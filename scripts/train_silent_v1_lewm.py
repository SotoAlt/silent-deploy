"""Silent v1 training — LeWM on audio observations.

Fork of /Users/rodrigosoto/repos/aura/scripts/train_relay_v9_lewm.py with the
smallest possible delta:

  v9 (RELAY)          →  Silent v1
  -----------           -----------
  3-channel RGB          4-channel log-mel-spec (N/E/S/W predator ears)
  224x224 input          224x224 input (mel-spec bilinear-upsampled from 64x50)
  action_dim=2           action_dim=3 (move_dx, move_dy, ping_amp)
  data: pixels           data: audio_obs
  data: actions          data: action
  data: states           data: state
  data: episode_ends     data: derived from ep_offset + ep_len

Everything else is identical: ViT-Tiny from scratch, AR causal predictor (depth=6,
heads=16, d=192), projectors, SIGReg, pure LeWM loss by default, optional joint
DexWM state head via --lambda-state > 0. Blackwell SDPA workaround retained.

Usage:
  # Local CPU smoke test (1 epoch on pilot data)
  PYTHONPATH=. python scripts/train_silent_v1_lewm.py \
      --h5 /tmp/silent_pilot.h5 --output /tmp/silent_smoke.pt \
      --epochs 1 --batch 4 --device cpu --num-workers 0 \
      --checkpoint-every 0

  # RunPod pilot: 20 epochs pure LeWM, checkpoint every 2 epochs
  python scripts/train_silent_v1_lewm.py \
      --h5 /tmp/silent_train.h5 --output checkpoints/silent/silent_v1.pt \
      --epochs 20 --batch 64 --device cuda \
      --checkpoint-every 2 --lambda-state 0.0

  # Pod pilot: 100 epochs joint-DexWM (v9 production recipe applied to audio)
  python scripts/train_silent_v1_lewm.py \
      --h5 /tmp/silent_train.h5 --output checkpoints/silent/silent_v1.pt \
      --epochs 100 --batch 128 --device cuda \
      --checkpoint-every 4 --lambda-state 10.0
"""
from __future__ import annotations

import argparse
import math
import pathlib
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===========================================================================
# SIGReg — isotropic-Gaussian regularizer (LeWM, Maes et al. 2025)
# Ported verbatim from /tmp/le-wm/module.py via RELAY v9's train_relay_predictor.
# ===========================================================================

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """proj: (T, B, D) — time-major matching LeWM paper usage."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ===========================================================================
# LeWM architecture — byte-for-byte port from /tmp/le-wm/
# ===========================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaln_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaln_mod[-1].weight)
        nn.init.zeros_(self.adaln_mod[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_mod(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class ARPredictor(nn.Module):
    def __init__(self, num_frames, input_dim, hidden_dim, output_dim,
                 depth=6, heads=16, dim_head=64, mlp_dim=2048,
                 dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        self.drop = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, emb, act_emb):
        x = self.in_proj(emb) + self.pos_emb[:, : emb.shape[1]]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, act_emb)
        return self.out_proj(self.norm(x))


class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.net(x)
        return x.reshape(*shape[:-1], x.shape[-1])


class ViTTinyEncoder(nn.Module):
    """ViT-Tiny trained from scratch. **4 input channels** for Silent's
    mel-spec tensor (N/E/S/W ears) instead of v9's 3 RGB channels.
    We replace timm's patch-embed conv with a fresh 4-channel one."""

    def __init__(self, image_size=224, patch_size=14, hidden_dim=192, in_channels=4):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=image_size, patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0, global_pool='',
        )
        self.hidden_dim = hidden_dim
        assert self.vit.embed_dim == hidden_dim, \
            f'timm ViT-Tiny embed_dim={self.vit.embed_dim}, expected {hidden_dim}'

    def forward(self, pixels):
        out = self.vit.forward_features(pixels)
        return out[:, 0]  # CLS token


class LeWM(nn.Module):
    def __init__(self, hidden_dim=192, embed_dim=192, action_dim=3, in_channels=4,
                 frameskip=5, history=3, depth=6, heads=16,
                 dim_head=64, mlp_dim=2048, dropout=0.1,
                 with_state_head=False, state_dim=10, head_hidden=256):
        super().__init__()
        self.history = history
        self.frameskip = frameskip
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.encoder = ViTTinyEncoder(hidden_dim=hidden_dim, in_channels=in_channels)
        self.projector = MLPProjector(hidden_dim, 2048, embed_dim)
        self.pred_proj = MLPProjector(hidden_dim, 2048, embed_dim)
        self.action_encoder = nn.Linear(frameskip * action_dim, embed_dim)
        self.predictor = ARPredictor(
            num_frames=history, input_dim=embed_dim, hidden_dim=hidden_dim,
            output_dim=hidden_dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
        )
        self.with_state_head = with_state_head
        if with_state_head:
            self.state_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, state_dim),
            )
        else:
            self.state_head = None

    def encode(self, obs_bt):
        """obs_bt: (B, T, C, H, W) → emb (B, T, embed_dim)."""
        B, T = obs_bt.shape[:2]
        flat = obs_bt.reshape(B * T, *obs_bt.shape[2:])
        h = self.encoder(flat)
        emb = self.projector(h)
        return emb.reshape(B, T, -1)

    def predict(self, emb_ctx, act_ctx):
        h = self.predictor(emb_ctx, act_ctx)
        return self.pred_proj(h)


# ===========================================================================
# Dataset: HDF5 with audio_obs (4, 64, 50) mel-specs, bilinear up to 224x224
# ===========================================================================

class AudioDataset(torch.utils.data.Dataset):
    """Silent HDF5 loader. Returns (obs, action, state) tuples where obs is
    (history + num_preds, 4, 224, 224) — mel-specs upsampled to ViT input size."""

    def __init__(self, h5_path, history=3, num_preds=1, frameskip=5,
                 split='train', val_frac=0.1, seed=42, preload=True,
                 input_size=224):
        self.h5_path = h5_path
        self.history = history
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.input_size = input_size
        self._h5 = None
        self._obs_ram = None
        self._action_ram = None
        self._state_ram = None
        if preload:
            t0 = time.time()
            with h5py.File(h5_path, 'r') as f:
                ep_offset = f['ep_offset'][:].astype(np.int64)
                ep_len = f['ep_len'][:].astype(np.int64)
                self.episode_ends = (ep_offset + ep_len).astype(np.int64)
                print(f'  preloading audio_obs → RAM…', flush=True)
                self._obs_ram = np.asarray(f['audio_obs'][:])
                self._action_ram = np.asarray(f['action'][:]).astype(np.float32)
                self._state_ram = np.asarray(f['state'][:]).astype(np.float32)
            print(f'  preload: obs {self._obs_ram.shape} {self._obs_ram.dtype} '
                  f'gb={self._obs_ram.nbytes / 1e9:.2f} in {time.time() - t0:.1f}s',
                  flush=True)
        else:
            with h5py.File(h5_path, 'r') as f:
                ep_offset = f['ep_offset'][:].astype(np.int64)
                ep_len = f['ep_len'][:].astype(np.int64)
                self.episode_ends = (ep_offset + ep_len).astype(np.int64)

        self.indices = []
        ep_starts = np.concatenate([[0], self.episode_ends[:-1]])
        for s, e in zip(ep_starts, self.episode_ends):
            min_i = s + (history - 1) * frameskip
            max_i = e - num_preds * frameskip - 1
            if max_i >= min_i:
                self.indices.extend(range(min_i, max_i + 1))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.indices))
        n_val = int(len(perm) * val_frac)
        if split == 'train':
            self.indices = [self.indices[j] for j in perm[n_val:]]
        else:
            self.indices = [self.indices[j] for j in perm[:n_val]]

    def _h(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        if self._obs_ram is not None:
            obs = self._obs_ram
            actions = self._action_ram
            states = self._state_ram
        else:
            h5 = self._h()
            obs = h5['audio_obs']
            actions = h5['action']
            states = h5['state']

        hist_starts = [i - k * self.frameskip for k in range(self.history - 1, -1, -1)]
        tgt_starts = [i + k * self.frameskip for k in range(1, self.num_preds + 1)]
        all_starts = hist_starts + tgt_starts

        frames = []
        for start in all_starts:
            mel = obs[start]              # (4, 64, 50) float32
            t = torch.from_numpy(mel).float()
            # Bilinear resize to (4, input_size, input_size) — ViT-Tiny expects 224
            if t.shape[-2:] != (self.input_size, self.input_size):
                t = F.interpolate(t.unsqueeze(0),
                                  size=(self.input_size, self.input_size),
                                  mode='bilinear', align_corners=False).squeeze(0)
            frames.append(t)
        obs_out = torch.stack(frames, dim=0)  # (T, 4, 224, 224)

        act_list = []
        for k in range(self.history):
            base = hist_starts[k]
            # frameskip actions starting at base, 3-D each (dx, dy, ping_amp)
            acts_k = actions[base: base + self.frameskip]  # (frameskip, 3)
            # Some indices may not have a full frameskip window if near ep end;
            # the indices list was pre-filtered to avoid this, but guard anyway.
            if acts_k.shape[0] < self.frameskip:
                pad = np.zeros((self.frameskip - acts_k.shape[0], 3), dtype=np.float32)
                acts_k = np.concatenate([acts_k, pad], axis=0)
            act_list.append(torch.from_numpy(acts_k.reshape(-1)).float())
        action_out = torch.stack(act_list, dim=0)  # (history, frameskip*3)

        state_out = torch.from_numpy(states[all_starts[-1]]).float()
        return obs_out, action_out, state_out


# ===========================================================================
# Training
# ===========================================================================

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--h5', required=True)
    pa.add_argument('--output', required=True, help="Final checkpoint path (best val)")
    pa.add_argument('--epochs', type=int, default=20)
    pa.add_argument('--batch', type=int, default=64)
    pa.add_argument('--lr', type=float, default=5e-5)
    pa.add_argument('--weight-decay', type=float, default=1e-3)
    pa.add_argument('--frameskip', type=int, default=5)
    pa.add_argument('--history', type=int, default=3)
    pa.add_argument('--lambda-sigreg', type=float, default=0.09)
    pa.add_argument('--lambda-state', type=float, default=0.0,
                    help='DexWM joint state-head weight. 0 = pure LeWM. v9 production uses 10.')
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda')
    pa.add_argument('--checkpoint-every', type=int, default=2,
                    help='Save intermediate checkpoint every N epochs. 0 = only final.')
    pa.add_argument('--checkpoint-dir', default=None,
                    help='Directory for intermediate checkpoints. Default: same dir as --output.')
    pa.add_argument('--resume-from', default=None)
    args = pa.parse_args()

    device = torch.device(args.device)
    use_bf16 = args.device == 'cuda'

    # Blackwell SDPA workaround — v9's main-branch fix
    if args.device == 'cuda':
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print('  SDPA: flash=off mem_eff=on math=on', flush=True)
        except Exception as e:
            print(f'  SDPA backend control unavailable: {e}', flush=True)

    print(f'Silent v1 LeWM training — {args.h5}', flush=True)
    train_ds = AudioDataset(args.h5, history=args.history, num_preds=1,
                            frameskip=args.frameskip, split='train', preload=True)
    val_ds = AudioDataset(args.h5, history=args.history, num_preds=1,
                          frameskip=args.frameskip, split='val', preload=False)
    val_ds._obs_ram = train_ds._obs_ram
    val_ds._action_ram = train_ds._action_ram
    val_ds._state_ram = train_ds._state_ram
    print(f'  train: {len(train_ds)} samples, val: {len(val_ds)} samples', flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
        pin_memory=(args.device == 'cuda'),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
        pin_memory=(args.device == 'cuda'),
    )

    model = LeWM(
        hidden_dim=192, embed_dim=192, action_dim=3, in_channels=4,
        frameskip=args.frameskip, history=args.history,
        with_state_head=(args.lambda_state > 0.0),
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    extra = ' + state head' if model.with_state_head else ''
    print(f'  model: {total:,} params (ViT-Tiny + AR pred + projectors{extra})  '
          f'λ_sigreg={args.lambda_sigreg} λ_state={args.lambda_state}', flush=True)

    if args.resume_from is not None:
        prev = torch.load(args.resume_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(prev['model_state'], strict=False)
        print(f'  resumed from {args.resume_from} (missing={len(missing)} unexp={len(unexpected)})',
              flush=True)

    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    ckpt_dir = pathlib.Path(args.checkpoint_dir or pathlib.Path(args.output).parent)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_state = None

    def save_checkpoint(path, state_dict, val_pred, epoch):
        model_sd = {k: v for k, v in state_dict.items() if not k.startswith('state_head.')}
        head_sd = {k.replace('state_head.', 'net.'): v
                   for k, v in state_dict.items() if k.startswith('state_head.')}
        out = {
            'model_state': model_sd,
            'config': {
                'hidden_dim': 192, 'embed_dim': 192, 'action_dim': 3, 'in_channels': 4,
                'frameskip': args.frameskip, 'history': args.history,
                'depth': 6, 'heads': 16, 'dim_head': 64, 'mlp_dim': 2048,
                'dropout': 0.1, 'lambda_sigreg': args.lambda_sigreg,
                'lambda_state': args.lambda_state,
                'joint_state_head': bool(args.lambda_state > 0.0),
                'game': 'silent', 'version': 'v1_lewm_joint' if args.lambda_state > 0.0 else 'v1_lewm_pure',
                'epoch': epoch,
            },
            'val_best_pred': val_pred,
        }
        if head_sd:
            out['state_head_state'] = head_sd
            out['state_head_config'] = {
                'embed_dim': 192, 'state_dim': 10, 'hidden': 256,
                'arch': 'mlp_192_256_256_10_gelu',
                'trained_jointly_with_predictor': True,
            }
        torch.save(out, str(path))

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        pred_losses, sig_losses, state_losses = [], [], []
        for step, (obs, actions, state) in enumerate(train_loader):
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True).float()
            with torch.autocast(device_type='cuda' if use_bf16 else 'cpu',
                                dtype=torch.bfloat16, enabled=use_bf16):
                emb = model.encode(obs)
                act_emb = model.action_encoder(actions)
                ctx_emb = emb[:, :args.history]
                tgt_emb = emb[:, 1:]
                pred_emb = model.predict(ctx_emb, act_emb)
                pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                sig_loss = sigreg(emb.transpose(0, 1))
                loss = pred_loss + args.lambda_sigreg * sig_loss
                if model.with_state_head:
                    pred_state = model.state_head(pred_emb[:, -1].float())
                    state_loss = (pred_state - state).pow(2).mean()
                    loss = loss + args.lambda_state * state_loss
                    state_losses.append(state_loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pred_losses.append(pred_loss.item())
            sig_losses.append(sig_loss.item())
            if step == 0 or (step + 1) % 50 == 0:
                dt = time.time() - t0
                extra = f' state={state_losses[-1]:.4f}' if state_losses else ''
                print(f'    ep{epoch + 1} step {step + 1}/{len(train_loader)}  '
                      f'pred={pred_loss.item():.5f} sig={sig_loss.item():.3f}{extra}  '
                      f'{dt:.0f}s', flush=True)
        scheduler.step()

        # Val
        model.eval()
        vl_p, vl_s, vl_st = [], [], []
        with torch.no_grad():
            for obs, actions, state in val_loader:
                obs = obs.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                state = state.to(device, non_blocking=True).float()
                with torch.autocast(device_type='cuda' if use_bf16 else 'cpu',
                                    dtype=torch.bfloat16, enabled=use_bf16):
                    emb = model.encode(obs)
                    act_emb = model.action_encoder(actions)
                    ctx_emb = emb[:, :args.history]
                    tgt_emb = emb[:, 1:]
                    pred_emb = model.predict(ctx_emb, act_emb)
                    p = (pred_emb - tgt_emb).pow(2).mean()
                    s = sigreg(emb.transpose(0, 1))
                    if model.with_state_head:
                        pred_state = model.state_head(pred_emb[:, -1].float())
                        st = (pred_state - state).pow(2).mean()
                        vl_st.append(st.item())
                vl_p.append(p.item())
                vl_s.append(s.item())
        val_pred = float(np.mean(vl_p)) if vl_p else 0.0
        val_sig = float(np.mean(vl_s)) if vl_s else 0.0
        val_state = float(np.mean(vl_st)) if vl_st else 0.0
        tracking = val_pred + (args.lambda_state * val_state if model.with_state_head else 0.0)

        dt = time.time() - t0
        print(f'  ep {epoch + 1:3d}/{args.epochs}  '
              f'train_pred={np.mean(pred_losses):.5f} train_sig={np.mean(sig_losses):.4f} '
              + (f'train_state={np.mean(state_losses):.5f} ' if state_losses else '')
              + f'| val_pred={val_pred:.5f} val_sig={val_sig:.4f} '
              + (f'val_state={val_state:.5f} ' if model.with_state_head else '')
              + f'| {dt:.0f}s',
              flush=True)

        # Track best
        if tracking < best_val:
            best_val = tracking
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Periodic checkpoint so we can hot-swap into the game live
        if args.checkpoint_every > 0 and ((epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs):
            iter_ckpt = ckpt_dir / f'silent_v1_ep{epoch + 1:03d}.pt'
            current_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(iter_ckpt, current_sd, val_pred, epoch + 1)
            print(f'    [checkpoint] → {iter_ckpt}', flush=True)

    # Final best
    save_checkpoint(pathlib.Path(args.output), best_state, best_val, args.epochs)
    print(f'\nSaved best → {args.output}  (best val={best_val:.5f})', flush=True)


if __name__ == '__main__':
    main()
