"""Silent — JEPA predator v2, state-space CEM cost via post-hoc head.

The v1 predator used embedding-space CEM cost vs a synthetic-goal embedding.
Two failure modes killed it:
  1. The encoder's projected-embedding space doesn't have a clean
     "prey caught" region — our preflight v2 showed player_xy R² ≈ 0.35
     on the raw projected embedding vs predator_xy R² ≈ 1.0.
  2. The synthetic "prey at me" goal collapsed to "loud audio near my
     ears," which the CEM gamed by pinging constantly.

v2 fixes this by adding a post-hoc state-decoder head trained on a frozen
encoder+projector. After CEM predicts `next_emb` for each candidate, we
decode `next_emb → state` via the head and use **state-space distance**
`|predator_xy - player_xy|` as the cost. No synthetic goal needed.

Requires: a post-hoc head checkpoint from scripts/train_silent_posthoc_head.py
(10-d state output: [pred_x, pred_y, pred_vx, pred_vy, player_x, player_y,
player_vx, player_vy, voice, dist]).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_silent_v1_lewm import LeWM


def _build_head(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class JEPAPredatorV2:
    """State-space CEM predator using a frozen JEPA + post-hoc state head.

    Config:
        n_samples — CEM population per iter
        n_iters   — CEM refinement iterations
        horizon   — rollout steps per candidate
        ping_weight — cost penalty per unit of ping amplitude (discourages spam)
        w_dist    — weight on state-space distance cost
    """

    def __init__(self, ckpt_path: str, head_path: str, device: str = 'cpu',
                 n_samples: int = 16, n_iters: int = 2, horizon: int = 1,
                 ping_weight: float = 30.0, w_dist: float = 1.0,
                 ping_every: int = 30, ping_amp: float = 0.6):
        # Production override: env vars let us tune the CEM cost without
        # rebuilding the image. SILENT_CEM_SAMPLES / SILENT_CEM_ITERS shrink
        # the loop on slow CPUs (CPX21 vCPU = ~4x slower than M3). Threads
        # cap is critical on shared vCPUs — too many threads thrash.
        import os as _os
        n_samples = int(_os.environ.get('SILENT_CEM_SAMPLES', n_samples))
        n_iters   = int(_os.environ.get('SILENT_CEM_ITERS',   n_iters))
        try:
            n_threads = int(_os.environ.get(
                'SILENT_TORCH_THREADS', max(1, _os.cpu_count() or 4)))
            torch.set_num_threads(max(1, n_threads))
        except Exception:
            pass
        """ping_every / ping_amp: periodic 'scan ping' to keep the audio scene
        illuminated. Without them, the predator's decoder goes stale after the
        player stops making noise. 18 ticks @ 10 Hz = 1.8 s cadence."""
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.horizon = horizon
        self.ping_weight = ping_weight
        self.w_dist = w_dist
        self.ping_every = ping_every
        self.ping_amp = ping_amp

        # Load frozen JEPA
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt['config']
        self.cfg = cfg
        self.model = LeWM(
            hidden_dim=cfg['hidden_dim'], embed_dim=cfg['embed_dim'],
            action_dim=cfg['action_dim'], in_channels=cfg['in_channels'],
            frameskip=cfg['frameskip'], history=cfg['history'],
            with_state_head=cfg['joint_state_head'],
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'], strict=False)
        self.model.eval()

        # Load post-hoc head
        head_ckpt = torch.load(head_path, map_location=self.device, weights_only=False)
        self.head_input_kind = head_ckpt.get('input_kind', 'projected')
        if self.head_input_kind != 'projected':
            raise ValueError(f"head must be trained on 'projected' embeddings "
                             f"(this one is '{self.head_input_kind}')")
        head_state = head_ckpt['head_state']
        # Derive hidden dim from state shape
        hidden = head_state['0.weight'].shape[0]
        in_dim = head_state['0.weight'].shape[1]
        out_dim = head_state['4.weight'].shape[0]
        self.head = _build_head(in_dim, hidden, out_dim).to(self.device)
        self.head.load_state_dict(head_state)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)

        self.Y_mu = head_ckpt['norm']['Y_mu'].to(self.device)       # (1, out_dim)
        self.Y_sig = head_ckpt['norm']['Y_sig'].to(self.device)

        self.history = cfg['history']
        self.frameskip = cfg['frameskip']
        self.action_dim = cfg['action_dim']

        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[np.ndarray] = []
        self.tick_count = 0
        # Federation data tap reuses the encoder forward we already run
        # for CEM (see commit e740976 for why re-encoding choked the
        # game). Read via cached_embedding(); None during warmup ticks.
        self._cached_emb: Optional[torch.Tensor] = None

    def reset(self):
        self._obs_buf.clear()
        self._act_buf.clear()
        self.tick_count = 0
        self._cached_emb = None

    def cached_embedding(self) -> Optional[torch.Tensor]:
        """Last act()'s context embedding, last timestep, detached.
        Shape (1, embed_dim). None until obs_buf has accumulated
        `history` frames (~3 ticks of warmup)."""
        return self._cached_emb

    def _obs_tensor(self, mel_spec_sequence: list[np.ndarray]) -> torch.Tensor:
        out = []
        for m in mel_spec_sequence:
            t = torch.from_numpy(np.asarray(m, dtype=np.float32))
            t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                              mode='bilinear', align_corners=False).squeeze(0)
            out.append(t)
        return torch.stack(out, dim=0).unsqueeze(0).to(self.device)

    def _decode_state(self, emb: torch.Tensor) -> torch.Tensor:
        """emb (B, D) → decoded state (B, 10) in original scale."""
        y_norm = self.head(emb)
        return y_norm * self.Y_sig + self.Y_mu

    def act(self, env) -> tuple[float, float, float]:
        import time as _t
        _t0 = _t.time()
        self.tick_count += 1

        obs = env.get_audio_obs()
        self._obs_buf.append(obs)
        if len(self._obs_buf) > self.history + 2:
            self._obs_buf.pop(0)

        if len(self._obs_buf) < self.history:
            return 0.0, 0.0, 0.0

        # Encode history
        obs_seq = self._obs_buf[-self.history:]
        obs_tensor = self._obs_tensor(obs_seq)            # (1, H, 4, 224, 224)
        with torch.no_grad():
            emb_ctx = self.model.encode(obs_tensor)       # (1, H, D)

        # Detach + clone so the next planner tick can reuse emb_ctx
        # storage without mutating what the tap is holding.
        self._cached_emb = emb_ctx[:, -1].detach().clone()  # (1, D)

        # CEM
        mean = torch.zeros(self.action_dim, device=self.device)
        std = torch.tensor([0.7, 0.7, 0.4], device=self.device)
        best_action = None

        with torch.no_grad():
            for it in range(self.n_iters):
                eps = torch.randn(self.n_samples, self.action_dim, device=self.device)
                cands = (mean + std * eps).clamp(-1.0, 1.0)
                cands[:, 2] = cands[:, 2].clamp(0.0, 1.0)
                B = self.n_samples

                # Build (B, H, fs*ad) action tensor
                hist_zeros = np.zeros((self.frameskip, self.action_dim), dtype=np.float32)
                prev_rows = []
                for h in range(self.history - 1):
                    offset = self.history - 1 - h
                    if offset <= len(self._act_buf):
                        prev_rows.append(self._act_buf[-offset])
                    else:
                        prev_rows.append(hist_zeros)
                if prev_rows:
                    prev_tensor = torch.from_numpy(
                        np.stack(prev_rows, axis=0).astype(np.float32)
                    ).reshape(self.history - 1, -1).to(self.device)
                else:
                    prev_tensor = torch.zeros(0, self.frameskip * self.action_dim,
                                              device=self.device)

                cand_rows = cands.unsqueeze(1).repeat(1, self.frameskip, 1).reshape(B, 1, -1)
                prev_b = prev_tensor.unsqueeze(0).expand(B, -1, -1)
                act_tensor = torch.cat([prev_b, cand_rows], dim=1)

                act_emb = self.model.action_encoder(act_tensor)
                emb_ctx_b = emb_ctx.expand(B, -1, -1)
                pred = self.model.predict(emb_ctx_b, act_emb)
                final_pred = pred[:, -1]                            # (B, D)

                # Decode state for each candidate's predicted next-embedding
                decoded = self._decode_state(final_pred)            # (B, 10)
                # State layout per silent.py get_state():
                #   [0:2]  = predator_xy (normalized to [-1, 1], range = WINDOW_SIZE/2)
                #   [4:6]  = player_xy (normalized the same way)
                #   [9]    = player-to-EXIT dist (NOT pred-player — trap!)
                # Compute pred→player distance directly from decoded positions,
                # scaled back to pixels for interpretable cost units.
                pred_xy = decoded[:, :2]                            # (B, 2) norm
                player_xy = decoded[:, 4:6]                         # (B, 2) norm
                dist_px = torch.norm(pred_xy - player_xy, dim=-1) * 256.0  # px

                # Cost: predator→player distance in pixels + ping penalty
                costs = self.w_dist * dist_px + self.ping_weight * cands[:, 2]

                elite_k = max(2, B // 4)
                _, elite_idx = costs.topk(elite_k, largest=False)
                elites = cands[elite_idx]
                mean = elites.mean(dim=0)
                std = elites.std(dim=0) + 0.05
                if it == self.n_iters - 1:
                    best_action = elites[0].clone()

        a = best_action.cpu().numpy()

        # Periodic scan ping so the audio scene stays illuminated between
        # CEM decisions. The JEPA can't decode a position from silence.
        if self.tick_count % self.ping_every == 0 or self.tick_count <= 2:
            a[2] = max(float(a[2]), self.ping_amp)

        fs_acts = np.tile(a.astype(np.float32), (self.frameskip, 1))
        self._act_buf.append(fs_acts)
        if len(self._act_buf) > self.history + 2:
            self._act_buf.pop(0)

        if self.tick_count % 10 == 0:
            print(f"[predator.act] tick={self.tick_count} latency={(_t.time()-_t0)*1000:.0f}ms", flush=True)

        return float(a[0]), float(a[1]), float(a[2])
