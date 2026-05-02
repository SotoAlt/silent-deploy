"""Silent — JEPA-controlled predator policy.

Loads a Silent v1 checkpoint and uses the trained encoder + AR predictor
to choose actions. This is the Phase 4 equivalent of RELAY's relay_planner_v9.

For pure LeWM checkpoints (no state head), we use LATENT CEM:
    1. Encode history of audio observations → emb_ctx (B, H, D)
    2. Sample N candidate action sequences from Gaussian
    3. Forward-roll predictor for each candidate → predicted next-embedding
    4. Score each by distance to a SYNTHETIC "prey-at-me" target embedding
    5. Pick best first action, refit Gaussian from top-K, iterate

For joint DexWM checkpoints (with state head), we can decode state directly
and score by predator-player distance — cleaner, future work.

Built for the player-facing game server (world_model/infer_silent.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_silent_v1_lewm import LeWM


class JEPAPredator:
    """LeWM-driven predator. Holds a rolling buffer of audio observations +
    actions to feed the predictor each tick. Uses latent CEM with a synthetic
    "prey-is-loud-at-my-location" target embedding as the goal.

    Config knobs (pass via __init__):
        n_samples   — CEM population size per iteration
        n_iters     — CEM refinement iterations
        horizon     — rollout steps per candidate (we use H=1: 1-step CEM,
                      re-encode every tick → matches AURA Rule 1)
        ping_every  — emit a real ping every N ticks regardless of plan
    """

    def __init__(self, ckpt_path: str, device: str = 'cpu',
                 n_samples: int = 32, n_iters: int = 3,
                 horizon: int = 1, ping_every: int = 25):
        # ping_every raised 8 → 25 (2.5s at 10Hz). Pinging every 0.8s was
        # both visually noisy and flooded the room with concurrent ripples
        # that contaminated the goal embedding.
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.horizon = horizon
        self.ping_every = ping_every
        self.tick_count = 0

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

        self.history = cfg['history']
        self.frameskip = cfg['frameskip']
        self.action_dim = cfg['action_dim']

        # Rolling buffers
        self._obs_buf: list[np.ndarray] = []    # each entry: (4, 64, 50)
        self._act_buf: list[np.ndarray] = []    # each entry: (frameskip, 3)
        self._goal_emb: Optional[torch.Tensor] = None

    def reset(self):
        self._obs_buf.clear()
        self._act_buf.clear()
        self._goal_emb = None
        self.tick_count = 0

    def _obs_tensor(self, mel_spec_sequence: list[np.ndarray]) -> torch.Tensor:
        """list of (4, 64, 50) → (1, T, 4, 224, 224) tensor on device."""
        out = []
        for m in mel_spec_sequence:
            t = torch.from_numpy(np.asarray(m, dtype=np.float32))
            t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                              mode='bilinear', align_corners=False).squeeze(0)
            out.append(t)
        return torch.stack(out, dim=0).unsqueeze(0).to(self.device)

    def _synth_goal_obs(self, env) -> np.ndarray:
        """Synthetic 'prey is at predator's current position' audio observation.

        We build a fake mel-spec where the footstep source is located exactly
        at the predator's position — which physically corresponds to the
        predator having caught the prey. Any CEM plan that moves the predator
        toward this target embedding is moving toward the real player.

        IMPORTANT: we also suppress the predator's own pings and reduce the
        synthetic voice to realistic levels. An earlier version left both
        intact, which made the goal embedding ≈ "loud audio near my ears,"
        and the easiest way to reach that embedding is just to ping louder —
        so the CEM collapsed to ping spam instead of pursuing the real player.
        """
        saved_player_pos = (float(env.player.position.x), float(env.player.position.y))
        saved_player_vel = (float(env.player.velocity.x), float(env.player.velocity.y))
        saved_voice = float(env._voice_amp)
        saved_pings = list(env._pings)   # shallow copy

        # Player at predator position, running at realistic speed + moderate voice
        env.player.position = (env.predator.position.x, env.predator.position.y)
        env.player.velocity = (120.0, 0.0)
        env._voice_amp = 0.5
        env._pings = []                   # suppress predator-own pings

        obs = env.get_audio_obs()

        env.player.position = saved_player_pos
        env.player.velocity = saved_player_vel
        env._voice_amp = saved_voice
        env._pings = saved_pings
        return obs

    def _encode_goal(self, env) -> torch.Tensor:
        """Compute the goal embedding by synthesizing 'prey-at-me' obs and
        encoding it. Cache for a tick to avoid recomputing."""
        goal_obs = self._synth_goal_obs(env)
        # Fake a trivial history (same obs repeated) to satisfy encode's shape
        goal_tensor = self._obs_tensor([goal_obs] * self.history)  # (1, H, 4, 224, 224)
        with torch.no_grad():
            goal_emb = self.model.encode(goal_tensor)  # (1, H, D)
        return goal_emb[:, -1:]   # (1, 1, D)

    def act(self, env) -> tuple[float, float, float]:
        """Return (move_dx, move_dy, ping_amp) for the predator this tick."""
        self.tick_count += 1

        # 1) Get current audio observation + add to buffers
        obs = env.get_audio_obs()
        self._obs_buf.append(obs)
        if len(self._obs_buf) > self.history + 2:
            self._obs_buf.pop(0)

        # Not enough history yet → coast
        if len(self._obs_buf) < self.history:
            ping = 1.0 if self.tick_count <= 1 else 0.0
            return 0.0, 0.0, ping

        # 2) Encode history + build context
        obs_seq = self._obs_buf[-self.history:]
        obs_tensor = self._obs_tensor(obs_seq)  # (1, H, 4, 224, 224)
        with torch.no_grad():
            emb_ctx = self.model.encode(obs_tensor)  # (1, H, D)

        # 3) Compute / refresh goal embedding
        if self._goal_emb is None or self.tick_count % 3 == 0:
            self._goal_emb = self._encode_goal(env)  # (1, 1, D)

        # 4) CEM over (dx, dy, ping) action
        mean = torch.zeros(self.action_dim, device=self.device)
        std = torch.tensor([0.7, 0.7, 0.4], device=self.device)

        # We emit a single action that will be applied; but the predictor expects
        # `history` action-packets each of (frameskip * action_dim). Build a packed
        # action buffer by repeating our candidate.
        best_action = None
        with torch.no_grad():
            for it in range(self.n_iters):
                # Sample N candidates around current mean
                eps = torch.randn(self.n_samples, self.action_dim, device=self.device)
                cands = (mean + std * eps).clamp(-1.0, 1.0)
                cands[:, 2] = cands[:, 2].clamp(0.0, 1.0)   # ping_amp in [0, 1]

                # Build (N, H, fs*action_dim) packed action tensor. The LAST
                # history row holds the candidate action; earlier rows hold
                # the most-recent actions we actually applied (padded w/ zeros
                # while _act_buf is still warming up at episode start).
                B = self.n_samples
                hist_zeros = np.zeros((self.frameskip, self.action_dim), dtype=np.float32)
                # Pre-compute the (history-1) earlier history rows (same for all B)
                prev_rows = []
                for h in range(self.history - 1):
                    # Want the h-th oldest-to-newest history row (excluding the
                    # candidate slot). Index into act_buf counting back from end.
                    offset = self.history - 1 - h   # 1 = oldest, then decreasing to newest-past
                    if offset <= len(self._act_buf):
                        prev_rows.append(self._act_buf[-offset])
                    else:
                        prev_rows.append(hist_zeros)
                prev_tensor = torch.from_numpy(
                    np.stack(prev_rows, axis=0).astype(np.float32)
                ).reshape(self.history - 1, -1) if prev_rows else torch.zeros(0, self.frameskip * self.action_dim)
                prev_tensor = prev_tensor.to(self.device)

                # Build candidate rows: shape (B, 1, fs*ad)
                cand_rows = cands.unsqueeze(1).repeat(1, self.frameskip, 1).reshape(B, 1, -1)
                # Broadcast prev_rows across B: (B, H-1, fs*ad)
                prev_b = prev_tensor.unsqueeze(0).expand(B, -1, -1)
                act_tensor = torch.cat([prev_b, cand_rows], dim=1)  # (B, H, fs*ad)

                # Forward-roll 1 step
                act_emb = self.model.action_encoder(act_tensor)            # (B, H, D)
                emb_ctx_b = emb_ctx.expand(B, -1, -1)                       # (B, H, D)
                pred = self.model.predict(emb_ctx_b, act_emb)               # (B, H, D)
                final_pred = pred[:, -1:]                                   # (B, 1, D)

                # Score: distance to goal embedding (MSE), low is better
                goal_b = self._goal_emb.expand(B, -1, -1)
                costs = F.mse_loss(final_pred, goal_b, reduction='none').mean(dim=(1, 2))

                # Elites: top-K lowest cost
                elite_k = max(2, B // 4)
                _, elite_idx = costs.topk(elite_k, largest=False)
                elites = cands[elite_idx]
                mean = elites.mean(dim=0)
                std = elites.std(dim=0) + 0.05  # refit + floor for exploration
                if it == self.n_iters - 1:
                    best_action = elites[0].clone()

        # 5) Override ping if we haven't pinged recently (keep room sonically active)
        a = best_action.cpu().numpy()
        if self.tick_count % self.ping_every == 0:
            a[2] = max(a[2], 0.9)

        # 6) Save the action to our history buffer (padded to frameskip)
        fs_acts = np.tile(a.astype(np.float32), (self.frameskip, 1))
        self._act_buf.append(fs_acts)
        if len(self._act_buf) > self.history + 2:
            self._act_buf.pop(0)

        return float(a[0]), float(a[1]), float(a[2])
