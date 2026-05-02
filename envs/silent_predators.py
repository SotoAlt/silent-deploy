"""Silent — scripted predator policies for Phase 0.

Two predators, toggleable in the UI:

  OraclePredator
    Knows the player's exact position (cheating). Moves toward it at full
    speed. Pings periodically for visual feedback only (the audio side is
    irrelevant for oracle). Baseline for "is the game fun when predator is
    perfect". The JEPA's ceiling.

  EcholocationPredator
    State-level stand-in for the JEPA during Phase 0. Does not peek at the
    player's true position. Instead, reads what WOULD be audio obs (when
    pyroomacoustics lands, Phase 0.2) and moves toward the direction of loudest
    return. Pings every N ticks to refresh its audio sample.

    Phase 0.1 stub: since audio isn't wired yet, this predator is a placeholder
    that moves in a random direction with periodic re-steering. It exists to
    validate the server/client loop works with a "dumb" predator that can lose.

Each predator exposes:
    reset()
    act(env) -> (dx, dy, ping_amp)      # action to pass to env.step(action, 'predator')
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


class OraclePredator:
    """Cheater baseline. Moves straight toward player at full speed, pings periodically."""
    def __init__(self, ping_interval_ticks: int = 10, seed: int = 0):
        self.ping_interval = ping_interval_ticks
        self.rng = np.random.default_rng(seed)
        self.ticks_since_ping = 0

    def reset(self):
        self.ticks_since_ping = 0

    def act(self, env) -> tuple[float, float, float]:
        px = env.player.position.x
        py = env.player.position.y
        ex = env.predator.position.x
        ey = env.predator.position.y
        dx = px - ex
        dy = py - ey
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0, 0.0, 0.0
        move_dx = dx / d
        move_dy = dy / d

        self.ticks_since_ping += 1
        ping = 0.0
        if self.ticks_since_ping >= self.ping_interval:
            ping = 1.0
            self.ticks_since_ping = 0
        return float(move_dx), float(move_dy), float(ping)


class EcholocationPredator:
    """Scripted audio-driven predator. Stand-in for the JEPA during Phase 0
    (and a permanent baseline for evaluation).

    Each tick:
      1. Reads 4-channel mel-spec obs (N/E/S/W cardioid ears) from env.get_audio_obs()
      2. Computes per-channel total energy → direction unit vector
      3. EMA-smooths the direction so it doesn't jitter between pings
      4. Walks that direction at 80% of max speed
      5. Pings periodically so the room is audibly scanned

    If all channels are silent (player still + silent + out of beacon range),
    the predator coasts its last heading and occasionally re-randomizes.
    """

    def __init__(self, ping_interval_ticks: int = 12, seed: int = 0):
        self.ping_interval = ping_interval_ticks
        self.rng = np.random.default_rng(seed)
        self.ticks_since_ping = 0
        self.heading_dx = 0.0
        self.heading_dy = 0.0
        self.silent_streak = 0   # ticks with no audible signal

    def reset(self):
        self.ticks_since_ping = 0
        self.heading_dx = 0.0
        self.heading_dy = 0.0
        self.silent_streak = 0

    def act(self, env) -> tuple[float, float, float]:
        obs = env.get_audio_obs()   # (4, N_MELS, T)
        # Per-channel energy (emphasize recent frames so ping echoes count more)
        recent = obs[:, :, :]       # use full 500-ms window
        energies = recent.mean(axis=(1, 2)).astype(np.float32)  # (4,)

        # Subtract a floor so near-zero signals don't bias the direction vector
        floor = 0.003
        e_eff = np.maximum(energies - floor, 0.0)
        e_total = float(e_eff.sum())

        # Lobe unit vectors (must match silent.py): N=(0,-1), E=(1,0), S=(0,1), W=(-1,0)
        lobe_axes = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

        if e_total > 1e-4:
            # Weighted mean of lobe directions → estimated source bearing
            weights = e_eff / e_total
            est = (lobe_axes * weights[:, None]).sum(axis=0)
            mag = float(np.hypot(est[0], est[1]))
            if mag > 1e-6:
                est_dx, est_dy = float(est[0] / mag), float(est[1] / mag)
            else:
                est_dx, est_dy = 0.0, 0.0
            self.silent_streak = 0
            # EMA smoothing (sticky direction so one ghost ping doesn't 180 us)
            alpha = 0.55
            self.heading_dx = (1 - alpha) * self.heading_dx + alpha * est_dx
            self.heading_dy = (1 - alpha) * self.heading_dy + alpha * est_dy
        else:
            # No audible signal — coast, occasionally re-randomize if long silence
            self.silent_streak += 1
            if self.silent_streak > 20:
                theta = self.rng.uniform(0, 2 * math.pi)
                self.heading_dx = math.cos(theta)
                self.heading_dy = math.sin(theta)
                self.silent_streak = 0

        # Renormalize heading (EMA can drift from unit length)
        h_mag = math.hypot(self.heading_dx, self.heading_dy)
        if h_mag > 1e-6:
            move_dx = self.heading_dx / h_mag
            move_dy = self.heading_dy / h_mag
        else:
            move_dx, move_dy = 0.0, 0.0

        # Walk at 80% of max speed
        move_dx *= 0.8
        move_dy *= 0.8

        # Ping periodically to keep the room sonically active
        self.ticks_since_ping += 1
        ping = 0.0
        if self.ticks_since_ping >= self.ping_interval:
            ping = 0.9
            self.ticks_since_ping = 0

        return float(move_dx), float(move_dy), float(ping)


class RandomWalkPredator:
    """Baseline: random heading, occasional re-pick, no audio use. Gives the
    JEPA a negative example — data where the predator's behavior is unrelated
    to the player's position so the model doesn't learn a shortcut."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks = 0
        self.ping_interval = 14

    def reset(self):
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks = 0

    def act(self, env):
        self.ticks += 1
        if self.ticks % 12 == 0:
            self.heading = self.rng.uniform(0, 2 * math.pi)
        ping = 0.9 if self.ticks % self.ping_interval == 0 else 0.0
        return float(math.cos(self.heading) * 0.6), float(math.sin(self.heading) * 0.6), float(ping)


class StationaryPredator:
    """Never moves. Pings periodically. Lets the JEPA see 'static agent + changing scene'
    which is important coverage for the observation distribution."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.ticks = 0
        self.ping_interval = 10

    def reset(self):
        self.ticks = 0

    def act(self, env):
        self.ticks += 1
        ping = 0.9 if self.ticks % self.ping_interval == 0 else 0.0
        return 0.0, 0.0, float(ping)


class SpiralingPredator:
    """Expands outward in a slow spiral. A deterministic trajectory that covers
    a lot of the room — useful for predictor horizon-stress examples."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.phase = 0.0
        self.ticks = 0
        self.ping_interval = 12

    def reset(self):
        self.phase = 0.0
        self.ticks = 0

    def act(self, env):
        self.ticks += 1
        self.phase += 0.12
        # Heading rotates while amplitude slowly increases — spiral outward
        dx = math.cos(self.phase) * 0.55
        dy = math.sin(self.phase) * 0.55
        ping = 0.9 if self.ticks % self.ping_interval == 0 else 0.0
        return float(dx), float(dy), float(ping)


def make_predator(name: str, seed: int = 0,
                  jepa_ckpt: str | None = None,
                  jepa_head: str | None = None,
                  jepa_test_ckpts: dict | None = None,
                  jepa_device: str = 'cpu'):
    """Factory for any predator mode.

    `jepa_test_ckpts` maps test-slot name → (ckpt_path, head_path) tuples.
    e.g. {'jepa_test1': (ckpt, head), 'jepa_test2': (ckpt, head), ...}
    so we can wire arbitrary numbers of test variants without changing
    the signature each time.
    """
    table = {
        'oracle':       OraclePredator,
        'echolocation': EcholocationPredator,
        'random_walk':  RandomWalkPredator,
        'stationary':   StationaryPredator,
        'spiraling':    SpiralingPredator,
    }
    if name == 'jepa_v1':
        if jepa_ckpt is None:
            raise ValueError("predator='jepa_v1' requires jepa_ckpt path")
        from envs.silent_jepa_predator import JEPAPredator
        return JEPAPredator(ckpt_path=jepa_ckpt, device=jepa_device)
    if name == 'jepa_v2':
        # Canonical baseline (silent-baseline-v1 git tag)
        if jepa_ckpt is None or jepa_head is None:
            raise ValueError("predator='jepa_v2' requires jepa_ckpt + jepa_head")
        from envs.silent_jepa_predator_v2 import JEPAPredatorV2
        return JEPAPredatorV2(ckpt_path=jepa_ckpt, head_path=jepa_head,
                              device=jepa_device)
    test_ckpts = jepa_test_ckpts or {}
    if name in test_ckpts:
        ckpt, head = test_ckpts[name]
        if ckpt is None or head is None:
            raise ValueError(f"predator={name!r} requires its corresponding ckpt + head")
        from envs.silent_jepa_predator_v2 import JEPAPredatorV2
        return JEPAPredatorV2(ckpt_path=ckpt, head_path=head, device=jepa_device)
    if name not in table:
        opts = list(table) + ['jepa_v1', 'jepa_v2'] + list(test_ckpts.keys())
        raise KeyError(f"Unknown predator {name!r}. Options: {opts}")
    return table[name](seed=seed)


PREDATOR_MIX = [
    ('echolocation', 0.40),
    ('oracle',       0.30),
    ('random_walk',  0.15),
    ('stationary',   0.10),
    ('spiraling',    0.05),
]


def sample_predator(rng: np.random.Generator, seed: int = 0):
    names = [n for n, _ in PREDATOR_MIX]
    probs = np.array([p for _, p in PREDATOR_MIX], dtype=np.float64)
    name = rng.choice(names, p=probs / probs.sum())
    return name, make_predator(name, seed=seed)
