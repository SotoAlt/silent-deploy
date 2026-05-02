"""Silent — scripted player (prey) policies for Phase 1 data generation.

During data collection we need varied player behavior so the JEPA sees the full
distribution of prey strategies, not just "straight to exit". Each policy
returns a 3-D action (move_dx, move_dy, voice_amp) per tick.

Mix targets (from the plan):
  30% RandomWalkPlayer
  25% WallHuggerPlayer
  20% CornerHiderPlayer
  15% BeaconSeekerPlayer
  10% DecoyVoicePlayer

The mix weights live in scripts/collect_silent_data.py, not here.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


class _BasePlayer:
    """Base with helper for clamping final action."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env) -> tuple[float, float, float]:
        raise NotImplementedError


class RandomWalkPlayer(_BasePlayer):
    """Wanders in random directions, re-picks heading every ~1 second."""
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks_since_change = 0

    def reset(self):
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks_since_change = 0

    def act(self, env) -> tuple[float, float, float]:
        self.ticks_since_change += 1
        if self.ticks_since_change >= 10:
            self.heading = self.rng.uniform(0, 2 * math.pi)
            self.ticks_since_change = 0
        dx = math.cos(self.heading) * 0.8
        dy = math.sin(self.heading) * 0.8
        return float(dx), float(dy), 0.0


class WallHuggerPlayer(_BasePlayer):
    """Stays close to the walls, moves along them. Creates directional audio
    from the periphery — tests how well the predator tracks an edge-runner."""
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.wall_pref = self.rng.choice(['N', 'E', 'S', 'W'])
        self.clockwise = bool(self.rng.integers(0, 2))
        self.ticks_since_change = 0

    def reset(self):
        self.wall_pref = self.rng.choice(['N', 'E', 'S', 'W'])
        self.clockwise = bool(self.rng.integers(0, 2))
        self.ticks_since_change = 0

    def act(self, env) -> tuple[float, float, float]:
        from envs.silent import WINDOW_SIZE
        px = env.player.position.x
        py = env.player.position.y
        target_dx, target_dy = 0.0, 0.0
        margin = 50.0
        # Steer toward the preferred wall first
        if self.wall_pref == 'N' and py > margin + 20: target_dy = -1
        elif self.wall_pref == 'S' and py < WINDOW_SIZE - margin - 20: target_dy = 1
        elif self.wall_pref == 'E' and px < WINDOW_SIZE - margin - 20: target_dx = 1
        elif self.wall_pref == 'W' and px > margin + 20: target_dx = -1
        else:
            # On wall — run along it in cw/ccw direction
            if self.wall_pref == 'N': target_dx = 1 if self.clockwise else -1
            elif self.wall_pref == 'S': target_dx = -1 if self.clockwise else 1
            elif self.wall_pref == 'E': target_dy = 1 if self.clockwise else -1
            else: target_dy = -1 if self.clockwise else 1
        # Occasionally switch walls
        self.ticks_since_change += 1
        if self.ticks_since_change >= 80:
            self.wall_pref = self.rng.choice(['N', 'E', 'S', 'W'])
            self.ticks_since_change = 0
        m = math.hypot(target_dx, target_dy) + 1e-6
        return float(target_dx / m * 0.7), float(target_dy / m * 0.7), 0.0


class CornerHiderPlayer(_BasePlayer):
    """Heads to a corner and stops. Tests stealth (silent + still = hard to find)."""
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.target = None
        self._pick_target = True

    def reset(self):
        self.target = None
        self._pick_target = True

    def act(self, env) -> tuple[float, float, float]:
        from envs.silent import WINDOW_SIZE
        if self._pick_target:
            corners = [(40, 40), (WINDOW_SIZE - 40, 40),
                       (40, WINDOW_SIZE - 40), (WINDOW_SIZE - 40, WINDOW_SIZE - 40)]
            # Pick corner furthest from predator at reset time
            pred_x = env.predator.position.x
            pred_y = env.predator.position.y
            self.target = max(corners, key=lambda c: math.hypot(c[0] - pred_x, c[1] - pred_y))
            self._pick_target = False
        dx = self.target[0] - env.player.position.x
        dy = self.target[1] - env.player.position.y
        d = math.hypot(dx, dy)
        if d < 20:
            return 0.0, 0.0, 0.0   # arrived — stay silent and still
        m = d + 1e-6
        return float(dx / m * 0.55), float(dy / m * 0.55), 0.0


class BeaconSeekerPlayer(_BasePlayer):
    """Runs straight toward the exit at full speed. Tests 'naive win' path."""
    def act(self, env) -> tuple[float, float, float]:
        ex = 0.5 * (env.level.exit_zone[0] + env.level.exit_zone[2])
        ey = 0.5 * (env.level.exit_zone[1] + env.level.exit_zone[3])
        dx = ex - env.player.position.x
        dy = ey - env.player.position.y
        d = math.hypot(dx, dy) + 1e-6
        return float(dx / d), float(dy / d), 0.0


class DecoyVoicePlayer(_BasePlayer):
    """Moves randomly but spams voice to decoy predator. Tests the voice
    mechanic under many geometries."""
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks_since_change = 0
        self.voice_on = False
        self.ticks_since_voice_change = 0

    def reset(self):
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.ticks_since_change = 0
        self.voice_on = False
        self.ticks_since_voice_change = 0

    def act(self, env) -> tuple[float, float, float]:
        self.ticks_since_change += 1
        if self.ticks_since_change >= 15:
            self.heading = self.rng.uniform(0, 2 * math.pi)
            self.ticks_since_change = 0
        self.ticks_since_voice_change += 1
        if self.ticks_since_voice_change >= int(self.rng.integers(5, 15)):
            self.voice_on = not self.voice_on
            self.ticks_since_voice_change = 0
        dx = math.cos(self.heading) * 0.5   # move slower while vocal-tricking
        dy = math.sin(self.heading) * 0.5
        voice = 1.0 if self.voice_on else 0.0
        return float(dx), float(dy), float(voice)


def make_player(name: str, seed: int = 0) -> _BasePlayer:
    table = {
        'random_walk':  RandomWalkPlayer,
        'wall_hugger':  WallHuggerPlayer,
        'corner_hider': CornerHiderPlayer,
        'beacon_seeker': BeaconSeekerPlayer,
        'decoy_voice':  DecoyVoicePlayer,
    }
    if name not in table:
        raise KeyError(f"Unknown player {name!r}. Options: {list(table)}")
    return table[name](seed=seed)


PLAYER_MIX = [
    ('random_walk',   0.30),
    ('wall_hugger',   0.25),
    ('corner_hider',  0.20),
    ('beacon_seeker', 0.15),
    ('decoy_voice',   0.10),
]


def sample_player(rng: np.random.Generator, seed: int = 0) -> tuple[str, _BasePlayer]:
    names = [n for n, _ in PLAYER_MIX]
    probs = np.array([p for _, p in PLAYER_MIX], dtype=np.float64)
    name = rng.choice(names, p=probs / probs.sum())
    return name, make_player(name, seed=seed)
