"""Silent — room specs (handcrafted + procedural).

Handcrafted: 5 fixed levels used for evaluation and Phase 0 play.
Procedural: infinite variety for training data generation (Phase 1).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from envs.silent import LevelSpec, PillarSpec, WINDOW_SIZE


# ---- Handcrafted evaluation set -------------------------------------------
LEVELS: dict[str, LevelSpec] = {
    'level_01': LevelSpec(
        name='level_01',
        pillars=[],
        pred_start=(80.0, 80.0),
        player_start=(430.0, 430.0),
    ),
    'level_02': LevelSpec(
        name='level_02',
        pillars=[PillarSpec(x=WINDOW_SIZE / 2, y=WINDOW_SIZE / 2, radius=32)],
        pred_start=(80.0, 80.0),
        player_start=(430.0, 430.0),
    ),
    'level_03': LevelSpec(
        name='level_03',
        pillars=[
            PillarSpec(x=140, y=260, radius=22),
            PillarSpec(x=260, y=260, radius=22),
            PillarSpec(x=380, y=260, radius=22),
        ],
        pred_start=(80.0, 80.0),
        player_start=(430.0, 430.0),
    ),
    'level_04': LevelSpec(
        name='level_04',
        pillars=[
            PillarSpec(x=160, y=150, radius=25),
            PillarSpec(x=320, y=200, radius=25),
            PillarSpec(x=200, y=340, radius=25),
            PillarSpec(x=380, y=380, radius=25),
            PillarSpec(x=90, y=260, radius=25),
        ],
        pred_start=(60.0, 60.0),
        player_start=(440.0, 440.0),
    ),
    'level_05': LevelSpec(
        name='level_05',
        pillars=[
            PillarSpec(x=WINDOW_SIZE / 2, y=60, radius=30),
            PillarSpec(x=WINDOW_SIZE / 2, y=140, radius=30),
            PillarSpec(x=WINDOW_SIZE / 2, y=220, radius=30),
            PillarSpec(x=WINDOW_SIZE / 2, y=400, radius=30),
            PillarSpec(x=WINDOW_SIZE / 2, y=460, radius=30),
        ],
        pred_start=(80.0, WINDOW_SIZE - 60),
        player_start=(WINDOW_SIZE - 80, WINDOW_SIZE - 60),
    ),
}


def get_level(name: str) -> LevelSpec:
    if name not in LEVELS:
        raise KeyError(f"Unknown level '{name}'. Available: {list(LEVELS)}")
    return LEVELS[name]


def list_levels() -> list[str]:
    return list(LEVELS.keys())


# ---- Procedural room generator (for Phase 1 training data) ----------------
# Exit zone is always 80x80 and lives in one of the 4 corners. Agent starts are
# constrained to opposite halves from the exit. Pillars are 0-4, placed to not
# overlap each other, agent starts, or the exit.

_EXIT_W, _EXIT_H = 80, 80
_EXIT_MARGIN = 30   # keep exit this far from walls
_MIN_PAIR_DIST = 140  # min distance between predator start and player start
_MIN_AGENT_CLEAR = 60  # agents must start this far from any pillar
_AGENT_CORNER_CLEAR = 40  # agents must be this far from walls


def _rand_exit_zone(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Pick a random corner for the exit zone."""
    corners = ['NE', 'NW', 'SE', 'SW']
    c = rng.choice(corners)
    if c == 'NE':
        x0 = WINDOW_SIZE - _EXIT_W - _EXIT_MARGIN
        y0 = _EXIT_MARGIN
    elif c == 'NW':
        x0 = _EXIT_MARGIN
        y0 = _EXIT_MARGIN
    elif c == 'SE':
        x0 = WINDOW_SIZE - _EXIT_W - _EXIT_MARGIN
        y0 = WINDOW_SIZE - _EXIT_H - _EXIT_MARGIN
    else:
        x0 = _EXIT_MARGIN
        y0 = WINDOW_SIZE - _EXIT_H - _EXIT_MARGIN
    return (float(x0), float(y0), float(x0 + _EXIT_W), float(y0 + _EXIT_H))


def _too_close(x: float, y: float, points: list, min_dist: float) -> bool:
    return any(math.hypot(x - p[0], y - p[1]) < min_dist for p in points)


def _in_exit_zone(x: float, y: float, zone: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = zone
    return x0 - 20 <= x <= x1 + 20 and y0 - 20 <= y <= y1 + 20


def random_level(seed: int) -> LevelSpec:
    """Generate a procedural level. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    exit_zone = _rand_exit_zone(rng)

    # Pillar count + placement
    n_pillars = int(rng.integers(0, 5))  # 0..4
    pillars: List[PillarSpec] = []
    existing = []
    for _ in range(n_pillars * 5):  # try up to 5x the count (may fail early)
        if len(pillars) >= n_pillars:
            break
        r = float(rng.uniform(20.0, 36.0))
        x = float(rng.uniform(60, WINDOW_SIZE - 60))
        y = float(rng.uniform(60, WINDOW_SIZE - 60))
        if _too_close(x, y, existing, r * 2 + 20):
            continue
        if _in_exit_zone(x, y, exit_zone):
            continue
        pillars.append(PillarSpec(x=x, y=y, radius=r))
        existing.append((x, y))

    # Predator + player starts — opposite halves, far from pillars + exit
    ex_cx = 0.5 * (exit_zone[0] + exit_zone[2])
    ey_cy = 0.5 * (exit_zone[1] + exit_zone[3])

    # Place player "near" exit side (has to cross room; or start near exit too — randomize)
    # Player always starts somewhere other than the exit corner, roughly diametrically opposite.
    def sample_start(preferred_region: str) -> Tuple[float, float]:
        for _ in range(200):
            if preferred_region == 'opposite_exit':
                # Mirror the exit corner
                x = WINDOW_SIZE - ex_cx + float(rng.normal(0, 30))
                y = WINDOW_SIZE - ey_cy + float(rng.normal(0, 30))
            elif preferred_region == 'near_exit':
                x = ex_cx + float(rng.normal(0, 40))
                y = ey_cy + float(rng.normal(0, 40))
            else:  # anywhere valid
                x = float(rng.uniform(_AGENT_CORNER_CLEAR, WINDOW_SIZE - _AGENT_CORNER_CLEAR))
                y = float(rng.uniform(_AGENT_CORNER_CLEAR, WINDOW_SIZE - _AGENT_CORNER_CLEAR))
            # Clamp to bounds
            x = max(_AGENT_CORNER_CLEAR, min(WINDOW_SIZE - _AGENT_CORNER_CLEAR, x))
            y = max(_AGENT_CORNER_CLEAR, min(WINDOW_SIZE - _AGENT_CORNER_CLEAR, y))
            # Check clear of pillars + exit
            if _in_exit_zone(x, y, exit_zone):
                continue
            if any(math.hypot(x - p.x, y - p.y) < p.radius + _MIN_AGENT_CLEAR for p in pillars):
                continue
            return x, y
        # Fallback: center
        return float(WINDOW_SIZE / 2), float(WINDOW_SIZE / 2)

    pred_start = sample_start('opposite_exit')
    # Player start must also be far from predator
    for attempt in range(200):
        player_start = sample_start(rng.choice(['near_exit', 'anywhere']))
        if math.hypot(pred_start[0] - player_start[0],
                      pred_start[1] - player_start[1]) >= _MIN_PAIR_DIST:
            break

    return LevelSpec(
        name=f'proc_{seed}',
        pillars=pillars,
        pred_start=pred_start,
        player_start=player_start,
        exit_zone=exit_zone,
        time_limit_sec=60.0,
    )


def random_levels(n: int, seed0: int = 0):
    """Yield n procedural levels."""
    for i in range(n):
        yield random_level(seed0 + i)


if __name__ == '__main__':
    # Sanity check: build 5 procedural levels and print their layouts.
    for lvl in random_levels(5, seed0=42):
        print(f"{lvl.name}: pillars={len(lvl.pillars)} pred={lvl.pred_start} "
              f"player={lvl.player_start} exit={lvl.exit_zone}")
