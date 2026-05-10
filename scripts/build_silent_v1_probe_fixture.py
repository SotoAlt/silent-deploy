"""Generate the tier-2 probe fixture for silent_v1 federation.

Records 5 reproducible (audio_obs, ground_truth_state) tuples by stepping
silent.py at fixed seeds and dumping the steady-state moment after a
short warmup. The output npz is consumed by
`aura-federated/federated/games/silent_v1/probe.py` per round to measure
predicted-player-xy MAE — analogous to v9_relay's pymunk-truth probe.

Why pre-compute instead of running silent live in the federation hub:
the audio physics + librosa melspectrogram makes silent.py heavy
(librosa, pyroomacoustics-like math, pygame). The hub should not depend
on those at runtime. Generating the fixture once + persisting as npz
keeps the hub lightweight and the probe deterministic across runs.

Usage:
    PYTHONPATH=. python3 scripts/build_silent_v1_probe_fixture.py \\
        --output /Users/rodrigosoto/repos/aura-federated/federated/\\
games/silent_v1/probe_fixture.npz \\
        --n-scenarios 5 --warmup 20 --seed-base 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless audio + display setup BEFORE pygame imports.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from envs.silent import Silent, WINDOW_SIZE  # noqa: E402
from envs.silent_rooms import get_level  # noqa: E402


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--output", required=True, type=Path)
    pa.add_argument("--n-scenarios", type=int, default=5)
    pa.add_argument("--history", type=int, default=3,
                    help="Number of context frames; fixture captures "
                         "history+1 frames so the probe can encode + "
                         "predict 1 step ahead (matches manifest.history).")
    pa.add_argument("--warmup", type=int, default=20,
                    help="Steps to run before capturing — gets past the "
                         "initial spawn transient into a representative "
                         "mid-game state.")
    pa.add_argument("--seed-base", type=int, default=100)
    pa.add_argument("--level", default="level_01",
                    help="Pass-through to Silent(level=...). Probe "
                         "results are level-conditional, so use the same "
                         "level here that production uses.")
    args = pa.parse_args()

    audio_obs_seqs: list = []  # (n_scenarios, history+1, 4, 64, 50)
    states: list = []          # (n_scenarios, 10)

    print(f"[fixture] building silent_v1 probe fixture: "
          f"{args.n_scenarios} scenarios × {args.history + 1} frames",
          flush=True)

    level_spec = get_level(args.level)
    # Both predator and player share action_dim=3 — see envs/silent.py
    # ACTION_DIM = (move_dx, move_dy, ping_or_voice_amp).
    zero_act = np.zeros(3, dtype=np.float32)

    for s in range(args.n_scenarios):
        seed = args.seed_base + s
        env = Silent(level=level_spec, seed=seed)
        env.reset(seed=seed)

        # Warmup: drive the PLAYER with a deterministic per-seed random
        # walk so each scenario produces a distinct player position by
        # capture time. Otherwise a zero-action warmup leaves the player
        # at the level's fixed spawn point and all N scenarios coincide
        # — the probe would then measure prediction error against a
        # constant target, which any state head trivially memorizes
        # (CLAUDE.md rule #34).
        rng = np.random.default_rng(seed)
        for _ in range(args.warmup):
            move = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            player_act = np.array([move[0], move[1], 0.0], dtype=np.float32)
            env.step(zero_act, who="predator")
            env.step(player_act, who="player")

        frames: list = []
        for k in range(args.history + 1):
            obs = env.get_audio_obs()  # (4, 64, 50) float32
            frames.append(obs.copy())
            env.step(zero_act, who="predator")
            env.step(zero_act, who="player")
        # The last step() in the loop above advances state to t+1 of
        # the last captured frame — that's the predictor's target.
        st = env.get_state().copy()

        audio_obs_seqs.append(np.stack(frames, axis=0))
        states.append(st)
        py = float(st[5] * (WINDOW_SIZE / 2) + WINDOW_SIZE / 2)
        px = float(st[4] * (WINDOW_SIZE / 2) + WINDOW_SIZE / 2)
        print(f"[fixture]   scenario {s} seed={seed}: "
              f"player_xy=({px:.1f}, {py:.1f}) abs", flush=True)

    audio_obs_arr = np.stack(audio_obs_seqs, axis=0).astype(np.float32)
    states_arr = np.stack(states, axis=0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        audio_obs=audio_obs_arr,            # (N, history+1, 4, 64, 50)
        states=states_arr,                  # (N, 10)
        n_scenarios=np.asarray([args.n_scenarios], dtype=np.int32),
        history=np.asarray([args.history], dtype=np.int32),
        seed_base=np.asarray([args.seed_base], dtype=np.int32),
        window_size=np.asarray([WINDOW_SIZE], dtype=np.int32),
    )
    print(f"[fixture] wrote {args.output}", flush=True)
    print(f"[fixture]   audio_obs shape: {audio_obs_arr.shape} "
          f"({audio_obs_arr.nbytes / 1e6:.2f} MB)", flush=True)
    print(f"[fixture]   states shape:    {states_arr.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
