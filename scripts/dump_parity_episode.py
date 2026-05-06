"""Dump a silent episode's intermediate tensors for the silent-local
JS parity harness.

Every tick records:
  - audio sources per-source (beacon, footsteps, voice, pings) before mixing
  - the env-state inputs to mix (predator/player positions, voice_amp,
    silent_ticks, ping list)
  - the 4-channel raw mixed audio (4, 8000)
  - the log-mel output (4, 64, 50)
  - the JEPA forward inputs (emb_ctx, action_history) and outputs (pred_emb)
  - the CEM-chosen action that the predator actually emitted
  - the goal embedding the CEM scored against
  - the decoded state from pred_emb

Saves to a single .npz so the JS harness can load it via Node's binary
parsers. The keys are namespaced by tick: `t{N:03d}_audio_raw`,
`t{N:03d}_mel`, etc.

Usage:
  PYTHONPATH=. python3 scripts/dump_parity_episode.py \\
      --jepa-ckpt checkpoints/silent_v1_3e_ep030.pt \\
      --jepa-head checkpoints/3e_ep030_head_uniform.pt \\
      --level level_01 --seed 42 --n-ticks 100 \\
      --out web/parity_harness/fixtures/parity.npz

The JS harness loads the .npz, runs the same env/audio/mel/forward/CEM
through its pure-JS port, and asserts numerical match within
manifest-defined tolerances. If mel-spec fails parity, the whole
silent-local project is gated.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless audio + display setup BEFORE pygame imports.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from envs.silent import (
    Silent, SAMPLE_RATE, OBS_WINDOW_SEC, N_MELS, N_FFT, HOP_LENGTH,
    UNITS_PER_METER, MAX_SPEED_PLAYER,
    _synth_beacon, _synth_footsteps, _synth_voice, _synth_ping,
)
from envs.silent_rooms import get_level
from envs.silent_jepa_predator_v2 import JEPAPredatorV2


def _snapshot_synth_inputs(env: Silent) -> dict:
    """Capture every input the audio mixer reads, plus the per-source
    raw signals (pre-mixing). The JS port must reproduce both."""
    n_samp = int(SAMPLE_RATE * OBS_WINDOW_SEC)

    # Beacon
    if not getattr(env, '_no_beacon', False):
        beacon_sig = _synth_beacon(n_samp)
        ex = 0.5 * (env.level.exit_zone[0] + env.level.exit_zone[2])
        ey = 0.5 * (env.level.exit_zone[1] + env.level.exit_zone[3])
    else:
        beacon_sig = np.zeros(n_samp, dtype=np.float32)
        ex = ey = 0.0

    # Footsteps (or idle breathing)
    speed = float(np.hypot(env.player.velocity.x, env.player.velocity.y))
    foot_amp = min(1.0, speed / MAX_SPEED_PLAYER)
    if foot_amp > 0.05:
        foot_sig = _synth_footsteps(foot_amp, n_samp)
        foot_used_amp = foot_amp
    elif env._silent_ticks >= 10:
        idle_amp = min(0.4, 0.15 + 0.005 * (env._silent_ticks - 10))
        foot_sig = _synth_footsteps(idle_amp, n_samp)
        foot_used_amp = idle_amp
    else:
        foot_sig = np.zeros(n_samp, dtype=np.float32)
        foot_used_amp = 0.0

    # Voice
    if env._voice_amp > 0.05:
        voice_sig = _synth_voice(env._voice_amp, n_samp)
    else:
        voice_sig = np.zeros(n_samp, dtype=np.float32)

    # Pings — sum all recent pings into one composite source map.
    # Each ping has its own (cx, cy, amplitude, age). For parity we
    # serialize each ping individually + let the JS port mix them.
    pings_data = []
    for ping in env._pings:
        if ping.age > 5:
            continue
        pings_data.append({
            'cx': float(ping.cx), 'cy': float(ping.cy),
            'amp': float(ping.amplitude), 'age': int(ping.age),
            'sig': _synth_ping(ping.amplitude, n_samp),
        })

    return {
        'predator_pos': np.array(
            [float(env.predator.position.x), float(env.predator.position.y)],
            dtype=np.float32),
        'player_pos': np.array(
            [float(env.player.position.x), float(env.player.position.y)],
            dtype=np.float32),
        'player_vel': np.array(
            [float(env.player.velocity.x), float(env.player.velocity.y)],
            dtype=np.float32),
        'voice_amp': np.float32(env._voice_amp),
        'silent_ticks': np.int32(env._silent_ticks),
        'no_beacon': np.bool_(getattr(env, '_no_beacon', False)),
        'beacon_pos': np.array([ex, ey], dtype=np.float32),
        'beacon_sig': beacon_sig,
        'foot_amp_used': np.float32(foot_used_amp),
        'foot_sig': foot_sig,
        'voice_sig': voice_sig,
        'pings': pings_data,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa-ckpt', required=True)
    ap.add_argument('--jepa-head', required=True)
    ap.add_argument('--level', default='level_01')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-ticks', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-beacon', action='store_true')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'[dump] level={args.level} seed={args.seed} n_ticks={args.n_ticks}')
    env = Silent(level=get_level(args.level), seed=args.seed)
    if args.no_beacon:
        env._no_beacon = True
    predator = JEPAPredatorV2(
        ckpt_path=args.jepa_ckpt, head_path=args.jepa_head, device='cpu')

    # Numpy savez accepts kwargs; collect everything in a flat dict
    # keyed by `t{NNN}_field`. For the pings list, store per-ping
    # entries as `t{NNN}_ping{K}_*`.
    bundle: dict = {
        'meta_sample_rate': np.int32(SAMPLE_RATE),
        'meta_obs_window_sec': np.float32(OBS_WINDOW_SEC),
        'meta_n_mels': np.int32(N_MELS),
        'meta_n_fft': np.int32(N_FFT),
        'meta_hop_length': np.int32(HOP_LENGTH),
        'meta_units_per_meter': np.float32(UNITS_PER_METER),
        'meta_n_ticks': np.int32(args.n_ticks),
        'meta_seed': np.int32(args.seed),
        'meta_level': np.array(args.level),
        'meta_jepa_ckpt': np.array(args.jepa_ckpt),
    }

    # Drive the env: scripted player wiggles around to vary the obs.
    rng = np.random.default_rng(args.seed)
    for tick in range(args.n_ticks):
        prefix = f't{tick:04d}'

        # Snapshot synth inputs BEFORE get_audio_obs so we know what the
        # JS port must produce.
        snap = _snapshot_synth_inputs(env)
        bundle[f'{prefix}_predator_pos'] = snap['predator_pos']
        bundle[f'{prefix}_player_pos'] = snap['player_pos']
        bundle[f'{prefix}_player_vel'] = snap['player_vel']
        bundle[f'{prefix}_voice_amp'] = snap['voice_amp']
        bundle[f'{prefix}_silent_ticks'] = snap['silent_ticks']
        bundle[f'{prefix}_no_beacon'] = snap['no_beacon']
        bundle[f'{prefix}_beacon_pos'] = snap['beacon_pos']
        bundle[f'{prefix}_beacon_sig'] = snap['beacon_sig']
        bundle[f'{prefix}_foot_amp_used'] = snap['foot_amp_used']
        bundle[f'{prefix}_foot_sig'] = snap['foot_sig']
        bundle[f'{prefix}_voice_sig'] = snap['voice_sig']
        bundle[f'{prefix}_n_pings'] = np.int32(len(snap['pings']))
        for k, p in enumerate(snap['pings']):
            bundle[f'{prefix}_ping{k}_cx']  = np.float32(p['cx'])
            bundle[f'{prefix}_ping{k}_cy']  = np.float32(p['cy'])
            bundle[f'{prefix}_ping{k}_amp'] = np.float32(p['amp'])
            bundle[f'{prefix}_ping{k}_sig'] = p['sig']

        # Mix the per-source signals into the 4-channel audio buffer
        # using the same logic as silent.py:480-495. Saving the post-mix
        # `audio_raw` lets the JS harness validate mel-spec INDEPENDENTLY
        # of mixing — diagnoses whether failures are STFT/mel or mixing.
        n_samp = int(SAMPLE_RATE * OBS_WINDOW_SEC)
        audio_raw = np.zeros((4, n_samp), dtype=np.float32)
        lobe_axes = np.array([
            [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        ], dtype=np.float32)
        sources = []
        if not snap['no_beacon']:
            sources.append((snap['beacon_pos'], snap['beacon_sig']))
        if snap['foot_amp_used'] > 0.0:
            sources.append((snap['player_pos'], snap['foot_sig']))
        if snap['voice_amp'] > 0.05:
            sources.append((snap['player_pos'], snap['voice_sig']))
        for p in snap['pings']:
            sources.append((np.array([p['cx'], p['cy']], dtype=np.float32), p['sig']))
        for src_pos, sig in sources:
            dx = float(src_pos[0]) - float(snap['predator_pos'][0])
            dy = float(src_pos[1]) - float(snap['predator_pos'][1])
            d_pix = float(np.hypot(dx, dy)) + 1e-6
            d_m = d_pix / UNITS_PER_METER
            atten = 1.0 / max(0.3, d_m)
            sd = np.array([dx / d_pix, dy / d_pix], dtype=np.float32)
            for ch in range(4):
                dot = float(np.dot(sd, lobe_axes[ch]))
                gain = max(0.0, (1.0 + dot) * 0.5) ** 2
                if gain > 0.0:
                    audio_raw[ch] += (atten * gain) * sig
        bundle[f'{prefix}_audio_raw'] = audio_raw

        # Now run get_audio_obs to capture the canonical mel output.
        mel = env.get_audio_obs()                             # (4, 64, 50)
        bundle[f'{prefix}_mel'] = mel

        # JEPA forward fixture: predator builds its own emb_ctx +
        # action_history internally inside .act(). Capture both before
        # the act so the JS harness can replay.
        # The simplest way is to mirror the predator's setup here.
        # Append obs to the predator's obs buf BEFORE calling act, so
        # the buf has length history when act runs.
        predator._obs_buf.append(mel)
        if len(predator._obs_buf) > predator.history + 2:
            predator._obs_buf.pop(0)

        # We can only run a meaningful forward once we have enough history.
        if len(predator._obs_buf) >= predator.history:
            import torch
            obs_seq = predator._obs_buf[-predator.history:]
            obs_tensor = predator._obs_tensor(obs_seq)        # (1, H, 4, 224, 224)
            with torch.no_grad():
                emb_ctx = predator.model.encode(obs_tensor)   # (1, H, D)
            bundle[f'{prefix}_emb_ctx'] = emb_ctx.cpu().numpy()

            # V2 predator uses state-head decoded |predator_xy - player_xy|
            # as the cost; no goal_emb. Just capture the chosen action.
            action = predator.act(env)                        # (3,) tuple
            bundle[f'{prefix}_action'] = np.array(action, dtype=np.float32)

            # Step env with chosen predator action + scripted player movement
            scripted_vx = float(rng.uniform(-0.5, 0.5))
            scripted_vy = float(rng.uniform(-0.5, 0.5))
            scripted_voice = float(rng.uniform(0.0, 0.3))
            env.step(np.array([scripted_vx, scripted_vy, scripted_voice], dtype=np.float32),
                     who='player')
            if not env.done:
                env.step(np.array(action, dtype=np.float32), who='predator')
        else:
            # Warmup: predator coasts. Step env with random action.
            scripted_vx = float(rng.uniform(-0.5, 0.5))
            scripted_vy = float(rng.uniform(-0.5, 0.5))
            scripted_voice = float(rng.uniform(0.0, 0.3))
            env.step(np.array([scripted_vx, scripted_vy, scripted_voice], dtype=np.float32),
                     who='player')

        if env.done:
            print(f'[dump] env done at tick {tick}; truncating')
            bundle['meta_n_ticks'] = np.int32(tick + 1)
            break

        if (tick + 1) % 20 == 0:
            print(f'[dump]   tick {tick + 1}/{args.n_ticks}')

    print(f'[dump] writing {out_path} ({len(bundle)} keys)')
    np.savez_compressed(out_path, **bundle)
    print(f'[dump] OK — {out_path.stat().st_size / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
