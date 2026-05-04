"""Silent — Phase 0 server (env + scripted predator, NO ML).

Owns the Silent env + runs the predator AI loop on the server side. The
client sends the player's action every tick (WASD + voice amp). The server
steps physics, advances the predator heuristic, and returns a rendered frame +
state + ping info for the client to visualize and spatialize.

Usage:
    PYTHONPATH=. python -m world_model.infer_silent_env --port 8801
    Open http://localhost:8801/

Messages (JSON text frames):
  Client -> Server:
    {type: "new_match", level: "level_01", predator: "oracle"|"echolocation", seed: 0}
    {type: "player_action", vx, vy, voice_amp}    # all in [-1..1] / [0..1]
    {type: "reset"}
  Server -> Client:
    {type: "frame", frame: <b64 PNG>, state: [10], done: bool, tick: int,
     elapsed_sec: float, time_limit_sec: float, win: "player"|"predator"|null,
     pings: [{x, y, r, amplitude}, ...], predator_mode: "...",
     predator_pos: [x, y], player_pos: [x, y], exit_zone: [x0,y0,x1,y1]}
    {type: "error", error: "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

# Must precede pygame import (keeps macOS dock icon / Alt-Tab clean)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import math as _math
from dataclasses import replace as _dc_replace
from envs.silent import Silent, WINDOW_SIZE
from envs.silent_rooms import LEVELS, get_level
from envs.silent_predators import make_predator


def _randomize_exit(env: Silent, rng: np.random.Generator,
                    margin: float = 50.0,
                    min_from_player: float = 220.0,
                    min_from_predator: float = 180.0,
                    exit_half: float = 40.0) -> None:
    """Randomize ONLY the exit position. Keeps predator/player at level
    defaults — those are part of what makes the JEPA's deployment-time
    behavior stable. Exit must be at least `min_from_player` from the
    player (so the run is meaningful) and `min_from_predator` from the
    predator (so the predator can't camp on the exit at spawn).
    """
    px = float(env.player.position.x)
    py = float(env.player.position.y)
    rx = float(env.predator.position.x)
    ry = float(env.predator.position.y)

    best = None
    best_score = -1.0
    for _ in range(200):
        ex = float(rng.uniform(margin, WINDOW_SIZE - margin))
        ey = float(rng.uniform(margin, WINDOW_SIZE - margin))
        d_player = _math.hypot(ex - px, ey - py)
        d_predator = _math.hypot(ex - rx, ey - ry)
        if d_player >= min_from_player and d_predator >= min_from_predator:
            best = (ex, ey)
            break
        # Track best-so-far in case constraints are unsatisfiable on a tight map
        score = min(d_player / min_from_player, d_predator / min_from_predator)
        if score > best_score:
            best_score = score
            best = (ex, ey)

    ex, ey = best
    new_exit = (ex - exit_half, ey - exit_half, ex + exit_half, ey + exit_half)
    env.level = _dc_replace(env.level, exit_zone=new_exit)
    print(f"[exit-rand] player=({px:.0f},{py:.0f}) "
          f"pred=({rx:.0f},{ry:.0f}) exit=({ex:.0f},{ey:.0f})  "
          f"dPlayer={_math.hypot(ex-px, ey-py):.0f} "
          f"dPred={_math.hypot(ex-rx, ey-ry):.0f}", flush=True)


app = FastAPI(title="Silent — Phase 0 Env Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_CLIENT_DIR = _ROOT / 'client' / 'silent'
app.mount("/static", StaticFiles(directory=str(_CLIENT_DIR)), name="static")

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
async def root():
    return FileResponse(_CLIENT_DIR / "index.html", headers=_NO_CACHE)


@app.get("/main.js")
async def main_js():
    return FileResponse(
        _CLIENT_DIR / "main.js",
        media_type="application/javascript",
        headers=_NO_CACHE,
    )


@app.get("/audio.js")
async def audio_js():
    path = _CLIENT_DIR / "audio.js"
    if path.exists():
        return FileResponse(path, media_type="application/javascript", headers=_NO_CACHE)
    return HTMLResponse(
        "// audio.js not built yet (Phase 0.3)",
        media_type="application/javascript",
        headers=_NO_CACHE,
    )


@app.get("/levels")
async def list_levels():
    return {"levels": list(LEVELS.keys())}


def _encode_frame(frame: np.ndarray) -> str:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _frame_payload(env: Silent, predator_mode: str) -> dict:
    pixels = env.render(size=512)
    return {
        'type': 'frame',
        'frame': _encode_frame(pixels),
        'state': env.get_state().tolist(),
        'done': env.done,
        'win': env.win,
        'tick': env.tick,
        'elapsed_sec': env._elapsed,
        'time_limit_sec': env.level.time_limit_sec,
        'pings': [
            {'x': float(p.cx), 'y': float(p.cy), 'r': float(p.radius),
             'amplitude': float(p.amplitude), 'age': int(p.age)}
            for p in env._pings
        ],
        'predator_mode': predator_mode,
        'predator_pos': [float(env.predator.position.x), float(env.predator.position.y)],
        'player_pos': [float(env.player.position.x), float(env.player.position.y)],
        'exit_zone': list(env.level.exit_zone),
        'level': env.level.name,
        # Scoring
        'score': float(env.score),
        'items_total': len(env._items),
        'items_collected': sum(1 for it in env._items if it[2] > 0.5),
        'proximity_active': bool(getattr(env, '_proximity_active', False)),
    }


def _make_survival(env: Silent, time_limit_sec: float = 90.0):
    """Convert a match to survival mode: keep the beacon AT ITS TRAINING
    POSITION (so the JEPA's audio distribution matches), but collapse
    the exit zone to zero area so the player can't trigger the
    'reached-exit' win. Win condition becomes: don't get caught for
    time_limit_sec.

    Why not move the beacon? Earlier attempt (1e6 px off-canvas) made
    the beacon 1/d-silent → encoder OOD (no_beacon ablation result) →
    predator collapsed. Keeping it audible at training cardinal
    direction preserves encoder distribution.
    """
    # Get the original exit center, then collapse the zone to a point
    # at that exact center. Beacon stays where it was; reach-check
    # requires player at exact coords (impossible).
    x0, y0, x1, y1 = env.level.exit_zone
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    point_exit = (cx, cy, cx, cy)   # zero-area rect → unreachable
    env.level = _dc_replace(env.level, exit_zone=point_exit,
                            time_limit_sec=time_limit_sec)


def _apply_ablation(env: Silent, ablation: str):
    """Monkey-patch env.get_audio_obs to apply an ablation, returning a
    callable that restores the original. Idempotent: calling with 'none'
    is a no-op. Used to let the user A/B audio conditions live."""
    if not hasattr(env, '_orig_get_audio_obs'):
        env._orig_get_audio_obs = env.get_audio_obs

    if ablation == 'none' or ablation == '':
        env.get_audio_obs = env._orig_get_audio_obs
        return

    if ablation == 'mute':
        def _muted():
            return np.zeros_like(env._orig_get_audio_obs())
        env.get_audio_obs = _muted

    elif ablation == 'one_ear':
        def _one_ear():
            obs = env._orig_get_audio_obs()
            out = np.zeros_like(obs)
            out[0] = obs[0]   # keep N
            return out
        env.get_audio_obs = _one_ear

    elif ablation == 'no_beacon':
        def _no_beacon():
            orig_exit = env.level.exit_zone
            env.level = _dc_replace(env.level, exit_zone=(1e7, 1e7, 1e7+80, 1e7+80))
            try:
                obs = env._orig_get_audio_obs()
            finally:
                env.level = _dc_replace(env.level, exit_zone=orig_exit)
            return obs
        env.get_audio_obs = _no_beacon


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    env: Silent | None = None
    predator = None
    predator_mode = 'oracle'
    ablation = 'none'   # 'none' | 'mute' | 'one_ear' | 'no_beacon'

    # Async planner state: the predator's CEM is too slow on shared vCPUs
    # to run inside the request-response loop (~200ms vs 100ms tick budget).
    # Background task continuously plans; WS handler uses whatever action
    # is most recent. Plan is at most 1 tick stale — fine for hunting since
    # player movement is bounded.
    latest_action: list = [0.0, 0.0, 0.0]   # [pdx, pdy, ping]; mutated in place
    planner_task: asyncio.Task | None = None
    planner_stop = asyncio.Event()

    async def planner_loop():
        # Read env + predator from the closure; they get rebound on new_match.
        # Intentionally read-only on env (CEM only needs current state to plan),
        # so the WS handler's env.step calls don't conflict beyond rare reads
        # of mid-step state which produce mostly-coherent plans anyway.
        while not planner_stop.is_set():
            if env is None or predator is None or env.done:
                await asyncio.sleep(0.05)
                continue
            try:
                pdx, pdy, ping = await asyncio.to_thread(predator.act, env)
                latest_action[0] = float(pdx)
                latest_action[1] = float(pdy)
                latest_action[2] = float(ping)
            except Exception as ex:
                print(f"[planner] error: {ex}", flush=True)
                await asyncio.sleep(0.1)

    try:
        while True:
            msg_raw = await ws.receive_text()
            try:
                msg = json.loads(msg_raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({'type': 'error', 'error': 'bad_json'}))
                continue

            t = msg.get('type')
            if t == 'new_match':
                try:
                    level = get_level(msg.get('level', 'level_01'))
                except KeyError as e:
                    await ws.send_text(json.dumps({'type': 'error', 'error': str(e)}))
                    continue
                seed = int(msg.get('seed', 0))
                env = Silent(level=level, seed=seed)
                # Random goal — only if the client requested it AND the
                # selected predator was trained for it. Canonical (jepa_v2)
                # was trained with fixed exit; jepa_test1 (Phase 3C) was
                # trained with random exit.
                random_goal = bool(msg.get('random_goal', False))
                if random_goal:
                    _randomize_exit(env, np.random.default_rng(seed * 7919 + 13))
                game_mode = msg.get('mode', 'escape')   # 'escape' | 'survival'
                if game_mode == 'survival':
                    _make_survival(env)
                    # Scoring incentives so standing still isn't optimal:
                    #   - 5 items spawn (must move to grab them)
                    #   - +bonus per tick when within 220 px of predator
                    env.spawn_items(n=8, rng=np.random.default_rng(seed * 1009 + 71))
                    env.set_proximity(True)
                predator_mode = msg.get('predator', 'oracle')
                try:
                    predator = await asyncio.to_thread(_get_predator, predator_mode, seed)
                except ValueError as e:
                    await ws.send_text(json.dumps({'type': 'error', 'error': str(e)}))
                    continue
                _apply_ablation(env, ablation)
                # Reset stale plan so the new match's first tick doesn't use
                # the previous match's last predator action.
                latest_action[0] = 0.0
                latest_action[1] = 0.0
                latest_action[2] = 0.0
                # Warm up: run ONE CEM synchronously so latest_action is
                # populated before the player sees the first frame. Without
                # this, predator stands still for the first ~1-2 ticks while
                # the background planner does its first pass.
                try:
                    pdx0, pdy0, ping0 = await asyncio.to_thread(predator.act, env)
                    latest_action[0] = float(pdx0)
                    latest_action[1] = float(pdy0)
                    latest_action[2] = float(ping0)
                except Exception as ex:
                    print(f"[warmup] error: {ex}", flush=True)
                # Spawn the background planner once we have an env + predator.
                if planner_task is None:
                    planner_task = asyncio.create_task(planner_loop())
                await ws.send_text(json.dumps(_frame_payload(env, predator_mode)))

            elif t == 'set_ablation':
                ablation = msg.get('kind', 'none')
                if env is not None:
                    _apply_ablation(env, ablation)
                print(f"[ablation] {ablation}", flush=True)

            elif t == 'reset':
                if env is None:
                    await ws.send_text(json.dumps({'type': 'error', 'error': 'no_env'}))
                    continue
                env.reset()
                if predator is not None:
                    predator.reset()
                await ws.send_text(json.dumps(_frame_payload(env, predator_mode)))

            elif t == 'player_action':
                if env is None or predator is None:
                    await ws.send_text(json.dumps({'type': 'error', 'error': 'no_env'}))
                    continue
                vx = float(msg.get('vx', 0.0))
                vy = float(msg.get('vy', 0.0))
                voice = float(msg.get('voice_amp', 0.0))
                if not env.done:
                    # Apply the player's action first
                    env.step(np.array([vx, vy, voice], dtype=np.float32), who='player')
                    # Then the predator's action — read whatever the background
                    # planner has most recently produced. Plan is at most ~1 tick
                    # stale on slow CPUs, which is fine for hunting.
                    if not env.done:
                        env.step(
                            np.array([latest_action[0], latest_action[1], latest_action[2]],
                                     dtype=np.float32),
                            who='predator')
                await ws.send_text(json.dumps(_frame_payload(env, predator_mode)))

            else:
                await ws.send_text(json.dumps({'type': 'error', 'error': f'unknown_type:{t}'}))

    except WebSocketDisconnect:
        return
    finally:
        # Clean up the background planner so it doesn't leak across reconnects.
        planner_stop.set()
        if planner_task is not None:
            planner_task.cancel()


_JEPA_CKPT_PATH: str | None = None   # canonical baseline (jepa_v2)
_JEPA_HEAD_PATH: str | None = None
_JEPA_TEST_CKPTS: dict = {}   # name → (ckpt, head); see --jepa-test

# Process-level predator cache. Loading a JEPA checkpoint on CPU is 3-8s
# (read 100MB ckpt + build ViT + warm up matmul). Cache by predator name
# so switching variants in the UI is instant after the first load.
_PREDATOR_CACHE: dict = {}   # name → predator instance

def _get_predator(predator_mode: str, seed: int):
    cached = _PREDATOR_CACHE.get(predator_mode)
    if cached is not None:
        try:
            cached.reset()
        except Exception:
            pass
        return cached
    pred = make_predator(
        predator_mode, seed=seed, jepa_device='cpu',
        jepa_ckpt=_JEPA_CKPT_PATH, jepa_head=_JEPA_HEAD_PATH,
        jepa_test_ckpts=_JEPA_TEST_CKPTS,
    )
    _PREDATOR_CACHE[predator_mode] = pred
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8801)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--jepa-ckpt', default=None,
                    help='Canonical JEPA ckpt for predator=jepa_v2')
    ap.add_argument('--jepa-head', default=None,
                    help='Canonical JEPA head for predator=jepa_v2')
    ap.add_argument('--jepa-test', action='append', default=[],
                    metavar='NAME:CKPT:HEAD',
                    help='Add a test variant. Repeatable. e.g. '
                         "--jepa-test jepa_test1:path/to/ckpt.pt:path/to/head.pt")
    args = ap.parse_args()
    global _JEPA_CKPT_PATH, _JEPA_HEAD_PATH, _JEPA_TEST_CKPTS
    _JEPA_HEAD_PATH = args.jepa_head
    _JEPA_CKPT_PATH = args.jepa_ckpt
    _JEPA_TEST_CKPTS = {}
    for spec in args.jepa_test:
        try:
            name, ckpt, head = spec.split(':', 2)
            _JEPA_TEST_CKPTS[name] = (ckpt, head)
            print(f"  test variant: {name} → {ckpt} + {head}")
        except ValueError:
            print(f"  ! ignoring malformed --jepa-test {spec!r}, want NAME:CKPT:HEAD")
    print(f"Silent env server — http://{args.host}:{args.port}/")
    print(f"Levels: {list(LEVELS.keys())}")
    # Pre-load the canonical (jepa_v2) predator so the first match avoids
    # the 3-8s checkpoint load. Other variants load lazily on first use,
    # cached process-wide thereafter.
    if _JEPA_CKPT_PATH and _JEPA_HEAD_PATH:
        try:
            print("[startup] pre-loading canonical predator (jepa_v2)…", flush=True)
            import time as _t
            t0 = _t.time()
            _get_predator('jepa_v2', seed=0)
            print(f"[startup] canonical loaded in {_t.time()-t0:.1f}s", flush=True)
        except Exception as ex:
            print(f"[startup] canonical pre-load failed: {ex}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
