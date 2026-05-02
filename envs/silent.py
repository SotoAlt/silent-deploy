"""Silent — 2D adversarial env for the JEPA-as-echolocating-predator task.

The human player is the prey. Both agents live in a 2D pymunk space. The
predator observes the scene only via a 4-channel mel-spectrogram built from a
pyroomacoustics simulation of its own pings + the player's footsteps/voice.
The player sees a top-down pixel view rendered on the server.

Architecturally mirrors world_model/envs/relay.py from the main AURA repo:
  - 512x512 pymunk physics space
  - Static outer walls + per-level static pillars
  - Two kinematic circular agents (predator, player)
  - step(action, who) pattern
  - render() returns HxWx3 uint8

Differences from RELAY:
  - Audio observation via pyroomacoustics (see `get_audio_obs`) — the predator's
    entire sensorium is audio; visual render is for the HUMAN viewer only.
  - No T-block. The player IS the target.
  - Additional "voice amp" channel in the action space (player only).
  - Exit door as a terminal win-zone for the player.

Phase 0 scope: physics + render + oracle/echolocation heuristics + stubbed audio.
The pyroomacoustics integration fills in get_audio_obs in the next commit.

State layout (10-D normalized, x'/y' = (val - 256) / 256):
    [0..3]  predator pose: pred_x', pred_y', pred_vx, pred_vy
    [4..7]  player pose:   player_x', player_y', player_vx, player_vy
    [8]     voice_amp in [0, 1] (last player vocalization this tick)
    [9]     distance_to_exit normalized in [0, 1]
"""
from __future__ import annotations

import math
_math_hypot_xy = math.hypot   # micro-cache for the per-tick proximity check
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame
import pymunk


# World / render constants mirror RELAY for pixel-parity ease
WINDOW_SIZE = 512
RENDER_SIZE = 512
DT = 0.01
CONTROL_HZ = 10
WALL_RADIUS = 2

# Audio simulation constants (pyroomacoustics + mel-spec)
UNITS_PER_METER = 100.0   # 512 world units = 5.12 m room — realistic acoustic scale
SAMPLE_RATE = 16000       # speech-quality, fast to simulate
OBS_WINDOW_SEC = 0.5      # 500 ms per observation → 8000 samples
N_MELS = 64               # AST-style mel-bin count
HOP_LENGTH = 160          # 10 ms hop → ~50 time frames
N_FFT = 512               # short FFT for 10-ms granularity
MIC_EAR_OFFSET = 0.05     # meters between predator center and each of 4 mics
# Note: pillars are NOT modeled in the acoustic simulation (MVP simplification).
# pyroomacoustics' image-source method requires convex rooms; circular pillars
# would punch holes. Pillars remain physical obstacles for movement only.
# Phase 1+ may revisit with ray-tracing if needed.

# Agent physics
PREDATOR_RADIUS = 14
PLAYER_RADIUS = 12
AGENT_MASS = 1.0
MAX_SPEED_PLAYER = 150.0   # base player speed; equals predator → spacing/audio matter
MAX_SPEED_PRED = 150.0     # match player base; player only escapes when far + smart
CATCH_DISTANCE = PREDATOR_RADIUS + PLAYER_RADIUS  # contact = caught

# Palette — dark chamber, single teal accent (matches Observatory aesthetic)
BG_COLOR = (14, 11, 8)
GRID_COLOR = (26, 21, 18)
WALL_COLOR = (38, 32, 26)
PILLAR_COLOR = (52, 44, 36)
PILLAR_RIM = (86, 74, 62)
PLAYER_COLOR = (63, 184, 161)     # teal — you
PLAYER_HALO = (30, 92, 80)
PREDATOR_COLOR = (196, 114, 101)  # coral — danger
PREDATOR_HALO = (110, 56, 48)
EXIT_COLOR = (217, 154, 95)       # warm amber
PING_COLOR = (196, 114, 101)
VOICE_COLOR = (63, 184, 161)


ACTION_DIM = 3   # (move_dx, move_dy, ping_or_voice_amp) in [-1, 1] / [0, 1]
STATE_DIM = 10


@dataclass
class PillarSpec:
    x: float
    y: float
    radius: float = 22.0


@dataclass
class LevelSpec:
    name: str
    pillars: List[PillarSpec] = field(default_factory=list)
    pred_start: Tuple[float, float] = (80.0, 80.0)
    player_start: Tuple[float, float] = (430.0, 430.0)
    exit_zone: Tuple[float, float, float, float] = (WINDOW_SIZE - 110, 30, WINDOW_SIZE - 30, 110)  # (x0, y0, x1, y1)
    time_limit_sec: float = 60.0


class _Ping:
    """Transient visual record of a predator ping — the audio effect is handled
    elsewhere (pyroomacoustics for predator's own observation, Web Audio on the
    client for the player's ears). This struct only keeps the visual ripple."""
    __slots__ = ('cx', 'cy', 'radius', 'age', 'amplitude', 'alive', 'speed', 'max_radius')

    def __init__(self, cx: float, cy: float, amplitude: float, speed: float = 900.0, max_radius: float = 900.0):
        self.cx = cx
        self.cy = cy
        self.amplitude = amplitude
        self.radius = 0.0
        self.age = 0
        self.speed = speed
        self.max_radius = max_radius
        self.alive = True

    def advance(self, dt: float) -> None:
        self.radius += self.speed * dt
        self.age += 1
        if self.radius > self.max_radius:
            self.alive = False


class Silent:
    """Adversarial 2D env. The JEPA (later) or a scripted predator drives the
    predator; the human drives the player. One shared pymunk space."""

    def __init__(self, level: LevelSpec, seed: int = 0):
        self.level = level
        self._rng = np.random.default_rng(seed)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.55   # slight friction so agents decelerate

        # Outer walls
        for a, b in [((0, 0), (0, WINDOW_SIZE)),
                     ((0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE)),
                     ((WINDOW_SIZE, WINDOW_SIZE), (WINDOW_SIZE, 0)),
                     ((WINDOW_SIZE, 0), (0, 0))]:
            seg = pymunk.Segment(self.space.static_body, a, b, WALL_RADIUS)
            seg.elasticity = 0.1
            seg.friction = 0.8
            self.space.add(seg)

        # Pillars (static)
        self.pillars: List[pymunk.Circle] = []
        for p in level.pillars:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (p.x, p.y)
            shape = pymunk.Circle(body, p.radius)
            shape.elasticity = 0.1
            shape.friction = 0.9
            self.space.add(body, shape)
            self.pillars.append(shape)

        # Predator body (kinematic-like: mass but velocity-controlled each step)
        pred_body = pymunk.Body(AGENT_MASS, pymunk.moment_for_circle(AGENT_MASS, 0, PREDATOR_RADIUS))
        pred_body.position = level.pred_start
        pred_shape = pymunk.Circle(pred_body, PREDATOR_RADIUS)
        pred_shape.elasticity = 0.1
        pred_shape.friction = 0.6
        self.space.add(pred_body, pred_shape)
        self.predator = pred_body

        # Player body
        p_body = pymunk.Body(AGENT_MASS, pymunk.moment_for_circle(AGENT_MASS, 0, PLAYER_RADIUS))
        p_body.position = level.player_start
        p_shape = pymunk.Circle(p_body, PLAYER_RADIUS)
        p_shape.elasticity = 0.1
        p_shape.friction = 0.6
        self.space.add(p_body, p_shape)
        self.player = p_body

        # Transient state
        self._voice_amp = 0.0         # last voice amplitude from player (0..1)
        self._pings: List[_Ping] = []  # visual ripples for active predator pings
        self._last_ping_tick: int = -100   # cooldown gate to keep ripples readable
        self._elapsed = 0.0
        self.tick = 0
        self.done = False
        self.win: Optional[str] = None  # 'player' (reached exit / survived), 'predator' (caught)
        # Silent-streak: how many ticks the player has been still. After 1s
        # of zero speed and no voice, we emit a faint "breathing" sound at
        # the player position so the predator can still hunt. Prevents
        # "stand still next to predator and farm proximity" exploit.
        self._silent_ticks: int = 0

        # Scoring (used in survival mode + future modes). Items spawn lazily
        # — only when score=='collect' mode is enabled via spawn_items().
        self.score: float = 0.0
        self._items: List[List[float]] = []   # each: [x, y, collected_flag, value, kind]
        self._spawn_rng = None                # set by spawn_items() for periodic respawn
        self._proximity_active: bool = False  # set to True to add proximity bonus per tick

        # Pygame surface for rendering
        pygame.init()
        self._surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # Full rebuild (pymunk doesn't expose a clean "reset")
        self.__init__(level=self.level, seed=seed if seed is not None else 0)
        return self.get_state()

    def step(self, action: np.ndarray, who: str) -> np.ndarray:
        """Apply a control action to the named agent for one tick (~0.1s).
        The passive agent keeps its current velocity (inertial glide).

        `who` ∈ {'predator', 'player'}. Action is 3-D: (dx, dy, amp) where
        - dx, dy ∈ [-1, 1] — desired velocity direction (normalized if > 1)
        - amp ∈ [0, 1] — ping amplitude (predator) OR voice amplitude (player)
        """
        if self.done:
            return self.get_state()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == ACTION_DIM
        dx, dy, amp = float(action[0]), float(action[1]), float(action[2])
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        amp = max(0.0, min(1.0, amp))

        if who == 'predator':
            v_mag = MAX_SPEED_PRED
            self.predator.velocity = (dx * v_mag, dy * v_mag)
            # Visible ripple cadence is throttled separately from the audio
            # input the JEPA receives: CEM samples can produce small nonzero
            # pings every tick which look like spam on screen. Threshold +
            # cooldown ensures at most ~2 ripples/sec.
            if amp > 0.15 and self.tick - self._last_ping_tick >= 5:
                self._pings.append(_Ping(
                    cx=self.predator.position.x,
                    cy=self.predator.position.y,
                    amplitude=amp,
                ))
                self._last_ping_tick = self.tick
        elif who == 'player':
            # Proximity slowdown: when the predator is close, the player
            # moves slower. Mirrors real predator-prey freeze response
            # AND raises the stakes of the proximity-bonus mechanic
            # (more points = closer = harder to escape).
            v_mag = MAX_SPEED_PLAYER
            d_pred = _math_hypot_xy(
                self.predator.position.x - self.player.position.x,
                self.predator.position.y - self.player.position.y,
            )
            slow_radius = 200.0     # px — start slowing inside this range
            slow_floor  = 0.55      # min speed multiplier (at d=0)
            if d_pred < slow_radius:
                t = max(0.0, d_pred) / slow_radius   # 0..1
                speed_mult = slow_floor + (1.0 - slow_floor) * t
                v_mag *= speed_mult
            self.player.velocity = (dx * v_mag, dy * v_mag)
            self._voice_amp = amp
            # Silent-streak: only updates on the player's tick (not the
            # predator's, which doesn't change player motion).
            req = max(abs(dx), abs(dy))
            if req > 0.1 or amp > 0.05:
                self._silent_ticks = 0
            else:
                self._silent_ticks += 1
        else:
            raise ValueError(f"who must be 'predator' or 'player', got {who!r}")

        # Physics sub-steps (10 sub-steps per control tick for stability)
        for _ in range(10):
            self.space.step(DT)
            # Advance ping ripples
            for p in self._pings:
                p.advance(DT)
            self._pings = [p for p in self._pings if p.alive]

        self._elapsed += 0.1
        self.tick += 1

        # Scoring: tick after both player + predator have moved on this
        # frame, otherwise we double-count (player.act → step('player') →
        # predator.act → step('predator')). We update only on predator
        # step to ensure full-tick state.
        if who == 'predator':
            self._score_tick()

        # Terminal checks
        self._check_done()
        return self.get_state()

    # ------------------------------------------------------------------
    # Scoring (mode-agnostic; activated by server via spawn_items / set_proximity)
    # ------------------------------------------------------------------
    PROXIMITY_RADIUS_PX = 220.0       # bonus zone radius (must stay > catch dist)
    PROXIMITY_PER_TICK = 1.0          # max points/tick when right next to predator
    ITEM_RADIUS_PX = 32.0              # collection radius

    # Item tiers — kind index → (value, weight_at_spawn, draw_color, rim_color)
    # Items stored as [x, y, collected_flag, value, kind].
    ITEM_TIERS = (
        # amber — common, small reward
        {'value':  25.0, 'weight': 0.65, 'color': (217, 154, 95),  'rim': (255, 220, 160)},
        # teal — uncommon, medium reward
        {'value':  50.0, 'weight': 0.25, 'color': (63, 184, 161),  'rim': (160, 230, 215)},
        # coral — rare, big reward
        {'value': 100.0, 'weight': 0.10, 'color': (232, 103, 93),  'rim': (255, 180, 170)},
    )
    ITEM_MAX_ON_FIELD = 12
    ITEM_RESPAWN_INTERVAL_TICKS = 60   # spawn one item every ~6 sec

    def _pick_item_kind(self, rng: np.random.Generator) -> int:
        weights = np.array([t['weight'] for t in self.ITEM_TIERS], dtype=np.float64)
        weights /= weights.sum()
        return int(rng.choice(len(self.ITEM_TIERS), p=weights))

    def _try_place_item(self, rng: np.random.Generator, kind: int,
                        margin: float = 60.0,
                        min_from_player: float = 90.0,
                        min_from_predator: float = 90.0,
                        min_from_other_items: float = 80.0,
                        max_attempts: int = 50) -> bool:
        for _ in range(max_attempts):
            x = float(rng.uniform(margin, WINDOW_SIZE - margin))
            y = float(rng.uniform(margin, WINDOW_SIZE - margin))
            if math.hypot(x - self.player.position.x, y - self.player.position.y) < min_from_player:
                continue
            if math.hypot(x - self.predator.position.x, y - self.predator.position.y) < min_from_predator:
                continue
            if any((it[2] < 0.5) and math.hypot(x - it[0], y - it[1]) < min_from_other_items for it in self._items):
                continue
            value = self.ITEM_TIERS[kind]['value']
            self._items.append([x, y, 0.0, value, float(kind)])
            return True
        return False

    def spawn_items(self, n: int, rng: np.random.Generator,
                    margin: float = 60.0,
                    min_from_player: float = 90.0,
                    min_from_predator: float = 90.0) -> None:
        """Initial spawn of n items with a tier mix (amber-heavy)."""
        self._items.clear()
        # Hold a persistent rng for periodic respawning so the seed behavior
        # is reproducible per match.
        self._spawn_rng = rng
        # Initial fixed mix: ensure at least one of each tier so the player
        # sees the color vocabulary right away.
        kinds = [2, 1, 1] + [0] * max(0, n - 3)   # 1 coral, 2 teal, rest amber
        kinds = kinds[:n]
        rng.shuffle(kinds)
        for k in kinds:
            self._try_place_item(rng, k, margin=margin,
                                 min_from_player=min_from_player,
                                 min_from_predator=min_from_predator)

    def maybe_respawn_item(self) -> None:
        """Tick-driven respawn: every ITEM_RESPAWN_INTERVAL_TICKS, if the
        field has fewer than ITEM_MAX_ON_FIELD active items, spawn one with
        weighted-random tier."""
        if not self._items:
            return   # spawn_items() not called → not in scoring mode
        if not hasattr(self, '_spawn_rng') or self._spawn_rng is None:
            return
        if self.tick % self.ITEM_RESPAWN_INTERVAL_TICKS != 0:
            return
        active = sum(1 for it in self._items if it[2] < 0.5)
        if active >= self.ITEM_MAX_ON_FIELD:
            return
        kind = self._pick_item_kind(self._spawn_rng)
        self._try_place_item(self._spawn_rng, kind)

    def set_proximity(self, on: bool) -> None:
        """Enable per-tick proximity bonus (encourages player to skirt
        the predator at minimum safe distance)."""
        self._proximity_active = bool(on)

    def _score_tick(self) -> None:
        # 0) Periodic respawn so the field stays interesting through the 90s.
        self.maybe_respawn_item()

        # 1) Item collection (per-item value)
        px, py = float(self.player.position.x), float(self.player.position.y)
        for it in self._items:
            if it[2] > 0.5:
                continue   # already collected
            if math.hypot(it[0] - px, it[1] - py) <= self.ITEM_RADIUS_PX:
                it[2] = 1.0
                self.score += float(it[3])   # use this item's value

        # 2) Proximity bonus: max points when right next to predator,
        #    zero outside PROXIMITY_RADIUS_PX. Linear falloff between.
        if self._proximity_active:
            d = math.hypot(
                self.predator.position.x - px,
                self.predator.position.y - py,
            )
            if d < self.PROXIMITY_RADIUS_PX:
                bonus = self.PROXIMITY_PER_TICK * (1.0 - d / self.PROXIMITY_RADIUS_PX)
                self.score += bonus

    def _check_done(self) -> None:
        # Caught?
        d = math.hypot(
            self.predator.position.x - self.player.position.x,
            self.predator.position.y - self.player.position.y,
        )
        if d < CATCH_DISTANCE:
            self.done = True
            self.win = 'predator'
            return
        # Reached exit?
        x0, y0, x1, y1 = self.level.exit_zone
        px, py = self.player.position.x, self.player.position.y
        if x0 <= px <= x1 and y0 <= py <= y1:
            self.done = True
            self.win = 'player'
            return
        # Timed out? (player wins — survived)
        if self._elapsed >= self.level.time_limit_sec:
            self.done = True
            self.win = 'player'

    def get_state(self) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0] = (self.predator.position.x - WINDOW_SIZE / 2) / (WINDOW_SIZE / 2)
        s[1] = (self.predator.position.y - WINDOW_SIZE / 2) / (WINDOW_SIZE / 2)
        s[2] = self.predator.velocity.x / MAX_SPEED_PRED
        s[3] = self.predator.velocity.y / MAX_SPEED_PRED
        s[4] = (self.player.position.x - WINDOW_SIZE / 2) / (WINDOW_SIZE / 2)
        s[5] = (self.player.position.y - WINDOW_SIZE / 2) / (WINDOW_SIZE / 2)
        s[6] = self.player.velocity.x / MAX_SPEED_PLAYER
        s[7] = self.player.velocity.y / MAX_SPEED_PLAYER
        s[8] = float(self._voice_amp)
        ex = (self.level.exit_zone[0] + self.level.exit_zone[2]) * 0.5
        ey = (self.level.exit_zone[1] + self.level.exit_zone[3]) * 0.5
        dist = math.hypot(self.player.position.x - ex, self.player.position.y - ey)
        s[9] = min(1.0, dist / WINDOW_SIZE)
        return s

    def get_audio_obs(self) -> np.ndarray:
        """Simulated 500-ms audio observation at the predator's 4 directional
        ears (N/E/S/W cardioid lobes). Returns log-mel tensor (4, N_MELS, 50).

        Physics model:
          - Direct-path propagation only (no wall reflections — AnechoicRoom-equivalent).
            Reflections are a nice-to-have and will be added via pyroomacoustics
            for offline training-data generation in Phase 1.
          - Distance attenuation: 1/d (sound-pressure-like; 1/d^2 is too aggressive
            at game scale where rooms are ~5 m).
          - Directional lobes: each ear is a cardioid pointing in one cardinal
            direction. Gain at ear c for a source at angle theta = ((1 + cos(theta - lobe_c)) / 2)^2
            Peak 1.0 at aligned, 0.0 at opposite.

        This is a faithful physical model (attenuation + directional sensing) —
        just without reflections. The JEPA will learn on the same model we
        play with, avoiding train/eval distribution mismatch.
        """
        from librosa.feature import melspectrogram

        n_samp = int(SAMPLE_RATE * OBS_WINDOW_SEC)
        signals = np.zeros((4, n_samp), dtype=np.float32)

        # Lobe axes (N, E, S, W) in screen coords (y grows down)
        # N = up = -y, E = right = +x, S = down = +y, W = left = -x
        lobe_axes = np.array([
            [0.0, -1.0],   # N
            [1.0,  0.0],   # E
            [0.0,  1.0],   # S
            [-1.0, 0.0],   # W
        ], dtype=np.float32)

        px = float(self.predator.position.x)
        py = float(self.predator.position.y)

        def add_source(src_x: float, src_y: float, sig: np.ndarray) -> None:
            """Attenuate by 1/d and spread through 4 cardioid lobes."""
            dx = src_x - px
            dy = src_y - py
            d_pix = math.hypot(dx, dy) + 1e-6
            d_m = d_pix / UNITS_PER_METER
            # Distance attenuation (clipped at close range so we don't blow up)
            atten = 1.0 / max(0.3, d_m)
            # Unit direction from predator to source
            src_dir = np.array([dx / d_pix, dy / d_pix], dtype=np.float32)
            for ch in range(4):
                dot = float(np.dot(src_dir, lobe_axes[ch]))
                # Cardioid squared — strong directional selectivity
                gain = max(0.0, (1.0 + dot) * 0.5) ** 2
                if gain > 0.0:
                    signals[ch] += (atten * gain) * sig

        # Source 1 — beacon hum at exit center.
        # Phase 3E: env-level toggle to suppress the beacon entirely. The
        # canonical model treats the beacon as a free spatial anchor, which
        # makes randomized-exit deployment break. Training without the
        # beacon forces the encoder to localize from player audio + own
        # pings alone — a more honest representation.
        if not getattr(self, '_no_beacon', False):
            ex = 0.5 * (self.level.exit_zone[0] + self.level.exit_zone[2])
            ey = 0.5 * (self.level.exit_zone[1] + self.level.exit_zone[3])
            add_source(ex, ey, _synth_beacon(n_samp))

        # Source 2 — player footsteps (speed-scaled)
        speed = math.hypot(self.player.velocity.x, self.player.velocity.y)
        foot_amp = min(1.0, speed / MAX_SPEED_PLAYER)
        if foot_amp > 0.05:
            add_source(
                float(self.player.position.x), float(self.player.position.y),
                _synth_footsteps(foot_amp, n_samp),
            )
        elif self._silent_ticks >= 10:
            # Idle "breathing" — after 1 second of stillness the player
            # leaks the same signature as a faint footstep so the predator
            # can still find them. Closes the "stand still and farm
            # proximity" exploit. Amplitude grows slowly with how long
            # they've been still (caps at 0.4).
            idle_amp = min(0.4, 0.15 + 0.005 * (self._silent_ticks - 10))
            add_source(
                float(self.player.position.x), float(self.player.position.y),
                _synth_footsteps(idle_amp, n_samp),
            )

        # Source 3 — player voice decoy
        if self._voice_amp > 0.05:
            add_source(
                float(self.player.position.x), float(self.player.position.y),
                _synth_voice(self._voice_amp, n_samp),
            )

        # Source 4 — recent predator pings
        for ping in self._pings:
            if ping.age > 5:
                continue
            add_source(ping.cx, ping.cy, _synth_ping(ping.amplitude, n_samp))

        # Per-channel log-mel
        mels = []
        for ch in range(4):
            m = melspectrogram(
                y=signals[ch], sr=SAMPLE_RATE,
                n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT, power=2.0,
            )
            mels.append(np.log1p(m))
        out = np.stack(mels, axis=0).astype(np.float32)   # (4, N_MELS, T)

        # Pad/truncate time axis to exactly 50 frames
        target_T = 50
        if out.shape[2] > target_T:
            out = out[:, :, :target_T]
        elif out.shape[2] < target_T:
            out = np.pad(out, ((0, 0), (0, 0), (0, target_T - out.shape[2])))
        return out

    # ------------------------------------------------------------------
    # Rendering — top-down view for the human player. Dark chamber + ping
    # ripples for predator pings + voice halo for player vocalizations.
    # ------------------------------------------------------------------
    def render(self, size: int = RENDER_SIZE) -> np.ndarray:
        surf = self._surface
        surf.fill(BG_COLOR)

        # Grid
        for g in range(64, WINDOW_SIZE, 64):
            pygame.draw.line(surf, GRID_COLOR, (g, 0), (g, WINDOW_SIZE), 1)
            pygame.draw.line(surf, GRID_COLOR, (0, g), (WINDOW_SIZE, g), 1)

        # Outer walls
        pygame.draw.rect(surf, WALL_COLOR, (0, 0, WINDOW_SIZE, WINDOW_SIZE), width=3)

        # Exit zone — dashed amber rect. Skip when the zone has zero area
        # (survive mode collapses it to a point so the beacon hum stays in
        # the same audio location as training, but visually no exit).
        x0, y0, x1, y1 = self.level.exit_zone
        if (x1 - x0) > 1 and (y1 - y0) > 1:
            rect = pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            fill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            fill.fill((*EXIT_COLOR, 42))
            surf.blit(fill, rect.topleft)
            _draw_dashed_rect(surf, rect, EXIT_COLOR, dash=10, gap=8, width=2)

        # Pillars
        for shape in self.pillars:
            cx = int(shape.body.position.x)
            cy = int(shape.body.position.y)
            r = int(shape.radius)
            pygame.draw.circle(surf, PILLAR_COLOR, (cx, cy), r)
            pygame.draw.circle(surf, PILLAR_RIM, (cx, cy), r, width=2)

        # Predator pings (ripples)
        for p in self._pings:
            _draw_ping_ring(surf, p.cx, p.cy, p.radius, p.amplitude)

        # Items — color + size by tier. amber=small/common, teal=mid, coral=big/rare.
        for it in self._items:
            ix, iy = int(it[0]), int(it[1])
            collected = it[2]
            kind = int(it[4]) if len(it) > 4 else 0
            tier = self.ITEM_TIERS[kind] if 0 <= kind < len(self.ITEM_TIERS) else self.ITEM_TIERS[0]
            if collected > 0.5:
                pygame.draw.circle(surf, GRID_COLOR, (ix, iy), 6, width=1)
            else:
                # Higher-tier items are slightly larger so the player can read value at a glance.
                core_r = 6 + kind * 2          # 6 / 8 / 10
                rim_r  = core_r + 2            # 8 / 10 / 12
                pygame.draw.circle(surf, tier['color'], (ix, iy), core_r)
                pygame.draw.circle(surf, tier['rim'],   (ix, iy), rim_r, width=1)

        # Voice halo on player (if vocalizing)
        if self._voice_amp > 0.05:
            halo_r = int(30 + 70 * self._voice_amp)
            halo_surf = pygame.Surface((halo_r * 2 + 4, halo_r * 2 + 4), pygame.SRCALPHA)
            c = halo_r + 2
            pygame.draw.circle(halo_surf, (*VOICE_COLOR, int(35 + 80 * self._voice_amp)), (c, c), halo_r, width=2)
            pygame.draw.circle(halo_surf, (*VOICE_COLOR, int(80 * self._voice_amp)), (c, c), halo_r - 10, width=2)
            surf.blit(halo_surf,
                      (int(self.player.position.x) - c, int(self.player.position.y) - c),
                      special_flags=pygame.BLEND_RGBA_ADD)

        # Player (teal) + halo
        _draw_agent(surf, int(self.player.position.x), int(self.player.position.y),
                    PLAYER_RADIUS, PLAYER_COLOR, PLAYER_HALO)

        # Predator (coral) + halo
        _draw_agent(surf, int(self.predator.position.x), int(self.predator.position.y),
                    PREDATOR_RADIUS, PREDATOR_COLOR, PREDATOR_HALO)

        arr = pygame.surfarray.array3d(surf)
        arr = np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))
        if size != WINDOW_SIZE:
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        return arr.astype(np.uint8, copy=False)


# ----------------------------------------------------------------------
# Audio signal synthesizers — each returns a float32 array of `n_samp`
# samples at SAMPLE_RATE. Used as sound sources in the pyroomacoustics sim.
# Keep dirt-simple: this is the training distribution the JEPA will see, so
# clarity and determinism matter more than realism.
# ----------------------------------------------------------------------
def _synth_beacon(n_samp: int) -> np.ndarray:
    """Very faint exit hum — provides audible signature of the goal but
    shouldn't drown out the main gameplay signal (player footsteps + voice).
    Tuned so that within ~1m of the exit the beacon is clearly the loudest
    source, but further away it fades below the player's footstep level."""
    t = np.arange(n_samp, dtype=np.float32) / SAMPLE_RATE
    return (0.02 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)


def _synth_footsteps(amplitude: float, n_samp: int) -> np.ndarray:
    """Repeating thuds spaced ~150-250 ms — louder/faster when amp is high."""
    t = np.arange(n_samp, dtype=np.float32) / SAMPLE_RATE
    step_hz = 4.0 + 6.0 * amplitude
    phase = (t * step_hz) % 1.0
    envelope = np.exp(-phase * 12) * (phase < 0.4)
    click = np.sin(2 * np.pi * 110.0 * t) * envelope
    click += 0.3 * np.random.randn(n_samp).astype(np.float32) * envelope
    # Boosted scale so running players are clearly audible above the beacon
    return (amplitude * 1.1 * click).astype(np.float32)


def _synth_voice(amplitude: float, n_samp: int) -> np.ndarray:
    """Sustained vocal tone with vibrato — the player's decoy mechanic."""
    t = np.arange(n_samp, dtype=np.float32) / SAMPLE_RATE
    vibrato = 20.0 * np.sin(2 * np.pi * 4.5 * t)
    sig = np.sin(2 * np.pi * (300.0 + vibrato) * t)
    return (amplitude * 0.55 * sig).astype(np.float32)


def _synth_ping(amplitude: float, n_samp: int) -> np.ndarray:
    """50 ms sine burst at 1 kHz at the START of the observation window,
    then silence. In real sim, the reverb tail will fill in the rest."""
    t = np.arange(n_samp, dtype=np.float32) / SAMPLE_RATE
    burst_sec = 0.05
    burst_samp = int(SAMPLE_RATE * burst_sec)
    sig = np.zeros(n_samp, dtype=np.float32)
    t_burst = np.arange(burst_samp, dtype=np.float32) / SAMPLE_RATE
    env = np.exp(-t_burst * 30)
    sig[:burst_samp] = amplitude * env * np.sin(2 * np.pi * 1000.0 * t_burst)
    return sig


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def _draw_dashed_rect(surf, rect, color, dash=10, gap=8, width=2):
    x0, y0, x1, y1 = rect.left, rect.top, rect.right, rect.bottom

    def dashed_line(p0, p1):
        px0, py0 = p0
        px1, py1 = p1
        dx, dy = px1 - px0, py1 - py0
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / length, dy / length
        pos = 0.0
        on = True
        while pos < length:
            step = dash if on else gap
            sx, sy = px0 + ux * pos, py0 + uy * pos
            ex, ey = px0 + ux * min(pos + step, length), py0 + uy * min(pos + step, length)
            if on:
                pygame.draw.line(surf, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
            pos += step
            on = not on

    dashed_line((x0, y0), (x1, y0))
    dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1))
    dashed_line((x0, y1), (x0, y0))


def _draw_ping_ring(surf, cx, cy, radius, amplitude):
    if radius < 3:
        return
    cx, cy, r = int(cx), int(cy), int(radius)
    halo_r = r + 16
    diam = halo_r * 2 + 4
    halo = pygame.Surface((diam, diam), pygame.SRCALPHA)
    hcx = diam // 2
    base_a = max(30, min(200, int(amplitude * 220)))
    pygame.draw.circle(halo, (*PING_COLOR, base_a // 5), (hcx, hcx), r + 12, width=6)
    pygame.draw.circle(halo, (*PING_COLOR, base_a // 3), (hcx, hcx), r + 6, width=4)
    pygame.draw.circle(halo, (*PING_COLOR, base_a), (hcx, hcx), r, width=2)
    surf.blit(halo, (cx - hcx, cy - hcx), special_flags=pygame.BLEND_RGBA_ADD)


def _draw_agent(surf, cx, cy, r, core_rgb, halo_rgb):
    halo = pygame.Surface((r * 4 + 8, r * 4 + 8), pygame.SRCALPHA)
    hcx = (r * 4 + 8) // 2
    pygame.draw.circle(halo, (*core_rgb, 38), (hcx, hcx), r * 2)
    pygame.draw.circle(halo, (*core_rgb, 70), (hcx, hcx), r + 4)
    surf.blit(halo, (cx - hcx, cy - hcx), special_flags=pygame.BLEND_RGBA_ADD)
    pygame.draw.circle(surf, core_rgb, (cx, cy), r)
    pygame.draw.circle(surf, (min(255, core_rgb[0] + 30),
                              min(255, core_rgb[1] + 30),
                              min(255, core_rgb[2] + 30)), (cx - r // 3, cy - r // 3), max(1, r // 4))


if __name__ == '__main__':
    from envs.silent_rooms import get_level
    env = Silent(level=get_level('level_01'), seed=42)
    frame = env.render()
    cv2.imwrite('/tmp/silent_smoke.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"frame shape: {frame.shape}, state dim: {env.get_state().shape}")
    print(f"saved /tmp/silent_smoke.png")
