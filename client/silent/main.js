// Silent — client
// WebSocket loop + WASD player input + score panel + footstep ripples.
// The server still owns the env physics + JEPA predator + audio sim and
// sends back rendered PNG frames + state. Nothing in this file affects
// the model — it's a pure presentation layer.

const BUILD_TAG = 'silent-0.4a';
console.log('[silent] build =', BUILD_TAG);

// Production: behind Caddy at https://jepa.waweapps.win/silent/, the WS
// endpoint lives at /silent/ws (handle_path strips /silent/ and proxies
// to silent:8801/ws). Local dev: same-host ws.
const WS_URL = (() => {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const path  = location.pathname.startsWith('/silent') ? '/silent/ws' : '/ws';
  return proto + '//' + location.host + path;
})();

const LEVELS = [
  { id: 'level_01', roman: 'I',   name: 'open field'      },
  { id: 'level_02', roman: 'II',  name: 'single pillar'   },
  { id: 'level_03', roman: 'III', name: 'three in a row'  },
  { id: 'level_04', roman: 'IV',  name: 'maze'            },
  { id: 'level_05', roman: 'V',   name: 'two rooms'       },
];

const $ = (id) => document.getElementById(id);

// ---- Audio-visualization FX -------------------------------------------
// Two kinds of pulses around the player's avatar that mirror what the
// predator's ears actually hear:
//   - Step ripples: sharp teal expanding rings, spawn while moving
//   - Idle pulses: slow, faint amber rings, spawn while standing still
//     (matches the env's anti-camp 'breathing' audio leak)
// Pure visual — the predator never sees rendered frames.
const ripples = [];        // step ripples
const idlePulses = [];     // idle / breathing pulses
let lastRippleTick = -1;
let lastIdlePulseTick = -1;
let lastPlayerPos = null;
let stillTicks = 0;

function spawnRipple(x, y) {
  ripples.push({ x, y, age: 0, maxAge: 18, radius: 6, maxRadius: 26 });
}
function spawnIdlePulse(x, y) {
  // Slower, larger, more transparent than step ripples
  idlePulses.push({ x, y, age: 0, maxAge: 42, radius: 4, maxRadius: 64 });
}

function spawnPointPop(text, x, y) {
  const div = document.createElement('div');
  div.className = 'point-pop';
  div.textContent = text;
  div.style.left = x + 'px';
  div.style.top  = y + 'px';
  el.canvasWrap.appendChild(div);
  setTimeout(() => div.remove(), 1100);
}

function drawFx() {
  const canvas = el.fx;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Step ripples (teal, sharp)
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i];
    r.age += 1;
    if (r.age >= r.maxAge) { ripples.splice(i, 1); continue; }
    const t = r.age / r.maxAge;
    const radius = r.radius + (r.maxRadius - r.radius) * t;
    const alpha = (1 - t) * 0.35;
    ctx.strokeStyle = `rgba(63, 184, 161, ${alpha})`;
    ctx.lineWidth = 1.5 * (1 - t * 0.5);
    ctx.beginPath();
    ctx.arc(r.x, r.y, radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Idle pulses (amber, slow, faint — visualizes the audio leak when
  // the player is camping)
  for (let i = idlePulses.length - 1; i >= 0; i--) {
    const p = idlePulses[i];
    p.age += 1;
    if (p.age >= p.maxAge) { idlePulses.splice(i, 1); continue; }
    const t = p.age / p.maxAge;
    const radius = p.radius + (p.maxRadius - p.radius) * t;
    const alpha = (1 - t) * 0.45;   // visible but distinct from sharp step rings
    ctx.strokeStyle = `rgba(217, 154, 95, ${alpha})`;   // amber
    ctx.lineWidth = 1.4 * (1 - t * 0.4);
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}
// Backward-compat name (some calls still reference drawRipples)
const drawRipples = drawFx;

const el = {
  // arena + canvas
  arena:        $('arena'),
  canvasWrap:   $('canvas-wrap'),
  frame:        $('game-frame'),
  fx:           $('fx-overlay'),
  banner:       $('banner'),
  bannerText:   $('banner-text'),

  // level nav + header
  levelNav:     $('level-nav'),
  helpBtn:      $('help-btn'),
  muteBtn:      $('mute-btn'),

  // score band (survive mode marquee)
  scorePanel:   $('score-panel'),
  scoreBig:     $('score-big'),
  scoreItems:   $('score-items'),
  scoreTime:    $('score-time'),

  // result overlay
  result:       $('result-overlay'),
  resultTitle:  $('result-title'),
  resultSub:    $('result-sub'),
  resultScore:  $('result-score'),
  resultScoreNum:   $('result-score-num'),
  resultScoreLabel: $('result-score-label'),
  resultStats:  $('result-stats'),
  btnReplay:    $('btn-replay'),
  btnNext:      $('btn-next'),

  // hud (right rail)
  hudTime:      $('hud-time'),
  hudDist:      $('hud-dist'),
  hudTick:      $('hud-tick'),
  hudScore:     $('hud-score'),
  hudScoreRow:  $('hud-score-row'),
  hudItems:     $('hud-items'),
  hudItemsRow:  $('hud-items-row'),

  // howto modal
  howto:         $('howto'),
  howtoBackdrop: $('howto-backdrop'),
  howtoDismiss:  $('howto-dismiss'),

  // toggle groups
  modeBtns:  document.querySelectorAll('.mode-btn'),
  mapBtns:   document.querySelectorAll('.map-btn'),
};

// Audio: on/off flag. AudioContext is created lazily on first user gesture.
let audioEnabled = true;
let audioUnlocked = false;
function unlockAudio() {
  if (audioUnlocked || !window.silentAudio) return;
  window.silentAudio.ensureContext();
  audioUnlocked = true;
}
function setAudioEnabled(on) {
  audioEnabled = on;
  if (window.silentAudio) window.silentAudio.setMuted(!on);
  if (el.muteBtn) el.muteBtn.textContent = on ? '♪ audio on' : '♪ audio off';
}

const state = {
  ws: null,
  connected: false,
  levelIdx: 0,
  predatorMode: 'jepa_v2',
  randomGoal: false,
  keys: { w: false, a: false, s: false, d: false },
  done: false,
  lastPayload: null,
  // score-tracking for animations
  lastScore: 0,
  lastItemsCollected: 0,
  // hold the very first match until the user dismisses the welcome modal.
  // Subsequent reopens of the howto don't re-pause; this flag flips once.
  matchStarted: false,
};

// ---- websocket + message loop -----------------------------------------
function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    state.ws = ws;
    state.connected = true;
    startTickLoop();
    // Don't auto-launch the first match — wait until the user dismisses the
    // welcome modal. After that, reconnects (e.g. server hiccup) do auto-resume.
    if (state.matchStarted) sendNewMatch();
  };
  ws.onclose = () => {
    state.connected = false;
    setTimeout(connect, 1200);
  };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    handleServer(msg);
  };
}
function send(obj) { if (state.connected) state.ws.send(JSON.stringify(obj)); }

function resetMatchClientState() {
  state.done = false;
  state.lastScore = 0;
  state.lastItemsCollected = 0;
  ripples.length = 0;
  idlePulses.length = 0;
  lastPlayerPos = null;
  // Critical: the tick gate counters are module-level let, so without
  // resetting them, after a replay (server tick=0) we'd never spawn
  // ripples again because msg.tick - lastRippleTick < 0 < 3. This was
  // the "step VFX missing in survive mode" bug.
  lastRippleTick = -1;
  lastIdlePulseTick = -1;
  stillTicks = 0;
  hideResult();
  if (window.silentAudio && audioUnlocked) window.silentAudio.silence();
}

function sendNewMatch() {
  resetMatchClientState();
  send({
    type: 'new_match',
    level: LEVELS[state.levelIdx].id,
    predator: state.predatorMode,
    mode: 'survival',
    random_goal: state.randomGoal,
    seed: Date.now() & 0xffff,
  });
}
function sendReset() {
  // Replay = fresh new_match with same settings. Using {type:'reset'}
  // would skip the server's mode/scoring re-init, so survive-mode
  // items + proximity flag would be lost and the score panel would
  // disappear. Always go through new_match for replay.
  sendNewMatch();
}

function handleServer(msg) {
  if (msg.type === 'error') { console.warn('server error', msg.error); return; }
  if (msg.type !== 'frame') return;

  state.lastPayload = msg;
  el.frame.src = 'data:image/png;base64,' + msg.frame;

  // Step + idle pulse FX. Both visualize what the predator's ears
  // actually hear:
  //   - Step ripples (teal): spawn while moving (player emits footsteps)
  //   - Idle pulses (amber): spawn while still for >= 1s (env's
  //     anti-camp 'breathing' audio leak)
  if (msg.player_pos) {
    const px = msg.player_pos[0];
    const py = msg.player_pos[1];
    if (lastPlayerPos !== null) {
      const dx = px - lastPlayerPos[0];
      const dy = py - lastPlayerPos[1];
      const moved = Math.hypot(dx, dy);
      if (moved > 4) {
        stillTicks = 0;
        if (msg.tick - lastRippleTick >= 3) {
          spawnRipple(px, py);
          lastRippleTick = msg.tick;
        }
      } else {
        stillTicks += 1;
        // After ~1s of stillness, start emitting slow amber pulses,
        // matching the env's idle-audio leak that kicks in at
        // _silent_ticks >= 10 server-side.
        if (stillTicks >= 10 && msg.tick - lastIdlePulseTick >= 14) {
          spawnIdlePulse(px, py);
          lastIdlePulseTick = msg.tick;
        }
      }
    }
    lastPlayerPos = [px, py];
  }
  drawFx();

  // ---- Run stats (always shown) ----
  const remaining = Math.max(0, msg.time_limit_sec - msg.elapsed_sec);
  el.hudTime.textContent = remaining.toFixed(1);
  el.hudTick.textContent = msg.tick;

  if (msg.predator_pos && msg.player_pos) {
    const dx = msg.player_pos[0] - msg.predator_pos[0];
    const dy = msg.player_pos[1] - msg.predator_pos[1];
    const d = Math.hypot(dx, dy);
    el.hudDist.textContent = d.toFixed(0) + 'px';
    if (!msg.done && d < 110) {
      el.banner.classList.add('on');
      el.bannerText.textContent = 'predator nearby';
    } else {
      el.banner.classList.remove('on');
    }
  }

  // ---- Scoring (survive mode only) ----
  const scoring = msg.proximity_active || (msg.items_total > 0);
  const score = msg.score || 0;
  if (scoring) {
    el.arena.classList.add('scoring');
    el.scorePanel.classList.add('on');
    el.scoreBig.textContent = score.toFixed(0);
    el.scoreItems.textContent = `${msg.items_collected || 0}/${msg.items_total}`;
    el.scoreTime.textContent = remaining.toFixed(1) + 's';

    // Score-num bump animation when crossing an integer threshold
    if (Math.floor(score) > Math.floor(state.lastScore || 0)) {
      el.scoreBig.classList.remove('bump');
      void el.scoreBig.offsetWidth;
      el.scoreBig.classList.add('bump');
      setTimeout(() => el.scoreBig.classList.remove('bump'), 180);
    }

    // "+25" pop near player when an item is collected
    const collectedNow = msg.items_collected || 0;
    if (collectedNow > (state.lastItemsCollected || 0)) {
      const wrap = el.canvasWrap.getBoundingClientRect();
      const sx = (msg.player_pos[0] / 512) * wrap.width;
      const sy = (msg.player_pos[1] / 512) * wrap.height;
      spawnPointPop('+25', sx, sy);
    }
    state.lastItemsCollected = collectedNow;
    state.lastScore = score;

    // Mirror values into the right-rail Run card too
    el.hudScoreRow.style.display = '';
    el.hudScore.textContent = score.toFixed(0);
    el.hudItemsRow.style.display = '';
    el.hudItems.textContent = `${msg.items_collected || 0}/${msg.items_total}`;
  } else {
    el.arena.classList.remove('scoring');
    el.scorePanel.classList.remove('on');
    el.hudScoreRow.style.display = 'none';
    el.hudItemsRow.style.display = 'none';
  }

  // ---- Game over ----
  if (msg.done && !state.done) {
    state.done = true;
    showResult(msg);
  }

  // Drive spatial audio from the server payload
  if (audioEnabled && window.silentAudio && audioUnlocked) {
    window.silentAudio.update(msg);
  }
}

// ---- player input -----------------------------------------------------
function computePlayerAction() {
  let vx = 0, vy = 0;
  if (state.keys.a) vx -= 1;
  if (state.keys.d) vx += 1;
  if (state.keys.w) vy -= 1;   // canvas y grows down; up = -y
  if (state.keys.s) vy += 1;
  const m = Math.hypot(vx, vy);
  if (m > 0) { vx /= m; vy /= m; }
  return { vx, vy, voice_amp: 0.0 };
}

// 10 Hz tick — matches server CONTROL_HZ
let tickTimer = null;
function startTickLoop() {
  if (tickTimer !== null) return;
  tickTimer = setInterval(() => {
    if (!state.connected || state.done) return;
    const a = computePlayerAction();
    send({ type: 'player_action', vx: a.vx, vy: a.vy, voice_amp: a.voice_amp });
  }, 100);
}

// ---- level nav --------------------------------------------------------
function buildLevelNav() {
  el.levelNav.innerHTML = '';
  LEVELS.forEach((lvl, i) => {
    const btn = document.createElement('button');
    btn.className = 'level-btn' + (i === state.levelIdx ? ' active' : '');
    btn.innerHTML = `<span class="num">${lvl.roman}</span><span>${lvl.name}</span>`;
    btn.addEventListener('click', () => setLevel(i));
    el.levelNav.appendChild(btn);
  });
}
function setLevel(i) {
  state.levelIdx = (i + LEVELS.length) % LEVELS.length;
  buildLevelNav();
  if (state.connected) sendNewMatch();
}

// ---- toggle groups ----------------------------------------------------
function bindToggleGroup(buttons, onSelect) {
  buttons.forEach((b) => b.addEventListener('click', () => {
    buttons.forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    onSelect(b);
  }));
}
bindToggleGroup(el.modeBtns, (b) => {
  state.predatorMode = b.dataset.mode;
  if (state.connected) sendNewMatch();
});
bindToggleGroup(el.mapBtns, (b) => {
  state.randomGoal = b.dataset.rand === '1';
  if (state.connected) sendNewMatch();
});

// ---- result overlay ---------------------------------------------------
function showResult(msg) {
  const win = msg && msg.win;
  const cls = win === 'player' ? 'win' : 'lose';
  el.result.classList.remove('win', 'lose');
  el.result.classList.add(cls, 'on');
  el.resultTitle.textContent = win === 'player' ? 'escaped' : 'caught';

  // Subtitle — survive mode prefers a different copy than escape
  const survive = msg && (msg.proximity_active || msg.items_total > 0);
  if (survive) {
    el.resultSub.textContent = win === 'player'
      ? 'you survived 90 seconds'
      : 'the hunter found you';
  } else {
    el.resultSub.textContent = win === 'player'
      ? 'you reached the exit'
      : 'the hunter found you';
  }

  // Final-score recap (only in survive mode)
  if (survive) {
    const finalScore = (msg.score || 0).toFixed(0);
    const items = `${msg.items_collected || 0}/${msg.items_total || 0}`;
    const timeSurvived = (msg.elapsed_sec || 0).toFixed(1);
    el.resultScoreNum.textContent = finalScore;
    el.resultScoreLabel.textContent = win === 'player' ? 'final score' : 'final score';
    el.resultScore.classList.add('on');
    el.resultStats.innerHTML =
      `<span>${items} items</span>` +
      `<span class="sep">·</span>` +
      `<span>survived ${timeSurvived}s</span>`;
    el.resultStats.classList.add('on');
  } else {
    el.resultScore.classList.remove('on');
    el.resultStats.classList.remove('on');
  }
}
function hideResult() {
  el.result.classList.remove('on', 'win', 'lose');
  el.resultScore.classList.remove('on');
  el.resultStats.classList.remove('on');
}
el.btnReplay.addEventListener('click', sendReset);
el.btnNext.addEventListener('click', () => setLevel(state.levelIdx + 1));

// ---- howto card -------------------------------------------------------
function showHowto() {
  el.howto.classList.remove('hidden');
  el.howtoBackdrop.classList.remove('hidden');
}
function hideHowto() {
  el.howto.classList.add('hidden');
  el.howtoBackdrop.classList.add('hidden');
  // First dismissal triggers the very first match. Subsequent reopens
  // are read-only references and don't restart anything.
  if (!state.matchStarted) {
    state.matchStarted = true;
    if (state.connected) sendNewMatch();
  }
}
function toggleHowto() {
  if (el.howto.classList.contains('hidden')) showHowto();
  else hideHowto();
}
el.helpBtn.addEventListener('click', showHowto);
el.howtoDismiss.addEventListener('click', hideHowto);
el.howtoBackdrop.addEventListener('click', hideHowto);

// ---- keyboard ---------------------------------------------------------
window.addEventListener('keydown', (e) => {
  unlockAudio();
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  const k = e.code;
  if (k === 'KeyW') state.keys.w = true;
  else if (k === 'KeyA') state.keys.a = true;
  else if (k === 'KeyS') state.keys.s = true;
  else if (k === 'KeyD') state.keys.d = true;
  else if (k === 'Space') { e.preventDefault(); hideHowto(); }
  else if (k === 'KeyR') { sendReset(); hideHowto(); }
  else if (k === 'KeyL') { setLevel(state.levelIdx + 1); }
  else if (k === 'KeyM') { setAudioEnabled(!audioEnabled); }
  else if (k === 'Escape') {
    if (!el.howto.classList.contains('hidden')) hideHowto();
    else toggleHowto();
  }
  else if (e.key === '?' || e.key === '/') { e.preventDefault(); toggleHowto(); }
});
window.addEventListener('keyup', (e) => {
  const k = e.code;
  if (k === 'KeyW') state.keys.w = false;
  else if (k === 'KeyA') state.keys.a = false;
  else if (k === 'KeyS') state.keys.s = false;
  else if (k === 'KeyD') state.keys.d = false;
});

if (el.muteBtn) el.muteBtn.addEventListener('click', () => { unlockAudio(); setAudioEnabled(!audioEnabled); });
el.helpBtn.addEventListener('click', unlockAudio);
document.addEventListener('click', unlockAudio, { once: false });

// ---- boot -------------------------------------------------------------
buildLevelNav();
hideResult();
connect();
