// Silent — client-side spatial audio (Phase 0.4)
// You hear the world from the player's position:
//   - beacon hum from the exit direction (low-frequency sine, stereo-panned)
//   - predator ping bursts panned to where the predator is
//   - a proximity growl that gets louder as the predator closes
//   - your own footsteps when you move (unpanned — own ears)
//
// Uses Web Audio primitives (OscillatorNode + StereoPannerNode + GainNode).
// AudioContext is created lazily on first user gesture (browser requirement).

(() => {
  const FREQ_BEACON = 82;
  const FREQ_GROWL  = 58;
  const FREQ_PING   = 1200;
  const FREQ_FOOT   = 95;

  // How fast sound falls off with distance (world units). Smaller = faster falloff.
  const PAN_HALF_WIDTH = 220;   // world-units difference that maps to pan=1
  const BEACON_RANGE   = 260;   // beyond this, beacon is inaudible
  const GROWL_RANGE    = 220;   // predator within this produces audible growl
  const PING_RANGE     = 340;   // how far a ping burst is audible

  const state = {
    ctx: null,
    master: null,
    beacon: { osc: null, pan: null, gain: null },
    growl:  { osc: null, pan: null, gain: null },
    lastFoot: 0,              // timestamp (ctx time) of last player footstep click
    playedPingIds: new Set(), // avoid retriggering the same ping on subsequent frames
    muted: false,
  };

  function ensureContext() {
    if (state.ctx) {
      if (state.ctx.state === 'suspended') state.ctx.resume();
      return;
    }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const master = ctx.createGain();
    master.gain.value = 0.45;
    master.connect(ctx.destination);

    // Beacon: sustained low hum, always running, pan + gain updated per frame
    const b = mkNode(ctx, 'sine', FREQ_BEACON);
    b.osc.start();
    b.gain.gain.value = 0;
    b.gain.connect(master);

    // Growl: low sawtooth, running but typically silent (gain=0) unless predator close
    const g = mkNode(ctx, 'sawtooth', FREQ_GROWL);
    g.osc.start();
    g.gain.gain.value = 0;
    g.gain.connect(master);

    state.ctx = ctx;
    state.master = master;
    state.beacon = b;
    state.growl = g;
  }

  // Small oscillator → panner → gain subchain
  function mkNode(ctx, type, freq) {
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = freq;
    const pan = ctx.createStereoPanner();
    const gain = ctx.createGain();
    osc.connect(pan);
    pan.connect(gain);
    return { osc, pan, gain };
  }

  function setPanGain(node, pan, gainValue, ramp_sec = 0.08) {
    if (!state.ctx) return;
    const t = state.ctx.currentTime;
    node.pan.pan.cancelScheduledValues(t);
    node.pan.pan.linearRampToValueAtTime(pan, t + ramp_sec);
    node.gain.gain.cancelScheduledValues(t);
    node.gain.gain.linearRampToValueAtTime(gainValue, t + ramp_sec);
  }

  function panFromDx(dx) {
    // Left = -1, Right = +1. World y irrelevant for L/R panning.
    return Math.max(-1, Math.min(1, dx / PAN_HALF_WIDTH));
  }

  // One-shot ping burst: short high sine, panned + attenuated
  function playPingBurst(dx, dy, amplitude) {
    if (!state.ctx || state.muted) return;
    const d = Math.hypot(dx, dy) + 1;
    if (d > PING_RANGE) return;
    const pan = panFromDx(dx);
    const vol = Math.min(0.25, amplitude * 90 / d);   // softer; previous cap (0.55) was ear-tingly

    const ctx = state.ctx;
    const t0 = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(FREQ_PING, t0);
    osc.frequency.exponentialRampToValueAtTime(FREQ_PING * 0.7, t0 + 0.25);

    const p = ctx.createStereoPanner();
    p.pan.setValueAtTime(pan, t0);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(vol, t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.28);

    osc.connect(p); p.connect(g); g.connect(state.master);
    osc.start(t0);
    osc.stop(t0 + 0.3);
  }

  // One-shot footstep click (player's own ears, unpanned)
  function playFootstep(amplitude) {
    if (!state.ctx || state.muted) return;
    const ctx = state.ctx;
    const t0 = ctx.currentTime;
    // Rate-limit — max one click every 150ms
    if (t0 - state.lastFoot < 0.15) return;
    state.lastFoot = t0;

    const osc = ctx.createOscillator();
    osc.type = 'square';
    osc.frequency.setValueAtTime(FREQ_FOOT, t0);
    const g = ctx.createGain();
    const v = Math.min(0.22, 0.08 + 0.2 * amplitude);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(v, t0 + 0.003);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.07);
    osc.connect(g); g.connect(state.master);
    osc.start(t0);
    osc.stop(t0 + 0.09);
  }

  // Call once per server frame with the latest payload. Drives all sustained
  // sounds (beacon, growl) and triggers one-shot sounds (pings, footsteps).
  function update(msg) {
    if (!state.ctx) return;
    if (state.ctx.state === 'suspended') state.ctx.resume();

    // Game over: ramp all sustained sounds to silence and skip everything else.
    // Beacon + growl oscillators keep running (cheap) — only gain is modulated.
    if (msg.done) {
      setPanGain(state.beacon, 0, 0, 0.4);
      setPanGain(state.growl,  0, 0, 0.4);
      return;
    }

    const [px, py] = msg.player_pos;
    const [ex, ey] = msg.predator_pos;
    const exit_cx = (msg.exit_zone[0] + msg.exit_zone[2]) * 0.5;
    const exit_cy = (msg.exit_zone[1] + msg.exit_zone[3]) * 0.5;

    // ---- Beacon (exit hum) --------------------------------------------------
    const b_dx = exit_cx - px;
    const b_dy = exit_cy - py;
    const b_d  = Math.hypot(b_dx, b_dy) + 1;
    const b_vol = Math.max(0, Math.min(0.28, (BEACON_RANGE - b_d) / BEACON_RANGE * 0.28));
    setPanGain(state.beacon, panFromDx(b_dx), b_vol);

    // ---- Growl (predator proximity) -----------------------------------------
    const g_dx = ex - px;
    const g_dy = ey - py;
    const g_d  = Math.hypot(g_dx, g_dy) + 1;
    const g_vol = g_d < GROWL_RANGE
      ? Math.min(0.20, (GROWL_RANGE - g_d) / GROWL_RANGE * 0.20)
      : 0;
    setPanGain(state.growl, panFromDx(g_dx), g_vol);

    // ---- Predator ping bursts (one-shot) ------------------------------------
    const currentIds = new Set();
    for (const ping of msg.pings || []) {
      const id = `${Math.round(ping.x)}-${Math.round(ping.y)}-${Math.round(ping.amplitude * 100)}`;
      currentIds.add(id);
      if (!state.playedPingIds.has(id)) {
        state.playedPingIds.add(id);
        playPingBurst(ping.x - px, ping.y - py, ping.amplitude);
      }
    }
    // GC old ping IDs
    for (const id of Array.from(state.playedPingIds)) {
      if (!currentIds.has(id)) state.playedPingIds.delete(id);
    }

    // ---- Player footsteps (driven by player velocity in state vector) -------
    // state[6], state[7] are player vx, vy normalized by MAX_SPEED_PLAYER
    if (msg.state && msg.state.length >= 8) {
      const vx_n = msg.state[6];
      const vy_n = msg.state[7];
      const speed_n = Math.hypot(vx_n, vy_n);
      if (speed_n > 0.15) {
        playFootstep(speed_n);
      }
    }
  }

  function setMuted(m) {
    state.muted = !!m;
    if (state.master && state.ctx) {
      state.master.gain.cancelScheduledValues(state.ctx.currentTime);
      state.master.gain.linearRampToValueAtTime(state.muted ? 0 : 0.45, state.ctx.currentTime + 0.1);
    }
  }

  // Explicit silence call — used on level change / reset to kill any lingering
  // beacon/growl tones between matches before the next frame update arrives.
  function silence() {
    if (!state.ctx) return;
    setPanGain(state.beacon, 0, 0, 0.15);
    setPanGain(state.growl,  0, 0, 0.15);
    state.playedPingIds.clear();
  }

  window.silentAudio = { ensureContext, update, setMuted, silence };
})();
