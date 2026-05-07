// silent-local: in-browser JEPA predator.
//
// Mirrors silent_jepa_predator_v2.py but in TF.js. Pipeline per tick:
//   1. Server sends audio_obs (4, 64, 50) mel-spec via the WS frame payload
//   2. We bilinear-upsample to (4, 224, 224)
//   3. Append to obs history; drop oldest. Once history==3, ready to plan.
//   4. Run vitTinyForward → (1, 3, 192) emb_ctx
//   5. CEM: 16 candidates × 2 iters
//      - Build (16, 3, 15) action tensors (history actions + tiled candidate)
//      - actionEncoderForward + predictorForward → pred (16, 3, 192)
//      - state-head decode pred[:, -1] → (16, 10)
//      - cost = ||decoded[:, 0:2] - decoded[:, 4:6]|| * 256 + ping_weight * cands[:, 2]
//      - elite topk → refit mean, std
//   6. Periodic ping override every 30 ticks (matches v2 predator)
//
// Inference happens on whatever TF.js backend is available — WebGL on
// most browsers, WebGPU on Chrome stable now. Forward is one
// tf.tidy() block to keep memory bounded.

import { parseRelayBlob, makeForward } from './federated_forward.js';

const HISTORY = 3;
const FRAMESKIP = 5;
const ACTION_DIM = 3;
const EMBED_DIM = 192;
// CEM 16×2 — matches server-side silent_jepa_predator_v2 fidelity.
// Speed comes from WebGPU backend (3-5× faster than WebGL on M3) +
// async planning (plan runs off the WS handler so game tick stays 10Hz).
const N_SAMPLES = 16;
const N_ITERS = 2;
const PING_WEIGHT = 30.0;
const W_DIST = 1.0;
const PING_EVERY = 30;
const PING_AMP = 0.6;
const ELITE_K = Math.max(2, Math.floor(N_SAMPLES / 4));
const STATE_SCALE_PX = 256.0;     // matches silent.py: state_xy * 256 = pixels


export class JepaPredator {
  constructor() {
    this.ready = false;
    this.weights = null;        // Map<string, tf.Tensor>
    this.forward = null;        // makeForward(tf) result
    this.obsBuf = [];           // raw (4,64,50) Float32Array entries
    this.actBuf = [];           // (FRAMESKIP, ACTION_DIM) Float32Array entries
    this.tickCount = 0;
    this.latestAction = [0.0, 0.0, 0.0];
    this.lastDecisionMs = 0;
    this._planning = false;     // async-plan in-flight guard
  }

  async init(weightsUrl) {
    if (typeof tf === 'undefined') throw new Error('TF.js not loaded yet');
    // Prefer WebGPU on M3 / modern Chrome — 3-5× faster than WebGL for
    // the predictor forward pass. Fall back to WebGL automatically.
    try {
      if (tf.engine().registryFactory && tf.engine().registryFactory['webgpu']) {
        await tf.setBackend('webgpu');
      }
    } catch (e) { /* fallback to default */ }
    await tf.ready();
    console.log('[jepa] TF.js backend:', tf.getBackend());
    console.log('[jepa] fetching', weightsUrl);
    const t0 = performance.now();
    const resp = await fetch(weightsUrl);
    if (!resp.ok) throw new Error(`weights fetch failed: ${resp.status}`);
    const buf = await resp.arrayBuffer();
    console.log(`[jepa] weights ${(buf.byteLength / 1e6).toFixed(1)} MB in ${((performance.now() - t0) | 0)} ms`);
    const arrs = parseRelayBlob(buf);
    this.weights = new Map();
    for (const [name, info] of arrs) {
      this.weights.set(name, tf.tensor(info.data, info.shape, 'float32'));
    }
    console.log('[jepa] weights loaded:', this.weights.size, 'tensors');
    this.forward = makeForward(tf);
    this.ready = true;
  }

  // Decode base64 audio_obs string → Float32Array (4*64*50).
  // Caller passes shape from the server; we sanity check it.
  // Plan runs async — WS handler returns immediately. The send loop reads
  // whatever latestAction is current. Mirrors the OG silent server's
  // background-planner pattern (CLAUDE.md: "Async background planner:
  // decouple game tick from CEM latency"). Without this, plan blocks the
  // main thread for ~160ms, capping the loop at ~3.8 Hz instead of 10 Hz.
  onObs(b64, shape) {
    if (!this.ready) return;
    if (!Array.isArray(shape) || shape[0] !== 4 || shape[1] !== 64 || shape[2] !== 50) {
      console.warn('[jepa] unexpected obs shape', shape);
      return;
    }
    const bin = atob(b64);
    const total = bin.length / 4;
    const arr = new Float32Array(total);
    const u8 = new Uint8Array(arr.buffer);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    if (arr.length !== 4 * 64 * 50) {
      console.warn('[jepa] obs length mismatch', arr.length);
      return;
    }
    this.obsBuf.push(arr);
    if (this.obsBuf.length > HISTORY + 1) this.obsBuf.shift();
    this.tickCount++;
    if (this.obsBuf.length < HISTORY) return;     // still warming up

    // Don't kick off a new plan if one is in flight — drop this obs's
    // chance to plan. The next obs that lands while idle will use the
    // most recent obsBuf, which is what we want.
    if (this._planning) return;
    this._planning = true;
    // Defer to the next macrotask so the WS onmessage handler can return
    // first; UI re-renders aren't starved by the long compute.
    setTimeout(() => this._runPlan(), 0);
  }

  _runPlan() {
    const t0 = performance.now();
    try {
      this.latestAction = this._planOneTick();
    } catch (e) {
      console.error('[jepa] plan failed:', e);
    } finally {
      this._planning = false;
    }
    this.lastDecisionMs = performance.now() - t0;

    // Periodic ping override (silent_jepa_predator_v2.py:228)
    if (this.tickCount % PING_EVERY === 0 || this.tickCount <= 2) {
      this.latestAction[2] = Math.max(this.latestAction[2], PING_AMP);
    }

    // Push the chosen action into the action buffer (tiled by frameskip).
    const fsActs = new Float32Array(FRAMESKIP * ACTION_DIM);
    for (let f = 0; f < FRAMESKIP; f++) {
      fsActs[f * ACTION_DIM]     = this.latestAction[0];
      fsActs[f * ACTION_DIM + 1] = this.latestAction[1];
      fsActs[f * ACTION_DIM + 2] = this.latestAction[2];
    }
    this.actBuf.push(fsActs);
    if (this.actBuf.length > HISTORY + 2) this.actBuf.shift();
  }

  getLatestAction() { return this.latestAction.slice(); }

  // Run one full forward + CEM. Returns [pdx, pdy, ping].
  _planOneTick() {
    return tf.tidy(() => {
      // ---- Build obs history tensor: (1, 3, 4, 224, 224) -----------------
      const obsHist = this.obsBuf.slice(-HISTORY);
      // For each obs (4,64,50), bilinear upsample to (4,224,224) via TF.js
      // resizeBilinear expects (H, W, C). We do it per-channel.
      const upsampled = [];
      for (const obs of obsHist) {
        const t = tf.tensor(obs, [4, 64, 50]);          // (4, 64, 50)
        const tHWC = tf.transpose(t, [1, 2, 0]);        // (64, 50, 4)
        const r = tf.image.resizeBilinear(tHWC, [224, 224], false);  // (224, 224, 4)
        const back = tf.transpose(r, [2, 0, 1]);        // (4, 224, 224)
        upsampled.push(back);
      }
      const obsStack = tf.stack(upsampled, 0);          // (3, 4, 224, 224)
      const pixels = tf.expandDims(obsStack, 0);        // (1, 3, 4, 224, 224)

      // ---- Encode history through ViT-Tiny → emb_ctx (1, 3, 192) -------
      const flatPixels = tf.reshape(pixels, [HISTORY, 4, 224, 224]);
      const h = this.forward.vitTinyForward(flatPixels, this.weights);
      const proj = this.forward.mlpProjectorForward(h, this.weights, 'projector');
      const embCtx1 = tf.reshape(proj, [1, HISTORY, EMBED_DIM]);
      const embCtxB = tf.tile(embCtx1, [N_SAMPLES, 1, 1]);  // (16, 3, 192)

      // ---- CEM: history-action prefix (same for every candidate) -------
      // Need the last (HISTORY - 1) action slots from actBuf, each
      // (FRAMESKIP * ACTION_DIM,) = (15,). Slot order matches silent_jepa_predator_v2.
      const prevRows = [];
      for (let h2 = 0; h2 < HISTORY - 1; h2++) {
        const offset = HISTORY - 1 - h2;
        if (offset <= this.actBuf.length) {
          prevRows.push(this.actBuf[this.actBuf.length - offset]);
        } else {
          prevRows.push(new Float32Array(FRAMESKIP * ACTION_DIM));
        }
      }
      const prevFlat = new Float32Array(prevRows.length * FRAMESKIP * ACTION_DIM);
      for (let i = 0; i < prevRows.length; i++) {
        prevFlat.set(prevRows[i], i * FRAMESKIP * ACTION_DIM);
      }
      const prevTensor = tf.tensor(prevFlat,
        [HISTORY - 1, FRAMESKIP * ACTION_DIM]);                      // (2, 15)
      const prevB = tf.tile(tf.expandDims(prevTensor, 0),
        [N_SAMPLES, 1, 1]);                                          // (16, 2, 15)

      // ---- CEM iterations -----------------------------------------------
      let mean = tf.zeros([ACTION_DIM]);
      let std = tf.tensor1d([0.7, 0.7, 0.4]);
      let bestAction = null;

      for (let it = 0; it < N_ITERS; it++) {
        const eps = tf.randomNormal([N_SAMPLES, ACTION_DIM]);
        let cands = tf.add(mean, tf.mul(std, eps));
        cands = tf.clipByValue(cands, -1, 1);
        // ping (col 2) ∈ [0, 1]
        const ping = tf.clipByValue(
          tf.slice(cands, [0, 2], [N_SAMPLES, 1]), 0, 1);
        const move = tf.slice(cands, [0, 0], [N_SAMPLES, 2]);
        cands = tf.concat([move, ping], 1);                         // (16, 3)

        // Tile candidate by frameskip → (16, 1, 15)
        const candRow = tf.reshape(
          tf.tile(tf.expandDims(cands, 1), [1, FRAMESKIP, 1]),
          [N_SAMPLES, 1, FRAMESKIP * ACTION_DIM]);

        const actTensor = tf.concat([prevB, candRow], 1);            // (16, 3, 15)
        const actEmb = this.forward.actionEncoderForward(
          actTensor, this.weights);                                  // (16, 3, 192)

        const predHidden = this.forward.predictorForward(
          embCtxB, actEmb, this.weights);                            // (16, 3, 192)
        const predOut = this.forward.mlpProjectorForward(
          predHidden, this.weights, 'pred_proj');                    // (16, 3, 192)

        // Take last token: pred[:, -1, :] → (16, 192)
        const finalPred = tf.slice(predOut, [0, HISTORY - 1, 0],
                                    [N_SAMPLES, 1, EMBED_DIM]).squeeze(1);

        // State head decode → (16, 10)
        const decoded = this._stateHeadForward(finalPred);
        const predXY   = tf.slice(decoded, [0, 0], [N_SAMPLES, 2]);
        const playerXY = tf.slice(decoded, [0, 4], [N_SAMPLES, 2]);
        const distNorm = tf.norm(tf.sub(predXY, playerXY), 'euclidean', 1);
        const distPx = tf.mul(distNorm, STATE_SCALE_PX);
        const pingFlat = tf.squeeze(ping, [1]);                      // (16,)
        const costs = tf.add(tf.mul(distPx, W_DIST),
                             tf.mul(pingFlat, PING_WEIGHT));         // (16,)

        // Elite topk (lowest costs)
        const negCosts = tf.neg(costs);
        const { values: _v, indices } = tf.topk(negCosts, ELITE_K);
        const elites = tf.gather(cands, indices);                    // (4, 3)

        const newMean = tf.mean(elites, 0);
        const newStd = tf.add(
          tf.sqrt(tf.mean(tf.square(tf.sub(elites, newMean)), 0)),
          0.05);
        mean = newMean;
        std = newStd;
        if (it === N_ITERS - 1) {
          // Take elite #0 — same as v2 predator
          bestAction = tf.slice(elites, [0, 0], [1, ACTION_DIM]).squeeze();
        }
      }
      const a = bestAction.dataSync();
      return [a[0], a[1], a[2]];
    });
  }

  // State head: Sequential(Linear(192,256), GELU, Linear(256,256), GELU,
  // Linear(256, 10)). Matches the head ckpt key layout fc0/fc1/fc2.
  // PyTorch nn.Linear stores W as (out, in); we transpose to feed matMul.
  // The head was trained on Y-standardized targets; we unnormalize back to
  // env state space [-1, 1] via state_mean / state_std (Y_mu / Y_sig in
  // the head ckpt). Without this, CEM optimizes a meaningless cost and the
  // predator wanders aimlessly.
  _stateHeadForward(emb) {
    const w0 = this.weights.get('state_head.fc0.weight');
    const b0 = this.weights.get('state_head.fc0.bias');
    const w1 = this.weights.get('state_head.fc1.weight');
    const b1 = this.weights.get('state_head.fc1.bias');
    const w2 = this.weights.get('state_head.fc2.weight');
    const b2 = this.weights.get('state_head.fc2.bias');
    const mean = this.weights.get('state_mean');   // (1, 10) — Y_mu
    const std  = this.weights.get('state_std');    // (1, 10) — Y_sig
    if (!w0) throw new Error('state_head weights missing');
    let x = tf.add(tf.matMul(emb, tf.transpose(w0)), b0);    // (B, 256)
    x = _gelu(x);
    x = tf.add(tf.matMul(x, tf.transpose(w1)), b1);          // (B, 256)
    x = _gelu(x);
    x = tf.add(tf.matMul(x, tf.transpose(w2)), b2);          // (B, 10) y_norm
    if (mean && std) {
      x = tf.add(tf.mul(x, std), mean);                      // → state ∈ [-1, 1]
    }
    return x;
  }
}

// Exact GELU matching PyTorch nn.GELU(approximate='none'):
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// The tanh-approx form differs by ~1e-4 per op; stacked across 6
// transformer layers and the projector + pred_proj that drift compounds
// and was a likely source of the predator-quality gap vs the PyTorch
// reference (silent_jepa_predator_v2). TF.js exposes tf.erf since 4.x.
function _gelu(x) {
  const SQRT_2 = Math.SQRT2;
  return tf.mul(tf.mul(x, 0.5),
                tf.add(1, tf.erf(tf.div(x, SQRT_2))));
}
