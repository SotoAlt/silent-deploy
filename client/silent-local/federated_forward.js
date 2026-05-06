// Shared TF.js v9 forward graph — used by both the parity test (Node) and
// the browser federated client. Designed to work with either tf-node or
// tf-browser; the caller imports tf and passes it in.
//
// `weights` is a Map<string, tf.Tensor | tf.Variable>. tf.Variable acts as a
// tensor under all ops and is what GradientTape tracks for backward, so the
// same forward function works for both inference (with tf.Tensor weights) and
// training (with tf.Variable weights).

export const MAGIC = [0x52, 0x45, 0x4c, 0x41, 0x59, 0, 0, 0];

// Architectural assumptions BAKED into this forward graph. The server
// validates each game's manifest against these at hub boot — a game
// whose embed_dim/depth/heads/dim_head don't match this file fails to
// register (loud failure beats silent shape mismatch at training time).
//
// embed_dim is the 192 baked into reshape calls; depth is the 6-iteration
// AdaLN loop in predictorForward; heads + dim_head are the qkv split
// constants. in_channels is intentionally NOT here — it lives in the
// encoder weights only and the browser path doesn't run the encoder.
export const ARCH_SIGNATURE = Object.freeze({
  embed_dim: 192,
  predictor_depth: 6,
  predictor_heads: 16,
  dim_head: 64,
});

// IEEE-754 half (fp16) → single (fp32) bulk decode.
// Reconstructs each fp32 bit pattern from fp16 fields and reinterprets
// via Uint32Array → Float32Array. Normals: sign + (exp+112)<<23 +
// frac<<13. Subnormals/zero/inf/NaN handled in the rare-path branches.
// 60 MB worth of weights decode in <500 ms on M-series CPUs.
function fp16ToFp32(uint16Buf) {
  const u16 = new Uint16Array(uint16Buf);
  const out = new Float32Array(u16.length);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h & 0x7c00) >>> 10;
    const frac = h & 0x03ff;
    if (exp === 0) {
      if (frac === 0) {
        u32[i] = sign;                 // ±0
      } else {
        // Subnormal: renormalize. m * 2^-24 in fp32 form.
        let m = frac, e = -14;
        while ((m & 0x0400) === 0) { m <<= 1; e -= 1; }
        m &= 0x03ff;
        u32[i] = sign | ((e + 127) << 23) | (m << 13);
      }
    } else if (exp === 0x1f) {
      u32[i] = sign | 0x7f800000 | (frac << 13);   // ±Inf or NaN
    } else {
      u32[i] = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
  }
  return out;
}

export function parseRelayBlob(buf) {
  const u = new Uint8Array(buf);
  const dv = new DataView(u.buffer, u.byteOffset, u.byteLength);
  for (let i = 0; i < 8; i++) {
    if (u[i] !== MAGIC[i]) throw new Error(`bad magic at byte ${i}`);
  }
  let off = 8;
  const n = dv.getUint32(off, true); off += 4;
  const out = new Map();
  for (let k = 0; k < n; k++) {
    const nameLen = dv.getUint32(off, true); off += 4;
    const name = new TextDecoder().decode(u.subarray(off, off + nameLen));
    off += nameLen;
    const dtype = u[off]; off += 1;
    const ndim = u[off]; off += 1;
    const shape = [];
    for (let d = 0; d < ndim; d++) { shape.push(dv.getUint32(off, true)); off += 4; }
    const dataLen = dv.getUint32(off, true); off += 4;
    const slice = u.buffer.slice(u.byteOffset + off, u.byteOffset + off + dataLen);
    off += dataLen;
    let data;
    if (dtype === 0) {
      data = new Float32Array(slice);
    } else if (dtype === 2) {
      // Wire is fp16; promote to fp32 so the rest of the training stack
      // (TF.js variables, optimizer, signSGD) doesn't have to know.
      data = fp16ToFp32(slice);
    } else {
      throw new Error(`unsupported dtype ${dtype} on ${name}`);
    }
    out.set(name, { shape, data });
  }
  return out;
}

export function makeForward(tf) {
  // Closure over a specific tf import (tfjs-node or tfjs browser).

  const T = (weights, name) => {
    const w = weights.get(name);
    if (!w) throw new Error(`missing weight: ${name}`);
    return w;
  };

  function gelu(x) {
    return tf.mul(tf.mul(x, 0.5),
                  tf.add(tf.scalar(1), tf.erf(tf.mul(x, 1.0 / Math.sqrt(2)))));
  }
  function silu(x) { return tf.mul(x, tf.sigmoid(x)); }
  function layerNorm(x, w, b, eps = 1e-6) {
    const mean = tf.mean(x, -1, true);
    const variance = tf.mean(tf.square(tf.sub(x, mean)), -1, true);
    const norm = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, eps)));
    if (w === null && b === null) return norm;
    return tf.add(tf.mul(norm, w), b);
  }
  function linear(x, w, b) {
    // tfjs 4.22 has a known matMul backward bug when batched input meets a
    // 2D Variable weight (gradient w.r.t. w doesn't sum over batch dim).
    // Sidestep by flattening leading batch dims into a single axis, doing
    // a 2D matMul, then reshaping back. Forward output is identical.
    const shape = x.shape;
    const lastDim = shape[shape.length - 1];
    const flat = shape.length > 2 ? tf.reshape(x, [-1, lastDim]) : x;
    let y = tf.matMul(flat, w, false, true);
    if (b !== null) y = tf.add(y, b);
    return shape.length > 2
      ? tf.reshape(y, [...shape.slice(0, -1), y.shape[1]])
      : y;
  }
  function patchEmbed(x, w, b) {
    const xNhwc = tf.transpose(x, [0, 2, 3, 1]);
    const wTfjs = tf.transpose(w, [2, 3, 1, 0]);
    let y = tf.conv2d(xNhwc, wTfjs, [14, 14], 'valid');
    y = tf.add(y, b);
    const [B, H, W, C] = y.shape;
    return tf.reshape(y, [B, H * W, C]);
  }
  function encoderAttention(x, qkvW, qkvB, projW, projB, numHeads) {
    const B = x.shape[0], Tlen = x.shape[1], D = x.shape[2];
    const Hd = D / numHeads;
    const flatX = tf.reshape(x, [B * Tlen, D]);
    let qkv = tf.add(tf.matMul(flatX, qkvW, false, true), qkvB);
    qkv = tf.reshape(qkv, [B, Tlen, 3, numHeads, Hd]);
    qkv = tf.transpose(qkv, [2, 0, 3, 1, 4]);
    const [q, k, v] = tf.split(qkv, 3, 0);
    const Q = tf.squeeze(q, [0]), K = tf.squeeze(k, [0]), V = tf.squeeze(v, [0]);
    let attn = tf.mul(tf.matMul(Q, K, false, true), 1.0 / Math.sqrt(Hd));
    attn = tf.softmax(attn, -1);
    let out = tf.matMul(attn, V);
    out = tf.transpose(out, [0, 2, 1, 3]);
    out = tf.reshape(out, [B * Tlen, D]);
    out = tf.add(tf.matMul(out, projW, false, true), projB);
    return tf.reshape(out, [B, Tlen, D]);
  }

  function vitTinyForward(pixels, weights, depth = 12) {
    let x = patchEmbed(pixels,
                       T(weights, 'encoder.vit.patch_embed.proj.weight'),
                       T(weights, 'encoder.vit.patch_embed.proj.bias'));
    const cls = T(weights, 'encoder.vit.cls_token');
    const pos = T(weights, 'encoder.vit.pos_embed');
    const B = x.shape[0];
    const clsB = tf.tile(cls, [B, 1, 1]);
    x = tf.concat([clsB, x], 1);
    x = tf.add(x, pos);
    for (let i = 0; i < depth; i++) {
      const p = `encoder.vit.blocks.${i}.`;
      const xn = layerNorm(x, T(weights, p + 'norm1.weight'), T(weights, p + 'norm1.bias'));
      const a = encoderAttention(xn,
                                 T(weights, p + 'attn.qkv.weight'),
                                 T(weights, p + 'attn.qkv.bias'),
                                 T(weights, p + 'attn.proj.weight'),
                                 T(weights, p + 'attn.proj.bias'),
                                 3);
      x = tf.add(x, a);
      const xn2 = layerNorm(x, T(weights, p + 'norm2.weight'), T(weights, p + 'norm2.bias'));
      let m = linear(xn2, T(weights, p + 'mlp.fc1.weight'), T(weights, p + 'mlp.fc1.bias'));
      m = gelu(m);
      m = linear(m, T(weights, p + 'mlp.fc2.weight'), T(weights, p + 'mlp.fc2.bias'));
      x = tf.add(x, m);
    }
    x = layerNorm(x,
                  T(weights, 'encoder.vit.norm.weight'),
                  T(weights, 'encoder.vit.norm.bias'));
    return tf.reshape(tf.slice(x, [0, 0, 0], [B, 1, 192]), [B, 192]);
  }

  function mlpProjectorForward(x, weights, prefix) {
    const bn = (y, idx) => {
      const w = T(weights, `${prefix}.net.${idx}.weight`);
      const b = T(weights, `${prefix}.net.${idx}.bias`);
      const rm = T(weights, `${prefix}.net.${idx}.running_mean`);
      const rv = T(weights, `${prefix}.net.${idx}.running_var`);
      return tf.add(tf.mul(tf.div(tf.sub(y, rm), tf.sqrt(tf.add(rv, 1e-5))), w), b);
    };
    const shape = x.shape;
    const lastDim = shape[shape.length - 1];
    const flat = tf.reshape(x, [-1, lastDim]);
    let y = linear(flat, T(weights, `${prefix}.net.0.weight`), T(weights, `${prefix}.net.0.bias`));
    y = bn(y, 1);
    y = tf.relu(y);
    y = linear(y, T(weights, `${prefix}.net.3.weight`), T(weights, `${prefix}.net.3.bias`));
    y = bn(y, 4);
    y = tf.relu(y);
    y = linear(y, T(weights, `${prefix}.net.6.weight`), T(weights, `${prefix}.net.6.bias`));
    return tf.reshape(y, [...shape.slice(0, -1), y.shape[1]]);
  }

  function actionEncoderForward(actions, weights) {
    return linear(actions,
                  T(weights, 'action_encoder.weight'),
                  T(weights, 'action_encoder.bias'));
  }

  function predictorAttention(x, qkvW, projW, projB, numHeads, dimHead, causal) {
    const B = x.shape[0], Tlen = x.shape[1];
    const inner = numHeads * dimHead;
    // Flatten batch dim before matMul against the 2D weight (tfjs gradient bug).
    const flatX = tf.reshape(x, [B * Tlen, x.shape[2]]);
    let qkv = tf.matMul(flatX, qkvW, false, true);    // [B*T, 3*inner]
    qkv = tf.reshape(qkv, [B, Tlen, 3, numHeads, dimHead]);
    qkv = tf.transpose(qkv, [2, 0, 3, 1, 4]);
    const [q, k, v] = tf.split(qkv, 3, 0);
    const Q = tf.squeeze(q, [0]), K = tf.squeeze(k, [0]), V = tf.squeeze(v, [0]);
    let attn = tf.mul(tf.matMul(Q, K, false, true), 1.0 / Math.sqrt(dimHead));
    if (causal) {
      const mask = tf.linalg.bandPart(tf.ones([Tlen, Tlen]), -1, 0);
      const negInf = tf.where(tf.equal(mask, 0),
                              tf.fill([Tlen, Tlen], -1e9),
                              tf.zeros([Tlen, Tlen]));
      attn = tf.add(attn, negInf);
    }
    attn = tf.softmax(attn, -1);
    let out = tf.matMul(attn, V);
    out = tf.transpose(out, [0, 2, 1, 3]);
    out = tf.reshape(out, [B * Tlen, inner]);
    out = tf.add(tf.matMul(out, projW, false, true), projB);
    return tf.reshape(out, [B, Tlen, x.shape[2]]);
  }

  function adaLNBlock(x, c, weights, prefix, numHeads, dimHead) {
    const c_silu = silu(c);
    const mod = linear(c_silu,
                       T(weights, `${prefix}.adaln_mod.1.weight`),
                       T(weights, `${prefix}.adaln_mod.1.bias`));
    const [shift1, scale1, gate1, shift2, scale2, gate2] = tf.split(mod, 6, -1);

    const x_norm1 = layerNorm(x, null, null);
    const h1 = tf.add(tf.mul(x_norm1, tf.add(tf.scalar(1), scale1)), shift1);
    const h1_inner = layerNorm(h1,
                               T(weights, `${prefix}.attn.norm.weight`),
                               T(weights, `${prefix}.attn.norm.bias`));
    const a = predictorAttention(h1_inner,
                                 T(weights, `${prefix}.attn.to_qkv.weight`),
                                 T(weights, `${prefix}.attn.to_out.0.weight`),
                                 T(weights, `${prefix}.attn.to_out.0.bias`),
                                 numHeads, dimHead, true);
    x = tf.add(x, tf.mul(gate1, a));

    const x_norm2 = layerNorm(x, null, null);
    const h2 = tf.add(tf.mul(x_norm2, tf.add(tf.scalar(1), scale2)), shift2);
    let m = layerNorm(h2,
                      T(weights, `${prefix}.mlp.net.0.weight`),
                      T(weights, `${prefix}.mlp.net.0.bias`));
    m = linear(m,
               T(weights, `${prefix}.mlp.net.1.weight`),
               T(weights, `${prefix}.mlp.net.1.bias`));
    m = gelu(m);
    m = linear(m,
               T(weights, `${prefix}.mlp.net.4.weight`),
               T(weights, `${prefix}.mlp.net.4.bias`));
    x = tf.add(x, tf.mul(gate2, m));
    return x;
  }

  function predictorForward(emb, act_emb, weights, depth = 6, numHeads = 16, dimHead = 64) {
    let x = linear(emb,
                   T(weights, 'predictor.in_proj.weight'),
                   T(weights, 'predictor.in_proj.bias'));
    const pos = T(weights, 'predictor.pos_emb');
    const Tlen = x.shape[1];
    x = tf.add(x, tf.slice(pos, [0, 0, 0], [1, Tlen, x.shape[2]]));
    for (let i = 0; i < depth; i++) {
      x = adaLNBlock(x, act_emb, weights, `predictor.blocks.${i}`, numHeads, dimHead);
    }
    x = layerNorm(x, null, null);
    return linear(x,
                  T(weights, 'predictor.out_proj.weight'),
                  T(weights, 'predictor.out_proj.bias'));
  }

  // Full predict path — the same fn the LeWM forward uses for the v9
  // composite loss. Returns { emb, act_emb, ctx_emb, pred_emb }.
  function fullPredict(pixels, actions, weights) {
    const [B, Tp1] = pixels.shape;
    const flatPixels = tf.reshape(pixels,
      [B * Tp1, pixels.shape[2], pixels.shape[3], pixels.shape[4]]);
    const h = vitTinyForward(flatPixels, weights);
    const emb = mlpProjectorForward(h, weights, 'projector');
    const embReshaped = tf.reshape(emb, [B, Tp1, 192]);
    const act_emb = actionEncoderForward(actions, weights);
    const ctx_emb = tf.slice(embReshaped, [0, 0, 0], [B, Tp1 - 1, 192]);
    const tgt_emb = tf.slice(embReshaped, [0, 1, 0], [B, Tp1 - 1, 192]);
    const predHidden = predictorForward(ctx_emb, act_emb, weights);
    const pred_emb = mlpProjectorForward(predHidden, weights, 'pred_proj');
    return { emb: embReshaped, act_emb, ctx_emb, tgt_emb, pred_emb };
  }

  // Lightweight forward for the browser federated client. Takes
  // already-encoded `emb` (B, T+1, 192) — the encoder forward is done
  // server-side and shipped via /training_batch, so the browser only runs
  // the trainable stack: action_encoder + predictor + pred_proj.
  // This is what the WebGL backend can train within a reasonable budget.
  function predictTrainable(emb, actions, weights) {
    const [B, Tp1] = emb.shape;
    const ctx_emb = tf.slice(emb, [0, 0, 0], [B, Tp1 - 1, 192]);
    const tgt_emb = tf.slice(emb, [0, 1, 0], [B, Tp1 - 1, 192]);
    const act_emb = actionEncoderForward(actions, weights);
    const pred_h = predictorForward(ctx_emb, act_emb, weights);
    const pred_emb = mlpProjectorForward(pred_h, weights, 'pred_proj');
    return { ctx_emb, tgt_emb, act_emb, pred_emb };
  }

  return { fullPredict, predictTrainable, vitTinyForward,
           mlpProjectorForward, actionEncoderForward, predictorForward };
}
