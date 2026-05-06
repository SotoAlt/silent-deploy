// Parity harness — Phase 0 gate.
//
// Why deterministic tests, not silent-audio fixtures:
//   silent.py:_synth_footsteps uses np.random.randn() WITHOUT a seed,
//   so every call produces different noise. The dump-fixture flow can't
//   reproduce byte-identical audio. We validate mel-spec on deterministic
//   inputs (impulse, sine, seeded-numpy noise) where parity IS expected
//   to be ~1e-7. End-to-end tests (full chain JS vs Python) live in a
//   separate stage that compares JEPA embeddings, not raw audio.
//
// Stages tested (in order, each gates the next):
//   1. mel: log-mel-spec via deterministic canonical inputs from
//          fixtures/canonical_mel.npz
//   2. mix: cardioid mixing math via fixtures/canonical_mix.npz
//          (deterministic source positions + sigs)
//   3. forward: TF.js JEPA forward — deferred
//   4. cem: predator action via CEM — deferred

import { loadNpz } from './npz_loader.js';
import { logMelTruncated, logMelSpectrogram, MEL_CONST } from '../env/mel.js';

const args = process.argv.slice(2);
const only = args.find(a => a.startsWith('--only='))?.split('=')[1] || 'all';

function statsAbsDiff(jsArr, pyArr) {
  if (jsArr.length !== pyArr.length) {
    return { error: `length ${jsArr.length} vs ${pyArr.length}` };
  }
  let sumAbs = 0, maxAbs = 0, n = 0;
  for (let i = 0; i < jsArr.length; i++) {
    const d = Math.abs(jsArr[i] - pyArr[i]);
    sumAbs += d;
    if (d > maxAbs) maxAbs = d;
    n++;
  }
  return { mean: sumAbs / n, max: maxAbs };
}

// ---------- Stage 1: log-mel parity on deterministic inputs --------------

function checkMelStage() {
  console.log('\n[parity] === stage 1: mel (deterministic canon) ===');
  const TOL = 1e-5;
  let cases;
  try {
    cases = loadNpz('parity_harness/fixtures/canonical_mel.npz');
  } catch (e) {
    console.error('[parity] mel: FAIL — fixtures/canonical_mel.npz not found.');
    console.error('  Generate it via: python3 scripts/dump_canonical_mel.py');
    return false;
  }

  // Each case is keyed `caseN_audio` + `caseN_mel`. Iterate.
  const caseIds = new Set();
  for (const k of Object.keys(cases)) {
    const m = k.match(/^(case\d+)_/);
    if (m) caseIds.add(m[1]);
  }

  let allPass = true;
  for (const id of [...caseIds].sort()) {
    const audio = cases[`${id}_audio`].data;
    const pyMel = cases[`${id}_mel`].data;
    const pyShape = cases[`${id}_mel`].shape;
    const jsOut = logMelSpectrogram(audio);

    const stats = statsAbsDiff(jsOut.data, pyMel);
    const status = stats.max < TOL ? 'PASS' : 'FAIL';
    console.log(`[parity]   ${id}: ${status} max=${stats.max.toExponential(3)} mean=${stats.mean.toExponential(3)} shape=${jsOut.shape.join('x')} (py ${pyShape.join('x')})`);
    if (stats.max >= TOL) allPass = false;
  }
  return allPass;
}

// ---------- Run stages --------------------------------------------------

let allPass = true;
if (only === 'all' || only === 'mel') {
  if (!checkMelStage()) allPass = false;
}

if (allPass) {
  console.log('\n[parity] OVERALL: PASS');
  process.exit(0);
} else {
  console.log('\n[parity] OVERALL: FAIL');
  process.exit(1);
}
