// JS port of librosa.feature.melspectrogram + log1p, parity-tested
// against the Python reference in silent.py:get_audio_obs.
//
// Pipeline:
//   1. Pad y on both sides by n_fft/2 (center=True default)
//   2. Frame into n_frames overlapping windows of length n_fft
//   3. Apply Hann window (periodic, matches librosa.filters.get_window)
//   4. Real-FFT each frame → (n_fft/2 + 1) complex coefficients
//   5. Power = |X|^power (power=2.0 here)
//   6. Mel filterbank @ power → (n_mels, n_frames)
//   7. log1p
//
// All constants match silent.py: SAMPLE_RATE=16000, N_FFT=512,
// HOP_LENGTH=160, N_MELS=64, power=2.0, htk=False, norm='slaney'.

const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const N_MELS = 64;
const POWER = 2.0;

// ---------- Hann window (periodic — matches scipy / librosa default) ----------
function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  }
  return w;
}

// ---------- Iterative Cooley-Tukey FFT (radix-2, in-place complex) -----------
// Input: re[], im[] of length n (n = power of 2). Output overwrites in place.
function fft(re, im) {
  const n = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wReBase = Math.cos(ang);
    const wImBase = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1, wIm = 0;
      for (let k = 0; k < half; k++) {
        const tRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
        const tIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
        re[i + k + half] = re[i + k] - tRe;
        im[i + k + half] = im[i + k] - tIm;
        re[i + k] += tRe;
        im[i + k] += tIm;
        const wReNext = wRe * wReBase - wIm * wImBase;
        const wImNext = wRe * wImBase + wIm * wReBase;
        wRe = wReNext; wIm = wImNext;
      }
    }
  }
}

// ---------- Mel filterbank (htk=False / slaney scale, norm='slaney') --------
//
// librosa.filters.mel reference:
//   mel(sr, n_fft, n_mels=128, fmin=0, fmax=sr/2, htk=False, norm='slaney')
//
// Slaney scale (mel_to_hz / hz_to_mel below) matches librosa's _mel_to_hz +
// _hz_to_mel with the htk=False branch.

function hzToMelSlaney(hz) {
  const f_min = 0.0;
  const f_sp = 200.0 / 3.0;
  const min_log_hz = 1000.0;
  const min_log_mel = (min_log_hz - f_min) / f_sp;
  const logstep = Math.log(6.4) / 27.0;
  if (hz < min_log_hz) return (hz - f_min) / f_sp;
  return min_log_mel + Math.log(hz / min_log_hz) / logstep;
}

function melToHzSlaney(mel) {
  const f_min = 0.0;
  const f_sp = 200.0 / 3.0;
  const min_log_hz = 1000.0;
  const min_log_mel = (min_log_hz - f_min) / f_sp;
  const logstep = Math.log(6.4) / 27.0;
  if (mel < min_log_mel) return f_min + f_sp * mel;
  return min_log_hz * Math.exp(logstep * (mel - min_log_mel));
}

function buildMelFilterbank({ sr, nFft, nMels, fmin = 0.0, fmax = null }) {
  if (fmax == null) fmax = sr / 2;
  const nFftBins = nFft / 2 + 1;

  // FFT bin centers in Hz: linspace(0, sr/2, n_fft/2 + 1)
  const fftFreqs = new Float64Array(nFftBins);
  for (let i = 0; i < nFftBins; i++) fftFreqs[i] = (i * sr) / nFft;

  // n_mels + 2 mel-band edges in Hz
  const minMel = hzToMelSlaney(fmin);
  const maxMel = hzToMelSlaney(fmax);
  const melPts = new Float64Array(nMels + 2);
  for (let i = 0; i < melPts.length; i++) {
    melPts[i] = minMel + ((maxMel - minMel) * i) / (nMels + 1);
  }
  const hzPts = melPts.map(melToHzSlaney);

  // Triangular filters (n_mels rows × n_fft_bins cols)
  const filt = new Float32Array(nMels * nFftBins);
  for (let m = 0; m < nMels; m++) {
    const left = hzPts[m];
    const center = hzPts[m + 1];
    const right = hzPts[m + 2];
    for (let k = 0; k < nFftBins; k++) {
      const f = fftFreqs[k];
      let v = 0;
      if (f >= left && f <= center) v = (f - left) / (center - left);
      else if (f > center && f <= right) v = (right - f) / (right - center);
      if (v < 0) v = 0;
      filt[m * nFftBins + k] = v;
    }
    // Slaney norm: divide by 2 / (right - left)
    // (== multiplying triangle by 2 / (f_high - f_low))
    const enorm = 2.0 / (right - left);
    for (let k = 0; k < nFftBins; k++) {
      filt[m * nFftBins + k] *= enorm;
    }
  }
  return { filt, nFftBins };
}

// ---------- Centered constant=0 padding for STFT (center=True) -------------
// librosa 0.10.2 default pad_mode is 'constant' (verified at runtime).
function padCenterConstant(y, padLen) {
  const out = new Float32Array(y.length + 2 * padLen);
  out.set(y, padLen);
  return out;
}

// ---------- Public API: logMelSpectrogram(audio) ---------------------------
//
// audio: Float32Array shape (n_samp,)  — single-channel mono
// returns: Float32Array shape (N_MELS, n_frames) in row-major,
//          values are log1p(mel-power).
//
// For multi-channel input call this once per channel.

let _MEL_CACHE = null;
function getMelCache() {
  if (_MEL_CACHE) return _MEL_CACHE;
  const win = hannWindow(N_FFT);
  const fb = buildMelFilterbank({ sr: SAMPLE_RATE, nFft: N_FFT, nMels: N_MELS });
  _MEL_CACHE = { win, ...fb };
  return _MEL_CACHE;
}

export function logMelSpectrogram(audio) {
  const { win, filt, nFftBins } = getMelCache();
  const padLen = N_FFT / 2;
  const padded = padCenterConstant(audio, padLen);
  const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP_LENGTH);

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  // power-spec scratch: (nFftBins,)
  const power = new Float32Array(nFftBins);
  // output rows: (n_mels, n_frames)
  const out = new Float32Array(N_MELS * nFrames);

  for (let f = 0; f < nFrames; f++) {
    const start = f * HOP_LENGTH;
    // copy + apply window
    for (let i = 0; i < N_FFT; i++) {
      re[i] = padded[start + i] * win[i];
      im[i] = 0;
    }
    fft(re, im);
    // |X|^2 over the rfft half
    for (let k = 0; k < nFftBins; k++) {
      const a = re[k], b = im[k];
      power[k] = a * a + b * b;
    }
    if (POWER !== 2.0) {
      for (let k = 0; k < nFftBins; k++) power[k] = Math.pow(power[k], POWER / 2);
    }
    // mel basis @ power → column f of output, length n_mels
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const base = m * nFftBins;
      for (let k = 0; k < nFftBins; k++) s += filt[base + k] * power[k];
      out[m * nFrames + f] = Math.log1p(s);
    }
  }

  return { data: out, shape: [N_MELS, nFrames] };
}

// Convenience: pad/truncate the time axis to a fixed target_T (silent.py
// pads/truncates to 50). For the parity harness we'll match this.
export function logMelTruncated(audio, targetT = 50) {
  const { data, shape } = logMelSpectrogram(audio);
  const [nMels, nFrames] = shape;
  if (nFrames === targetT) return { data, shape };
  if (nFrames > targetT) {
    const out = new Float32Array(nMels * targetT);
    for (let m = 0; m < nMels; m++) {
      for (let t = 0; t < targetT; t++) {
        out[m * targetT + t] = data[m * nFrames + t];
      }
    }
    return { data: out, shape: [nMels, targetT] };
  }
  // Pad with zeros
  const out = new Float32Array(nMels * targetT);
  for (let m = 0; m < nMels; m++) {
    for (let t = 0; t < nFrames; t++) {
      out[m * targetT + t] = data[m * nFrames + t];
    }
  }
  return { data: out, shape: [nMels, targetT] };
}

export const MEL_CONST = { SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, POWER };
