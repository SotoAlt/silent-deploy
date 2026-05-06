"""Generate deterministic mel-spec test cases for the silent-local
parity harness.

Why this exists:
  silent.py:_synth_footsteps calls np.random.randn() without a seed,
  so byte-parity on real silent audio is fundamentally impossible —
  Python and JS will see different noise samples.

  Instead, validate the JS mel-spec port against canonical inputs
  where reproducibility is guaranteed: impulse, sine waves at known
  frequencies, white noise with a fixed seed. If JS mel matches
  Python's librosa output to ~1e-7 on these, the implementation is
  correct, and statistical-noise differences in the live silent
  pipeline are encoder-robustness territory, not a port bug.

Output: parity_harness/fixtures/canonical_mel.npz with case0_*, case1_*,
etc. Each case has `_audio` (input) + `_mel` (librosa reference).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from librosa.feature import melspectrogram


SAMPLE_RATE = 16000
OBS_WINDOW_SEC = 0.5
N_MELS = 64
HOP_LENGTH = 160
N_FFT = 512


def _mel(audio: np.ndarray) -> np.ndarray:
    m = melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT, power=2.0,
    )
    return np.log1p(m).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='web/parity_harness/fixtures/canonical_mel.npz')
    args = ap.parse_args()

    n_samp = int(SAMPLE_RATE * OBS_WINDOW_SEC)  # 8000
    t = np.arange(n_samp, dtype=np.float32) / SAMPLE_RATE
    cases: dict = {}

    # case0: silence
    cases['case0_audio'] = np.zeros(n_samp, dtype=np.float32)
    cases['case0_mel']   = _mel(cases['case0_audio'])

    # case1: unit impulse at sample 0
    a = np.zeros(n_samp, dtype=np.float32); a[0] = 1.0
    cases['case1_audio'] = a
    cases['case1_mel']   = _mel(a)

    # case2: 440 Hz sine, amplitude 0.5
    cases['case2_audio'] = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    cases['case2_mel']   = _mel(cases['case2_audio'])

    # case3: 80 Hz sine (matches the silent beacon hum)
    cases['case3_audio'] = (0.02 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)
    cases['case3_mel']   = _mel(cases['case3_audio'])

    # case4: 1000 Hz sine (matches silent ping carrier)
    cases['case4_audio'] = (0.7 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    cases['case4_mel']   = _mel(cases['case4_audio'])

    # case5: seeded white noise, amp 0.1
    rng = np.random.default_rng(42)
    cases['case5_audio'] = (0.1 * rng.standard_normal(n_samp)).astype(np.float32)
    cases['case5_mel']   = _mel(cases['case5_audio'])

    # case6: seeded white noise, amp 1.0 (heavier)
    rng = np.random.default_rng(7)
    cases['case6_audio'] = (1.0 * rng.standard_normal(n_samp)).astype(np.float32)
    cases['case6_mel']   = _mel(cases['case6_audio'])

    # case7: mixed sine + impulse + noise (stress the linearity)
    rng = np.random.default_rng(123)
    sig = 0.3 * np.sin(2 * np.pi * 220.0 * t) \
        + 0.2 * np.sin(2 * np.pi * 1500.0 * t) \
        + 0.1 * rng.standard_normal(n_samp)
    sig[100] += 0.5
    cases['case7_audio'] = sig.astype(np.float32)
    cases['case7_mel']   = _mel(cases['case7_audio'])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cases)
    print(f'[canon-mel] wrote {len(cases) // 2} cases to {out_path}')
    print(f'[canon-mel] file size: {out_path.stat().st_size / 1e6:.2f} MB')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
