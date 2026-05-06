"""Extract silent_v1 weights (encoder + projector + predictor +
pred_proj + action_encoder + state_head) into a single fp16 RELAY-binary
blob the in-browser TF.js port can fetch + parse via parseRelayBlob.

Format mirrors federated/protocol.py encode_arrays — magic + n_arrays +
per-array {name, dtype, ndim, shape, data}.

Usage (inside the silent container):
  python3 -m scripts.extract_silent_weights \\
    --jepa-ckpt /app/checkpoints/silent_v1_3e_ep030.pt \\
    --jepa-head /app/checkpoints/3e_ep030_head_uniform.pt \\
    --out /tmp/silent_v1.weights.bin \\
    --fp16
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Build the same LeWM the silent predator builds at runtime.
from scripts.train_silent_v1_lewm import LeWM


# ---- Federation RELAY-binary encoder (vendored to avoid the dep) ----
_MAGIC = b"RELAY\x00\x00\x00"
_DT_FP32, _DT_INT8, _DT_FP16 = 0, 1, 2

def encode_arrays(arrays: dict, cast_fp16: bool = False) -> bytes:
    chunks = [_MAGIC, struct.pack("<I", len(arrays))]
    for name, arr in arrays.items():
        if cast_fp16 and arr.dtype == np.dtype("float32"):
            arr = arr.astype("<f2", copy=False)
        if arr.dtype == np.dtype("float32"):
            dt, wire = _DT_FP32, "<f4"
        elif arr.dtype == np.dtype("float16"):
            dt, wire = _DT_FP16, "<f2"
        elif arr.dtype == np.dtype("int8"):
            dt, wire = _DT_INT8, "|i1"
        else:
            raise TypeError(f"unsupported dtype {arr.dtype} on {name}")
        arr_le = arr.astype(wire, copy=False)
        nb = name.encode("utf-8")
        chunks += [struct.pack("<I", len(nb)), nb,
                   struct.pack("<BB", dt, arr.ndim)]
        if arr.ndim > 0:
            chunks.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        d = arr_le.tobytes()
        chunks += [struct.pack("<I", len(d)), d]
    return b"".join(chunks)


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--jepa-ckpt", required=True, type=Path)
    pa.add_argument("--jepa-head", required=True, type=Path)
    pa.add_argument("--out", required=True, type=Path)
    pa.add_argument("--fp16", action="store_true",
                    help="Cast fp32 weights to fp16 — halves file size, "
                         "in-browser parseRelayBlob auto-promotes back.")
    args = pa.parse_args()

    print(f"[extract] loading {args.jepa_ckpt}", flush=True)
    ckpt = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt['config']

    model = LeWM(
        in_channels=cfg.get('in_channels', 4),
        hidden_dim=cfg['hidden_dim'],
        embed_dim=cfg['embed_dim'],
        action_dim=cfg.get('action_dim', 3),
        frameskip=cfg['frameskip'],
        history=cfg['history'],
        depth=cfg['depth'],
        heads=cfg['heads'],
        dim_head=cfg['dim_head'],
        mlp_dim=cfg['mlp_dim'],
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()

    # State head: small MLP 192 -> 256 -> 256 -> 10 (silent state_dim=10).
    # silent-deploy head ckpt format: {head_state, norm: {Y_mu, Y_sig}, ...}
    # head_state has Sequential keys '0.weight', '0.bias', '2.*', '4.*'.
    head_ckpt = torch.load(args.jepa_head, map_location="cpu", weights_only=False)
    head_sd = head_ckpt['head_state']
    norm = head_ckpt.get('norm', {})
    state_mean = norm.get('Y_mu', None)
    state_std  = norm.get('Y_sig', None)

    keep_prefixes = ("encoder.", "projector.", "predictor.",
                     "pred_proj.", "action_encoder.")
    arrays: dict = {}
    for name, p in model.named_parameters():
        if name.startswith(keep_prefixes):
            arrays[name] = p.detach().cpu().numpy().astype("float32")
    for name, b in model.named_buffers():
        if name.startswith(keep_prefixes) and ("running_mean" in name or "running_var" in name):
            arrays[name] = b.detach().cpu().numpy().astype("float32")

    # State-head MLP — Sequential weights '0.weight', '0.bias', '2.*', '4.*'.
    # Normalize to state_head.fc{0,1,2}.{weight,bias} for the JS port.
    layer_map = {'0': 'fc0', '2': 'fc1', '4': 'fc2'}
    for k, v in head_sd.items():
        idx, kind = k.split('.', 1)        # '0', 'weight'
        new_key = f"state_head.{layer_map[idx]}.{kind}"
        arrays[new_key] = v.detach().cpu().numpy().astype("float32")

    if state_mean is not None:
        arrays['state_mean'] = state_mean.detach().cpu().numpy().astype("float32") \
            if hasattr(state_mean, 'detach') else np.asarray(state_mean, dtype="float32")
    if state_std is not None:
        arrays['state_std']  = state_std.detach().cpu().numpy().astype("float32") \
            if hasattr(state_std, 'detach') else np.asarray(state_std, dtype="float32")

    print(f"[extract] {len(arrays)} tensors:", flush=True)
    total = 0
    for n, a in arrays.items():
        total += a.nbytes if not args.fp16 or a.dtype != np.dtype("float32") else a.nbytes // 2
        print(f"  {n:55s} {str(tuple(a.shape)):20s}  {a.dtype}", flush=True)
    print(f"[extract] effective size {total / 1e6:.1f} MB ({'fp16' if args.fp16 else 'fp32'})", flush=True)

    blob = encode_arrays(arrays, cast_fp16=args.fp16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"[extract] wrote {args.out} ({len(blob):,} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
