#!/usr/bin/env python3
"""
convert_weights.py  —  SafeTensors → flat binary (.llmbin) for llama.c
=======================================================================

Output file layout (all little-endian):
┌─────────────────────────────────────────────────────────────────┐
│ HEADER                                                          │
│   magic      : u32  = 0x4C4C4D42  ("LLMB")                     │
│   version    : u32  = 1                                         │
│   n_tensors  : u32  = number of tensors                         │
│   _pad       : u32  = 0  (align to 16 bytes)                    │
├─────────────────────────────────────────────────────────────────│
│ TENSOR INDEX  (n_tensors entries, each 344 bytes)               │
│   name       : char[256]   null-terminated                      │
│   dtype      : u32                                              │
│                  0 = F32                                        │
│                  1 = BF16                                       │
│                  2 = F16                                        │
│                  3 = FP8_E4M3FN   (IEEE 4-bit exp, 3-bit mant) │
│                  4 = FP8_E5M2     (IEEE 5-bit exp, 2-bit mant) │
│                  5 = FP8_E4M3FNUZ (saturating NaN variant)     │
│                  6 = FP8_E5M2FNUZ (saturating NaN variant)     │
│   ndim       : u32                                              │
│   shape      : u64[8]                                           │
│   offset     : u64  (byte offset from start of DATA section)    │
│   nbytes     : u64  (byte length of tensor data)                │
├─────────────────────────────────────────────────────────────────│
│ DATA                                                            │
│   Raw tensor bytes, 64-byte aligned per tensor.                 │
│   Default (--f32): everything converted to F32.                 │
│   --keep-dtype: stored in native dtype; C must decode.          │
│   FP8 tensors stored as raw uint8 bytes (1 byte per element).   │
└─────────────────────────────────────────────────────────────────┘

Usage:
    # Single shard
    python convert_weights.py model.safetensors model.llmbin

    # Multiple shards → merged into one .llmbin  (most common for 7B/8B/70B)
    python convert_weights.py model-00001-of-00004.safetensors \
                              model-00002-of-00004.safetensors \
                              model-00003-of-00004.safetensors \
                              model-00004-of-00004.safetensors \
                              model.llmbin

    # Or use a glob (shell expands it):
    python convert_weights.py model-*.safetensors model.llmbin

    --keep-dtype        Store tensors in their native dtype (no conversion).
                        FP8 stays as FP8, BF16 stays BF16, etc.
                        Smaller file; your C inference code must handle each dtype.

    --target-dtype X    Convert everything to X before storing.
                        Choices: f32 (default), f16, bf16
                        fp8 tensors are always decoded to at least f16 or f32
                        since there is no portable "store as fp8 and read in C"
                        yet — use --keep-dtype if you want raw fp8 bytes.

Multi-shard note:
    Shards are processed in the order you pass them.  Tensor names must be
    unique across shards (they always are for HuggingFace split checkpoints).
    A duplicate tensor name is a fatal error.

FP8 notes:
    Two families exist in the wild:
      e4m3fn   — 4-bit exponent, 3-bit mantissa, no inf, NaN=0x7F/0xFF
                 Range ±448.  Used by NVIDIA H100, many quantised LLMs.
      e5m2     — 5-bit exponent, 2-bit mantissa, has inf/NaN
                 Range ±57344.  Better for gradients / activations.
    "FNUZ" variants shift the bias so the negative-zero bit codes NaN instead.

    This script handles all four: e4m3fn, e5m2, e4m3fnuz, e5m2fnuz.
    Requires:  pip install ml_dtypes   (provides numpy-compatible fp8 dtypes)

Requires:  pip install safetensors numpy tqdm ml_dtypes
"""

import argparse
import struct
import sys
import os

# ── mandatory deps ────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: pip install numpy")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("Missing dependency: pip install safetensors")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── optional: ml_dtypes for FP8 ───────────────────────────────────────────────
try:
    import ml_dtypes
    HAS_ML_DTYPES = True
    _FP8_E4M3FN   = ml_dtypes.float8_e4m3fn
    _FP8_E5M2     = ml_dtypes.float8_e5m2
    _FP8_E4M3FNUZ = ml_dtypes.float8_e4m3fnuz
    _FP8_E5M2FNUZ = ml_dtypes.float8_e5m2fnuz
except ImportError:
    HAS_ML_DTYPES = False
    _FP8_E4M3FN = _FP8_E5M2 = _FP8_E4M3FNUZ = _FP8_E5M2FNUZ = None

# ── constants ─────────────────────────────────────────────────────────────────
MAGIC   = 0x4C4C4D42   # "LLMB"
VERSION = 1

# dtype codes stored in the .llmbin index
DTYPE_F32        = 0
DTYPE_BF16       = 1
DTYPE_F16        = 2
DTYPE_FP8_E4M3FN   = 3
DTYPE_FP8_E5M2     = 4
DTYPE_FP8_E4M3FNUZ = 5
DTYPE_FP8_E5M2FNUZ = 6

# index entry size: 256 (name) + 4 (dtype) + 4 (ndim) + 64 (shape×8) + 8 (offset) + 8 (nbytes) = 344
INDEX_ENTRY_BYTES = 344
ALIGN = 64   # byte alignment between tensors in DATA section

# ── dtype normalisation ───────────────────────────────────────────────────────
_DTYPE_ALIASES = {
    # F32
    "FLOAT32": "F32", "FLOAT": "F32", "F32": "F32",
    # BF16
    "BFLOAT16": "BF16", "BF16": "BF16",
    # F16
    "FLOAT16": "F16", "HALF": "F16", "F16": "F16",
    # FP8 e4m3fn
    "FLOAT8_E4M3FN":   "FP8_E4M3FN",
    "F8_E4M3FN":       "FP8_E4M3FN",
    "F8_E4M3":         "FP8_E4M3FN",
    # FP8 e5m2
    "FLOAT8_E5M2":     "FP8_E5M2",
    "F8_E5M2":         "FP8_E5M2",
    # FP8 e4m3fnuz
    "FLOAT8_E4M3FNUZ": "FP8_E4M3FNUZ",
    "F8_E4M3FNUZ":     "FP8_E4M3FNUZ",
    # FP8 e5m2fnuz
    "FLOAT8_E5M2FNUZ": "FP8_E5M2FNUZ",
    "F8_E5M2FNUZ":     "FP8_E5M2FNUZ",
}

_DTYPE_TO_CODE = {
    "F32":          DTYPE_F32,
    "BF16":         DTYPE_BF16,
    "F16":          DTYPE_F16,
    "FP8_E4M3FN":   DTYPE_FP8_E4M3FN,
    "FP8_E5M2":     DTYPE_FP8_E5M2,
    "FP8_E4M3FNUZ": DTYPE_FP8_E4M3FNUZ,
    "FP8_E5M2FNUZ": DTYPE_FP8_E5M2FNUZ,
}

_DTYPE_NAMES = {v: k for k, v in _DTYPE_TO_CODE.items()}   # code → name

def _normalise_dtype(raw: str) -> str:
    """Turn any raw dtype string into our canonical name, or raise."""
    key = raw.upper().replace("-", "_")
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    raise ValueError(
        f"Unknown dtype '{raw}'.  Known: {sorted(_DTYPE_ALIASES.keys())}"
    )

# ── conversion helpers ────────────────────────────────────────────────────────

def _bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    """BF16 raw uint16 → float32  (zero-extend mantissa)."""
    u32 = arr.view(np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)

def _f16_to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.view(np.float16).astype(np.float32)

def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """float32 → BF16 raw uint16  (truncate mantissa, round-to-nearest-even)."""
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    rounding = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    rounded  = (u32 + rounding) >> 16
    return rounded.astype(np.uint16)

def _f32_to_f16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32).astype(np.float16)

def _f32_to_fp8(arr: np.ndarray, canonical: str = "FP8_E4M3FN") -> np.ndarray:
    """float32 -> FP8 raw uint8 bytes.  Requires ml_dtypes."""
    if not HAS_ML_DTYPES:
        raise RuntimeError("FP8 conversion requires ml_dtypes: pip install ml_dtypes")
    fp8_dtype_map = {
        "FP8_E4M3FN":   _FP8_E4M3FN,
        "FP8_E5M2":     _FP8_E5M2,
        "FP8_E4M3FNUZ": _FP8_E4M3FNUZ,
        "FP8_E5M2FNUZ": _FP8_E5M2FNUZ,
    }
    fp8_dtype = fp8_dtype_map[canonical]
    return arr.astype(np.float32).astype(fp8_dtype).view(np.uint8)

def _fp8_to_f32(arr: np.ndarray, canonical: str) -> np.ndarray:
    if not HAS_ML_DTYPES:
        raise RuntimeError(
            "FP8 tensors require ml_dtypes: pip install ml_dtypes"
        )
    fp8_dtype_map = {
        "FP8_E4M3FN":   _FP8_E4M3FN,
        "FP8_E5M2":     _FP8_E5M2,
        "FP8_E4M3FNUZ": _FP8_E4M3FNUZ,
        "FP8_E5M2FNUZ": _FP8_E5M2FNUZ,
    }
    fp8_dtype = fp8_dtype_map[canonical]
    raw = arr.view(np.uint8) if arr.dtype != np.uint8 else arr
    return raw.view(fp8_dtype).astype(np.float32)

def _fp8_to_f16(arr: np.ndarray, canonical: str) -> np.ndarray:
    return _fp8_to_f32(arr, canonical).astype(np.float16)

def to_target(arr: np.ndarray, src_dtype: str, target: str) -> tuple[np.ndarray, str]:
    """
    Convert arr from src_dtype to target dtype.
    Returns (converted_array, actual_output_dtype_canonical).

    target: "F32" | "F16" | "BF16" | "FP8_E4M3FN" | "FP8_E5M2" | "KEEP"
    """
    if target == "KEEP":
        if src_dtype.startswith("FP8_"):
            return arr.view(np.uint8), src_dtype
        return arr, src_dtype

    # ── source → F32 first ──────────────────────────────────────────────────
    if src_dtype == "F32":
        f32 = arr.astype(np.float32)
    elif src_dtype == "BF16":
        f32 = _bf16_to_f32(arr)
    elif src_dtype == "F16":
        f32 = _f16_to_f32(arr)
    elif src_dtype.startswith("FP8_"):
        f32 = _fp8_to_f32(arr, src_dtype)
    else:
        raise ValueError(f"Don't know how to decode dtype: {src_dtype}")

    # ── F32 → target ────────────────────────────────────────────────────────
    if target == "F32":
        return f32, "F32"
    elif target == "F16":
        return _f32_to_f16(f32), "F16"
    elif target == "BF16":
        return _f32_to_bf16(f32).view(np.uint16), "BF16"
    elif target.startswith("FP8_"):
        return _f32_to_fp8(f32, target), target
    else:
        raise ValueError(f"Unknown target dtype: {target}")


# ── dtype info for printing ───────────────────────────────────────────────────

def _bytes_per_element(canonical: str) -> float:
    return {
        "F32": 4, "BF16": 2, "F16": 2,
        "FP8_E4M3FN": 1, "FP8_E5M2": 1,
        "FP8_E4M3FNUZ": 1, "FP8_E5M2FNUZ": 1,
    }.get(canonical, 4)

def _is_fp8(canonical: str) -> bool:
    return canonical.startswith("FP8_")

# ── per-shard scanning ────────────────────────────────────────────────────────

def _scan_shard(path: str, target: str, data_cursor: int,
                seen_names: set, dtype_counts: dict) -> tuple[list, int]:
    meta = []

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    with safe_open(path, framework="pt", device="cpu") as f:
        it = tqdm(keys, desc=os.path.basename(path), leave=False) if HAS_TQDM else keys
        for key in it:
            if key in seen_names:
                raise ValueError(
                    f"Duplicate tensor '{key}' found in {path}. "
                    "Shards must not share tensor names."
                )
            seen_names.add(key)

            t = f.get_tensor(key)

            raw_dtype_str = str(t.dtype).split(".")[-1]
            try:
                src_dtype = _normalise_dtype(raw_dtype_str)
            except ValueError as e:
                print(f"\nWARNING: {e} — skipping tensor '{key}'")
                continue

            try:
                arr = t.numpy()
            except Exception:
                import torch
                arr = t.view(torch.uint8).numpy()

            out_arr, out_dtype = to_target(arr, src_dtype, target)
            raw_bytes = out_arr.tobytes()
            nbytes    = len(raw_bytes)

            dtype_counts[out_dtype] = dtype_counts.get(out_dtype, 0) + 1

            meta.append({
                "name":      key,
                "dtype":     _DTYPE_TO_CODE[out_dtype],
                "dtype_str": out_dtype,
                "shape":     list(arr.shape),
                "offset":    data_cursor,
                "nbytes":    nbytes,
                "raw":       raw_bytes,
            })
            data_cursor += align_up(nbytes, ALIGN)

    return meta, data_cursor


# ── main conversion ───────────────────────────────────────────────────────────

def convert(input_paths: list[str], output_path: str, target: str = "F32"):
    n_shards = len(input_paths)
    print(f"Shards : {n_shards}")
    for p in input_paths:
        print(f"         {p}")
    print(f"Output : {output_path}")
    print(f"Target : {target if target != 'KEEP' else 'keep native dtype'}")
    if not HAS_ML_DTYPES:
        print("WARNING: ml_dtypes not installed — FP8 tensors will fail.")
        print("         pip install ml_dtypes")
    print()

    all_meta:    list  = []
    seen_names:  set   = set()
    dtype_counts: dict = {}
    data_cursor = 0

    print("Pass 1: scanning shards …")
    for shard_path in input_paths:
        if not os.path.exists(shard_path):
            sys.exit(f"ERROR: shard not found: {shard_path}")
        shard_meta, data_cursor = _scan_shard(
            shard_path, target, data_cursor, seen_names, dtype_counts
        )
        all_meta.extend(shard_meta)
        print(f"  {os.path.basename(shard_path):50s}  {len(shard_meta):4d} tensors")

    total_data_bytes = data_cursor
    n_tensors        = len(all_meta)

    print(f"\nTotal tensors : {n_tensors}")
    print("Dtype breakdown in output:")
    for dtype_name, count in sorted(dtype_counts.items()):
        bpe = _bytes_per_element(dtype_name)
        tag = "  ← stored as raw uint8" if _is_fp8(dtype_name) and target == "KEEP" else ""
        print(f"  {dtype_name:18s}  {count:4d} tensors  ({bpe} B/elem){tag}")

    header_bytes = 16
    index_bytes  = n_tensors * INDEX_ENTRY_BYTES
    data_start   = align_up(header_bytes + index_bytes, ALIGN)
    total_bytes  = data_start + total_data_bytes

    print(f"\nLayout:")
    print(f"  Header : {header_bytes} B         @ offset 0")
    print(f"  Index  : {index_bytes/1024:.1f} KB       @ offset {header_bytes}  ({n_tensors} × {INDEX_ENTRY_BYTES} B)")
    print(f"  Data   : {total_data_bytes/1024/1024:.1f} MB      @ offset {data_start}")
    print(f"  Total  : {total_bytes/1024/1024:.1f} MB")
    print()

    print("Pass 2: writing …")
    with open(output_path, "wb") as out:
        # header (16 bytes)
        out.write(struct.pack("<IIII", MAGIC, VERSION, n_tensors, 0))

        # index entries
        for m in all_meta:
            name_padded  = m["name"].encode("utf-8")[:255].ljust(256, b"\x00")
            shape_padded = m["shape"] + [0] * (8 - len(m["shape"]))
            entry = (
                name_padded
                + struct.pack("<II",  m["dtype"], len(m["shape"]))
                + struct.pack("<8Q", *shape_padded)
                + struct.pack("<QQ",  m["offset"], m["nbytes"])
            )
            assert len(entry) == INDEX_ENTRY_BYTES, \
                f"BUG: entry size {len(entry)} != {INDEX_ENTRY_BYTES}"
            out.write(entry)

        # pad to data_start
        cur = header_bytes + index_bytes
        if cur < data_start:
            out.write(b"\x00" * (data_start - cur))

        # tensor data
        it = tqdm(all_meta, desc="writing tensors") if HAS_TQDM else all_meta
        for m in it:
            out.write(m["raw"])
            pad = align_up(len(m["raw"]), ALIGN) - len(m["raw"])
            if pad:
                out.write(b"\x00" * pad)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nDone! → {output_path}  ({size_mb:.1f} MB)")


# ── helpers ───────────────────────────────────────────────────────────────────

def align_up(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert .safetensors → .llmbin (flat binary for llama.c)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single shard, default F32
  python conv.py model.safetensors model.llmbin

  # Multiple shards merged (pass output path last)
  python conv.py model-00001-of-00004.safetensors \\
                 model-00002-of-00004.safetensors \\
                 model-00003-of-00004.safetensors \\
                 model-00004-of-00004.safetensors \\
                 model.llmbin

  # Glob (shell expands; output must end with .llmbin)
  python conv.py model-*.safetensors model.llmbin

  # Store as F16 (half the size, tiny quality loss)
  python conv.py model-*.safetensors model.llmbin --target-dtype f16

  # Keep native dtype (your C code must handle each dtype)
  python conv.py model-*.safetensors model.llmbin --keep-dtype
""",
    )
    # Accept one-or-more positional args; the last one is always the output.
    ap.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="One or more .safetensors input shards followed by the .llmbin output path",
    )

    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--keep-dtype", action="store_true",
        help="Store tensors in their native dtype (no conversion). "
             "FP8 is kept as raw uint8 bytes.",
    )
    group.add_argument(
        "--target-dtype",
        choices=["f32", "f16", "bf16", "fp8", "fp8_e4m3fn", "fp8_e5m2"],
        default="f32",
        metavar="DTYPE",
        help="Convert all tensors to this dtype. Choices: f32 (default), f16, bf16, fp8, fp8_e4m3fn, fp8_e5m2.",
    )

    args = ap.parse_args()

    if len(args.paths) < 2:
        ap.error("Provide at least one input .safetensors file and one output .llmbin path.")

    *input_paths, output_path = args.paths

    # Basic sanity checks
    for p in input_paths:
        if not p.endswith(".safetensors"):
            ap.error(f"Expected a .safetensors input, got: {p!r}")
    if not output_path.endswith(".llmbin"):
        ap.error(f"Output path must end with .llmbin, got: {output_path!r}")

    if args.keep_dtype:
        target = "KEEP"
    else:
        target = {
            "f32":        "F32",
            "f16":        "F16",
            "bf16":       "BF16",
            "fp8":        "FP8_E4M3FN",   # default FP8 variant (best for weights)
            "fp8_e4m3fn": "FP8_E4M3FN",
            "fp8_e5m2":   "FP8_E5M2",
        }[args.target_dtype]

    convert(input_paths, output_path, target=target)


if __name__ == "__main__":
    main()