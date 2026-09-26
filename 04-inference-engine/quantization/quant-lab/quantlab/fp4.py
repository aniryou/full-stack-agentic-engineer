"""fp4.py — FP4 block formats (NVFP4, MXFP4): the grid, the two-level scales, the bytes, and when they pay.

One idea: four bits cannot hold a useful float on their own — E2M1 has 15 values, {0, 0.5, 1,
1.5, 2, 3, 4, 6} and their negatives — so FP4 formats attach a scale to every *small* block.
MXFP4 (OCP Microscaling) shares one power-of-two scale (E8M0: an 8-bit exponent) per 32 values:
4 + 8/32 = 4.25 bits per weight. NVFP4 shares an FP8 E4M3 scale per 16 values *and* multiplies by
one FP32 scale per tensor: 4 + 8/16 = 4.5 bits, finer steps and finer blocks for 0.25 more bits.
What FP4 buys depends on the tensor cores: on Blackwell (sm_100/sm_120) activations are quantized
too and the multiply runs in FP4 (W4A4); everywhere else vLLM runs NVFP4 weights through a
weight-only kernel (Marlin: dequantize to 16-bit, multiply in 16-bit), which saves memory and
decode bandwidth but not FLOPs. Conventions follow compressed-tensors (``nvfp4/helpers.py``,
``mxfp_utils.py``) and vLLM (``nvfp4_emulation_utils.py``); product numbers are (verify).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import numerics as N, stio

E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])   # nibble code = index | sign << 3
E2M1_MAX = 6.0
E4M3_MAX = 448.0
MX_ELEM_EMAX = 2          # floor(log2(6)): compressed-tensors _MX_ELEM_OFFSET[4]


def fp4_round(x) -> np.ndarray:
    """Round to the nearest E2M1 value, ties to even, saturating at +-6 (vLLM's ``cast_to_fp4``:
    0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), E2M1_MAX)
    step = np.where(a < 2, 0.5, np.where(a < 4, 1.0, 2.0))
    r = np.round(a / step) * step                          # half-to-even on the local step
    return np.copysign(np.minimum(r, E2M1_MAX), x)


def fp4_encode(x) -> np.ndarray:
    """Values (already on the E2M1 grid, or rounded here) -> 4-bit codes ``sign << 3 | index``."""
    v = fp4_round(x)
    idx = np.searchsorted(E2M1_GRID, np.abs(v))
    return (idx | (np.signbit(v).astype(np.int64) << 3)).astype(np.uint8)


def fp4_decode(codes) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    return np.where(c & 8, -1.0, 1.0) * E2M1_GRID[c & 7]


def pack_fp4(codes) -> np.ndarray:
    """Two codes per byte along the last axis, the **first value in the low nibble**
    (compressed-tensors ``pack_fp4_to_uint8``; gpt-oss reads ``blk & 0x0F`` first)."""
    c = np.asarray(codes, dtype=np.uint8)
    if c.shape[-1] % 2:
        raise ValueError("FP4 packing needs an even number of values per row")
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_fp4(packed) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2], out[..., 1::2] = p & 0x0F, p >> 4
    return out


# ---------------------------------------------------------------------------------------------
# NVFP4: E4M3 scale per 16 + one FP32 multiplier per tensor
# ---------------------------------------------------------------------------------------------
@dataclass
class NVFP4Tensor:
    values: np.ndarray        # E2M1 values, [rows, cols]
    scales: np.ndarray        # local scales as stored (E4M3-representable), [rows, cols / group]
    global_scale: float       # compressed-tensors' multiplier: 448 * 6 / amax(tensor)
    group: int = 16

    def dequantize(self) -> np.ndarray:
        s = np.repeat(self.scales, self.group, axis=1) / self.global_scale
        return self.values * s

    @property
    def bits_per_weight(self) -> float:
        return 4 + 8 / self.group + 32 / self.values.size


def nvfp4_global_scale(amax: float) -> float:
    """``448 * 6 / amax``: maps the tensor's largest block scale onto E4M3's largest value
    (compressed-tensors ``generate_gparam``; a multiplier, not a divisor — do not invert it)."""
    return E4M3_MAX * E2M1_MAX / max(float(amax), 1e-12)


def nvfp4_quantize(w, group: int = 16, global_scale: float | None = None) -> NVFP4Tensor:
    """Local scale = E4M3(global * block_amax / 6); value = E2M1(w / (local / global))."""
    w = np.asarray(w, dtype=np.float64)
    rows, cols = w.shape
    if cols % group:
        raise ValueError(f"in_features {cols} not divisible by {group} (transformers' NVFP4 path requires it)")
    gs = nvfp4_global_scale(np.abs(w).max()) if global_scale is None else global_scale
    blocks = w.reshape(rows, cols // group, group)
    local = N.minifloat_round(np.clip(gs * np.abs(blocks).max(-1) / E2M1_MAX, -E4M3_MAX, E4M3_MAX), "e4m3")
    local = np.where(local == 0, N.E4M3.min_subnormal, local)
    vals = fp4_round(blocks / (local[..., None] / gs))
    return NVFP4Tensor(vals.reshape(rows, cols), local, gs, group)


def nvfp4_checkpoint_tensors(q: NVFP4Tensor) -> dict:
    """The tensors compressed-tensors' ``nvfp4-pack-quantized`` format stores for one Linear
    (keys are suffixes): packed uint8 ``[out, in/2]``, E4M3 scales ``[out, in/16]``, FP32 global ``[1]``."""
    return {".weight_packed": stio.Tensor("U8", pack_fp4(fp4_encode(q.values))),
            ".weight_scale": stio.Tensor("F8_E4M3", N.fp8_encode(q.scales, "e4m3")),
            ".weight_global_scale": stio.Tensor("F32", np.array([q.global_scale], dtype=np.float32))}


def nvfp4_from_checkpoint(t: dict, group: int = 16) -> NVFP4Tensor:
    t = {k: stio.as_tensor(v) for k, v in t.items()}
    vals = fp4_decode(unpack_fp4(t[".weight_packed"].numpy()))
    return NVFP4Tensor(vals, t[".weight_scale"].numpy().astype(np.float64),
                       float(t[".weight_global_scale"].numpy()[0]), group)


# ---------------------------------------------------------------------------------------------
# MXFP4: a power-of-two (E8M0) scale per 32
# ---------------------------------------------------------------------------------------------
@dataclass
class MXFP4Tensor:
    values: np.ndarray        # E2M1 values
    exponents: np.ndarray     # E8M0 biased exponents (uint8), [rows, cols / 32]
    group: int = 32

    def dequantize(self) -> np.ndarray:
        s = np.ldexp(1.0, np.repeat(self.exponents.astype(np.int64) - 127, self.group, axis=1))
        return self.values * s

    @property
    def bits_per_weight(self) -> float:
        return 4 + 8 / self.group


def mxfp4_scale_exponent(block_amax) -> np.ndarray:
    """E8M0 code for a block, as compressed-tensors computes it: round ``amax`` to a power of two
    ``2^e`` (down, unless its mantissa is >= 1.75, then up: ``round_to_power_2``), then
    ``127 + e - 2`` (2 = floor(log2 6), E2M1's top exponent). The block max lands in [3.5, 7);
    anything above 6 x scale saturates to 6 — MX trades a little clipping for power-of-two scales."""
    a = np.maximum(np.asarray(block_amax, dtype=np.float64), 2.0 ** -126)
    m, e = np.frexp(a)                       # a = m * 2^e, m in [0.5, 1): mantissa 1.x = 2m
    e = (e - 1) + (2 * m >= 1.75)
    return np.clip(127 + e - MX_ELEM_EMAX, 0, 254).astype(np.uint8)


def mxfp4_quantize(w, group: int = 32) -> MXFP4Tensor:
    w = np.asarray(w, dtype=np.float64)
    rows, cols = w.shape
    if cols % group:
        raise ValueError(f"in_features {cols} not divisible by {group}")
    blocks = w.reshape(rows, cols // group, group)
    e = mxfp4_scale_exponent(np.abs(blocks).max(-1))
    vals = fp4_round(blocks / np.ldexp(1.0, e.astype(np.int64) - 127)[..., None])
    return MXFP4Tensor(vals.reshape(rows, cols), e, group)


# ---------------------------------------------------------------------------------------------
# Error and bytes
# ---------------------------------------------------------------------------------------------
def sqnr_db(w, w_hat) -> float:
    w, w_hat = np.asarray(w, float), np.asarray(w_hat, float)
    return float(10 * np.log10(np.sum(w ** 2) / max(np.sum((w - w_hat) ** 2), 1e-300)))


def int4_group_fake_quant(w, group: int = 128) -> np.ndarray:
    """Symmetric INT4 with a scale per ``group`` inputs, compressed-tensors' grid (-8..7, amax/7.5)."""
    w = np.asarray(w, dtype=np.float64)
    b = w.reshape(w.shape[0], -1, group)
    s = np.maximum(np.abs(b).max(-1, keepdims=True), 1e-12) / 7.5
    return (np.clip(np.round(b / s), -8, 7) * s).reshape(w.shape)


BPW = {  # bits per stored weight including block scales (the per-tensor NVFP4 FP32 is negligible)
    "bf16": 16.0, "fp8": 8.0, "fp8-block128": 8 + 32 / (128 * 128), "int8": 8.0,
    "int4-g128": 4 + 16 / 128, "int4-g128-asym": 4 + 16 / 128 + 4 / 128, "int4-g32": 4 + 16 / 32,
    "mxfp4": 4 + 8 / 32, "nvfp4": 4 + 8 / 16,
}


def layout(out_features: int, in_features: int, fmt: str) -> dict:
    """Tensor names, dtypes, shapes and bytes one Linear becomes in a checkpoint of format ``fmt``."""
    o, i = out_features, in_features
    if fmt == "nvfp4":
        t = {"weight_packed": ("U8", (o, i // 2)), "weight_scale": ("F8_E4M3", (o, i // 16)),
             "weight_global_scale": ("F32", (1,))}
    elif fmt == "mxfp4":
        t = {"weight_packed": ("U8", (o, i // 2)), "weight_scale": ("U8", (o, i // 32))}
    elif fmt == "int4-g128":
        t = {"weight_packed": ("I32", (o, i // 8)), "weight_scale": ("BF16", (o, i // 128)), "weight_shape": ("I64", (2,))}
    elif fmt == "fp8":
        t = {"weight": ("F8_E4M3", (o, i)), "weight_scale": ("BF16", (o, 1))}
    elif fmt == "bf16":
        t = {"weight": ("BF16", (o, i))}
    else:
        raise KeyError(f"unknown format {fmt!r}")
    return {k: (dt, shp, int(np.prod(shp)) * stio.BYTES[dt]) for k, (dt, shp) in t.items()}


def layer_bytes(out_features: int, in_features: int, fmt: str) -> int:
    return sum(b for _, _, b in layout(out_features, in_features, fmt).values())


# ---------------------------------------------------------------------------------------------
# The Blackwell throughput model (roofline; peaks from datasheets, verify)
# ---------------------------------------------------------------------------------------------
def gemm_times(M: int, K: int, N: int, gpu="B200", weight_only_eff: float = 1.0) -> dict:
    """Seconds for one ``[M, K] x [K, N]`` GEMM per path: BF16, FP8 W8A8, NVFP4 W4A4 (FP4 tensor cores on
    sm_100+) and NVFP4 weight-only (FP4 bytes, BF16 math at ``weight_only_eff`` of the BF16 GEMM's rate).
    SIMULATED: 100% of peak unless an efficiency is given; B200 2,250 / 4,500 / 9,000 dense TFLOP/s, 8 TB/s (verify)."""
    from . import bench
    from .serve import gpu as _gpu
    g = _gpu(gpu)
    out = {"bf16": bench.gemm_time(M, K, N, g, "bf16"), "fp8": bench.gemm_time(M, K, N, g, "fp8")}
    if "fp4" in g.tflops:
        out["w4a4-nvfp4"] = bench.gemm_time(M, K, N, g, "w4a4-nvfp4")
    w, a, _ = bench.GEMM_PATH["w4a16-nvfp4"]
    out["w4a16-nvfp4"] = max(2.0 * M * K * N / (g.peak("bf16") * weight_only_eff),
                             (K * N * w + M * K * a + M * N * 2) / (g.mem_bw_gbs * 1e9))
    return out


def weight_only_slower_from(K: int, N: int, gpu="B200", weight_only_eff: float = 0.7, max_m: int = 1 << 15):
    """Smallest tokens-per-step ``M`` at which NVFP4 weight-only loses to BF16, or None (never at eff >= 1).
    On a B200 at 70% efficiency: ~210 tokens — every prefill chunk (SIMULATED)."""
    for m in range(1, max_m):
        t = gemm_times(m, K, N, gpu, weight_only_eff)
        if t["w4a16-nvfp4"] > t["bf16"]:
            return m
    return None
