"""formats.py - a number format is a grid of representable values; quantizing is rounding onto it.

The one idea: every low-precision format is a finite grid plus a scale that stretches it over the
data. Integer grids are evenly spaced, so the absolute error is the same everywhere (at most half a
step) and each extra bit halves it: ~6 dB of signal-to-noise per bit. Floating grids (FP8 E4M3/E5M2,
FP4 E2M1) are spaced by powers of two, so the *relative* error is the same everywhere and exponent
bits buy range instead of precision. Block formats (MXFP4, NVFP4) give every 32 or 16 values their
own small scale so a 4-bit float grid can follow the data's local magnitude. A format's cost is its
bits per weight: element bits plus scale bits amortised over the group (`bits_per_weight`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FloatFormat:
    """sign + `exp_bits` exponent (with `bias`) + `man_bits` mantissa; saturates at `max_value`."""
    name: str
    exp_bits: int
    man_bits: int
    bias: int
    max_value: float

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias - self.man_bits)

    def grid(self) -> np.ndarray:
        """Every non-negative finite value, from the bit patterns (codes above max_value are NaN/inf)."""
        vals = set()
        for e in range(2 ** self.exp_bits):
            for m in range(2 ** self.man_bits):
                frac = m / 2 ** self.man_bits
                v = frac * 2.0 ** (1 - self.bias) if e == 0 else (1 + frac) * 2.0 ** (e - self.bias)
                if v <= self.max_value:
                    vals.add(v)
        return np.array(sorted(vals))


# E4M3 "fn": no infinities, S.1111.111 is NaN, so the largest finite value is 1.75 x 2^8 = 448.
E4M3 = FloatFormat("fp8_e4m3", 4, 3, 7, 448.0)
E5M2 = FloatFormat("fp8_e5m2", 5, 2, 15, 57344.0)      # IEEE-like: exponent 31 is inf/NaN
E2M1 = FloatFormat("fp4_e2m1", 2, 1, 1, 6.0)           # grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}
FLOATS = {"fp8": E4M3, "fp8_e4m3": E4M3, "fp8_e5m2": E5M2, "fp4": E2M1, "fp4_e2m1": E2M1}


def to_float(x, fmt: FloatFormat = E4M3) -> np.ndarray:
    """Round to the nearest value of `fmt` (ties to even mantissa), saturating at +-max_value.
    Within one binade [2^e, 2^(e+1)) the step is 2^(e - man_bits); below min_normal it stays fixed."""
    x = np.clip(np.asarray(x, float), -fmt.max_value, fmt.max_value)
    e = np.floor(np.log2(np.maximum(np.abs(x), fmt.min_normal)))
    step = 2.0 ** (e - fmt.man_bits)
    return np.clip(np.round(x / step) * step, -fmt.max_value, fmt.max_value)


# ------------------------------------------------------------------------------------------------
# Integers: symmetric (zero maps to code 0) or asymmetric (a zero point shifts an unsigned grid)
# ------------------------------------------------------------------------------------------------
def int_range(bits: int, symmetric: bool = True, convention: str = "restricted") -> tuple[int, int]:
    """Code range. 'restricted' symmetric: +-(2^(b-1) - 1), e.g. INT4 -7..7 (minengine, SmoothQuant).
    'full' symmetric: -2^(b-1)..2^(b-1) - 1, e.g. INT4 -8..7 (GPTQ, compressed-tensors checkpoints).
    Asymmetric: unsigned 0..2^b - 1 with a zero point (AWQ, KIVI)."""
    if not symmetric:
        return 0, 2 ** bits - 1
    if convention == "restricted":
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    if convention == "full":
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    raise ValueError(f"convention must be 'restricted' or 'full', not {convention!r}")


def int_scale(amax, bits: int, convention: str = "restricted"):
    """Symmetric scale. restricted: amax / (2^(b-1) - 1) (INT8 /127, INT4 /7). full: amax / ((2^b - 1)/2)
    (INT8 /127.5, INT4 /7.5 - compressed-tensors' `_calculate_range`, GPTQ's (xmax - xmin)/maxq)."""
    denom = 2 ** (bits - 1) - 1 if convention == "restricted" else (2 ** bits - 1) / 2
    return np.maximum(np.asarray(amax, float), 1e-12) / denom


def asym_params(xmin, xmax, bits: int):
    """Asymmetric min-max: scale = (max - min) / (2^b - 1), zero = round(-min / scale). The range is
    widened to include 0 so that 0 is exactly representable (padding, ReLU outputs)."""
    xmin, xmax = np.minimum(np.asarray(xmin, float), 0), np.maximum(np.asarray(xmax, float), 0)
    scale = np.maximum(xmax - xmin, 1e-12) / (2 ** bits - 1)
    return scale, np.round(-xmin / scale)


def quantize_int(x, scale, bits: int, zero=None, convention: str = "restricted") -> np.ndarray:
    """Codes: round(x / scale) (+ zero), clipped to the code range. `zero is None` means symmetric."""
    lo, hi = int_range(bits, zero is None, convention)
    q = np.round(np.asarray(x, float) / scale) + (0 if zero is None else zero)
    return np.clip(q, lo, hi)


def dequantize_int(codes, scale, zero=None) -> np.ndarray:
    return (np.asarray(codes, float) - (0 if zero is None else zero)) * scale


# ------------------------------------------------------------------------------------------------
# Block formats: a 4-bit float element plus a scale shared by a short block
# ------------------------------------------------------------------------------------------------
def _blocks(x, block: int):
    x = np.asarray(x, float)
    if x.shape[-1] % block:
        raise ValueError(f"last dimension {x.shape[-1]} is not a multiple of the block size {block}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // block, block)


def mxfp4(x, block: int = 32):
    """OCP MXFP4: E2M1 elements, one E8M0 scale (a power of two) per 32 along the last axis.
    Shared exponent = floor(log2(block amax)) - 2 (2 = floor(log2(6)), E2M1's largest exponent), so the
    block max lands in [4, 8) and anything above 6 saturates. Returns (elements, scale_code, x_hat);
    scale_code is the E8M0 byte (exponent + 127) a checkpoint stores."""
    xb = _blocks(x, block)
    amax = np.abs(xb).max(-1, keepdims=True)
    exp = np.clip(np.floor(np.log2(np.maximum(amax, 2.0 ** -127))) - 2, -127, 127)
    elems = to_float(xb / 2.0 ** exp, E2M1)
    return elems, (exp + 127).astype(np.uint8), (elems * 2.0 ** exp).reshape(np.shape(x))


def nvfp4(x, block: int = 16):
    """NVFP4: E2M1 elements, one FP8-E4M3 scale per 16, one FP32 scale per tensor. The tensor scale is a
    multiplier g = 448 x 6 / amax(tensor) (compressed-tensors `generate_gparam`); each block's scale is
    E4M3(g x block_amax / 6), so block scales use E4M3's whole range. x_hat = element x block_scale / g.
    Returns (elements, block_scales, g, x_hat)."""
    xb = _blocks(x, block)
    g = E4M3.max_value * E2M1.max_value / max(float(np.abs(xb).max()), 1e-12)
    local = to_float(g * np.abs(xb).max(-1, keepdims=True) / E2M1.max_value, E4M3)
    safe = np.where(local == 0, 1.0, local)
    elems = to_float(xb * g / safe, E2M1)
    return elems, local, g, (elems * local / g).reshape(np.shape(x))


# ------------------------------------------------------------------------------------------------
# What it costs, and what it gets you
# ------------------------------------------------------------------------------------------------
def bits_per_weight(bits: float, group_size: int | None = None, scale_bits: float = 16,
                    zero_point_bits: float = 0, tensor_scale_bits: float = 0, numel: int | None = None) -> float:
    """Storage per weight including the scales. INT4 g128 with a 16-bit scale: 4.125; plus a 4-bit zero
    point (AWQ, servelab.sizing): 4.15625; MXFP4 (8-bit scale per 32): 4.25; NVFP4 (8-bit per 16): 4.5
    (+ one 32-bit tensor scale over `numel` weights); FP8 with fp32 scales per 128x128 block: 8.002."""
    per_group = (scale_bits + zero_point_bits) / group_size if group_size else 0.0
    return bits + per_group + (tensor_scale_bits / numel if numel else 0.0)


def sqnr_rule_db(bits: int, crest: float) -> float:
    """The uniform-quantizer model: noise power step^2 / 12 with step = 2 amax / 2^b gives
    SQNR = 6.02 b + 4.77 - 20 log10(amax / rms). A full-scale sine (crest sqrt 2) gives the textbook
    6.02 b + 1.76; every 10x of amax over rms (an outlier) costs 20 dB, i.e. more than 3 bits."""
    return 20 * np.log10(2) * bits + 10 * np.log10(3) - 20 * np.log10(crest)


# ------------------------------------------------------------------------------------------------
# Packing: how the codes are laid out in a checkpoint
# ------------------------------------------------------------------------------------------------
def pack_int4(codes) -> np.ndarray:
    """compressed-tensors `pack_to_int32`: signed codes -8..7 offset by +8, eight per int32 along the last
    axis, element 0 in the lowest 4 bits. [-8,-7,0,1,2,3,4,7] -> 0xfcba9810."""
    c = (np.asarray(codes, np.int64) + 8).astype(np.uint32)
    c = c.reshape(*c.shape[:-1], c.shape[-1] // 8, 8)
    packed = np.zeros(c.shape[:-1], np.uint32)
    for i in range(8):
        packed |= c[..., i] << np.uint32(4 * i)
    return packed.view(np.int32)


def unpack_int4(packed) -> np.ndarray:
    p = np.asarray(packed, np.int32).view(np.uint32)
    c = np.stack([(p >> np.uint32(4 * i)) & np.uint32(0xF) for i in range(8)], -1)
    return c.reshape(*p.shape[:-1], p.shape[-1] * 8).astype(np.int64) - 8


def fp4_nibbles(values) -> np.ndarray:
    """E2M1 value -> 4-bit code: index in the grid {0, .5, 1, 1.5, 2, 3, 4, 6} | sign << 3."""
    v = np.asarray(values, float)
    idx = np.searchsorted(E2M1.grid(), np.abs(v))
    return (idx | (np.signbit(v) << 3)).astype(np.uint8)


def pack_fp4(values) -> np.ndarray:
    """Two E2M1 values per byte along the last axis, the first in the low nibble (gpt-oss, compressed-tensors)."""
    n = fp4_nibbles(values)
    return (n[..., 0::2] | (n[..., 1::2] << 4)).astype(np.uint8)


def unpack_fp4(packed) -> np.ndarray:
    p = np.asarray(packed, np.uint8)
    nib = np.stack([p & 0xF, p >> 4], -1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    return np.where(nib & 8, -1.0, 1.0) * E2M1.grid()[nib & 7]
