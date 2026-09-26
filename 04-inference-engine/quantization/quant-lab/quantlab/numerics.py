"""numerics.py — the bytes on disk: bf16, FP8 (E4M3/E5M2) and packed INT4, bit for bit.

One idea: a quantized checkpoint is only a set of bit patterns plus a rule for reading them, and
every rule here is small enough to write in numpy. ``bf16`` keeps float32's 8 exponent bits and
drops 16 mantissa bits; FP8 E4M3 (1-4-3, bias 7, largest finite 448, no infinities) trades range
for precision against E5M2 (1-5-2, bias 15, largest finite 57,344); INT4 codes are stored eight to
an int32 word in the order compressed-tensors uses (``pack_to_int32``: add 8 so -8..7 becomes
0..15, element 0 in the lowest four bits). Everything is round-to-nearest-even, like the hardware
casts, and saturating (quantizers clamp to the largest finite value before they cast). The tests
check the encoders against ``torch.float8_e4m3fn`` / ``float8_e5m2`` / ``bfloat16`` when torch is
installed; this module itself needs only numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MiniFloat:
    """An IEEE-like small float: ``exp_bits`` exponent bits with ``bias``, ``man_bits`` mantissa bits."""
    name: str
    exp_bits: int
    man_bits: int
    bias: int
    max_finite: float
    has_inf: bool

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias - self.man_bits)


E4M3 = MiniFloat("float8_e4m3fn", 4, 3, 7, 448.0, has_inf=False)       # S.1111.111 is NaN; no inf
E5M2 = MiniFloat("float8_e5m2", 5, 2, 15, 57344.0, has_inf=True)       # IEEE-style: inf and NaNs
FORMATS = {"e4m3": E4M3, "fp8_e4m3": E4M3, "float8_e4m3fn": E4M3, "fp8": E4M3,
           "e5m2": E5M2, "fp8_e5m2": E5M2, "float8_e5m2": E5M2}


def _fmt(fmt) -> MiniFloat:
    return fmt if isinstance(fmt, MiniFloat) else FORMATS[str(fmt).lower()]


def minifloat_round(x, fmt="e4m3") -> np.ndarray:
    """Round to the nearest representable value (ties to even), saturating at +-max_finite."""
    f = _fmt(fmt)
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), f.max_finite)
    _, e = np.frexp(np.maximum(a, f.min_normal))          # a = m * 2**e with m in [0.5, 1)
    step = np.ldexp(1.0, e - 1 - f.man_bits)               # spacing inside this binade (or the subnormals)
    r = np.minimum(np.round(a / step) * step, f.max_finite)   # np.round is round-half-to-even
    return np.where(np.isnan(x), np.nan, np.copysign(r, x))


def fp8_encode(x, fmt="e4m3") -> np.ndarray:
    """float -> uint8 bit patterns (sign | exponent | mantissa), rounded and saturated."""
    f = _fmt(fmt)
    v = minifloat_round(x, f)
    sign = (np.signbit(v)).astype(np.uint8) << 7
    a = np.abs(v)
    m, e = np.frexp(np.where(a > 0, a, 1.0))
    normal = a >= f.min_normal
    exp_field = np.where(normal, e - 1 + f.bias, 0)
    man = np.where(normal, (m * 2 - 1) * 2 ** f.man_bits, a / f.min_subnormal)
    code = (exp_field.astype(np.int64) << f.man_bits) | np.round(man).astype(np.int64)
    code = np.where(a > 0, code, 0).astype(np.uint8)
    return code | sign


def fp8_decode(codes, fmt="e4m3") -> np.ndarray:
    """uint8 bit patterns -> float32. E4M3's S.1111.111 and E5M2's all-ones exponent decode to NaN/inf."""
    f = _fmt(fmt)
    c = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(c & 0x80, -1.0, 1.0)
    exp_field = (c >> f.man_bits) & ((1 << f.exp_bits) - 1)
    man = c & ((1 << f.man_bits) - 1)
    val = np.where(exp_field == 0, man * f.min_subnormal,
                   (1 + man / 2 ** f.man_bits) * np.ldexp(1.0, exp_field - f.bias))
    top = (1 << f.exp_bits) - 1
    if f.has_inf:
        val = np.where(exp_field == top, np.where(man == 0, np.inf, np.nan), val)
    else:
        val = np.where((exp_field == top) & (man == (1 << f.man_bits) - 1), np.nan, val)
    return (sign * val).astype(np.float32)


def representable(fmt="e4m3") -> np.ndarray:
    """Every finite non-negative value of the format, sorted (E4M3: 127 values from 0 to 448)."""
    vals = fp8_decode(np.arange(128, dtype=np.uint8), fmt)
    return np.unique(vals[np.isfinite(vals)])


# ---------------------------------------------------------------------------------------------
# bfloat16: the top half of a float32
# ---------------------------------------------------------------------------------------------
def bf16_encode(x) -> np.ndarray:
    """float -> uint16 bfloat16 bit patterns, round to nearest even (NaN kept quiet)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    return np.where(nan, np.uint16(0x7FC0), rounded).astype(np.uint16)


def bf16_decode(u16) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(x) -> np.ndarray:
    """What storing ``x`` as bfloat16 does to it (8 bits of precision, float32's range)."""
    return bf16_decode(bf16_encode(x))


def fp16_round(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Packed integers, compressed-tensors order
# ---------------------------------------------------------------------------------------------
def pack_int32(q, num_bits: int = 4) -> np.ndarray:
    """Pack signed ``num_bits`` codes (range -2^(b-1) .. 2^(b-1)-1) along the last axis into int32.

    compressed-tensors ``pack_to_int32``: codes are offset by ``1 << (b-1)`` (INT4: -8..7 -> 0..15)
    and element ``i`` of each run of ``32 // b`` lands at bit ``b * i`` — the first element in the
    lowest bits. ``[-8, -7, 0, 1, 2, 3, 4, 7]`` -> ``0xfcba9810``. A ``[4096, 896]`` INT4 weight
    packs to ``[4096, 112]``. Only widths that divide 32 (1, 2, 4, 8) are supported here."""
    if 32 % num_bits:
        raise ValueError("this packer supports num_bits in (1, 2, 4, 8)")
    q = np.asarray(q, dtype=np.int64)
    lo, hi = -(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1
    if q.min(initial=0) < lo or q.max(initial=0) > hi:
        raise ValueError(f"codes outside [{lo}, {hi}]")
    per = 32 // num_bits
    rows, cols = q.reshape(-1, q.shape[-1]).shape
    pad = (-cols) % per
    u = np.pad(q.reshape(rows, cols) + (1 << (num_bits - 1)), ((0, 0), (0, pad))).astype(np.uint64)
    u = u.reshape(rows, -1, per)
    shifts = (np.arange(per, dtype=np.uint64) * num_bits)
    words = (u << shifts).sum(axis=-1).astype(np.uint32)
    return words.view(np.int32).reshape(*q.shape[:-1], -1)


def unpack_int32(packed, num_bits: int, cols: int) -> np.ndarray:
    """Inverse of :func:`pack_int32`: int32 words -> signed codes, ``cols`` per row."""
    p = np.asarray(packed, dtype=np.int32)
    per = 32 // num_bits
    words = p.view(np.uint32).astype(np.uint64).reshape(-1, p.shape[-1], 1)
    shifts = np.arange(per, dtype=np.uint64) * num_bits
    u = (words >> shifts) & ((1 << num_bits) - 1)
    q = u.reshape(words.shape[0], -1)[:, :cols].astype(np.int64) - (1 << (num_bits - 1))
    return q.reshape(*p.shape[:-1], cols).astype(np.int8)
