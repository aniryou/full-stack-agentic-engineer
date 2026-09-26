"""Count the work before you time it: FLOPs and bytes for every operation this lab measures.

A benchmark number is only as good as its numerator. "GB/s" means nothing until you say
*which* bytes you counted, and "TFLOP/s" nothing until you say how you counted FLOPs.
This module is the one place the lab counts them; every benchmark takes its cost from
here and the tests pin each formula to a hand-computed value.

The conventions (the same ones the primer uses in §2, "The roofline model"):

* A multiply and an add are **two** FLOPs, so a GEMM ``(m×k)·(k×n)`` costs ``2·m·n·k``.
* GEMM bytes are the **compulsory** traffic: read A and B once, write C once,
  ``(m·k + k·n + m·n)·b``. A real kernel moves at least this much; more if its tiles
  do not fit on chip. This is the byte count the roofline uses.
* STREAM kernels are counted John McCalpin's way: every element of every array the
  kernel names is read or written once, and nothing else. The hidden
  *write-allocate* read that most CPUs add is **not** counted (``write_allocate_bytes``
  tells you how big it is).
* An implementation that needs several passes (numpy has no one-pass ``b + q·c``) is
  charged for every pass it actually makes (``passes_cost``). The lab reports both:
  the bytes the code moved, and the bytes the STREAM convention would count.
* A **transfer** (host to device, GPU to GPU, disk to RAM) counts the bytes delivered,
  once. The link carries them once; counting read + write would double the rate.

``OpCost.basis`` records which of the last two rules applies, so a ``Measurement`` can
never divide the wrong byte count by its time.
"""
from __future__ import annotations

from dataclasses import dataclass

# bytes per element; "tf32" is stored as fp32 (4 bytes) and only *computed* at lower precision
DTYPE_BYTES = {
    "float64": 8, "float32": 4, "tf32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "uint8": 1,
}
_ALIASES = {
    "fp64": "float64", "f64": "float64", "double": "float64",
    "fp32": "float32", "f32": "float32", "float": "float32",
    "fp16": "float16", "f16": "float16", "half": "float16",
    "bf16": "bfloat16",
    "fp8": "float8_e4m3fn", "e4m3": "float8_e4m3fn", "float8": "float8_e4m3fn", "fp8_e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "u8": "uint8", "i8": "int8",
}


def canonical_dtype(name: str) -> str:
    """Normalise a dtype name: ``"bf16"`` -> ``"bfloat16"``, ``"fp8"`` -> ``"float8_e4m3fn"``."""
    key = str(name).lower().replace("torch.", "")
    key = _ALIASES.get(key, key)
    if key not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return key


def dtype_bytes(name: str) -> int:
    """Bytes per element of a dtype (``tf32`` counts as 4: it is stored as fp32)."""
    return DTYPE_BYTES[canonical_dtype(name)]


def _b(dtype_or_bytes) -> int:
    return int(dtype_or_bytes) if isinstance(dtype_or_bytes, (int, float)) else dtype_bytes(dtype_or_bytes)


@dataclass(frozen=True)
class OpCost:
    """What one call of an operation computes and moves.

    ``basis="memory"``: the rate counts ``bytes_read + bytes_written`` (GEMM, STREAM).
    ``basis="transfer"``: the rate counts the bytes *delivered* (``bytes_written``) —
    host to device, GPU to GPU, disk to RAM.
    """
    flops: float
    bytes_read: float
    bytes_written: float
    basis: str = "memory"

    def __post_init__(self) -> None:
        if self.basis not in ("memory", "transfer"):
            raise ValueError("basis must be 'memory' or 'transfer'")

    @property
    def bytes(self) -> float:
        """The byte count a bandwidth figure divides by (see the class docstring)."""
        if self.basis == "transfer":
            return self.bytes_written
        return self.bytes_read + self.bytes_written

    @property
    def intensity(self) -> float:
        """Arithmetic intensity, FLOPs per byte (``inf`` for an op that moves nothing)."""
        return self.flops / self.bytes if self.bytes else float("inf")

    def __add__(self, other: "OpCost") -> "OpCost":
        if self.basis != other.basis:
            raise ValueError("cannot add costs counted on different bases")
        return OpCost(self.flops + other.flops, self.bytes_read + other.bytes_read,
                      self.bytes_written + other.bytes_written, self.basis)

    def times(self, k: float) -> "OpCost":
        return OpCost(self.flops * k, self.bytes_read * k, self.bytes_written * k, self.basis)

    def to_dict(self) -> dict:
        return {"flops": self.flops, "bytes_read": self.bytes_read, "bytes_written": self.bytes_written,
                "basis": self.basis, "bytes": self.bytes}


# -- GEMM ------------------------------------------------------------------------------------
def gemm_cost(m: int, n: int, k: int, dtype_or_bytes, out_bytes=None, accumulate: bool = False) -> OpCost:
    """C(m×n) = A(m×k) · B(k×n): ``2mnk`` FLOPs; compulsory bytes read A, B, write C.

    ``out_bytes`` defaults to the input width (FP8 GEMMs usually write bf16: pass 2).
    ``accumulate=True`` models ``C += A·B``, which must also read C.
    """
    b_in = _b(dtype_or_bytes)
    b_out = b_in if out_bytes is None else _b(out_bytes)
    read = (m * k + k * n) * b_in + (m * n * b_out if accumulate else 0)
    return OpCost(flops=2.0 * m * n * k, bytes_read=float(read), bytes_written=float(m * n * b_out))


def gemm_intensity(m: int, n: int, k: int, dtype_or_bytes) -> float:
    """``2mnk / ((mk + kn + mn)·b)`` — the formula in primer §2."""
    return gemm_cost(m, n, k, dtype_or_bytes).intensity


# -- elementwise and STREAM ------------------------------------------------------------------
def elementwise_cost(n: int, dtype_or_bytes, reads: int, writes: int, flops_per_element: float) -> OpCost:
    """One pass over ``n`` elements that reads ``reads`` arrays and writes ``writes`` arrays."""
    b = _b(dtype_or_bytes)
    return OpCost(flops=float(flops_per_element) * n, bytes_read=float(reads * n * b),
                  bytes_written=float(writes * n * b))


# name: (arrays read, arrays written, FLOPs per element) — McCalpin's STREAM definitions
STREAM = {
    "copy": (1, 1, 0),    # c = a
    "scale": (1, 1, 1),   # b = q·c
    "add": (2, 1, 1),     # c = a + b
    "triad": (2, 1, 2),   # a = b + q·c
}
STREAM_FORMULA = {"copy": "c = a", "scale": "b = q·c", "add": "c = a + b", "triad": "a = b + q·c"}


def stream_cost(kernel: str, n: int, dtype_or_bytes) -> OpCost:
    """STREAM's own byte and FLOP count for one kernel over arrays of ``n`` elements."""
    if kernel not in STREAM:
        raise ValueError(f"unknown STREAM kernel {kernel!r}; one of {list(STREAM)}")
    reads, writes, f = STREAM[kernel]
    return elementwise_cost(n, dtype_or_bytes, reads, writes, f)


def write_allocate_bytes(kernel: str, n: int, dtype_or_bytes) -> float:
    """The read STREAM does not count: on a write-allocate cache, storing to a line that is
    not in cache first *reads* it (read-for-ownership). One extra read per array written,
    unless the code uses non-temporal (streaming) stores."""
    _, writes, _ = STREAM[kernel]
    return float(writes * n * _b(dtype_or_bytes))


def passes_cost(n: int, dtype_or_bytes, passes) -> OpCost:
    """Cost of an implementation that makes several passes; ``passes`` is a list of
    ``(arrays read, arrays written, FLOPs per element)``, one tuple per pass."""
    total = OpCost(0.0, 0.0, 0.0)
    for reads, writes, f in passes:
        total = total + elementwise_cost(n, dtype_or_bytes, reads, writes, f)
    return total


# numpy has no single-pass ``b + q*c``: the lab's numpy triad is ``a = q*c`` then ``a = a + b``
NUMPY_TRIAD_PASSES = [(1, 1, 1), (2, 1, 1)]


def chain_cost(n: int, dtype_or_bytes, k: int, fused: bool) -> OpCost:
    """``k`` in-place elementwise ops (one FLOP each) over an array of ``n`` elements.

    Unfused: every op is its own pass, so the array is read and written ``k`` times.
    Fused (all ``k`` ops applied to a block while it sits in cache): read once, write once.
    FLOPs are the same either way — fusion only removes bytes.
    """
    b = _b(dtype_or_bytes)
    trips = 1 if fused else k
    return OpCost(flops=float(k * n), bytes_read=float(trips * n * b), bytes_written=float(trips * n * b))


# -- transfers ---------------------------------------------------------------------------------
def transfer_cost(nbytes: int, directions: int = 1) -> OpCost:
    """Bytes delivered across a link (or from disk into memory). ``directions=2`` is a
    bidirectional test: two transfers of ``nbytes`` in flight at once, rate = aggregate."""
    total = float(nbytes) * directions
    return OpCost(flops=0.0, bytes_read=total, bytes_written=total, basis="transfer")
