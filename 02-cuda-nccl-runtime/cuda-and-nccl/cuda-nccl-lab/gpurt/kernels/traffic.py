"""The bytes and FLOPs each kernel *should* move — the model that traces and timings are checked against.

The one idea: for a memory-bound kernel, **time ~ bytes / bandwidth**, so predicting its speed starts
with counting the bytes the algorithm must move. After a run, *effective bandwidth* = those bytes /
measured time; dividing by the spec-sheet peak tells you how close to the memory roofline you are.
A copy kernel at 80-90 % of peak is doing well; a naive transpose at 20 % is leaving most of the
machine idle (primer §3; the roofline itself is layer 01).

Everything here is arithmetic — no GPU needed. The spec table is dated and marked (verify): check a
datasheet before quoting a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    mem_gbps: float  # peak DRAM bandwidth, GB/s (1e9 bytes/s)
    fp32_tflops: float  # peak non-tensor FP32, TFLOP/s
    cc: tuple[int, int]  # compute capability
    memory_gb: int


# Datasheet peaks, Sep 2026 snapshot (verify each against the vendor datasheet before quoting).
GPUS: dict[str, GpuSpec] = {g.name: g for g in (
    GpuSpec("T4", 320, 8.1, (7, 5), 16),
    GpuSpec("P100", 732, 9.3, (6, 0), 16),
    GpuSpec("L4", 300, 30.3, (8, 9), 24),
    GpuSpec("A10", 600, 31.2, (8, 6), 24),
    GpuSpec("RTX 4090", 1008, 82.6, (8, 9), 24),
    GpuSpec("A100 40GB SXM", 1555, 19.5, (8, 0), 40),
    GpuSpec("A100 80GB SXM", 2039, 19.5, (8, 0), 80),
    GpuSpec("H100 SXM", 3350, 67.0, (9, 0), 80),
    GpuSpec("H200 SXM", 4800, 67.0, (9, 0), 141),
)}

# (reads, writes) of the full vector per element
ELEMENTWISE = {"copy": (1, 1), "vec_add": (2, 1), "saxpy": (2, 1)}


def elementwise_bytes(kind: str, n: int, itemsize: int = 4) -> int:
    reads, writes = ELEMENTWISE[kind]
    return (reads + writes) * n * itemsize


def transpose_bytes(rows: int, cols: int, itemsize: int = 4) -> int:
    """Read every element once, write it once — the same bytes as a copy."""
    return 2 * rows * cols * itemsize


def reduction_bytes(n: int, itemsize: int = 4) -> int:
    """Read every element once; the partial sums are negligible."""
    return n * itemsize


def matmul_flops(M: int, N: int, K: int) -> int:
    """One multiply and one add per (i, j, k)."""
    return 2 * M * N * K


def matmul_global_loads(M: int, N: int, K: int, tile: int = 1) -> int:
    """Element loads the kernel issues to global memory (before caches), for M, N multiples of the
    tile: every one of the M*N threads loads ceil(K / tile) elements of A and as many of B.
    ``tile=1`` is the naive kernel (2*M*N*K)."""
    return 2 * M * N * math.ceil(K / tile)


def matmul_min_bytes(M: int, N: int, K: int, itemsize: int = 4) -> int:
    """Compulsory traffic: read A and B once, write C once (a perfect cache)."""
    return (M * K + K * N + M * N) * itemsize


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """FLOPs per byte moved. Compare with the machine balance (peak FLOP/s / peak B/s)."""
    return flops / nbytes


def machine_balance(gpu: GpuSpec) -> float:
    """FP32 FLOPs the GPU can do per byte of DRAM traffic — the roofline's ridge point."""
    return gpu.fp32_tflops * 1e12 / (gpu.mem_gbps * 1e9)


# Full-matrix element accesses per element of x, by array: {array: (reads, writes)}
SOFTMAX_ACCESSES = {
    "unfused": {"x": (2, 0), "e": (2, 1), "out": (0, 1)},
    "fused": {"x": (2, 0), "out": (0, 1)},
}


def softmax_bytes(rows: int, cols: int, variant: str = "fused", itemsize: int = 4) -> int:
    """Bytes of full-matrix traffic (the O(rows) row statistics are ignored): unfused moves the
    matrix 6 times (24 B/element in float32), fused 3 times (12 B/element)."""
    per_elem = sum(r + w for r, w in SOFTMAX_ACCESSES[variant].values())
    return per_elem * rows * cols * itemsize


def effective_gbps(nbytes: float, seconds: float) -> float:
    """Achieved bandwidth in GB/s (1e9), from the bytes the algorithm must move."""
    return nbytes / seconds / 1e9


def predicted_seconds(nbytes: float, gpu: GpuSpec, efficiency: float = 0.8) -> float:
    """A *model prediction*, not a measurement: bytes / (peak x efficiency). ``efficiency`` is an
    assumption (well-written streaming kernels typically reach ~0.8-0.9 of peak — measure yours)."""
    return nbytes / (gpu.mem_gbps * 1e9 * efficiency)
