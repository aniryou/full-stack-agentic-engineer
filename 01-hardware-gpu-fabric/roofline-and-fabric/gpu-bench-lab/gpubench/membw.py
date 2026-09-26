"""Memory bandwidth: STREAM's kernels, Little's law, fusion, and the cache ladder.

Four experiments, all real measurements on whatever machine runs them:

1. **STREAM** (``stream_suite``): copy, scale, add, triad over arrays at least 4× the last-level
   cache, best of N — John McCalpin's rules, so the number is the *memory's* bandwidth and not
   the cache's. Bytes are counted by accounting.py; the numpy triad is charged for both passes.
2. **Threads** (``thread_scaling``): one core cannot fill a memory bus. Little's law says
   bandwidth = bytes in flight ÷ latency, and one core only has ~10–20 cache-line misses in
   flight; more cores, more misses in flight, more bandwidth — until the DRAM channels saturate.
   A GPU is this idea taken to the limit (tens of thousands of threads in flight).
3. **Fusion** (``fusion``): ``k`` elementwise ops as ``k`` passes over DRAM, versus the same ops
   applied block by block while each block sits in cache. Same FLOPs, ``k``× fewer DRAM bytes.
4. **The cache ladder** (``cache_ladder``): bandwidth versus working-set size. Each plateau is a
   level of the hierarchy (L1/L2/L3/DRAM on a CPU; L2/HBM on a GPU); tiny sizes are dominated by
   per-call overhead instead — the α of the α-β model.
"""
from __future__ import annotations

import os

from .accounting import STREAM, canonical_dtype, dtype_bytes
from .measure import Measurement, measure

KERNELS = tuple(STREAM)


def stream_elems(be, llc_bytes: int | None = None, factor: int = 4, min_elems: int = 1 << 22,
                 max_bytes_per_array: int | None = None, dtype: str | None = None) -> int:
    """Elements per array. CPU: each array ≥ ``factor`` × last-level cache (STREAM's rule).
    GPU: ≥ 256 MiB and ≥ 8× L2, capped at ~5% of device memory per array."""
    b = dtype_bytes(dtype or be.stream_dtype)
    if be.is_gpu:
        d = be.describe()
        l2 = d.get("l2_cache_bytes") or (50 << 20)
        want = max(256 << 20, 8 * l2)
        cap = max_bytes_per_array or max(64 << 20, int(0.05 * d.get("memory_bytes", 16 << 30)))
        return max(1, min(want, cap) // b)
    if llc_bytes is None:
        from .inventory import cache_sizes
        llc_bytes = max([v for v in cache_sizes().values() if v] or [32 << 20])
    n = max(min_elems, factor * llc_bytes // b)
    if max_bytes_per_array:
        n = min(n, max_bytes_per_array // b)
    return int(n)


def run_stream(be, kernel: str, n: int, dtype: str | None = None, threads: int = 1, *,
               repeats: int = 5, min_time: float = 0.05) -> Measurement:
    dtype = canonical_dtype(dtype or be.stream_dtype)
    op = be.make_stream(kernel, n, dtype, threads=threads)
    return measure(be, op, f"stream.{kernel}", {"n": n, "dtype": dtype, "threads": threads},
                   repeats=repeats, min_time=min_time)


def stream_suite(be, n: int | None = None, threads: int = 1, kernels=KERNELS, dtype: str | None = None,
                 repeats: int = 5, min_time: float = 0.05) -> list:
    n = n or stream_elems(be, dtype=dtype)
    return [run_stream(be, k, n, dtype, threads, repeats=repeats, min_time=min_time) for k in kernels]


def thread_scaling(be, kernel: str = "add", n: int | None = None, threads=None, repeats: int = 5,
                   min_time: float = 0.05) -> list:
    """The same kernel with 1, 2, 4, ... threads (CPU only: a GPU kernel is already parallel)."""
    if be.is_gpu:
        raise ValueError("thread scaling is a CPU experiment; on a GPU one kernel already uses every SM")
    cpus = os.cpu_count() or 1
    threads = threads or sorted({1, 2, 4, 8, 16, 32, 64, cpus} & set(range(1, cpus + 1)))
    n = n or stream_elems(be)
    return [run_stream(be, kernel, n, None, t, repeats=repeats, min_time=min_time) for t in threads]


def peak(measurements, stat: str = "best") -> Measurement:
    """The measurement that moved bytes fastest — the slanted roof of the roofline."""
    return max(measurements, key=lambda m: m.bytes_per_s(stat))


def fusion(be, n: int | None = None, k: int = 8, block_elems: int = 1 << 15, repeats: int = 5,
           min_time: float = 0.05) -> tuple:
    """Unfused (``k`` DRAM passes) vs fused (block-wise, one DRAM pass): returns both Measurements."""
    if not hasattr(be, "make_chain"):
        raise ValueError(f"the {be.name} backend has no fusion experiment (it is a CPU/numpy demo)")
    n = n or stream_elems(be)
    out = []
    for fused in (False, True):
        op = be.make_chain(n, k=k, fused=fused, block_elems=block_elems)
        out.append(measure(be, op, "chain.fused" if fused else "chain.unfused",
                           {"n": n, "k": k, "block_elems": block_elems if fused else None},
                           repeats=repeats, min_time=min_time))
    return tuple(out)


def ladder_sizes(min_bytes: int = 16 << 10, max_bytes: int = 256 << 20, factor: int = 2) -> list:
    sizes, s = [], min_bytes
    while s <= max_bytes:
        sizes.append(s)
        s *= factor
    return sizes


def cache_ladder(be, sizes=None, repeats: int = 3, min_time: float = 0.01) -> list:
    """In-place ``x *= 1`` (one read + one write per element) over growing working sets."""
    sizes = sizes or (ladder_sizes(64 << 10, 1 << 30, 2) if be.is_gpu else ladder_sizes())
    out = []
    for s in sizes:
        op = be.make_scale_inplace(s)
        out.append(measure(be, op, "ladder", {"working_set": op.extras["working_set_bytes"]},
                           repeats=repeats, min_time=min_time))
    return out


def bytes_in_flight(bandwidth: float, latency: float) -> float:
    """Little's law for memory: to sustain ``bandwidth`` (B/s) at ``latency`` (s) you must keep
    ``bandwidth × latency`` bytes of requests outstanding."""
    return bandwidth * latency
