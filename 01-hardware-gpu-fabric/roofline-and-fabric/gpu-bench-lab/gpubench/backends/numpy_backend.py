"""The numpy backend: real measurements of the CPU this runs on (tier T0).

A CPU has the same two roofs as a GPU — a FLOP/s ceiling (cores × clock × vector lanes × 2
for FMA × FMA units) and a bandwidth ceiling (memory channels × transfer rate) — only 10–100×
lower. numpy reaches the first through its BLAS library (OpenBLAS or MKL, multi-threaded)
and the second through ufunc loops, which run on one core per call; the lab adds threads by
splitting arrays into slices (numpy releases the GIL inside large ufunc loops).

What numpy cannot do, and what the lab does about it:

* no single-pass ``b + q·c`` — the triad takes two passes and is charged for both;
* no fast float16 GEMM — there is no BLAS path, so it crawls: a real and instructive number;
* no device — transfers and P2P need the torch backend; T0 measures host ``memcpy`` instead.

Constants are chosen so repeated calls never overflow or decay into denormals (which are
slow on x86): multiplying by 1.0 costs the hardware exactly as much as multiplying by 3.0.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..accounting import (NUMPY_TRIAD_PASSES, canonical_dtype, chain_cost, elementwise_cost, gemm_cost,
                          passes_cost, stream_cost, transfer_cost, write_allocate_bytes)
from ..measure import Op
from ..timing import bench

_POOLS: dict = {}


def _pool(n: int) -> ThreadPoolExecutor:
    if n not in _POOLS:
        _POOLS[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="gpubench")
    return _POOLS[n]


def split(n: int, parts: int) -> list:
    """``parts`` contiguous, nearly equal slices covering ``range(n)``."""
    parts = max(1, min(int(parts), n))
    edges = [n * i // parts for i in range(parts + 1)]
    return [slice(edges[i], edges[i + 1]) for i in range(parts)]


def _parallel(kernel, slices) -> None:
    if len(slices) == 1:
        kernel(slices[0])
    else:
        list(_pool(len(slices)).map(kernel, slices))


def blas_info() -> str:
    """Which BLAS library numpy links (it decides your GEMM roof)."""
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas.get('name', '?')} {blas.get('version', '')}".strip()
    except Exception:  # older numpy, unusual builds
        return "unknown"


class NumpyBackend:
    name = "numpy"
    device = "cpu"
    is_gpu = False
    stream_dtype = "float64"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def describe(self) -> dict:
        from ..inventory import cpu_info
        d = {"backend": "numpy", "numpy": np.__version__, "device": "cpu", "blas": blas_info()}
        d.update(cpu_info())
        d["name"] = d.get("model", "CPU")
        return d

    def gemm_dtypes(self) -> list:
        return ["float64", "float32", "float16"]

    def time(self, fn, *, method: str = "auto", **kw):
        return bench(fn, **kw)      # numpy calls are synchronous: the wall clock is exact

    def sync(self) -> None:
        pass

    # -- compute ----------------------------------------------------------------------------------
    def make_gemm(self, m: int, n: int, k: int, dtype: str = "float64") -> Op:
        dt = np.dtype(canonical_dtype(dtype))
        rng = np.random.default_rng(self.seed)
        a = rng.standard_normal((m, k)).astype(dt)
        b = rng.standard_normal((k, n)).astype(dt)
        c = np.empty((m, n), dtype=dt)

        def fn():
            np.matmul(a, b, out=c)

        def verify() -> bool:       # a benchmark that computes nothing measures nothing
            ref = a.astype(np.float64) @ b.astype(np.float64)
            tol = 8 * (np.finfo(dt).eps / 2) * k * max(1.0, math.sqrt(k))
            return bool(np.allclose(c.astype(np.float64), ref, rtol=0.0, atol=tol))

        note = "numpy has no BLAS path for float16: an emulated loop, slow on purpose" if dt == np.float16 else ""
        return Op(fn, gemm_cost(m, n, k, dt.itemsize), note=note, verify=verify)

    # -- memory -----------------------------------------------------------------------------------
    def make_stream(self, kernel: str, n: int, dtype: str = "float64", threads: int = 1) -> Op:
        """One STREAM kernel over three arrays of ``n`` elements, split across ``threads``."""
        dt = np.dtype(canonical_dtype(dtype))
        a, b, c = np.full(n, 1.0, dt), np.full(n, 2.0, dt), np.full(n, 0.5, dt)   # np.full touches every page
        q = dt.type(3.0)
        if kernel == "copy":
            def k(s): np.copyto(c[s], a[s])
        elif kernel == "scale":
            def k(s): np.multiply(c[s], q, out=b[s])
        elif kernel == "add":
            def k(s): np.add(a[s], b[s], out=c[s])
        elif kernel == "triad":
            def k(s):                           # two passes: a = q·c, then a = a + b
                np.multiply(c[s], q, out=a[s])
                np.add(a[s], b[s], out=a[s])
        else:
            raise ValueError(f"unknown STREAM kernel {kernel!r}")
        slices = split(n, threads)
        cost = passes_cost(n, dt.itemsize, NUMPY_TRIAD_PASSES) if kernel == "triad" else stream_cost(kernel, n, dt.itemsize)
        target, value = {"copy": (c, 1.0), "scale": (b, 1.5), "add": (c, 3.0), "triad": (a, 3.5)}[kernel]
        extras = {"stream_bytes": stream_cost(kernel, n, dt.itemsize).bytes,
                  "write_allocate_bytes": write_allocate_bytes(kernel, n, dt.itemsize),
                  "passes": 2 if kernel == "triad" else 1}
        note = "numpy triad = 2 passes (5 arrays of traffic, STREAM counts 3)" if kernel == "triad" else ""
        return Op(lambda: _parallel(k, slices), cost, note=note,
                  verify=lambda: bool(np.all(target == value)), extras=extras)

    def make_scale_inplace(self, nbytes: int, dtype: str = "float64") -> Op:
        """``x *= 1.0`` over ``nbytes``: one read and one write per element — the cache-ladder probe."""
        dt = np.dtype(canonical_dtype(dtype))
        n = max(1, nbytes // dt.itemsize)
        x = np.full(n, 1.0, dt)
        one = dt.type(1.0)
        return Op(lambda: np.multiply(x, one, out=x), elementwise_cost(n, dt.itemsize, 1, 1, 1),
                  extras={"working_set_bytes": n * dt.itemsize})

    def make_chain(self, n: int, k: int = 8, fused: bool = False, block_elems: int = 32768,
                   dtype: str = "float64") -> Op:
        """``k`` in-place elementwise ops over ``n`` elements: one pass each (unfused), or all
        ``k`` applied block by block while the block sits in cache (fused)."""
        dt = np.dtype(canonical_dtype(dtype))
        x = np.full(n, 1.0, dt)
        steps = [(np.multiply, dt.type(1.0)) if i % 2 == 0 else (np.add, dt.type(0.0)) for i in range(k)]

        def apply(v):
            for f, const in steps:
                f(v, const, out=v)

        if fused:
            views = [x[i:i + block_elems] for i in range(0, n, block_elems)]

            def fn():
                for v in views:
                    apply(v)
        else:
            def fn():
                apply(x)
        extras = {"k": k, "fused": fused, "block_bytes": block_elems * dt.itemsize if fused else None,
                  "in_cache_bytes": 2 * (k - 1) * n * dt.itemsize if fused else 0}
        return Op(fn, chain_cost(n, dt.itemsize, k, fused), extras=extras,
                  verify=lambda: bool(np.all(x == 1.0)))

    def make_memcpy(self, nbytes: int) -> Op:
        """Host-to-host copy of ``nbytes``: the T0 stand-in for a host-to-device transfer."""
        src = np.ones(nbytes, dtype=np.uint8)
        dst = np.zeros(nbytes, dtype=np.uint8)
        return Op(lambda: np.copyto(dst, src), transfer_cost(nbytes),
                  note="host memcpy (RAM to RAM), not PCIe", extras={"stream_bytes": 2 * nbytes},
                  verify=lambda: bool(dst[0] == 1 and dst[-1] == 1))
