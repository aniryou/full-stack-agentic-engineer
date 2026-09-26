"""Row softmax: why fusion wins for memory-bound work.

The one idea: when a computation is memory-bound, **time follows bytes**, and every kernel boundary
is a round trip through DRAM. Softmax written as separate ops — the way an eager framework runs
``exp(x - x.max(1)) / sum(...)`` — is four kernels:

    row_max   read x                      -> m (one value per row)
    sub_exp   read x, read m, write e     -> e = exp(x - m)
    row_sum   read e                      -> s
    divide    read e, read s, write out

That is 4 full reads + 2 full writes of the matrix (24 bytes per float32 element). The **fused**
kernel keeps each row inside one thread block and uses the *online softmax* recurrence
(Milakov & Gimelshein, 2018) to get the max and the normaliser in one pass:

    m' = max(m, x);   d' = d * exp(m - m') + exp(x - m')

then a second pass writes ``exp(x - m) / d``: 2 reads + 1 write (12 bytes per element), one launch.
The same recurrence, applied tile by tile to attention scores, is the heart of FlashAttention
(see ``04-inference-engine/flash-attention``).

Per-block partial ``(m, d)`` pairs are merged with a shared-memory tree, exactly like
``reduction.py`` but combining pairs:  ``m = max(m1, m2);  d = d1*e^(m1-m) + d2*e^(m2-m)``.

How this maps to primer §3.5 (``gpusim.tiling.softmax_traffic``): the primer's unfused chain is five
kernels — max, subtract, exp, sum, divide: 8RC (+4R) element accesses. This unfused path already merges
subtract and exp into one kernel (``sub_exp``), so it is four kernels and 6RC. The fused kernel here is the
primer's *online* row (3RC: two reads and a write), not its 2RC variant that holds a whole row on chip and
reads it once — that one is ``triton_kernels.softmax`` (the Triton tutorial's kernel).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from numba import cuda, float32

from . import blocks_for

NEG_INF = np.float32(-np.inf)


def _check_pow2(threads: int) -> None:
    if threads < 2 or threads & (threads - 1):
        raise ValueError("threads must be a power of two >= 2")


@lru_cache(maxsize=None)
def make_softmax_fused(threads: int = 128):
    """One block per row; ``threads`` (power of two) is a compile-time constant."""
    _check_pow2(threads)

    @cuda.jit
    def softmax_fused(x, out):
        bm = cuda.shared.array(threads, float32)
        bd = cuda.shared.array(threads, float32)
        row = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        n = x.shape[1]
        m = NEG_INF
        d = float32(0.0)
        for j in range(tid, n, threads):  # pass 1: running max and normaliser (1 read)
            v = x[row, j]
            m_new = max(m, v)
            d = d * math.exp(m - m_new) + math.exp(v - m_new)
            m = m_new
        bm[tid] = m
        bd[tid] = d
        cuda.syncthreads()
        stride = threads // 2
        while stride > 0:  # merge (m, d) pairs across the block
            if tid < stride:
                m1 = bm[tid]
                m2 = bm[tid + stride]
                mm = max(m1, m2)
                if mm > NEG_INF:  # both empty (-inf, 0): nothing to merge, avoid exp(nan)
                    bd[tid] = bd[tid] * math.exp(m1 - mm) + bd[tid + stride] * math.exp(m2 - mm)
                    bm[tid] = mm
            cuda.syncthreads()
            stride //= 2
        m = bm[0]
        d = bd[0]
        for j in range(tid, n, threads):  # pass 2: normalise (1 read + 1 write)
            out[row, j] = math.exp(x[row, j] - m) / d

    return softmax_fused


@lru_cache(maxsize=None)
def make_row_reductions(threads: int = 128):
    """``(row_max, row_sum)``: one block per row, shared-memory tree — the unfused building blocks."""
    _check_pow2(threads)

    @cuda.jit
    def row_max(x, m):
        buf = cuda.shared.array(threads, float32)
        row = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        v = NEG_INF
        for j in range(tid, x.shape[1], threads):
            v = max(v, x[row, j])
        buf[tid] = v
        cuda.syncthreads()
        stride = threads // 2
        while stride > 0:
            if tid < stride:
                buf[tid] = max(buf[tid], buf[tid + stride])
            cuda.syncthreads()
            stride //= 2
        if tid == 0:
            m[row] = buf[0]

    @cuda.jit
    def row_sum(e, s):
        buf = cuda.shared.array(threads, float32)
        row = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        v = float32(0.0)
        for j in range(tid, e.shape[1], threads):
            v += e[row, j]
        buf[tid] = v
        cuda.syncthreads()
        stride = threads // 2
        while stride > 0:
            if tid < stride:
                buf[tid] += buf[tid + stride]
            cuda.syncthreads()
            stride //= 2
        if tid == 0:
            s[row] = buf[0]

    return row_max, row_sum


@cuda.jit
def sub_exp(x, m, e):
    col, row = cuda.grid(2)
    if row < x.shape[0] and col < x.shape[1]:
        e[row, col] = math.exp(x[row, col] - m[row])


@cuda.jit
def divide_rows(e, s, out):
    col, row = cuda.grid(2)
    if row < e.shape[0] and col < e.shape[1]:
        out[row, col] = e[row, col] / s[row]


ELEMENTWISE_BLOCK = (32, 8)


def elementwise_grid(rows: int, cols: int):
    bx, by = ELEMENTWISE_BLOCK
    return (blocks_for(cols, bx), blocks_for(rows, by)), ELEMENTWISE_BLOCK


def softmax_reference(x) -> np.ndarray:
    """NumPy softmax over rows, in float64 — the answer both kernels are checked against."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def unfused_buffers(rows: int, cols: int):
    """Device buffers ``(m, e, s, out)`` for the unfused path — allocate once, outside any timing loop."""
    return (cuda.device_array(rows, dtype=np.float32), cuda.device_array((rows, cols), dtype=np.float32),
            cuda.device_array(rows, dtype=np.float32), cuda.device_array((rows, cols), dtype=np.float32))


def run_softmax_unfused(d_x, threads: int = 128, buffers=None):
    """Four launches on a device array; returns the device output."""
    rows, cols = d_x.shape
    row_max, row_sum = make_row_reductions(threads)
    d_m, d_e, d_s, d_out = buffers if buffers is not None else unfused_buffers(rows, cols)
    grid, block = elementwise_grid(rows, cols)
    row_max[rows, threads](d_x, d_m)
    sub_exp[grid, block](d_x, d_m, d_e)
    row_sum[rows, threads](d_e, d_s)
    divide_rows[grid, block](d_e, d_s, d_out)
    return d_out


def run_softmax(x, fused: bool = True, threads: int = 128) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("softmax expects a 2-D (rows, cols) array")
    d_x = cuda.to_device(x)
    if fused:
        d_out = cuda.device_array_like(x)
        make_softmax_fused(threads)[x.shape[0], threads](d_x, d_out)
    else:
        d_out = run_softmax_unfused(d_x, threads)
    return d_out.copy_to_host()
