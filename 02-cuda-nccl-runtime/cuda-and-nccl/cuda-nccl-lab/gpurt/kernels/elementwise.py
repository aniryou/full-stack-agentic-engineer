"""Elementwise kernels: one thread per element, and the grid-stride loop.

The one idea: ``i = cuda.grid(1)`` is ``blockIdx.x * blockDim.x + threadIdx.x`` — the loop index of
the thread. The grid is rounded up to whole blocks, so the last block has threads with no element
and **the bounds check is not optional** (the simulator turns a missing check into an IndexError;
a real GPU silently writes past the end or dies with "illegal memory access").

A *grid-stride loop* (``for i in range(cuda.grid(1), n, cuda.gridsize(1))``) decouples the launch
from the data size: launch a few waves of blocks sized to the GPU and let each thread walk the array.

These kernels do almost no arithmetic per byte (vector add: 1 FLOP per 12 bytes), so on every GPU
they are **memory-bound** — which makes them the yardstick for achievable DRAM bandwidth (primer §2-§3).
"""

from __future__ import annotations

import numpy as np

from . import blocks_for
from numba import cuda  # noqa: I001  (after gpurt.kernels chose the mode)


@cuda.jit
def vec_add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:  # tail threads of the last block do nothing
        out[i] = a[i] + b[i]


@cuda.jit
def saxpy(alpha, x, y, out):
    """out = alpha * x + y with a grid-stride loop: any grid size covers any n."""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, out.size, stride):
        out[i] = alpha * x[i] + y[i]


@cuda.jit
def copy(src, dst):
    """The bandwidth baseline: read once, write once."""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, dst.size, stride):
        dst[i] = src[i]


# --------------------------------------------------------------------------- host wrappers
def _f32(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def run_vec_add(a, b, threads: int = 256) -> np.ndarray:
    """One thread per element: ``blocks_for(n, threads)`` blocks."""
    a, b = _f32(a), _f32(b)
    d_out = cuda.device_array_like(a)
    vec_add[blocks_for(a.size, threads), threads](cuda.to_device(a), cuda.to_device(b), d_out)
    return d_out.copy_to_host()


def run_saxpy(alpha: float, x, y, blocks: int = 4, threads: int = 128) -> np.ndarray:
    """Deliberately launches fewer threads than elements: the grid-stride loop covers the rest.

    ``alpha`` is passed as ``np.float32`` — a plain Python float is float64 to Numba, and the
    multiply would silently run in FP64 (1/32 of the FP32 rate on a T4, 1/64 on an L4; verify).
    """
    x, y = _f32(x), _f32(y)
    d_out = cuda.device_array_like(x)
    saxpy[blocks, threads](np.float32(alpha), cuda.to_device(x), cuda.to_device(y), d_out)
    return d_out.copy_to_host()


def run_copy(src, blocks: int = 64, threads: int = 256) -> np.ndarray:
    src = _f32(src)
    d_dst = cuda.device_array_like(src)
    copy[blocks, threads](cuda.to_device(src), d_dst)
    return d_dst.copy_to_host()
