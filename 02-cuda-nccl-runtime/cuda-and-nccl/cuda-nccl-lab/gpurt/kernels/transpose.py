"""Transpose: the cleanest demonstration of memory coalescing and shared-memory bank conflicts.

The one idea: DRAM is read in 32-byte **sectors**; a warp's 32 loads (or stores) are served by the
set of sectors they touch. A transpose must read rows and write columns, so one side of a naive
kernel is **strided**: 32 lanes write 32 different rows -> 32 sectors for 128 useful bytes
(12.5 % efficiency). The tiled kernel reads a 32x32 tile with coalesced row loads into shared
memory, synchronises, and writes it back as coalesced rows of the output — both sides now touch
4 sectors per warp request. Same bytes, far fewer transactions.

The shared tile is declared ``(TILE, TILE + 1)``: without the one-column pad, reading a *column*
of a 32-wide float tile hits the same shared-memory bank 32 times (a 32-way bank conflict, served
serially); the pad shifts each row by one bank. ``make_transpose_tiled(pad=0)`` keeps the conflict
so you can time the difference on a GPU.

Launch shape follows NVIDIA's classic transpose example: blocks of ``TILE x ROWS = 32 x 8``
threads, each thread handling ``TILE / ROWS = 4`` elements. ``copy2d`` uses the identical indexing
without the transpose — the upper bound a transpose can hope to match (primer §3).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numba import cuda, float32

from . import blocks_for

TILE = 32  # one warp spans a tile row
ROWS = 8  # threads per block in y; each thread covers TILE // ROWS rows


@cuda.jit
def copy2d(inp, out):
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    for j in range(0, TILE, ROWS):
        if x < inp.shape[1] and y + j < inp.shape[0]:
            out[y + j, x] = inp[y + j, x]


@cuda.jit
def transpose_naive(inp, out):
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    for j in range(0, TILE, ROWS):
        if x < inp.shape[1] and y + j < inp.shape[0]:
            out[x, y + j] = inp[y + j, x]  # read: row-contiguous; write: stride = out row length


@lru_cache(maxsize=None)
def make_transpose_tiled(pad: int = 1):
    """Tiled transpose through a ``(TILE, TILE + pad)`` shared tile. ``pad=1`` avoids bank conflicts."""
    width = TILE + pad

    @cuda.jit
    def transpose_tiled(inp, out):
        tile = cuda.shared.array((TILE, width), float32)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        x = cuda.blockIdx.x * TILE + tx
        y = cuda.blockIdx.y * TILE + ty
        for j in range(0, TILE, ROWS):
            if x < inp.shape[1] and y + j < inp.shape[0]:
                tile[ty + j, tx] = inp[y + j, x]  # coalesced read of a tile row
        cuda.syncthreads()
        x = cuda.blockIdx.y * TILE + tx  # swap the block coordinates for the output
        y = cuda.blockIdx.x * TILE + ty
        for j in range(0, TILE, ROWS):
            if x < out.shape[1] and y + j < out.shape[0]:
                out[y + j, x] = tile[tx, ty + j]  # coalesced write; column read of the tile
    return transpose_tiled


transpose_tiled = make_transpose_tiled(1)


def launch_config(rows: int, cols: int):
    """``(grid, block)`` for an input of shape (rows, cols)."""
    return (blocks_for(cols, TILE), blocks_for(rows, TILE)), (TILE, ROWS)


def run_transpose(a, variant: str = "tiled", pad: int = 1) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    rows, cols = a.shape
    grid, block = launch_config(rows, cols)
    if variant == "copy":
        kernel, out_shape = copy2d, (rows, cols)
    elif variant == "naive":
        kernel, out_shape = transpose_naive, (cols, rows)
    else:
        kernel, out_shape = make_transpose_tiled(pad), (cols, rows)
    d_out = cuda.device_array(out_shape, dtype=np.float32)
    kernel[grid, block](cuda.to_device(a), d_out)
    return d_out.copy_to_host()
