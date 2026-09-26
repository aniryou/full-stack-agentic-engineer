"""The same two ideas in Triton — optional, T1 only (needs torch + triton + a GPU; imported lazily).

The one idea: Triton moves the unit of programming from the *thread* to the *block*. A Triton
program instance handles a whole ``BLOCK_SIZE`` vector (``tl.arange``) with masked loads and
stores; the compiler decides how lanes, warps, vectorised loads and shared memory map onto it.
Compare with ``elementwise.vec_add`` (one element per thread, explicit bounds check) and
``softmax.make_softmax_fused`` (explicit shared-memory tree): same algorithms, far less code, and
this is what ``torch.compile`` generates for fused elementwise/reduction kernels.

Kernels follow the official Triton tutorials (vector add, fused softmax). One row must fit in one
block here (``BLOCK_SIZE = next_power_of_2(n_cols)``), as in the tutorial.
"""

from __future__ import annotations

import importlib.util


def available() -> bool:
    """True when triton and torch import and torch sees a CUDA device."""
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements  # the bounds check, vectorised
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)  # the row stays in registers: one read, one write
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)

    return triton, add_kernel, softmax_kernel


_CACHE: dict = {}


def _kernels():
    if not _CACHE:
        _CACHE["triton"], _CACHE["add"], _CACHE["softmax"] = _build()
    return _CACHE["triton"], _CACHE["add"], _CACHE["softmax"]


def vec_add(x, y, block_size: int = 1024):
    """x + y for CUDA torch tensors of equal shape."""
    import torch

    triton, add_kernel, _ = _kernels()
    out = torch.empty_like(x)
    n = out.numel()
    add_kernel[(triton.cdiv(n, block_size),)](x, y, out, n, BLOCK_SIZE=block_size)
    return out


def softmax(x):
    """Row softmax of a 2-D CUDA torch tensor, one program per row."""
    import torch

    triton, _, softmax_kernel = _kernels()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols,
                              BLOCK_SIZE=triton.next_power_of_2(n_cols))
    return out
