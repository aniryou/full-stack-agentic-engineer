"""w8a8.py - quantize weights and activations, multiply the codes, apply both scales once at the end.

The one idea: a W8A8 GEMM multiplies 8-bit codes on the tensor cores and accumulates in a wide
register (INT32 for INT8, FP32 for FP8). Scales factor out of the sum, so they are applied once per
output element in the epilogue: y[t, j] = s_x[t] * s_w[j] * sum_k qx[t, k] qw[j, k]. That is why the
activation scale may vary per token and the weight scale per output channel, but neither may vary
along k, the reduction axis - and why block formats (1x128 activations, 128x128 weights, DeepSeek-V3)
need one partial sum per 128-wide k block, rescaled before it is added. Static activation scales come
from calibration and saturate tokens larger than anything calibration saw; dynamic ones cost one
max-reduction per token.
"""
from __future__ import annotations

import numpy as np

from .formats import FLOATS, int_scale, quantize_int, to_float


def _codes(A, fmt: str, amax):
    """Codes and scale for rows of A given each row's amax (INT8 restricted +-127, or an FP8 grid)."""
    if fmt in FLOATS:
        s = np.maximum(amax, 1e-12) / FLOATS[fmt].max_value
        return to_float(A / s, FLOATS[fmt]), s
    s = int_scale(amax, int(fmt.removeprefix("int")))
    return quantize_int(A, s, int(fmt.removeprefix("int"))), s


def w8a8_matmul(X, W, fmt: str = "int8", act: str = "token", weight: str = "channel",
                static_amax: float | None = None):
    """Emulate a W8A8 GEMM, X (tokens, in) times W (out, in)^T. act: 'token' (dynamic per-token) or
    'tensor' (static, from `static_amax`, else this batch's amax); weight: 'channel' or 'tensor'.
    Integer codes are multiplied in int64 (a stand-in for INT32 accumulation), FP8 codes in float64.
    Returns (Y, info) with info['saturated']: the fraction of activations clipped by a static scale."""
    X, W = np.asarray(X, float), np.asarray(W, float)
    if act == "token":
        x_amax = np.abs(X).max(1, keepdims=True)
    else:
        x_amax = np.full((1, 1), float(np.abs(X).max()) if static_amax is None else static_amax)
    w_amax = np.abs(W).max(1, keepdims=True) if weight == "channel" else np.full((1, 1), np.abs(W).max())
    qx, sx = _codes(X, fmt, x_amax)
    qw, sw = _codes(W, fmt, w_amax)
    acc = qx.astype(np.int64) @ qw.astype(np.int64).T if fmt.startswith("int") else qx @ qw.T
    return acc * sx * sw.T, {"saturated": float((np.abs(X) > x_amax).mean())}


def block_fp8_matmul(X, W, block: int = 128):
    """DeepSeek-V3-style FP8: weights with one scale per 128x128 block, activations with one scale per
    token per 128 channels (computed on the fly). Each k-block's FP32 partial sum is multiplied by its
    own two scales before accumulation - the 'per-block rescale' the kernels implement."""
    X, W = np.asarray(X, float), np.asarray(W, float)
    (t, k), (o, _) = X.shape, W.shape
    Y = np.zeros((t, o))
    for kb in range(0, k, block):
        xb = X[:, kb:kb + block]
        qx, sx = _codes(xb, "fp8", np.abs(xb).max(1, keepdims=True))
        for ob in range(0, o, block):
            wb = W[ob:ob + block, kb:kb + block]
            qw, sw = _codes(wb, "fp8", np.abs(wb).max())
            Y[:, ob:ob + block] += (qx @ qw.T) * sx * sw
    return Y
