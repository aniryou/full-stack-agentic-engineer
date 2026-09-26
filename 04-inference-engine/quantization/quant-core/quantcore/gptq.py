"""gptq.py - quantize one input column at a time and let the columns not yet quantized absorb the error.

The one idea: round-to-nearest (RTN) rounds every weight independently, but what matters is the
layer's output error ||X W^T - X Q^T||^2, and that couples the weights of one row through the input
correlations H = 2/n X^T X. GPTQ walks the input columns in order; after rounding column j it spreads
the error e = (w_j - q_j) / [H^-1]_jj over the columns still to come, in proportion to row j of the
(upper Cholesky factor of the) inverse Hessian - the Optimal Brain Surgeon update - so later columns
lean against the damage. Calibration data enters only through H; damping (1% of the mean diagonal)
keeps H invertible when some inputs never fire. The output is an ordinary INT4 checkpoint: same
format and kernels as RTN, better-chosen codes.

This is the reference algorithm (`gptq.py: fasterquant`, IST-DASLab) without its "lazy batch" trick,
which defers updates to columns beyond a 128-column block for GPU efficiency. The OBS updates are the
same either way; the results are identical per channel, or when every group starts on a block boundary.
For groups smaller than the block, the reference computes a mid-block group's scale from the global W,
which lacks that block's in-flight updates, while this loop uses the fully updated weights - a slightly
different (not worse) choice of scales (tests/test_gptq.py shows both). llm-compressor's GPTQModifier
does neither: its group scales come from the weight observer on the original weights, before the loop.
"""
from __future__ import annotations

import numpy as np

from .formats import asym_params, dequantize_int, int_scale, quantize_int
from .granularity import Quantized, quantize


def hessian(X) -> np.ndarray:
    """H = 2/n sum_t x_t x_t^T over calibration tokens X (n, in): the curvature of the output error."""
    X = np.asarray(X, float)
    return 2.0 / len(X) * X.T @ X


def rtn(W, bits: int = 4, group_size: int | None = 128, symmetric: bool = True, convention: str = "full") -> Quantized:
    """Round-to-nearest baseline: no data, every weight rounded on its own group's grid."""
    gran = "group" if group_size else "channel"
    return quantize(W, fmt=f"int{bits}", granularity=gran, group_size=group_size or 0,
                    symmetric=symmetric, convention=convention)


def _params(cols, bits, symmetric, convention):
    """Scale (and zero) for one group of columns, per output row."""
    if symmetric:
        return int_scale(np.abs(cols).max(1), bits, convention), None
    return asym_params(cols.min(1), cols.max(1), bits)


def gptq(W, X=None, H=None, bits: int = 4, group_size: int | None = 128, symmetric: bool = True,
         percdamp: float = 0.01, actorder: bool = False, convention: str = "full") -> Quantized:
    """GPTQ on W (out, in) with calibration inputs X (n, in) or a precomputed Hessian H.
    Group parameters are computed when the loop reaches a group, on the already-updated weights (the
    reference's behaviour when groups align with its 128-column blocks; see the module docstring).
    actorder: visit columns by decreasing H_jj (the most-used inputs first, while
    the most columns remain to absorb their error), with each group's parameters fixed up front from
    the original weights ('static groups') so the checkpoint layout does not change."""
    W = np.asarray(W, float).copy()
    H = hessian(X) if H is None else np.asarray(H, float).copy()
    out, n = W.shape
    g = group_size or n
    dead = np.diag(H) == 0                                  # inputs that never fire: no information
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = np.argsort(-np.diag(H), kind="stable") if actorder else np.arange(n)
    static = [_params(W[:, k:k + g], bits, symmetric, convention) for k in range(0, n, g)] if actorder else None
    W, H = W[:, perm], H[perm][:, perm]
    H[np.diag_indices(n)] += percdamp * np.mean(np.diag(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T              # H^-1 = U^T U, U upper triangular
    Q = np.zeros_like(W)
    scales = np.zeros((out, n // g))
    zeros = None if symmetric else np.zeros((out, n // g))
    for j in range(n):
        grp = perm[j] // g
        if actorder:
            scale, zero = static[grp]
        elif j % g == 0:
            scale, zero = _params(W[:, j:j + g], bits, symmetric, convention)
        scales[:, grp] = scale
        if zeros is not None:
            zeros[:, grp] = zero
        q = dequantize_int(quantize_int(W[:, j], scale, bits, zero, convention), scale, zero)
        err = (W[:, j] - q) / U[j, j]
        W[:, j + 1:] -= np.outer(err, U[j, j + 1:])         # the OBS update: later columns compensate
        Q[:, j] = q
    w_hat = np.empty_like(Q)
    w_hat[:, perm] = Q
    return Quantized(None, scales, zeros, w_hat)


def layer_loss(W, W_hat, X) -> float:
    """The objective GPTQ minimises: mean squared output error over the calibration tokens."""
    X = np.asarray(X, float)
    return float(np.mean((X @ (np.asarray(W, float) - np.asarray(W_hat, float)).T) ** 2))
