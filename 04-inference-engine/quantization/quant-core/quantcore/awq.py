"""awq.py - protect the weights that meet large activations by scaling them up before rounding.

The one idea: a weight's rounding error reaches the output multiplied by the activation it meets, so
the few input channels with large activations own most of the output error. AWQ multiplies those
weight columns by s > 1 before quantizing and divides the activations by the same s, which is folded
into the previous RMSNorm gain or linear layer and so costs nothing at run time: X W^T = (X/s)(W s)^T.
A scaled-up column uses more of its group's grid, so its relative rounding error shrinks (the other
columns in the group pay a little). s = mean|x|^alpha, with alpha chosen by a 20-point grid search
that minimises the layer's output error on calibration data. The result is still a plain weight-only
INT4 checkpoint; calibration only picks s (`llm-awq/awq/quantize/auto_scale.py`).
"""
from __future__ import annotations

import numpy as np

from .granularity import Quantized, quantize


def act_mean(X) -> np.ndarray:
    """AWQ's activation statistic: mean |x| per input channel over the calibration tokens."""
    return np.abs(np.asarray(X, float)).mean(0)


def _q(W, bits, group_size, symmetric):
    return quantize(W, fmt=f"int{bits}", granularity="group" if group_size else "channel",
                    group_size=group_size or 0, symmetric=symmetric, convention="full")


def search_scale(W, X, bits: int = 4, group_size: int | None = 128, n_grid: int = 20,
                 symmetric: bool = False, duo: bool = False):
    """Grid search over alpha in {0, 1/20, ..., 19/20}: s = x_mean^alpha, normalised by sqrt(max s * min s);
    `duo` (llm-compressor's variant) divides by w_mean^(1 - alpha). Loss = mean squared output error of
    (X / s) Q(W s)^T against X W^T. alpha = 0 is plain RTN, so the search can never do worse than RTN on
    the calibration set. Returns (s, best_alpha, losses)."""
    W, X = np.asarray(W, float), np.asarray(X, float)
    ref = X @ W.T
    x_mean = act_mean(X)
    w_mean = np.abs(W).mean(0)
    losses, best = [], (np.inf, None, None)
    for k in range(n_grid):
        a = k / n_grid
        s = np.maximum(x_mean ** a / ((w_mean ** (1 - a) + 1e-4) if duo else 1.0), 1e-4)
        s = s / np.sqrt(s.max() * s.min())
        loss = float(np.mean(((X / s) @ _q(W * s, bits, group_size, symmetric).w_hat.T - ref) ** 2))
        losses.append(loss)
        if loss < best[0]:
            best = (loss, s, a)
    return best[1], best[2], np.array(losses)


def awq(W, X, bits: int = 4, group_size: int | None = 128, symmetric: bool = False, **kw):
    """Search s, quantize W * s, and return (Quantized for W * s, s). At run time the layer computes
    (x / s) @ Q(W s)^T: fold 1/s into the previous norm gain (`tinymodel.TinyModel.fold`); the weight the
    original activations effectively meet is Q(W s) / s."""
    s, _, _ = search_scale(W, X, bits, group_size, symmetric=symmetric, **kw)
    return _q(np.asarray(W, float) * s, bits, group_size, symmetric), s

