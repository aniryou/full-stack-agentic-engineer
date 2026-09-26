"""smoothquant.py - move activation outliers into the weights, where per-channel scales can absorb them.

The one idea: W8A8 quantizes activations per token at run time, and a few input channels are 10-100x
larger than the rest in almost every token. A per-token scale is set by them, so the other channels
round to a handful of levels. Weights have no such channels. SmoothQuant divides activation channel j
by s_j and multiplies weight column j by s_j - X W^T = (X / s)(W s)^T exactly - with
s_j = max|X_j|^alpha / max|W_j|^(1 - alpha). alpha = 0.5 splits the difficulty evenly (the reference
default); models with strong outliers want 0.75-0.9 (the repo's tuned values: Llama-3-8B 0.85). The
1/s is folded into the preceding LayerNorm/RMSNorm gain, so run time pays nothing
(`smoothquant/smooth.py: smooth_ln_fcs`).
"""
from __future__ import annotations

import numpy as np

from .granularity import fake_quant, output_error, quantize_activations


def smooth_scales(x_absmax, w_absmax, alpha: float = 0.5) -> np.ndarray:
    """Per input channel: max|X_j|^alpha / max|W_j|^(1 - alpha), clamped away from 0 as in the reference."""
    return (np.maximum(np.asarray(x_absmax, float), 1e-5) ** alpha
            / np.maximum(np.asarray(w_absmax, float), 1e-5) ** (1 - alpha))


def smooth(X, W, alpha: float = 0.5):
    """Return (X / s, W * s, s) for activations X (tokens, in) and weight W (out, in)."""
    X, W = np.asarray(X, float), np.asarray(W, float)
    s = smooth_scales(np.abs(X).max(0), np.abs(W).max(0), alpha)
    return X / s, W * s, s


def w8a8_error(X, W, fmt: str = "int8", alpha: float | None = None) -> float:
    """Relative output error of W8A8 (per-channel weights, per-token dynamic activations), optionally
    after smoothing with `alpha`. The error is measured against the unsmoothed full-precision X W^T."""
    Xs, Ws = (np.asarray(X, float), np.asarray(W, float)) if alpha is None else smooth(X, W, alpha)[:2]
    W_hat = fake_quant(Ws, fmt=fmt, granularity="channel")
    return output_error(Xs, Ws, W_hat, X_hat=quantize_activations(Xs, fmt, per="token"))


def alpha_sweep(X, W, alphas=(0.0, 0.25, 0.5, 0.75, 0.85, 1.0), fmt: str = "int8") -> dict:
    """Output error per alpha: 0 flattens the weights and leaves all the difficulty in the activations,
    1 flattens the activations and moves all of it into the weights; the best value lies in between."""
    return {a: w8a8_error(X, W, fmt, a) for a in alphas}
