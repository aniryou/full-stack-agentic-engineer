"""granularity.py - who shares a scale decides who pays for an outlier.

The one idea: a scale is set by the largest magnitude among the values that share it, so an outlier
coarsens the grid for its whole group. Per tensor, one value hurts everything; per output channel (a
row of W), one row; per group of 32-128 inputs, one short run, for 16/g extra bits per weight.
Activations are quantized at run time, so theirs is per token (dynamic) or one calibrated scale per
tensor (static); DeepSeek-V3-style blocks (128x128 weights, 1x128 activations) sit in between.
Aggregate error metrics average over groups and hide the damage next to an outlier
(`error` vs `row_errors`).

Layout: W is (out_features, in_features) as in torch.nn.Linear and checkpoints, y = x @ W.T.
Scales come back in compressed-tensors' shapes: tensor (1,), channel (out, 1), group (out, in/g),
block (out/r, in/c).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formats import FLOATS, asym_params, dequantize_int, int_scale, quantize_int, to_float


@dataclass
class Quantized:
    codes: np.ndarray        # integer codes or float-grid values, grouped
    scale: np.ndarray        # one per group, in the checkpoint's shape
    zero: np.ndarray | None  # asymmetric only
    w_hat: np.ndarray        # dequantized, the weight the model effectively computes with


def _group(W, granularity: str, group_size: int, block: tuple):
    o, i = W.shape
    if granularity == "tensor":
        return W.reshape(1, -1), (1,)
    if granularity in ("channel", "token"):
        return W, (o, 1)
    if granularity == "group":
        if i % group_size:
            raise ValueError(f"in_features {i} % group_size {group_size} != 0 (llm-compressor and Marlin reject this)")
        return W.reshape(o * i // group_size, group_size), (o, i // group_size)
    if granularity == "block":
        r, c = block
        if o % r or i % c:
            raise ValueError(f"shape {W.shape} is not a multiple of the block {block}")
        return W.reshape(o // r, r, i // c, c).transpose(0, 2, 1, 3).reshape(-1, r * c), (o // r, i // c)
    raise ValueError(f"unknown granularity {granularity!r}")


def _ungroup(G, shape, granularity: str, block: tuple):
    o, i = shape
    if granularity == "block":
        r, c = block
        return G.reshape(o // r, i // c, r, c).transpose(0, 2, 1, 3).reshape(o, i)
    return G.reshape(o, i)


def quantize(W, fmt: str = "int8", granularity: str = "channel", group_size: int = 128,
             block: tuple = (128, 128), symmetric: bool = True, convention: str = "restricted") -> Quantized:
    """Quantize a 2-D array. fmt: 'int8', 'int4', 'int<b>', or a float format ('fp8', 'fp8_e5m2', 'fp4').
    Integer symmetric uses `convention` ('restricted' +-(2^(b-1)-1), or 'full' as in checkpoints);
    asymmetric uses min-max with a zero point. Float formats scale amax onto the format's max value."""
    W = np.asarray(W, float)
    G, sshape = _group(W, granularity, group_size, block)
    zero = None
    if fmt in FLOATS:
        f = FLOATS[fmt]
        scale = np.maximum(np.abs(G).max(1, keepdims=True), 1e-12) / f.max_value
        codes = to_float(G / scale, f)
        g_hat = codes * scale
    else:
        bits = int(fmt.removeprefix("int"))
        if symmetric:
            scale = int_scale(np.abs(G).max(1, keepdims=True), bits, convention)
        else:
            scale, zero = asym_params(G.min(1, keepdims=True), G.max(1, keepdims=True), bits)
        codes = quantize_int(G, scale, bits, zero, convention)
        g_hat = dequantize_int(codes, scale, zero)
    return Quantized(codes, scale.reshape(sshape), None if zero is None else zero.reshape(sshape),
                     _ungroup(g_hat, W.shape, granularity, block))


def fake_quant(W, **kw) -> np.ndarray:
    """Quantize then dequantize: the weight a W4A16/W8A16 kernel reconstructs in registers."""
    return quantize(W, **kw).w_hat


def quantize_activations(X, fmt: str = "int8", per: str = "token", static_amax: float | None = None,
                         convention: str = "restricted") -> np.ndarray:
    """Activations (tokens, features). Dynamic: one scale per token, computed from that token at run
    time. Static: one scale per tensor from a calibrated amax; tokens beyond it clip (saturate)."""
    X = np.asarray(X, float)
    if static_amax is None:
        return fake_quant(X, fmt=fmt, granularity="token" if per == "token" else "tensor", convention=convention)
    if fmt in FLOATS:
        s = static_amax / FLOATS[fmt].max_value
        return to_float(X / s, FLOATS[fmt]) * s
    bits = int(fmt.removeprefix("int"))
    s = int_scale(static_amax, bits, convention)
    return dequantize_int(quantize_int(X, s, bits, None, convention), s)


# ------------------------------------------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------------------------------------------
def error(w, w_hat) -> dict:
    """rel = ||w - w_hat|| / ||w||; max_abs; sqnr_db = 10 log10(sum w^2 / sum (w - w_hat)^2).
    (The same definitions as minengine.quant.error, so numbers compare across the two cores.)"""
    w, w_hat = np.asarray(w, float), np.asarray(w_hat, float)
    noise = np.sum((w - w_hat) ** 2)
    return {"rel": float(np.sqrt(noise / np.sum(w ** 2))), "max_abs": float(np.abs(w - w_hat).max()),
            "sqnr_db": float(10 * np.log10(np.sum(w ** 2) / max(noise, 1e-300)))}


def row_errors(w, w_hat) -> np.ndarray:
    """Relative error of each row (output channel, or token): what an aggregate number averages away."""
    w, w_hat = np.asarray(w, float), np.asarray(w_hat, float)
    return np.sqrt(((w - w_hat) ** 2).sum(1) / np.maximum((w ** 2).sum(1), 1e-300))


def output_error(X, W, W_hat, X_hat=None) -> float:
    """Relative error of the layer output X W^T, the quantity that reaches the next layer."""
    X = np.asarray(X, float)
    ref = X @ np.asarray(W, float).T
    return float(np.linalg.norm(ref - (X if X_hat is None else X_hat) @ np.asarray(W_hat, float).T)
                 / np.linalg.norm(ref))


def output_error_by_input(X, W, W_hat) -> np.ndarray:
    """Where a layer's output error comes from, per input channel c: ||X[:, c]||^2 x ||W[:, c] - W_hat[:, c]||^2,
    the energy of that channel's term in X (W - W_hat)^T (cross terms between channels left out). A column's
    rounding error reaches the output multiplied by its input, so a channel with large activations owns the
    error even when its weight column is ordinary - the case AWQ and SmoothQuant are for."""
    X, E = np.asarray(X, float), np.asarray(W, float) - np.asarray(W_hat, float)
    return (X ** 2).sum(0) * (E ** 2).sum(0)


def argmax_agreement(ref_logits, test_logits) -> float:
    """Fraction of positions whose top-1 prediction is unchanged."""
    return float((np.argmax(ref_logits, -1) == np.argmax(test_logits, -1)).mean())
