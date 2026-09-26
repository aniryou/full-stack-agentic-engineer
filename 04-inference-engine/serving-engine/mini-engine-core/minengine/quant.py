"""quant.py - quantization: fewer bits per number, plus a scale that says what the bits mean.

The one idea: w ~ code * scale. The code is a small integer (INT8: -127..127, INT4: -7..7) or an
FP8 value; the scale is a higher-precision number shared by a GROUP of weights. The group is the
design choice: one scale per tensor is cheap but one outlier ruins everyone's resolution; one per
output channel isolates outlier rows; one per 32/128 inputs (group-wise, as in GPTQ/AWQ INT4)
isolates outliers further for 16/g extra bits per weight. What quantization buys depends on the
bottleneck: decode is memory-bound, so fewer weight BYTES are faster tokens; prefill is
compute-bound, so only formats the tensor cores compute in natively (FP8/INT8 W8A8) make it faster.
Everything here runs in float64 and returns the dequantized ("fake-quant") result, so the error is
visible and measurable.
"""
from __future__ import annotations

import numpy as np

FP8_E4M3_MAX = 448.0
QMAX = {"int8": 127, "int4": 7, "fp8": FP8_E4M3_MAX}


def fp8_e4m3(x):
    """Round to the nearest FP8-E4M3 value: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits,
    no infinities, largest finite 448, smallest subnormal 2^-9. Saturates out-of-range values."""
    x = np.clip(np.asarray(x, float), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    e = np.floor(np.log2(np.maximum(np.abs(x), 2.0 ** -6)))    # binade; below 2^-6 the spacing is fixed
    step = 2.0 ** (e - 3)                                      # 3 mantissa bits -> 8 values per binade
    return np.round(x / step) * step


def quantize(w, fmt: str = "int8", granularity: str = "channel", group_size: int = 128):
    """w: (d_in, d_out), used as x @ w. granularity: 'tensor' (one scale), 'channel' (one per output
    column), 'group' (one per group_size consecutive inputs of each column).
    Returns (codes, scales, w_hat) with w_hat = codes * scales, the dequantized weight."""
    w = np.asarray(w, float)
    d_in, d_out = w.shape
    g = group_size if granularity == "group" else d_in
    wg = w.reshape(d_in // g, g, d_out)
    amax = np.abs(wg).max() if granularity == "tensor" else np.abs(wg).max(axis=1, keepdims=True)
    scale = np.maximum(amax, 1e-12) / QMAX[fmt]
    codes = fp8_e4m3(wg / scale) if fmt == "fp8" else np.clip(np.round(wg / scale), -QMAX[fmt], QMAX[fmt])
    return codes, scale, (codes * scale).reshape(d_in, d_out)


def fake_quant(w, **kw):
    """Quantize then dequantize: the weight the model effectively computes with."""
    return quantize(w, **kw)[2]


def quantize_activations(x, fmt: str = "int8", per: str = "token"):
    """Dynamic activation quantization for W8A8: one scale per token (row) or per tensor."""
    return quantize(np.asarray(x, float).T, fmt, "channel" if per == "token" else "tensor")[2].T


def smoothquant_scales(x_absmax, w_absmax, alpha: float = 0.5):
    """SmoothQuant: per input channel j, s_j = max|X_j|^alpha / max|W_j|^(1-alpha). Using X / s and
    s * W leaves X @ W unchanged but moves activation outliers into the weights, which tolerate them."""
    return np.maximum(x_absmax, 1e-8) ** alpha / np.maximum(w_absmax, 1e-8) ** (1 - alpha)


def error(w, w_hat) -> dict:
    """rel: ||w - w_hat|| / ||w||; max_abs: worst element; sqnr_db: signal-to-quantization-noise ratio."""
    w, w_hat = np.asarray(w, float), np.asarray(w_hat, float)
    noise = np.sum((w - w_hat) ** 2)
    return {"rel": float(np.sqrt(noise / np.sum(w ** 2))), "max_abs": float(np.abs(w - w_hat).max()),
            "sqnr_db": float(10 * np.log10(np.sum(w ** 2) / max(noise, 1e-300)))}


def bits_per_weight(bits: float, group_size: int | None = None, scale_bits: int = 16) -> float:
    """Storage cost including the scales: INT4 with a 16-bit scale per 128 weights = 4.125 bits."""
    return bits + (scale_bits / group_size if group_size else 0.0)


def weight_gb(params: float, bits: float, group_size: int | None = None, scale_bits: int = 16,
              keep16_params: float = 0.0) -> float:
    """Weights in GB: `params` at bits_per_weight, except `keep16_params` of them kept in 16-bit - the
    embedding table and LM head, which GPTQ/AWQ/FP8 checkpoints do not quantize."""
    low = (params - keep16_params) * bits_per_weight(bits, group_size, scale_bits)
    return (low + keep16_params * 16) / 8 / 1e9


def compare_logits(ref, test) -> dict:
    """Model-level damage over many positions: mean KL(p_ref || p_test) and top-1 agreement."""
    lp = ref - ref.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    lq = test - test.max(1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(1, keepdims=True))
    kl = (np.exp(lp) * (lp - lq)).sum(1).mean()
    return {"kl": float(kl), "top1": float((ref.argmax(1) == test.argmax(1)).mean())}
