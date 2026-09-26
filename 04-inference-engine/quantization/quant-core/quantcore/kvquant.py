"""kvquant.py - the KV cache is re-read every decode step, so its bytes are the other thing to shrink.

The one idea: K and V are activations stored for later - quantized once when written, dequantized on
every read. FP8 E4M3 halves the bytes with 3 mantissa bits, and needs a scale that fits the range:
vLLM uses 1.0 unless the checkpoint carries calibrated k_scale/v_scale, which is fine for values of
order 1 and wrong for values far below it (they flush to zero) or above 448 (they saturate). Keys have
a few channels that are large in every token; values do not. So KIVI quantizes keys per channel (one
scale and minimum per channel per group of tokens) and values per token, keeping the newest tokens in
16-bit until a group fills. The error that matters is the attention output's (`attention_error`).
"""
from __future__ import annotations

import numpy as np

from .formats import E4M3, asym_params, dequantize_int, quantize_int, to_float


def synthetic_qkv(n_tokens: int = 512, head_dim: int = 128, n_queries: int = 8, outlier_channels=(5, 37, 70, 101),
                  outlier_size: float = 12.0, seed: int = 0):
    """One head's decode queries Q, keys K and values V with the structure KIVI reports: a few key
    channels carry a large, nearly constant magnitude in every token; values vary in size per token and
    share a per-channel mean (real value vectors are not zero-mean, so the output is not averaged noise)."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((n_tokens, head_dim))
    for c in outlier_channels:
        K[:, c] = outlier_size * rng.choice([-1, 1]) * (1 + 0.1 * rng.standard_normal(n_tokens))
    V = rng.standard_normal((n_tokens, head_dim)) * np.exp(0.5 * rng.standard_normal((n_tokens, 1)))
    V += rng.standard_normal(head_dim)
    return rng.standard_normal((n_queries, head_dim)), K, V


def attention(Q, K, V) -> np.ndarray:
    S = Q @ K.T / np.sqrt(K.shape[1])
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V


def attention_error(Q, K, V, K_hat, V_hat) -> float:
    """Relative error of the attention output when the cache holds K_hat, V_hat instead of K, V."""
    O = attention(Q, K, V)
    return float(np.linalg.norm(attention(Q, K_hat, V_hat) - O) / np.linalg.norm(O))


def fp8_kv(X, scale="tensor") -> np.ndarray:
    """E4M3 cache. scale: a number (vLLM's uncalibrated default is 1.0), 'tensor' (calibrated amax / 448,
    one per layer as llm-compressor writes k_scale/v_scale), or 'token' (dynamic, one per cached token)."""
    X = np.asarray(X, float)
    if scale == "tensor":
        s = np.abs(X).max() / E4M3.max_value
    elif scale == "token":
        s = np.abs(X).max(1, keepdims=True) / E4M3.max_value
    else:
        s = float(scale)
    return to_float(X / s, E4M3) * s


def quant_groups(X, axis: int, bits: int, group: int) -> np.ndarray:
    """Asymmetric min-max INT quantization where each group of `group` consecutive entries along `axis`
    shares a scale and a minimum. axis=0 on (tokens, channels): per-channel scales over token groups
    (KIVI keys); axis=1: per-token scales over channel groups (KIVI values)."""
    X = np.asarray(X, float)
    Xt = X if axis == 1 else X.T
    r, c = Xt.shape
    G = Xt.reshape(r, c // group, group)
    scale, zero = asym_params(G.min(-1, keepdims=True), G.max(-1, keepdims=True), bits)
    out = dequantize_int(quantize_int(G, scale, bits, zero), scale, zero).reshape(r, c)
    return out if axis == 1 else out.T


def kivi(K, V, bits: int = 2, group: int = 32, residual: int = 32, key_axis: int = 0):
    """KIVI: keys per channel (key_axis=0), values per token, both asymmetric in groups of `group`; the
    newest tokens - the last `residual`, plus any that do not fill a whole group - stay full precision.
    key_axis=1 quantizes keys per token instead, to show why KIVI does not."""
    n = len(K)
    nq = max(0, (n - residual) // group * group)
    K_hat, V_hat = np.array(K, float), np.array(V, float)
    if nq:
        K_hat[:nq] = quant_groups(K_hat[:nq], key_axis, bits, group)
        V_hat[:nq] = quant_groups(V_hat[:nq], 1, bits, group)
    return K_hat, V_hat


def kv_bits_per_element(bits: float, group: int | None = None, scale_bits: float = 16, zero_bits: float = 0) -> float:
    """Storage per cached element: KIVI 2-bit, groups of 32 with a 16-bit scale and minimum: 2 + 32/32 = 3."""
    return bits + ((scale_bits + zero_bits) / group if group else 0.0)

