"""W8A8 GEMMs (the epilogue), static vs dynamic activation scales, and KV-cache quantization."""
import numpy as np
import pytest

from quantcore import granularity as G, kvquant as K, w8a8
from quantcore.cost import MODELS


def test_int8_gemm_with_integer_accumulation_equals_fake_quant():
    rng = np.random.default_rng(0)
    X, W = rng.standard_normal((16, 64)), rng.standard_normal((32, 64))
    Y, info = w8a8.w8a8_matmul(X, W, "int8")
    np.testing.assert_allclose(Y, G.quantize_activations(X, "int8") @ G.fake_quant(W, fmt="int8").T, atol=1e-9)
    assert info["saturated"] == 0.0


def test_static_scale_saturates_unseen_magnitudes():
    rng = np.random.default_rng(1)
    X, W = rng.standard_normal((256, 64)), rng.standard_normal((32, 64))
    Y, info = w8a8.w8a8_matmul(X * 4, W, "int8", act="tensor", static_amax=np.abs(X).max())
    assert info["saturated"] > 0.1
    dyn, _ = w8a8.w8a8_matmul(X * 4, W, "int8")
    ref = (X * 4) @ W.T
    assert np.linalg.norm(dyn - ref) < np.linalg.norm(Y - ref) / 5


def test_block_fp8_matches_per_block_fake_quant():
    rng = np.random.default_rng(2)
    X, W = rng.standard_normal((8, 256)), rng.standard_normal((256, 256))
    W[:128, :128] *= 30
    Y = w8a8.block_fp8_matmul(X, W)
    Xh = np.hstack([G.fake_quant(X[:, k:k + 128], fmt="fp8", granularity="token") for k in (0, 128)])
    np.testing.assert_allclose(Y, Xh @ G.fake_quant(W, fmt="fp8", granularity="block").T, atol=1e-9)


def test_fp8_kv_default_scale_one_flushes_small_and_saturates_large():
    Q, Kk, V = K.synthetic_qkv()
    err = lambda f, scale: K.attention_error(Q, Kk, V * f, Kk, K.fp8_kv(V * f, scale))
    assert err(1, 1.0) < 0.005 and err(1, "tensor") < 0.005          # values of order 1: scale 1.0 is fine
    assert err(1e-3, 1.0) > 10 * err(1e-3, "tensor")                # far below 1: subnormals and zeros
    assert err(1e3, 1.0) > 0.5 and err(1e3, "tensor") < 0.005       # above 448: saturation


def test_kivi_keys_per_channel_beat_keys_per_token():
    Q, Kk, V = K.synthetic_qkv()
    for bits in (4, 2):
        ch = K.attention_error(Q, Kk, V, *K.kivi(Kk, V, bits, 32, 32, key_axis=0))
        tok = K.attention_error(Q, Kk, V, *K.kivi(Kk, V, bits, 32, 32, key_axis=1))
        assert ch < tok
    assert K.attention_error(Q, Kk, V, *K.kivi(Kk, V, 4, 32, 32)) < 0.6 * K.attention_error(
        Q, Kk, V, *K.kivi(Kk, V, 4, 32, 32, key_axis=1))


def test_kivi_keeps_the_residual_window_exact():
    _, Kk, V = K.synthetic_qkv(n_tokens=100)
    Kh, Vh = K.kivi(Kk, V, 2, 32, 32)                            # (100 - 32) // 32 * 32 = 64 tokens quantized
    np.testing.assert_array_equal(Kh[64:], Kk[64:])
    assert not np.allclose(Kh[:64], Kk[:64])


def test_kv_bytes_hand_computed():
    assert K.kv_bits_per_element(2, 32, 16, 16) == 3.0 and K.kv_bits_per_element(4, 32, 16, 16) == 5.0
    m = MODELS["llama-3.1-8b"]
    assert m.kv_bytes_per_token(16) == 131072 and m.kv_bytes_per_token(8) == 65536
    assert m.kv_bytes_per_token(3) == 24576 and round(16 / 3, 1) == 5.3
