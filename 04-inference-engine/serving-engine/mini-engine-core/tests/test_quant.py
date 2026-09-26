"""Quantization: formats, granularity, overheads and model-level damage."""
import numpy as np

from minengine import TinyLM, encode
from minengine.model import CORPUS
from minengine.quant import (bits_per_weight, compare_logits, error, fake_quant, fp8_e4m3, quantize,
                             quantize_activations, smoothquant_scales, weight_gb)


def test_fp8_e4m3_grid():
    x = np.array([448.0, 500.0, 0.1, 2.0 ** -9, 1e-4, 1.0625, 15.9, -3.3])
    np.testing.assert_array_equal(fp8_e4m3(x), [448.0, 448.0, 0.1015625, 2.0 ** -9, 0.0, 1.0, 16.0, -3.25])


def test_int8_per_channel_error_is_at_most_half_a_step():
    w = np.random.default_rng(0).standard_normal((64, 32))
    codes, scale, w_hat = quantize(w, "int8", "channel")
    assert codes.min() >= -127 and codes.max() <= 127 and scale.shape == (1, 1, 32)
    assert np.all(np.abs(w - w_hat) <= scale.reshape(1, 32) / 2 + 1e-12)


def test_one_outlier_ruins_per_tensor_but_not_per_channel():
    w = np.random.default_rng(1).standard_normal((64, 32)) * 0.02
    w[:, 3] *= 100                                                   # one outlier output channel
    tensor, channel = fake_quant(w, fmt="int8", granularity="tensor"), fake_quant(w, fmt="int8", granularity="channel")
    rest = [c for c in range(32) if c != 3]
    assert error(w, tensor)["rel"] < 0.05                            # the aggregate looks fine...
    assert error(w[:, rest], tensor[:, rest])["rel"] > 0.5           # ...while 31 of 32 channels are wrecked
    assert error(w, channel)["rel"] < 0.01


def test_int4_groups_beat_int4_per_channel():
    w = np.random.default_rng(2).standard_normal((128, 16)) * np.linspace(0.1, 3, 128)[:, None]
    g32 = error(w, fake_quant(w, fmt="int4", granularity="group", group_size=32))["rel"]
    ch = error(w, fake_quant(w, fmt="int4", granularity="channel"))["rel"]
    assert g32 < ch


def test_bits_per_weight_and_weight_gb_hand_computed():
    assert bits_per_weight(4, 128) == 4.125 and bits_per_weight(4, 32) == 4.5 and bits_per_weight(8) == 8
    assert np.isclose(weight_gb(8e9, 4, 128), 4.125) and np.isclose(weight_gb(8e9, 16), 16.0)
    embed = 2 * 128256 * 4096                                        # Llama-3.1-8B: untied embedding + LM head
    assert np.isclose(weight_gb(8.03e9, 4, 128, keep16_params=embed), (8.03e9 - embed) * 4.125 / 8e9 + embed * 2 / 1e9)
    assert round(weight_gb(8.03e9, 4, 128, keep16_params=embed), 2) == 5.70   # not 4.14: the 16-bit tables stay


def test_smoothquant_keeps_the_product_and_tames_activation_outliers():
    rng = np.random.default_rng(3)
    x, w = rng.standard_normal((32, 64)), rng.standard_normal((64, 16)) * 0.05
    x[:, 5] *= 60                                                    # an outlier activation channel
    s = smoothquant_scales(np.abs(x).max(0), np.abs(w).max(1), alpha=0.5)
    xs, ws = x / s, w * s[:, None]
    np.testing.assert_allclose(xs @ ws, x @ w, atol=1e-9)
    naive = error(x @ w, quantize_activations(x) @ fake_quant(w))["rel"]
    smooth = error(x @ w, quantize_activations(xs) @ fake_quant(ws))["rel"]
    assert smooth < naive / 2


def test_model_level_damage_int8_small_int4_larger():
    model, ids = TinyLM(), encode(CORPUS[:200])
    ref = model.forward_dense(ids)
    int8 = compare_logits(ref, model.quantized(fmt="int8", granularity="channel").forward_dense(ids))
    int4 = compare_logits(ref, model.quantized(fmt="int4", granularity="group", group_size=32).forward_dense(ids))
    assert int8["kl"] < int4["kl"] and int8["top1"] >= 0.97 and int4["kl"] < 0.05
