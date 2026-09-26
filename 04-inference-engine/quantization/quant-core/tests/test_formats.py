"""Number formats: grids from bit patterns, rounding, block formats, bits per weight, packing."""
import numpy as np
import pytest

from quantcore import formats as F
from quantcore import granularity as G


def test_fp8_e4m3_grid_from_bit_patterns():
    g = F.E4M3.grid()
    assert len(g) == 127 and 2 * len(g) - 1 == 253          # 254 finite codes (+0, -0), 2 NaN codes
    assert g[-1] == 448 and g[1] == 2.0 ** -9 and F.E4M3.min_normal == 2.0 ** -6
    assert np.log2(F.E4M3.max_value / F.E4M3.min_normal) == pytest.approx(14.807, abs=1e-3)


def test_fp8_e5m2_trades_precision_for_range():
    g = F.E5M2.grid()
    assert g[-1] == 57344 and F.E5M2.min_normal == 2.0 ** -14 and F.E5M2.min_subnormal == 2.0 ** -16
    x = np.random.default_rng(0).standard_normal(10000)
    rel = lambda f: (np.abs(F.to_float(x, f) - x) / np.abs(x))[np.abs(x) > f.min_normal].max()
    assert rel(F.E4M3) <= 1 / 16 and rel(F.E5M2) <= 1 / 8 and rel(F.E5M2) > rel(F.E4M3)


def test_minengine_fp8_values_are_reproduced():
    x = np.array([448.0, 500.0, 0.1, 2.0 ** -9, 1e-4, 1.0625, 15.9, -3.3])   # minengine tests/test_quant.py
    np.testing.assert_array_equal(F.to_float(x), [448.0, 448.0, 0.1015625, 2.0 ** -9, 0.0, 1.0, 16.0, -3.25])


def test_e2m1_grid_and_vllm_tie_thresholds():
    np.testing.assert_array_equal(F.E2M1.grid(), [0, 0.5, 1, 1.5, 2, 3, 4, 6])
    # vLLM cast_to_fp4: <=0.25 -> 0, [0.75, 1.25] -> 1, [1.75, 2.5] -> 2, [3.5, 5] -> 4, > 5 -> 6
    x = np.array([0.25, 0.26, 0.75, 1.25, 1.3, 1.75, 2.5, 2.6, 3.5, 5.0, 5.01, 9.0])
    np.testing.assert_array_equal(F.to_float(x, F.E2M1), [0, 0.5, 1, 1, 1.5, 2, 2, 3, 4, 4, 6, 6])


def test_to_float_is_nearest_grid_value():
    for f in (F.E4M3, F.E5M2, F.E2M1):
        g = np.concatenate([-f.grid()[::-1], f.grid()])
        x = np.random.default_rng(1).uniform(-f.max_value, f.max_value, 2000) * np.random.default_rng(2).uniform(0, 1, 2000) ** 4
        nearest = g[np.abs(x[:, None] - g[None]).argmin(1)]
        assert np.all(np.abs(F.to_float(x, f) - x) <= np.abs(nearest - x) + 1e-15)


def test_integer_conventions():
    assert F.int_range(4) == (-7, 7) and F.int_range(4, convention="full") == (-8, 7) and F.int_range(4, False) == (0, 15)
    assert F.int_scale(7.0, 4) == 1.0 and F.int_scale(7.5, 4, "full") == 1.0 and F.int_scale(127.5, 8, "full") == 1.0
    codes = F.quantize_int(np.array([-1.0, 1.0]), F.int_scale(1.0, 4, "full"), 4, convention="full")
    np.testing.assert_array_equal(codes, [-8, 7])      # -amax uses the extra code, +amax clips by half a step


def test_asymmetric_keeps_zero_exact():
    x = np.array([-0.3, 0.0, 0.4, 1.2])
    scale, zero = F.asym_params(x.min(), x.max(), 4)
    assert scale == pytest.approx(1.5 / 15) and zero == 3
    assert F.dequantize_int(F.quantize_int(0.0, scale, 4, zero), scale, zero) == 0.0


def test_bits_per_weight_hand_computed():
    assert F.bits_per_weight(4, 128) == 4.125                              # minengine.quant convention
    assert F.bits_per_weight(4, 128, zero_point_bits=4) == 4.15625         # servelab.sizing: 4 + 2.5 B / 128
    assert F.bits_per_weight(4, 32) == 4.5 and F.bits_per_weight(4, 32, zero_point_bits=4) == 4.625
    assert F.bits_per_weight(4, 32, scale_bits=8) == 4.25                  # MXFP4
    assert F.bits_per_weight(4, 16, scale_bits=8) == 4.5                   # NVFP4
    assert F.bits_per_weight(8, 128 * 128, scale_bits=32) == pytest.approx(8.002, abs=1e-3)   # DeepSeek-V3 FP8 blocks


def test_mxfp4_shared_exponent():
    x = np.zeros(64)
    x[0], x[32] = 5.0, 0.75
    elems, code, x_hat = F.mxfp4(x)
    np.testing.assert_array_equal(code.ravel(), [127, 124])               # 2^0 and 2^-3
    assert x_hat[0] == 5.0 - 1.0 and x_hat[32] == 0.75                     # 5 rounds to 4 (tie to even); 0.75 x 8 = 6
    _, _, sat = F.mxfp4(np.full(32, 7.5))
    assert np.all(sat == 6.0)                                              # 7.5 / 2^0 > 6 saturates: MX floors the exponent


def test_mxfp4_compressed_tensors_rounds_the_exponent_up_at_1_75():
    """compressed-tensors' round_to_power_2 (what llm-compressor writes): amax rounds up to the next power of two
    when its mantissa is >= 1.75, so the block max lands in [3.5, 7) instead of the OCP rule's [4, 8)."""
    for amax, ocp, ct in ((5.0, 127, 127), (7.0, 127, 128), (7.5, 127, 128), (6.9, 127, 127), (3.6, 126, 127)):
        assert F.mxfp4(np.full(32, amax))[1].item() == ocp and F.mxfp4(np.full(32, amax), rule="compressed-tensors")[1].item() == ct
    _, _, x_hat = F.mxfp4(np.full(32, 7.5), rule="compressed-tensors")
    assert np.all(x_hat == 8.0)                                            # 7.5 / 2 = 3.75 -> 4, x 2: no saturation
    W = np.random.default_rng(0).standard_normal((256, 512))
    top = {r: np.abs(W.reshape(256, 16, 32)).max(-1) / 2.0 ** (F.mxfp4(W, rule=r)[1][..., 0].astype(float) - 127)
           for r in ("ocp", "compressed-tensors")}
    assert 4 <= top["ocp"].min() and top["ocp"].max() < 8 and 3.5 <= top["compressed-tensors"].min() and top["compressed-tensors"].max() < 7
    with pytest.raises(ValueError):
        F.mxfp4(W, rule="nearest")


def test_the_full_convention_tie_does_not_depend_on_the_last_bit():
    """Under the full convention -amax / scale is exactly -7.5 in exact arithmetic; one ulp either side must not
    decide between -7 and -8 (that ulp differs between BLAS builds, and GPTQ propagates it)."""
    s = F.int_scale(1.0, 4, "full")
    x = np.array([-1.0, np.nextafter(-1.0, 0), np.nextafter(-1.0, -2), 2.5 * s, np.nextafter(2.5 * s, 0)])
    np.testing.assert_array_equal(F.quantize_int(x, s, 4, convention="full"), [-8, -8, -8, 2, 2])
    assert F.quantize_int(np.array([-1.0 + 1e-6]), s, 4, convention="full")[0] == -7    # a real difference still counts


def test_nvfp4_two_level_scales_beat_mxfp4():
    rng = np.random.default_rng(0)
    for W in (rng.standard_normal((256, 512)) * 0.02, rng.standard_t(3, (256, 512)) * 0.02):
        elems, local, g, x_hat = F.nvfp4(W)
        assert g == pytest.approx(448 * 6 / np.abs(W).max()) and local.shape == (256, 32, 1)
        assert np.all(np.isin(np.abs(elems), F.E2M1.grid()))
        assert G.error(W, x_hat)["rel"] < G.error(W, F.mxfp4(W)[2])["rel"]


def test_float_grid_beats_int_grid_on_heavy_tails():
    rng = np.random.default_rng(0)
    gauss, heavy = rng.standard_normal((256, 512)), rng.standard_t(3, (256, 512))
    e = lambda W, fmt: G.error(W, G.fake_quant(W, fmt=fmt, granularity="group", group_size=32, convention="full"))["rel"]
    assert e(gauss, "int4") < e(gauss, "fp4") and e(heavy, "fp4") < e(heavy, "int4")


def test_sqnr_rule_six_db_per_bit():
    assert F.sqnr_rule_db(8, np.sqrt(2)) == pytest.approx(6.02 * 8 + 1.76, abs=0.01)
    w = np.random.default_rng(0).standard_normal((128, 256))
    crest = np.mean(np.abs(w).max(1) / np.sqrt((w ** 2).mean(1)))
    sq = {b: G.error(w, G.fake_quant(w, fmt=f"int{b}", convention="full"))["sqnr_db"] for b in (4, 5, 6, 8)}
    for b, v in sq.items():
        assert abs(v - F.sqnr_rule_db(b, crest)) < 1.0
    assert 5.9 < (sq[8] - sq[4]) / 4 < 6.4


def test_pack_int4_matches_compressed_tensors():
    packed = F.pack_int4(np.array([[-8, -7, 0, 1, 2, 3, 4, 7]]))
    assert packed.dtype == np.int32 and packed[0, 0] == np.uint32(0xFCBA9810).view(np.int32)
    codes = np.random.default_rng(0).integers(-8, 8, (4096, 896))
    p = F.pack_int4(codes)
    assert p.shape == (4096, 112)
    np.testing.assert_array_equal(F.unpack_int4(p), codes)


def test_pack_fp4_low_nibble_first():
    v = np.array([[0.5, -6.0, 1.5, 0.0]])
    p = F.pack_fp4(v)
    np.testing.assert_array_equal(p, [[0x1 | (0xF << 4), 0x3 | (0x0 << 4)]])
    np.testing.assert_array_equal(F.unpack_fp4(p), v)
