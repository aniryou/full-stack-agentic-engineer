"""FP4: the E2M1 grid, NVFP4's two-level scale, MXFP4's power-of-two scale, and the bytes."""
import numpy as np
import pytest

from quantlab import fp4


def test_e2m1_grid_and_vllm_rounding_thresholds():
    assert fp4.E2M1_GRID.tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    x = [0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 9.0, -0.8]
    assert fp4.fp4_round(x).tolist() == [0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6, -1]    # vLLM cast_to_fp4
    codes = fp4.fp4_encode([0.5, 6.0, -6.0, -0.5])
    assert codes.tolist() == [1, 7, 15, 9] and fp4.fp4_decode(codes).tolist() == [0.5, 6, -6, -0.5]


def test_packing_first_value_in_low_nibble():
    p = fp4.pack_fp4(np.array([[1, 7, 15, 9]], dtype=np.uint8))
    assert p.tolist() == [[0x71, 0x9F]]
    assert fp4.unpack_fp4(p).tolist() == [[1, 7, 15, 9]]


def test_nvfp4_scales_are_compressed_tensors_convention():
    w = np.random.default_rng(0).standard_normal((32, 64)) * 0.02
    q = fp4.nvfp4_quantize(w)
    amax = np.abs(w).max()
    assert q.global_scale == pytest.approx(448 * 6 / amax)                   # a multiplier, 2688 / amax
    assert q.scales.shape == (32, 4) and q.values.shape == (32, 64)
    assert np.abs(q.scales).max() <= 448 and set(np.abs(q.values).ravel()) <= set(fp4.E2M1_GRID)
    assert q.bits_per_weight == pytest.approx(4.5, abs=0.02)
    back = fp4.nvfp4_from_checkpoint(fp4.nvfp4_checkpoint_tensors(q))
    np.testing.assert_allclose(back.dequantize(), q.dequantize(), rtol=1e-6)


def test_mxfp4_scale_is_a_power_of_two_per_32():
    assert fp4.mxfp4_scale_exponent([1.0, 1.7, 1.75, 6.0, 0.01]).tolist() == [125, 125, 126, 127, 118]
    w = np.random.default_rng(1).standard_normal((8, 64))
    m = fp4.mxfp4_quantize(w)
    assert m.exponents.shape == (8, 2) and m.bits_per_weight == 4.25
    assert np.abs(m.dequantize() - w).max() < np.abs(w).max() / 2


def test_finer_blocks_win_on_outlier_heavy_weights():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((64, 256)) * 0.02
    w[rng.random(w.shape) < 0.01] *= 20                                       # 1% outliers
    nv, mx = fp4.nvfp4_quantize(w).dequantize(), fp4.mxfp4_quantize(w).dequantize()
    g128 = fp4.int4_group_fake_quant(w, 128)
    assert fp4.sqnr_db(w, nv) > fp4.sqnr_db(w, mx) and fp4.sqnr_db(w, nv) > fp4.sqnr_db(w, g128)


def test_bits_and_layouts():
    assert fp4.BPW["nvfp4"] == 4.5 and fp4.BPW["mxfp4"] == 4.25 and fp4.BPW["int4-g128"] == 4.125
    assert fp4.BPW["int4-g128-asym"] == 4.15625
    lay = fp4.layout(4096, 4096, "nvfp4")
    assert lay["weight_packed"][:2] == ("U8", (4096, 2048)) and lay["weight_scale"][:2] == ("F8_E4M3", (4096, 256))
    assert fp4.layer_bytes(4096, 4096, "nvfp4") * 8 / (4096 * 4096) == pytest.approx(4.5, abs=1e-5)


def test_blackwell_model():
    t = fp4.gemm_times(4096, 14336, 4096, "B200")
    assert t["bf16"] / t["w4a4-nvfp4"] == pytest.approx(4.0, rel=0.01)          # FP4 tensor cores: 4x BF16 FLOP/s
    assert t["w4a16-nvfp4"] == pytest.approx(t["bf16"])                            # weight-only: BF16 math
    assert "w4a4-nvfp4" not in fp4.gemm_times(4096, 14336, 4096, "H100-80GB")
    assert fp4.weight_only_slower_from(14336, 4096, "B200", 1.0) is None
    assert 150 < fp4.weight_only_slower_from(14336, 4096, "B200", 0.7) < 300
