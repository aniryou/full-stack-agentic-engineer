"""Pins the formulas in fa_calculators.py to hand-computed values quoted in
flash-attention-deep-dive.md. Offline, CPU only:  python3 -m pytest -q test_fa_calculators.py
"""
import math

import numpy as np
import pytest

import fa_calculators as fc


# --- section 1: the naive schedule on the roofline ---------------------------------------
def test_naive_traffic_n4k_d128_bf16():
    t = fc.naive_traffic(4096, 128)
    # 4*N^2*b + 4*N*d*b = 4*16,777,216*2 + 4*4096*128*2
    assert t["bytes"] == 134_217_728 + 4_194_304 == 138_412_032
    assert t["s_matrix_bytes"] == 33_554_432
    assert t["flops"] == 4 * 4096 * 4096 * 128 == 8_589_934_592
    assert t["intensity"] == pytest.approx(62.06, abs=0.01)


def test_naive_intensity_tends_to_d_over_b():
    # N*d / (b*(N+d)) -> d/b: independent of N, batch and heads
    assert fc.naive_traffic(1 << 20, 128)["intensity"] == pytest.approx(64.0, rel=1e-3)
    assert fc.naive_traffic(1 << 20, 64)["intensity"] == pytest.approx(32.0, rel=1e-3)


def test_fp32_scores_or_extra_passes_double_the_quadratic_term():
    base = fc.naive_traffic(4096, 128)
    assert fc.naive_traffic(4096, 128, b_s=4)["bytes"] == 272_629_760
    assert fc.naive_traffic(4096, 128, extra_elementwise_passes=1)["bytes"] == base["bytes"] + 2 * 4096**2 * 2


def test_roofline_naive_is_memory_bound_on_h100_and_l4():
    t = fc.naive_traffic(4096, 128)
    h = fc.roofline(t["flops"], t["bytes"], "H100")
    assert h["bound"] == "memory"
    assert h["t_memory_s"] * 1e6 == pytest.approx(41.32, abs=0.01)
    assert h["t_compute_s"] * 1e6 == pytest.approx(8.68, abs=0.01)
    assert h["fraction_of_peak"] == pytest.approx(0.21, abs=0.005)
    l4 = fc.roofline(t["flops"], t["bytes"], "L4")
    assert l4["t_memory_s"] * 1e6 == pytest.approx(461.4, abs=0.1)
    t32 = fc.naive_traffic(32768, 128)
    assert fc.roofline(t32["flops"], t32["bytes"], "H100")["t_memory_s"] * 1e3 == pytest.approx(2.574, abs=1e-3)


def test_ridge_points():
    assert fc.DEVICES["H100"].ridge == pytest.approx(295.3, abs=0.1)
    assert fc.DEVICES["L4"].ridge == pytest.approx(403.3, abs=0.1)
    assert fc.DEVICES["A100"].ridge == pytest.approx(153.0, abs=0.1)


# --- section 3: tiled schedules ------------------------------------------------------------
def test_compulsory_traffic_makes_flash_compute_bound():
    c = fc.compulsory_bytes(4096, 128)
    assert c == 4 * 4096 * 128 * 2 + 4 * 4096 == 4_210_688
    assert fc.roofline(fc.attention_flops(4096, 4096, 128), c, "H100")["bound"] == "compute"


def test_fa1_and_fa2_schedule_traffic_no_l2_model():
    assert fc.flash_traffic(4096, 128, 128, 128, schedule="fa1")["bytes"] == 104_857_600
    f2 = fc.flash_traffic(4096, 128, 128, 64, schedule="fa2")
    assert f2["parts"]["K,V re-reads"] == 32 * 2 * 4096 * 128 * 2 == 67_108_864
    assert f2["bytes"] == 69_222_400
    # the K/V term gives an intensity of ~B_r FLOP/B in bf16 (2*B_r/b), whatever B_c is
    big = fc.flash_traffic(1 << 18, 128, 128, 64, schedule="fa2")
    assert big["intensity"] == pytest.approx(128, rel=0.01)
    assert fc.flash_traffic(4096, 128, 128, 64, causal=True)["bytes"] == 36_716_544


def test_smem_footprints_match_fa2_source_comments():
    assert fc.tile_smem_bytes(128, 64, 128) == 64 * 1024     # A100/H100 d=128
    assert fc.tile_smem_bytes(128, 32, 128) == 48 * 1024     # sm86/89: "128 x 32 (48 KB smem)"
    assert fc.tile_smem_bytes(128, 64, 256) == 128 * 1024    # A100 d=256: "128 x 64 (128KB smem)"
    assert fc.tile_smem_bytes(64, 64, 256) == 96 * 1024      # H100 d=256: "64 x 64 (96KB smem)"


def test_attention_flop_conventions():
    f = fc.attention_flops(4096, 4096, 128)
    assert fc.attention_flops(4096, 4096, 128, causal=True) == f / 2
    assert fc.attention_flops(4096, 4096, 128, pass_="bwd") == 2.5 * f
    assert fc.attention_flops(4096, 4096, 128, pass_="fwd_bwd") == 3.5 * f
    assert fc.causal_fraction(8, 8) == 36 / 64
    assert fc.causal_fraction(4096, 4096) == pytest.approx(0.500122, abs=1e-6)


# --- section 4: causal and sliding-window tile counts ------------------------------------
def test_causal_tiles():
    assert fc.causal_tiles(4096, 4096, 128, 128) == (528, 32, 1024)
    assert fc.causal_tiles(4096, 4096, 128, 64) == (1056, 64, 2048)
    assert fc.causal_tiles(512, 512, 128, 64) == (20, 8, 32)          # the diagram
    v, _, full = fc.causal_tiles(32768, 32768, 128, 64)
    assert (v, full) == (65792, 131072)


def test_window_tiles_hand_count():
    # m < 32: 2m+2 tiles; m >= 32: 66 tiles  ->  1,056 + 224*66
    assert fc.window_tiles(32768, 4096, 128, 64) == 1056 + 224 * 66 == 15_840


# --- section 6: split-KV decode ----------------------------------------------------------
def test_split_heuristic_matches_upstream_comment_example():
    # hopper/heuristics.h: "batch * n_heads = 48 and we have 108 SMs, having 2 splits
    # (efficiency = 0.89) is better than having 3 splits (efficiency = 0.67)"
    assert fc.num_splits_heuristic(48, 108, 64, 128) == 2
    assert fc.num_splits_heuristic(200, 216, 64, 128) == 1          # already >= 0.8 * SMs


def test_fa2_decode_splits_worked_examples():
    h = fc.fa2_decode_splits(1, 32, 8, 32768, 128, 132)
    assert (h["ctas_without_split"], h["splits"], h["ctas"], h["blocks_per_split"]) == (8, 29, 232, 9)
    assert fc.fa2_decode_splits(1, 32, 8, 32768, 128, 108)["splits"] == 24
    assert fc.fa2_decode_splits(1, 32, 8, 32768, 128, 58)["splits"] == 13
    assert fc.fa2_decode_splits(8, 32, 8, 32768, 128, 132)["splits"] == 4
    assert fc.fa2_decode_splits(64, 32, 8, 4096, 128, 132)["splits"] == 1


def test_decode_bytes_and_intensity():
    assert fc.kv_bytes_per_token(32, 8, 128) == 131_072                   # Llama-3-8B-like, bf16
    assert fc.decode_attention_bytes(32 * 8192, 32, 8, 128) == 34_359_738_368
    assert [fc.decode_intensity(g) for g in (1, 4, 8)] == [1.0, 4.0, 8.0]
    assert fc.mla_decode_intensity() == pytest.approx(241.78, abs=0.01)
    assert fc.mla_decode_intensity(b=1) == pytest.approx(483.56, abs=0.01)


def test_prefill_share_and_ring_threshold():
    lin = 4096 * 4096 * 2 + 4096 * 1024 * 2 + 3 * 4096 * 14336
    assert lin == 218_103_808
    assert fc.prefill_attention_share(53_248, 32, 128, lin) == pytest.approx(1.0)
    assert fc.prefill_attention_share(8192, 32, 128, lin) == pytest.approx(0.154, abs=1e-3)
    assert fc.ring_min_tokens_per_gpu(989.4, 450) == pytest.approx(2198.7, abs=0.1)


# --- section 2: online-softmax algebra ---------------------------------------------------
S_BLOCKS = [[4.0, 8.0], [6.0, 12.0]]
V_BLOCKS = [[[1, 0], [0, 1]], [[1, 1], [2, -1]]]


@pytest.mark.parametrize("use_exp2", [False, True])
def test_two_block_worked_example(use_exp2):
    steps, o, lse = fc.online_trace(S_BLOCKS, V_BLOCKS, use_exp2=use_exp2, scale=0.5)
    assert steps[0]["l"] == pytest.approx(1.135335, abs=1e-6)
    assert steps[1]["alpha"] == pytest.approx(math.exp(-2), abs=1e-12)
    assert steps[1]["l"] == pytest.approx(1.203438, abs=1e-6)
    np.testing.assert_allclose(steps[1]["o_unnorm"], [2.068103, -0.814878], atol=1e-6)
    np.testing.assert_allclose(o, [1.718495, -0.677125], atol=1e-6)
    assert lse == pytest.approx(6.185182, abs=1e-6)
    ref, ref_lse = fc.reference_attention_row(np.array([4, 8, 6, 12.0]),
                                              np.array([[1, 0], [0, 1], [1, 1], [2, -1.0]]), 0.5)
    np.testing.assert_allclose(o, ref, atol=1e-12)
    assert lse == pytest.approx(ref_lse, abs=1e-12)


def test_lse_form_merge_of_the_example():
    o1, l1 = fc.block_state(np.array([2.0, 4.0]), np.array([[1, 0], [0, 1.0]])).finalize()
    o2, l2 = fc.block_state(np.array([3.0, 6.0]), np.array([[1, 1], [2, -1.0]])).finalize()
    assert (l1, l2) == (pytest.approx(4.126928, abs=1e-6), pytest.approx(6.048587, abs=1e-6))
    o, lse = fc.merge_lse(o1, l1, o2, l2)
    np.testing.assert_allclose(o, [1.718495, -0.677125], atol=1e-6)
    assert lse == pytest.approx(6.185182, abs=1e-6)


def test_merge_is_associative_commutative_with_identity():
    rng = np.random.default_rng(0)
    states = [fc.block_state(rng.normal(size=5) * 3, rng.normal(size=(5, 3))) for _ in range(3)]
    a, b, c = states
    left, right = fc.merge(fc.merge(a, b), c), fc.merge(a, fc.merge(b, c))
    np.testing.assert_allclose(left.o / left.l, right.o / right.l, rtol=1e-12)
    ab, ba = fc.merge(a, b), fc.merge(b, a)
    np.testing.assert_allclose(ab.o, ba.o, rtol=1e-12)
    e = fc.merge(a, fc.empty_state(3))
    np.testing.assert_allclose(e.o, a.o)
    assert (e.m, e.l) == (a.m, a.l)
    both_empty = fc.merge(fc.empty_state(3), fc.empty_state(3))
    assert both_empty.l == 0.0 and not np.isnan(both_empty.o).any()


@pytest.mark.parametrize("n_splits", [1, 2, 7, 64])
def test_split_kv_decode_matches_reference(n_splits):
    rng = np.random.default_rng(n_splits)
    q, k, v = rng.normal(size=64), rng.normal(size=(1000, 64)) * 2, rng.normal(size=(1000, 64))
    o, lse, parts = fc.split_kv_decode(q, k, v, n_splits)
    ref, ref_lse = fc.reference_attention_row(k @ q, v, 1 / 8)
    np.testing.assert_allclose(o, ref, rtol=1e-10, atol=1e-12)
    assert lse == pytest.approx(ref_lse, abs=1e-10)
    assert len(parts) == n_splits


# --- section 9: FP8 emulation and incoherent processing -----------------------------------
def test_round_to_e4m3_known_values():
    x = np.array([1.0625, 1.1875, -3.3, 500.0, 2.0**-9, 2.0**-10, 0.0, 448.0, 0.3])
    got = fc.round_to_e4m3(x)
    np.testing.assert_array_equal(got[:8], [1.0, 1.25, -3.25, 448.0, 2.0**-9, 0.0, 0.0, 448.0])
    assert got[8] == pytest.approx(0.3125)            # 0.3 in [0.25, 0.5): step 2^-5


def test_hadamard_rotation_preserves_scores():
    m = fc.random_hadamard(64, seed=3)
    np.testing.assert_allclose(m @ m.T, np.eye(64), atol=1e-12)
    rng = np.random.default_rng(1)
    q, k = rng.normal(size=(4, 64)), rng.normal(size=(6, 64))
    np.testing.assert_allclose((q @ m) @ (k @ m).T, q @ k.T, atol=1e-10)
    with pytest.raises(ValueError):
        fc.hadamard(48)
