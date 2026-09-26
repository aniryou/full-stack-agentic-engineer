"""kerncore.flash: tiled == exact, causal tile skipping, the -inf trap, and the counters against fa_calculators."""
import numpy as np
import pytest

from kerncore import _repo, flash

fa = flash.fa


def qkv(n, d=8, seed=0, n_k=None):
    rng = np.random.default_rng(seed)
    n_k = n if n_k is None else n_k
    return rng.standard_normal((n, d)), rng.standard_normal((n_k, d)), rng.standard_normal((n_k, d))


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_equals_exact(causal):
    for bm, bn in ((1, 1), (2, 3), (4, 4), (8, 3), (16, 16)):     # ragged tails, tiles larger than N
        Q, K, V = qkv(13, seed=bm * 10 + bn)
        O_ref, lse_ref = flash.attention(Q, K, V, causal=causal)
        O, lse, st = flash.flash_attention(Q, K, V, bm, bn, causal=causal)
        np.testing.assert_allclose(O, O_ref, atol=1e-12)
        np.testing.assert_allclose(lse, lse_ref, atol=1e-12)
        assert st["nan_rows"] == 0


def test_exact_attention_lse_matches_a_direct_logsumexp():
    Q, K, V = qkv(5)
    _, lse = flash.attention(Q, K, V)
    S = Q @ K.T / np.sqrt(8)
    np.testing.assert_allclose(lse, np.log(np.exp(S).sum(axis=1)), atol=1e-12)


def test_forward_loop_needs_no_guard_first_tile_holds_key_0():
    """Deep dive §11.2: walking key tiles forward from key 0, every row's first tile holds a key it may see."""
    Q, K, V = qkv(512)
    O_ref, _ = flash.attention(Q, K, V, causal=True)
    for bm, bn in ((128, 64), (64, 128), (32, 32), (48, 16)):
        with np.errstate(all="raise"):                    # any exp(-inf - -inf) would raise here
            O, _, st = flash.flash_attention(Q, K, V, bm, bn, causal=True, order="forward", guard=False)
        assert st["nan_rows"] == 0
        np.testing.assert_allclose(O, O_ref, atol=1e-12)


def test_backward_loop_without_the_guard_gives_nan_rows():
    """FA2 walks from the diagonal down; with B_r > B_c a row's first tile can be fully masked."""
    Q, K, V = qkv(512)
    O_ref, _ = flash.attention(Q, K, V, causal=True)
    _, _, st = flash.flash_attention(Q, K, V, 128, 64, causal=True, order="backward", guard=False)
    assert st["nan_rows"] > 0
    O, _, st = flash.flash_attention(Q, K, V, 128, 64, causal=True, order="backward", guard=True)
    assert st["nan_rows"] == 0
    np.testing.assert_allclose(O, O_ref, atol=1e-12)


def test_rows_with_no_visible_key_need_the_guard():
    """Bottom-right alignment with n_q > n_k: the first n_q - n_k rows see no key at all."""
    Q, K, V = qkv(10, n_k=6)
    O_ref, lse_ref = flash.attention(Q, K, V, causal=True)
    O, lse, st = flash.flash_attention(Q, K, V, 4, 2, causal=True, guard=True)
    np.testing.assert_allclose(O, O_ref, atol=1e-12)
    assert np.all(O[:4] == 0) and np.all(np.isneginf(lse[:4])) and np.allclose(lse[4:], lse_ref[4:])


def test_mask_only_is_exact_but_visits_every_tile():
    Q, K, V = qkv(64)
    O_skip, _, s1 = flash.flash_attention(Q, K, V, 16, 8, causal=True)
    O_all, _, s2 = flash.flash_attention(Q, K, V, 16, 8, causal=True, skip_masked_tiles=False)
    np.testing.assert_allclose(O_skip, O_all, atol=1e-12)
    assert s2["visited"] == s2["grid"] == 32 and s1["visited"] < s2["visited"]


def test_counters_match_fa_calculators():
    for n, bm, bn in ((512, 128, 64), (512, 64, 128), (500, 128, 64), (256, 32, 32), (96, 64, 48)):
        Q, K, V = qkv(n, d=16)
        for causal in (False, True):
            _, _, st = flash.flash_attention(Q, K, V, bm, bn, causal=causal)
            model = flash.cost_model(n, 16, bm, bn, causal=causal)
            assert (st["visited"], st["masked"], st["grid"]) == (model["visited"], model["masked"], model["grid"])
            assert st["bytes"] == model["bytes"] == fa.flash_traffic(n, 16, bm, bn, schedule="fa2",
                                                                     causal=causal)["bytes"]
            assert model["bytes"] < model["naive_bytes"]


def test_matches_flash_attention_minimal():
    fam = _repo.load("flash_attention_minimal")
    Q, K, V = qkv(12, d=4, seed=3)
    for causal in (False, True):
        O_theirs, L_theirs = fam.flash_attention(Q, K, V, block_q=4, block_kv=3, causal=causal)
        O, lse, _ = flash.flash_attention(Q, K, V, 4, 3, causal=causal)
        np.testing.assert_allclose(O, O_theirs, atol=1e-12)
        np.testing.assert_allclose(lse, L_theirs, atol=1e-12)


def test_online_softmax_steps_match_fa_online_trace():
    s_blocks = [[4.0, 8.0], [6.0, 12.0]]
    v_blocks = [[[1, 0], [0, 1]], [[1, 1], [2, -1]]]
    _, o_fa, lse_fa = fa.online_trace(s_blocks, v_blocks, scale=0.5)
    _, final = flash.online_softmax_steps([np.array(b) * 0.5 for b in s_blocks], v_blocks)
    np.testing.assert_allclose(final["o"], o_fa, atol=1e-12)
    assert abs(final["lse"] - lse_fa) < 1e-12


def test_bad_order_is_rejected():
    with pytest.raises(ValueError):
        flash.flash_attention(*qkv(4), order="sideways")
