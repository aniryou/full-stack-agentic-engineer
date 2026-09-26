"""kerncore.kv: sizing, and the tiny decoder's headline -- cached decode == recomputing the prefix."""
import numpy as np
import pytest

from kerncore import flash, kv

PROMPT = np.random.default_rng(0).integers(0, 256, 16)


@pytest.fixture(scope="module")
def small():
    return kv.TinyDecoder(kv.random_params(vocab=256, d_model=64, n_heads=4, n_layers=2, seed=0))


@pytest.fixture(scope="module")
def run(small):
    return kv.compare(small, PROMPT, 24)


# --- sizing ------------------------------------------------------------------------------------------
def test_formula_matches_fa_calculators_and_per_layer_sum():
    for shape in kv.MODELS.values():
        for b in (1, 2):
            per_tok = kv.kv_bytes_per_token(shape.n_layers, shape.n_kv_heads, shape.head_dim, b)
            assert per_tok == flash.fa.kv_bytes_per_token(shape.n_layers, shape.n_kv_heads, shape.head_dim, b)
            assert per_tok == shape.n_layers * kv.kv_bytes_per_token_per_layer(shape.n_kv_heads, shape.head_dim, b)
            assert kv.kv_cache_bytes(shape.n_layers, shape.n_kv_heads, shape.head_dim, 1000, 3, b) == per_tok * 3000


def test_mha_gqa_mqa_table():
    rows = {(r["kind"], r["dtype"]): r for r in kv.kv_table(kv.MODELS["llama-3-8b"])}
    assert rows[("MHA", "fp16")]["kv_heads"] == 32 and rows[("GQA", "fp16")]["kv_heads"] == 8
    assert rows[("MQA", "fp16")]["kv_heads"] == 1
    assert rows[("GQA", "fp16")]["per_layer"] == 4096 and rows[("GQA", "fp8")]["per_layer"] == 2048
    assert rows[("MHA", "fp16")]["per_token"] == 32 * rows[("MQA", "fp16")]["per_token"]
    with pytest.raises(ValueError):
        kv.kv_heads_for("GQA", 32, group=5)


def test_fmt_bytes_units():
    assert kv.fmt_bytes(131_072) == "128 KiB"
    assert kv.fmt_bytes(819_200) == "800 KiB"
    assert kv.fmt_bytes(kv.GiB) == "1.00 GiB (1.07 GB)"
    assert kv.fmt_bytes(128 * kv.MiB) == "128.00 MiB (134.22 MB)"


def test_sessions_upper_bound_and_reserve():
    per = kv.kv_bytes_per_token(32, 8, 128)
    assert kv.sessions_per_gpu(24e9, 16e9, 8192, per) == 7
    assert kv.sessions_per_gpu(24e9, 16e9, 8192, per, reserve_bytes=2e9) == 5
    assert kv.sessions_per_gpu(16e9, 16.06e9, 8192, per) == 0          # the weights do not fit: none


# --- the tiny decoder ----------------------------------------------------------------------------------
def test_cached_decode_is_identical_to_recompute(run):
    assert run["same_ids"] and run["identical"], run["max_logit_diff"]
    assert run["max_logit_diff"] <= 1e-9


def test_one_decode_step_equals_last_row_of_a_full_forward(small):
    seq = np.array([5, 17, 99, 3])
    full, _ = small.forward(seq)
    _, cache = small.forward(seq[:3])
    step, _ = small.forward(seq[3:], cache)
    np.testing.assert_allclose(step[0], full[-1], atol=1e-10)


def test_chunk_on_top_of_the_cache_and_cache_contents(small):
    """A multi-token chunk after a cached prefix (chunked prefill) needs the mask by absolute position AND
    the cache in time order; a reversed concatenation passes one-token decode but fails here."""
    seq = np.array([1, 2, 3, 4, 5, 6])
    full, full_cache = small.forward(seq)
    _, cache = small.forward(seq[:2])
    chunk, grown = small.forward(seq[2:], cache)
    np.testing.assert_allclose(chunk, full[2:], atol=1e-10)
    for (k1, v1), (k2, v2) in zip(grown, full_cache):
        np.testing.assert_allclose(k1, k2, atol=1e-12)
        np.testing.assert_allclose(v1, v2, atol=1e-12)


def test_causal_future_token_does_not_change_the_past(small):
    a, b = np.array([7, 8, 9, 10, 11]), np.array([7, 8, 9, 10, 12])
    la, _ = small.forward(a)
    lb, _ = small.forward(b)
    np.testing.assert_allclose(la[:-1], lb[:-1], atol=1e-12)
    assert not np.allclose(la[-1], lb[-1])


def test_cache_bytes_equal_the_formula_and_gqa_shrinks_it():
    for kvh in (4, 2, 1):                                                # MHA, GQA group 2, MQA
        m = kv.TinyDecoder(kv.random_params(64, 64, 4, 3, n_kv_heads=kvh, seed=1))
        _, cache = m.forward(np.arange(10))
        assert kv.cache_nbytes(cache) == kv.kv_cache_bytes(3, kvh, 16, 10, bytes_per=8)   # float64
    g = kv.TinyDecoder(kv.random_params(64, 64, 4, 2, n_kv_heads=2, seed=2))
    r = kv.compare(g, np.arange(8), 10)
    assert r["identical"]


def test_step_cost_grows_with_the_cache_not_the_prefix(run, small):
    cfg = small.cfg
    cached = kv.per_step_costs(run["cached_costs"])[1:]                 # drop the prefill
    naive = kv.per_step_costs(run["naive_costs"])
    slope = 2 * cfg["n_heads"] * cfg["head_dim"] * cfg["n_layers"]      # scores + PV for one more key
    assert np.all(np.diff(cached) == slope)                             # linear in cache length, small slope
    per_token = cached[0] - slope * (len(PROMPT) + 1)                   # cost of one token with no keys
    assert np.all(naive > (len(PROMPT) - 1) * per_token)                # naive reprocesses the whole prefix
    assert naive[-1] / cached[-1] > len(PROMPT)
    reads = kv.per_step_costs(run["cached_costs"], "kv_bytes_attended")[1:]
    assert np.all(np.diff(reads) == 2 * cfg["n_kv_heads"] * cfg["head_dim"] * 8 * cfg["n_layers"])


def test_token_passes_of_the_worked_notebook():
    naive, cached = kv.token_passes(16, 256)
    assert (naive, cached) == (36_736, 272) and round(naive / cached) == 135


def test_max_len_is_enforced():
    m = kv.TinyDecoder(kv.random_params(16, 16, 2, 1, max_len=8))
    with pytest.raises(ValueError):
        m.forward(np.arange(9) % 16)
