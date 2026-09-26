"""The paged forward pass computes exactly what textbook attention computes."""
import numpy as np

from minengine.model import SMALL, Batch, ModelConfig, PagedKVCache, TinyLM, encode

MODEL = TinyLM()
TEXT = encode("The engine runs a loop. Each step it picks the requests to run.")


def _run_chunks(model, cache, seqs, chunk_sizes, rng):
    """Feed several sequences through forward() in random chunks; each owns a scattered block table.
    Returns {seq index: {position: logits}} for the last token of every chunk."""
    B = cache.block_size
    free = list(rng.permutation(cache.k.shape[1] // B))
    tables = [[free.pop() for _ in range(-(-len(s) // B))] for s in seqs]
    done, got = [0] * len(seqs), {i: {} for i in range(len(seqs))}
    while any(d < len(s) for d, s in zip(done, seqs)):
        toks, pos, slots, starts, bts, ctx, who = [], [], [], [0], [], [], []
        for i, s in enumerate(seqs):
            if done[i] >= len(s):
                continue
            n = min(int(rng.choice(chunk_sizes)), len(s) - done[i])
            p = range(done[i], done[i] + n)
            toks += s[done[i]:done[i] + n]
            pos += p
            slots += [tables[i][q // B] * B + q % B for q in p]
            starts.append(starts[-1] + n)
            bts.append(tables[i])
            ctx.append(done[i] + n)
            who.append(i)
            done[i] += n
        logits = model.forward(Batch(np.array(toks), np.array(pos), np.array(slots), starts, bts, ctx), cache)
        for i, row, c in zip(who, logits, ctx):
            got[i][c - 1] = row
    return got


def test_paged_forward_equals_dense_forward():
    rng = np.random.default_rng(0)
    seqs = [TEXT[:37], TEXT[5:29], TEXT[:13]]
    cache = PagedKVCache(MODEL.cfg, num_blocks=64, block_size=4)
    got = _run_chunks(MODEL, cache, seqs, [1, 3, 7], rng)
    for i, s in enumerate(seqs):
        dense = MODEL.forward_dense(s)
        for pos, row in got[i].items():
            np.testing.assert_allclose(row, dense[pos], atol=1e-10)


def test_chunked_prefill_equals_one_shot_prefill():
    rng = np.random.default_rng(1)
    one = _run_chunks(MODEL, PagedKVCache(MODEL.cfg, 32, 4), [TEXT[:40]], [40], rng)[0][39]
    chunked = _run_chunks(MODEL, PagedKVCache(MODEL.cfg, 32, 4), [TEXT[:40]], [5, 11], rng)[0][39]
    np.testing.assert_allclose(chunked, one, atol=1e-10)


def test_kv_bytes_per_token_formula():
    assert ModelConfig().kv_bytes_per_token() == 2 * 2 * 2 * 16 * 2 == 256
    llama8b = ModelConfig(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8)
    assert llama8b.kv_bytes_per_token() == 131072                      # 128 KiB per token in bf16


def test_fp8_kv_cache_is_close_but_not_exact():
    rng = np.random.default_rng(2)
    exact = _run_chunks(MODEL, PagedKVCache(MODEL.cfg, 32, 4), [TEXT[:30]], [30], rng)[0][29]
    fp8 = _run_chunks(MODEL, PagedKVCache(MODEL.cfg, 32, 4, kv_dtype="fp8"), [TEXT[:30]], [30], rng)[0][29]
    rel = np.linalg.norm(fp8 - exact) / np.linalg.norm(exact)
    assert 0 < rel < 0.05


def test_weights_are_deterministic_and_draft_is_smaller():
    a, b = TinyLM(seed=0), TinyLM(seed=0)
    assert all(np.array_equal(a.layers[0][k], b.layers[0][k]) for k in a.layers[0])
    assert TinyLM(SMALL).num_params() < a.num_params() / 5
