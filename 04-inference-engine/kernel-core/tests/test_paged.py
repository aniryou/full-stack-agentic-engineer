"""kerncore.paged: the pool, copy-on-write, and paged attention == contiguous attention."""
import numpy as np
import pytest

from kerncore import _repo, paged

D = 8


def fill(seq, n, rng):
    for _ in range(n):
        seq.append(rng.standard_normal(D), rng.standard_normal(D))


# --- the pool ------------------------------------------------------------------------------------------
def test_pool_allocate_share_release_and_exhaustion():
    pool = paged.BlockPool(num_blocks=3, block_size=4, head_dim=D)
    a, b, c = pool.allocate(), pool.allocate(), pool.allocate()
    assert len({a, b, c}) == 3 and pool.blocks_in_use() == 3
    with pytest.raises(MemoryError):
        pool.allocate()
    pool.share(a)
    pool.release(a)
    assert pool.blocks_in_use() == 3                     # still held once
    for x in (a, b, c):
        pool.release(x)
    assert pool.blocks_in_use() == 0 and not pool.refcount


def test_block_table_allocates_on_demand():
    pool = paged.BlockPool(num_blocks=16, block_size=4, head_dim=D)
    seq = paged.PagedSequence(pool)
    fill(seq, 10, np.random.default_rng(1))
    assert len(seq.block_table) == 3 == paged.blocks_needed(10, 4)
    for t, (k, _) in enumerate(seq.log):
        np.testing.assert_array_equal(pool.K[seq.block_table[t // 4], t % 4], k)
    seq.free()
    assert pool.blocks_in_use() == 0


# --- attention through the block table ------------------------------------------------------------------
def test_gather_and_blockwise_equal_contiguous():
    for n in (1, 3, 16, 37):                              # one partial block, exact fit, many blocks
        rng = np.random.default_rng(n)
        pool = paged.BlockPool(num_blocks=16, block_size=4, head_dim=D)
        seq = paged.PagedSequence(pool)
        fill(seq, n, rng)
        q = rng.standard_normal(D)
        ref = paged.attention(q, *seq.contiguous())
        np.testing.assert_allclose(paged.paged_attention_gather(q, seq), ref, atol=1e-12)
        np.testing.assert_allclose(paged.paged_attention_blockwise(q, seq), ref, atol=1e-12)


def test_running_max_matters():
    """Block 0 scores about +800, block 1 about -800 (the practice notebook's exercise 4 case)."""
    rng = np.random.default_rng(3)
    pool = paged.BlockPool(num_blocks=4, block_size=4, head_dim=D)
    seq = paged.PagedSequence(pool)
    for t in range(8):
        sign = 1.0 if t < 4 else -1.0
        seq.append(sign * 800 / np.sqrt(D) + rng.standard_normal(D), rng.standard_normal(D))
    q = np.ones(D)
    ref = paged.attention(q, *seq.contiguous())
    np.testing.assert_allclose(paged.paged_attention_blockwise(q, seq), ref, atol=1e-12)
    with np.errstate(all="ignore"):
        wrong = paged.paged_attention_blockwise(q, seq, running_max=False)
    assert not np.all(np.isfinite(wrong))


# --- sharing and copy-on-write --------------------------------------------------------------------------
def three_samples(release_on_cow=True):
    rng = np.random.default_rng(4)
    pool = paged.BlockPool(num_blocks=64, block_size=4, head_dim=D)
    parent = paged.PagedSequence(pool)
    fill(parent, 6, rng)                                  # one full block + one half-full
    samples = [parent, parent.fork(), parent.fork()]
    assert [pool.refcount[b] for b in parent.block_table] == [3, 3]
    for s in samples:
        for _ in range(3):
            s.append(rng.standard_normal(D), rng.standard_normal(D), release_on_cow=release_on_cow)
    return pool, samples, rng


def test_cow_shares_full_blocks_and_copies_the_partial_one():
    pool, samples, rng = three_samples()
    assert len({s.block_table[0] for s in samples}) == 1
    assert len({s.block_table[1] for s in samples}) == 3
    assert pool.copies == 2                               # the last writer owns the original alone
    q = rng.standard_normal(D)
    for s in samples:
        np.testing.assert_allclose(paged.paged_attention_blockwise(q, s), paged.attention(q, *s.contiguous()),
                                   atol=1e-12)


def test_cow_releases_the_shared_block_and_the_pool_recovers():
    pool, samples, _ = three_samples()
    assert pool.audit(samples) == []
    assert pool.blocks_in_use() == len({b for s in samples for b in s.block_table}) == 7   # 9 without sharing
    for s in samples:
        s.free()
    assert pool.blocks_in_use() == 0 and not pool.refcount


def test_cow_without_release_leaks_and_the_audit_says_so():
    pool, samples, _ = three_samples(release_on_cow=False)
    problems = pool.audit(samples)
    assert problems and any("leaked" in p for p in problems)          # the original: refcount 3, no owner
    for s in samples:
        s.free()
    assert pool.blocks_in_use() > 0                       # the shared block never comes back


# --- the same algorithm as paged_attention_minimal.py ---------------------------------------------------
def test_matches_paged_attention_minimal():
    pam = _repo.load("paged_attention_minimal")
    mgr = pam.BlockManager(num_blocks=pam.NUM_BLOCKS)
    theirs = pam.Sequence(mgr)
    pool = paged.BlockPool(num_blocks=pam.NUM_BLOCKS, block_size=pam.BLOCK_SIZE, head_dim=pam.D_HEAD)
    ours = paged.PagedSequence(pool)
    rng = np.random.default_rng(7)
    for _ in range(13):
        k, v = rng.standard_normal(pam.D_HEAD), rng.standard_normal(pam.D_HEAD)
        theirs.append_kv(k, v)
        ours.append(k, v)
    q = rng.standard_normal(pam.D_HEAD)
    assert len(ours.block_table) == len(theirs.block_table) == 4
    np.testing.assert_allclose(paged.paged_attention_blockwise(q, ours), pam.paged_attention_blockwise(q, theirs),
                               atol=1e-12)
    np.testing.assert_allclose(paged.paged_attention_gather(q, ours), pam.paged_attention_gather(q, theirs),
                               atol=1e-12)
    child_t, child_o = pam.fork(theirs), ours.fork()
    for seq_t, seq_o in ((theirs, ours), (child_t, child_o)):
        k, v = rng.standard_normal(pam.D_HEAD), rng.standard_normal(pam.D_HEAD)
        seq_t.append_kv(k, v)
        seq_o.append(k, v)
    assert sorted(mgr.refcount.values()) == sorted(pool.refcount.values())
    np.testing.assert_allclose(paged.paged_attention_blockwise(q, child_o), pam.paged_attention_blockwise(q, child_t),
                               atol=1e-12)
    for s in (theirs, child_t):
        s.free()


def test_allocation_waste_contiguous_vs_paged():
    w = paged.allocation_waste([16, 100, 256], block_size=16, reserve=512)
    assert w["used"] == 372 and w["paged_slots"] == 16 + 112 + 256
    assert round(w["contiguous_waste"], 3) == round(1 - 372 / 1536, 3)
    assert w["paged_waste"] < 0.04 and w["max_paged_waste_per_seq"] == 15
