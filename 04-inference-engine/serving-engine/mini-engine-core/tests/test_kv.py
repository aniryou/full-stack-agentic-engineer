"""Block pool, refcounts, LRU eviction and the hash-chained prefix cache."""
import random

from minengine.kv import KVCacheManager, block_hashes, hash_block


def _admit(kv, rid, toks):
    hits = kv.lookup(toks)
    c = len(hits) * kv.block_size
    assert kv.allocate_slots(rid, toks, c, len(toks) - c, hits) is not None
    kv.cache_blocks(rid, toks, len(toks))
    return hits


def test_block_names_chain_through_the_parent():
    a, b = [1, 2, 3, 4, 5, 6], [9, 9, 3, 4, 5, 6]            # same 2nd/3rd blocks, different 1st
    ha, hb = block_hashes(a, 2), block_hashes(b, 2)
    assert ha[1] != hb[1] and ha[2] != hb[2]                 # a name commits to everything before it
    assert block_hashes(a + [7], 2) == ha                     # a partial last block has no name
    assert block_hashes(a, 2, extra="lora-A") != ha           # an adapter id changes every name


def test_lookup_never_covers_the_last_token():
    kv = KVCacheManager(8, block_size=4)
    _admit(kv, "a", list(range(8)))                           # two full blocks, both published
    kv.free("a")
    assert len(kv.lookup(list(range(8)))) == 1                # (8 - 1) // 4: one token must be recomputed
    assert len(kv.lookup(list(range(9)))) == 2


def test_refcounts_sharing_and_freed_blocks_stay_hittable():
    kv = KVCacheManager(8, block_size=2)
    _admit(kv, "a", [1, 2, 3, 4, 5])
    hits = _admit(kv, "b", [1, 2, 3, 4, 9])
    assert hits == kv.tables["a"][:2]
    assert [kv.blocks[b].ref_cnt for b in hits] == [2, 2]
    kv.free("a")
    kv.free("b")
    assert kv.num_free_blocks == 8 and len(kv.lookup([1, 2, 3, 4, 7])) == 2   # free, yet still a hit
    kv.check()


def test_lru_evicts_the_tail_of_a_freed_request_first():
    kv = KVCacheManager(3, block_size=2)
    _admit(kv, "a", [1, 2, 3, 4, 5, 6])                       # blocks: head, middle, tail
    head, mid, tail = kv.tables["a"]
    kv.free("a")
    assert list(kv.free_queue) == [tail, mid, head]           # eviction order
    _admit(kv, "b", [7, 7])                                   # needs one block: the tail goes
    assert kv.tables["b"] == [tail] and kv.stats.evictions == 1
    assert len(kv.lookup([1, 2, 3, 4, 0])) == 2               # the shared head survives


def test_allocation_failure_changes_nothing():
    kv = KVCacheManager(4, block_size=2)
    _admit(kv, "a", [1, 2, 3, 4, 5])                          # 3 blocks
    before = (list(kv.free_queue), dict(kv.tables))
    assert kv.allocate_slots("b", [6, 7, 8, 9], 0, 4) is None
    assert (list(kv.free_queue), dict(kv.tables)) == before
    kv.check()


def test_hit_accounting_is_in_tokens():
    kv = KVCacheManager(16, block_size=4)
    _admit(kv, "a", list(range(10)))
    _admit(kv, "b", list(range(10)))                          # hits 2 blocks = 8 tokens of 10
    assert (kv.stats.queries, kv.stats.hits) == (20, 8)
    assert kv.stats.hit_rate == 0.4


def _oracle_run(hash_fn, seed=0, rounds=400):
    """Random prompts over a 3-letter alphabet (so blocks repeat under different prefixes). An
    oracle remembers which full token prefix each block's K/V was computed from; every hit is
    checked against it. Returns the number of hits that served the wrong prefix."""
    kv, rng, B = KVCacheManager(12, block_size=2, hash_fn=hash_fn), random.Random(seed), 2
    oracle, live, wrong = {}, [], 0
    for step in range(rounds):
        if live and (rng.random() < 0.4 or kv.num_free_blocks < 5):
            kv.free(live.pop(rng.randrange(len(live))))
            continue
        toks = [rng.randrange(3) for _ in range(rng.randrange(3, 9))]
        hits = kv.lookup(toks)
        wrong += sum(oracle.get(b) != tuple(toks[:(i + 1) * B]) for i, b in enumerate(hits))
        rid = f"r{step}"
        if kv.allocate_slots(rid, toks, len(hits) * B, len(toks) - len(hits) * B, hits) is None:
            continue
        kv.cache_blocks(rid, toks, len(toks))
        for i in range(len(hits), len(toks) // B):            # blocks this request computed itself
            oracle[kv.tables[rid][i]] = tuple(toks[:(i + 1) * B])
        live.append(rid)
        if hash_fn is hash_block:
            kv.check()                                        # (a broken hash even adopts one block twice)
    return wrong


def test_prefix_cache_never_serves_a_block_whose_prefix_differs():
    assert _oracle_run(hash_block) == 0


def test_a_name_without_the_parent_would_serve_wrong_kv():
    parentless = lambda parent, tokens, extra=None: hash_block(None, tokens, extra)
    assert _oracle_run(parentless) > 0                        # why the chain is not optional
