"""Pseudo-tokens, chained block hashes and the approximate prefix index."""
import random

from igwlab.router.prefix import (PrefixIndex, PrefixMatch, block_hashes, effective_block_size, max_blocks_for)
from igwlab.router.tokens import estimate_tokens, pack_bytes, request_bytes


def test_pack_bytes_is_4_bytes_per_token_little_endian():
    assert pack_bytes(b"") == []
    assert pack_bytes(b"abcd") == [int.from_bytes(b"abcd", "little")]
    assert pack_bytes(b"abcde") == [int.from_bytes(b"abcd", "little"), int.from_bytes(b"e\x00\x00\x00", "little")]


def test_chat_bytes_are_tools_then_role_content_pairs():
    body = {"tools": [{"b": 1, "a": 2}], "messages": [{"role": "system", "content": "S"},
                                                       {"role": "user", "content": [{"type": "text", "text": "U"}]}]}
    assert request_bytes(body) == b'[{"a":2,"b":1}]' + b"systemS" + b"userU"
    assert len(estimate_tokens({"prompt": "x" * 400})) == 100          # 400 bytes -> 100 pseudo-tokens


def test_chained_hashes_share_exactly_the_common_prefix():
    a = list(range(1000))
    b = list(range(640)) + [7] * 360                 # identical for the first 640 tokens = 10 blocks of 64
    ha, hb = block_hashes(a, 64), block_hashes(b, 64)
    assert len(ha) == 16                             # ceil(1000/64): the last partial block is hashed too
    assert ha[:10] == hb[:10] and ha[10] != hb[10]
    assert block_hashes(a, 64, model="m1")[0] != block_hashes(a, 64, model="m2")[0]   # model seeds the chain
    assert block_hashes(a, 64, cache_salt="tenant-a")[0] != ha[0]
    assert len(block_hashes(a, 64, max_blocks=3)) == 3


def test_block_size_floor_and_caps():
    assert effective_block_size(16, None) == 64                  # upstream floor of 64 tokens
    assert effective_block_size(16, 128) == 128                  # auto-tuned from the engine
    assert effective_block_size(256, 16, auto_tune=False) == 256
    assert max_blocks_for(64) == 131072 // 64 == 2048            # the token cap wins when > 0
    assert max_blocks_for(64, 0, 100) == 100 and max_blocks_for(64, 0, 0) is None


def test_match_counts_leading_blocks_per_endpoint():
    idx = PrefixIndex()
    h = block_hashes(list(range(64 * 4)), 64)
    idx.add(h[:3], "A")
    idx.add(h[:2], "B")
    assert idx.match(h) == {"A": 3, "B": 2}
    assert PrefixMatch(3, 4, 64).ratio == 0.75 and PrefixMatch(3, 4, 64).matched_tokens == 192
    assert idx.match(block_hashes([9] * 64, 64)) == {}


def test_lru_evicts_the_tail_first_and_an_oversized_prompt_keeps_its_head():
    idx = PrefixIndex(capacity_per_endpoint=4)
    h = block_hashes(list(range(64 * 6)), 64)
    idx.add(h, "A")                                   # 6 blocks into a 4-entry LRU
    assert idx.match(h) == {"A": 4}                   # head kept, tail evicted
    other = block_hashes([5] * 128, 64)
    idx.add(other, "A")                               # 2 more entries evict the 2 deepest blocks
    assert idx.longest_prefix(h, "A") == 2 and idx.block_counts() == {"A": 4}


def test_head_first_insertion_of_v0_10_evicts_the_head_and_the_greedy_scan_undercounts():
    h = block_hashes(list(range(64 * 6)), 64)
    lab = PrefixIndex(capacity_per_endpoint=4)                     # tail-first (llm-d-router main)
    v010 = PrefixIndex(capacity_per_endpoint=4, head_first=True)   # llm-d-router v0.10.0
    lab.add(h, "A")
    v010.add(h, "A")
    assert lab.match(h) == {"A": 4}                   # keeps blocks 0-3: a 4-block match
    assert v010.block_counts() == {"A": 4}            # also holds 4 blocks (2-5) ...
    assert v010.match(h) == {}                        # ... but block 0 is gone, so the scan stops at once
    roomy = PrefixIndex(head_first=True)              # with room to spare, the order does not matter
    roomy.add(h, "A")
    assert roomy.match(h) == {"A": 6}


def test_ttl_expires_entries(monkeypatch):
    now = [100.0]
    idx = PrefixIndex(ttl_s=10, clock=lambda: now[0])
    h = block_hashes(list(range(256)), 64)
    idx.add(h, "A")
    now[0] = 105.0
    assert idx.match(h) == {"A": 4}
    now[0] = 111.0
    assert idx.match(h) == {}                         # expired entries are purged lazily on lookup


def test_greedy_union_match_equals_per_endpoint_longest_prefix():
    """The lab's tail-first insertion (llm-d-router main; v0.10.0 inserts head-first) keeps every
    endpoint's chain hole-free, so the greedy scan gives the same count as a per-endpoint
    contiguous match (randomized check)."""
    rng = random.Random(1)
    prompts = [[p] * 64 * 3 + list(range(rng.randrange(0, 64 * 8))) for p in range(5)]
    idx = PrefixIndex(capacity_per_endpoint=12)
    for _ in range(300):
        toks = rng.choice(prompts)[: rng.randrange(64, 64 * 11)]
        idx.add(block_hashes(toks, 64), rng.choice("ABC"))
        q = block_hashes(rng.choice(prompts), 64)
        greedy = idx.match(q)
        for ep in "ABC":
            assert greedy.get(ep, 0) == idx.longest_prefix(q, ep)
