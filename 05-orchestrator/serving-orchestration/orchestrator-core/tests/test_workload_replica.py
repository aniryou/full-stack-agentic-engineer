"""Workload identity (chain hashes) and the engine model (prefix cache, eviction order, preemption, step time)."""
import dataclasses
import random

import pytest

from fleetsim import L4_8B, H100_8B, LLAMA_8B_KV, BlockPool, HashChain, Replica, agentic, arrivals, chat, expand, step_time
from fleetsim.workload import Request


def test_chain_hash_identifies_prefix_not_content():
    a = HashChain(16).extend(1, 40).extend(2, 40)          # 80 tokens -> 5 full blocks
    b = HashChain(16).extend(1, 40).extend(3, 40)          # same first 40 tokens, then different content
    assert len(a.hashes) == 5 and a.hashes[:2] == b.hashes[:2] and a.hashes[2] != b.hashes[2]
    c = HashChain(16).extend(9, 16).extend(1, 40)          # segment 1 at a different offset: no shared blocks
    assert not set(c.hashes) & set(a.hashes[:2])
    assert HashChain(16).extend(1, 15).hashes == []        # a partial block is not cacheable


def test_agentic_turn_reuses_the_previous_turns_kv():
    t1, t2 = expand([agentic(0.2, 60, seed=1, turns=(3, 3))[0]])[:2]
    assert t2.prompt > t1.prompt + t1.output                         # the whole history + a new tool result
    rep = Replica(0, L4_8B)
    rep.enqueue(t1, 0.0)
    t = _drain(rep)
    rep.enqueue(t2, t)
    _drain(rep, t)
    # every full block of turn 1's prompt + output (the last sampled token never enters the KV) is a hit
    assert t2.cached == (t1.prompt + t1.output - 1) // 16 * 16


def test_poisson_arrivals_rate_and_determinism():
    xs = arrivals(5.0, 2000, random.Random(0))
    assert abs(len(xs) / 2000 - 5.0) < 0.15
    assert arrivals([(0, 1.0), (10, 0.0)], 100, random.Random(1)) == arrivals([(0, 1.0), (10, 0.0)], 100, random.Random(1))
    assert max(arrivals([(0, 1.0), (10, 0.0)], 100, random.Random(1))) < 10


def test_engine_profile_is_spec_sheet_arithmetic():
    assert LLAMA_8B_KV == 131072                                    # 2 x 32 x 8 x 128 x 2 bytes
    assert L4_8B.kv_blocks == 2193                                  # (24e9 x 0.9 - 16e9 - 1e9) // (131072 x 16)
    assert L4_8B.weight_read_s == pytest.approx(16e9 / 0.3e12)      # 53.3 ms: the decode floor on an L4
    assert L4_8B.compute_tok_s == pytest.approx(3781.25)            # 121e12 x 0.5 / (2 x 8e9)
    assert H100_8B.compute_tok_s == pytest.approx(30906.25)


def test_step_time_is_a_roofline():
    # 16 decodes at 2,000 tokens of context: memory-bound -> 3 ms + 53.3 ms + 32,016 x 0.437 us = 70.3 ms
    assert step_time(L4_8B, 16, 16 * 2001) == pytest.approx(0.07032, abs=1e-4)
    # a 2,048-token prefill chunk: compute-bound -> 3 ms + 2048 / 3781.25 = 544.6 ms (every decode waits for it)
    assert step_time(L4_8B, 2048, 2048) == pytest.approx(0.54462, abs=1e-4)


def _req(rid, prompt, output, seg_id=7, block=16):
    ch = HashChain(block).extend(seg_id, prompt).extend(1000 + rid, output)
    return Request(rid, 0.0, prompt, output, ch.hashes, (prompt + output) // block, block)


def _drain(rep, t=0.0):
    while True:
        dur = rep.start_step(t)
        if dur is None:
            return t
        t += dur
        rep.end_step(t)


def test_prefix_cache_hit_caps_at_prompt_minus_one_token():
    rep = Replica(0, L4_8B)
    rep.enqueue(_req(0, 64, 4), 0.0)
    _drain(rep)
    again = _req(1, 64, 4)                          # the same 64-token prompt
    rep.enqueue(again, 0.0)
    _drain(rep)
    assert again.cached == 48                       # (64 - 1) // 16 = 3 blocks: the last token is always computed
    assert rep.stats["prompt"] == 128 and rep.stats["cached"] == 48


def test_freed_blocks_stay_cached_and_evict_tail_first():
    pool = BlockPool(4)
    blocks = [pool.allocate() for _ in range(3)]
    for b, h in zip(blocks, (11, 12, 13)):
        pool.register(b, h)
    pool.release(blocks)                            # refcount 0, still cached; tail block queued first
    assert pool.usage() == 0.0 and pool.match([11, 12, 13], 3) == blocks
    pool.allocate()                                 # the never-used block goes first
    pool.allocate()                                 # then the request's TAIL block (hash 13) is evicted
    assert pool.match([11, 12, 13], 3) == blocks[:2]


def test_preemption_by_recompute_when_kv_runs_out():
    tiny = dataclasses.replace(L4_8B, kv_blocks=12)          # 192 tokens of KV
    rep = Replica(0, tiny)
    a, b = _req(0, 60, 60, seg_id=1), _req(1, 60, 60, seg_id=2)
    rep.enqueue(a, 0.0)
    rep.enqueue(b, 0.0)
    _drain(rep)
    assert a.t_done and b.t_done                   # both finish...
    assert rep.stats["preempted"] >= 1 and b.preempted >= 1   # ...after the newest one was preempted and recomputed
    with pytest.raises(ValueError):
        rep.enqueue(_req(2, 300, 10), 0.0)          # can never fit: refused up front


def test_lora_request_without_a_free_slot_is_skipped_not_blocking():
    rep = Replica(0, dataclasses.replace(L4_8B, max_loras=1))
    a, b, c = _req(0, 32, 40, seg_id=1), _req(1, 32, 5, seg_id=2), _req(2, 32, 5, seg_id=3)
    a.lora, b.lora = "x", "y"                        # one slot: "y" must wait until "x" has no running request
    for r in (a, b, c):
        rep.enqueue(r, 0.0)
    _drain(rep)
    assert c.t_first < b.t_first                     # the adapter-free request overtook the stuck one
    assert b.t_first >= a.t_done and tuple(rep.loras) == ("y",)    # "x" evicted only once idle
    assert rep.stats["lora_loads"] == 2                              # x, then y: each load stalls a step


def test_chat_lengths_have_the_requested_mean():
    reqs = chat(20.0, 200, seed=4, system=0, user=200, output=150, cv=0.6)
    assert abs(sum(r.output for r in reqs) / len(reqs) - 150) < 10


def test_kv_ready_admission_reuses_the_cached_prefix():
    """P/D: a request whose KV arrives over the link still takes its cached prefix from the local pool."""
    rep = Replica(0, L4_8B)
    first = _req(0, 3200, 4, seg_id=7)
    rep.enqueue(first, 0.0)
    t = _drain(rep)
    ch = HashChain(16).extend(7, 3200).extend(99, 800).extend(98, 4)
    second = Request(1, t, 4000, 4, ch.hashes, 4004 // 16, 16)
    second.t_first = t                                   # its first token came with the transferred KV
    rep.enqueue(second, t, kv_ready=True)
    rep.start_step(t)
    prefix = rep.pool.match(second.hashes, 200)          # the 3,200 shared tokens: 200 blocks
    assert len(prefix) == 200 and rep.running[0].blocks[:200] == prefix
    assert all(rep.pool.ref[b] == 1 for b in prefix)     # shared, not duplicated
