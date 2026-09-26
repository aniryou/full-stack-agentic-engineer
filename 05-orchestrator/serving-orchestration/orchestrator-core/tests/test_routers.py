"""Routers: each algorithm pinned to its defining property, scorers to the upstream formulas."""
import math
from collections import Counter

import pytest

from fleetsim import (L4_8B, ConsistentHashBoundedLoad, HashChain, HashRing, KVCacheUtilizationScorer,
                      LeastOutstanding, LoraAffinityFilter, PowerOfTwo, PrefixAffinityFilter, PrefixCacheScorer,
                      PrefixHash, PreciseIndex, QueueScorer, Replica, RoundRobin, WeightedScorer)
from fleetsim.workload import Request


class Stub:
    """A replica as a router sees it: an id, a profile and a /metrics scrape."""
    def __init__(self, rid, waiting=0, kv=0.0, loras=()):
        self.rid, self.p = rid, L4_8B
        self._m = {"waiting": waiting, "running": 0, "kv": kv, "loras": tuple(loras)}

    def metrics(self, now):
        return self._m


def req(rid=0, seg=5, prompt=160, lora=None):
    ch = HashChain(16).extend(seg, prompt)
    return Request(rid, 0.0, prompt, 1, ch.hashes, prompt // 16, 16, lora=lora)


def test_round_robin_cycles():
    reps, rr = [Stub(i) for i in range(3)], RoundRobin()
    assert [rr.pick(req(), reps, 0).rid for _ in range(6)] == [0, 1, 2, 0, 1, 2]


def test_least_outstanding_uses_its_own_counters():
    reps, lo = [Stub(i) for i in range(3)], LeastOutstanding()
    for _ in range(2):
        lo.on_dispatch(req(), reps[0], 0)
    lo.on_dispatch(req(), reps[1], 0)
    assert lo.pick(req(), reps, 0).rid == 2


def test_power_of_two_picks_the_lighter_of_two_random_replicas():
    reps = [Stub(i, waiting=w) for i, w in enumerate([0, 5, 5, 5])]
    p2c = PowerOfTwo(load="waiting", seed=1)
    share = Counter(p2c.pick(req(), reps, 0).rid for _ in range(20000))[0] / 20000
    assert share == pytest.approx(0.5, abs=0.02)            # 1 - C(3,2)/C(4,2): chosen whenever it is sampled
    heavy = [Stub(i, waiting=w) for i, w in enumerate([9, 1, 1, 1])]
    assert all(p2c.pick(req(), heavy, 0).rid != 0 for _ in range(2000))   # never the unique maximum


def test_consistent_hashing_moves_about_one_nth_of_keys():
    ring, keys = HashRing(vnodes=64), range(0, 20000 * 7919, 7919)
    before = {k: next(ring.walk(k * 0x9E3779B97F4A7C15 % 2**64, (0, 1, 2, 3))) for k in keys}
    after = {k: next(ring.walk(k * 0x9E3779B97F4A7C15 % 2**64, (0, 1, 2, 3, 4))) for k in keys}
    moved = [k for k in keys if before[k] != after[k]]
    assert 0.12 < len(moved) / len(before) < 0.28              # ~1/5, and...
    assert all(after[k] == 4 for k in moved)                    # ...only onto the new replica


def test_prefix_hash_is_sticky():
    reps, ph = [Stub(i) for i in range(4)], PrefixHash()
    assert len({ph.pick(req(rid=i, seg=5), reps, 0).rid for i in range(50)}) == 1


def test_bounded_load_capacity_and_bound():
    ch = ConsistentHashBoundedLoad(eps=0.25)
    for i in range(7):
        ch.outstanding[i % 4] += 1
    assert ch.capacity((0, 1, 2, 3)) == math.ceil(1.25 * 8 / 4) == 3
    ch, reps = ConsistentHashBoundedLoad(eps=0.25), [Stub(i) for i in range(4)]
    for i in range(100):                                        # 100 requests, ALL with the same prefix
        r = ch.pick(req(rid=i), reps, 0)
        assert ch.outstanding[r.rid] < ch.capacity((0, 1, 2, 3))
        ch.on_dispatch(req(rid=i), r, 0)
    assert max(ch.outstanding.values()) <= math.ceil(1.25 * 100 / 4)    # vs 100 on one replica for PrefixHash


def test_queue_and_kv_scorers_match_upstream_formulas():
    reps = [Stub(0, waiting=0, kv=0.2), Stub(1, waiting=2, kv=0.5), Stub(2, waiting=4, kv=0.9)]
    assert QueueScorer().score(req(), reps, None, 0) == {0: 1.0, 1: 0.5, 2: 0.0}      # (max - q) / (max - min)
    assert QueueScorer().score(req(), [Stub(0, 3), Stub(1, 3)], None, 0) == {0: 1.0, 1: 1.0}  # all equal: neutral
    assert KVCacheUtilizationScorer().score(req(), reps, None, 0)[2] == pytest.approx(0.1)   # 1 - usage


def test_prefix_scorer_is_matched_over_total_blocks():
    r0, r1 = Replica(0, L4_8B), Replica(1, L4_8B)
    warm = req(rid=1, prompt=160)                               # 10 blocks
    r0.enqueue(warm, 0.0)
    t = 0.0
    while (d := r0.start_step(t)) is not None:
        t += d
        r0.end_step(t)
    longer = Request(2, 0.0, 320, 1, HashChain(16).extend(5, 160).extend(6, 160).hashes, 20, 16)
    scores = PrefixCacheScorer(PreciseIndex()).score(longer, [r0, r1], None, 0)
    assert scores == {0: 0.5, 1: 0.0}                           # 10 of 20 blocks cached on replica 0


def test_weighted_scorer_trades_cache_for_queue():
    class Fixed:
        def __init__(self, s):
            self.s = s

        def score(self, *a):
            return self.s
    reps = [Stub(0), Stub(1)]
    prefix, queue = Fixed({0: 1.0, 1: 0.0}), Fixed({0: 0.0, 1: 1.0})
    assert WeightedScorer([(prefix, 3), (queue, 2)]).pick(req(), reps, 0).rid == 0     # 3 > 2: go where it's cached
    assert WeightedScorer([(prefix, 1), (queue, 2)]).pick(req(), reps, 0).rid == 1     # 1 < 2: go where it's quiet
    assert WeightedScorer([(Fixed({0: 7.0, 1: 0.5}), 1)]).pick(req(), reps, 0).rid == 0  # clamped to [0, 1]


def test_lora_affinity_filter():
    reps = [Stub(0, loras=("a",)), Stub(1, loras=("b", "c", "d", "e")), Stub(2, loras=())]
    f = LoraAffinityFilter()
    assert [r.rid for r in f.filter(req(lora="a"), reps, None, 0)] == [0]          # already loaded
    assert [r.rid for r in f.filter(req(lora="z"), reps, None, 0)] == [0, 2]       # has a free slot
    assert f.filter(req(), reps, None, 0) == reps                                  # no adapter: no filter


def test_affinity_filter_sticks_until_saturated():
    class Idx:
        def match(self, rq, r):
            return 10 if r.rid == 0 else 0
    reps, router = [Stub(0), Stub(1)], LeastOutstanding()
    f = PrefixAffinityFilter(Idx(), threshold=0.8, max_ttft_penalty_s=2.0, peak_prefill_tok_s=1000.0)
    assert [r.rid for r in f.filter(req(), reps, router, 0)] == [0]               # sticky
    router.inflight_tokens[0] = 2500                                                # 2.5 s of estimated TTFT
    assert f.filter(req(), reps, router, 0) == reps                                 # > 2 s penalty: spread
