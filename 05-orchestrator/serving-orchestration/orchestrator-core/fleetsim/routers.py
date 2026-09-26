"""Routers: which replica gets this request. Every algorithm here answers one tension differently — send the
request where its prefix is cached (locality), or where the queue is short (load)?

    RoundRobin, LeastOutstanding, PowerOfTwo     load only (blind to cache and to request size)
    PrefixHash                                   locality only (a hot prefix melts one replica)
    ConsistentHashBoundedLoad                    locality, capped at (1 + eps) x average load
    WeightedScorer                               the llm-d EPP shape: filters -> weighted scorers -> max-score pick

A router sees two kinds of state: its own dispatch counters (always fresh — it just made those decisions) and
replica metrics scraped a little while ago (possibly stale). Scorers mirror the upstream llm-d Router plugins
(prefix-cache-scorer, queue-scorer, kv-cache-utilization-scorer, token-load-scorer, prefix-cache-affinity-filter).
"""
from __future__ import annotations

import bisect
import math
import random
from collections import OrderedDict, defaultdict

from .workload import mix64


class Router:
    """Base class: pick(req, replicas, now) returns one routable replica. The on_* hooks keep the router's own,
    always-fresh counters: requests in flight and uncached prompt tokens not yet prefilled, per replica."""
    name = "router"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.outstanding = defaultdict(int)       # requests this router sent and not yet finished, per replica
        self.inflight_tokens = defaultdict(int)   # uncached prompt tokens sent and not yet prefilled, per replica
        self._tok = {}

    def pick(self, req, replicas, now):
        raise NotImplementedError

    def cached_estimate(self, req, r) -> int:     # what this router believes r has cached for req (tokens)
        return 0

    def on_dispatch(self, req, r, now):
        t = max(0, req.prompt - self.cached_estimate(req, r))
        self.outstanding[r.rid] += 1
        self.inflight_tokens[r.rid] += t
        self._tok[req.rid] = (r.rid, t)

    def on_first_token(self, req, r):
        rid, t = self._tok.pop(req.rid, (None, 0))
        if rid is not None:
            self.inflight_tokens[rid] -= t

    def on_complete(self, req, r):
        self.outstanding[r.rid] -= 1

    def _argmin(self, replicas, load):
        best = min(load(r) for r in replicas)
        top = [r for r in replicas if load(r) == best]
        return top[0] if len(top) == 1 else self.rng.choice(top)     # ties at random, not "first wins"

    def _load(self, kind, now):
        if kind == "outstanding":
            return lambda r: self.outstanding[r.rid]
        return lambda r: r.metrics(now)[kind]                        # scraped: "waiting", "running", "kv"


class RoundRobin(Router):
    """Next replica in turn. Fair in request count, blind to an order-of-magnitude spread in request cost."""
    name = "round-robin"

    def __init__(self, seed=0):
        super().__init__(seed)
        self._i = -1

    def pick(self, req, replicas, now):
        self._i += 1
        return replicas[self._i % len(replicas)]


class LeastOutstanding(Router):
    """argmin of load over all replicas. With load='outstanding' it uses the router's own counters; with a scraped
    metric ('waiting', 'running', 'kv') every request in a burst sees the same stale minimum and herds onto it."""
    name = "least-outstanding"

    def __init__(self, load="outstanding", seed=0):
        super().__init__(seed)
        self.load = load

    def pick(self, req, replicas, now):
        return self._argmin(replicas, self._load(self.load, now))


class PowerOfTwo(Router):
    """Power of two choices (Mitzenmacher 2001): sample d replicas uniformly, send to the less loaded one.
    Max load drops from ~log n/log log n (random) to ~log log n/log d, O(1) per decision, robust to stale data."""
    name = "power-of-two"

    def __init__(self, d=2, load="outstanding", seed=0):
        super().__init__(seed)
        self.d, self.load = d, load

    def pick(self, req, replicas, now):
        return self._argmin(self.rng.sample(replicas, min(self.d, len(replicas))), self._load(self.load, now))


class HashRing:
    """Consistent hashing: each replica owns `vnodes` points on a 64-bit ring; a key belongs to the next point
    clockwise. Adding or removing one of n replicas moves only ~1/n of the keys."""

    def __init__(self, vnodes=64):
        self.vnodes, self._ids, self._pts, self._own = vnodes, None, [], []

    def walk(self, key: int, ids: tuple):
        """Replica ids in clockwise order from `key`, each once."""
        if ids != self._ids:
            pts = sorted((mix64((i << 16) + v), i) for i in ids for v in range(self.vnodes))
            self._pts, self._own, self._ids = [p for p, _ in pts], [i for _, i in pts], ids
        start, seen = bisect.bisect(self._pts, key), set()
        for k in range(len(self._pts)):
            rid = self._own[(start + k) % len(self._pts)]
            if rid not in seen:
                seen.add(rid)
                yield rid
                if len(seen) == len(ids):
                    return


class PrefixHash(Router):
    """Pure affinity: hash the request's first `prefix_blocks` KV blocks (or its session) onto a ring. Every request
    with that prefix lands on the same replica — maximum cache locality, zero load awareness."""
    name = "prefix-hash"

    def __init__(self, prefix_blocks=8, by="prefix", vnodes=64, seed=0):
        super().__init__(seed)
        self.prefix_blocks, self.by, self.ring = prefix_blocks, by, HashRing(vnodes)

    def key(self, req) -> int:
        if self.by == "session" and req.session >= 0:
            return mix64(req.session)
        k = min(self.prefix_blocks, req.pblocks)
        return req.hashes[k - 1] if k else mix64(req.rid)

    def pick(self, req, replicas, now):
        by_id = {r.rid: r for r in replicas}
        return by_id[next(self.ring.walk(self.key(req), tuple(by_id)))]


class ConsistentHashBoundedLoad(PrefixHash):
    """Consistent hashing with bounded loads (Mirrokni, Thorup, Zadimoghaddam, SODA 2018): no replica may hold more
    than ceil((1 + eps) * (m + 1) / n) of the m requests in flight (+1 for this one); walk clockwise past full ones.
    eps -> infinity is PrefixHash; small eps trades locality for balance. (Envoy/HAProxy: hash_balance_factor.)"""
    name = "chwbl"

    def __init__(self, eps=0.25, **kw):
        super().__init__(**kw)
        self.eps = eps

    def capacity(self, ids) -> int:
        m = sum(self.outstanding[i] for i in ids)
        return math.ceil((1 + self.eps) * (m + 1) / len(ids))

    def pick(self, req, replicas, now):
        by_id = {r.rid: r for r in replicas}
        cap = self.capacity(tuple(by_id))
        for rid in self.ring.walk(self.key(req), tuple(by_id)):
            if self.outstanding[rid] < cap:
                return by_id[rid]
        raise AssertionError("unreachable: n x cap > m, so some replica is below capacity")


# -- the EPP model: indexes, scorers, filters ----------------------------------------------------------------
class ApproxPrefixIndex:
    """What the router *believes* each replica caches: it records the prompt blocks of every request it sends
    (LRU, `capacity` blocks per replica). It never hears about evictions, so it goes stale when caches are small."""

    def __init__(self, capacity=None):
        self.capacity, self.lru = capacity, defaultdict(OrderedDict)

    def match(self, req, r) -> int:
        lru, n = self.lru[r.rid], 0
        for h in req.hashes[:req.pblocks]:
            if h not in lru:
                break
            n += 1
        return n

    def record(self, req, r):
        lru, cap = self.lru[r.rid], self.capacity or r.p.kv_blocks
        for h in req.hashes[:req.pblocks]:
            lru[h] = None
            lru.move_to_end(h)
        while len(lru) > cap:
            lru.popitem(last=False)


class PreciseIndex:
    """What each replica actually holds — the view llm-d builds from vLLM KV-cache events (block stored/removed,
    published over ZMQ). The simulator reads the pool directly; a real index lags by the event latency."""

    def match(self, req, r) -> int:
        return len(r.pool.match(req.hashes, req.pblocks))

    def record(self, req, r):
        pass


class PrefixCacheScorer:
    """prefix-cache-scorer: matched prefix blocks / total prompt blocks (its default, matchLengthWeight = 0)."""
    name = "prefix-cache-scorer"

    def __init__(self, index=None):
        self.index = index or ApproxPrefixIndex()

    def score(self, req, replicas, router, now):
        total = req.pblocks
        return {r.rid: (self.index.match(req, r) / total if total else 0.0) for r in replicas}


class QueueScorer:
    """queue-scorer: (maxQ - q) / (maxQ - minQ) over scraped vllm:num_requests_waiting; 1.0 for all if equal."""
    name = "queue-scorer"

    def score(self, req, replicas, router, now):
        q = {r.rid: r.metrics(now)["waiting"] for r in replicas}
        lo, hi = min(q.values()), max(q.values())
        return {k: 1.0 if hi == lo else (hi - v) / (hi - lo) for k, v in q.items()}


class KVCacheUtilizationScorer:
    """kv-cache-utilization-scorer: 1 - vllm:kv_cache_usage_perc."""
    name = "kv-cache-utilization-scorer"

    def score(self, req, replicas, router, now):
        return {r.rid: 1.0 - r.metrics(now)["kv"] for r in replicas}


class TokenLoadScorer:
    """token-load-scorer: 1 - min(1, tokens / threshold), tokens = the endpoint's uncached prompt tokens in flight
    + this request's tokens the endpoint has not cached (upstream InFlightLoad + UncachedRequestTokens). It counts
    work, not requests, and prices cache warmth in the same units: a warm endpoint costs this request less."""
    name = "token-load-scorer"

    def __init__(self, threshold_tokens=4_194_304):
        self.threshold = threshold_tokens

    def tokens(self, req, r, router) -> int:
        return router.inflight_tokens[r.rid] + max(0, req.prompt - router.cached_estimate(req, r))

    def score(self, req, replicas, router, now):
        return {r.rid: 1.0 - min(1.0, self.tokens(req, r, router) / self.threshold) for r in replicas}


class LoraAffinityFilter:
    """Keep replicas with the request's adapter loaded; else those with a free adapter slot; else all."""
    name = "lora-affinity-filter"

    def filter(self, req, replicas, router, now):
        if not req.lora:
            return replicas
        hot = [r for r in replicas if req.lora in r.metrics(now)["loras"]]
        room = [r for r in replicas if len(r.metrics(now)["loras"]) < r.p.max_loras]
        return hot or room or replicas


class PrefixAffinityFilter:
    """prefix-cache-affinity-filter, "sticky until saturated": keep replicas whose prefix score >= threshold; if the
    best sticky replica's estimated TTFT (its in-flight tokens / peak prefill tok/s) exceeds the best non-sticky
    one's by more than max_ttft_penalty_s, break stickiness and keep everyone (penalty 0 = always stick).
    Upstream defaults: 0.80, 18 s, 15928. `decisions` counts outcomes like the upstream
    llm_d_epp_prefix_cache_affinity_filter_decisions_total: no_match, sticky, load_override (a gate break)."""
    name = "prefix-cache-affinity-filter"

    def __init__(self, index=None, threshold=0.8, max_ttft_penalty_s=18.0, peak_prefill_tok_s=15928.0):
        self.index, self.threshold = index or ApproxPrefixIndex(), threshold
        self.penalty, self.peak = max_ttft_penalty_s, peak_prefill_tok_s
        self.decisions = defaultdict(int)

    def filter(self, req, replicas, router, now):
        total = req.pblocks
        sticky = [r for r in replicas if total and self.index.match(req, r) / total >= self.threshold]
        others = [r for r in replicas if r not in sticky]
        if not sticky:
            self.decisions["no_match"] += 1
            return replicas
        if self.penalty > 0 and others:
            est = lambda r: router.inflight_tokens[r.rid] / self.peak      # noqa: E731
            if min(map(est, sticky)) - min(map(est, others)) > self.penalty:
                self.decisions["load_override"] += 1
                return replicas                                           # sticky set is saturated: spread
        self.decisions["sticky"] += 1
        return sticky


class WeightedScorer(Router):
    """The llm-d EPP scheduling profile: filters -> weighted scorers -> max-score picker, ties broken at random.
    total(r) = sum_i weight_i * clamp(score_i(r), 0, 1)."""
    name = "epp"

    def __init__(self, scorers, filters=(), seed=0, name=None):
        super().__init__(seed)
        self.scorers, self.filters = list(scorers), list(filters)
        self.indexes = {id(x.index): x.index for x in [s for s, _ in self.scorers] + self.filters
                        if hasattr(x, "index")}
        self.name = name or self.name

    def pick(self, req, replicas, now):
        cand = replicas
        for f in self.filters:
            cand = f.filter(req, cand, self, now) or cand
        total = {r.rid: 0.0 for r in cand}
        for scorer, w in self.scorers:
            for rid, v in scorer.score(req, cand, self, now).items():
                total[rid] += w * min(1.0, max(0.0, v))
        best = max(total.values())
        top = [r for r in cand if total[r.rid] >= best - 1e-12]
        return top[0] if len(top) == 1 else self.rng.choice(top)

    def cached_estimate(self, req, r):
        return max((ix.match(req, r) for ix in self.indexes.values()), default=0) * req.block

    def on_dispatch(self, req, r, now):
        super().on_dispatch(req, r, now)
        for ix in self.indexes.values():
            ix.record(req, r)


def epp(weights=(3, 2, 2), index="approx", seed=0) -> WeightedScorer:
    """An EPP-style profile: prefix-cache (approx or precise index), queue and kv-utilization scorers + LoRA filter.
    The 3:2:2 weights are illustrative — tune them against your own hit rate and TTFT (notebook 02)."""
    ix = PreciseIndex() if index == "precise" else ApproxPrefixIndex()
    wp, wq, wk = weights
    return WeightedScorer([(PrefixCacheScorer(ix), wp), (QueueScorer(), wq), (KVCacheUtilizationScorer(), wk)],
                          [LoraAffinityFilter()], seed=seed, name=f"epp {wp}:{wq}:{wk} {index}")


def sticky_until_saturated(peak_prefill_tok_s, max_ttft_penalty_s=18.0, index="approx", seed=0) -> WeightedScorer:
    """The shape of llm-d's optimized-baseline (Sep 2026): prefix-cache-affinity-filter + token-load-scorer."""
    ix = PreciseIndex() if index == "precise" else ApproxPrefixIndex()
    return WeightedScorer([(TokenLoadScorer(), 1)],
                          [LoraAffinityFilter(), PrefixAffinityFilter(ix, 0.8, max_ttft_penalty_s, peak_prefill_tok_s)],
                          seed=seed, name="affinity+token-load")
