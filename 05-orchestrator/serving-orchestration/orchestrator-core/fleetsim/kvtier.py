"""KV cache beyond HBM: is it cheaper to fetch a prefix's KV than to recompute it, and how big is the working set
of multi-turn agent sessions?

    onload_s     = latency + tokens x KV bytes per token / tier bandwidth
    recompute_s  = tokens / prefill tokens per s                      (prefill is compute-bound)
    fetch beats recompute when  tier bandwidth > KV bytes per token x prefill tokens per s   (breakeven_gb_s)
    working set  = concurrent sessions x context tokens x KV bytes per token

A tier is only worth its hit rate, so TieredKV keeps exclusive LRU tiers of whole-session KV (HBM -> DRAM -> disk,
demote on evict — the shape of vLLM's OffloadingConnector, LMCache, Mooncake Store, SGLang HiCache), and
simulate_sessions() replays agent sessions over per-replica tiers plus an optional shared (cross-replica) tier.
"""
from __future__ import annotations

import heapq
import random
from collections import OrderedDict
from dataclasses import dataclass

from .metrics import percentile


@dataclass(frozen=True)
class Tier:
    name: str
    capacity_gb: float
    read_gb_s: float
    latency_s: float = 0.0


def onload_s(tokens, kv_bytes_per_token, tier: Tier) -> float:
    return tier.latency_s + tokens * kv_bytes_per_token / (tier.read_gb_s * 1e9)


def recompute_s(tokens, prefill_tok_s) -> float:
    return tokens / prefill_tok_s


def breakeven_gb_s(kv_bytes_per_token, prefill_tok_s) -> float:
    """Tier bandwidth above which fetching KV beats recomputing it: the KV bytes prefill produces per second."""
    return kv_bytes_per_token * prefill_tok_s / 1e9


def working_set_gb(sessions, context_tokens, kv_bytes_per_token) -> float:
    return sessions * context_tokens * kv_bytes_per_token / 1e9


class TieredKV:
    """Exclusive LRU tiers of whole-session KV. put() lands in tier 0; each eviction demotes the LRU session one
    tier down; the last tier drops it. lookup() says where a session lives and how many tokens are stored."""

    def __init__(self, tiers, kv_bytes_per_token):
        self.tiers, self.kvb = list(tiers), kv_bytes_per_token
        self.lru = [OrderedDict() for _ in self.tiers]      # key -> tokens, front = least recently used
        self.used = [0.0] * len(self.tiers)

    def lookup(self, key):
        for i, lru in enumerate(self.lru):
            if key in lru:
                return i, lru[key]
        return None, 0

    def put(self, key, tokens):
        for i, lru in enumerate(self.lru):
            if key in lru:
                self.used[i] -= lru.pop(key) * self.kvb / 1e9
        self._insert(0, key, tokens)

    def _insert(self, i, key, tokens):
        while i < len(self.tiers) and tokens * self.kvb / 1e9 > self.tiers[i].capacity_gb:
            i += 1                                          # too big for this tier: straight to a larger one
        if i == len(self.tiers):
            return
        self.lru[i][key] = tokens
        self.used[i] += tokens * self.kvb / 1e9
        while self.used[i] > self.tiers[i].capacity_gb:
            victim, vt = self.lru[i].popitem(last=False)
            self.used[i] -= vt * self.kvb / 1e9
            self._insert(i + 1, victim, vt)


def simulate_sessions(sessions, *, tiers, kv_bytes_per_token, prefill_tok_s, replicas=1, sticky=True,
                      shared: Tier | None = None, task=300, tool=800, output=150, turns=(4, 12),
                      think=(2.0, 20.0), turn_s=3.0, session_rate=1.0, seed=0) -> dict:
    """Turn-level replay of agent sessions over per-replica tiers, plus an optional shared tier every replica reads.

    Only each session's own context (task, outputs, tool results) is tracked: the shared system prompt is hit by
    every session and assumed resident (notebook 02). Each turn: find the longest stored copy of the session's
    context (the chosen replica's tiers, then the shared tier), pay onload for it and recompute for the rest,
    then store the grown context (written through to the shared tier). No queueing — this isolates the cache
    effect. Scores resumed turns (2..n): the share served by each tier and the prefix cost (simulated)."""
    rng = random.Random(seed)
    local = [TieredKV(tiers, kv_bytes_per_token) for _ in range(replicas)]
    remote = TieredKV([shared], kv_bytes_per_token) if shared else None
    by_name = {x.name: x for x in list(tiers) + ([shared] if shared else [])}
    heap, t = [], 0.0
    for s in range(sessions):
        t += rng.expovariate(session_rate)
        heapq.heappush(heap, (t, s, 0, rng.randint(*turns), task))
    served, costs, recomputed, total = {}, [], 0, 0
    while heap:
        t, s, k, n, ctx = heapq.heappop(heap)             # ctx = this turn's session-specific prompt tokens
        rep = local[s % replicas if sticky else rng.randrange(replicas)]
        where, stored = rep.lookup(s)
        tier = tiers[where].name if where is not None else "recompute"
        if remote is not None and remote.lookup(s)[1] > stored:
            tier, stored = shared.name, remote.lookup(s)[1]
        stored = min(stored, ctx)
        tier = tier if stored else "recompute"
        if k:                                             # score resumed turns: only they can reuse their KV
            served[tier] = served.get(tier, 0) + 1
            costs.append((onload_s(stored, kv_bytes_per_token, by_name[tier]) if stored else 0.0)
                         + recompute_s(ctx - stored, prefill_tok_s))
            recomputed, total = recomputed + ctx - stored, total + ctx
        rep.put(s, ctx + output)                          # the KV now covers this turn's prompt and output
        if remote is not None:
            remote.put(s, ctx + output)
        if k + 1 < n:
            nxt = ctx + output + rng.randint(tool // 2, tool * 3 // 2)       # + the next tool result
            heapq.heappush(heap, (t + turn_s + rng.uniform(*think), s, k + 1, n, nxt))
    n_turns = len(costs)
    return {"turns": n_turns, **{f"share_{k}": v / n_turns for k, v in sorted(served.items())},
            "recompute_token_share": recomputed / total, "prefix_cost_p50_s": percentile(costs, 50),
            "prefix_cost_p95_s": percentile(costs, 95)}
