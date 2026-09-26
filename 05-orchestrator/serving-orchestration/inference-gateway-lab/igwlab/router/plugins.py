"""Plugins: producers prepare data, filters drop endpoints, scorers rank them, a picker chooses.

The one idea: an endpoint picker is a pipeline of small decisions, each testable on its own,
combined by a *weighted sum*:

    candidates --filters--> survivors --each scorer s: score_s(e) in [0,1]-->
    total(e) = sum_s weight_s * clamp(score_s(e), 0, 1)  --picker--> endpoint

Every class here mirrors an llm-d-router plugin of the same `TYPE` and parameter names
(Sep 2026, v0.10), so a config that only uses these plugins can be pasted into the real EPP.
Semantics worth knowing by heart (all pinned in tests/test_plugins.py):

* `prefix-cache-scorer`      w*min(1, matched_tokens/scale)^2 + (1-w)*match_blocks/total_blocks
* `queue-scorer`             (maxQ - q)/(maxQ - minQ); all equal -> 1.0; unscraped endpoints unscored
* `kv-cache-utilization-scorer`  1 - kv_usage; unscraped endpoints unscored
* `running-requests-size-scorer` like queue-scorer on running requests
* `active-request-scorer`    <= idleThreshold -> 1.0 else (max - n)/max * maxBusyScore  (router-local counts)
* `token-load-scorer`        1 - min(1, inflight_uncached_tokens / queueThresholdTokens)
* `lora-affinity-scorer`     1.0 adapter active | 0.8 room for one more | 0.6 adapter waiting | 0.0
* `max-score-picker`         highest total; a tie tier is rotated by a per-request counter. Here the
                             candidates are name-sorted first, so with *no* scorers the lab router is
                             exact round-robin. Upstream rotates the same way, but its candidate list
                             comes from a Go map (random order), so in the real EPP ties land
                             effectively at random — even on average, not strictly alternating.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .prefix import (DEFAULT_LRU_CAPACITY_PER_SERVER, DEFAULT_MAX_PREFIX_BLOCKS, DEFAULT_MAX_PREFIX_TOKENS,
                     PrefixIndex, PrefixMatch, block_hashes, effective_block_size, max_blocks_for)

__all__ = ["RequestCtx", "InFlightLoad", "Plugin", "REGISTRY", "make_plugin", "clamp"]


def clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class RequestCtx:
    """Everything a plugin may read about one request."""
    request_id: str
    model: str                               # target model (a base model or a LoRA adapter name)
    tokens: list                             # pseudo-tokens (see tokens.py)
    body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    cache_salt: str = ""
    priority: int = 0
    data: dict = field(default_factory=dict)   # producer name -> {endpoint name -> attribute}
    state: dict = field(default_factory=dict)  # plugin name -> private per-request scratch


@dataclass(frozen=True)
class InFlightLoad:
    requests: int
    tokens: int


class Plugin:
    TYPE = ""
    KIND = ""                      # producer | filter | scorer | picker | handler | datasource | extractor
    PARAMS: dict = {}              # parameter name -> default
    LAB_ONLY: frozenset = frozenset()

    def __init__(self, name: str | None = None, rng: random.Random | None = None, clock=time.monotonic, **params):
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise ValueError(f"{self.TYPE}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(self.PARAMS)}")
        self.name = name or self.TYPE
        self.params = {**self.PARAMS, **params}
        self.rng = rng or random.Random(0)
        self.clock = clock
        self.validate()

    def validate(self):
        pass

    def __repr__(self):
        return f"<{self.TYPE} {self.name!r}>"


# ================================================================== producers
class Producer(Plugin):
    KIND = "producer"

    def produce(self, ctx: RequestCtx, endpoints) -> None: ...
    def pre_request(self, ctx: RequestCtx, ep) -> None: ...
    def on_first_token(self, ctx: RequestCtx, ep) -> None: ...
    def on_complete(self, ctx: RequestCtx, ep) -> None: ...


class ApproxPrefixCacheProducer(Producer):
    """Hashes the request into blocks, looks them up in the index, records the pick afterwards."""
    TYPE = "approx-prefix-cache-producer"
    PARAMS = {"autoTune": True, "blockSizeTokens": 16, "maxPrefixBlocksToMatch": DEFAULT_MAX_PREFIX_BLOCKS,
              "maxPrefixTokensToMatch": DEFAULT_MAX_PREFIX_TOKENS, "lruCapacityPerServer": DEFAULT_LRU_CAPACITY_PER_SERVER,
              "ttlSeconds": None}
    LAB_ONLY = frozenset({"ttlSeconds"})

    def validate(self):
        p = self.params
        if not p["autoTune"] and p["blockSizeTokens"] <= 0:
            raise ValueError("blockSizeTokens must be > 0 when autoTune is false")
        if p["maxPrefixTokensToMatch"] < 0 or p["maxPrefixBlocksToMatch"] < 0:
            raise ValueError("match caps must be >= 0")
        self.index = PrefixIndex(p["lruCapacityPerServer"], ttl_s=p["ttlSeconds"], clock=self.clock)

    def block_size(self, endpoints) -> int:
        engine_bs = endpoints[0].metrics.block_size if endpoints else None   # upstream reads endpoints[0]
        return effective_block_size(self.params["blockSizeTokens"], engine_bs, self.params["autoTune"])

    def capacity_for(self, ep, index_block_size: int) -> int:
        """LRU entries for this endpoint. Auto-tuned from the engine's KV geometry, converted
        to index blocks (lab: upstream uses the engine's block count directly)."""
        m = ep.metrics
        if self.params["autoTune"] and m.num_gpu_blocks > 0 and m.block_size > 0:
            return max(1, m.num_gpu_blocks * m.block_size // index_block_size)
        return self.params["lruCapacityPerServer"]

    def produce(self, ctx, endpoints):
        bs = self.block_size(endpoints)
        cap = max_blocks_for(bs, self.params["maxPrefixTokensToMatch"], self.params["maxPrefixBlocksToMatch"])
        hashes = block_hashes(ctx.tokens, bs, ctx.model, ctx.cache_salt, cap)
        matches = self.index.match(hashes)
        ctx.state[self.name] = (hashes, bs)
        ctx.data[self.name] = {e.name: PrefixMatch(matches.get(e.name, 0), len(hashes), bs) for e in endpoints}

    def pre_request(self, ctx, ep):
        hashes, bs = ctx.state.get(self.name, ([], 0))
        if hashes:
            self.index.add(hashes, ep.name, self.capacity_for(ep, bs))


class InFlightLoadProducer(Producer):
    """Router-local load: requests (released at completion) and *uncached* prompt tokens
    (released at the first streamed chunk, when prefill is over)."""
    TYPE = "inflight-load-producer"
    PARAMS = {"prefixMatchInfoProducerName": None}

    def produce(self, ctx, endpoints):
        ctx.data[self.name] = {e.name: InFlightLoad(e.inflight_requests, e.inflight_tokens) for e in endpoints}

    def pre_request(self, ctx, ep):
        pname = self.params["prefixMatchInfoProducerName"] or ApproxPrefixCacheProducer.TYPE
        pm = ctx.data.get(pname, {}).get(ep.name)
        cached = min(pm.matched_tokens, len(ctx.tokens)) if pm else 0
        uncached = max(0, len(ctx.tokens) - cached)
        ctx.state[self.name] = uncached
        ep.inflight_tokens += uncached

    def _release_tokens(self, ctx, ep):
        n = ctx.state.pop(self.name, 0)
        ep.inflight_tokens -= n

    def on_first_token(self, ctx, ep):
        self._release_tokens(ctx, ep)

    def on_complete(self, ctx, ep):
        self._release_tokens(ctx, ep)


# ================================================================== filters
class Filter(Plugin):
    KIND = "filter"

    def filter(self, ctx: RequestCtx, endpoints: list) -> list:
        raise NotImplementedError


def _inflight(ctx, ep, producer_name=None) -> InFlightLoad | None:
    return ctx.data.get(producer_name or InFlightLoadProducer.TYPE, {}).get(ep.name)


class UtilizationFilter(Filter):
    """Drop endpoints over any cap; missing metrics count as 0; optional fail-open fallback."""
    TYPE = "utilization-filter"
    PARAMS = {"conditions": [], "fallbackOnEmpty": False, "inFlightLoadProducerName": None}
    METRICS = ("active-requests", "running-requests", "waiting-queue", "kv-cache-utilization")

    def validate(self):
        conds = self.params["conditions"]
        if not conds:
            raise ValueError("utilization-filter needs at least one condition")
        for c in conds:
            if c.get("metric") not in self.METRICS:
                raise ValueError(f"utilization-filter metric must be one of {self.METRICS}")
            mv = float(c.get("maxValue", -1))
            if mv < 0 or (c["metric"] == "kv-cache-utilization" and mv > 1):
                raise ValueError(f"bad maxValue for {c['metric']}: {c.get('maxValue')}")

    def value(self, ctx, ep, metric):
        m = ep.metrics
        if metric == "active-requests":
            load = _inflight(ctx, ep, self.params["inFlightLoadProducerName"])
            return load.requests if load else 0
        if not m.fresh:
            return 0
        return {"running-requests": m.running, "waiting-queue": m.waiting, "kv-cache-utilization": m.kv_usage}[metric]

    def filter(self, ctx, endpoints):
        keep = [e for e in endpoints
                if all(self.value(ctx, e, c["metric"]) <= float(c["maxValue"]) for c in self.params["conditions"])]
        if not keep and self.params["fallbackOnEmpty"]:
            return list(endpoints)
        return keep


def _prefix_info(ctx, ep, producer_name=None) -> PrefixMatch | None:
    return ctx.data.get(producer_name or ApproxPrefixCacheProducer.TYPE, {}).get(ep.name)


class PrefixCacheAffinityFilter(Filter):
    """Keep only 'sticky' endpoints (prefix match ratio >= threshold) unless that would make a
    request wait too long: the TTFT gate estimates TTFT = inflight_tokens / peakPrefillThroughput
    and breaks stickiness when the best sticky endpoint is more than maxTTFTPenaltyMs slower
    than the best non-sticky one ("sticky until saturated")."""
    TYPE = "prefix-cache-affinity-filter"
    PARAMS = {"affinityThreshold": 0.80, "explorationProbability": 0.0, "maxTTFTPenaltyMs": 18000.0,
              "ttftSource": "prefillThroughput", "peakPrefillThroughput": 15928.0,
              "prefixMatchInfoProducerName": None, "inFlightLoadProducerName": None}

    def validate(self):
        if self.params["ttftSource"] != "prefillThroughput":
            raise ValueError("lab router supports ttftSource: prefillThroughput only")
        if self.params["peakPrefillThroughput"] <= 0:
            raise ValueError("peakPrefillThroughput must be > 0")
        self.last_outcome = ""

    def filter(self, ctx, endpoints):
        p = self.params
        ratio = {e.name: (_prefix_info(ctx, e, p["prefixMatchInfoProducerName"]) or PrefixMatch(0, 0, 0)).ratio
                 for e in endpoints}
        sticky = [e for e in endpoints if ratio[e.name] >= p["affinityThreshold"]]
        if not sticky:
            self.last_outcome = "no_sticky"
            return list(endpoints)
        if p["explorationProbability"] > 0 and self.rng.random() < p["explorationProbability"]:
            self.last_outcome = "explore"
            return list(endpoints)
        others = [e for e in endpoints if e not in sticky]
        if p["maxTTFTPenaltyMs"] > 0 and others:
            loads = {e.name: _inflight(ctx, e, p["inFlightLoadProducerName"]) for e in endpoints}
            if all(v is not None for v in loads.values()):
                est = {n: v.tokens / p["peakPrefillThroughput"] * 1000.0 for n, v in loads.items()}
                best_sticky = min(est[e.name] for e in sticky)
                best_other = min(est[e.name] for e in others)
                if best_sticky - best_other > p["maxTTFTPenaltyMs"]:
                    self.last_outcome = "load_override"
                    return list(endpoints)
        self.last_outcome = "sticky"
        return sticky


# ================================================================== scorers
class Scorer(Plugin):
    KIND = "scorer"

    def score(self, ctx: RequestCtx, endpoints: list) -> dict:
        """-> {endpoint name: score in [0,1]}; endpoints left out contribute 0."""
        raise NotImplementedError


class PrefixCacheScorer(Scorer):
    TYPE = "prefix-cache-scorer"
    PARAMS = {"matchLengthWeight": 0.0, "matchLengthScaleTokens": 8192, "prefixMatchInfoProducerName": None}

    def validate(self):
        if not 0.0 <= self.params["matchLengthWeight"] <= 1.0:
            raise ValueError("matchLengthWeight must be in [0, 1]")
        if self.params["matchLengthScaleTokens"] <= 0:
            raise ValueError("matchLengthScaleTokens must be > 0")

    @staticmethod
    def formula(match_blocks, total_blocks, block_size, w=0.0, scale=8192) -> float:
        if total_blocks <= 0:
            return 0.0
        ratio = match_blocks / total_blocks
        length = min(1.0, match_blocks * block_size / scale) ** 2
        return w * length + (1.0 - w) * ratio

    def score(self, ctx, endpoints):
        out = {}
        for e in endpoints:
            pm = _prefix_info(ctx, e, self.params["prefixMatchInfoProducerName"])
            out[e.name] = 0.0 if pm is None else self.formula(pm.match_blocks, pm.total_blocks, pm.block_size,
                                                             self.params["matchLengthWeight"],
                                                             self.params["matchLengthScaleTokens"])
        return out


def _minmax_scores(values: dict) -> dict:
    """(max - v)/(max - min); if all equal, every endpoint gets the neutral 1.0."""
    if not values:
        return {}
    hi, lo = max(values.values()), min(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (hi - v) / (hi - lo) for k, v in values.items()}


class QueueScorer(Scorer):
    TYPE = "queue-scorer"

    def score(self, ctx, endpoints):
        return _minmax_scores({e.name: e.metrics.waiting for e in endpoints if e.metrics.fresh})


class RunningRequestsSizeScorer(Scorer):
    TYPE = "running-requests-size-scorer"

    def score(self, ctx, endpoints):
        return _minmax_scores({e.name: e.metrics.running for e in endpoints if e.metrics.fresh})


class KVCacheUtilizationScorer(Scorer):
    TYPE = "kv-cache-utilization-scorer"

    def score(self, ctx, endpoints):
        return {e.name: 1.0 - e.metrics.kv_usage for e in endpoints if e.metrics.fresh}


class ActiveRequestScorer(Scorer):
    TYPE = "active-request-scorer"
    PARAMS = {"idleThreshold": 0, "maxBusyScore": 1.0, "requestTimeout": None}

    def score(self, ctx, endpoints):
        counts = {e.name: (_inflight(ctx, e) or InFlightLoad(0, 0)).requests for e in endpoints}
        mx = max(counts.values(), default=0)
        out = {}
        for n, c in counts.items():
            out[n] = 1.0 if c <= self.params["idleThreshold"] else (mx - c) / mx * self.params["maxBusyScore"]
        return out


class TokenLoadScorer(Scorer):
    TYPE = "token-load-scorer"
    PARAMS = {"queueThresholdTokens": 4_194_304}

    def validate(self):
        if self.params["queueThresholdTokens"] <= 0:
            raise ValueError("queueThresholdTokens must be > 0")

    def score(self, ctx, endpoints):
        t = self.params["queueThresholdTokens"]
        return {e.name: 1.0 - min(1.0, (_inflight(ctx, e) or InFlightLoad(0, 0)).tokens / t) for e in endpoints}


class LoraAffinityScorer(Scorer):
    TYPE = "lora-affinity-scorer"

    def score(self, ctx, endpoints):
        out = {}
        for e in endpoints:
            m = e.metrics
            union = len(m.active_models | m.waiting_models)
            if ctx.model in m.active_models:
                out[e.name] = 1.0
            elif union < m.max_active_models:
                out[e.name] = 0.8
            elif ctx.model in m.waiting_models:
                out[e.name] = 0.6
            else:
                out[e.name] = 0.0
        return out


# ================================================================== pickers
class Picker(Plugin):
    KIND = "picker"
    PARAMS = {"maxNumOfEndpoints": 1}

    def validate(self):
        if self.params["maxNumOfEndpoints"] <= 0:
            raise ValueError("maxNumOfEndpoints must be > 0")

    def pick(self, ctx: RequestCtx, scored: list) -> list:
        raise NotImplementedError


class MaxScorePicker(Picker):
    """Sort by score (desc, then name — the lab's deterministic base order) and rotate each tie
    tier by a per-request counter (upstream: same rotation over a map-ordered, i.e. random, list)."""
    TYPE = "max-score-picker"

    def validate(self):
        super().validate()
        self.counter = 0

    def pick(self, ctx, scored):
        order = sorted(scored, key=lambda es: (-es[1], es[0].name))
        c = self.counter
        self.counter += 1
        n = self.params["maxNumOfEndpoints"]
        start = 0
        while start < len(order) and start < n:
            end = start + 1
            while end < len(order) and order[end][1] == order[start][1]:
                end += 1
            tier = order[start:end]
            if len(tier) > 1:
                s = c % len(tier)
                order[start:end] = tier[s:] + tier[:s]
            start = end
        return [e for e, _ in order[:n]]


class RandomPicker(Picker):
    TYPE = "random-picker"

    def pick(self, ctx, scored):
        eps = [e for e, _ in sorted(scored, key=lambda es: es[0].name)]
        self.rng.shuffle(eps)
        return eps[: self.params["maxNumOfEndpoints"]]


class WeightedRandomPicker(Picker):
    """A-Res weighted sampling: key = U^(1/w); all scores <= 0 -> uniform random."""
    TYPE = "weighted-random-picker"

    def pick(self, ctx, scored):
        scored = sorted(scored, key=lambda es: es[0].name)
        if all(s <= 0 for _, s in scored):
            eps = [e for e, _ in scored]
            self.rng.shuffle(eps)
            return eps[: self.params["maxNumOfEndpoints"]]
        keyed = []
        for e, s in scored:
            u = self.rng.random() or 1e-12
            keyed.append(((u ** (1.0 / s)) if s > 0 else -1.0, e))
        keyed.sort(key=lambda ke: -ke[0])
        return [e for _, e in keyed[: self.params["maxNumOfEndpoints"]]]


# ================================================================== accepted, no-op here
class SingleProfileHandler(Plugin):
    TYPE, KIND = "single-profile-handler", "handler"


class MetricsDataSource(Plugin):
    """Scrape settings; the lab's Scraper reads `path` and `interval` from here."""
    TYPE, KIND = "metrics-data-source", "datasource"
    PARAMS = {"scheme": "http", "path": "/metrics", "insecureSkipVerify": True, "interval": None,
              "caCertPath": None, "clientCertPath": None, "clientKeyPath": None}


class CoreMetricsExtractor(Plugin):
    TYPE, KIND = "core-metrics-extractor", "extractor"
    PARAMS = {"engineLabelKey": "llm-d.ai/engine-type", "defaultEngine": "vllm", "engineConfigs": None}

    def validate(self):
        if self.params["defaultEngine"] != "vllm" or self.params["engineConfigs"]:
            raise ValueError("lab router extracts vLLM metric names only")


REGISTRY = {c.TYPE: c for c in (
    ApproxPrefixCacheProducer, InFlightLoadProducer,
    UtilizationFilter, PrefixCacheAffinityFilter,
    PrefixCacheScorer, QueueScorer, RunningRequestsSizeScorer, KVCacheUtilizationScorer,
    ActiveRequestScorer, TokenLoadScorer, LoraAffinityScorer,
    MaxScorePicker, RandomPicker, WeightedRandomPicker,
    SingleProfileHandler, MetricsDataSource, CoreMetricsExtractor)}


def make_plugin(type_: str, name: str | None = None, parameters: dict | None = None, **kw) -> Plugin:
    cls = REGISTRY.get(type_)
    if cls is None:
        raise ValueError(f"unknown plugin type {type_!r}; the lab router implements: {sorted(REGISTRY)}")
    return cls(name=name, **kw, **(parameters or {}))
