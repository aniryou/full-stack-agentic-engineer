"""Filters, scorers, pickers and the EndpointPickerConfig loader, pinned to hand-computed values."""
import pytest
import yaml

from igwlab.router import ConfigError, Datastore, Endpoint, EndpointMetrics, RequestCtx, Scheduler, load_config
from igwlab.router.plugins import (ActiveRequestScorer, InFlightLoad, KVCacheUtilizationScorer, LoraAffinityScorer,
                                   MaxScorePicker, PrefixCacheAffinityFilter, PrefixCacheScorer, QueueScorer,
                                   TokenLoadScorer, UtilizationFilter, WeightedRandomPicker, make_plugin)
from igwlab.router.prefix import PrefixMatch


def ep(name, waiting=0, running=0, kv=0.0, fresh=True, **kw):
    m = EndpointMetrics(waiting=waiting, running=running, kv_usage=kv, update_time=1.0 if fresh else 0.0, **kw)
    return Endpoint(name=name, url=f"http://{name}", metrics=m)


def ctx(**data):
    return RequestCtx("r1", "lab/llm", list(range(10)), data=data)


def test_prefix_cache_scorer_formula():
    f = PrefixCacheScorer.formula
    assert f(3, 4, 64) == 0.75                                    # default: ratio only
    # w=0.5, scale=512: 0.5 * min(1, 192/512)^2 + 0.5 * 0.75 = 0.5*0.140625 + 0.375
    assert f(3, 4, 64, w=0.5, scale=512) == pytest.approx(0.4453125)
    assert f(0, 0, 64) == 0.0
    s = PrefixCacheScorer()
    c = ctx(**{"approx-prefix-cache-producer": {"a": PrefixMatch(2, 8, 64)}})
    assert s.score(c, [ep("a"), ep("b")]) == {"a": 0.25, "b": 0.0}     # missing info scores 0


def test_queue_scorer_min_max_ties_and_never_scraped_looks_idle():
    q = QueueScorer()
    assert q.score(ctx(), [ep("a", waiting=0), ep("b", waiting=5), ep("c", waiting=10)]) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert q.score(ctx(), [ep("a", waiting=3), ep("b", waiting=3)]) == {"a": 1.0, "b": 1.0}
    # upstream reads GetMetrics() with no freshness check: a never-scraped endpoint has queue 0 -> best score
    assert q.score(ctx(), [ep("a", waiting=4), ep("b", fresh=False), ep("c", waiting=8)]) == {"a": 0.5, "b": 1.0, "c": 0.0}


def test_kv_active_token_and_lora_scorers():
    assert KVCacheUtilizationScorer().score(ctx(), [ep("a", kv=0.25), ep("b", fresh=False)]) == {"a": 0.75, "b": 1.0}
    loads = {"inflight-load-producer": {"a": InFlightLoad(0, 0), "b": InFlightLoad(2, 100), "c": InFlightLoad(4, 900)}}
    eps = [ep("a"), ep("b"), ep("c")]
    assert ActiveRequestScorer().score(ctx(**loads), eps) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert ActiveRequestScorer(idleThreshold=2, maxBusyScore=0.5).score(ctx(**loads), eps) == {"a": 1.0, "b": 1.0, "c": 0.0}
    assert TokenLoadScorer(queueThresholdTokens=1000).score(ctx(**loads), eps) == {"a": 1.0, "b": 0.9, "c": pytest.approx(0.1)}
    # the request's own uncached tokens on each endpoint count too (upstream UncachedRequestTokens):
    # b: 1 - (100 + 10)/1000 = 0.89; c: 1 - min(1, (900 + 200)/1000) = 0.0
    loads2 = {"inflight-load-producer": {"a": InFlightLoad(0, 0, 0), "b": InFlightLoad(2, 100, 10), "c": InFlightLoad(4, 900, 200)}}
    assert TokenLoadScorer(queueThresholdTokens=1000).score(ctx(**loads2), eps) == {"a": 1.0, "b": pytest.approx(0.89), "c": 0.0}


def test_uncached_input_tokens_counts_whole_blocks_and_the_tail_like_upstream():
    from igwlab.router.plugins import uncached_input_tokens
    assert uncached_input_tokens(None, 300) == 300                          # no prefix info: everything
    assert uncached_input_tokens(PrefixMatch(3, 5, 64), 300) == 128         # (5 - 3) x 64: the partial block counts whole
    assert uncached_input_tokens(PrefixMatch(5, 5, 64), 300) == 0
    assert uncached_input_tokens(PrefixMatch(2, 4, 64), 1000) == 128 + 744  # 4 x 64 indexed (capped), 744-token tail


def test_token_load_scorer_prefers_the_endpoint_that_caches_the_prompt():
    """End to end through the producers: two idle endpoints, one already holding this prompt.
    (The prefix producer is optional input to inflight-load-producer, upstream too: without one
    every endpoint is charged the whole prompt.)"""
    cfg = load_config({"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
                       "plugins": [{"type": "approx-prefix-cache-producer"},
                                   {"type": "token-load-scorer", "parameters": {"queueThresholdTokens": 1024}}]})
    tokens = list(range(640))                                   # 10 blocks of 64
    ds = Datastore([ep("warm"), ep("cold")])
    sched = Scheduler(cfg, ds)
    first = RequestCtx("r0", "lab/llm", tokens)
    sched.schedule(first)
    sched.pre_request(first, ds.get("warm"))                    # the index now says: warm holds these 10 blocks
    sched.on_complete(first, ds.get("warm"))                    # ... and warm is idle again
    d = sched.schedule(RequestCtx("r1", "lab/llm", tokens + list(range(64))))     # 11 blocks, 10 cached on warm
    # warm: 1 - 64/1024 = 0.9375; cold: 1 - 704/1024 = 0.3125
    assert d.scores["token-load-scorer"] == pytest.approx({"warm": 0.9375, "cold": 0.3125}) and d.endpoint == "warm"
    l = LoraAffinityScorer()
    c = RequestCtx("r", "sql-lora", [])
    eps = [ep("act", active_models={"sql-lora"}, max_active_models=2),
           ep("room", active_models={"x"}, max_active_models=2),
           ep("wait", active_models={"x", "y"}, waiting_models={"sql-lora"}, max_active_models=2),
           ep("full", active_models={"x", "y"}, max_active_models=2)]
    # precedence: active 1.0 > room for one more adapter 0.8 > adapter already waiting 0.6 > 0.0
    assert l.score(c, eps) == {"act": 1.0, "room": 0.8, "wait": 0.6, "full": 0.0}
    eps[1] = ep("room", active_models={"x"}, waiting_models={"sql-lora"}, max_active_models=3)
    assert l.score(c, eps)["room"] == 0.8                     # union {x, sql-lora} < 3: room wins over waiting


def test_max_score_picker_rotates_ties_round_robin():
    p = MaxScorePicker()
    eps = [ep("a"), ep("b"), ep("c")]
    picks = [p.pick(ctx(), [(e, 0.0) for e in eps])[0].name for _ in range(6)]
    assert picks == ["a", "b", "c", "a", "b", "c"]            # no scorers == exact round-robin
    assert p.pick(ctx(), [(eps[0], 1.0), (eps[1], 3.0), (eps[2], 3.0)])[0].name in {"b", "c"}
    assert [e.name for e in MaxScorePicker(maxNumOfEndpoints=2).pick(ctx(), [(eps[0], 1.0), (eps[1], 3.0), (eps[2], 2.0)])] == ["b", "c"]


def test_weighted_random_picker_prefers_high_scores():
    p = WeightedRandomPicker()
    eps = [ep("a"), ep("b")]
    n = sum(p.pick(ctx(), [(eps[0], 0.9), (eps[1], 0.1)])[0].name == "a" for _ in range(2000))
    assert 0.85 < n / 2000 < 0.95                              # P(a) = 0.9/(0.9+0.1) under A-Res


def test_utilization_filter_caps_and_fallback():
    f = UtilizationFilter(conditions=[{"metric": "waiting-queue", "maxValue": 2}, {"metric": "kv-cache-utilization", "maxValue": 0.8}])
    eps = [ep("a", waiting=1, kv=0.5), ep("b", waiting=3), ep("c", kv=0.9), ep("d", fresh=False)]
    assert [e.name for e in f.filter(ctx(), eps)] == ["a", "d"]           # never scraped: metrics are 0, like upstream
    strict = UtilizationFilter(conditions=[{"metric": "waiting-queue", "maxValue": 0}])
    fallback = UtilizationFilter(conditions=[{"metric": "waiting-queue", "maxValue": 0}], fallbackOnEmpty=True)
    assert strict.filter(ctx(), [eps[1]]) == [] and fallback.filter(ctx(), [eps[1]]) == [eps[1]]
    with pytest.raises(ValueError):
        UtilizationFilter(conditions=[{"metric": "kv-cache-utilization", "maxValue": 1.5}])


def test_prefix_affinity_filter_threshold_and_ttft_gate():
    eps = [ep("hot"), ep("cold")]
    pm = {"hot": PrefixMatch(9, 10, 64), "cold": PrefixMatch(0, 10, 64)}
    f = PrefixCacheAffinityFilter(maxTTFTPenaltyMs=100, peakPrefillThroughput=10_000)
    light = {"hot": InFlightLoad(1, 500), "cold": InFlightLoad(0, 0)}           # 50 ms vs 0 ms
    heavy = {"hot": InFlightLoad(5, 5_000), "cold": InFlightLoad(0, 0)}         # 500 ms vs 0 ms
    c1 = ctx(**{"approx-prefix-cache-producer": pm, "inflight-load-producer": light})
    assert [e.name for e in f.filter(c1, eps)] == ["hot"] and f.last_outcome == "sticky"
    c2 = ctx(**{"approx-prefix-cache-producer": pm, "inflight-load-producer": heavy})
    assert len(f.filter(c2, eps)) == 2 and f.last_outcome == "load_override"
    c3 = ctx(**{"approx-prefix-cache-producer": {"hot": PrefixMatch(7, 10, 64), "cold": PrefixMatch(0, 10, 64)}})
    assert len(f.filter(c3, eps)) == 2 and f.last_outcome == "no_sticky"       # 0.7 < 0.8


def test_scheduler_weighted_sum_is_clamped_and_explained():
    cfg = load_config("default-weighted")
    ds = Datastore([ep("a", waiting=0, kv=0.9), ep("b", waiting=4, kv=0.1)])
    d = Scheduler(cfg, ds).schedule(RequestCtx("r", "lab/llm", list(range(256))))
    # queue: a=1, b=0 (x2); kv: a=0.1, b=0.9 (x2); prefix: 0 for both (x3)
    assert d.totals == pytest.approx({"a": 2.2, "b": 1.8}) and d.endpoint == "a"
    assert "queue-scorer" in d.table() and "<- picked" in d.table()


def test_empty_filter_result_means_no_endpoint():
    doc = {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
           "plugins": [{"type": "utilization-filter", "parameters": {"conditions": [{"metric": "waiting-queue", "maxValue": 0}]}}]}
    cfg = load_config(doc)
    d = Scheduler(cfg, Datastore([ep("a", waiting=2)])).schedule(RequestCtx("r", "m", []))
    assert d.endpoint is None and "eliminated" in d.reason


def test_config_default_injection_like_upstream():
    cfg = load_config({"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
                       "plugins": [{"type": "prefix-cache-scorer"}, {"type": "active-request-scorer"}]})
    assert cfg.profile.name == "default" and [w for _, w in cfg.profile.scorers] == [1.0, 1.0]
    assert cfg.profile.picker.TYPE == "max-score-picker"
    assert [p.TYPE for p in cfg.producers] == ["approx-prefix-cache-producer", "inflight-load-producer"]
    assert {"metrics-data-source", "core-metrics-extractor", "single-profile-handler"} <= set(cfg.plugins)


def test_config_rejects_what_it_cannot_run():
    base = {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig"}
    for bad in ({**base, "apiVersion": "inference.networking.x-k8s.io/v1alpha1"},
                {**base, "plugins": [{"type": "no-such-scorer"}]},
                {**base, "plugins": [{"type": "queue-scorer", "parameters": {"typo": 1}}]},
                {**base, "plugins": [{"type": "queue-scorer"}], "schedulingProfiles": [
                    {"name": "prefill", "plugins": [{"pluginRef": "queue-scorer"}]},
                    {"name": "decode", "plugins": [{"pluginRef": "queue-scorer"}]}]},
                {**base, "plugins": [{"type": "queue-scorer"}], "schedulingProfiles": [
                    {"name": "default", "plugins": [{"pluginRef": "missing"}]}]},
                {**base, "surprise": 1}):
        with pytest.raises(ConfigError):
            load_config(bad)
    assert load_config({**base, "apiVersion": "llm-d.ai/v1alpha1"}).warnings   # deprecated but accepted
    fc = load_config({**base, "featureGates": ["flowControl"], "plugins": [{"type": "queue-scorer"}]})
    assert fc.flow_control and any("NOT implemented" in w for w in fc.warnings)   # accepted, never silently
    assert "legacy shedding" in fc.describe()


def test_to_upstream_strips_lab_only_parameters():
    doc = {"apiVersion": "llm-d.ai/v1alpha1", "kind": "EndpointPickerConfig",
           "plugins": [{"type": "approx-prefix-cache-producer", "parameters": {"ttlSeconds": 30, "blockSizeTokens": 64}},
                       {"type": "prefix-cache-scorer"}]}
    up = load_config(doc).to_upstream()
    assert up["apiVersion"] == "llm-d.ai/v1" and up["plugins"][0]["parameters"] == {"blockSizeTokens": 64}
    assert yaml.safe_load(load_config(doc).to_upstream_yaml()) == up


def test_every_preset_loads_and_plugin_types_are_upstream_names():
    from igwlab.router import preset_names
    from igwlab.router.plugins import REGISTRY
    assert {"round-robin", "default-weighted", "sticky-until-saturated"} <= set(preset_names())
    for n in preset_names():
        load_config(n)
    assert all(t == t.lower() and " " not in t for t in REGISTRY)
    with pytest.raises(ValueError):
        make_plugin("queue-scorer", parameters={"weight": 2})
