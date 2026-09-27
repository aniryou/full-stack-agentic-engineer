"""Exact and semantic caching: what is cacheable, the namespace, the entity guard, cache_salt, the sweep."""
import pytest

from gwlab.gateway.cache import (GatewayCache, HashingEmbedder, cache_salt, cacheable, canonical, classify_query, entities,
                                 load_sample, sweep, valid_cache_salt)
from gwlab.gateway.config import CacheCfg
from gwlab.gateway.store import Store


def body(q, system="You are helpful.", cls="faq", user=None, **kw):
    meta = {k: v for k, v in (("cache_class", cls), ("user", user)) if v is not None}
    return {"model": "chat", "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}],
            **({"metadata": meta} if meta else {}), **kw}


def test_what_is_cacheable_is_declared_not_inferred():
    assert cacheable(body("How do I export a report as CSV?")) == (True, "faq")
    assert cacheable(body("How do I export a report as CSV?", cls=None))[1].startswith("no cache_class declared")
    assert cacheable(body("How do I export a report as CSV?", cls="marketing"))[0] is False      # not allowlisted
    assert cacheable({**body("x"), "temperature": 0.7})[0] is False
    assert cacheable({k: v for k, v in body("x").items() if k != "temperature"})[0] is False     # default is 1
    assert cacheable(body("x", tools=[{"type": "function"}]))[1].startswith("tools")
    # the regex only vetoes a declared shared class; it never makes anything cacheable
    assert cacheable(body("What is the status of my order 1234?"))[1] == "declared 'faq', vetoed by the regex guard (personal)"
    assert cacheable(body("Is the service down right now?"))[1].endswith("(time_sensitive)")
    assert cacheable(body("What is the status of my order 1234?", cls=None))[0] is False
    # per-user classes need a user, and are not second-guessed by the regex
    assert cacheable(body("What is my plan limit?", cls="account"))[0] is False
    assert cacheable(body("What is my plan limit?", cls="account", user="alice")) == (True, "account")


def test_why_the_class_is_declared_the_regex_misreads_both_ways():
    assert classify_query("How do I cancel my subscription?") == "personal"      # an FAQ, read as personal
    assert classify_query("What is my plan limit?") == "general"                 # personal, read as general


def test_classifier_stand_in_agrees_with_the_sample_labels():
    for q in load_sample():
        assert classify_query(q["query"]) == q["class"], q


def test_canonical_ignores_transport_fields():
    a = body("hi", stream=True, user="u1", stream_options={"include_usage": True})
    assert canonical(a) == canonical(body("hi"))
    assert canonical(body("hi", max_completion_tokens=5)) != canonical(body("hi"))


def test_entities():
    assert entities("How many vCPUs does plan 4 include?") != entities("How many vCPUs does plan 8 include?")
    assert entities("error E1042") == entities("error e-1042") == ("e1042",)
    assert entities("the 2026-09-01 release") == ("20260901",)


def test_cache_salt_is_a_valid_vllm_salt_per_tenant():
    a, b = cache_salt("team-a", "s3cret"), cache_salt("team-b", "s3cret")
    assert len(a) == 43 and a != b and a == cache_salt("team-a", "s3cret") and a != cache_salt("team-a", "other")
    assert valid_cache_salt(a) and valid_cache_salt(b)
    for bad in ("", "a/b", "a@b", "a\\b", "a\x00", "x" * 129, None, 5):
        assert not valid_cache_salt(bad)
    assert valid_cache_salt("x" * 128)


def make_cache(**kw):
    t = [0.0]
    return GatewayCache(Store(), CacheCfg(**kw), clock=lambda: t[0]), t


def test_exact_then_semantic_hits_and_the_guard():
    c, _ = make_cache(threshold=0.8)
    resp = {"choices": [{"message": {"content": "4 vCPUs"}}]}
    assert c.lookup("team-a", "chat", body("How many vCPUs does plan 4 include?")).kind is None
    assert c.put("team-a", "chat", body("How many vCPUs does plan 4 include?"), resp)
    assert c.lookup("team-a", "chat", body("How many vCPUs does plan 4 include?")).kind == "exact"
    hit = c.lookup("team-a", "chat", body("how many vCPUs does plan 4 include"))
    assert hit.kind == "semantic" and hit.response == resp
    near = c.lookup("team-a", "chat", body("How many vCPUs does plan 8 include?"))
    assert near.kind is None and near.reason == "entity guard" and near.score > 0.8
    c.cfg.entity_guard = False
    assert c.lookup("team-a", "chat", body("How many vCPUs does plan 8 include?")).kind == "semantic"   # a false hit


def test_namespaces_ttl_and_invalidation():
    c, t = make_cache(ttl_s=10)
    c.put("team-a", "chat", body("How do I export a report as CSV?"), {"x": 1})
    assert c.lookup("team-b", "chat", body("How do I export a report as CSV?")).kind is None       # another tenant
    assert c.lookup("team-a", "chat-strong", body("How do I export a report as CSV?")).kind is None # another alias
    assert c.lookup("team-a", "chat", body("How do I export a report as CSV?", system="Be terse.")).kind is None
    assert c.lookup("team-a", "chat", body("How do I export a report as CSV?")).kind == "exact"
    t[0] = 11.0
    assert c.lookup("team-a", "chat", body("How do I export a report as CSV?")).kind is None       # expired
    c.put("team-a", "chat", body("How do I export a report as CSV?"), {"x": 1})
    assert c.invalidate("team-a") == 2 and c.lookup("team-a", "chat", body("How do I export a report as CSV?")).kind is None


def test_a_per_user_class_is_namespaced_by_user():
    c, _ = make_cache()
    c.put("team-a", "chat", body("What is my plan limit?", cls="account", user="alice"), {"answer": "alice: 10 seats"})
    assert c.lookup("team-a", "chat", body("What is my plan limit?", cls="account", user="alice")).kind == "exact"
    assert c.lookup("team-a", "chat", body("What is my plan limit?", cls="account", user="bob")).kind is None
    assert c.lookup("team-a", "chat", body("what is my plan limit", cls="account", user="bob")).kind is None
    assert c.namespace("t", "a", body("q", cls="account", user="x|y")) != c.namespace("t", "a", body("q", cls="account", user="x"))
    c.put("team-a", "chat", body("How do I export a report as CSV?", user="alice"), {"answer": "Reports > Export"})
    assert c.lookup("team-a", "chat", body("How do I export a report as CSV?", user="bob")).kind == "exact"   # faq: shared


def test_multi_turn_is_exact_only():
    c, _ = make_cache()
    b = {"model": "chat", "temperature": 0, "metadata": {"cache_class": "faq"},
         "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                      {"role": "user", "content": "How do I export a report as CSV?"}]}
    c.put("team-a", "chat", b, {"x": 1})
    assert c.store.query("SELECT COUNT(*) AS n FROM cache WHERE kind='semantic'")[0]["n"] == 0


def test_sweep_on_the_bundled_sample():
    s = load_sample()
    rows = {(r["threshold"], g): r for g in (False, True) for r in sweep(s, [0.8, 0.85, 0.9, 0.95], guard=g)}
    assert rows[(0.9, True)]["n"] == 52 and rows[(0.9, True)]["bypassed"] == 8
    assert (rows[(0.9, True)]["hits"], rows[(0.9, True)]["false_hits"]) == (9, 1)
    assert (rows[(0.95, True)]["hits"], rows[(0.95, True)]["false_hits"]) == (6, 0)
    assert (rows[(0.85, False)]["false_hits"], rows[(0.85, True)]["false_hits"]) == (9, 6)      # the guard's catch
    for thr in (0.8, 0.85, 0.9, 0.95):
        assert rows[(thr, True)]["false_hits"] <= rows[(thr, False)]["false_hits"]
    hr = [rows[(t, True)]["served_rate"] for t in (0.8, 0.85, 0.9, 0.95)]
    assert hr == sorted(hr, reverse=True)                                                        # stricter = fewer hits
    r = rows[(0.9, True)]
    assert r["correct_rate"] == pytest.approx(r["served_rate"] - r["false_hit_rate"]) and r["correct_rate"] <= r["reachable"]


def test_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    v = e.encode("How do I export a report as CSV?")
    assert v.dtype.name == "float32" and float(v @ v) == pytest.approx(1.0)
    assert float(v @ e.encode("how do i export a report as csv")) == pytest.approx(1.0)
    assert "not semantic" in e.model_name
