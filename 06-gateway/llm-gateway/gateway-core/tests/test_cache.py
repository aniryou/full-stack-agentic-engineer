"""§3: what may be cached, tenant namespaces, the hashing embedder, and the threshold sweep on labelled traffic."""
import numpy as np
import pytest

from gwcore import cache as C

SWEEP = [0.6, 0.8, 0.9, 0.95]


def test_route_declares_what_is_cacheable():
    assert C.cacheable({"metadata": {"cache_class": "faq"}}) == (True, "faq")
    assert not C.cacheable({"metadata": {"cache_class": "personal"}})[0]
    assert not C.cacheable({"metadata": {"cache_class": "faq"}, "tools": [{}]})[0]
    assert not C.cacheable({})[0]                      # nothing declared: not cacheable


def test_exact_key_is_namespaced_and_ignores_stream():
    req = {"model": "chat", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    assert C.exact_key("acme", req) != C.exact_key("globex", req)
    assert C.exact_key("acme", req) == C.exact_key("acme", {**req, "stream": True, "user": "u1"})
    assert C.exact_key("acme", req) != C.exact_key("acme", {**req, "temperature": 1})


def test_exact_cache_ttl():
    c = C.ExactCache(ttl=60)
    c.put("k", "v", now=0)
    assert c.get("k", 59) == "v" and c.get("k", 60) is None


def test_hashing_embedder_is_bag_of_words():
    e = C.HashingEmbedder()
    v = e.encode("Reset my password")
    assert v.dtype == np.float32 and np.isclose(np.linalg.norm(v), 1.0)
    assert float(e.encode("my password reset!") @ v) == pytest.approx(1.0)      # order and punctuation vanish
    assert float(e.encode("holiday") @ e.encode("vacation")) == 0.0               # lexical, not semantic


def test_tenants_never_share_a_semantic_namespace():
    c = C.SemanticCache(0.9)
    c.store("acme", "How do I reset my password?", "acme-answer")
    assert c.lookup("globex", "How do I reset my password?") is None
    assert c.lookup("acme", "how do i reset my password")[0] == "acme-answer"


def test_entity_guard_blocks_a_different_date():
    c = C.SemanticCache(0.8, guard=False)
    c.store("t", "What were the Q3 2025 revenue figures?", "q3-2025")
    assert c.lookup("t", "What were the Q3 2024 revenue figures?")[0] == "q3-2025"   # sim 0.882: a false hit
    c.guard = True
    assert c.lookup("t", "What were the Q3 2024 revenue figures?") is None
    assert C.entities("order 1234 in Q3 2025 via SSO") == {"1234", "Q3", "2025", "SSO"}


def test_sweep_numbers_pinned():
    s = C.load_sample()
    assert sum(p["label"] == "same" for p in s["probes"]) == 23 and sum(p["label"] == "near_miss" for p in s["probes"]) == 15
    off = {r["threshold"]: (r["right"], r["wrong"]) for r in C.sweep_thresholds(SWEEP, guard=False)}
    on = {r["threshold"]: (r["right"], r["wrong"]) for r in C.sweep_thresholds(SWEEP, guard=True)}
    assert off == {0.6: (23, 15), 0.8: (11, 13), 0.9: (5, 2), 0.95: (4, 0)}
    assert on == {0.6: (22, 8), 0.8: (10, 7), 0.9: (4, 2), 0.95: (3, 0)}


def test_the_lexical_embedder_ranks_near_misses_above_paraphrases():
    s, e = C.load_sample(), C.HashingEmbedder()
    seeds = {x["id"]: e.encode(x["q"]) for x in s["seeds"]}
    sim = lambda label: np.mean([seeds[p["seed"]] @ e.encode(p["q"]) for p in s["probes"] if p["label"] == label])
    assert sim("near_miss") > sim("same")               # why no threshold separates them on this embedder
