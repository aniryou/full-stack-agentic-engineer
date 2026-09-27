"""Fallback classes, the consecutive-failure breaker, chain planning and the chain arithmetic."""
import pytest

from gwlab.gateway.config import load_config
from gwlab.gateway.routing import FAIL, FALLTHROUGH, Breaker, Needs, Router, chain_availability, classify, expected_chain


@pytest.mark.parametrize("args, kw, want", [
    ((429,), {}, FALLTHROUGH), ((500,), {}, FALLTHROUGH), ((503,), {}, FALLTHROUGH), ((529,), {}, FALLTHROUGH),
    ((None,), {"timeout": True}, FALLTHROUGH), ((None,), {"connect_error": True}, FALLTHROUGH),
    ((400, "context_length_exceeded"), {}, FALLTHROUGH),
    ((400,), {}, FAIL), ((401,), {}, FAIL), ((403,), {}, FAIL), ((404,), {}, FAIL),
    ((400, "content_filter"), {}, FAIL), ((500, "content_filter"), {}, FAIL),
])
def test_what_falls_through(args, kw, want):
    assert classify(*args, **kw) == want


def test_breaker_counts_consecutive_failures_and_probes_once():
    t = [0.0]
    b = Breaker(threshold=3, recovery_s=5.0, clock=lambda: t[0])
    b.record(False)
    b.record(False)
    b.record(True)                                   # a success resets the count (07.2's rule)
    b.record(False)
    b.record(False)
    assert b.state == "closed"
    b.record(False)
    assert b.state == "open" and not b.allow() and b.rejections == 1
    t[0] = 5.0
    assert b.state == "half_open" and b.allow() and not b.allow()          # one probe at a time
    b.record(False)
    assert b.state == "open" and b.opens == 2
    t[0] = 10.0
    assert b.allow()
    b.record(True)
    assert b.state == "closed"


def test_plan_filters_orders_and_respects_residency():
    cfg = load_config("lab")
    r = Router(cfg)
    p = r.plan("chat", cfg.tenants["team-a"], Needs(100, 100))
    assert p.ids() == ["acme/fast", "bolt/haiku"]
    p = r.plan("chat", cfg.tenants["team-eu"], Needs(100, 100))
    assert p.ids() == ["bolt/haiku"] and p.skipped == [("acme/fast", "residency")]
    p = r.plan("chat", cfg.tenants["team-a"], Needs(9000, 500))
    assert p.ids() == ["bolt/haiku"] and ("acme/fast", "context_window") in p.skipped
    # cheapest for this shape: gpt-5.4-mini row ($0.75/$4.50) < haiku ($1/$5) < gemini-3.5-flash ($1.50/$9)
    assert r.plan("chat-cheap", None, Needs(1000, 200)).ids() == ["acme/fast", "bolt/haiku", "acme/strong"]


def test_ewma_and_canary_policies():
    cfg = load_config("lab", {"aliases": {"fast": {"policy": "ewma_ttft", "targets": ["acme/fast", "bolt/haiku"]},
                                          "canary": {"policy": "canary", "canary_weight": 0.1,
                                                     "targets": ["acme/fast", "bolt/haiku"]}},
                              "tenants": {"team-a": {"aliases": ["chat", "fast", "canary"]}}})
    r = Router(cfg)
    r.record("acme/fast", True, 0.8)
    r.record("bolt/haiku", True, 0.2)
    assert r.plan("fast", None, Needs(10, 10)).ids()[0] == "bolt/haiku"
    firsts = [r.plan("canary", None, Needs(10, 10, request_id=f"req-{i}")).ids()[0] for i in range(2000)]
    share = firsts.count("bolt/haiku") / len(firsts)
    assert 0.08 < share < 0.12
    assert firsts == [r.plan("canary", None, Needs(10, 10, request_id=f"req-{i}")).ids()[0] for i in range(2000)]


def test_chain_availability_and_expected_cost_by_hand():
    assert chain_availability([0.99, 0.99]) == pytest.approx(0.9999)
    assert chain_availability([0.99, 0.99], common_mode=0.001) == pytest.approx(0.999 * 0.9999)
    assert chain_availability([0.995]) == pytest.approx(0.995)
    # primary fails 10 % of the time after 50 ms; fallback answers in 400 ms at twice the price
    e = expected_chain([(0.1, 0.05, 0.3, 0.001), (0.0, 0.0, 0.4, 0.002)])
    assert e["p_success"] == pytest.approx(1.0)
    assert e["ttft_s"] == pytest.approx(0.9 * 0.3 + 0.1 * (0.05 + 0.4))
    assert e["cost_usd"] == pytest.approx(0.9 * 0.001 + 0.1 * 0.002)
