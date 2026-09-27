"""§2: which failures fall through, breakers per target, chain availability, expected latency and cost."""
import pytest

from gwcore import routing as R
from gwcore.metering import price_call


def test_what_falls_through_and_what_must_not():
    assert all(R.falls_through(s) for s in (408, 429, 500, 502, 503, 504, 529, None))   # 529: Anthropic's overloaded_error
    assert not any(R.falls_through(s) for s in (400, 401, 403, 404))
    assert R.falls_through(400, "context_length_exceeded") and not R.falls_through(400, "content_policy")


def test_breaker_opens_on_consecutive_failures_and_one_probe_decides():
    b = R.Breaker(threshold=3, cooldown=30)
    for t in (0, 1):
        b.record(False, t)
    b.record(True, 2)                                   # a success resets the count: consecutive, not total
    for t in (3, 4, 5):
        assert b.allow(t)
        b.record(False, t)
    assert b.state(6) == "open" and not b.allow(6)
    assert b.state(35) == "half_open" and b.allow(35) and not b.allow(35)   # exactly one probe
    b.record(False, 35)
    assert b.state(36) == "open" and b.trips == 2
    assert b.allow(65)
    b.record(True, 65)
    assert b.state(66) == "closed"


def test_a_probe_without_a_health_verdict_is_handed_back():
    b = R.Breaker(threshold=1, cooldown=30)
    assert b.allow(0)
    b.record(False, 0)
    assert b.allow(30) and not b.allow(30)             # the probe is out
    b.release()                                         # ... and came back with a 400: it decided nothing
    assert b.state(31) == "half_open" and b.allow(31)   # so the next request probes, instead of refusing forever


def test_capability_context_and_region_filters():
    chain = [R.Target("anthropic", "claude-haiku-4-5"), R.Target("openai", "gpt-5.4-mini"), R.Target("google", "gemini-3.5-flash"),
             R.Target("self", "lab/llm", "us-central1")]
    r = R.Router({"chat": chain})
    assert [t.model for t in r.candidates("chat", {"tools": [{}]}, now=0, prompt_tokens=100)] == [t.model for t in chain[:3]]
    assert [t.model for t in r.candidates("chat", {}, now=0, prompt_tokens=280_000)] == ["gemini-3.5-flash"]
    assert [t.model for t in r.candidates("chat", {"reasoning_effort": "high"}, now=0, regions={"us-central1"})] == []
    assert [t.model for t in r.candidates("chat", {"max_completion_tokens": 500}, now=0, prompt_tokens=100,
                                          policy="cheapest")][:2] == ["lab/llm", "gpt-5.4-mini"]


def test_tier_chains_ewma_and_canary():
    gold, std = [R.Target("openai", "gpt-5.4-mini")], [R.Target("google", "gemini-3.5-flash-lite")]
    r = R.Router({"chat": std, "chat@gold": gold})
    assert r.candidates("chat", {}, now=0, tier="gold") == gold and r.candidates("chat", {}, now=0) == std
    assert R.ewma(None, 1.0) == 1.0 and R.ewma(1.0, 2.0, 0.2) == pytest.approx(1.2)
    a, b = R.Target("a", "gpt-5.4-mini", weight=9), R.Target("b", "claude-haiku-4-5", weight=1)
    r = R.Router({"chat": [a, b]})
    first = [r.candidates("chat", {}, now=0, policy="canary", request_id=f"r{i}")[0] for i in range(2000)]
    assert 0.07 < first.count(b) / 2000 < 0.13        # ~10 % to the canary, stable per request id
    assert r.candidates("chat", {}, now=0, policy="canary", request_id="r7") == r.candidates("chat", {}, now=0, policy="canary", request_id="r7")


def test_chain_availability_independent_and_common_mode():
    assert R.chain_availability([0.995, 0.99]) == pytest.approx(0.99995)            # 1 - 0.005 x 0.01
    assert R.chain_availability([0.995, 0.99], 0.001) == pytest.approx(0.99895005)  # capped by the shared 0.1 %
    assert R.chain_availability([0.995, 0.99, 0.99], 0.001) < 0.999


def test_chain_cost_a_timeout_dominates_latency():
    c = {m: price_call(m, 5000, 350, 2700) for m in ("gemini-3.5-flash", "gpt-5.4-mini", "claude-haiku-4-5")}
    steps = lambda tf: [dict(p_ok=0.95, t_ok=0.6, t_fail=tf, cost_ok=c["gemini-3.5-flash"]),
                        dict(p_ok=0.99, t_ok=0.5, t_fail=0.15, cost_ok=c["gpt-5.4-mini"]),
                        dict(p_ok=0.99, t_ok=0.7, t_fail=0.15, cost_ok=c["claude-haiku-4-5"])]
    fast, slow = R.chain_cost(steps(0.15)), R.chain_cost(steps(10.0))
    # hand: 0.95*0.6 + 0.05*0.15 + 0.05*(0.99*0.5 + 0.01*0.15) + 0.0005*(0.99*0.7 + 0.01*0.15)
    assert fast["latency"] == pytest.approx(0.57 + 0.0075 + 0.05 * 0.4965 + 0.0005 * 0.6945)
    assert slow["latency"] - fast["latency"] == pytest.approx(0.05 * (10 - 0.15))
    assert fast["p_fail"] == pytest.approx(0.05 * 0.01 * 0.01) and fast["cost"] == slow["cost"]
    assert fast["cost"] == pytest.approx(0.95 * 0.007005 + 0.05 * (0.99 * 0.0035025 + 0.01 * 0.99 * 0.00432))
