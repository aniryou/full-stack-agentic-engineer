"""One test per concept. Run with `pytest -q` (about 10 seconds)."""

import asyncio
import random

import pytest

from scalelab.admission import AdmissionConfig, AdmissionController
from scalelab.capacity import Scenario, cost_per_call, gsus_needed, plan
from scalelab.clock import CLOCK
from scalelab.loop import Budget, Store, run_turn
from scalelab.model import FakeModel, SharedPool
from scalelab.resilience import CircuitBreaker, CircuitOpen, RateLimited, TokenBucket, backoff, call_with_retries
from scalelab.sim import compare, make_setup, simulate
from scalelab.tools import Tools


@pytest.fixture(autouse=True)
def fast_clock():
    CLOCK.reset(0.02)  # 50x faster than real time


# --- capacity -------------------------------------------------------------------------------

def test_capacity_plan_reproduces_the_primer_numbers():
    p = plan(Scenario())
    assert abs(p["rates"]["peak"]["calls_per_s"] - 45.8) < 0.1
    assert abs(p["tokens"]["peak"]["vs_baseline"] - 1.375) < 0.01
    assert abs(p["concurrency"]["peak"]["inflight_turns"] - 125) < 0.5
    assert 84 < p["concurrency"]["max_inflight_for_baseline"] < 86
    assert 0.067 < p["cost"]["per_conversation_routed_cached"] < 0.069
    assert p["pt"]["gsus"]["average"] == 69
    assert 0.74 < p["pt"]["breakeven_utilisation"]["1y"] < 0.76
    assert p["breaks_first"][0]["resource"] == "model TPM baseline"


def test_docs_part3_figures_are_the_capacity_plan():
    """The primer and its long-form companion quote plan(Scenario()); pin the text to the model."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs"
    p = plan(Scenario())
    call = cost_per_call("gemini-3.5-flash", 5000, 350, 2700)  # the scenario's per-call working
    assert abs(call - (2300 * 1.50 + 2700 * 0.15 + 350 * 9.00) / 1e6) < 1e-12 and round(call, 4) == 0.0070
    figures = [  # (text as the doc prints it, the model's value, the number that text stands for)
        ("15.3", p["rates"]["average"]["calls_per_s"], 15.3),
        ("45.8", p["rates"]["peak"]["calls_per_s"], 45.8),
        ("13.75 M", p["tokens"]["peak"]["input_tpm"] / 1e6, 13.75),
        ("45.8 M", p["tokens"]["incident"]["input_tpm"] / 1e6, 45.8),
        ("**125**", p["concurrency"]["peak"]["inflight_turns"], 125),
        ("**417**", p["concurrency"]["incident"]["inflight_turns"], 417),
        ("$0.141", p["cost"]["per_conversation_no_cache"], 0.141),
        ("$0.092", p["cost"]["per_conversation_cached"], 0.092),
        ("**$0.068**", p["cost"]["per_conversation_routed_cached"], 0.068),
        ("69 GSUs", p["pt"]["gsus"]["average"], 69),
        ("688", p["pt"]["gsus"]["incident"], 688),
    ]
    text = (docs / "scaling-agentic-solutions-on-google-cloud.md").read_text()
    for quoted, value, number in figures:
        assert quoted in text, quoted
        assert abs(value - number) <= 0.006 * number, (quoted, value)  # rounding only
    for doc in ("01-scaling-primer.md", "scaling-agentic-solutions-on-google-cloud.md"):
        assert "2,300 uncached" in (docs / doc).read_text() and "2,000 uncached" not in (docs / doc).read_text()


def test_pricing_arithmetic():
    # 2,000 uncached × $1.50 + 3,000 cached × $0.15 + 350 out × $9 per 1M tokens
    assert abs(cost_per_call("gemini-3.5-flash", 5000, 350, 3000) - (3000 + 450 + 3150) / 1e6) < 1e-12
    assert abs(gsus_needed("gemini-3.5-flash", 10, 5000, 350, 3000) - 10 * (2000 + 300 + 2100) / 675) < 1e-9


# --- resilience -----------------------------------------------------------------------------

async def test_token_bucket_paces_to_its_rate():
    b = TokenBucket(rate=10, capacity=10)
    t0 = CLOCK.now()
    for _ in range(30):
        await b.acquire(1)
    assert 1.7 < CLOCK.now() - t0 < 2.6  # 10 from the burst, then 20 at 10/s


def test_backoff_is_capped_and_jittered():
    rng = random.Random(1)
    assert backoff(4, jitter=False) == 4.0 and backoff(9, jitter=False) == 8.0
    assert all(0 <= backoff(a, rng=rng) <= 8.0 for a in range(1, 10) for _ in range(20))


async def test_breaker_trips_on_failure_ratio_and_recovers():
    cb = CircuitBreaker(threshold=3, min_calls=4, ratio=0.5, cooldown=2.0)
    for ok in (True, False, False, False):
        cb.record(ok)
    assert cb.state == "open"
    with pytest.raises(CircuitOpen):
        cb.before_call()
    await CLOCK.sleep(2.1)
    assert cb.state == "half_open"
    cb.record(True)
    assert cb.state == "closed"


async def test_call_with_retries_honours_deadline():
    calls = []

    async def always_429():
        calls.append(1)
        raise RateLimited(retry_after=5.0)

    with pytest.raises(RateLimited):
        await call_with_retries(always_429, deadline=CLOCK.now() + 3.0)
    assert len(calls) == 1  # a 5 s hint cannot fit in a 3 s deadline: no second attempt


# --- the loop -------------------------------------------------------------------------------

async def test_turn_runs_tools_then_answers():
    r = await run_turn("t1", "my bill is higher than usual", model=FakeModel(), tools=Tools(), store=Store())
    assert r.status == "completed" and r.tool_names == ["get_customer", "get_invoice"] and r.steps == 5
    assert r.cost_usd > 0 and 4 < r.latency_s < 9


async def test_budget_ends_the_turn_gracefully():
    r = await run_turn("t2", "my bill is wrong", model=FakeModel(), tools=Tools(), store=Store(), budget=Budget(max_steps=2))
    assert r.status == "failed" and r.error == "budget:steps" and "sorry" in r.text


async def test_crash_and_resume_creates_one_ticket():
    store, tools, model = Store(), Tools(), FakeModel()

    class Crash(Exception):
        pass

    original = store.append_step

    def crash_after_ticket(turn_id, step):
        original(turn_id, step)
        if step["kind"] == "tool" and "create_ticket" in step["payload"]["tools"]:
            raise Crash()

    store.append_step = crash_after_ticket
    with pytest.raises(Crash):
        await run_turn("c1", "I want to complain to a human", model=model, tools=tools, store=store)
    store.append_step = original
    r = await run_turn("c1", "I want to complain to a human", model=model, tools=tools, store=store)  # the redelivery
    assert r.status == "completed" and r.resumed_steps == 4 and r.model_attempts == 1
    assert tools.tickets == ["T-1001"]  # exactly one side effect


async def test_degrade_level_2_withholds_writes():
    tools = Tools()
    r = await run_turn("d1", "I want to complain to a human", model=FakeModel(), tools=tools, store=Store(), degrade_level=2)
    assert "create_ticket" not in r.tool_names and tools.tickets == []


# --- admission --------------------------------------------------------------------------------

def test_admission_levels_and_shedding():
    ac = AdmissionController(AdmissionConfig(max_inflight=4, dwell_s=0))
    assert ac.admit().admitted and ac.level == 0
    for _ in range(3):
        ac.admit()
    d = ac.admit()
    assert not d.admitted and d.level == 3 and d.retry_after
    assert ac.admit(priority=1).admitted          # priority bypasses the cap
    for _ in range(10):
        ac.note_model_call(rate_limited=True)
    for _ in range(5):
        ac.release()
    assert ac.compute_level() == 2                # 67 % of calls rate limited


def test_level_hysteresis():
    ac = AdmissionController(AdmissionConfig(max_inflight=100, dwell_s=10))
    for _ in range(4):
        ac.note_model_call(True)
    assert ac.compute_level() == 2
    ac._recent.clear()
    assert ac.compute_level() == 2  # held for the dwell time


# --- simulation ---------------------------------------------------------------------------------

async def test_overload_regimes():
    results = {}
    for name, kw, users in [("naive", dict(pool_tpm=3_000_000, naive=True), 100),
                            ("capped", dict(pool_tpm=3_000_000, max_inflight=30), 100)]:
        CLOCK.reset(0.02)
        results[name] = await simulate(make_setup(**kw), users=users, duration_s=40, think_s=5)
    s = compare(results)
    assert s.loc["naive", "rate_limited_calls"] > 20 and s.loc["naive", "p95_s"] > s.loc["capped", "p95_s"]
    assert s.loc["capped", "rate_limited_calls"] == 0 and s.loc["capped", "shed"] > 0 and s.loc["capped", "failed"] == 0
