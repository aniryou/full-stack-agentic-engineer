import asyncio
import random

import pytest

from agentlab.agents import ToolContext, ToolTransientError, tool
from agentlab.agents.state import TaskStore
from agentlab.llm import FakeLLM, scripted
from agentlab.reliability import (BreakerState, Bulkhead, BulkheadFull, CircuitBreaker, CircuitOpen, Deadline,
                                DeadlineExceeded, FallbackChain, FallbackExhausted, GracefulTool, IdempotentCall,
                                PermanentError, RetryableError, RetryPolicy, backoff_schedule, classify_http,
                                is_retryable, lab_fallback_chain, per_hop_budget, retry, with_deadline, with_retry)


# ------------------------------------------------------------------ helpers
class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Sleeps:
    """Records requested sleeps instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class LossyTransport:
    """The ambiguous failure: the operation is applied, then the response is lost."""

    def __init__(self, lose_first: int = 1) -> None:
        self.lose_first = lose_first
        self.lost = 0

    async def send(self, fn):
        result = await fn()
        if self.lost < self.lose_first:
            self.lost += 1
            raise asyncio.TimeoutError("response lost after the write was applied")
        return result


# ------------------------------------------------------------------- retry
def test_classify_http_retryable_vs_permanent():
    assert all(classify_http(s) for s in (408, 429, 502, 503, 504))
    assert not any(classify_http(s) for s in (400, 401, 403, 404, 409, 422, 500, 501))


def test_is_retryable_by_type_and_status():
    assert is_retryable(RetryableError("503")) and is_retryable(asyncio.TimeoutError()) and is_retryable(ToolTransientError("x"))
    assert not is_retryable(PermanentError("400")) and not is_retryable(KeyError("bug"))
    assert is_retryable(RetryableError("rate limited", status=429))
    assert not is_retryable(PermanentError("bad request", status=400))


def test_backoff_schedule_is_capped_exponential_with_jitter():
    exact = backoff_schedule(RetryPolicy(max_attempts=7, base_s=0.5, cap_s=8.0, jitter_s=0.0), random.Random(0))
    assert exact == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    jittered = backoff_schedule(RetryPolicy(), random.Random(0))
    assert [round(d, 4) for d in jittered] == [0.7533, 1.2274, 2.1262]
    assert all(0.0 <= j - e <= 0.3 for j, e in zip(jittered, [0.5, 1.0, 2.0]))
    assert backoff_schedule(RetryPolicy(max_attempts=1), random.Random(0)) == []


async def test_retry_recovers_from_transient_failures_and_reports_attempts():
    sleeps, seen, calls = Sleeps(), [], {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("503", status=503)
        return "ok"

    result = await retry(flaky, RetryPolicy(), sleep=sleeps, on_attempt=lambda a, e, d: seen.append((a, type(e).__name__)))
    assert result == "ok" and calls["n"] == 3
    assert seen == [(1, "RetryableError"), (2, "RetryableError")]
    assert [round(s, 4) for s in sleeps.calls] == [0.7533, 1.2274]


async def test_retry_stops_immediately_on_permanent_error():
    sleeps, calls = Sleeps(), {"n": 0}

    async def broken():
        calls["n"] += 1
        raise PermanentError("404 not found", status=404)

    with pytest.raises(PermanentError):
        await retry(broken, RetryPolicy(max_attempts=5), sleep=sleeps)
    assert calls["n"] == 1 and sleeps.calls == []


async def test_retry_gives_up_after_max_attempts_and_honours_retry_after():
    sleeps, calls = Sleeps(), {"n": 0}

    async def always_429():
        calls["n"] += 1
        raise RetryableError("rate limited", status=429, retry_after_s=5.0)

    with pytest.raises(RetryableError):
        await retry(always_429, RetryPolicy(max_attempts=3), sleep=sleeps)
    assert calls["n"] == 3 and sleeps.calls == [5.0, 5.0]   # server hint beats the 0.75 s / 1.2 s schedule


async def test_with_retry_decorator():
    sleeps, calls = Sleeps(), {"n": 0}

    @with_retry(RetryPolicy(max_attempts=3, jitter_s=0.0), sleep=sleeps)
    async def fetch(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.TimeoutError()
        return x * 2

    assert await fetch(21) == 42 and sleeps.calls == [0.5]


async def test_naive_retry_double_applies_but_idempotent_call_does_not():
    ledger: list[tuple[str, float]] = []

    async def refund(order_id: str, amount: float) -> dict:
        ledger.append((order_id, amount))
        return {"refunded": amount}

    naive = LossyTransport(lose_first=1)
    await retry(lambda: naive.send(lambda: refund("O-1", 20.0)), RetryPolicy(jitter_s=0.0), sleep=Sleeps())
    assert ledger == [("O-1", 20.0), ("O-1", 20.0)], "timeout-after-success made the naive retry refund twice"

    ledger.clear()
    keyed = IdempotentCall()
    safe = LossyTransport(lose_first=1)
    result = await retry(lambda: safe.send(lambda: keyed("refund:O-1:turn-3", lambda: refund("O-1", 20.0))),
                         RetryPolicy(jitter_s=0.0), sleep=Sleeps())
    assert result == {"refunded": 20.0}
    assert ledger == [("O-1", 20.0)] and keyed.applied == 1 and keyed.replayed == 1


# ------------------------------------------------------------------ breaker
async def test_breaker_state_transitions_with_fake_clock():
    clock = FakeClock()
    br = CircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0, half_open_max_calls=1, clock=clock)

    async def down():
        raise RetryableError("503")

    async def up():
        return "ok"

    for _ in range(3):
        with pytest.raises(RetryableError):
            await br.call(down)
    assert br.state is BreakerState.OPEN and br.metrics.opens == 1 and br.metrics.failures == 3

    with pytest.raises(CircuitOpen) as info:
        await br.call(up)
    assert info.value.retry_after_s == 30.0 and br.metrics.rejections == 1

    clock.advance(29.9)
    assert br.state is BreakerState.OPEN
    clock.advance(0.1)
    assert br.state is BreakerState.HALF_OPEN

    with pytest.raises(RetryableError):        # failed probe: back to open, timer restarted
        await br.call(down)
    assert br.state is BreakerState.OPEN and br.metrics.opens == 2

    clock.advance(30.0)
    assert await br.call(up) == "ok"           # successful probe closes the circuit
    assert br.state is BreakerState.CLOSED


async def test_breaker_half_open_admits_bounded_probes_and_ignores_business_errors():
    clock = FakeClock()
    br = CircuitBreaker(failure_threshold=1, recovery_timeout_s=10.0, half_open_max_calls=1, clock=clock)
    with pytest.raises(PermanentError):        # a 404 says nothing about dependency health
        await br.call(lambda: (_ for _ in ()).throw(PermanentError("404")))
    assert br.state is BreakerState.CLOSED and br.metrics.failures == 0

    with pytest.raises(RetryableError):
        await br.call(lambda: (_ for _ in ()).throw(RetryableError("503")))
    clock.advance(10.0)
    assert br.state is BreakerState.HALF_OPEN

    gate = asyncio.Event()

    async def slow_probe():
        await gate.wait()
        return "probe ok"

    probe = asyncio.create_task(br.call(slow_probe))
    await asyncio.sleep(0)
    with pytest.raises(CircuitOpen):           # only one probe allowed while half-open
        await br.call(lambda: "second")
    gate.set()
    assert await probe == "probe ok" and br.state is BreakerState.CLOSED


async def test_bulkhead_rejects_beyond_slots_plus_queue():
    bh = Bulkhead(max_concurrent=2, max_queue=1)
    release = asyncio.Event()

    async def slow():
        await release.wait()
        return "done"

    tasks = [asyncio.create_task(bh.run(slow)) for _ in range(5)]
    await asyncio.sleep(0)
    assert bh.active == 2 and bh.waiting == 1 and bh.rejections == 2
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(r == "done" for r in results) == 3 and sum(isinstance(r, BulkheadFull) for r in results) == 2
    assert bh.active == 0 and bh.peak_active == 2


async def test_deadline_and_per_hop_budget():
    clock = FakeClock()
    turn = Deadline(10.0, clock=clock)
    clock.advance(7.5)
    assert turn.remaining() == 2.5 and not turn.expired()
    assert turn.sub(5.0).seconds == 2.5          # a hop never outlives its turn
    clock.advance(3.0)
    assert turn.expired()

    async def never():
        await asyncio.sleep(10)

    with pytest.raises(DeadlineExceeded):        # already expired: rejected without waiting
        await with_deadline(never(), turn)
    with pytest.raises(DeadlineExceeded):        # real clock, tiny budget: cut off
        await with_deadline(never(), Deadline(0.01))
    assert await with_deadline(asyncio.sleep(0, result="fast"), Deadline(1.0)) == "fast"

    assert per_hop_budget(10.0, 4, reserve_s=2.0) == 2.0
    with pytest.raises(ValueError):
        per_hop_budget(2.0, 3, reserve_s=2.0)


# ----------------------------------------------------------------- fallback
async def test_fallback_chain_marks_degraded_and_records_errors():
    async def primary(q):
        raise RetryableError("primary overloaded")

    async def smaller(q):
        return f"small:{q}"

    chain = FallbackChain([("primary", primary), ("smaller", smaller)])
    r = await chain.run("hello")
    assert r.value == "small:hello" and r.step == "smaller" and r.degraded
    assert r.errors == [("primary", "RetryableError: primary overloaded")]
    assert chain.served["smaller"] == 1 and chain.failed["primary"] == 1 and chain.degraded_share() == 1.0

    ok_chain = FallbackChain([("primary", smaller)])
    assert not (await ok_chain.run("x")).degraded

    with pytest.raises(FallbackExhausted):
        await FallbackChain([("only", primary)]).run("x")


async def test_lab_chain_degrades_to_follow_up_task():
    class DownLLM(FakeLLM):
        async def generate(self, messages, tools=None, **o):
            raise RetryableError("503 model overloaded")

    tasks = TaskStore()
    chain = lab_fallback_chain(DownLLM(), DownLLM(), cache={"What is the fee?": "SGD 5"}, tasks=tasks)
    hit = await chain.run("What is the fee?")
    assert hit.step == "cached_answer" and hit.value == "SGD 5" and hit.degraded
    miss = await chain.run("Close my account")
    assert miss.step == "enqueue_followup" and miss.value["acknowledged"]
    assert tasks.get(miss.value["task_id"]).checkpoint == {"request": "Close my account"}

    healthy = lab_fallback_chain(scripted("The fee is SGD 5."), DownLLM(), cache={}, tasks=tasks)
    assert not (await healthy.run("What is the fee?")).degraded


async def test_graceful_tool_reports_unavailable_instead_of_raising():
    calls = {"n": 0}

    @tool
    def legacy_lookup(q: str) -> dict:
        """Legacy system."""
        calls["n"] += 1
        raise ToolTransientError("503 from legacy")

    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_s=60.0, clock=clock)
    graceful = GracefulTool(legacy_lookup, breaker, RetryPolicy(max_attempts=5, jitter_s=0.0), sleep=Sleeps())
    assert graceful.spec is legacy_lookup.spec

    r = await graceful.run({"q": "x"}, ToolContext())
    assert not r.ok and r.error.type == "unavailable" and r.error.retryable
    assert "do not call it again" in r.error.hint.lower()
    assert calls["n"] == 3 and breaker.state is BreakerState.OPEN    # retries stopped when the circuit opened

    r2 = await graceful.run({"q": "x"}, ToolContext())
    assert r2.error.type == "unavailable" and calls["n"] == 3        # open circuit: dependency not touched

    exhausted = GracefulTool(legacy_lookup, CircuitBreaker(failure_threshold=10, clock=clock), RetryPolicy(max_attempts=2, jitter_s=0.0), sleep=Sleeps())
    r3 = await exhausted.run({"q": "x"}, ToolContext())
    assert r3.error.type == "transient" and calls["n"] == 5          # retries exhausted before the breaker tripped


async def test_graceful_tool_with_bulkhead_reports_overloaded():
    release = asyncio.Event()

    @tool
    async def slow(q: str) -> dict:
        """Slow."""
        await release.wait()
        return {"q": q}

    graceful = GracefulTool(slow, CircuitBreaker(), RetryPolicy(max_attempts=1), bulkhead=Bulkhead(1, max_queue=0))
    first = asyncio.create_task(graceful.run({"q": "a"}, ToolContext()))
    await asyncio.sleep(0)
    second = await graceful.run({"q": "b"}, ToolContext())
    assert second.error.type == "overloaded" and second.error.retryable
    release.set()
    assert (await first).ok
