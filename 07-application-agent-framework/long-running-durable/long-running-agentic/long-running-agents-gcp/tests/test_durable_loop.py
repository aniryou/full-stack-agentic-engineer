"""Pattern 1 — durable loop. Every test runs offline with the in-memory backends."""

import pytest

from lragents.core import (
    Budget,
    Decision,
    FakeClock,
    FaultInjector,
    InMemoryDispatcher,
    InMemoryIdempotencyStore,
    InMemoryRunStore,
    LeaseHeld,
    NaivePaymentGateway,
    PaymentGateway,
    PriceCard,
    RunStatus,
    ScriptedLLM,
    SimulatedCrash,
    StepKind,
    StepStatus,
    Tool,
    ToolError,
    ToolRegistry,
)
from lragents.patterns import DurableAgentLoop


def make_tools(gateway):
    def lookup(args, ctx):
        return {"price": 42.0, "sku": args["sku"]}

    def charge(args, ctx):
        return gateway.charge(float(args["amount"]), idempotency_key=ctx.idempotency_key)

    def flaky(args, ctx):
        raise ToolError("upstream 503")

    return ToolRegistry([
        Tool("lookup_price", "price of a sku", lookup),
        Tool("charge_card", "charge the card", charge),
        Tool("flaky", "always fails", flaky),
    ])


def make_loop(script, gateway=None, clock=None, faults=None, **kw):
    clock = clock or FakeClock()
    gateway = gateway or PaymentGateway()
    dispatcher = InMemoryDispatcher(clock=clock)
    loop = DurableAgentLoop(
        store=InMemoryRunStore(clock=clock), llm=ScriptedLLM(script), tools=make_tools(gateway),
        dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(), clock=clock,
        faults=faults or FaultInjector(), price=PriceCard(0.5, 3.0), **kw,
    )
    return loop, dispatcher, gateway, clock


SCRIPT = [Decision.call("lookup_price", sku="ABC"), Decision.call("charge_card", amount=42.0), Decision.final("Charged 42.00 for ABC")]


def test_happy_path_completes_and_journals_everything():
    loop, dispatcher, gateway, _ = make_loop(SCRIPT)
    run = loop.start("buy ABC")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert final.result == "Charged 42.00 for ABC"
    kinds = [(s.kind, s.status) for s in final.journal]
    assert kinds == [(StepKind.LLM, StepStatus.DONE), (StepKind.TOOL, StepStatus.DONE)] * 2 + [(StepKind.LLM, StepStatus.DONE)]
    assert len(gateway.charges) == 1
    assert final.usage.tokens > 0 and final.usage.cost_usd > 0
    assert final.lease is None


def test_crash_after_side_effect_does_not_double_charge():
    faults = FaultInjector()
    loop, dispatcher, gateway, clock = make_loop(SCRIPT, faults=faults)
    run = loop.start("buy ABC")
    dispatcher.deliver_one(loop.handle)                # step 0: lookup ok
    faults.crash_once_at("after_side_effect")          # die after charging, before checkpoint #2
    with pytest.raises(SimulatedCrash):
        loop.handle(dispatcher.queue.pop(0))           # step 1: charge → crash
    assert len(gateway.charges) == 1
    mid = loop.store.get(run.run_id)
    assert mid.pending_step() is not None and mid.pending_step().name == "charge_card"
    assert mid.lease is not None                       # the dead worker still "holds" the lease...
    with pytest.raises(LeaseHeld):                     # ...so an immediate retry is refused
        loop2 = DurableAgentLoop(store=loop.store, llm=loop.llm, tools=loop.tools, dispatcher=dispatcher,
                                 idempotency=loop.idem, worker_id="worker-2", clock=clock)
        loop2.step(run.run_id, expected_index=2)
    clock.advance(61)                                  # ...until it expires
    loop2.step(run.run_id, expected_index=2)           # recovery path re-executes the intent
    dispatcher.drain(loop2.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert len(gateway.charges) == 1                   # still exactly one charge
    assert final.state["recoveries"] == 1
    assert final.journal[3].output["replayed"] is True # the retry found the memoised result


def test_crash_after_intent_before_side_effect_executes_exactly_once():
    faults = FaultInjector()
    loop, dispatcher, gateway, clock = make_loop(SCRIPT, faults=faults)
    run = loop.start("buy ABC")
    dispatcher.deliver_one(loop.handle)
    faults.crash_once_at("after_intent")               # die after journaling the intent, before charging
    with pytest.raises(SimulatedCrash):
        loop.handle(dispatcher.queue.pop(0))
    assert gateway.charges == []                       # never charged
    clock.advance(61)
    loop.step(run.run_id, expected_index=2)
    dispatcher.drain(loop.handle)
    assert len(gateway.charges) == 1
    assert loop.store.get(run.run_id).status == RunStatus.SUCCEEDED


def test_duplicate_delivery_is_harmless():
    loop, dispatcher, gateway, _ = make_loop(SCRIPT)
    run = loop.start("buy ABC")
    dispatcher.deliver_one(loop.handle)
    dispatcher.duplicate_next()                        # Cloud Tasks delivers step 1 twice
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert len(gateway.charges) == 1
    assert len(final.journal) == 5                     # no duplicate journal entries


def test_naive_gateway_shows_why_idempotency_keys_matter():
    """Same crash, but the downstream API has no idempotency keys AND we bypass the
    memo store: the retry charges again. This is the bug the pattern prevents."""
    gateway = NaivePaymentGateway()

    class NoMemo(InMemoryIdempotencyStore):
        def __contains__(self, key):
            return False

    faults = FaultInjector()
    faults.crash_once_at("after_side_effect")
    clock = FakeClock()
    dispatcher = InMemoryDispatcher(clock=clock)

    def charge(args, ctx):
        return gateway.charge(float(args["amount"]))

    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM(SCRIPT[1:]),
                            tools=ToolRegistry([Tool("charge_card", "charge", charge)]), dispatcher=dispatcher,
                            idempotency=NoMemo(), clock=clock, faults=faults)
    run = loop.start("charge")
    with pytest.raises(SimulatedCrash):
        loop.handle(dispatcher.queue.pop(0))
    clock.advance(61)
    loop.step(run.run_id, expected_index=2)
    assert len(gateway.charges) == 2                   # double charge: no key, no memo


def test_budget_stops_runaway_loop():
    endless = [Decision.call("lookup_price", sku="X")] * 50
    loop, dispatcher, _, _ = make_loop(endless)
    run = loop.start("loop forever", budget=Budget(max_steps=6))
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.FAILED
    assert "max_steps" in final.error
    assert final.usage.steps == 6


def test_tool_error_becomes_an_observation_and_loop_continues():
    script = [Decision.call("flaky"), Decision.final("gave up gracefully")]
    loop, dispatcher, _, _ = make_loop(script)
    run = loop.start("try flaky")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert final.journal[1].status == StepStatus.FAILED
    assert "TOOL_ERROR[flaky]" in loop.messages(final)[-1]["content"]


def test_unknown_tool_is_surfaced_not_fatal():
    script = [Decision.call("no_such_tool"), Decision.final("ok")]
    loop, dispatcher, _, _ = make_loop(script)
    run = loop.start("x")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert any(s.kind == StepKind.SYSTEM for s in final.journal)


def test_deadline_budget():
    clock = FakeClock()
    loop, dispatcher, _, _ = make_loop(SCRIPT, clock=clock)
    run = loop.start("x", budget=Budget(deadline_epoch=clock() + 10))
    dispatcher.deliver_one(loop.handle)
    clock.advance(11)
    dispatcher.drain(loop.handle)
    assert loop.store.get(run.run_id).error == "budget: deadline passed"
