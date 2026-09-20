"""Checks shared by the practice notebooks and ``tests/test_solutions.py``.

Each ``check_*`` takes the learner's implementation and raises AssertionError
with a helpful message, or returns a short success string.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .core import (
    Decision,
    FakeClock,
    FaultInjector,
    InMemoryDispatcher,
    InMemoryIdempotencyStore,
    InMemoryRunStore,
    PaymentGateway,
    RunStatus,
    ScriptedLLM,
    SimulatedCrash,
    StepKind,
    Tool,
    ToolError,
    ToolRegistry,
)


# ------------------------------------------------------------ exercise 1
def check_run_idempotent(fn: Callable) -> str:
    store = InMemoryIdempotencyStore()
    calls = []
    out1, replayed1 = fn(store, "k1", lambda: calls.append(1) or "first")
    out2, replayed2 = fn(store, "k1", lambda: calls.append(2) or "second")
    assert (out1, replayed1) == ("first", False), "first call must execute and report replayed=False"
    assert (out2, replayed2) == ("first", True), "second call must return the memoised result with replayed=True"
    assert calls == [1], "the side effect ran more than once"
    return "run_idempotent ✓"


def check_mini_loop(MiniLoop: type) -> str:
    """MiniLoop(store, llm, tools, idem, faults).step(run_id) must survive a crash after the side effect."""
    gateway = PaymentGateway()
    tools = ToolRegistry([
        Tool("charge_card", "charge", lambda a, c: gateway.charge(float(a["amount"]), idempotency_key=c.idempotency_key)),
    ])
    faults = FaultInjector()
    store, idem = InMemoryRunStore(), InMemoryIdempotencyStore()
    llm = ScriptedLLM([Decision.call("charge_card", amount=10.0), Decision.final("done")])
    loop = MiniLoop(store=store, llm=llm, tools=tools, idem=idem, faults=faults)
    run = loop.start("charge 10")
    faults.crash_once_at("after_side_effect")
    try:
        loop.step(run.run_id)
    except SimulatedCrash:
        pass
    else:
        raise AssertionError("expected the injected crash to propagate (do not swallow SimulatedCrash)")
    assert len(gateway.charges) == 1, "the charge should have happened once before the crash"
    mid = store.get(run.run_id)
    assert mid.pending_step() is not None, "the tool intent must be journaled (STARTED) before the side effect"
    loop.step(run.run_id)                         # recovery
    loop.step(run.run_id)                         # final answer
    final = store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED, f"run should succeed, got {final.status}"
    assert len(gateway.charges) == 1, f"double charge! {len(gateway.charges)} charges"
    assert llm.remaining == 0, "the LLM should have been asked exactly twice"
    return "mini durable loop ✓ (one charge across a crash)"


# ------------------------------------------------------------ exercise 2
def check_fan_in_mutate(mutate_factory: Callable[[str, Any], Callable]) -> str:
    """mutate_factory(subtask_id, result) -> mutate(run) -> all_done: bool"""
    from .core.models import Run

    run = Run(run_id="r", goal="g", state={"fan": {"expected": 3, "completed": 0, "results": {}}})
    assert mutate_factory("0", "a")(run) is False
    assert mutate_factory("0", "a")(run) is False, "duplicate delivery must not double count"
    assert run.state["fan"]["completed"] == 1
    assert mutate_factory("1", "b")(run) is False
    assert mutate_factory("2", "c")(run) is True, "the completion that reaches expected must report all_done"
    assert mutate_factory("2", "c")(run) is True, "a late duplicate must ALSO see all_done (idempotent enqueue handles it)"
    assert run.state["fan"]["completed"] == 3
    return "fan-in mutate ✓"


def check_fan_out_fan_in(FanClass: type) -> str:
    calls = []

    def worker(task):
        calls.append(task["title"])
        return {"summary": f"done {task['title']}"}

    subtasks = [{"title": f"t{i}", "instructions": "..."} for i in range(4)]
    dispatcher = InMemoryDispatcher()
    f = FanClass(store=InMemoryRunStore(), planner=ScriptedLLM([Decision.final(json.dumps(subtasks))]),
                 aggregator=ScriptedLLM([lambda msgs: Decision.final("SYNTH")]), dispatcher=dispatcher,
                 worker=worker, idempotency=InMemoryIdempotencyStore())
    run = f.start("four things")
    dispatcher.duplicate_next()
    dispatcher.drain(f.handle)
    if dispatcher.dead_letter:
        raise AssertionError(f"handle_subtask kept failing: {dispatcher.dead_letter[0][1]}")
    final = f.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and final.result == "SYNTH"
    assert sorted(calls) == ["t0", "t1", "t2", "t3"], f"workers ran {calls}"
    assert sum(1 for e in dispatcher.delivered if e.kind == "aggregate") == 1, "aggregate must run exactly once"
    return "fan-out/fan-in ✓"


# ------------------------------------------------------------ exercise 3
def check_approve(approve_fn: Callable) -> str:
    from .patterns import DurableAgentLoop

    gateway = PaymentGateway()
    tools = ToolRegistry([Tool("charge_card", "charge", lambda a, c: gateway.charge(float(a["amount"]), idempotency_key=c.idempotency_key),
                               requires_approval=True)])
    dispatcher = InMemoryDispatcher()
    llm = ScriptedLLM([Decision.call("charge_card", amount=5.0), Decision.final("ok")])
    loop = DurableAgentLoop(store=InMemoryRunStore(), llm=llm, tools=tools, dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore())
    run = loop.start("x")
    dispatcher.drain(loop.handle)
    parked = loop.store.get(run.run_id)
    assert parked.status == RunStatus.WAITING_HUMAN
    approve_fn(loop, run.run_id, parked.waiting_on["token"], True, "alice")
    assert llm.remaining == 1, "approve must NOT consult the LLM"
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED, final.error
    assert len(gateway.charges) == 1 and gateway.charges[0].amount == 5.0, "approved call must execute exactly once with approved args"
    assert [s.kind for s in final.journal] == [StepKind.LLM, StepKind.HUMAN, StepKind.TOOL, StepKind.LLM], [s.kind for s in final.journal]
    approve_fn(loop, run.run_id, parked.waiting_on["token"], True, "alice")   # second call: no-op
    assert len(gateway.charges) == 1
    return "approve ✓"


def check_saga_next(next_fn: Callable) -> str:
    """next_fn(order, saga_state) -> (step_name, phase) | ('__done__', 'succeeded'|'failed')"""
    order = ["flight", "hotel", "card"]
    assert next_fn(order, {"phase": "forward", "cursor": 0, "comp_cursor": None}) == ("flight", "action")
    assert next_fn(order, {"phase": "forward", "cursor": 2, "comp_cursor": None}) == ("card", "action")
    assert next_fn(order, {"phase": "forward", "cursor": 3, "comp_cursor": None}) == ("__done__", "succeeded")
    assert next_fn(order, {"phase": "compensating", "cursor": 2, "comp_cursor": 1}) == ("hotel", "compensate")
    assert next_fn(order, {"phase": "compensating", "cursor": 2, "comp_cursor": 0}) == ("flight", "compensate")
    assert next_fn(order, {"phase": "compensating", "cursor": 2, "comp_cursor": -1}) == ("__done__", "failed")
    return "saga next-step ✓"


# ------------------------------------------------------------ exercise 4
def _run(coro):
    """asyncio.run that also works inside Jupyter (which already runs a loop)."""
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def _t():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_t)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def check_adk_workflow(build_workflow_fn: Callable) -> str:

    from google.adk.apps import App, ResumabilityConfig
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from .adk.nightly_workflow import ASK_BUDGET, WAKE, Venue, run_until_interrupt

    async def go():
        venue = Venue()
        wf = build_workflow_fn(venue)
        assert getattr(wf, "edges", None), "the Workflow has no edges — wire START → agree_budget → ... in the edges list"
        app = App(name="p", root_agent=wf, resumability_config=ResumabilityConfig(is_resumable=True))
        svc = InMemorySessionService()
        runner = Runner(app=app, session_service=svc)
        sess = await svc.create_session(app_name="p", user_id="u", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
        r1 = await run_until_interrupt(runner, "u", sess.id, text="go")
        assert r1["interrupts"] == [ASK_BUDGET], f"first stop must be the budget gate, got {r1['interrupts']}"
        r2 = await run_until_interrupt(runner, "u", sess.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
        assert r2["interrupts"] == [WAKE], f"after the budget the run must park on the queue, got {r2['interrupts']}"
        st = (await svc.get_session(app_name="p", user_id="u", session_id=sess.id)).state
        venue.advance(st["ticket"], 5000)
        r3 = await run_until_interrupt(runner, "u", sess.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
        assert r3["interrupts"] == [WAKE], "still queued → must park again"
        assert venue.calls.count("join_queue") == 1, "queue_up re-ran on resume: set rerun_on_resume=False"
        venue.advance(st["ticket"], 99_999)
        await run_until_interrupt(runner, "u", sess.id, invocation_id=r3["invocation_id"], answers={WAKE: {"ok": True}})
        assert len(venue.orders) == 1, "exactly one purchase expected"
        assert venue.calls == ["join_queue", "purchase"], venue.calls
        return "adk workflow ✓"

    return _run(go())
