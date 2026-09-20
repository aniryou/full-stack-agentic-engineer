"""Patterns 2–7 and the HTTP surface. All offline."""

import json

import pytest
from fastapi.testclient import TestClient

from lragents.core import (
    Budget,
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
    SlowJobService,
    StepKind,
    Tool,
    ToolError,
    ToolRegistry,
)
from lragents.core.transport import Envelope
from lragents.patterns import (
    ApprovalError,
    DurableAgentLoop,
    FanOutFanIn,
    ReflectionLoop,
    SagaRunner,
    SagaStep,
    ScheduledAgent,
    approve,
    expire_stale_approvals,
)
from lragents.service.app import build_services, create_app


# ----------------------------------------------------------------- HITL
def hitl_loop(clock=None, notify=None):
    clock = clock or FakeClock()
    gateway = PaymentGateway()
    tools = ToolRegistry([
        Tool("charge_card", "charge", lambda a, c: gateway.charge(float(a["amount"]), idempotency_key=c.idempotency_key),
             requires_approval=True),
    ])
    dispatcher = InMemoryDispatcher(clock=clock)
    llm = ScriptedLLM([Decision.call("charge_card", amount=99.0), Decision.final("done")])
    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=llm, tools=tools, dispatcher=dispatcher,
                            idempotency=InMemoryIdempotencyStore(), clock=clock, notify=notify)
    return loop, dispatcher, gateway, clock


def test_hitl_parks_then_executes_exactly_what_was_approved():
    notified = []
    loop, dispatcher, gateway, _ = hitl_loop(notify=notified.append)
    run = loop.start("charge 99")
    dispatcher.drain(loop.handle)
    parked = loop.store.get(run.run_id)
    assert parked.status == RunStatus.WAITING_HUMAN and notified and gateway.charges == []
    assert dispatcher.pending() == 0                     # nothing scheduled while parked
    assert loop.llm.remaining == 1                       # LLM not consulted again
    with pytest.raises(ApprovalError):
        approve(loop, run.run_id, "wrong-token", True, "alice")
    approve(loop, run.run_id, parked.waiting_on["token"], True, "alice", "ok")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert len(gateway.charges) == 1 and gateway.charges[0].amount == 99.0
    assert [s.kind for s in final.journal] == [StepKind.LLM, StepKind.HUMAN, StepKind.TOOL, StepKind.LLM]
    # approving twice (double click / retried webhook) is a no-op
    approve(loop, run.run_id, parked.waiting_on["token"], True, "alice")
    assert len(gateway.charges) == 1


def test_hitl_rejection_feeds_back_to_the_model():
    loop, dispatcher, gateway, _ = hitl_loop()
    run = loop.start("charge 99")
    dispatcher.drain(loop.handle)
    tok = loop.store.get(run.run_id).waiting_on["token"]
    approve(loop, run.run_id, tok, False, "bob", "too expensive")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and gateway.charges == []
    assert "too expensive" in loop.llm.calls[-1][-1]["content"]


def test_stale_approval_expires():
    loop, dispatcher, _, clock = hitl_loop()
    run = loop.start("charge 99")
    dispatcher.drain(loop.handle)
    assert expire_stale_approvals(loop, ttl_s=3600) == []
    clock.advance(3601)
    escalated = []
    expired = expire_stale_approvals(loop, ttl_s=3600, escalate=escalated.append)
    assert [r.run_id for r in expired] == [run.run_id] and escalated
    assert loop.store.get(run.run_id).status == RunStatus.FAILED


# ----------------------------------------------------------- async tools
def async_loop(clock, use_callback=False):
    jobs = SlowJobService(clock, duration_s=120)
    tools = ToolRegistry([
        Tool("run_report", "start a long report job", lambda a, c: jobs.submit(a, idempotency_key=c.idempotency_key),
             is_async=True, poll=jobs.status),
    ])
    dispatcher = InMemoryDispatcher(clock=clock)
    llm = ScriptedLLM([Decision.call("run_report", region="apac"), lambda msgs: Decision.final(msgs[-1]["content"])])
    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=llm, tools=tools, dispatcher=dispatcher,
                            idempotency=InMemoryIdempotencyStore(), clock=clock, poll_base_s=10, poll_max_s=60)
    return loop, dispatcher, jobs


def test_async_tool_parks_and_polls_with_backoff():
    clock = FakeClock()
    loop, dispatcher, jobs = async_loop(clock)
    run = loop.start("report")
    dispatcher.drain(loop.handle)                        # submits, parks
    parked = loop.store.get(run.run_id)
    assert parked.status == RunStatus.WAITING_EVENT and dispatcher.pending() == 1
    delays = []
    while loop.store.get(run.run_id).status == RunStatus.WAITING_EVENT:
        nxt = dispatcher.queue[0].not_before
        delays.append(nxt - clock())
        clock.t = nxt
        dispatcher.drain(loop.handle)
    assert delays[:3] == [10, 20, 40] and max(delays) <= 60       # exponential, capped
    final = loop.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and "processed" in final.result
    assert [s.kind for s in final.journal] == [StepKind.LLM, StepKind.TOOL, StepKind.EVENT, StepKind.LLM]


def test_async_tool_callback_resumes_and_stale_callback_ignored():
    clock = FakeClock()
    loop, dispatcher, jobs = async_loop(clock)
    run = loop.start("report")
    dispatcher.drain(loop.handle)
    ticket = loop.store.get(run.run_id).waiting_on["ticket"]
    assert loop.resume_with_event(run.run_id, "wrong-ticket", {"x": 1}).status == RunStatus.WAITING_EVENT
    loop.resume_with_event(run.run_id, ticket, {"summary": "via webhook"})
    dispatcher.drain(loop.handle)
    assert loop.store.get(run.run_id).status == RunStatus.SUCCEEDED
    # the leftover scheduled poll is now a harmless no-op
    clock.advance(1000)
    dispatcher.drain(loop.handle)
    assert loop.store.get(run.run_id).status == RunStatus.SUCCEEDED


# ---------------------------------------------------------- fan-out/fan-in
def fan(faults=None, n=5):
    calls = []

    def worker(task):
        calls.append(task["title"])
        return {"summary": f"done {task['title']}"}

    subtasks = [{"title": f"t{i}", "instructions": "..."} for i in range(n)]
    planner = ScriptedLLM([Decision.final(json.dumps(subtasks))])
    aggregator = ScriptedLLM([lambda msgs: Decision.final("SYNTH:" + msgs[0]["content"])])
    dispatcher = InMemoryDispatcher()
    f = FanOutFanIn(store=InMemoryRunStore(), planner=planner, aggregator=aggregator, dispatcher=dispatcher,
                    worker=worker, idempotency=InMemoryIdempotencyStore(), faults=faults or FaultInjector())
    return f, dispatcher, calls


def test_fan_out_fan_in_aggregates_once():
    f, dispatcher, calls = fan()
    run = f.start("research 5 things")
    assert dispatcher.pending() == 5
    dispatcher.duplicate_next()                           # at-least-once on the first subtask
    dispatcher.drain(f.handle)
    final = f.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and final.result.startswith("SYNTH:")
    assert sorted(calls) == [f"t{i}" for i in range(5)]  # each worker ran exactly once
    assert final.state["fan"]["completed"] == 5
    assert sum(1 for e in dispatcher.delivered if e.kind == "aggregate") == 1


def test_fan_in_survives_crash_after_commit_before_enqueue():
    faults = FaultInjector()
    f, dispatcher, calls = fan(faults=faults, n=2)
    run = f.start("two things")
    dispatcher.deliver_one(f.handle)
    faults.crash_once_at("after_commit")                  # last worker commits, then dies before enqueuing aggregate
    dispatcher.drain(f.handle)                            # retry → duplicate observes all_done → enqueues (dedup-safe)
    final = f.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and len(calls) == 2


# ------------------------------------------------------------------ saga
def make_saga(fail_at=None, faults=None, stuck=None):
    log = []

    def action(name):
        def _a(ctx, key):
            if name == fail_at:
                raise ToolError(f"{name} unavailable")
            log.append(f"+{name}")
            return {"booking": f"{name}-{key[-4:]}"}
        return _a

    def comp(name):
        def _c(ctx, prior, key):
            log.append(f"-{name}")
        return _c

    steps = [SagaStep(n, action(n), comp(n)) for n in ("flight", "hotel", "card")]
    dispatcher = InMemoryDispatcher()
    runner = SagaRunner(store=InMemoryRunStore(), dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(),
                        steps=steps, faults=faults or FaultInjector(), on_stuck=stuck)
    return runner, dispatcher, log


def test_saga_happy_path():
    runner, dispatcher, log = make_saga()
    run = runner.start({"trip": "SIN-AMS"})
    dispatcher.drain(runner.handle)
    final = runner.store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED and log == ["+flight", "+hotel", "+card"]
    assert set(final.result) == {"flight", "hotel", "card"}


def test_saga_compensates_in_reverse_and_survives_crash():
    faults = FaultInjector()
    runner, dispatcher, log = make_saga(fail_at="card", faults=faults)
    run = runner.start({"trip": "SIN-AMS"})
    dispatcher.deliver_one(runner.handle)                 # +flight
    dispatcher.deliver_one(runner.handle)                 # +hotel
    dispatcher.deliver_one(runner.handle)                 # card fails → compensating
    faults.crash_once_at("after_side_effect")             # die right after compensating hotel
    dispatcher.drain(runner.handle)                       # retry finds the STARTED intent, memo says done
    final = runner.store.get(run.run_id)
    assert final.status == RunStatus.FAILED and "card failed" in final.error
    assert log == ["+flight", "+hotel", "-hotel", "-flight"]  # hotel compensated exactly once


def test_saga_stuck_compensation_escalates():
    stuck = []

    def bad_comp(ctx, prior, key):
        raise ToolError("refund API down")

    steps = [SagaStep("a", lambda c, k: "ok", bad_comp), SagaStep("b", lambda c, k: (_ for _ in ()).throw(ToolError("boom")), lambda c, p, k: None)]
    dispatcher = InMemoryDispatcher()
    runner = SagaRunner(store=InMemoryRunStore(), dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(), steps=steps,
                        on_stuck=lambda r, why: stuck.append(why))
    run = runner.start({})
    dispatcher.drain(runner.handle)
    assert "MANUAL INTERVENTION" in runner.store.get(run.run_id).error and stuck


# ------------------------------------------------------------- scheduled
def test_scheduled_agent_leases_and_due_times():
    clock = FakeClock()
    store = InMemoryRunStore(clock=clock)
    work_log = []
    a = ScheduledAgent(store=store, work=lambda st: work_log.append(clock()) or "checked", interval_s=300, clock=clock, lease_ttl_s=120)
    b = ScheduledAgent(store=store, work=lambda st: work_log.append("B"), interval_s=300, clock=clock, worker_id="worker-2", lease_ttl_s=120)
    a.ensure("presale-monitor")
    assert a.tick("presale-monitor").ran
    assert not a.tick("presale-monitor").ran             # duplicate tick: not due
    clock.advance(301)
    store.acquire_lease("presale-monitor", "zombie", 120)   # a dead worker still holds a lease
    assert "overlap" in b.tick("presale-monitor").reason
    clock.advance(121)                                    # lease expired
    assert b.tick("presale-monitor").ran
    assert len(work_log) == 2


# ------------------------------------------------------------ reflection
def test_reflection_stops_on_threshold_and_checkpoints_each_iteration():
    gen = ScriptedLLM([Decision.final("draft v1"), Decision.final("draft v2"), Decision.final("draft v3")])
    critic = ScriptedLLM([Decision.final(json.dumps({"score": 5, "feedback": "weak intro"})),
                          Decision.final(json.dumps({"score": 9, "feedback": "good"}))])
    dispatcher = InMemoryDispatcher()
    store = InMemoryRunStore()
    r = ReflectionLoop(store=store, generator=gen, critic=critic, dispatcher=dispatcher, threshold=8, max_iters=3)
    run = r.start("write an abstract")
    dispatcher.drain(r.handle)
    final = store.get(run.run_id)
    assert final.status == RunStatus.SUCCEEDED
    assert final.result == {"draft": "draft v2", "score": 9, "iterations": 2}
    assert [s.name for s in final.journal] == ["generate", "critique", "revise", "critique"]
    assert store.save_count >= 4                           # one checkpoint per phase


def test_reflection_iteration_cap():
    gen = ScriptedLLM([Decision.final(f"draft v{i}") for i in range(1, 5)])
    critic = ScriptedLLM([Decision.final(json.dumps({"score": 3, "feedback": "meh"}))] * 4)
    dispatcher = InMemoryDispatcher()
    r = ReflectionLoop(store=InMemoryRunStore(), generator=gen, critic=critic, dispatcher=dispatcher, threshold=8, max_iters=2)
    run = r.start("x")
    dispatcher.drain(r.handle)
    assert r.store.get(run.run_id).result["iterations"] == 2


# --------------------------------------------------------------- service
def test_http_service_end_to_end_with_cloud_tasks_style_delivery():
    svc = build_services("memory")
    client = TestClient(create_app(svc))
    assert client.get("/healthz").json()["backend"] == "memory"
    run_id = client.post("/runs", json={"goal": "buy ABC"}).json()["run_id"]
    # Play the role of Cloud Tasks: pop envelopes and POST them at the handler
    while svc.dispatcher.pending():
        env = svc.dispatcher.queue.pop(0)
        r = client.post(f"/internal/tasks/{env.kind}", json=env.__dict__)
        assert r.status_code == 200, r.text
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "WAITING_HUMAN"
    tok = body["waiting_on"]["token"]
    bad = client.post(f"/runs/{run_id}/approve", json={"token": "nope", "approved": True, "approver": "a"})
    assert bad.status_code == 403                             # bad token is rejected server-side
    r = client.post(f"/runs/{run_id}/approve", json={"token": tok, "approved": True, "approver": "alice"})
    assert r.json()["status"] == "RUNNING"
    while svc.dispatcher.pending():
        env = svc.dispatcher.queue.pop(0)
        client.post(f"/internal/tasks/{env.kind}", json=env.__dict__)
    assert client.get(f"/runs/{run_id}").json()["status"] == "SUCCEEDED"
    assert len(svc.gateway.charges) == 1
