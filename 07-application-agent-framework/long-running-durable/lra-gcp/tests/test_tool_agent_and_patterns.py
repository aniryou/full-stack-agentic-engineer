"""A model-chosen tool loop (examples/tool_agent), scheduled ticks and long-running tools (patterns).

These carry over the durable-loop lessons of the layer's earlier labs: journal the model's decision
before acting on it, so a retry never re-asks the model; approve what you execute; a scheduler tick
is at-least-once; a slow tool parks the run on a back-off timer or a callback."""

from __future__ import annotations

from datetime import timedelta

import pytest

from lra import Engine, Event, RunStatus, SimulatedCrash, Workflow, Next, Done
from lra.adapters.memory import FakeClock, FakeLLM, InMemoryEventBus, InMemoryStateStore, InMemoryTaskQueue, LocalRunner
from lra.examples.tool_agent import PaymentAPI, ScriptedDecider, make_tool_agent
from lra.patterns.async_tool import backoff, job_key, poll_job
from lra.patterns.hitl import approval_key
from lra.patterns.scheduled import tick, window_of

CHARGE_THEN_DONE = [{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}]


def engine_for(*workflows, chaos=None):
    clock = FakeClock()
    store, queue = InMemoryStateStore(), InMemoryTaskQueue(clock)
    engine = Engine(store=store, queue=queue, bus=InMemoryEventBus(), llm=FakeLLM(), clock=clock,
                    workflows=list(workflows), worker_id="w", lease_ttl=timedelta(seconds=60), chaos=chaos)
    return engine, LocalRunner(engine, queue, clock), clock, store


def approve(engine, run, n=1, decision="approve", comment=""):
    return engine.resume(Event(run_id=run.run_id, key=approval_key_for(run, n),
                               payload={"decision": decision, "by": "cfo", "comment": comment}))


def approval_key_for(run, n):
    return f"tool-{n}:{run.run_id}"


def test_approval_key_matches_the_pattern():
    class Ctx:
        run_id = "r1"
    assert approval_key(Ctx(), "tool-1") == approval_key_for(type("R", (), {"run_id": "r1"}), 1)


def test_ungated_tool_runs_once_and_the_loop_finishes():
    pay, decider = PaymentAPI(), ScriptedDecider(CHARGE_THEN_DONE)
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}, gated=()))
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and r.result == {"result": "Charged 42."}
    assert pay.charges == {f"{run.run_id}:1": 42.0} and decider.calls == 2
    assert [e["type"] for e in r.state["journal"]] == ["decision", "intent", "decision"]


def test_crash_after_the_charge_retries_the_same_call_and_never_reasks_the_model():
    crashed = []

    def chaos(point, run):
        if point == "after_step_before_commit" and run.current_step == "act" and not crashed:
            crashed.append(1)
            raise SimulatedCrash()

    pay, decider = PaymentAPI(), ScriptedDecider(CHARGE_THEN_DONE)
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}, gated=()), chaos=chaos)
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    with pytest.raises(SimulatedCrash):
        runner.run_until_idle()
    assert pay.calls == 1                                    # the money moved; the checkpoint did not
    clock.advance(seconds=61)                                # the dead worker's lease expires
    engine.reap()
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED
    assert pay.calls == 1 and list(pay.charges.values()) == [42.0]   # the effect record, not a second call
    assert decider.calls == 2                                # decide, then final: the retry re-asked nothing


def test_crash_after_the_decision_reuses_the_recorded_decision():
    crashed = []

    def chaos(point, run):
        if point == "after_step_before_commit" and run.current_step == "decide" and not crashed:
            crashed.append(1)
            raise SimulatedCrash()

    pay = PaymentAPI()
    # a model that would choose differently if it were asked again
    decider = ScriptedDecider([{"tool": "charge", "args": {"amount": 42}}, {"tool": "charge", "args": {"amount": 99}},
                               {"final": "done"}])
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}, gated=()), chaos=chaos)
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    with pytest.raises(SimulatedCrash):
        runner.run_until_idle()
    clock.advance(seconds=61)
    engine.reap()
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and pay.charges == {f"{run.run_id}:1": 42.0, f"{run.run_id}:3": 99.0}
    assert decider.calls == 3                                # 42 (recorded, reused), 99, final: never 99 twice


def test_gate_parks_the_run_and_executes_exactly_what_was_approved():
    pay, decider = PaymentAPI(), ScriptedDecider(CHARGE_THEN_DONE)
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}))
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.WAITING and pay.calls == 0 and decider.calls == 1
    assert approve(engine, run) is not None
    assert approve(engine, run) is None                      # a double click is a no-op
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and pay.charges == {f"{run.run_id}:1": 42.0} and decider.calls == 2


def test_rejection_is_fed_back_to_the_model():
    pay = PaymentAPI()
    decider = ScriptedDecider([{"tool": "charge", "args": {"amount": 42}}, {"final": "Understood, not paying."}])
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}))
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    runner.run_until_idle()
    approve(engine, run, decision="reject", comment="vendor not onboarded")
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and pay.charges == {}
    assert r.state["journal"][1]["result"] == {"rejected": "vendor not onboarded"}


def test_approval_timeout_fails_the_run_and_the_step_budget_stops_a_model_that_never_finishes():
    pay, decider = PaymentAPI(), ScriptedDecider(CHARGE_THEN_DONE)
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}, approval_timeout=timedelta(hours=1)))
    run = engine.start("tool_agent", {"goal": "pay invoice 42"})
    runner.run_until_idle()
    clock.advance(hours=2)
    engine.reap()
    r = store.get(run.run_id)
    assert r.status == RunStatus.FAILED and "timed out" in r.error and pay.charges == {}

    pay, decider = PaymentAPI(), ScriptedDecider([{"tool": "charge", "args": {"amount": 1}}] * 10)
    engine, runner, clock, store = engine_for(make_tool_agent(decider, {"charge": pay}, gated=(), max_steps=6))
    run = engine.start("tool_agent", {"goal": "loop"})
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.FAILED and "budget" in r.error
    assert len(pay.charges) == 3                             # distinct keys, then the budget stopped it


# ---------------------------------------------------------------- scheduled ticks
def nightly():
    wf = Workflow("reconcile")

    @wf.step(start=True)
    def work(ctx):
        ctx.state["did"] = ctx.input["window"]
        return Done(ctx.input["window"])

    return wf


def test_duplicate_ticks_in_one_window_start_one_run():
    engine, runner, clock, store = engine_for(nightly())
    a = tick(engine, "reconcile", schedule="nightly", interval=timedelta(hours=24))
    b = tick(engine, "reconcile", schedule="nightly", interval=timedelta(hours=24))   # at-least-once delivery
    assert a.run_id == b.run_id == f"nightly:{window_of(clock.now(), timedelta(hours=24))}"
    runner.run_until_idle()
    assert len(store.list_runs()) == 1 and store.get(a.run_id).status == RunStatus.SUCCEEDED


def test_an_overlapping_tick_is_skipped_until_the_previous_run_is_done():
    wf = Workflow("slow")

    @wf.step(start=True)
    def work(ctx):
        return Next("finish") if ctx.state.setdefault("n", 0) else Done("x")

    @wf.step()
    def finish(ctx):
        return Done("x")

    engine, runner, clock, store = engine_for(wf)
    first = tick(engine, "slow", schedule="every5m", interval=timedelta(minutes=5))      # not drained yet
    clock.advance(minutes=5)
    assert tick(engine, "slow", schedule="every5m", interval=timedelta(minutes=5)) is None
    runner.run_until_idle()
    assert store.get(first.run_id).status == RunStatus.SUCCEEDED
    assert tick(engine, "slow", schedule="every5m", interval=timedelta(minutes=5)) is not None


# ---------------------------------------------------------------- long-running tools
class Job:
    def __init__(self, ready_after_checks):
        self.left, self.checks = ready_after_checks, 0

    def status(self):
        self.checks += 1
        self.left -= 1
        return {"rows": 1200} if self.left <= 0 else None


def export_workflow(job, deadline=timedelta(hours=6)):
    wf = Workflow("export")

    @wf.step(start=True)
    def submit(ctx):
        ctx.state["ticket"] = "job-1"
        return Next("wait_export")

    @wf.step()
    def wait_export(ctx):
        return poll_job(ctx, job="export", check=job.status, then="summarise", deadline=deadline,
                        base=timedelta(seconds=30), cap=timedelta(minutes=4))

    @wf.step()
    def summarise(ctx):
        return Done(ctx.state["jobs"]["export"])

    return wf


def drive_timers(engine, runner, clock, store, run_id, max_wakes=20):
    for _ in range(max_wakes):
        runner.run_until_idle()
        r = store.get(run_id)
        if r.status != RunStatus.WAITING:
            return r
        clock._now = r.wait.timeout_at                      # the reaper fires when the timer is due
        engine.reap()
    return store.get(run_id)


def test_backoff_doubles_and_is_capped():
    assert [backoff(i, timedelta(seconds=30), timedelta(minutes=4)).total_seconds() for i in range(6)] == \
        [30, 60, 120, 240, 240, 240]


def test_polling_parks_on_a_timer_between_checks_and_finishes():
    job = Job(ready_after_checks=4)
    engine, runner, clock, store = engine_for(export_workflow(job))
    run = engine.start("export", {})
    t0 = clock.now()
    r = drive_timers(engine, runner, clock, store, run.run_id)
    assert r.status == RunStatus.SUCCEEDED and job.checks == 4
    assert r.result["result"] == {"rows": 1200} and r.result["via"] == "poll" and r.result["polls"] == 3
    assert clock.now() - t0 == timedelta(seconds=30 + 60 + 120)


def test_a_callback_resumes_the_run_early_without_another_poll():
    job = Job(ready_after_checks=100)
    engine, runner, clock, store = engine_for(export_workflow(job))
    run = engine.start("export", {})
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.WAITING and r.wait.kind == "timer" and job.checks == 1
    engine.resume(Event(run_id=run.run_id, key=r.wait.key, payload={"result": {"rows": 7}}))   # the webhook
    runner.run_until_idle()
    r = store.get(run.run_id)
    assert r.status == RunStatus.SUCCEEDED and r.result["via"] == "callback" and job.checks == 1


def test_the_poll_deadline_fails_the_run():
    job = Job(ready_after_checks=10_000)
    engine, runner, clock, store = engine_for(export_workflow(job, deadline=timedelta(minutes=30)))
    run = engine.start("export", {})
    r = drive_timers(engine, runner, clock, store, run.run_id)
    assert r.status == RunStatus.FAILED and "not done after" in r.error


def test_job_key_is_run_scoped():
    class Ctx:
        run_id = "r9"
    assert job_key(Ctx(), "export") == "job:export:r9"
