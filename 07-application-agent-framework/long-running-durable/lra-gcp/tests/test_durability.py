"""Durable execution: every arrow in the engine docstring is a test here."""

from __future__ import annotations

from datetime import timedelta

import pytest

from lra import Budget, Done, Engine, LeaseHeldError, Next, RunStatus, SimulatedCrash, StepFailed, StepTask, Workflow
from lra.adapters.memory import LocalRunner
from tests.conftest import Harness, make_harness


def test_research_pipeline_happy_path(h: Harness) -> None:
    run = h.start("research_pipeline", {"goal": "durable agents"})
    h.drain()

    r = h.run(run.run_id)
    assert r.status == RunStatus.WAITING and r.wait and r.wait.kind == "approval"
    assert r.current_step == "publish"
    children = [c for c in h.store.list_runs() if c.parent_run_id == run.run_id]
    assert len(children) == 3 and all(c.status == RunStatus.SUCCEEDED for c in children)
    assert r.state["reflect_loop"]["exit_reason"].startswith("threshold met")

    assert h.approve(run.run_id) is not None
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.SUCCEEDED
    assert r.result["post_id"].startswith("post_") and r.result["iterations"] == 1
    assert [s for s, _, _ in h.history(run.run_id)] == [
        "plan", "synthesize", "reflect_critique", "reflect_revise", "reflect_critique", "request_review", "publish", "notify",
    ]
    assert r.budget.steps_used == 8 and r.budget.cost_usd > 0


def test_crash_after_commit_before_enqueue_is_repaired_by_reaper() -> None:
    """Worker dies after checkpointing step N but before enqueueing N+1."""
    crashed: list[str] = []

    def chaos(point: str, run) -> None:
        if point == "after_commit_before_enqueue" and run.current_step == "charge_payment" and not crashed:
            crashed.append(run.run_id)
            raise SimulatedCrash()

    h = make_harness(chaos=chaos, lease_ttl_s=60)
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 10})
    with pytest.raises(SimulatedCrash):
        h.drain()

    r = h.run(run.run_id)
    assert r.status == RunStatus.RUNNING and r.current_step == "charge_payment"
    assert r.lease is not None, "a crashed worker never releases its lease"

    # Cloud Tasks redelivers the crashed task; it is stale (already committed) and acked.
    h.drain()
    assert h.runner.trace[-1][1] == "stale" and len(h.queue) == 0, "the *next* task was never enqueued"
    assert h.run(run.run_id).status == RunStatus.RUNNING

    # Nothing happens until the lease expires...
    assert h.engine.reap()["leases_recovered"] == []
    h.clock.advance(seconds=61)
    assert h.engine.reap()["leases_recovered"] == [run.run_id]
    h.drain()
    assert h.run(run.run_id).status == RunStatus.SUCCEEDED


def test_crash_after_effect_before_checkpoint_does_not_repeat_effect() -> None:
    """Side effect applied, then crash before the checkpoint: retry must skip the effect."""
    from lra.examples.procurement_saga import ExternalSystems

    ExternalSystems.reset()
    crashed: list[str] = []

    def chaos(point: str, run) -> None:
        if point == "after_step_before_commit" and run.current_step == "charge_payment" and not crashed:
            crashed.append(run.run_id)
            raise SimulatedCrash()

    h = make_harness(chaos=chaos)
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 10})
    with pytest.raises(SimulatedCrash):
        h.drain()
    assert [c[0] for c in ExternalSystems.calls] == ["reserve_stock", "charge"], "charge happened once before the crash"

    h.clock.advance(seconds=61)  # lease expires; Cloud Tasks redelivers, reaper is a no-op duplicate
    assert h.engine.reap()["leases_recovered"] == [run.run_id]
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.SUCCEEDED
    assert [c[0] for c in ExternalSystems.calls] == ["reserve_stock", "charge", "book_shipment"], "no double charge"
    assert r.attempt_of("charge_payment") == 1, "same attempt was re-delivered, not a retry"


def test_duplicate_delivery_is_a_noop(h: Harness) -> None:
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 10})
    task = h.queue.pop_due()
    assert task is not None
    assert h.engine.execute_task(task) == "ok"
    assert h.engine.execute_task(task) == "stale", "same (step, attempt) delivered twice"
    assert h.engine.execute_task(StepTask(run_id=run.run_id, step="charge_payment", attempt=5)) == "stale"
    assert h.engine.execute_task(StepTask(run_id="nope", step="x", attempt=1)) == "unknown-run"


def test_lease_blocks_a_second_worker() -> None:
    h = make_harness(worker_id="worker-A")
    other = Engine(
        store=h.store, queue=h.queue, bus=h.bus, llm=h.llm, clock=h.clock, workflows=h.engine.registry, worker_id="worker-B"
    )
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 10})
    task = h.queue.pop_due()
    assert task is not None
    h.store.acquire_lease(run.run_id, "worker-A", timedelta(seconds=60), h.clock.now())  # A is mid-step
    assert other.execute_task(task) == "lease-held"
    with pytest.raises(LeaseHeldError):
        h.store.acquire_lease(run.run_id, "worker-B", timedelta(seconds=60), h.clock.now())
    h.clock.advance(seconds=61)  # A died; lease expired
    assert other.execute_task(task) == "ok"


def test_transient_failures_retry_with_backoff_then_succeed() -> None:
    h = make_harness(fail_times=2)  # first two model calls raise ConnectionError
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 10, "flaky_at": "charge_payment"})
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.SUCCEEDED
    assert r.attempt_of("charge_payment") == 2
    retries = h.events("step.retry")
    assert retries and retries[0]["step"] == "charge_payment" and retries[0]["delay_s"] >= 0
    assert [t[1] for t in h.runner.trace] == ["ok", "retry", "ok", "done"]


def test_permanent_failure_exhausts_attempts_and_fails() -> None:
    wf = Workflow("always_fails")

    @wf.step(start=True, max_attempts=3, backoff_base_s=0.1)
    def boom(ctx):
        raise RuntimeError("downstream is down")

    h = make_harness()
    h.engine.registry.register(wf)
    run = h.start("always_fails")
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and "after 3 attempts" in (r.error or "")
    assert [t[1] for t in h.runner.trace] == ["retry", "retry", "failed"]


def test_step_failed_is_not_retried() -> None:
    wf = Workflow("business_rule")

    @wf.step(start=True, max_attempts=5)
    def check(ctx):
        raise StepFailed("customer is on the sanctions list")

    h = make_harness()
    h.engine.registry.register(wf)
    run = h.start("business_rule")
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and r.attempt_of("check") == 1 and "sanctions" in r.error


def test_idempotent_start_with_client_supplied_id(h: Harness) -> None:
    a = h.start("procurement", {"sku": "X", "qty": 1, "amount": 1}, run_id="order-1")
    b = h.start("procurement", {"sku": "X", "qty": 1, "amount": 1}, run_id="order-1")
    assert a.run_id == b.run_id and len(h.queue) == 1


def test_unknown_next_step_is_caught_before_persisting() -> None:
    wf = Workflow("typo")

    @wf.step(start=True, max_attempts=1)
    def s(ctx):
        return Next("does_not_exist")

    h = make_harness()
    h.engine.registry.register(wf)
    run = h.start("typo")
    h.drain()
    assert h.run(run.run_id).status == RunStatus.FAILED
