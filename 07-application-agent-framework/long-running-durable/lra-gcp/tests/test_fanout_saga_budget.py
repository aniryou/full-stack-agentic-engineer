"""Orchestrator/worker fan-out, saga compensation, budgets, versioning."""

from __future__ import annotations

from datetime import timedelta

import pytest

from lra import Budget, Done, Next, RunStatus, SimulatedCrash, Workflow
from lra.examples.procurement_saga import ExternalSystems
from tests.conftest import Harness, make_harness, research_routes


# ------------------------------------------------------------------ fan-out
def test_fan_out_children_are_runs_with_their_own_budget(h: Harness) -> None:
    run = h.start("research_pipeline", {"goal": "fan-out"})
    h.runner.step()  # plan -> FanOut
    parent = h.run(run.run_id)
    assert parent.status == RunStatus.WAITING and parent.wait.kind == "children"
    assert parent.fan_in.expected == 3 and parent.fan_in.completed == 0
    children = [c for c in h.store.list_runs() if c.parent_run_id == run.run_id]
    assert {c.run_id for c in children} == {f"{run.run_id}--{c.child_key}" for c in children}, "deterministic ids"
    assert all(c.budget.max_cost_usd == 0.25 for c in children)
    h.drain()
    assert h.run(run.run_id).status == RunStatus.WAITING and h.run(run.run_id).wait.kind == "approval"
    assert sorted(h.run(run.run_id).state["children"]["results"]) == sorted(c.child_key for c in children)


def test_partial_child_failure_is_data_for_the_aggregator() -> None:
    routes = research_routes()
    calls = {"n": 0}

    def flaky_research(prompt: str) -> str:
        calls["n"] += 1
        if "Subtopic 2" in prompt:
            raise RuntimeError("source unavailable")  # every attempt fails for one child
        return "- finding"

    routes[r"Research this subtopic"] = flaky_research
    h = make_harness(routes=routes)
    run = h.start("research_pipeline", {"goal": "partial"})
    h.drain()
    parent = h.run(run.run_id)
    assert parent.status == RunStatus.WAITING and parent.wait.kind == "approval", "2 of 3 children is enough"
    assert list(parent.state["partial_failures"]) and len(parent.state["children"]["results"]) == 2
    failed = [c for c in h.store.list_runs() if c.parent_run_id == run.run_id and c.status == RunStatus.FAILED]
    assert len(failed) == 1 and failed[0].attempt_of("research") == 3


def test_crash_between_fanout_checkpoint_and_spawn_is_repaired_by_reaper() -> None:
    crashed: list[str] = []

    def chaos(point: str, run) -> None:
        if point == "after_commit_before_enqueue" and run.fan_in and not crashed:
            crashed.append(run.run_id)
            raise SimulatedCrash()

    h = make_harness(chaos=chaos)
    run = h.start("research_pipeline", {"goal": "crash"})
    with pytest.raises(SimulatedCrash):
        h.drain()
    assert h.run(run.run_id).status == RunStatus.WAITING and h.run(run.run_id).fan_in.expected == 3
    assert not [c for c in h.store.list_runs() if c.parent_run_id == run.run_id], "no children spawned"

    h.drain()  # redelivered plan task is stale
    assert h.engine.reap()["children_respawned"] == [run.run_id]
    h.drain()
    assert h.run(run.run_id).status == RunStatus.WAITING and h.run(run.run_id).wait.kind == "approval"


# --------------------------------------------------------------------- saga
def test_saga_compensates_completed_steps_in_reverse_order(h: Harness) -> None:
    ExternalSystems.reset()
    run = h.start("procurement", {"sku": "X", "qty": 2, "amount": 99, "fail_at": "book_shipment"})
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.COMPENSATED and "book_shipment rejected" in r.error
    assert [c[0] for c in ExternalSystems.calls] == ["reserve_stock", "charge", "refund", "release_stock"]
    assert r.compensated_steps == ["charge_payment", "reserve_stock"]
    assert h.history(run.run_id)[-2:] == [("charge_payment", "compensate", "ok"), ("reserve_stock", "compensate", "ok")]
    assert h.events("run.compensating") and h.events("run.compensated")


def test_saga_retry_does_not_double_charge(h: Harness) -> None:
    ExternalSystems.reset()
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 5, "flaky_at": "charge_payment"})
    h.drain()
    assert h.run(run.run_id).status == RunStatus.SUCCEEDED
    assert [c[0] for c in ExternalSystems.calls].count("charge") == 1


def test_compensation_uses_the_effect_record_and_is_idempotent(h: Harness) -> None:
    ExternalSystems.reset()
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 5, "fail_at": "charge_payment"})
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.COMPENSATED
    refunds = [c for c in ExternalSystems.calls if c[0] == "refund"]
    assert refunds == [], "charge never happened, so no refund"
    releases = [c for c in ExternalSystems.calls if c[0] == "release_stock"]
    assert len(releases) == 1 and releases[0][1]["reservation_id"] == r.state["reservation_id"]
    assert h.store.effect_get(f"{run.run_id}:undo:reserve") is not None, "undo is recorded like any other effect"


def test_failed_compensation_parks_the_run_with_an_alert() -> None:
    wf = Workflow("bad_undo")

    def undo(ctx):
        raise RuntimeError("refund API 500")

    @wf.step(start=True, compensate=undo, max_attempts=2, backoff_base_s=0.1)
    def charge(ctx):
        return Next("ship")

    @wf.step(max_attempts=1)
    def ship(ctx):
        raise RuntimeError("carrier down")

    h = make_harness()
    h.engine.registry.register(wf)
    run = h.start("bad_undo")
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and "compensation of charge failed after 2 attempts" in r.error
    alerts = [m for m in h.bus.published if m["topic"] == "agent-alerts"]
    assert alerts and alerts[0]["payload"]["type"] == "run.compensation_failed"


# ------------------------------------------------------------------- budget
def test_step_budget_fails_closed_before_another_model_call() -> None:
    h = make_harness(routes=research_routes(first_score=1, second_score=1))  # never good enough -> would loop
    run = h.start("research_pipeline", {"goal": "loop"}, budget=Budget(max_steps=6, max_cost_usd=10))
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and "step budget exhausted" in r.error
    assert r.budget.steps_used == 6
    n_llm_calls_after = len(h.llm.calls)
    h.drain()
    assert len(h.llm.calls) == n_llm_calls_after, "a failed run makes no further model calls"


def test_cost_budget_is_charged_per_step(h: Harness) -> None:
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 1}, budget=Budget(max_cost_usd=0.0000001))
    h.drain()
    r = h.run(run.run_id)
    # The procurement steps make no model calls, so cost never exceeds; the budget only bites on LLM steps.
    assert r.status == RunStatus.SUCCEEDED and r.budget.cost_usd == 0


def test_deadline_is_absolute(h: Harness) -> None:
    deadline = h.clock.now() + timedelta(hours=1)
    run = h.start("research_pipeline", {"goal": "deadline"}, budget=Budget(deadline=deadline))
    h.drain()  # parks at approval
    h.clock.advance(hours=2)
    h.approve(run.run_id)
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and "deadline passed" in r.error
    assert "publish" not in r.completed_steps


def test_reflection_loop_stops_at_max_iterations() -> None:
    h = make_harness(routes=research_routes(first_score=3, second_score=3))
    run = h.start("research_pipeline", {"goal": "stubborn"})
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.WAITING
    assert r.state["reflect_loop"]["iteration"] == 2 and r.state["reflect_loop"]["exit_reason"].startswith("max iterations")


# --------------------------------------------------------------- versioning
def test_in_flight_runs_keep_their_workflow_version() -> None:
    v1 = Workflow("wf", version="1")

    @v1.step(start=True)
    def a(ctx):
        return Next("b")

    @v1.step()
    def b(ctx):
        return Done("v1 result")

    v2 = Workflow("wf", version="2")

    @v2.step(start=True)
    def a2(ctx):
        return Done("v2 result")

    h = make_harness()
    h.engine.registry.register(v1)
    run = h.start("wf")
    h.runner.step()  # a done; b queued
    h.engine.registry.register(v2)  # deploy v2 while run is in flight
    new_run = h.start("wf")
    assert new_run.workflow_version == "2"
    h.drain()
    assert h.run(run.run_id).result == "v1 result" and h.run(new_run.run_id).result == "v2 result"


def test_missing_version_fails_loudly_instead_of_guessing() -> None:
    v1 = Workflow("wf", version="1")

    @v1.step(start=True)
    def a(ctx):
        return Next("b")

    @v1.step()
    def b(ctx):
        return Done()

    h = make_harness()
    h.engine.registry.register(v1)
    run = h.start("wf")
    h.runner.step()
    h.engine.registry._by_key.pop(("wf", "1"))  # simulate a deploy that dropped v1
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.FAILED and "no longer deployed" in r.error
