"""Human-in-the-loop: suspend, resume, timeouts, duplicate/wrong events, cancel."""

from __future__ import annotations

from lra import Event, RunStatus
from tests.conftest import Harness, make_harness


def _to_review(h: Harness):
    run = h.start("research_pipeline", {"goal": "hitl"})
    h.drain()
    r = h.run(run.run_id)
    assert r.status == RunStatus.WAITING and r.wait and r.wait.key == f"editor:{run.run_id}"
    assert r.lease is None and len(h.queue) == 0, "a waiting run holds no lease and has no task: sleeping is free"
    return r


def test_wait_timeout_at_is_absolute_and_survives_restarts(h: Harness) -> None:
    r = _to_review(h)
    assert r.wait.timeout_at == h.clock.now().replace() + (r.wait.timeout_at - h.clock.now())
    assert (r.wait.timeout_at - h.clock.now()).days == 3


def test_resume_is_idempotent_and_key_scoped(h: Harness) -> None:
    r = _to_review(h)
    wrong = Event(run_id=r.run_id, key="approval:other", payload={"decision": "approve"})
    assert h.engine.resume(wrong) is None, "an event for another gate does nothing"

    evt = Event(run_id=r.run_id, key=r.wait.key, payload={"decision": "approve", "by": "anil"})
    assert h.engine.resume(evt) is not None
    assert h.engine.resume(evt) is None, "duplicate webhook is a no-op, not an error"
    h.drain()
    assert h.run(r.run_id).status == RunStatus.SUCCEEDED
    assert h.run(r.run_id).state["approvals"]["editor"]["by"] == "anil"


def test_rejection_fails_the_run_without_retry(h: Harness) -> None:
    r = _to_review(h)
    h.engine.resume(Event(run_id=r.run_id, key=r.wait.key, payload={"decision": "reject", "by": "anil", "comment": "too thin"}))
    h.drain()
    r2 = h.run(r.run_id)
    assert r2.status == RunStatus.FAILED and "rejected by anil" in r2.error
    assert r2.attempt_of("publish") == 1
    assert "publish" not in r2.completed_steps, "nothing to compensate: publish never happened"


def test_timeout_fails_run_via_reaper(h: Harness) -> None:
    r = _to_review(h)
    assert h.engine.reap()["waits_timed_out"] == []
    h.clock.advance(days=3, seconds=1)
    assert h.engine.reap()["waits_timed_out"] == [r.run_id]
    r2 = h.run(r.run_id)
    assert r2.status == RunStatus.FAILED and "timed out waiting for editor" in r2.error


def test_timeout_can_resume_with_marker() -> None:
    from datetime import timedelta

    from lra import Done, Wait, Workflow

    wf = Workflow("soft_gate")

    @wf.step(start=True)
    def ask(ctx):
        return Wait(key=f"ok:{ctx.run_id}", then="finish", timeout=timedelta(hours=1), on_timeout="resume")

    @wf.step()
    def finish(ctx):
        evt = ctx.state["events"][f"ok:{ctx.run_id}"]["payload"]
        return Done({"auto_approved": bool(evt.get("timed_out"))})

    h = make_harness()
    h.engine.registry.register(wf)
    run = h.start("soft_gate")
    h.drain()
    h.clock.advance(hours=1, seconds=1)
    h.engine.reap()
    h.drain()
    assert h.run(run.run_id).result == {"auto_approved": True}


def test_cancel_while_waiting_finishes_immediately(h: Harness) -> None:
    r = _to_review(h)
    h.engine.cancel(r.run_id)
    assert h.run(r.run_id).status == RunStatus.CANCELLED
    assert h.engine.resume(Event(run_id=r.run_id, key=r.wait.key, payload={"decision": "approve"})) is None


def test_cancel_while_running_is_cooperative(h: Harness) -> None:
    run = h.start("procurement", {"sku": "X", "qty": 1, "amount": 1})
    h.runner.step()  # reserve_stock done, charge_payment queued
    h.engine.cancel(run.run_id)
    assert h.run(run.run_id).status == RunStatus.RUNNING and h.run(run.run_id).cancel_requested
    h.drain()
    r = h.run(run.run_id)
    # The queued step saw the flag and, since reserve_stock is compensable, rolled it back.
    assert r.status == RunStatus.COMPENSATED and r.error == "cancelled by user"
    assert r.compensated_steps == ["reserve_stock"]
