"""Pattern 3 — Human-in-the-loop gate (suspend / resume).

When the loop wants to call a tool marked ``requires_approval`` it parks the run
(``WAITING_HUMAN``), stores the *exact* proposed call plus a one-time token, and
notifies a human (Slack, email, a queue in an ops UI). Nothing runs, nothing
costs money, for hours or days. A decision arrives through an HTTP endpoint
(``POST /runs/{id}/approve``) or a Cloud Workflows callback.

Two rules that make approvals meaningful:

1. **Execute what was approved, approve what will execute.** After an approval
   we do *not* re-ask the model. We journal the approved call as a STARTED
   intent and let the durable loop execute it under its idempotency key. If we
   re-asked the model it might pick different arguments and the human would
   have approved something that never ran.
2. **Every gate has a deadline.** ``expire_stale_approvals`` is run by Cloud
   Scheduler (or a delayed Cloud Task created at park time) so a forgotten
   approval becomes an explicit FAILED/escalated state, not a zombie.

The token is single-use and compared server-side; the approval link should
carry it, and the endpoint must be authenticated (IAP / Identity Platform).
"""

from __future__ import annotations

import hmac
import time
from typing import Callable

from ..core.models import Run, RunStatus, StepKind, StepRecord, StepStatus
from .durable_loop import DurableAgentLoop


class ApprovalError(Exception):
    pass


def approve(loop: DurableAgentLoop, run_id: str, token: str, approved: bool, approver: str, comment: str = "") -> Run:
    run = loop.store.acquire_lease(run_id, loop.worker_id, loop.lease_ttl_s)
    try:
        if run.status != RunStatus.WAITING_HUMAN:
            return run                                        # already decided (double-click, retried webhook)
        w = run.waiting_on or {}
        if not hmac.compare_digest(str(w.get("token", "")), token):
            raise ApprovalError("bad approval token")

        human = StepRecord(index=run.next_index(), kind=StepKind.HUMAN, status=StepStatus.DONE, name="approval",
                           input={"tool": w["tool"], "args": w["args"]})
        human.finish(output={"approved": approved, "approver": approver, "comment": comment})
        run.append(human)
        run.waiting_on = None
        run.status = RunStatus.RUNNING
        if approved:                                          # journal the approved call as a durable intent
            idx = run.next_index()
            run.append(StepRecord(index=idx, kind=StepKind.TOOL, status=StepStatus.STARTED, name=w["tool"],
                                  input=w["args"], idempotency_key=f"{run.run_id}:{idx}"))
        run = loop.store.save(run)
        loop._enqueue_next(run)                               # the loop's recovery path executes the intent
        return run
    finally:
        loop.store.release_lease(run_id, loop.worker_id)


def expire_stale_approvals(loop: DurableAgentLoop, ttl_s: float, now: float | None = None,
                           escalate: Callable[[Run], None] | None = None) -> list[Run]:
    """Run periodically (Cloud Scheduler → /internal/scheduler/tick)."""
    now = now if now is not None else loop.clock()
    expired: list[Run] = []
    for run in loop.store.list_runs(RunStatus.WAITING_HUMAN):
        w = run.waiting_on or {}
        if now - w.get("requested_at", now) < ttl_s:
            continue
        run = loop.store.acquire_lease(run.run_id, loop.worker_id, loop.lease_ttl_s)
        try:
            if run.status != RunStatus.WAITING_HUMAN:
                continue
            if escalate:
                escalate(run)
            rec = StepRecord(index=run.next_index(), kind=StepKind.SYSTEM, status=StepStatus.DONE, name="approval_timeout")
            rec.finish(output={"after_s": now - w.get("requested_at", now)})
            run.append(rec)
            run.status = RunStatus.FAILED
            run.error = "approval timed out"
            run.waiting_on = None
            expired.append(loop.store.save(run))
        finally:
            loop.store.release_lease(run.run_id, loop.worker_id)
    return expired


def approval_link(base_url: str, run: Run) -> str:
    w = run.waiting_on or {}
    return f"{base_url}/runs/{run.run_id}/approve?token={w.get('token', '')}"


__all__ = ["approve", "expire_stale_approvals", "approval_link", "ApprovalError", "time"]
