"""Human-in-the-loop (HITL): suspend, wait days, resume.

The agent must *truly sleep* while a human decides — no polling loop, no
container pinned for three days. Mechanically that is a :class:`Wait`
outcome: the run is checkpointed as WAITING, no task exists for it, and an
external ``POST /runs/{id}/events`` (from an approval UI, Slack action,
e-mail link, or a Cloud Workflows callback) wakes it.

On GCP the same shape exists at three layers:

* **This engine**: ``Wait(key=..., then=..., timeout=...)`` + ``engine.resume``.
* **Cloud Workflows**: ``events.create_callback_endpoint`` + ``events.await_callback``
  (see workflows/research_approval.yaml).
* **ADK / Agent Engine**: ``LongRunningFunctionTool`` returns a pending
  status; the app resumes with the human's response (examples/adk_agent_engine).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..core.workflow import StepContext, StepFailed, Wait


def approval_key(ctx: StepContext, gate: str = "approval") -> str:
    """Stable, run-scoped key. Include the run id so a stale link cannot resume the wrong run."""
    return f"{gate}:{ctx.run_id}"


def request_approval(
    ctx: StepContext,
    *,
    then: str,
    gate: str = "approval",
    summary: str | None = None,
    timeout: timedelta = timedelta(days=3),
    on_timeout: str = "fail",
) -> Wait:
    """Record what is being approved and suspend until a human responds."""
    key = approval_key(ctx, gate)
    ctx.state.setdefault("approvals", {})[gate] = {"status": "pending", "summary": summary, "key": key}
    ctx.emit("approval.requested", {"gate": gate, "key": key, "summary": summary, "timeout_s": timeout.total_seconds()})
    return Wait(key=key, then=then, kind="approval", timeout=timeout, on_timeout=on_timeout)


def approval_decision(ctx: StepContext, gate: str = "approval") -> dict[str, Any]:
    """Read the human's decision after resume; fail the run on rejection or timeout.

    The event payload contract is ``{"decision": "approve"|"reject", "by": str, "comment": str}``.
    """
    key = approval_key(ctx, gate)
    evt = ctx.state.get("events", {}).get(key)
    if evt is None:
        raise StepFailed(f"no approval event recorded for {key}")
    payload = evt.get("payload", {})
    if payload.get("timed_out"):
        raise StepFailed(f"approval {gate} timed out")
    decision = payload.get("decision", "reject")
    ctx.state["approvals"][gate].update({"status": decision, "by": payload.get("by"), "comment": payload.get("comment")})
    if decision != "approve":
        raise StepFailed(f"approval {gate} rejected by {payload.get('by', 'unknown')}: {payload.get('comment', '')}")
    return payload
