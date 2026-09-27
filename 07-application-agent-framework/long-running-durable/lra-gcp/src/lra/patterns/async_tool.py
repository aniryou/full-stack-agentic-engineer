"""Long-running tools: a ticket now, the result later, by callback or by polling with back-off.

A BigQuery export, a fine-tune job or a crawler returns a *ticket* immediately and finishes minutes
to hours later. No process may wait for it. The run parks on a timer instead (``Wait(kind="timer")``,
``on_timeout="resume"``): the reaper wakes it when the timer is due (on GCP, Cloud Scheduler ->
``/internal/reap``; a delayed Cloud Task works too), and the polling step checks the job again with
exponential back-off, capped, until a total deadline. If the external system can call back, it
resumes the same wait key early with the result (``engine.resume(Event(key=job_key(...), payload=...))``)
and the poll never happens. ADK's ``LongRunningFunctionTool`` is the same shape.

Use :func:`poll_job` as the body of the polling step::

    @wf.step()
    def wait_export(ctx):
        return poll_job(ctx, job="export", check=lambda: bq.job_result(ctx.state["export_ticket"]),
                        then="summarise", deadline=timedelta(hours=6))
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from ..core.workflow import Next, StepContext, StepFailed, Wait


def job_key(ctx: StepContext, job: str) -> str:
    """The wait key a callback must present to resume the run early."""
    return f"job:{job}:{ctx.run_id}"


def backoff(polls: int, base: timedelta, cap: timedelta) -> timedelta:
    """base, 2*base, 4*base, ... capped at ``cap``."""
    return min(cap, base * (2 ** polls))


def poll_job(
    ctx: StepContext,
    *,
    job: str,
    check: Callable[[], dict[str, Any] | None],
    then: str,
    deadline: timedelta,
    base: timedelta = timedelta(seconds=30),
    cap: timedelta = timedelta(minutes=10),
) -> Next | Wait:
    """Return ``Next(then)`` once the job's result is known (stored in ``ctx.state["jobs"][job]["result"]``),
    otherwise park on a back-off timer and come back to this same step."""
    jobs = ctx.state.setdefault("jobs", {})
    j = jobs.setdefault(job, {"polls": 0, "started_at": ctx.now().isoformat()})
    key = job_key(ctx, job)
    evt = ctx.state.get("events", {}).pop(key, None)          # consume the wake-up so the next wait is clean
    payload = (evt or {}).get("payload", {})
    result = payload.get("result") if evt and not payload.get("timed_out") else None
    j["via"] = "callback" if result is not None else "poll"
    if result is None:
        result = check()                                        # one cheap status call per wake-up
    if result is not None:
        j["result"] = result
        return Next(then)
    if ctx.now() - datetime.fromisoformat(j["started_at"]) >= deadline:
        raise StepFailed(f"job {job} not done after {deadline}")   # a business failure: no retry storm
    delay = backoff(j["polls"], base, cap)
    j["polls"] += 1
    return Wait(key=key, then=ctx.step_name, kind="timer", timeout=delay, on_timeout="resume")
