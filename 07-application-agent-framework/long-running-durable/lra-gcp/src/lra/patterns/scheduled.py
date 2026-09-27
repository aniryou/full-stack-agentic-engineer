"""Scheduled (heartbeat) runs: "check every 5 minutes", "reconcile nightly".

A prompt cannot start itself; something with a clock has to wake the agent. On GCP that is
Cloud Scheduler -> (Pub/Sub ->) an HTTP handler that calls :func:`tick`. Three things go wrong with
naive cron-driven agents, and this is how the engine's primitives fix each:

1. **Duplicate ticks.** Scheduler and Pub/Sub are at-least-once, so the same tick can arrive twice.
   Each tick maps to a run id derived from its time window (``schedule:window``) and
   :meth:`Engine.start` is idempotent on ``run_id``: the second delivery returns the same run.
2. **Overlap.** Tick N+1 fires while tick N's run is still working. With ``allow_overlap=False``
   the tick is skipped while the previous window's run is not terminal.
3. **Zombies.** A worker dies mid-tick. Nothing special: the run's lease expires and the reaper
   re-drives it (``Engine.reap``), as for any other run.

Keep the per-tick work small; hand long work to a normal run (or a :class:`~lra.core.workflow.FanOut`).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..core.models import Run


def window_of(now: datetime, interval: timedelta) -> int:
    """The index of the time window ``now`` falls in (UTC epoch seconds // interval)."""
    return int(now.timestamp() // interval.total_seconds())


def tick(
    engine: Any,
    workflow: str,
    *,
    schedule: str,
    interval: timedelta,
    input: dict[str, Any] | None = None,
    now: datetime | None = None,
    allow_overlap: bool = False,
) -> Run | None:
    """Handle one scheduler delivery. Returns the window's run, or None when skipped for overlap."""
    now = now or engine.clock.now()
    w = window_of(now, interval)
    if not allow_overlap:
        prev = engine.store.get(f"{schedule}:{w - 1}")
        if prev is not None and not prev.status.terminal:
            return None
    return engine.start(workflow, {**(input or {}), "window": w}, run_id=f"{schedule}:{w}")
