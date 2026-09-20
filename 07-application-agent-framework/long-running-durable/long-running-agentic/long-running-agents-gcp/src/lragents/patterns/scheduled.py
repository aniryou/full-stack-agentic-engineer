"""Pattern 5 — Scheduled (heartbeat) agent.

"Check the presale every 5 minutes", "reconcile the ledger nightly", "watch the
inbox". A prompt cannot do this: prompts are read only when something invokes
the model. Something with a clock has to wake the agent — on GCP that is Cloud
Scheduler → (Pub/Sub →) Cloud Run.

Three things go wrong with naive cron-driven agents and how this module fixes them:

1. **Overlap** – tick N is still running when tick N+1 fires (Scheduler retries,
   slow steps). Fix: a lease. The second tick sees ``LeaseHeld`` and exits.
2. **Duplicate ticks** – Scheduler and Pub/Sub are at-least-once. Fix: a
   ``next_due`` timestamp stored with the run; a tick that arrives early is a no-op.
3. **Zombies** – the worker dies holding the lease. Fix: leases expire (TTL), so
   the next tick after expiry proceeds. Long work extends the lease ("heartbeat").

The unit of work per tick is small and checkpointed; anything long is handed
to the durable loop rather than done inside the tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..core.models import Run, RunStatus, StepKind, StepRecord, StepStatus
from ..core.store import LeaseHeld, RunNotFound, RunStore


@dataclass
class TickResult:
    ran: bool
    reason: str
    output: Any = None


class ScheduledAgent:
    def __init__(self, *, store: RunStore, work: Callable[[dict[str, Any]], Any], interval_s: float,
                 worker_id: str = "worker-1", lease_ttl_s: float = 120.0, clock: Callable[[], float] = time.time) -> None:
        self.store, self.work, self.interval_s = store, work, interval_s
        self.worker_id, self.lease_ttl_s, self.clock = worker_id, lease_ttl_s, clock

    def ensure(self, name: str, state: dict[str, Any] | None = None) -> Run:
        """Idempotently create the long-lived 'monitor' run for this schedule."""
        try:
            return self.store.get(name)
        except RunNotFound:
            run = Run(run_id=name, goal=f"scheduled:{name}", status=RunStatus.RUNNING,
                      state={**(state or {}), "next_due": 0.0, "ticks": 0})
            return self.store.create(run)

    def tick(self, name: str) -> TickResult:
        now = self.clock()
        try:
            run = self.store.acquire_lease(name, self.worker_id, self.lease_ttl_s)
        except LeaseHeld as e:
            return TickResult(False, f"overlap: {e}")
        try:
            if run.status.terminal:
                return TickResult(False, "terminal")
            if now < run.state.get("next_due", 0.0):
                return TickResult(False, f"not due until {run.state['next_due']:.0f}")
            rec = StepRecord(index=run.next_index(), kind=StepKind.SYSTEM, status=StepStatus.STARTED, name="tick")
            run.append(rec)
            run = self.store.save(run)                 # intent
            out = self.work(run.state)                 # small, idempotent-by-design unit of work
            run.journal[-1].finish(output=out)
            run.state["ticks"] += 1
            run.state["next_due"] = now + self.interval_s
            run.state["last_tick_at"] = now
            self.store.save(run)
            return TickResult(True, "ran", out)
        finally:
            self.store.release_lease(name, self.worker_id)

    def heartbeat(self, name: str) -> None:
        """Extend the lease from inside a long unit of work."""
        self.store.acquire_lease(name, self.worker_id, self.lease_ttl_s)
