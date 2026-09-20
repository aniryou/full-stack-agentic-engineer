"""Pattern 4 — Saga (compensating transactions).

An agent that books a flight, then a hotel, then charges a card is performing
a distributed transaction across systems that share no transaction manager. If
step 3 fails, steps 1–2 must be *undone* with compensating actions (cancel
hotel, cancel flight) — not rolled back, because they already happened.

Saga rules encoded here:
* Every forward action and every compensation is recorded in the journal with
  an idempotency key *before* it runs (write-ahead intent, same as the loop).
* One action per wake-up, so the saga survives a crash at any point and
  resumes from the journal.
* Compensations run in reverse order and must themselves be idempotent and
  retryable — a compensation that can fail permanently needs a human escalation
  path (``on_stuck``), never silent abandonment.
* The LLM may *decide* the plan (which steps, which arguments) but the saga
  runner, not the model, owns the forward/compensate state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..core.models import Run, RunStatus, StepKind, StepRecord, StepStatus, new_run_id
from ..core.store import RunStore
from ..core.tools import FaultInjector, IdempotencyStore, ToolError, run_idempotent
from ..core.transport import Dispatcher, Envelope


@dataclass
class SagaStep:
    name: str
    action: Callable[[dict[str, Any], str], Any]                  # (context, idempotency_key) -> result
    compensate: Callable[[dict[str, Any], Any, str], Any]         # (context, action_result, idempotency_key) -> None


class SagaRunner:
    def __init__(self, *, store: RunStore, dispatcher: Dispatcher, idempotency: IdempotencyStore, steps: list[SagaStep],
                 worker_id: str = "worker-1", lease_ttl_s: float = 60.0, clock: Callable[[], float] = time.time,
                 faults: FaultInjector | None = None, on_stuck: Callable[[Run, str], None] | None = None) -> None:
        self.store, self.dispatcher, self.idem = store, dispatcher, idempotency
        self.steps = {s.name: s for s in steps}
        self.order = [s.name for s in steps]
        self.worker_id, self.lease_ttl_s, self.clock = worker_id, lease_ttl_s, clock
        self.faults = faults or FaultInjector()
        self.on_stuck = on_stuck or (lambda run, why: None)

    def start(self, context: dict[str, Any], run_id: str | None = None) -> Run:
        run = Run(run_id=run_id or new_run_id("saga"), goal="saga", status=RunStatus.RUNNING,
                  state={"ctx": context, "saga": {"phase": "forward", "cursor": 0, "results": {}, "comp_cursor": None}})
        run = self.store.create(run)
        self.dispatcher.enqueue(Envelope(run.run_id, run.next_index(), "step"))
        return run

    def handle(self, env: Envelope) -> Run:
        return self.advance(env.run_id)

    # ---------------------------------------------------------------- core
    def advance(self, run_id: str) -> Run:
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        try:
            if run.status.terminal:
                return run
            pending = run.pending_step()
            if pending is not None:                               # crashed mid-action or mid-compensation
                return self._execute(run, pending)
            saga = run.state["saga"]
            if saga["phase"] == "forward":
                if saga["cursor"] >= len(self.order):
                    run.status, run.result = RunStatus.SUCCEEDED, saga["results"]
                    return self.store.save(run)
                name = self.order[saga["cursor"]]
                return self._intend_and_execute(run, name, "action")
            # compensating: walk back from comp_cursor
            if saga["comp_cursor"] < 0:
                run.status, run.error = RunStatus.FAILED, run.error or "saga compensated"
                return self.store.save(run)
            name = self.order[saga["comp_cursor"]]
            return self._intend_and_execute(run, name, "compensate")
        finally:
            self.store.release_lease(run_id, self.worker_id)

    def _intend_and_execute(self, run: Run, name: str, phase: str) -> Run:
        idx = run.next_index()
        rec = StepRecord(index=idx, kind=StepKind.TOOL, status=StepStatus.STARTED, name=f"{name}:{phase}",
                         input={"step": name, "phase": phase}, idempotency_key=f"{run.run_id}:{name}:{phase}")
        run.append(rec)
        run = self.store.save(run)                                  # write-ahead intent
        self.faults.maybe_crash("after_intent")
        return self._execute(run, run.journal[-1])

    def _execute(self, run: Run, rec: StepRecord) -> Run:
        step_name, phase = rec.input["step"], rec.input["phase"]
        step = self.steps[step_name]
        saga = run.state["saga"]
        ctx = run.state["ctx"]
        key = rec.idempotency_key or f"{run.run_id}:{step_name}:{phase}"
        try:
            if phase == "action":
                result, _ = run_idempotent(self.idem, key, lambda: step.action(ctx, key))
                self.faults.maybe_crash("after_side_effect")
                rec.finish(output={"result": result})
                saga["results"][step_name] = result
                saga["cursor"] += 1
            else:
                prior = saga["results"].get(step_name)
                run_idempotent(self.idem, key, lambda: step.compensate(ctx, prior, key))
                self.faults.maybe_crash("after_side_effect")
                rec.finish(output={"compensated": step_name})
                saga["comp_cursor"] -= 1
        except ToolError as e:
            rec.finish(error=str(e))
            if phase == "action":                                   # flip to compensation of everything before this step
                saga["phase"] = "compensating"
                saga["comp_cursor"] = saga["cursor"] - 1
                run.error = f"{step_name} failed: {e}"
            else:                                                   # a compensation failed: needs a human
                run.status = RunStatus.FAILED
                run.error = f"compensation {step_name} failed: {e} — MANUAL INTERVENTION REQUIRED"
                run = self.store.save(run)
                self.on_stuck(run, run.error)
                return run
        run = self.store.save(run)
        if not run.status.terminal:
            self.dispatcher.enqueue(Envelope(run.run_id, run.next_index(), "step"))
        return run
