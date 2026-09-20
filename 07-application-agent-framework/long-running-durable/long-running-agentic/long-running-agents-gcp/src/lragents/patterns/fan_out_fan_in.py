"""Pattern 2 — Orchestrator / workers (fan-out, fan-in).

A planner splits a goal into N independent subtasks, publishes them (Pub/Sub
topic or one Cloud Task each), N workers execute in parallel and write results,
and the *last* completion triggers aggregation. The fan-in is the hard part:

* Workers must be idempotent (Pub/Sub is at-least-once; a subtask may arrive twice).
* The completion counter is updated in a transaction; only the write that takes
  ``completed`` from N-1 to N sees ``all_done``. Because the aggregate wake-up is
  a *named* task, a late duplicate that also observes ``all_done`` re-enqueues
  harmlessly (de-duplicated) — so "crash after commit, before enqueue" is safe.
* Do the side effect (the worker's real work) OUTSIDE the transaction, under an
  idempotency key. Transactions may re-run.

Scaling note: N workers incrementing one Firestore document is fine for tens of
subtasks. For hundreds+, write one document per subtask
(``runs/{id}/subtasks/{sid}``) and have the orchestrator (a delayed Cloud Task
or a Cloud Workflows loop) poll a count query — or use Cloud Workflows'
``parallel`` step, which does the join for you (see infra/workflows/fan_out_fan_in.yaml).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..core.budget import charge
from ..core.llm import LLM, PriceCard
from ..core.models import Run, RunStatus, StepKind, StepRecord, StepStatus, new_run_id
from ..core.store import RunStore, transact
from ..core.tools import FaultInjector, IdempotencyStore, run_idempotent
from ..core.transport import Dispatcher, Envelope

PLANNER_PROMPT = "Split the goal into independent subtasks. Reply ONLY with a JSON array of objects, each with 'title' and 'instructions'."
AGGREGATOR_PROMPT = "You are given the results of parallel subtasks. Synthesize a single, final answer."


class FanOutFanIn:
    def __init__(
        self,
        *,
        store: RunStore,
        planner: LLM,
        aggregator: LLM,
        dispatcher: Dispatcher,
        worker: Callable[[dict[str, Any]], Any],
        idempotency: IdempotencyStore,
        price: PriceCard | None = None,
        worker_id: str = "worker-1",
        lease_ttl_s: float = 60.0,
        clock: Callable[[], float] = time.time,
        faults: FaultInjector | None = None,
        max_subtasks: int = 50,
    ) -> None:
        self.store, self.planner, self.aggregator = store, planner, aggregator
        self.dispatcher, self.worker, self.idem = dispatcher, worker, idempotency
        self.price = price or PriceCard()
        self.worker_id, self.lease_ttl_s, self.clock = worker_id, lease_ttl_s, clock
        self.faults = faults or FaultInjector()
        self.max_subtasks = max_subtasks

    # ------------------------------------------------------------------ API
    def start(self, goal: str, subtasks: list[dict[str, Any]] | None = None, run_id: str | None = None) -> Run:
        run = Run(run_id=run_id or new_run_id("fan"), goal=goal, status=RunStatus.RUNNING)
        if subtasks is None:
            d = self.planner.decide(PLANNER_PROMPT, [{"role": "user", "content": goal}], [])
            cost = charge(run, d.tokens_in, d.tokens_out, self.price)
            rec = StepRecord(index=0, kind=StepKind.LLM, status=StepStatus.DONE, name="plan", tokens_in=d.tokens_in,
                             tokens_out=d.tokens_out, cost_usd=cost)
            subtasks = json.loads(d.text or "[]")
            rec.finish(output={"subtasks": subtasks})
            run.append(rec)
        if not 0 < len(subtasks) <= self.max_subtasks:
            raise ValueError(f"need 1..{self.max_subtasks} subtasks, got {len(subtasks)}")
        run.state["fan"] = {"expected": len(subtasks), "completed": 0, "results": {}, "subtasks": subtasks}
        run = self.store.create(run)
        for i, st in enumerate(subtasks):                     # fan-out: one message per subtask
            self.dispatcher.enqueue(Envelope(run.run_id, 0, "subtask", payload={"subtask_id": str(i), "task": st}))
        return run

    def handle(self, env: Envelope) -> Run:
        if env.kind == "subtask":
            return self.handle_subtask(env.run_id, env.payload["subtask_id"], env.payload["task"])
        if env.kind == "aggregate":
            return self.aggregate(env.run_id)
        raise ValueError(env.kind)

    # -------------------------------------------------------------- workers
    def handle_subtask(self, run_id: str, subtask_id: str, task: dict[str, Any]) -> Run:
        key = f"{run_id}:sub:{subtask_id}"
        result, replayed = run_idempotent(self.idem, key, lambda: self.worker(task))   # side effect outside the txn
        self.faults.maybe_crash("after_worker")

        def mutate(run: Run) -> bool:                          # pure; may be retried on VersionConflict
            fan = run.state["fan"]
            if subtask_id not in fan["results"]:
                fan["results"][subtask_id] = result
                fan["completed"] += 1
                rec = StepRecord(index=run.next_index(), kind=StepKind.TOOL, status=StepStatus.DONE, name=f"subtask:{subtask_id}",
                                 input=task, idempotency_key=key)
                rec.finish(output={"result": result, "replayed": replayed})
                run.append(rec)
            return fan["completed"] == fan["expected"]

        run, all_done = transact(self.store, run_id, mutate)
        self.faults.maybe_crash("after_commit")
        if all_done:                                          # last one out (or a late duplicate) → idempotent enqueue
            self.dispatcher.enqueue(Envelope(run_id, 1, "aggregate"))
        return run

    # ------------------------------------------------------------ aggregate
    def aggregate(self, run_id: str) -> Run:
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        try:
            if run.status.terminal:
                return run
            fan = run.state["fan"]
            if fan["completed"] != fan["expected"]:
                return run                                    # premature wake-up; ignore
            ordered = [fan["results"][k] for k in sorted(fan["results"], key=int)]
            d = self.aggregator.decide(AGGREGATOR_PROMPT, [{"role": "user", "content": json.dumps({"goal": run.goal, "results": ordered}, default=str)}], [])
            cost = charge(run, d.tokens_in, d.tokens_out, self.price)
            rec = StepRecord(index=run.next_index(), kind=StepKind.LLM, status=StepStatus.DONE, name="aggregate",
                             tokens_in=d.tokens_in, tokens_out=d.tokens_out, cost_usd=cost)
            rec.finish(output={"text": d.text})
            run.append(rec)
            run.result = d.text
            run.status = RunStatus.SUCCEEDED
            return self.store.save(run)
        finally:
            self.store.release_lease(run_id, self.worker_id)
