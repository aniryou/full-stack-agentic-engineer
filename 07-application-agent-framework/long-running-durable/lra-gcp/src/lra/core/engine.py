"""Durable-execution engine.

One :class:`Engine` instance runs inside every worker replica (Cloud Run). It
is stateless: all coordination happens through the :class:`StateStore`
(Firestore) and :class:`TaskQueue` (Cloud Tasks). That is what lets Cloud Run
scale to zero between steps and back up to N replicas under load.

Lifecycle of one step::

    queue delivers StepTask(run_id, step, attempt)
      -> stale/duplicate guard (does the run still expect this exact task?)
      -> acquire lease (one live worker per run)
      -> budget + cancellation checks
      -> execute step fn against a deep copy of state
      -> commit checkpoint atomically (optimistic concurrency)
      -> enqueue next task (dedup key = run:step:attempt)
      -> publish progress events
      -> release lease

Every arrow is a place a worker can die. The invariants that make that safe:

1. The checkpoint is written *before* the next task is enqueued, so a crash
   between the two leaves a consistent run that the reaper can re-drive.
2. A task is only executed if ``(step, attempt)`` matches the run document,
   so duplicate or late deliveries are no-ops.
3. Side effects go through ``ctx.effect(key, fn)`` which records the effect
   *before* the checkpoint, so "effect happened but checkpoint didn't" is
   detected on retry.
"""

from __future__ import annotations

import logging
import os
import socket
from datetime import timedelta
from typing import Any, Callable, Iterable

from .models import (
    Budget,
    ChildSpec,
    Event,
    FanIn,
    Lease,
    Run,
    RunStatus,
    StepRecord,
    StepTask,
    WaitCondition,
    backoff_delay,
    digest,
    new_id,
)
from .ports import LLM, Clock, ConflictError, EventBus, LeaseHeldError, StateStore, TaskQueue
from .workflow import Done, FanOut, Next, Step, StepContext, StepFailed, Wait, Workflow

log = logging.getLogger("lra.engine")


class SimulatedCrash(Exception):
    """Raised by a chaos hook to emulate the process dying at a specific point."""


class WorkflowRegistry:
    """Holds every deployed (name, version) so in-flight runs keep their code."""

    def __init__(self, workflows: Iterable[Workflow] = ()) -> None:
        self._by_key: dict[tuple[str, str], Workflow] = {}
        self._latest: dict[str, Workflow] = {}
        for wf in workflows:
            self.register(wf)

    def register(self, wf: Workflow) -> None:
        wf.validate()
        self._by_key[(wf.name, wf.version)] = wf
        cur = self._latest.get(wf.name)
        if cur is None or _version_key(wf.version) >= _version_key(cur.version):
            self._latest[wf.name] = wf

    def latest(self, name: str) -> Workflow:
        try:
            return self._latest[name]
        except KeyError as e:
            raise KeyError(f"unknown workflow {name!r}") from e

    def get(self, name: str, version: str) -> Workflow | None:
        return self._by_key.get((name, version))

    def names(self) -> list[str]:
        return sorted(self._latest)


def _version_key(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in v.split("."))
    except ValueError:
        return (0,)


class Engine:
    PROGRESS_TOPIC = "agent-events"

    def __init__(
        self,
        *,
        store: StateStore,
        queue: TaskQueue,
        bus: EventBus,
        llm: LLM,
        clock: Clock,
        workflows: Iterable[Workflow] | WorkflowRegistry = (),
        worker_id: str | None = None,
        lease_ttl: timedelta = timedelta(seconds=60),
        chaos: Callable[[str, Run], None] | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.bus = bus
        self.llm = llm
        self.clock = clock
        self.registry = workflows if isinstance(workflows, WorkflowRegistry) else WorkflowRegistry(workflows)
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{new_id('w')[2:8]}"
        self.lease_ttl = lease_ttl
        self.chaos = chaos  # test hook: chaos(point, run) may raise SimulatedCrash

    # ------------------------------------------------------------------ start
    def start(
        self,
        workflow: str,
        input: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
        parent_run_id: str | None = None,
        child_key: str | None = None,
    ) -> Run:
        """Create a run and enqueue its first step.

        Passing ``run_id`` makes creation idempotent: a client that retries
        ``POST /runs`` with the same id gets the same run, not a duplicate.
        """
        wf = self.registry.latest(workflow)
        existing = self.store.get(run_id) if run_id else None
        if existing is not None:
            return existing
        run = Run(
            run_id=run_id or new_id(),
            workflow=wf.name,
            workflow_version=wf.version,
            status=RunStatus.PENDING,
            current_step=wf.start,
            input=dict(input or {}),
            budget=(budget or wf.default_budget).model_copy(deep=True),
            parent_run_id=parent_run_id,
            child_key=child_key,
            created_at=self.clock.now(),
            updated_at=self.clock.now(),
        )
        attempt = run.bump_attempt(wf.start)  # type: ignore[arg-type]
        try:
            run = self.store.create(run)
        except ConflictError:
            return self.store.get(run.run_id)  # type: ignore[return-value]  # lost a create race: same id, same run
        self._enqueue(run, wf.start, attempt)  # type: ignore[arg-type]
        self._publish(run, "run.started", {"workflow": wf.name})
        return run

    # ---------------------------------------------------------------- execute
    def execute_task(self, task: StepTask) -> str:
        """Worker entrypoint. Returns a short outcome code (useful for HTTP mapping).

        Outcome codes: ``ok``, ``done``, ``waiting``, ``fanout``, ``retry``,
        ``failed``, ``compensating``, ``compensated``, ``budget``,
        ``cancelled``, ``stale``, ``unknown-run``, ``lease-held``,
        ``version-missing``.
        """
        run = self.store.get(task.run_id)
        if run is None:
            log.warning("task for unknown run %s; acking", task.run_id)
            return "unknown-run"
        if run.status.terminal:
            return "stale"
        if not self._task_matches(run, task):
            log.info("stale task %s for run %s (current=%s attempts=%s)", task.dedup_key, run.run_id, run.current_step, run.attempts)
            return "stale"

        now = self.clock.now()
        try:
            run = self.store.acquire_lease(run.run_id, self.worker_id, self.lease_ttl, now)
        except LeaseHeldError:
            return "lease-held"

        released = False
        try:
            # Cancellation and budget gate *forward* progress only: a run that is
            # already compensating must be allowed to finish undoing its effects.
            if task.kind == "step":
                if run.cancel_requested:
                    self._fail_or_compensate(run, "cancelled by user", final=RunStatus.CANCELLED)
                    return "cancelled"
                reason = run.budget.exceeded(now)
                if reason:
                    self._fail_or_compensate(run, reason)
                    return "budget"
            wf = self.registry.get(run.workflow, run.workflow_version)
            if wf is None:
                self._fail_or_compensate(
                    run,
                    f"workflow {run.workflow}@{run.workflow_version} is no longer deployed; "
                    "redeploy that version or migrate the run",
                )
                return "version-missing"
            if task.kind == "compensate":
                return self._run_compensation(run, wf, task)
            return self._run_step(run, wf, task)
        except SimulatedCrash:
            released = True  # a real crash never releases its lease; the reaper must.
            raise
        finally:
            if not released:
                self.store.release_lease(run.run_id, self.worker_id)

    def _task_matches(self, run: Run, task: StepTask) -> bool:
        if task.kind == "step":
            if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                return False
        else:
            if run.status != RunStatus.COMPENSATING:
                return False
        return run.current_step == task.step and run.attempt_of(_attempt_key(task)) == task.attempt

    # -------------------------------------------------------------- run step
    def _run_step(self, run: Run, wf: Workflow, task: StepTask) -> str:
        step = wf.get(task.step)
        self._chaos("before_step", run)
        working = run.model_copy(deep=True)
        ctx = StepContext(
            working, store=self.store, llm=self.llm, bus=self.bus, clock=self.clock, step_name=step.name, attempt=task.attempt
        )
        started = self.clock.now()
        try:
            outcome = step.fn(ctx)
        except StepFailed as e:
            self._record(run, step.name, task.attempt, "error", started, ctx, error=str(e))
            self._fail_or_compensate(run, f"{step.name}: {e}")
            return "failed"
        except SimulatedCrash:
            raise
        except Exception as e:  # noqa: BLE001 - any infra/model error is retryable
            self._record(run, step.name, task.attempt, "error", started, ctx, error=repr(e))
            if task.attempt < step.max_attempts:
                next_attempt = run.bump_attempt(step.name)
                run.status = RunStatus.RUNNING
                run = self._save(run)
                delay = backoff_delay(task.attempt, step.backoff_base_s, step.backoff_cap_s)
                self._enqueue(run, step.name, next_attempt, delay=delay)
                self._publish(run, "step.retry", {"step": step.name, "attempt": next_attempt, "delay_s": delay.total_seconds(), "error": repr(e)})
                return "retry"
            self._fail_or_compensate(run, f"{step.name} failed after {task.attempt} attempts: {e!r}")
            return "failed"

        self._chaos("after_step_before_commit", run)

        # ---- commit: everything below is one atomic checkpoint --------------
        run.state = ctx.state
        run.budget.charge(tokens=ctx.tokens, cost_usd=ctx.cost_usd, steps=1)
        run.completed_steps.append(step.name)

        target = getattr(outcome, "step", None) or getattr(outcome, "then", None)
        if not isinstance(outcome, (Next, Done, Wait, FanOut)) or (target and target not in wf.steps):
            # A programming error (typo'd transition, wrong return type). Retrying cannot fix it.
            self._record(run, step.name, task.attempt, "error", started, ctx, error=f"invalid outcome {outcome!r}")
            self._fail_or_compensate(run, f"{step.name} returned invalid outcome {outcome!r}")
            return "failed"

        if isinstance(outcome, Next):
            self._record(run, step.name, task.attempt, "ok", started, ctx, output=outcome.step)
            run.current_step = outcome.step
            run.status = RunStatus.RUNNING
            attempt = run.bump_attempt(outcome.step)
            run = self._save(run)
            self._chaos("after_commit_before_enqueue", run)
            self._enqueue(run, outcome.step, attempt)
            self._flush(ctx, run)
            return "ok"

        if isinstance(outcome, Done):
            self._record(run, step.name, task.attempt, "ok", started, ctx, output=outcome.result)
            run.status = RunStatus.SUCCEEDED
            run.result = outcome.result
            run.current_step = None
            run = self._save(run)
            self._flush(ctx, run)
            self._publish(run, "run.succeeded", {"result": outcome.result})
            self._notify_parent(run)
            return "done"

        if isinstance(outcome, Wait):
            self._record(run, step.name, task.attempt, "suspended", started, ctx, output=outcome.key)
            run.status = RunStatus.WAITING
            run.current_step = outcome.then
            run.wait = WaitCondition(
                kind=outcome.kind,  # type: ignore[arg-type]
                key=outcome.key,
                then=outcome.then,
                timeout_at=(self.clock.now() + outcome.timeout) if outcome.timeout else None,
                on_timeout=outcome.on_timeout,  # type: ignore[arg-type]
            )
            run = self._save(run)
            self._flush(ctx, run)
            self._publish(run, "run.waiting", {"kind": outcome.kind, "key": outcome.key, "timeout_at": run.wait.timeout_at})
            return "waiting"

        if isinstance(outcome, FanOut):
            self._record(run, step.name, task.attempt, "fanout", started, ctx, output=[c.child_key for c in outcome.children])
            keys = [c.child_key for c in outcome.children]
            if len(set(keys)) != len(keys):
                raise ValueError("child_key values must be unique within a FanOut")
            run.status = RunStatus.WAITING
            run.current_step = outcome.then
            run.fan_in = FanIn(expected=len(outcome.children), then=outcome.then)
            run.wait = WaitCondition(kind="children", key=f"children:{step.name}", then=outcome.then)
            run.state["_pending_children"] = [c.model_dump(mode="json") for c in outcome.children]
            run = self._save(run)  # checkpoint BEFORE spawning: children must find fan_in
            self._chaos("after_commit_before_enqueue", run)
            self._spawn_children(run)
            self._flush(ctx, run)
            return "fanout"

        raise AssertionError("unreachable")

    # ---------------------------------------------------------- compensation
    def _run_compensation(self, run: Run, wf: Workflow, task: StepTask) -> str:
        step = wf.get(task.step)
        assert step.compensate is not None
        working = run.model_copy(deep=True)
        ctx = StepContext(
            working, store=self.store, llm=self.llm, bus=self.bus, clock=self.clock, step_name=step.name, attempt=task.attempt
        )
        started = self.clock.now()
        try:
            step.compensate(ctx)
        except Exception as e:  # noqa: BLE001
            self._record(run, step.name, task.attempt, "error", started, ctx, kind="compensate", error=repr(e))
            if task.attempt < step.max_attempts:
                next_attempt = run.bump_attempt(_attempt_key(task))
                run = self._save(run)
                self._enqueue(run, step.name, next_attempt, kind="compensate", delay=backoff_delay(task.attempt, step.backoff_base_s, step.backoff_cap_s))
                return "retry"
            # Compensation itself failed: this needs a human. Park the run FAILED with a loud event.
            run.status = RunStatus.FAILED
            run.error = f"{run.error} | compensation of {step.name} failed after {task.attempt} attempts: {e!r}"
            run.current_step = None
            run = self._save(run)
            self._publish(run, "run.compensation_failed", {"step": step.name, "error": repr(e)}, topic="agent-alerts")
            self._notify_parent(run)
            return "failed"

        self._record(run, step.name, task.attempt, "ok", started, ctx, kind="compensate")
        run.state = ctx.state
        run.compensated_steps.append(step.name)
        nxt = self._next_compensation(run, wf)
        if nxt:
            run.current_step = nxt
            attempt = run.bump_attempt(f"~{nxt}")
            run = self._save(run)
            self._enqueue(run, nxt, attempt, kind="compensate")
            self._flush(ctx, run)
            return "compensating"
        run.status = RunStatus.COMPENSATED
        run.current_step = None
        run = self._save(run)
        self._flush(ctx, run)
        self._publish(run, "run.compensated", {"error": run.error})
        self._notify_parent(run)
        return "compensated"

    def _next_compensation(self, run: Run, wf: Workflow) -> str | None:
        seen: list[str] = []
        for s in run.completed_steps:  # dedupe, keep first occurrence order
            if s not in seen:
                seen.append(s)
        for s in reversed(seen):
            if s in run.compensated_steps:
                continue
            step = wf.steps.get(s)
            if step and step.compensate:
                return s
        return None

    def _fail_or_compensate(self, run: Run, reason: str, *, final: RunStatus = RunStatus.FAILED) -> Run:
        run.error = reason
        run.wait = None
        run.fan_in = None
        wf = self.registry.get(run.workflow, run.workflow_version)
        nxt = self._next_compensation(run, wf) if wf else None
        if nxt:
            run.status = RunStatus.COMPENSATING
            run.current_step = nxt
            attempt = run.bump_attempt(f"~{nxt}")
            run = self._save(run)
            self._enqueue(run, nxt, attempt, kind="compensate")
            self._publish(run, "run.compensating", {"error": reason, "first": nxt})
            return run
        run.status = final
        run.current_step = None
        run = self._save(run)
        self._publish(run, f"run.{final.value.lower()}", {"error": reason})
        self._notify_parent(run)
        return run

    # ------------------------------------------------------------- resume
    def resume(self, event: Event) -> Run | None:
        """Wake a WAITING run whose wait key matches. Idempotent per ``event_id``.

        Returns the updated run, or None if the event did not apply (already
        resumed, wrong key, terminal). Callers should treat None as success —
        duplicate webhooks are normal.
        """
        for _ in range(8):  # optimistic-concurrency retry loop
            run = self.store.get(event.run_id)
            if run is None:
                raise KeyError(f"unknown run {event.run_id}")
            if run.status != RunStatus.WAITING or run.wait is None or run.wait.key != event.key:
                return None
            if run.wait.kind == "children":
                return None  # children resume via record_child_result
            events = run.state.setdefault("events", {})
            events[event.key] = {
                "event_id": event.event_id,
                "payload": event.payload,
                "occurred_at": event.occurred_at.isoformat(),
            }
            then = run.wait.then
            run.wait = None
            run.status = RunStatus.RUNNING
            run.current_step = then
            attempt = run.bump_attempt(then)
            try:
                run = self._save(run)
            except ConflictError:
                continue
            self._enqueue(run, then, attempt)
            self._publish(run, "run.resumed", {"key": event.key, "then": then})
            return run
        raise ConflictError(f"could not resume {event.run_id} after repeated conflicts")

    def cancel(self, run_id: str) -> Run | None:
        for _ in range(8):
            run = self.store.get(run_id)
            if run is None or run.status.terminal:
                return run
            run.cancel_requested = True
            if run.status == RunStatus.WAITING:
                # Nothing will ever deliver a task for a waiting run, so finish it here.
                return self._fail_or_compensate(run, "cancelled by user", final=RunStatus.CANCELLED)
            try:
                return self._save(run)  # a running step sees the flag on its next task
            except ConflictError:
                continue
        raise ConflictError(f"could not cancel {run_id}")

    # ------------------------------------------------------------- fan-out
    def _spawn_children(self, run: Run) -> int:
        """Start every child in ``_pending_children`` that does not exist yet.

        Deterministic child ids (``parent--childkey``) make this safe to call
        repeatedly — from the step commit path and from the reaper.
        """
        specs = [ChildSpec.model_validate(c) for c in run.state.get("_pending_children", [])]
        spawned = 0
        for spec in specs:
            child_id = f"{run.run_id}--{spec.child_key}"
            if self.store.get(child_id) is None:
                self.start(
                    spec.workflow,
                    spec.input,
                    run_id=child_id,
                    budget=spec.budget,
                    parent_run_id=run.run_id,
                    child_key=spec.child_key,
                )
                spawned += 1
        return spawned

    def _notify_parent(self, child: Run) -> None:
        if not child.parent_run_id or not child.child_key:
            return
        error = None if child.status == RunStatus.SUCCEEDED else (child.error or child.status.value)
        parent = self.store.record_child_result(child.parent_run_id, child.child_key, child.result, error)
        self._maybe_complete_fan_in(parent)

    def _maybe_complete_fan_in(self, parent: Run) -> Run | None:
        for _ in range(8):
            fi = parent.fan_in
            if parent.status != RunStatus.WAITING or fi is None or fi.completed < fi.expected:
                return None
            then = fi.then
            parent.state["children"] = {"results": fi.results, "failures": fi.failures}
            parent.state.pop("_pending_children", None)
            parent.fan_in = None
            parent.wait = None
            parent.status = RunStatus.RUNNING
            parent.current_step = then
            attempt = parent.bump_attempt(then)
            try:
                parent = self._save(parent)
            except ConflictError:
                refreshed = self.store.get(parent.run_id)
                if refreshed is None:
                    return None
                parent = refreshed
                continue
            self._enqueue(parent, then, attempt)
            self._publish(parent, "run.fan_in_complete", {"then": then, "failures": list(fi.failures)})
            return parent
        raise ConflictError(f"fan-in transition for {parent.run_id} kept conflicting")

    # ------------------------------------------------------------- reaper
    def reap(self) -> dict[str, list[str]]:
        """Periodic repair job (Cloud Scheduler -> POST /internal/reap).

        Recovers from every crash window the engine cannot cover in-line:
        expired leases (worker died mid-step), wait timeouts, and fan-outs that
        were checkpointed but whose children were not all spawned.
        """
        now = self.clock.now()
        report: dict[str, list[str]] = {"leases_recovered": [], "orphans_redriven": [], "waits_timed_out": [], "children_respawned": [], "fan_in_completed": []}

        stale_after = self.lease_ttl * 2
        for status in (RunStatus.PENDING, RunStatus.RUNNING, RunStatus.COMPENSATING):
            for run in self.store.list_runs(status=status.value):
                if not run.current_step:
                    continue
                expired = run.lease is not None and run.lease.expired(now)
                orphaned = run.lease is None and now - run.updated_at >= stale_after
                if not (expired or orphaned):
                    continue
                # Re-deliver the *current* attempt. If the original task still exists in
                # Cloud Tasks, the name collision makes this a no-op — which is correct.
                run.lease = None
                try:
                    run = self._save(run)
                except ConflictError:
                    continue
                kind = "compensate" if run.status == RunStatus.COMPENSATING else "step"
                self._enqueue(run, run.current_step, run.attempt_of(_attempt_key_for(kind, run.current_step)), kind=kind)
                report["leases_recovered" if expired else "orphans_redriven"].append(run.run_id)

        for run in self.store.list_runs(status=RunStatus.WAITING.value):
            w = run.wait
            if w is None:
                continue
            if w.kind == "children":
                if self._spawn_children(run):
                    report["children_respawned"].append(run.run_id)
                if run.fan_in and run.fan_in.completed >= run.fan_in.expected and self._maybe_complete_fan_in(run):
                    report["fan_in_completed"].append(run.run_id)
                continue
            if w.timeout_at and now >= w.timeout_at:
                if w.on_timeout == "resume":
                    self.resume(Event(run_id=run.run_id, key=w.key, payload={"timed_out": True}))
                else:
                    self._fail_or_compensate(run, f"timed out waiting for {w.key}")
                report["waits_timed_out"].append(run.run_id)
        return report

    # ------------------------------------------------------------ plumbing
    def _save(self, run: Run) -> Run:
        run.updated_at = self.clock.now()
        return self.store.save(run, expected_version=run.version)

    def _enqueue(self, run: Run, step: str, attempt: int, *, kind: str = "step", delay: timedelta | None = None) -> None:
        task = StepTask(run_id=run.run_id, step=step, attempt=attempt, kind=kind)  # type: ignore[arg-type]
        if delay:
            task.not_before = self.clock.now() + delay
        self.queue.enqueue(task, delay=delay)

    def _record(
        self,
        run: Run,
        step: str,
        attempt: int,
        status: str,
        started: Any,
        ctx: StepContext,
        *,
        kind: str = "step",
        error: str | None = None,
        output: Any = None,
    ) -> None:
        run.history.append(
            StepRecord(
                step=step,
                attempt=attempt,
                kind=kind,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                started_at=started,
                finished_at=self.clock.now(),
                worker=self.worker_id,
                tokens=ctx.tokens,
                cost_usd=ctx.cost_usd,
                error=error,
                output_digest=digest(output) if output is not None else None,
            )
        )

    def _flush(self, ctx: StepContext, run: Run) -> None:
        for event_type, payload in ctx.buffered_events:
            self._publish(run, event_type, payload)

    def _publish(self, run: Run, event_type: str, payload: dict[str, Any] | None = None, *, topic: str | None = None) -> None:
        body = {
            "type": event_type,
            "run_id": run.run_id,
            "workflow": run.workflow,
            "status": run.status.value,
            "step": run.current_step,
            "at": self.clock.now().isoformat(),
            **(payload or {}),
        }
        try:
            self.bus.publish(topic or self.PROGRESS_TOPIC, body, {"run_id": run.run_id, "type": event_type})
        except Exception:  # noqa: BLE001 - progress events are best-effort
            log.exception("publish failed for %s", event_type)

    def _chaos(self, point: str, run: Run) -> None:
        if self.chaos:
            self.chaos(point, run)


def _attempt_key(task: StepTask) -> str:
    return _attempt_key_for(task.kind, task.step)


def _attempt_key_for(kind: str, step: str) -> str:
    return step if kind == "step" else f"~{step}"
