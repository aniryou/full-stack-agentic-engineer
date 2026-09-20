"""Pattern 1 — The Durable Agent Loop ("wake, do one thing, checkpoint, sleep").

A ReAct loop where *each wake-up performs exactly one step* and everything the
next step needs is in the store. The process holding the loop can die at any
line and the run still converges. The mechanics:

    wake-up (Cloud Tasks → POST /internal/tasks/step)
      │
      ├─ acquire lease            (exclusivity: only one worker advances a run)
      ├─ pending STARTED step?    (crash recovery: finish what we intended)
      ├─ duplicate delivery?      (at-least-once guard)
      ├─ budget check             (deterministic circuit breaker)
      ├─ LLM decides              (non-deterministic! record it immediately)
      │     final answer → SUCCEEDED
      │     tool needing approval → park WAITING_HUMAN (see hitl.py)
      │     tool call → journal STARTED intent + idempotency key
      ├─ checkpoint #1            (intent is durable BEFORE the side effect)
      ├─ execute tool idempotently
      │     async tool → park WAITING_EVENT, schedule a poll (see below)
      ├─ checkpoint #2            (result is durable)
      └─ enqueue next wake-up     (named task → duplicates collapse)

Why record intent before acting? Because the LLM is not deterministic. If we
crash after the tool ran but before we saved, a retry that *re-asks the model*
might choose different arguments and produce a second, different side effect.
With the intent journaled, the retry re-executes the *same* call under the
*same* idempotency key and the downstream system de-duplicates it.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Callable

from ..core.budget import BudgetExceeded, charge, check_budget
from ..core.llm import LLM, PriceCard
from ..core.models import Budget, Run, RunStatus, StepKind, StepRecord, StepStatus, new_run_id
from ..core.store import RunStore
from ..core.tools import FaultInjector, IdempotencyStore, SimulatedCrash, ToolContext, ToolError, ToolRegistry, run_idempotent
from ..core.transport import Dispatcher, Envelope

SYSTEM_PROMPT = (
    "You are an autonomous agent completing a long-running task. You may call tools. "
    "A journal of everything that already happened is provided; never repeat completed work. "
    "When the goal is met, reply with a concise final answer instead of calling a tool."
)


class DurableAgentLoop:
    def __init__(
        self,
        *,
        store: RunStore,
        llm: LLM,
        tools: ToolRegistry,
        dispatcher: Dispatcher,
        idempotency: IdempotencyStore,
        price: PriceCard | None = None,
        worker_id: str = "worker-1",
        lease_ttl_s: float = 60.0,
        clock: Callable[[], float] = time.time,
        faults: FaultInjector | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        notify: Callable[[Run], None] | None = None,
        poll_base_s: float = 5.0,
        poll_max_s: float = 300.0,
        poll_max_wait_s: float = 6 * 3600,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tools = tools
        self.dispatcher = dispatcher
        self.idem = idempotency
        self.price = price or PriceCard()
        self.worker_id = worker_id
        self.lease_ttl_s = lease_ttl_s
        self.clock = clock
        self.faults = faults or FaultInjector()
        self.system_prompt = system_prompt
        self.notify = notify or (lambda run: None)
        self.poll_base_s, self.poll_max_s, self.poll_max_wait_s = poll_base_s, poll_max_s, poll_max_wait_s

    # ------------------------------------------------------------------ API
    def start(self, goal: str, budget: Budget | None = None, state: dict[str, Any] | None = None, run_id: str | None = None) -> Run:
        run = Run(run_id=run_id or new_run_id(), goal=goal, budget=budget or Budget(), state=state or {})
        run = self.store.create(run)
        self.dispatcher.enqueue(Envelope(run.run_id, 0, "step"))
        return run

    def handle(self, env: Envelope) -> Run:
        """Route a wake-up envelope (what the HTTP handler calls)."""
        if env.kind == "step":
            return self.step(env.run_id, expected_index=env.step_index)
        if env.kind == "poll":
            return self.poll(env.run_id)
        raise ValueError(f"unknown envelope kind {env.kind}")

    def step(self, run_id: str, expected_index: int | None = None) -> Run:
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        released = False
        try:
            if run.status.terminal or run.status in (RunStatus.WAITING_HUMAN, RunStatus.WAITING_EVENT):
                return run                                   # parked or done: a stray wake-up is a no-op
            pending = run.pending_step()
            if pending is not None:                         # we died mid-step last time
                run.state["recoveries"] = run.state.get("recoveries", 0) + 1
                return self._execute_and_continue(run, pending)
            if expected_index is not None and run.next_index() > expected_index:
                self._enqueue_next(run)                      # duplicate delivery; make sure the chain continues
                return run
            return self._advance(run)
        except SimulatedCrash:
            released = True                                  # a dead process never reaches `finally`:
            raise                                            # the lease must EXPIRE, not be released
        finally:
            if not released:                                 # normal exit or ordinary exception → release, let the platform retry
                self.store.release_lease(run_id, self.worker_id)

    # ------------------------------------------------------------ internals
    def _advance(self, run: Run) -> Run:
        run.status = RunStatus.RUNNING
        now = self.clock()
        try:
            check_budget(run, now)
        except BudgetExceeded as e:
            return self._fail(run, f"budget: {e}")

        decision = self.llm.decide(self.system_prompt, self.messages(run), self.tools.specs())
        cost = charge(run, decision.tokens_in, decision.tokens_out, self.price)
        rec = StepRecord(
            index=run.next_index(), kind=StepKind.LLM, status=StepStatus.DONE, name="decide",
            tokens_in=decision.tokens_in, tokens_out=decision.tokens_out, cost_usd=cost,
        )
        rec.finish(output={"kind": decision.kind, "tool": decision.tool_name, "args": decision.tool_args, "text": decision.text})
        run.append(rec)

        if decision.kind == "final":
            run.result = decision.text
            run.status = RunStatus.SUCCEEDED
            return self.store.save(run)

        if decision.tool_name not in self.tools:              # let the model correct itself next step
            obs = StepRecord(index=run.next_index(), kind=StepKind.SYSTEM, status=StepStatus.DONE, name="error")
            obs.finish(output=f"unknown tool {decision.tool_name!r}")
            run.append(obs)
            run = self.store.save(run)
            self._enqueue_next(run)
            return run

        tool = self.tools.get(decision.tool_name)
        idx = run.next_index()
        if tool.requires_approval:
            run.status = RunStatus.WAITING_HUMAN
            run.waiting_on = {
                "type": "approval", "token": secrets.token_urlsafe(16), "tool": tool.name,
                "args": decision.tool_args, "step_index": idx, "requested_at": now,
            }
            run = self.store.save(run)
            self.notify(run)
            return run

        intent = StepRecord(
            index=idx, kind=StepKind.TOOL, status=StepStatus.STARTED, name=tool.name,
            input=decision.tool_args, idempotency_key=f"{run.run_id}:{idx}",
        )
        run.append(intent)
        run = self.store.save(run)                            # checkpoint #1: intent
        self.faults.maybe_crash("after_intent")
        return self._execute_and_continue(run, run.journal[-1])

    def _execute_and_continue(self, run: Run, rec: StepRecord) -> Run:
        tool = self.tools.get(rec.name)
        ctx = ToolContext(run.run_id, rec.index, rec.idempotency_key or f"{run.run_id}:{rec.index}", run.state)
        try:
            output, replayed = run_idempotent(self.idem, ctx.idempotency_key, lambda: tool.fn(rec.input, ctx))
            self.faults.maybe_crash("after_side_effect")
        except ToolError as e:
            rec.finish(error=str(e))
            run = self.store.save(run)
            self._enqueue_next(run)
            return run

        if tool.is_async:                                     # tool returned a ticket; park until the world calls back
            rec.finish(output={"ticket": output, "pending": True, "replayed": replayed})
            run.status = RunStatus.WAITING_EVENT
            run.waiting_on = {"type": "event", "tool": tool.name, "ticket": output, "step_index": rec.index,
                              "submitted_at": self.clock(), "attempt": 0}
            run = self.store.save(run)
            self._schedule_poll(run)
            return run

        rec.finish(output={"result": output, "replayed": replayed})
        run.usage.steps += 1
        run = self.store.save(run)                            # checkpoint #2: result
        self.faults.maybe_crash("before_enqueue")
        self._enqueue_next(run)
        return run

    def _enqueue_next(self, run: Run) -> None:
        """Wake-up names are keyed by the journal index the wake-up will work on:
        a pending STARTED step's index, else the next free index. Duplicates collapse."""
        if run.status == RunStatus.RUNNING:
            pending = run.pending_step()
            idx = pending.index if pending else run.next_index()
            self.dispatcher.enqueue(Envelope(run.run_id, idx, "step"))

    def _fail(self, run: Run, reason: str) -> Run:
        run.status = RunStatus.FAILED
        run.error = reason
        return self.store.save(run)

    # ---------------------------------------------------- async tool support
    def _schedule_poll(self, run: Run) -> None:
        w = run.waiting_on or {}
        delay = min(self.poll_base_s * (2 ** w.get("attempt", 0)), self.poll_max_s)
        self.dispatcher.enqueue(Envelope(run.run_id, w["step_index"], "poll", payload={"attempt": w.get("attempt", 0)},
                                         not_before=self.clock() + delay))

    def poll(self, run_id: str) -> Run:
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        try:
            if run.status != RunStatus.WAITING_EVENT:
                return run
            w = run.waiting_on or {}
            tool = self.tools.get(w["tool"])
            assert tool.poll is not None, f"async tool {tool.name} has no poll()"
            status = tool.poll(w["ticket"])
            if status.get("done"):
                return self._resume_from_event(run, status.get("result"))
            if self.clock() - w["submitted_at"] > self.poll_max_wait_s:
                return self._fail(run, f"external job {w['ticket']} timed out")
            w["attempt"] = w.get("attempt", 0) + 1
            run = self.store.save(run)
            self._schedule_poll(run)
            return run
        finally:
            self.store.release_lease(run_id, self.worker_id)

    def resume_with_event(self, run_id: str, ticket: str, result: Any) -> Run:
        """Webhook path: the external system calls us instead of us polling it."""
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        try:
            if run.status != RunStatus.WAITING_EVENT or (run.waiting_on or {}).get("ticket") != ticket:
                return run                                    # stale or duplicate callback: ignore
            return self._resume_from_event(run, result)
        finally:
            self.store.release_lease(run_id, self.worker_id)

    def _resume_from_event(self, run: Run, result: Any) -> Run:
        w = run.waiting_on or {}
        rec = StepRecord(index=run.next_index(), kind=StepKind.EVENT, status=StepStatus.DONE, name=f"{w['tool']}:result",
                         input={"ticket": w.get("ticket")})
        rec.finish(output={"result": result})
        run.append(rec)
        run.usage.steps += 1
        run.waiting_on = None
        run.status = RunStatus.RUNNING
        run = self.store.save(run)
        self._enqueue_next(run)
        return run

    # ------------------------------------------------------------- prompting
    def messages(self, run: Run) -> list[dict[str, Any]]:
        """Rebuild the conversation from the journal (event sourcing → prompt)."""
        msgs: list[dict[str, Any]] = [{"role": "user", "content": f"GOAL: {run.goal}\nSTATE: {json.dumps(run.state, default=str)}"}]
        for s in run.journal:
            if s.kind == StepKind.LLM:
                o = s.output or {}
                if o.get("kind") == "tool_call":
                    msgs.append({"role": "model", "content": f"CALL {o['tool']}({json.dumps(o['args'], default=str)})"})
            elif s.kind == StepKind.TOOL:
                if s.status == StepStatus.DONE:
                    msgs.append({"role": "user", "content": f"OBSERVATION[{s.name}]: {json.dumps(s.output, default=str)}"})
                elif s.status == StepStatus.FAILED:
                    msgs.append({"role": "user", "content": f"TOOL_ERROR[{s.name}]: {s.error}"})
            elif s.kind in (StepKind.HUMAN, StepKind.EVENT, StepKind.SYSTEM):
                msgs.append({"role": "user", "content": f"{s.kind.value.upper()}[{s.name}]: {json.dumps(s.output, default=str)}"})
        return msgs
