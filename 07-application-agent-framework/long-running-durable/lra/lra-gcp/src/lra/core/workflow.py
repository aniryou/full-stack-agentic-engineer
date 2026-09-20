"""Workflow definition DSL.

A workflow is a named set of *steps*. Each step is a plain Python function
``fn(ctx) -> outcome`` where ``outcome`` is one of:

* :class:`Next`   – continue at another step (the transition may be chosen by an
  LLM at runtime; that is what makes this an *agentic* state machine rather
  than a fixed DAG).
* :class:`Done`   – finish the run with a result.
* :class:`Wait`   – suspend until an external event/approval/timer arrives.
* :class:`FanOut` – spawn child runs and suspend until they all complete.

Steps may mutate ``ctx.state``; the engine checkpoints the mutated state
atomically with the outcome. If a step raises, nothing it wrote is persisted
(the step is retried from the same checkpoint), so steps should be written to
be *re-runnable*: side effects go through :meth:`StepContext.effect`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from .models import Budget, ChildSpec, LLMResponse, Run, digest
from .ports import LLM, Clock, EventBus, StateStore

log = logging.getLogger("lra.workflow")


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
@dataclass
class Next:
    step: str


@dataclass
class Done:
    result: Any = None


@dataclass
class Wait:
    """Suspend the run.

    ``key`` must match the ``Event.key`` that resumes it. Keep keys stable and
    unguessable enough for the channel (e.g. ``approval:<run_id>``).
    """

    key: str
    then: str
    kind: str = "event"  # event | approval | timer
    timeout: timedelta | None = None
    on_timeout: str = "fail"  # fail | resume


@dataclass
class FanOut:
    children: list[ChildSpec]
    then: str


@dataclass
class StepFailed(Exception):
    """Raise to fail the run *without* retrying (a business-rule failure)."""

    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.reason


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------
class StepContext:
    """What a step can see and do.

    Everything mutable here (``state``, budget charges, buffered events) is
    committed *together* by the engine after the step returns. If the step
    raises, all of it is discarded.
    """

    def __init__(
        self,
        run: Run,
        *,
        store: StateStore,
        llm: LLM,
        bus: EventBus,
        clock: Clock,
        step_name: str,
        attempt: int,
    ) -> None:
        self.run = run
        self.state: dict[str, Any] = run.state  # engine passes a deep copy
        self.input = run.input
        self.step_name = step_name
        self.attempt = attempt
        self._store = store
        self._llm = llm
        self._bus = bus
        self._clock = clock
        self.tokens = 0
        self.cost_usd = 0.0
        self.buffered_events: list[tuple[str, dict[str, Any]]] = []
        self.log = logging.getLogger(f"lra.step.{run.workflow}.{step_name}")

    # -- time & identity ----------------------------------------------------
    def now(self) -> datetime:
        return self._clock.now()

    @property
    def run_id(self) -> str:
        return self.run.run_id

    # -- LLM with budget accounting ----------------------------------------
    def llm(self, prompt: str, **kw: Any) -> LLMResponse:
        """Call the model and charge the run's budget.

        Budget is enforced *before* the call using what has been spent so far,
        so an agent that is already over budget cannot make one more call.
        """
        reason = self.run.budget.exceeded(self.now())
        if reason:
            raise StepFailed(f"budget exceeded before LLM call: {reason}")
        resp = self._llm.generate(prompt, **kw)
        self.tokens += resp.usage.total_tokens
        self.cost_usd += resp.usage.cost_usd
        return resp

    # -- idempotent side effects -------------------------------------------
    def effect(self, key: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run ``fn`` at most once per ``key`` across retries and duplicate deliveries.

        The effect record is written to the store *before* the step's
        checkpoint. That ordering is deliberate: if we crash after the effect
        but before the checkpoint, the retry finds the record and skips the
        effect instead of, say, charging a card twice.
        """
        full_key = f"{self.run_id}:{key}"
        existing = self._store.effect_get(full_key)
        if existing is not None:
            self.log.info("effect %s already applied; skipping", key)
            return existing
        value = fn()
        if not self._store.effect_put(full_key, value):
            # Lost a race with a concurrent duplicate; use its result.
            return self._store.effect_get(full_key) or value
        return value

    # -- progress / notifications ------------------------------------------
    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Buffer a progress event; published after the checkpoint commits."""
        self.buffered_events.append((event_type, payload or {}))

    # -- cooperative cancellation ------------------------------------------
    def check_cancelled(self) -> None:
        fresh = self._store.get(self.run_id)
        if fresh and fresh.cancel_requested:
            raise StepFailed("cancelled by user")

    # -- convenience --------------------------------------------------------
    def fingerprint(self, obj: Any) -> str:
        return digest(obj)


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    fn: Callable[[StepContext], Any]
    compensate: Callable[[StepContext], Any] | None = None
    max_attempts: int = 3
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 300.0


@dataclass
class Workflow:
    name: str
    version: str = "1"
    start: str | None = None
    steps: dict[str, Step] = field(default_factory=dict)
    default_budget: Budget = field(default_factory=Budget)

    def step(
        self,
        name: str | None = None,
        *,
        start: bool = False,
        compensate: Callable[[StepContext], Any] | None = None,
        max_attempts: int = 3,
        backoff_base_s: float = 2.0,
        backoff_cap_s: float = 300.0,
    ) -> Callable[[Callable[[StepContext], Any]], Callable[[StepContext], Any]]:
        """Decorator registering a step function."""

        def deco(fn: Callable[[StepContext], Any]) -> Callable[[StepContext], Any]:
            step_name = name or fn.__name__
            if step_name in self.steps:
                raise ValueError(f"duplicate step {step_name!r} in workflow {self.name!r}")
            self.steps[step_name] = Step(
                name=step_name,
                fn=fn,
                compensate=compensate,
                max_attempts=max_attempts,
                backoff_base_s=backoff_base_s,
                backoff_cap_s=backoff_cap_s,
            )
            if start:
                if self.start is not None:
                    raise ValueError(f"workflow {self.name!r} already has a start step")
                self.start = step_name
            return fn

        return deco

    def get(self, name: str) -> Step:
        try:
            return self.steps[name]
        except KeyError as e:
            raise KeyError(f"workflow {self.name!r} has no step {name!r}") from e

    def validate(self) -> None:
        if not self.start:
            raise ValueError(f"workflow {self.name!r} has no start step (use @wf.step(start=True))")
