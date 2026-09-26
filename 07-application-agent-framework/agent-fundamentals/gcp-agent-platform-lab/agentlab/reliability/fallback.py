"""Graceful degradation: a fallback chain, and tools that tell the model when they are down (notebook 10).

The failure a user remembers is not the outage, it is the agent that pretended nothing
was wrong. Two mechanisms fix that:

* ``FallbackChain`` tries steps in order (primary model → smaller model → cached answer →
  "I'll follow up" with a durable task) and *labels* the result as degraded, so the caller
  can say so and dashboards can count it.
* ``GracefulTool`` wraps an agentlab tool with retry + circuit breaker and converts an open
  circuit into a structured ``unavailable`` result whose hint tells the model what to do
  instead of looping on a dead dependency.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ..agents.state import TaskRecord, TaskStatus, TaskStore
from ..agents.tools import Tool, ToolContext, ToolResult
from ..llm.types import LLM
from .breaker import Bulkhead, BulkheadFull, CircuitBreaker, CircuitOpen
from .retry import RetryableError, RetryPolicy, retry

StepFn = Callable[..., Awaitable[Any]]


# ------------------------------------------------------------ fallback chain
@dataclass
class Step:
    name: str
    fn: StepFn


@dataclass
class FallbackResult:
    value: Any
    step: str                       # which step served the request
    degraded: bool                  # True whenever the primary did not
    errors: list[tuple[str, str]] = field(default_factory=list)   # (step, error) for the steps that failed


class FallbackExhausted(RuntimeError):
    def __init__(self, errors: list[tuple[str, str]]):
        super().__init__("every fallback step failed: " + "; ".join(f"{s}: {e}" for s, e in errors))
        self.errors = errors


class FallbackChain:
    """Try each step in order; the first that returns wins.

    ``fallback_on`` decides which errors fall through (default: all of them — a fallback
    chain exists precisely for the failures nobody predicted). ``served`` and ``failed``
    count outcomes per step; a rising ``degraded`` share is the alert.
    """

    def __init__(self, steps: list[Step] | list[tuple[str, StepFn]], fallback_on: Callable[[BaseException], bool] = lambda exc: True):
        self.steps = [s if isinstance(s, Step) else Step(*s) for s in steps]
        if not self.steps:
            raise ValueError("a fallback chain needs at least one step")
        self.fallback_on = fallback_on
        self.served: Counter[str] = Counter()
        self.failed: Counter[str] = Counter()

    async def run(self, *args: Any, **kwargs: Any) -> FallbackResult:
        errors: list[tuple[str, str]] = []
        for index, step in enumerate(self.steps):
            try:
                value = await step.fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - fallback_on classifies
                self.failed[step.name] += 1
                errors.append((step.name, f"{type(exc).__name__}: {exc}"))
                if not self.fallback_on(exc):
                    raise
                continue
            self.served[step.name] += 1
            return FallbackResult(value=value, step=step.name, degraded=index > 0, errors=errors)
        raise FallbackExhausted(errors)

    def degraded_share(self) -> float:
        total = sum(self.served.values())
        return 0.0 if total == 0 else 1.0 - self.served[self.steps[0].name] / total


# ------------------------------------------------- lab steps for the chain
def model_step(llm: LLM) -> StepFn:
    """A step that answers ``request`` with one model call (primary or smaller model)."""

    async def step(request: str) -> str:
        resp = await llm.generate([{"role": "user", "content": request}])
        return resp.text or ""

    return step


def cached_answer_step(cache: Mapping[str, str]) -> StepFn:
    """Serve a previously computed answer; a miss is a failure so the chain moves on."""

    async def step(request: str) -> str:
        if request not in cache:
            raise KeyError(f"no cached answer for {request!r}")
        return cache[request]

    return step


def enqueue_followup_step(tasks: TaskStore, kind: str = "followup") -> StepFn:
    """Last resort: record a durable task and acknowledge it, instead of failing silently.

    The acknowledgement is the honest answer when nothing can serve the request now — the
    task survives the process, and the user is told what will happen next.
    """

    async def step(request: str) -> dict[str, Any]:
        task = tasks.create(TaskRecord(kind=kind, stage="queued", checkpoint={"request": request},
                                       status_message="queued for follow-up"))
        return {"acknowledged": True, "task_id": task.id, "status": TaskStatus.WORKING.value,
                "message": "I can't complete this right now. I've queued it and will follow up."}

    return step


def lab_fallback_chain(primary: LLM, smaller: LLM, cache: Mapping[str, str], tasks: TaskStore) -> FallbackChain:
    """The chain the notebook demonstrates: primary → smaller → cached → follow-up task."""
    return FallbackChain([
        ("primary_model", model_step(primary)),
        ("smaller_model", model_step(smaller)),
        ("cached_answer", cached_answer_step(cache)),
        ("enqueue_followup", enqueue_followup_step(tasks)),
    ])


# -------------------------------------------------------------- graceful tool
class _RetryableToolFailure(RetryableError):
    """Carries a structured retryable ToolResult through retry/breaker as an exception."""

    def __init__(self, result: ToolResult):
        super().__init__(result.error.message if result.error else "retryable tool failure")
        self.result = result


class GracefulTool:
    """Retry + breaker (+ optional bulkhead) around an agentlab tool, with structured degradation.

    Ordering is retry → breaker → tool: every attempt passes through the breaker so a failing
    dependency trips it quickly, and ``CircuitOpen`` is not retryable, so the retry loop stops
    the moment the circuit opens. Whatever happens, the model receives a ``ToolResult`` — an
    open circuit becomes ``unavailable`` with a hint — and never a stack trace or a hang.

    The retry schedule must fit inside the hop's deadline (the loop's ``tool_timeout_s``); the
    defaults sum to under 5 s. ``sleep``/``rng`` are injectable so tests run instantly.
    """

    def __init__(
        self,
        tool: Tool,
        breaker: CircuitBreaker,
        policy: RetryPolicy | None = None,
        *,
        bulkhead: Bulkhead | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.tool = tool
        self.spec = tool.spec
        self.breaker = breaker
        self.policy = policy or RetryPolicy()
        self.bulkhead = bulkhead
        self.sleep = sleep
        self.rng = rng
        self.attempts = 0

    async def _attempt(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.attempts += 1
        result = await self.tool.run(args, ctx)
        if not result.ok and result.error is not None and result.error.retryable:
            raise _RetryableToolFailure(result)   # lets retry and the breaker see it as a failure
        return result

    async def _guarded(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self.bulkhead is None:
            return await self.breaker.call(self._attempt, args, ctx)
        return await self.bulkhead.run(self.breaker.call, self._attempt, args, ctx)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        try:
            return await retry(lambda: self._guarded(args, ctx), self.policy, sleep=self.sleep, rng=self.rng)
        except CircuitOpen as exc:
            return ToolResult.failure(
                "unavailable", f"{name} is temporarily unavailable (circuit open; retry in {exc.retry_after_s:.0f}s)",
                retryable=True,
                hint=f"{name} is down right now. Do not call it again this turn; tell the user this part could not be "
                     f"completed and offer to follow up when it recovers.")
        except BulkheadFull:
            return ToolResult.failure(
                "overloaded", f"{name} is at capacity", retryable=True,
                hint=f"{name} is overloaded. Answer with what you already have or tell the user to try again shortly.")
        except _RetryableToolFailure as exc:
            return exc.result   # retries exhausted: hand back the last structured error
