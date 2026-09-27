"""Ports: the seams between the engine and infrastructure.

The engine only talks to these protocols. ``adapters/memory`` implements them
in-process (tests, notebooks, local dev); ``adapters/gcp`` implements them with
Firestore, Cloud Tasks, Pub/Sub and Gemini on Vertex AI.

Keeping this boundary thin is what lets the same workflow code run in a
notebook, in ``pytest``, and on Cloud Run without modification.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Protocol, runtime_checkable

from .models import Run, StepTask, LLMResponse


class ConflictError(RuntimeError):
    """Optimistic-concurrency failure: the run changed under us. Reload and retry."""


class LeaseHeldError(RuntimeError):
    """Another live worker holds the lease. Back off; do not run the step."""


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class StateStore(Protocol):
    """Durable run storage with optimistic concurrency and idempotent effects."""

    def create(self, run: Run) -> Run: ...

    def get(self, run_id: str) -> Run | None: ...

    def save(self, run: Run, *, expected_version: int) -> Run:
        """Persist ``run`` iff the stored version == ``expected_version``.

        Bumps ``run.version`` and returns the saved run. Raises ConflictError
        otherwise. This is the only write path for run state, which is what
        makes checkpoints atomic and duplicate deliveries harmless.
        """
        ...

    def acquire_lease(self, run_id: str, owner: str, ttl: timedelta, now: datetime) -> Run:
        """Atomically claim the run. Raises LeaseHeldError if a live lease exists."""
        ...

    def release_lease(self, run_id: str, owner: str) -> None: ...

    def list_runs(self, *, status: str | None = None, limit: int = 100) -> list[Run]: ...

    # Idempotent side effects -------------------------------------------------
    def effect_get(self, key: str) -> dict[str, Any] | None: ...

    def effect_put(self, key: str, value: dict[str, Any]) -> bool:
        """Record an effect result. Returns False if the key already existed."""
        ...

    # Fan-in bookkeeping (must be atomic) ------------------------------------
    def record_child_result(self, parent_run_id: str, child_key: str, result: Any, error: str | None) -> Run:
        """Atomically record a child outcome on the parent and return the updated parent."""
        ...


@runtime_checkable
class TaskQueue(Protocol):
    """At-least-once delivery of :class:`StepTask` with name-based dedup and delays."""

    def enqueue(self, task: StepTask, *, delay: timedelta | None = None) -> bool:
        """Returns False if a task with the same dedup_key already exists."""
        ...


@runtime_checkable
class EventBus(Protocol):
    """Fire-and-forget notifications (progress, completion, DLQ)."""

    def publish(self, topic: str, payload: dict[str, Any], attributes: dict[str, str] | None = None) -> str: ...


@runtime_checkable
class LLM(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse: ...


# Convenience alias for step functions used by the workflow decorator.
StepFn = Callable[["StepContext"], Any]  # noqa: F821  (StepContext lives in workflow.py)
