"""Domain model for durable, long-running agent runs.

Everything the engine needs to *resume* a run after a crash, a scale-to-zero,
or a multi-day wait lives in the :class:`Run` document. The document is the
single source of truth; conversation history is *not* the state machine.

Design rules encoded here:

* State is explicit (``Run.state``), versioned (``Run.version``) and small
  enough to fit in one Firestore document (1 MiB). Large artefacts go to GCS
  and only their URIs live in ``state``.
* Every unit of work is addressed by ``(run_id, step, attempt)``. That triple
  is the idempotency key for task delivery.
* Budgets are first-class so a runaway agent fails *closed*.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def digest(obj: Any) -> str:
    """Stable content hash used for output fingerprints and idempotency keys."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class RunStatus(str, Enum):
    PENDING = "PENDING"          # created, first task enqueued
    RUNNING = "RUNNING"          # a step is claimable / executing
    WAITING = "WAITING"          # suspended on an external event, approval, timer or children
    COMPENSATING = "COMPENSATING"  # saga rollback in progress
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"  # failed, and compensations completed
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.COMPENSATED,
            RunStatus.CANCELLED,
        }


class Budget(BaseModel):
    """Hard limits that make an agent fail closed.

    ``deadline`` is absolute so it survives restarts (a relative timeout would
    silently reset on every resume).
    """

    max_steps: int = 100
    max_tokens: int = 1_000_000
    max_cost_usd: float = 10.0
    deadline: datetime | None = None

    steps_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0

    def charge(self, *, tokens: int = 0, cost_usd: float = 0.0, steps: int = 0) -> None:
        self.tokens_used += tokens
        self.cost_usd += cost_usd
        self.steps_used += steps

    def exceeded(self, now: datetime | None = None) -> str | None:
        """Return a human-readable reason if any limit is breached, else None."""
        now = now or utcnow()
        if self.steps_used >= self.max_steps:
            return f"step budget exhausted ({self.steps_used}/{self.max_steps})"
        if self.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.tokens_used}/{self.max_tokens})"
        if self.cost_usd >= self.max_cost_usd:
            return f"cost budget exhausted (${self.cost_usd:.4f}/${self.max_cost_usd:.2f})"
        if self.deadline and now >= self.deadline:
            return f"deadline passed ({self.deadline.isoformat()})"
        return None


class Lease(BaseModel):
    """Exclusive claim on a run by one worker, with expiry for crash recovery."""

    owner: str
    expires_at: datetime

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at


WaitKind = Literal["event", "approval", "timer", "children"]


class WaitCondition(BaseModel):
    kind: WaitKind
    key: str                       # event key the resume must match
    then: str                      # step to run once the wait is satisfied
    timeout_at: datetime | None = None
    on_timeout: Literal["fail", "resume"] = "fail"  # resume => continue at `then` with a timeout marker


class ChildSpec(BaseModel):
    workflow: str
    input: dict[str, Any] = Field(default_factory=dict)
    child_key: str                 # stable key so re-spawning after a crash is idempotent
    budget: Budget | None = None


class FanIn(BaseModel):
    expected: int
    completed: int = 0
    then: str
    results: dict[str, Any] = Field(default_factory=dict)  # child_key -> result
    failures: dict[str, str] = Field(default_factory=dict)  # child_key -> error


class StepRecord(BaseModel):
    step: str
    attempt: int
    kind: Literal["step", "compensate"] = "step"
    status: Literal["ok", "error", "suspended", "skipped", "fanout"]
    started_at: datetime
    finished_at: datetime
    worker: str
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    output_digest: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000


class Run(BaseModel):
    run_id: str = Field(default_factory=new_id)
    workflow: str
    workflow_version: str = "1"
    status: RunStatus = RunStatus.PENDING

    current_step: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None

    # Completed forward steps in completion order (drives saga compensation).
    completed_steps: list[str] = Field(default_factory=list)
    compensated_steps: list[str] = Field(default_factory=list)
    history: list[StepRecord] = Field(default_factory=list)

    # Per-step attempt counters. A task is only valid if its attempt matches.
    attempts: dict[str, int] = Field(default_factory=dict)

    budget: Budget = Field(default_factory=Budget)
    wait: WaitCondition | None = None
    fan_in: FanIn | None = None
    lease: Lease | None = None

    parent_run_id: str | None = None
    child_key: str | None = None
    cancel_requested: bool = False

    version: int = 0  # optimistic concurrency token, bumped on every write
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # --- helpers -----------------------------------------------------------
    def attempt_of(self, step: str) -> int:
        return self.attempts.get(step, 0)

    def bump_attempt(self, step: str) -> int:
        self.attempts[step] = self.attempt_of(step) + 1
        return self.attempts[step]

    def to_doc(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "Run":
        return cls.model_validate(doc)


class StepTask(BaseModel):
    """The unit of work delivered by the queue (Cloud Tasks / Pub/Sub / in-memory).

    ``dedup_key`` is the Cloud Tasks task *name*; the queue rejects duplicates
    for the same name, giving cheap at-most-once *enqueue* on top of the
    at-least-once *delivery* we still have to tolerate in the worker.
    """

    run_id: str
    step: str
    attempt: int
    kind: Literal["step", "compensate"] = "step"
    not_before: datetime | None = None

    @property
    def dedup_key(self) -> str:
        return f"{self.run_id}--{self.kind}--{self.step}--{self.attempt}"


class Event(BaseModel):
    """An external signal that can wake a WAITING run."""

    run_id: str
    key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    occurred_at: datetime = Field(default_factory=utcnow)


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    text: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = "fake"

    def json(self) -> Any:  # type: ignore[override]
        """Parse the response as JSON, tolerating markdown fences."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)


def backoff_delay(attempt: int, base_s: float = 2.0, cap_s: float = 300.0) -> timedelta:
    """Exponential backoff with full jitter (AWS-style), deterministic cap."""
    import random

    raw = min(cap_s, base_s * (2 ** (attempt - 1)))
    return timedelta(seconds=random.uniform(0, raw))
