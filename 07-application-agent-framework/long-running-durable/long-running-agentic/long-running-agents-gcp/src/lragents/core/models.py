"""Core data model for long-running agent runs.

Everything a long-running agent needs to survive a process death lives in a
``Run``:

* ``journal``  – an append-only log of what happened (LLM decisions, tool calls,
  human decisions, external events). This is the *event-sourced* part: replaying
  the journal reconstructs the prompt for the next LLM call.
* ``state``    – a small working-state dictionary (the *checkpoint* part).
* ``version``  – an optimistic-concurrency token. Every save must present the
  version it read; a mismatch means someone else wrote first.
* ``lease``    – who is allowed to advance the run right now, and until when.
* ``budget`` / ``usage`` – hard stops so a loop can never run away.

The model is deliberately plain dataclasses + JSON-serialisable dicts so it
maps 1:1 onto a Firestore document (see ``firestore_store.py``).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"   # parked on an approval gate
    WAITING_EVENT = "WAITING_EVENT"   # parked on an external system / async tool
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)


class StepKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    HUMAN = "human"
    EVENT = "event"
    SYSTEM = "system"


class StepStatus(str, Enum):
    STARTED = "started"   # intent recorded, side effect may or may not have happened
    DONE = "done"
    FAILED = "failed"


@dataclass
class StepRecord:
    """One journal entry. Immutable once DONE/FAILED."""

    index: int
    kind: StepKind
    status: StepStatus
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    idempotency_key: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    def finish(self, output: Any = None, error: str | None = None) -> "StepRecord":
        self.output = output
        self.error = error
        self.status = StepStatus.FAILED if error else StepStatus.DONE
        self.finished_at = time.time()
        return self


@dataclass
class Budget:
    """Hard limits. A run that crosses any of them FAILS with a clear reason."""

    max_steps: int = 25
    max_tokens: int = 200_000
    max_cost_usd: float = 2.0
    deadline_epoch: float | None = None   # absolute wall-clock deadline


@dataclass
class Usage:
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Lease:
    owner: str
    expires_at: float

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


@dataclass
class Run:
    run_id: str
    goal: str
    status: RunStatus = RunStatus.PENDING
    state: dict[str, Any] = field(default_factory=dict)
    journal: list[StepRecord] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    usage: Usage = field(default_factory=Usage)
    lease: Lease | None = None
    waiting_on: dict[str, Any] | None = None   # e.g. {"type": "approval", "token": "..."}
    result: Any = None
    error: str | None = None
    version: int = 0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- journal helpers -------------------------------------------------
    def next_index(self) -> int:
        return len(self.journal)

    def append(self, rec: StepRecord) -> StepRecord:
        assert rec.index == self.next_index(), "journal indices must be contiguous"
        self.journal.append(rec)
        return rec

    def pending_step(self) -> StepRecord | None:
        """The last STARTED (unfinished) step, if any — the crash-recovery hook."""
        if self.journal and self.journal[-1].status == StepStatus.STARTED:
            return self.journal[-1]
        return None

    def tool_steps(self) -> list[StepRecord]:
        return [s for s in self.journal if s.kind == StepKind.TOOL]

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        for s in d["journal"]:
            s["kind"] = StepKind(s["kind"]).value
            s["status"] = StepStatus(s["status"]).value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Run":
        d = dict(d)
        d["status"] = RunStatus(d["status"])
        d["journal"] = [
            StepRecord(**{**s, "kind": StepKind(s["kind"]), "status": StepStatus(s["status"])})
            for s in d.get("journal", [])
        ]
        d["budget"] = Budget(**d.get("budget", {}))
        d["usage"] = Usage(**d.get("usage", {}))
        d["lease"] = Lease(**d["lease"]) if d.get("lease") else None
        return cls(**d)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
