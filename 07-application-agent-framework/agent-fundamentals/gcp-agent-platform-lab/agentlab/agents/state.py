"""Sessions, events, working state, and durable task records (notebook 03).

The invariants behind them — the store is the only memory, effectively-once
side effects, leases — are the long-running-durable primer's §3
(07-application-agent-framework/long-running-durable/00_primer.md).

The event log is the source of truth; the model's view of the conversation is
*derived* from it. Working state is a small typed dict kept alongside the log,
not inside the transcript. Stores use optimistic concurrency (a version number)
so two turns racing on one session cannot silently overwrite each other.
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from ..llm.types import Message, Usage

# ------------------------------------------------------------------- events
EVENT_KINDS = {
    "user",              # a user message
    "model",             # an assistant turn (text and/or tool calls)
    "tool_call",         # a tool call about to run (payload: name, args, id)
    "tool_result",       # a tool result (payload: name, id, ok, content)
    "final",             # marker: the turn produced its final answer
    "approval_required", # the loop paused for a human decision
    "approval",          # the human decision
    "delegation",        # a sub-agent was invoked as a tool
    "state",             # a working-state change (payload: key, value)
    "error",             # a failure the runtime recorded
    "note",              # free-form annotations (compaction summaries, etc.)
}


@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    agent: str | None = None
    step: int | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: "ev_" + uuid.uuid4().hex[:10])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        if d.get("usage"):
            d["usage"] = Usage(**d["usage"])
        return cls(**d)


class SessionStatus(str, Enum):
    ACTIVE = "active"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


# ------------------------------------------------------------------ session
@dataclass
class Session:
    id: str
    tenant: str = "default"
    user: str = "anonymous"
    events: list[Event] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)   # scoped keys: "app:", "user:", "temp:" or plain
    status: SessionStatus = SessionStatus.ACTIVE
    pending: dict[str, Any] | None = None                  # what the session is waiting on (approval payload)
    version: int = 0                                       # optimistic concurrency token
    created_at: float = field(default_factory=time.time)

    # -- log ------------------------------------------------------------
    def append(self, event: Event) -> Event:
        if event.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {event.kind!r}")
        self.events.append(event)
        return event

    def set_state(self, key: str, value: Any, agent: str | None = None) -> None:
        self.state[key] = value
        self.events.append(Event(kind="state", payload={"key": key, "value": value}, agent=agent))

    def clear_temp_state(self) -> None:
        for k in [k for k in self.state if k.startswith("temp:")]:
            del self.state[k]

    # -- derived views --------------------------------------------------
    def messages(self, since_event: int = 0) -> list[Message]:
        """Derive the model-facing conversation from the event log.

        Only user, model and tool_result events become messages. Everything else
        (state, approvals, notes) stays out of the model's context unless a
        context builder chooses to inject it.
        """
        out: list[Message] = []
        for ev in self.events[since_event:]:
            if ev.kind == "user":
                out.append({"role": "user", "content": ev.payload.get("content", "")})
            elif ev.kind == "model":
                m: Message = {"role": "assistant", "content": ev.payload.get("content", "") or ""}
                if ev.payload.get("tool_calls"):
                    m["tool_calls"] = ev.payload["tool_calls"]
                out.append(m)
            elif ev.kind == "tool_result":
                out.append({
                    "role": "tool",
                    "name": ev.payload.get("name"),
                    "tool_call_id": ev.payload.get("id"),
                    "content": ev.payload.get("content", ""),
                })
        return out

    def last_final_text(self) -> str | None:
        for ev in reversed(self.events):
            if ev.kind == "final":
                return ev.payload.get("text")
        return None

    def usage(self) -> Usage:
        total = Usage()
        for ev in self.events:
            if ev.usage:
                total = total + ev.usage
        return total

    def turn_count(self) -> int:
        return sum(1 for ev in self.events if ev.kind == "user")

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "tenant": self.tenant, "user": self.user,
            "events": [e.to_dict() for e in self.events],
            "state": self.state, "status": self.status.value, "pending": self.pending,
            "version": self.version, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        return cls(
            id=d["id"], tenant=d.get("tenant", "default"), user=d.get("user", "anonymous"),
            events=[Event.from_dict(e) for e in d.get("events", [])],
            state=dict(d.get("state", {})), status=SessionStatus(d.get("status", "active")),
            pending=d.get("pending"), version=int(d.get("version", 0)), created_at=d.get("created_at", time.time()),
        )

    def branch(self, suffix: str) -> "Session":
        """A deep copy for parallel branches; merge results back via state."""
        s = copy.deepcopy(self)
        s.id = f"{self.id}#{suffix}"
        return s


# ------------------------------------------------------------------- stores
class VersionConflict(RuntimeError):
    def __init__(self, session_id: str, expected: int, actual: int):
        super().__init__(f"session {session_id}: expected version {expected}, store has {actual}")
        self.session_id, self.expected, self.actual = session_id, expected, actual


class SessionNotFound(KeyError):
    pass


class SessionStore(Protocol):
    def create(self, session: Session) -> Session: ...
    def get(self, session_id: str) -> Session: ...
    def put(self, session: Session) -> Session: ...
    def get_or_create(self, session_id: str, **kwargs: Any) -> Session: ...


class InMemorySessionStore:
    """Stores serialised copies (no aliasing) and enforces compare-and-set on ``version``."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def create(self, session: Session) -> Session:
        if session.id in self._data:
            raise ValueError(f"session {session.id} exists")
        session.version = 1
        self._data[session.id] = session.to_dict()
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self._data:
            raise SessionNotFound(session_id)
        return Session.from_dict(copy.deepcopy(self._data[session_id]))

    def put(self, session: Session) -> Session:
        """Compare-and-set: the caller's ``session.version`` must match the stored one."""
        stored = self._data.get(session.id)
        if stored is None:
            raise SessionNotFound(session.id)
        if stored["version"] != session.version:
            raise VersionConflict(session.id, session.version, stored["version"])
        session.version += 1
        self._data[session.id] = copy.deepcopy(session.to_dict())
        return session

    def get_or_create(self, session_id: str, **kwargs: Any) -> Session:
        try:
            return self.get(session_id)
        except SessionNotFound:
            return self.create(Session(id=session_id, **kwargs))

    def ids(self) -> list[str]:
        return list(self._data)


class JsonFileSessionStore(InMemorySessionStore):
    """Same semantics, one JSON file per session — survives a process restart."""

    def __init__(self, root: str | Path):
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for p in self.root.glob("*.json"):
            self._data[p.stem] = json.loads(p.read_text())

    def _flush(self, session_id: str) -> None:
        (self.root / f"{session_id}.json").write_text(json.dumps(self._data[session_id], default=str, indent=1))

    def create(self, session: Session) -> Session:
        s = super().create(session)
        self._flush(s.id)
        return s

    def put(self, session: Session) -> Session:
        s = super().put(session)
        self._flush(s.id)
        return s


# ------------------------------------------------------------ durable tasks
class TaskStatus(str, Enum):
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


@dataclass
class TaskRecord:
    """A long-running unit of work with a checkpoint (mirrors the MCP Tasks shape)."""
    id: str = field(default_factory=lambda: "task_" + uuid.uuid4().hex[:10])
    kind: str = "generic"
    status: TaskStatus = TaskStatus.WORKING
    stage: str = "start"                                 # where in the state machine we are
    checkpoint: dict[str, Any] = field(default_factory=dict)  # everything needed to resume
    completed_steps: list[str] = field(default_factory=list)  # idempotency: steps already applied
    input_requests: dict[str, Any] = field(default_factory=dict)
    input_responses: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: dict[str, Any] | None = None
    status_message: str = ""
    ttl_ms: int = 24 * 3600 * 1000
    poll_interval_ms: int = 500
    version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def mark(self, status: TaskStatus, message: str = "") -> None:
        self.status = status
        self.status_message = message
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class TaskStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def create(self, task: TaskRecord) -> TaskRecord:
        task.version = 1
        self._data[task.id] = copy.deepcopy(task.to_dict())
        return task

    def get(self, task_id: str) -> TaskRecord:
        d = copy.deepcopy(self._data[task_id])
        d["status"] = TaskStatus(d["status"])
        return TaskRecord(**d)

    def put(self, task: TaskRecord) -> TaskRecord:
        stored = self._data[task.id]
        if stored["version"] != task.version:
            raise VersionConflict(task.id, task.version, stored["version"])
        task.version += 1
        task.updated_at = time.time()
        self._data[task.id] = copy.deepcopy(task.to_dict())
        return task

    def ids(self) -> list[str]:
        return list(self._data)
