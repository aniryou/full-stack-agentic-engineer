"""audit.py — one structured event per memory read, write and forget, and never the memory itself.

The one idea (PRIMER §8; identity primer §9): the gateway side of memory is an audit trail keyed by the
verified principal. This mirrors the identity lab's ``AuditEvent`` field names
(``06-gateway/identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py``) — a standalone
copy, as ``sandboxcore.audit`` is — with three new ``event_type`` values: ``memory.write``,
``memory.read`` and ``memory.forget``. Arguments and results are *hashed* (``args_digest``: sha256 of
canonical JSON, first 16 hex), so the log never becomes one more copy a forget has to chase.

``extra`` carries the OpenTelemetry GenAI memory names from ``open-telemetry/semantic-conventions-genai``
(``gen_ai.operation.name`` = ``create_memory`` / ``search_memory`` / ``delete_memory``,
``gen_ai.memory.store.id``, ``gen_ai.memory.record.count``) — status *development*, unreleased (verify);
the opt-in ``gen_ai.memory.query.text`` / ``gen_ai.memory.records`` are sensitive and deliberately omitted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVENT_TYPES = ("memory.write", "memory.read", "memory.forget")
GENAI_OPERATION = {"memory.write": "create_memory", "memory.read": "search_memory", "memory.forget": "delete_memory"}


def args_digest(args: Any) -> str:
    """sha256 of canonical JSON, first 16 hex — identical to the identity lab's recipe."""
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class AuditEvent:
    # ---- fields shared with the identity lab's AuditEvent (same names) ----
    event_type: str                     # memory.write | memory.read | memory.forget
    agent: str                          # the workload that acted (the token's actor, or the service)
    authority: str = "delegated"        # own | delegated (identity primer §3.5): whose rights were used
    user: str | None = None             # the verified subject whose memory it is
    tool: str | None = "memory"
    decision: str | None = None         # allow | deny | quarantine | replay
    reasons: list[str] = field(default_factory=list)
    args_hash: str | None = None
    result_hash: str | None = None
    approver: str | None = None
    provenance: list[str] = field(default_factory=list)
    trace_id: str | None = None
    invocation_id: str | None = None
    session_id: str | None = None
    latency_ms: float | None = None
    ts: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra: dict[str, Any] = field(default_factory=dict)
    # ---- the field memory adds ----
    tenant: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        self.extra.setdefault("gen_ai.operation.name", GENAI_OPERATION[self.event_type])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """In memory, and optionally appended as JSON lines to ``path`` (what a log pipeline would ingest)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            self.events.append(event)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")
        return event

    def json_lines(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self.events)

    def timeline(self, width: int = 60) -> str:
        return "\n".join(f"{e.ts[11:19]} {e.event_type:14} {(e.decision or '-'):10} tenant={e.tenant} "
                         f"user={e.user} agent={e.agent} {'; '.join(e.reasons)[:width]}" for e in self.events)


def read_json_lines(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
