"""Structured audit events with dual identity (user + agent) for every consequential step.

One event per decision — model call screened, tool call evaluated/allowed/denied/confirmed,
credential retrieved, egress attempted — carrying: trace and invocation IDs, the agent's SPIFFE
ID, the user (when delegated), the authority mode, the tool and a hash (or redacted copy) of its
arguments, the policy decision and reasons, the approver if a human confirmed, provenance tags
of the inputs, and timing.

Sinks: in-memory (tests/notebooks), JSON lines to a file/stdout, and Cloud Logging
(``CloudLoggingSink``) which lands as ``jsonPayload`` so you can route it with a log sink to
BigQuery and join it with Cloud Audit Logs on the agent principal.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import IO, Any

from ..secrets.redaction import redact


def args_digest(args: Any) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class AuditEvent:
    event_type: str  # tool.decision | tool.result | model.screen | credential.retrieve | egress | confirmation
    agent: str
    authority: str
    user: str | None = None
    tool: str | None = None
    decision: str | None = None  # allow | deny | confirm | blocked | ok
    reasons: list[str] = field(default_factory=list)
    args_hash: str | None = None
    args_redacted: Any = None
    result_hash: str | None = None
    approver: str | None = None
    provenance: list[str] = field(default_factory=list)
    trace_id: str | None = None
    invocation_id: str | None = None
    session_id: str | None = None
    latency_ms: float | None = None
    ts: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["args_redacted"] = redact(d["args_redacted"])
        d["extra"] = redact(d["extra"])
        return d


class AuditSink:
    def write(self, event: AuditEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class MemorySink(AuditSink):
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class JsonLinesSink(AuditSink):
    def __init__(self, stream: IO[str] | None = None):
        self.stream = stream or sys.stdout

    def write(self, event: AuditEvent) -> None:
        self.stream.write(json.dumps(event.to_dict(), default=str) + "\n")
        self.stream.flush()


class CloudLoggingSink(AuditSink):
    """Writes events to Cloud Logging as structured ``jsonPayload`` (lazy client)."""

    def __init__(self, log_name: str = "agentsec-audit", client: Any | None = None):
        self.log_name = log_name
        self._client = client

    def write(self, event: AuditEvent) -> None:
        if self._client is None:
            from google.cloud import logging as cloud_logging  # lazy

            self._client = cloud_logging.Client()
        logger = self._client.logger(self.log_name)
        severity = "WARNING" if event.decision in {"deny", "blocked"} else "INFO"
        logger.log_struct(event.to_dict(), severity=severity, labels={"agent": event.agent[:63]})


class AuditLog:
    """Fan-out to sinks, with convenience query helpers for notebooks."""

    def __init__(self, sinks: Iterable[AuditSink] | None = None):
        self.sinks: list[AuditSink] = list(sinks) if sinks else [MemorySink()]

    @property
    def memory(self) -> MemorySink | None:
        return next((s for s in self.sinks if isinstance(s, MemorySink)), None)

    def emit(self, event: AuditEvent) -> AuditEvent:
        for sink in self.sinks:
            sink.write(event)
        return event

    def record(self, **fields: Any) -> AuditEvent:
        return self.emit(AuditEvent(**fields))

    # ---- query helpers (memory sink) ---------------------------------------------------
    def events(self, predicate: Callable[[AuditEvent], bool] | None = None) -> list[AuditEvent]:
        mem = self.memory
        if mem is None:
            return []
        return [e for e in mem.events if predicate is None or predicate(e)]

    def by_agent(self, spiffe_id: str) -> list[AuditEvent]:
        return self.events(lambda e: e.agent == spiffe_id)

    def denials(self) -> list[AuditEvent]:
        return self.events(lambda e: e.decision in {"deny", "blocked"})

    def timeline(self) -> list[str]:
        rows = []
        for e in self.events():
            row = f"{e.ts[11:19]} {e.event_type:<14} {e.decision or '-':<8} {e.tool or '-':<22} user={e.user or '-':<22} agent={e.agent.rsplit('/', 1)[-1]}"
            if e.approver:
                row += f" approver={e.approver}"
            if e.reasons and e.decision in {"deny", "blocked", "confirm"}:
                row += f"  ({e.reasons[0]})"
            rows.append(row)
        return rows
