"""One structured event per execution: who ran code, under what policy, what it used, how it ended.

The one idea: every execution must leave a record the way every tool call does. This mirrors the identity
lab's ``AuditEvent`` field names (``06-gateway/identity-security/.../agentsec/audit/log.py``) — a standalone
copy, not an import, so the core stays dependency-free — and adds the fields a sandbox needs:
``budgets_used``, ``exit_reason`` and ``policy_decision``. The argument/result hashing uses the identity
lab's ``args_digest`` recipe (sha256 of canonical JSON, first 16 hex). Emit the event from the executor
layer so it exists even when the code crashes or is killed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def args_digest(args: Any) -> str:
    """sha256 of canonical JSON, first 16 hex — identical to the identity lab's recipe."""
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class AuditEvent:
    # ---- fields shared with the identity lab's AuditEvent (same names) ----
    event_type: str                     # sandbox.decision | sandbox.result | egress | confirmation
    agent: str                          # the principal that asked to run code
    authority: str = "own"              # own | delegated (identity primer §3.2)
    user: str | None = None
    tool: str | None = "run_code"
    decision: str | None = None         # allow | deny | confirm
    reasons: list[str] = field(default_factory=list)
    args_hash: str | None = None        # hash of the code + inputs (never the code itself)
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
    # ---- fields the sandbox adds ----
    budgets_used: dict[str, Any] = field(default_factory=dict)   # cpu_s, wall_s, max_rss_mb, disk, output
    exit_reason: str | None = None                               # contract.EXIT_REASONS
    policy_decision: str | None = None                           # the policy Effect that let it run
    isolation: dict[str, Any] = field(default_factory=dict)      # what boundary actually ran it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """An in-memory sink (tests/notebooks). A real deployment writes JSON lines to a log pipeline."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        return event

    def json_lines(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self.events)

    def timeline(self) -> str:
        """A readable one-line-per-event dump, like the identity lab's ``AuditLog.timeline``."""
        rows = []
        for e in self.events:
            rows.append(f"{e.ts[11:19]} {e.event_type:16} {(e.decision or '-'):7} "
                        f"exit={e.exit_reason or '-':13} agent={e.agent:18} "
                        f"{'; '.join(e.reasons)[:60]}")
        return "\n".join(rows)

    def counts(self) -> dict[str, int]:
        """Exit-reason histogram, for the abuse-detection view (kill reasons over time)."""
        out: dict[str, int] = {}
        for e in self.events:
            key = e.exit_reason or e.decision or "?"
            out[key] = out.get(key, 0) + 1
        return out
