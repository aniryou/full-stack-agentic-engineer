"""audit.py — one structured event per decision and per execution, and the alerts they add up to.

One idea: every execution leaves a record that answers who asked, under which policy, what ran
(by hash), with which budgets, how much it used and *why it stopped* — and the stop reasons are
the abuse signal (PRIMER §8 "Observability, audit and abuse detection"). The event shape is the
identity lab's ``AuditEvent`` (06-gateway ``agentsec/audit/log.py``, identity primer §9: trace and
invocation ids, agent, authority, tool, argument hash, decision and reasons, result hash,
latency) — copied field for field, not imported — plus the sandbox's own fields:
``isolation``, ``budgets_used``, ``exit_reason``, ``policy_decision``, ``idempotency_key``.

``detect()`` turns a stream of events into alerts: fork bombs and escaped processes, repeated
CPU-limit kills (a miner or a runaway loop), output floods, egress attempts the proxy refused,
and credentials reflected back through the proxy.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


def args_digest(args: Any) -> str:
    """The identity lab's recipe: sha256 of canonical JSON, first 16 hex characters."""
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class AuditEvent:
    event_type: str      # tool.decision | tool.result | sandbox.execution | egress
    agent: str
    authority: str       # "own" | "delegated"
    user: str | None = None
    tool: str | None = None
    decision: str | None = None           # allow | deny | ok | blocked
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
    ts: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra: dict[str, Any] = field(default_factory=dict)
    # -- sandbox additions
    isolation: str | None = None
    budgets_used: dict | None = None
    exit_reason: str | None = None
    policy_decision: str | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    """In-memory sink with JSON-lines export (a Cloud Logging or file sink is one method away)."""

    def __init__(self):
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        return event

    def from_proxy(self, proxy_events: list[dict], session_id: str | None = None) -> None:
        """Fold the egress proxy's JSON lines (same field names) into this log."""
        for e in proxy_events:
            known = {k: e[k] for k in ("event_type", "agent", "authority", "tool", "decision", "reasons", "latency_ms",
                                        "ts", "id", "extra") if k in e}
            self.emit(AuditEvent(**known, session_id=session_id))

    def jsonl(self) -> str:
        return "".join(json.dumps(e.to_dict(), default=str) + "\n" for e in self.events)

    def of_type(self, t: str) -> list[AuditEvent]:
        return [e for e in self.events if e.event_type == t]


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(events: list[AuditEvent]) -> dict:
    """The dashboard: executions by exit reason, p50/p95 run and start-up time, denials by reason."""
    ex = [e for e in events if e.event_type == "sandbox.execution"]
    run = [e.budgets_used.get("wall_s") for e in ex if e.budgets_used and e.budgets_used.get("wall_s") is not None]
    start = [e.budgets_used.get("startup_s") for e in ex if e.budgets_used and e.budgets_used.get("startup_s") is not None]
    denials = Counter(r for e in events if e.decision == "deny" for r in e.reasons)
    return {"executions": len(ex), "exit_reasons": dict(Counter(e.exit_reason for e in ex)),
            "run_s_p50": _pct(run, 0.5), "run_s_p95": _pct(run, 0.95),
            "startup_s_p50": _pct(start, 0.5), "startup_s_p95": _pct(start, 0.95),
            "cpu_s_total": round(sum((e.budgets_used or {}).get("cpu_s") or 0 for e in ex), 3),
            "denials": dict(denials)}


@dataclass
class Alert:
    severity: str        # "high" | "medium" | "low"
    rule: str
    session_id: str | None
    evidence: str


def detect(events: list[AuditEvent], *, cpu_kills_threshold: int = 3) -> list[Alert]:
    """Abuse signals from the audit stream. Each rule is one line of reasoning a responder can check."""
    out: list[Alert] = []
    by_session: dict[str | None, list[AuditEvent]] = {}
    for e in events:
        by_session.setdefault(e.session_id, []).append(e)
    for sid, evs in by_session.items():
        ex = [e for e in evs if e.event_type == "sandbox.execution"]
        for e in ex:
            used = e.budgets_used or {}
            if e.exit_reason == "pids" or "fork" in " ".join(e.reasons).lower():
                out.append(Alert("high", "fork-bomb", sid, f"execution {e.invocation_id}: process limit hit"))
            if used.get("stragglers_killed"):
                out.append(Alert("high", "escaped-process", sid,
                                 f"execution {e.invocation_id}: {used['stragglers_killed']} process(es) outlived the run"))
            if e.exit_reason == "output_limit":
                out.append(Alert("low", "output-flood", sid, f"execution {e.invocation_id}: output limit"))
            if e.exit_reason == "memory":
                out.append(Alert("low", "memory-limit", sid, f"execution {e.invocation_id}: memory limit"))
        cpu_kills = sum(e.exit_reason in ("cpu_time", "wall_timeout") for e in ex)
        if cpu_kills >= cpu_kills_threshold:
            out.append(Alert("medium", "repeated-cpu-kills", sid,
                             f"{cpu_kills} executions hit the CPU/wall limit (mining or a runaway loop)"))
        for e in evs:
            if e.event_type == "egress" and e.decision == "deny":
                target = e.extra.get("host") or e.extra.get("route") or e.extra.get("target")
                out.append(Alert("high", "egress-denied", sid, f"egress to {target!r} refused: {'; '.join(e.reasons)}"))
            if e.event_type == "egress" and any("redacted" in r for r in e.reasons):
                out.append(Alert("high", "credential-reflection", sid,
                                 "an upstream response contained the injected credential (redacted by the proxy)"))
            if e.event_type == "tool.decision" and e.decision == "deny":
                out.append(Alert("medium", "tool-denied", sid, f"{e.tool}: {'; '.join(e.reasons)}"))
    return out


def execution_event(result, *, agent: str, session_id: str | None, invocation_id: str | None, code: str,
                    idempotency_key: str | None = None, policy_decision: str = "allow",
                    authority: str = "own", user: str | None = None) -> AuditEvent:
    """The sandbox.execution record for one ``ExecResult``: the code by hash, the output by hash."""
    used = {"wall_s": result.wall_s, "cpu_s": result.cpu_s, "max_rss_kib": result.max_rss_kib,
            "stdout_bytes": result.stdout_bytes, "stderr_bytes": result.stderr_bytes,
            "startup_s": round(result.startup_s, 4), "stragglers_killed": result.stragglers_killed}
    return AuditEvent("sandbox.execution", agent, authority, user=user, tool="run_code", decision="ok",
                      reasons=list(result.notes)[:3], args_hash=args_digest({"code": code}),
                      result_hash=hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest()[:16],
                      session_id=session_id, invocation_id=invocation_id, latency_ms=round(result.total_s * 1000, 1),
                      isolation=result.isolation, budgets_used=used, exit_reason=result.exit_reason,
                      policy_decision=policy_decision, idempotency_key=idempotency_key,
                      extra={"simulated": result.simulated})


def run_times(events: list[AuditEvent]) -> list[float]:
    return [e.budgets_used["wall_s"] for e in events if e.event_type == "sandbox.execution" and e.budgets_used]


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None
