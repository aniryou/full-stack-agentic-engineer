"""consolidate.py - turn a window of episodes into facts, on a schedule, as a run that survives a crash.

The one idea: consolidation is a batch job with rules, not a summary. Per (user, window) it reads the raw
episodes, groups the statements by slot and applies three rules: the highest-precedence source wins
(human > user > tool > inferred); within that source, newer supersedes older - and the older fact is KEPT,
closed with valid_to (bi-temporal, as Graphiti closes an edge with invalid_at instead of deleting it); a
weaker source that contradicts is flagged for review, never applied. Quarantined (tool) episodes are not
read at all, so nothing a tool said is promoted without review.

It runs like any durable job (lra-gcp primer §3.3, §3.13): a deterministic run id per (user, window), so a
double fire is one run; a lease with a TTL, so two workers never run it together and a dead worker's run is
picked up after the lease expires; a checkpoint after every slot, so a resumed run skips finished work; and
fact ids derived from (run id, slot, position), so re-applying a slot after a crash overwrites instead of
duplicating. `reflect` is the generative-agents idea in brief: when the importance of new episodes sums past a
trigger (150 in the reference code), write insights that cite their evidence, with explicit exits.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from .budget import BudgetExceeded
from .records import DAY, SOURCE_TRUST, MemoryRecord, Scope
from .write import SLOTS, fact_text, read_facts


class LeaseHeld(RuntimeError):
    pass


class Crash(RuntimeError):
    """An injected crash (tests, notebooks): the process dies holding the lease."""


def plan_key(statements, existing: MemoryRecord | None = None):
    """statements: [(time, value, source, evidence_id)] for ONE slot. Returns (facts, flags):
    facts = [value, valid_from, valid_to, source, evidence ids, existing id or None] in time order."""
    best = max(SOURCE_TRUST[s] for _, _, s, _ in statements)
    if existing is not None and existing.trust > best:
        return [], [(st, f"weaker than the {existing.source} fact") for st in statements
                    if st[1].lower() != existing.value.lower()]
    facts = [[existing.value, existing.valid_from, None, existing.source, existing.provenance, existing.id]] \
        if existing is not None else []
    for t, v, s, ev in sorted(st for st in statements if SOURCE_TRUST[st[2]] == best):
        if facts and facts[-1][0].lower() == v.lower():
            facts[-1][4] = tuple(facts[-1][4]) + (ev,)            # same value again: more evidence, no new fact
            continue
        if facts:
            facts[-1][2] = t                                      # newer supersedes older: close it, keep it
        facts.append([v, t, None, s, (ev,), None])
    def value_at(t):
        return next((f[0] for f in reversed(facts) if f[1] <= t), None)
    flags = [(st, "a weaker source contradicts") for st in statements
             if SOURCE_TRUST[st[2]] < best and (value_at(st[0]) or "").lower() != st[1].lower()]
    return facts, flags


@dataclass
class ConsolidationReport:
    run_id: str
    written: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    flagged: list[tuple] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    status: str = "running"


class Consolidator:
    def __init__(self, store, lease_ttl: float = 60.0):
        self.store, self.lease_ttl = store, lease_ttl
        self.leases: dict[str, tuple[str, float]] = {}            # run id -> (worker, expires at)
        self.checkpoints: dict[str, dict] = {}                     # run id -> {"done": [...], "status": ...}

    @staticmethod
    def run_id(scope: Scope, start: float, end: float) -> str:
        return f"consolidate:{scope.tenant}:{scope.user}:day{start / DAY:g}-{end / DAY:g}"

    def acquire(self, rid: str, worker: str, now: float) -> None:
        holder = self.leases.get(rid)
        if holder and holder[0] != worker and holder[1] > now:
            raise LeaseHeld(f"{rid} is leased by {holder[0]} until t={holder[1]:g}")
        self.leases[rid] = (worker, now + self.lease_ttl)

    def run(self, scope: Scope, start: float, end: float, *, now: float, worker: str = "w1",
            crash_after: int | None = None) -> ConsolidationReport:
        rid = self.run_id(scope, start, end)
        ck = self.checkpoints.setdefault(rid, {"done": [], "status": "running"})
        report = ConsolidationReport(rid)
        if ck["status"] == "done":
            report.status, report.skipped = "already done", list(ck["done"])
            return report
        self.acquire(rid, worker, now)
        by_key = defaultdict(list)
        for ep in self.store.records(scope, kind="episodic"):     # active only: quarantined episodes never count
            if start <= ep.created_at < end:
                for key, value in read_facts(ep.text):
                    by_key[key].append((ep.created_at, value, ep.source, ep.id))
        processed = 0
        for key in sorted(by_key):
            if key in ck["done"]:
                report.skipped.append(key)
                continue
            self._apply(scope, rid, key, by_key[key], now, report)
            ck["done"].append(key)                                  # checkpoint after every slot
            processed += 1
            if crash_after is not None and processed >= crash_after:
                raise Crash(f"crashed after {processed} slot(s) of {rid}")
        ck["status"] = report.status = "done"
        self.leases.pop(rid, None)
        return report

    def _apply(self, scope, rid, key, statements, now, report) -> None:
        current = [r for r in self.store.find(scope, key) if r.meta.get("run_id") != rid]   # not a half-applied self
        facts, flags = plan_key(statements, max(current, key=lambda r: r.valid_from) if current else None)
        report.flagged += [(key, st[1], st[2], why) for st, why in flags]
        for i, (value, vfrom, vto, source, evidence, existing_id) in enumerate(facts):
            if existing_id is not None:
                old = self.store.get(scope, existing_id)
                old.provenance = tuple(dict.fromkeys(old.provenance + tuple(evidence)))
                if vto is not None and old.status == "active":
                    old.valid_to, old.superseded_at, old.status = vto, now, "superseded"
                    report.superseded.append(old.id)
                self.store.update(old)
                continue
            fid = hashlib.sha256(f"{rid}:{key}:{i}".encode()).hexdigest()[:12]
            self.store.put(MemoryRecord(
                fact_text(key, value), "procedural" if key == "procedure" else "semantic", scope, source,
                key=key, value=value, provenance=tuple(evidence), importance=SLOTS[key][1], created_at=now,
                valid_from=vfrom, valid_to=vto, superseded_at=now if vto is not None else None,
                status="superseded" if vto is not None else "active", id=fid, meta={"run_id": rid}))
            report.written.append(fid)


def reflect(store, scope: Scope, *, now: float, since: float = 0.0, threshold: float = 150.0,
            max_insights: int = 3, budget=None):
    """If new episodes' importance sums past `threshold`, write insights citing evidence.
    Returns (insights, exit): below_trigger | done | max_insights | budget_exhausted (lra-gcp primer §3.8's exits)."""
    recent = [e for e in store.records(scope, kind="episodic") if e.created_at >= since]
    if sum(e.importance for e in recent) < threshold:
        return [], "below_trigger"
    mentions = defaultdict(list)
    for e in recent:
        for key, _ in read_facts(e.text):
            mentions[key].append(e)
    insights = []
    for key, eps in sorted(mentions.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(eps) < 2:
            break
        if len(insights) == max_insights:
            return insights, "max_insights"
        if budget is not None:
            try:
                budget.charge("llm_calls")
            except BudgetExceeded:
                return insights, "budget_exhausted"
        text = f"The user keeps returning to their {SLOTS[key][0]} ({len(eps)} mentions)."
        insights.append(store.put(MemoryRecord(text, "semantic", scope, "inferred", created_at=now,
                                               importance=max(e.importance for e in eps),
                                               provenance=tuple(e.id for e in eps), meta={"insight": True})))
    return insights, "done"
