"""memory.py — the write path and the read path around a store: policy, resolution, audit, scope.

The one idea (PRIMER §2): what reaches the store is decided in code, not by the model. A write passes
a ``WritePolicy`` (which kinds each source may write, a confidence floor, screening before persistence —
secrets are rejected, injection-shaped text and every tool output are quarantined), then is *resolved*
against what the partition already holds: ADD a new fact, NOOP a repeat (and refresh it), UPDATE by
superseding an older value of a single-valued slot — kept, closed at the new fact's ``valid_from`` —
or FLAG a contradiction from a weaker source (precedence human > user > tool > inferred, PRIMER §7).
mem0 v1.x did extract → compare → ADD/UPDATE/DELETE/NONE with an LLM call; mem0 2.x is ADD-only
(facts sheet §5). Here resolution is deterministic, so every run is reproducible.

``LocalMemory`` is that pipeline in-process for one ``(tenant, user)``; ``service.MemoryService`` serves
the same pipeline over HTTP with the scope taken from a verified token; ``service.RemoteMemory`` is
its client. The agent (``agent.py``) talks to either through the same four methods.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .audit import AuditEvent, AuditLog, args_digest
from .deletion import DeletionReport
from .extract import SLOTS, day, question_slot, render_memory, screen
from .records import TRUST_ORDER, MemoryRecord, content_hash, count_tokens, default_deletion_key, outranks
from .store.sqlite import SQLiteMemoryStore, pack


@dataclass
class Decision:
    action: str                    # write | quarantine | reject
    reasons: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return {"write": "active", "quarantine": "quarantined"}.get(self.action, "rejected")


ALL_KINDS = frozenset({"episodic", "semantic", "procedural"})


@dataclass
class WritePolicy:
    """Policy as data. Defaults: users and human reviewers write any kind; a tool's output may only land
    as a quarantined record (it replays in later sessions otherwise — ASI06, identity primer §2); the
    model's own inferences only as semantic facts above the floor; secrets never persist."""
    confidence_floor: float = 0.7
    kinds_by_source: dict = field(default_factory=lambda: {
        "human": ALL_KINDS, "user": ALL_KINDS, "tool": frozenset({"episodic", "semantic"}),
        "inferred": frozenset({"semantic"}), "consolidation": frozenset({"semantic", "procedural"})})
    quarantine_sources: frozenset = frozenset({"tool"})

    def decide(self, rec: MemoryRecord) -> Decision:
        if rec.kind not in self.kinds_by_source.get(rec.source, frozenset()):
            return Decision("reject", [f"kind {rec.kind!r} is not writable from source {rec.source!r}"])
        if rec.confidence < self.confidence_floor:
            return Decision("reject", [f"confidence {rec.confidence:.2f} < floor {self.confidence_floor:.2f}"])
        findings = screen(rec.text).findings
        secrets = [f for f in findings if f.startswith("secret:")]
        if secrets:
            return Decision("reject", ["a secret never persists: " + ", ".join(secrets)])
        reasons = [f for f in findings if f.startswith("injection:")]
        if rec.source in self.quarantine_sources:
            reasons.append(f"source {rec.source!r} is quarantined until a human reviews it")
        return Decision("quarantine", reasons) if reasons else Decision("write")


def resolve(existing: list[MemoryRecord], new: MemoryRecord) -> tuple[str, MemoryRecord | None]:
    """Compare a new fact with the partition's *active* facts of the same slot.

    Returns ``(action, other)``: ``("ADD", None)``, ``("NOOP", same)``, ``("UPDATE", superseded)`` or
    ``("FLAG", stronger)``. A multi-valued slot (trips) only ever ADDs or NOOPs. A newer value from a
    source at least as trusted supersedes; a newer value from a *weaker* source is flagged, not applied;
    an *older* value (a late-arriving episode) never supersedes a newer one — it is kept as history.
    """
    same = [r for r in existing if r.slot == new.slot and r.status == "active" and r.slot]
    for r in same:
        if (r.value or "").lower() == (new.value or "").lower():
            return "NOOP", r
    if not same or (new.slot in SLOTS and SLOTS[new.slot].multi):
        return "ADD", None
    current = max(same, key=lambda r: r.valid_from)
    if outranks(current.trust, new.trust):
        return "FLAG", current
    if new.valid_from < current.valid_from:
        return "ADD_HISTORY", current
    return "UPDATE", current


class LocalMemory:
    """One ``(tenant, user)``'s memory, in-process: policy → resolve → store → audit."""

    def __init__(self, store: SQLiteMemoryStore, tenant: str, user: str, *, agent: str = "agent:memlab",
                 policy: WritePolicy | None = None, audit: AuditLog | None = None, clock=None):
        self.store, self.tenant, self.user, self.agent = store, tenant, user, agent
        self.policy = policy or WritePolicy()
        self.audit = audit or AuditLog()
        self.clock = clock or store.clock or time.time
        self.scope = f"{tenant}/{user}"

    # -- write -----------------------------------------------------------------------------------
    def remember(self, text: str, *, kind: str = "semantic", slot: str | None = None, value: str | None = None,
                 source: str = "user", confidence: float = 0.9, importance: float = 5.0, provenance=(),
                 session: str | None = None, scope: str = "user", ttl_s: float | None = None,
                 idempotency_key: str | None = None, deletion_key: str | None = None,
                 valid_from: float | None = None) -> dict:
        now = self.clock()
        trust = source if source in TRUST_ORDER else "inferred"
        rec = MemoryRecord(self.tenant, self.user, text, kind=kind, scope=scope, source=source, trust=trust,
                           session=session, agent=self.agent, slot=slot, value=value, provenance=list(provenance),
                           confidence=confidence, importance=importance, created_at=now,
                           valid_from=valid_from if valid_from is not None else now, ttl_s=ttl_s,
                           deletion_key=deletion_key or default_deletion_key(self.tenant, self.user, slot or
                                                                            content_hash(text)))
        decision = self.policy.decide(rec)
        key = idempotency_key or content_hash(self.tenant, self.user, kind, slot, value, text)
        out = {"ok": decision.action != "reject", "decision": decision.action, "reasons": decision.reasons}
        if decision.action == "reject":
            self._audit("memory.write", "deny", decision.reasons, {"slot": slot, "kind": kind, "text": text},
                        None, session, rec.provenance)
            return out
        rec.status = decision.status
        action, other = ("ADD", None)
        if decision.action == "write" and kind == "semantic" and slot:
            action, other = resolve(self.store.records(self.tenant, self.user, statuses=("active",), slot=slot), rec)
        if action == "NOOP":
            self.store.set_fields(other.id, last_accessed=now, confidence=max(other.confidence, confidence))
            out.update(id=other.id, action="NOOP", replayed=False)
        else:
            if action == "FLAG":
                rec.status = "flagged"
                out["reasons"] = out["reasons"] + [f"contradicts a {other.trust}-sourced fact; flagged, not applied"]
            if action == "ADD_HISTORY":
                rec.valid_to = other.valid_from
                rec.status = "superseded"
                rec.superseded_at = now
            rid, created = self.store.add(rec, idempotency_key=key)
            if action == "UPDATE" and created:
                self.store.supersede(other.id, at=rec.valid_from, now=now)
            out.update(id=rid, action=action if created else "REPLAY", replayed=not created, status=rec.status,
                       superseded=other.id if action == "UPDATE" else None)
        self._audit("memory.write", "replay" if out.get("replayed") else
                    ("quarantine" if rec.status == "quarantined" else "allow"),
                    out["reasons"] + [out.get("action", "")], {"slot": slot, "kind": kind, "text": text},
                    out.get("id"), session, rec.provenance)
        return out

    # -- read ------------------------------------------------------------------------------------
    def recall(self, query: str, k: int = 5, budget_tokens: int | None = None, *, as_of: float | None = None,
               kinds=None, session: str | None = None, mode: str = "hybrid") -> list[dict]:
        hits = self.store.search(self.tenant, self.user, query, k, as_of=as_of, kinds=kinds, session=session,
                                 mode=mode, now=self.clock())
        if budget_tokens is not None:
            hits = pack(hits, budget_tokens)
        out = [{"id": h.record.id, "text": h.record.text, "kind": h.record.kind, "date": day(h.record.valid_from),
                "trust": h.record.trust, "score": round(h.score, 6), "tokens": h.tokens} for h in hits]
        self._audit("memory.read", "allow", [f"{len(out)} records"], {"query": query, "k": k},
                    [o["id"] for o in out], session, [])
        return out

    def profile(self, max_items: int = 6, budget_tokens: int | None = None) -> list[dict]:
        """The pinned profile: the partition's active semantic facts by importance, then slot and text —
        sorted, so it renders byte-identically on every turn of a session (the prefix cache survives)."""
        recs = [r for r in self.store.records(self.tenant, self.user, statuses=("active",), kind="semantic")
                if r.scope in ("user", "tenant")]
        recs.sort(key=lambda r: (-r.importance, r.slot or "", r.text))
        out, used = [], 0
        for r in recs[:max_items]:
            t = count_tokens(r.text)
            if budget_tokens is not None and used + t > budget_tokens:
                continue
            used += t
            out.append({"id": r.id, "text": r.text, "kind": r.kind, "date": day(r.valid_from), "trust": r.trust,
                        "tokens": t})
        return out

    # -- forget ----------------------------------------------------------------------------------
    def deletion_key_for(self, subject: str) -> str:
        """A forget request names a subject ("my address", "address") or a full deletion key."""
        if subject.count("/") == 2:
            return subject
        slot = subject if subject in SLOTS else question_slot(subject) or subject
        return default_deletion_key(self.tenant, self.user, slot)

    def forget(self, subject: str, *, mode: str = "purge", needles=()) -> DeletionReport:
        key = self.deletion_key_for(subject)
        rep = self.store.forget(self.tenant, key, user=self.user, mode=mode, needles=needles)
        self._audit("memory.forget", "allow", [f"{len(rep.deleted_ids)} records", mode], {"deletion_key": key},
                    rep.deleted_ids, None, [])
        return rep

    def render(self, items: list[dict]) -> str:
        return render_memory([(i["text"], i.get("date")) for i in items], self.scope)

    # -- audit -----------------------------------------------------------------------------------
    def _audit(self, event_type, decision, reasons, args, result, session, provenance) -> None:
        self.audit.record(AuditEvent(event_type, self.agent, "delegated", user=self.user, decision=decision,
                                     reasons=[r for r in reasons if r], args_hash=args_digest(args),
                                     result_hash=args_digest(result) if result is not None else None,
                                     provenance=list(provenance), session_id=session, tenant=self.tenant,
                                     extra={"gen_ai.memory.store.id": "memlab-sqlite"}))
