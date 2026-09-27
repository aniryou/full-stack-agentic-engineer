"""write.py - the write path: extract, check the policy, merge, and write each memory exactly once.

The one idea: what reaches memory is decided in code, not by the model. After a turn, an extractor proposes
candidate records (a template extractor stands in for the LLM call). A `WritePolicy` then decides per
candidate: which kinds each source may write (a tool may never write procedural memory - that is what a
standing instruction planted by a web page looks like), a confidence floor, and screening before persistence
(secrets are rejected, injection phrasing and every tool-sourced fact are quarantined - stored, never
retrieved until a human promotes them; 07.2 notebook 11 §4). Survivors are merged against what the partition
already holds: the same value is a NOOP, a new value from an equal or stronger source is an UPDATE that closes
the old fact (valid_to) instead of deleting it, and a weaker contradiction is quarantined. Every write carries
an idempotency key, so a retried turn writes once (durable primer §3.2).

This is extract -> compare -> ADD / UPDATE / NOOP, the shape mem0 used before its 2.0.0 release (its events were
ADD / UPDATE / DELETE / NONE; mem0 2.x is ADD-only). Deletion is not a write-path event here: `forget` is.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .records import KINDS, MemoryRecord, Scope

# key -> (noun used in the fact's text, importance 1..10, statement patterns the extractor recognises)
SLOTS = {
    "name": ("name", 7, [r"(?i:my name is) (?P<v>[A-Z][a-z]+)"]),
    "home_city": ("home city", 6, [r"(?i:i live in) (?P<v>[A-Z][a-z]+)", r"(?i:i (?:have )?moved to) (?P<v>[A-Z][a-z]+)"]),
    "employer": ("employer", 5, [r"(?i:i work (?:at|for)) (?P<v>[A-Z][a-z]+)"]),
    "allergy": ("allergy", 9, [r"(?i:i(?: am|'m) allergic to) (?P<v>[a-z]+)"]),
    "seat_preference": ("seat preference", 4, [r"(?i:i prefer) (?P<v>window|aisle)(?i: seats)"]),
    "pet": ("pet", 3, [r"(?i:my pet is an?) (?P<v>[a-z]+)"]),
    "language": ("language", 4, [r"(?i:i am learning) (?P<v>[A-Z][a-z]+)"]),
    "procedure": ("standing preference", 6, [r"(?i:please always) (?P<v>[a-z][a-z ]+)"]),
}
NOUN_TO_KEY = {noun: key for key, (noun, _, _) in SLOTS.items()}
FACT_RE = re.compile(r"(?i:the user's) (?P<noun>[a-z ]+) is (?P<v>[A-Za-z][A-Za-z ]*)\.")


def fact_text(key: str, value: str) -> str:
    return f"The user's {SLOTS[key][0]} is {value}."


def read_facts(text: str) -> list[tuple[str, str]]:
    """(key, value) pairs a reader finds in a memory's text - a distilled fact or a raw statement."""
    out = [(NOUN_TO_KEY[m["noun"]], m["v"]) for m in FACT_RE.finditer(text) if m["noun"] in NOUN_TO_KEY]
    for key, (_, _, pats) in SLOTS.items():
        out += [(key, m["v"]) for p in pats for m in re.finditer(p, text)]
    return out


def extract(text: str, scope: Scope, *, source: str = "user", at: float = 0.0, turn_id: str = "",
            confidence: float = 0.9) -> list[MemoryRecord]:
    """One episodic record for the turn, plus one semantic (or procedural) candidate per recognised statement."""
    facts = []
    for key, value in read_facts(text):
        noun, importance, _ = SLOTS[key]
        facts.append(MemoryRecord(fact_text(key, value), "procedural" if key == "procedure" else "semantic", scope,
                                  source, key=key, value=value, confidence=confidence, importance=importance,
                                  created_at=at, provenance=(turn_id,) if turn_id else ()))
    prefix = {"user": "User said", "tool": "Tool returned", "human": "Operator noted", "inferred": "Noted"}[source]
    episode = MemoryRecord(f"{prefix}: {text}", "episodic", scope, source, created_at=at, confidence=1.0,
                           importance=max([f.importance for f in facts], default=2),
                           provenance=(turn_id,) if turn_id else ())
    for f in facts:
        f.provenance = f.provenance + (episode.id,)
    return [episode] + facts


# --- screening before persistence (a few rules; 07.2's agentlab.security.screen has the full set) ----------
INJECTION = [r"(?i)ignore (?:all |any )?(?:previous|prior) instructions", r"(?i)\byou are now\b",
             r"(?i)\bnew instructions\b", r"(?i)\bfrom now on\b", r"(?i)^\s*system\s*:"]
SECRETS = [r"\bsk-[A-Za-z0-9]{16,}", r"(?i)\bpassword\s*[:=]\s*\S+", r"\b(?:\d[ -]?){13,19}\b"]


def screen(text: str) -> list[str]:
    return ([f"secret:{p}" for p in SECRETS if re.search(p, text)]
            + [f"injection:{p}" for p in INJECTION if re.search(p, text)])


@dataclass
class WritePolicy:
    kinds_by_source: dict = field(default_factory=lambda: {
        "human": set(KINDS), "user": set(KINDS), "inferred": {"semantic", "procedural"},
        "tool": {"episodic", "semantic"}})             # a tool never writes procedural memory
    min_confidence: float = 0.6
    quarantine_sources: tuple = ("tool",)              # stored, never retrieved until a human promotes it

    def check(self, rec: MemoryRecord) -> tuple[str | None, list[str]]:
        """REJECT / QUARANTINE with reasons, or (None, []) to continue to the merge."""
        if rec.kind not in self.kinds_by_source.get(rec.source, set()):
            return "REJECT", [f"source {rec.source!r} may not write {rec.kind} memory"]
        if rec.confidence < self.min_confidence:
            return "REJECT", [f"confidence {rec.confidence:.2f} < floor {self.min_confidence:.2f}"]
        findings = screen(rec.text)
        if any(f.startswith("secret:") for f in findings):
            return "REJECT", ["a secret is never persisted"] + findings
        if findings:
            return "QUARANTINE", findings
        if rec.source in self.quarantine_sources:
            return "QUARANTINE", [f"source {rec.source!r} is quarantined until reviewed"]
        return None, []


@dataclass
class WriteResult:
    action: str                          # ADD | UPDATE | NOOP | QUARANTINE | REJECT
    record: MemoryRecord | None
    reasons: list[str] = field(default_factory=list)
    superseded: str | None = None        # id of the fact an UPDATE closed


def idempotency_key(session: str, turn: int, index: int, text: str) -> str:
    """session:turn:index plus a content hash - stable across retries of the same turn, unique across turns."""
    return f"{session}:{turn}:{index}:" + hashlib.sha256(text.encode()).hexdigest()[:8]


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


class Writer:
    def __init__(self, store, policy: WritePolicy | None = None):
        self.store, self.policy = store, policy or WritePolicy()
        self.seen: dict[str, WriteResult] = {}

    def write(self, rec: MemoryRecord, idempotency_key: str | None = None) -> WriteResult:
        if idempotency_key is not None and idempotency_key in self.seen:
            return self.seen[idempotency_key]            # a replayed turn: same answer, no second write
        result = self._write(rec)
        if idempotency_key is not None:
            self.seen[idempotency_key] = result
        return result

    def _write(self, rec: MemoryRecord) -> WriteResult:
        verdict, reasons = self.policy.check(rec)
        if verdict == "REJECT":
            return WriteResult("REJECT", None, reasons)
        if verdict == "QUARANTINE":
            return WriteResult("QUARANTINE", self.store.put(rec.copy(status="quarantined")), reasons)
        if rec.kind == "episodic":
            return WriteResult("ADD", self.store.put(rec))
        same = (self.store.find(rec.scope, rec.key) if rec.key else
                [r for r in self.store.records(rec.scope, kind=rec.kind) if _norm(r.text) == _norm(rec.text)])
        if not same:
            return WriteResult("ADD", self.store.put(rec))
        old = max(same, key=lambda r: r.valid_from)
        if _norm(old.value or old.text) == _norm(rec.value or rec.text):     # merge: provenance, confidence
            old.provenance = tuple(dict.fromkeys(old.provenance + rec.provenance))
            old.confidence = max(old.confidence, rec.confidence)
            return WriteResult("NOOP", self.store.update(old), ["already known"])
        if rec.trust < old.trust:
            return WriteResult("QUARANTINE", self.store.put(rec.copy(status="quarantined")),
                               [f"contradicts {old.id} from a stronger source ({old.source})"])
        old.valid_to, old.superseded_at, old.status = rec.valid_from, rec.created_at, "superseded"
        self.store.update(old)
        return WriteResult("UPDATE", self.store.put(rec.copy(meta={**rec.meta, "supersedes": old.id})),
                           [f"supersedes {old.id}"], superseded=old.id)
