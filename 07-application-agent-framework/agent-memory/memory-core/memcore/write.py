"""write.py - the write path: extract, check the policy, merge, and write each memory exactly once.

The one idea: what reaches memory is decided in code, not by the model. After a turn, an extractor proposes
candidate records (a template extractor stands in for the LLM call). A `WritePolicy` then decides per
candidate: which kinds each source may write (a tool may never write procedural memory - that is what a
standing instruction planted by a web page looks like), a confidence floor, and screening before persistence
(secrets are rejected, injection phrasing and every tool-sourced fact are quarantined - stored, never
retrieved until a human promotes them; 07.2 notebook 11 §4). Survivors are merged against what the partition
already holds: the same value is a NOOP, a new value from an equal or stronger source is an UPDATE that closes
the old fact (valid_to) instead of deleting it, an OLDER value that arrives late (a backfill, an out-of-order
background extraction) is kept as closed history and never supersedes a newer one, and a weaker contradiction is
quarantined. Every write carries an idempotency key that names the step - (session, turn, call index), never the
content - so a retried turn replays its first result instead of writing again (durable primer §3.2).

This is extract -> compare -> ADD / UPDATE / NOOP, the shape mem0 used before its 2.0.0 release (its events were
ADD / UPDATE / DELETE / NONE; mem0 2.x is ADD-only). Deletion is not a write-path event here: `forget` is.
"""
from __future__ import annotations

import hashlib
import re
import zlib
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
    "procedure": ("standing preference", 6, [r"(?i:please always|for this user, always) (?P<v>[a-z][a-z0-9 -]*[a-z0-9])"]),
}
NOUN_TO_KEY = {noun: key for key, (noun, _, _) in SLOTS.items()}
FACT_RE = re.compile(r"(?i:the user's) (?P<noun>[a-z ]+) is (?P<v>[A-Za-z][A-Za-z ]*)\.")
# A family of open-ended slots ("My favourite colour is teal." -> favourite_colour), so the harness can give a user
# far more facts than a profile holds. Importance is a fixed pseudo-rating per thing (2..8), like a model's rating.
FAVOURITE_RE = re.compile(r"(?i:my favourite) (?P<thing>[a-z]+) (?i:is) (?P<v>[a-z]+)")


def slot_info(key: str) -> tuple[str, int]:
    """(noun used in the fact's text, importance 1..10) for a fixed slot or a favourite_<thing> slot."""
    if key in SLOTS:
        return SLOTS[key][0], SLOTS[key][1]
    if key.startswith("favourite_"):
        return "favourite " + key[len("favourite_"):], 2 + zlib.crc32(key.encode()) % 7
    raise KeyError(key)


def _noun_key(noun: str) -> str | None:
    if noun in NOUN_TO_KEY:
        return NOUN_TO_KEY[noun]
    parts = noun.split()
    return "favourite_" + parts[1] if len(parts) == 2 and parts[0] == "favourite" else None


def fact_text(key: str, value: str) -> str:
    return f"For this user, always {value}." if key == "procedure" else f"The user's {slot_info(key)[0]} is {value}."


def read_facts(text: str) -> list[tuple[str, str]]:
    """(key, value) pairs a reader finds in a memory's text - a distilled fact or a raw statement."""
    out = [(_noun_key(m["noun"]), m["v"]) for m in FACT_RE.finditer(text) if _noun_key(m["noun"])]
    for key, (_, _, pats) in SLOTS.items():
        out += [(key, m["v"]) for p in pats for m in re.finditer(p, text)]
    out += [("favourite_" + m["thing"].lower(), m["v"]) for m in FAVOURITE_RE.finditer(text)]
    return out


def extract(text: str, scope: Scope, *, source: str = "user", at: float = 0.0, turn_id: str = "",
            confidence: float = 0.9) -> list[MemoryRecord]:
    """One episodic record for the turn, plus one semantic (or procedural) candidate per recognised statement."""
    facts = []
    for key, value in read_facts(text):
        noun, importance = slot_info(key)
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
    action: str                          # ADD | UPDATE | ADD_HISTORY | NOOP | QUARANTINE | REJECT
    record: MemoryRecord | None
    reasons: list[str] = field(default_factory=list)
    superseded: str | None = None        # id of the fact an UPDATE closed


class IdempotencyConflict(ValueError):
    """A known key arrived with a different write: the caller re-derived the step instead of replaying it.
    Never a second write (an HTTP API answers 422; the lab's service does)."""


def idempotency_key(session: str, turn: int, index: int) -> str:
    """session:turn:index - it names the STEP, not its content. A retried turn re-asks the extractor and may get
    different text back; a key with the text in it would then be new, and the retry would write a second, different
    fact (the durable primer's §1.3 failure, which its §3.2 fixes: "a naive retry re-asks the model ... and produces
    a second, different side effect"). With the step as the key, the retry replays the first result, or fails loudly."""
    return f"{session}:{turn}:{index}"


def fingerprint(rec: MemoryRecord) -> str:
    return hashlib.sha256(repr((rec.kind, rec.key, _norm(rec.value), _norm(rec.text))).encode()).hexdigest()[:16]


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


class Writer:
    """The write path. `journal` maps ((tenant, user), key) -> (fingerprint, result): keys are unique per partition
    (two users may both have a session "s1"). It is in-process here; in production it is a table written in the
    same transaction as the record (memory-lab's `idempotency_key` column), so it survives the crash that caused
    the retry."""

    def __init__(self, store, policy: WritePolicy | None = None):
        self.store, self.policy = store, policy or WritePolicy()
        self.journal: dict[tuple, tuple[str, WriteResult]] = {}

    def write(self, rec: MemoryRecord, idempotency_key: str | None = None) -> WriteResult:
        slot = (rec.scope.partition, idempotency_key)
        if idempotency_key is not None and slot in self.journal:
            fp, result = self.journal[slot]
            if fp != fingerprint(rec):
                raise IdempotencyConflict(f"key {idempotency_key!r} already wrote {result.action} "
                                          f"{result.record.text if result.record else ''!r}: replay the journaled "
                                          f"step instead of re-extracting it")
            return result                                 # a replayed turn: same answer, no second write
        result = self._write(rec)
        if idempotency_key is not None:
            self.journal[slot] = (fingerprint(rec), result)
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
            old.valid_from = min(old.valid_from, rec.valid_from)             # older evidence: true since then
            return WriteResult("NOOP", self.store.update(old), ["already known"])
        if rec.trust < old.trust:
            return WriteResult("QUARANTINE", self.store.put(rec.copy(status="quarantined")),
                               [f"contradicts {old.id} from a stronger source ({old.source})"])
        if rec.valid_from < old.valid_from:           # a late, OLDER value: history, never the current fact
            end = history_end(self.store.find(rec.scope, rec.key, status=None), rec.valid_from) if rec.key else None
            return WriteResult("ADD_HISTORY", self.store.put(rec.copy(status="superseded", valid_to=end,
                                                                      superseded_at=rec.created_at)),
                               [f"older than {old.id} (valid from day {old.valid_from / 86400:g}): kept as history"])
        old.valid_to, old.superseded_at, old.status = rec.valid_from, rec.created_at, "superseded"
        self.store.update(old)
        return WriteResult("UPDATE", self.store.put(rec.copy(meta={**rec.meta, "supersedes": old.id})),
                           [f"supersedes {old.id}"], superseded=old.id)


def history_end(records, valid_from: float) -> float | None:
    """Where a late, older value stops being true: the valid_from of the next known value of the slot (active or
    already superseded), or None if nothing later is known (memory-lab's memory.history_end)."""
    later = [r.valid_from for r in records if r.valid_from > valid_from and r.status in ("active", "superseded")]
    return min(later) if later else None
