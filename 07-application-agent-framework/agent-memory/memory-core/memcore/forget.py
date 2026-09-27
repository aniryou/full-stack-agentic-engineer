"""forget.py - forgetting is four mechanisms; deletion is a propagation problem.

The one idea: an agent forgets in four ways - decay (old, unimportant memories stop ranking), TTL (a record
expires), the per-turn budget (what does not fit is not seen) and a cap per scope (the weakest are evicted) -
and none of them is deletion. Deleting a fact means finding every copy: the record, its vector and its
full-text postings (store.delete), the facts derived from it (provenance), anything else that quotes it, the
prompt-cache blocks that hold it (a cached prefix cannot be edited, only evicted: evict the user's salt), the
logs, the eval set, and the backups (which cannot be rewritten: they expire, or their key is shredded).
`propagate` walks that list and returns a DeletionReport with a count per surface and the residue left.
Embeddings are the data: a vector of a deleted text is not anonymous (embeddings primer §15).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .records import DAY, Scope


def retention(rec, now: float, half_life_s: float = 30 * DAY) -> float:
    """Decay: importance (0..1) halved for every `half_life_s` since the memory was last used."""
    return rec.importance / 10 * 0.5 ** ((now - rec.last_accessed) / half_life_s)


def expire(store, scope: Scope, now: float) -> list[str]:
    """TTL: delete every record past created_at + ttl_s."""
    gone = [r.id for r in store.records(scope, status=None) if r.expired(now)]
    for rid in gone:
        store.delete(scope, rid)
    return gone


def cap(store, scope: Scope, max_records: int, now: float, kind: str = "episodic",
        half_life_s: float = 30 * DAY) -> list[str]:
    """A cap per scope: keep the `max_records` with the highest retention, delete the rest."""
    recs = sorted(store.records(scope, kind=kind), key=lambda r: (-retention(r, now, half_life_s), r.id))
    gone = [r.id for r in recs[max_records:]]
    for rid in gone:
        store.delete(scope, rid)
    return gone


@dataclass
class Surfaces:
    """Every place a memory's content can end up, beyond the store itself."""
    store: object
    cache: object = None                                     # budget.PrefixCache, salted per (tenant, user)
    logs: list[str] = field(default_factory=list)
    eval_cases: list[dict] = field(default_factory=list)     # {"text": ..., "provenance": (ids)}
    backups: list[dict] = field(default_factory=list)        # snapshots {record id: text}


@dataclass
class DeletionReport:
    removed: dict[str, int]
    residue: dict[str, int]
    ids: list[str]

    def checklist(self) -> str:
        rows = []
        for surface in self.removed:
            left = self.residue.get(surface, 0)
            mark = "✓" if left == 0 else ("pending" if surface == "backups" else "LEFT")
            rows.append(f"{mark:8} {surface:18} removed {self.removed[surface]:3d}   still holding it: {left}")
        return "\n".join(rows)


def residue(surfaces: Surfaces, scope: Scope, needles) -> dict[str, int]:
    """Search every surface for the deleted content - the only test of a deletion that means anything."""
    has = lambda text: any(n.lower() in text.lower() for n in needles)
    s = surfaces.store
    recs = {r.id: r for r in s.records(scope, status=None)}
    rows = s.partitions.get(scope.partition).ids if scope.partition in s.partitions else []
    return {"records": sum(has(r.text) for r in recs.values()),
            "vectors": sum(rid not in recs or has(recs[rid].text) for rid in rows),    # orphans count too
            "fulltext_postings": sum(len(s.fulltext_search(scope, w)) for n in needles for w in n.lower().split()),
            "prompt_cache_blocks": surfaces.cache.count("/".join(scope.partition)) if surfaces.cache else 0,
            "logs": sum(has(line) for line in surfaces.logs),
            "eval_cases": sum(has(c["text"]) for c in surfaces.eval_cases),
            "backups": sum(has(t) for snap in surfaces.backups for t in snap.values())}


def propagate(surfaces: Surfaces, scope: Scope, *, ids=(), key: str | None = None) -> DeletionReport:
    """Delete the target records (by id, or every version of a slot), everything derived from them, and every
    other copy of their values; evict the user's cached prefixes; redact logs; drop eval cases. Backups stay."""
    store = surfaces.store
    recs = {r.id: r for r in store.records(scope, status=None)}
    targets = set(ids) | {r.id for r in recs.values() if key is not None and r.key == key}
    needles = {recs[t].value or recs[t].text for t in targets if t in recs}
    derived, frontier = set(targets), set(targets)
    while frontier:                                          # downstream: anything whose provenance cites a target
        frontier = {r.id for r in recs.values() if r.id not in derived and set(r.provenance) & derived}
        derived |= frontier
    quoting = {r.id for r in recs.values() if any(n.lower() in r.text.lower() for n in needles)}
    removed = {"records": 0, "vectors": 0, "fulltext_postings": 0, "derived_facts": len(derived - targets),
               "prompt_cache_blocks": 0, "logs": 0, "eval_cases": 0, "backups": 0}
    doomed = sorted((derived | quoting) & set(recs))
    for rid in doomed:
        for surface, n in store.delete(scope, rid).items():
            removed[surface] += n
    if surfaces.cache is not None:
        removed["prompt_cache_blocks"] = surfaces.cache.evict("/".join(scope.partition))
    hit = lambda text: any(n.lower() in text.lower() for n in needles)
    removed["logs"] = sum(hit(line) for line in surfaces.logs)
    surfaces.logs[:] = ["[redacted: deleted memory]" if hit(line) else line for line in surfaces.logs]
    keep = [c for c in surfaces.eval_cases if not (hit(c["text"]) or set(c.get("provenance", ())) & set(doomed))]
    removed["eval_cases"] = len(surfaces.eval_cases) - len(keep)
    surfaces.eval_cases[:] = keep
    return DeletionReport(removed, residue(surfaces, scope, needles), doomed)
