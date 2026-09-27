"""forget.py - forgetting is four mechanisms; deletion is a propagation problem.

The one idea: an agent forgets in four ways - decay (old, unimportant memories stop ranking), TTL (a record
expires), the per-turn budget (what does not fit is not seen) and a cap per scope (the weakest are evicted) -
and none of them is deletion. Deleting a fact means finding every copy: the record, its vector and its
full-text postings (store.delete), the facts derived from it (provenance), anything else that quotes it, the
prompt-cache blocks that hold it, the logs, the eval set, and the backups (which cannot be rewritten: they expire,
or their key is shredded). A cached prefix cannot be edited, and vLLM cannot evict by salt: rotating the tenant's
`cache_salt` makes its blocks unreachable at once, and LRU eviction (or an operator's full reset) removes them.
`propagate` walks that list and returns a DeletionReport with a count per surface, the residue left, and a review
list. Content matching finds only VERBATIM copies - "the user lives in Portugal's second city" does not contain
"Porto" - so a record derived (by provenance) from a deleted record that does not quote the value is listed for
review: it may be a paraphrase, or an unrelated fact extracted from the same turn. Values are matched on word
boundaries, so forgetting a pet "cat" never touches "education" or "Catalyst".
Embeddings are the data: a vector of a deleted text is not anonymous (embeddings primer §15).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .records import DAY, Scope


def mentions(text: str, needle: str) -> bool:
    """True if `needle` occurs in `text` as whole words (case-insensitive): "cat" in "my cat" but not "education"."""
    return re.search(r"(?<![a-z0-9])" + re.escape(needle.lower()) + r"(?![a-z0-9])", text.lower()) is not None


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
    cache: object = None                                     # budget.PrefixCache shared by every tenant
    salts: object = None                                     # budget.Salts: the current cache_salt per tenant
    logs: list[str] = field(default_factory=list)
    eval_cases: list[dict] = field(default_factory=list)     # {"text": ..., "provenance": (ids)}
    backups: list[dict] = field(default_factory=list)        # snapshots {record id: text}


# Surfaces whose residue is expected after a correct deletion, and why: they are not rewritten, they age out.
PENDING = {"backups": "expire on their retention schedule (or their key is shredded)",
           "prompt_cache_blocks": "unreachable under the rotated salt; resident until LRU eviction or a reset"}


@dataclass
class DeletionReport:
    removed: dict[str, int]
    residue: dict[str, int]
    ids: list[str]
    review: list[str] = field(default_factory=list)   # derived from a deleted record, no verbatim copy: check them

    def checklist(self) -> str:
        rows = []
        for surface in self.removed:
            left = self.residue.get(surface, 0)
            mark = "✓" if left == 0 else ("pending" if surface in PENDING else "LEFT")
            rows.append(f"{mark:8} {surface:19} removed {self.removed[surface]:3d}   still holding it: {left}"
                        + (f"  ({PENDING[surface]})" if left and surface in PENDING else ""))
        if self.review:
            rows.append(f"{'review':8} {'derived records':19} {len(self.review):3d} cite a deleted record without "
                        f"quoting it (a paraphrase, or another fact from the same turn)")
        return "\n".join(rows)


def _derived_from(recs: dict, seeds: set) -> set:
    """`seeds` plus every record whose provenance cites one of them, to a fixed point."""
    found, frontier = set(seeds), set(seeds)
    while frontier:
        frontier = {r.id for r in recs.values() if r.id not in found and set(r.provenance) & found}
        found |= frontier
    return found


def residue(surfaces: Surfaces, scope: Scope, needles) -> dict[str, int]:
    """Search every surface for the deleted content - the only test of a deletion that means anything.
    The prompt cache counts every block still resident under this tenant's current or retired salts (a view only
    the simulator has; on a real engine you know the blocks are there until evicted, not how many)."""
    has = lambda text: any(mentions(text, n) for n in needles)
    s = surfaces.store
    recs = {r.id: r for r in s.records(scope, status=None)}
    rows = s.partitions.get(scope.partition).ids if scope.partition in s.partitions else []
    cache_blocks = 0
    if surfaces.cache is not None and surfaces.salts is not None:
        salts = [surfaces.salts.salt(scope.tenant)] + surfaces.salts.retired.get(scope.tenant, [])
        cache_blocks = sum(surfaces.cache.count(x) for x in salts)
    return {"records": sum(has(r.text) for r in recs.values()),
            "vectors": sum(rid not in recs or has(recs[rid].text) for rid in rows),    # orphans count too
            "fulltext_postings": sum(len(s.fulltext_search(scope, w)) for n in needles for w in n.lower().split()),
            "prompt_cache_blocks": cache_blocks,
            "logs": sum(has(line) for line in surfaces.logs),
            "eval_cases": sum(has(c["text"]) for c in surfaces.eval_cases),
            "backups": sum(has(t) for snap in surfaces.backups for t in snap.values())}


def propagate(surfaces: Surfaces, scope: Scope, *, ids=(), key: str | None = None) -> DeletionReport:
    """Delete the target records (by id, or every version of a slot), everything derived from them, and every other
    record that quotes their values verbatim; list for review what was derived from a deleted quoting record without
    quoting it; rotate the tenant's cache salt; redact logs; drop eval cases. Backups stay (pending)."""
    store = surfaces.store
    recs = {r.id: r for r in store.records(scope, status=None)}
    targets = set(ids) | {r.id for r in recs.values() if key is not None and r.key == key}
    needles = {recs[t].value or recs[t].text for t in targets if t in recs}
    hit = lambda text: any(mentions(text, n) for n in needles)
    derived = _derived_from(recs, targets)                   # downstream of the facts themselves: delete
    quoting = {r.id for r in recs.values() if hit(r.text)} - derived
    doomed_set = (derived | quoting) & set(recs)
    review = sorted(_derived_from(recs, quoting) - doomed_set)   # downstream of a quoting record: review
    removed = {"records": 0, "vectors": 0, "fulltext_postings": 0, "derived_facts": len(derived - targets),
               "prompt_cache_blocks": 0, "logs": 0, "eval_cases": 0, "backups": 0}
    doomed = sorted(doomed_set)
    for rid in doomed:
        for surface, n in store.delete(scope, rid).items():
            removed[surface] += n
    if surfaces.cache is not None and surfaces.salts is not None:
        old = surfaces.salts.rotate(scope.tenant)            # no eviction by salt exists: make them unreachable
        removed["prompt_cache_blocks"] = surfaces.cache.count(old)
    removed["logs"] = sum(hit(line) for line in surfaces.logs)
    surfaces.logs[:] = ["[redacted: deleted memory]" if hit(line) else line for line in surfaces.logs]
    keep = [c for c in surfaces.eval_cases if not (hit(c["text"]) or set(c.get("provenance", ())) & set(doomed))]
    removed["eval_cases"] = len(surfaces.eval_cases) - len(keep)
    surfaces.eval_cases[:] = keep
    return DeletionReport(removed, residue(surfaces, scope, needles), doomed, review)
