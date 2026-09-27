"""store.py - a memory store partitioned by (tenant, user), with a flat vector index that can DELETE.

The one idea: scope is a partition, not a filter. Every record lives in exactly one (tenant, user) partition
and search runs inside one partition only, so a missing `WHERE user = ...` cannot leak another user's memory
(vector-databases primer §9: "if a filter is always present and highly selective ... make it a partition").
Each partition holds three copies of a memory - the record, its vector, and its terms in a full-text index -
and `delete` must remove all three. The flat index removes the row outright, so there are no tombstones;
minifaiss's HNSW (07.4) has no delete at all, and a production HNSW marks deletions and vacuums later
(vector-databases primer §8).

The embedder is ragkit's crc32 hashing embedder, re-implemented (no import): lexical, deterministic, offline.
"lives" and "live" share nothing - the paraphrase miss the harness measures on purpose.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from .records import MemoryRecord, Scope

_TOKEN_RE = re.compile(r"[a-z0-9]+")
LABEL = "hashing embedder (T0; lexical, not semantic)"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Bag-of-words feature hashing into `dim` buckets, L2-normalised (ragkit.embed.HashingEmbedder's recipe)."""

    def __init__(self, dim: int = 1024):
        self.dim, self.model_name = dim, LABEL

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        """str -> (dim,) ; list[str] -> (n, dim). Rows are unit length, so a dot product is a cosine."""
        if isinstance(texts, str):
            return self._vec(texts)
        texts = list(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype="float32")


@dataclass
class Partition:
    records: dict[str, MemoryRecord] = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)              # row i of `vecs` belongs to ids[i]
    vecs: np.ndarray | None = None
    fulltext: dict[str, set[str]] = field(default_factory=dict)   # term -> record ids


class MemoryStore:
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self.partitions: dict[tuple[str, str], Partition] = {}

    def _part(self, scope: Scope) -> Partition:
        return self.partitions.setdefault(scope.partition, Partition())

    def put(self, rec: MemoryRecord) -> MemoryRecord:
        """Write the record, its vector and its full-text terms. Re-putting an id replaces all three."""
        p = self._part(rec.scope)
        if rec.id in p.records:
            self.delete(rec.scope, rec.id)
        p.records[rec.id] = rec
        v = self.embedder.encode(rec.text)[None, :]
        p.vecs = v if p.vecs is None else np.vstack([p.vecs, v])
        p.ids.append(rec.id)
        for term in set(tokenize(rec.text)):
            p.fulltext.setdefault(term, set()).add(rec.id)
        return rec

    def update(self, rec: MemoryRecord) -> MemoryRecord:
        """Change fields in place (status, validity, access time). The text - and so the vector - is unchanged."""
        self._part(rec.scope).records[rec.id] = rec
        return rec

    def get(self, scope: Scope, rid: str) -> MemoryRecord | None:
        return self._part(scope).records.get(rid)

    def records(self, scope: Scope, status: str | None = "active", kind: str | None = None) -> list[MemoryRecord]:
        return [r for r in self._part(scope).records.values()
                if (status is None or r.status == status) and (kind is None or r.kind == kind)]

    def find(self, scope: Scope, key: str, status: str | None = "active") -> list[MemoryRecord]:
        return [r for r in self.records(scope, status) if r.key == key]

    def similarity(self, scope: Scope, query: str, rids: list[str]) -> np.ndarray:
        """Cosine between the query and each record - an exact scan of this partition's rows only."""
        p = self._part(scope)
        if not rids:
            return np.zeros(0)
        rows = [p.ids.index(r) for r in rids]
        return p.vecs[rows] @ self.embedder.encode(query)

    def search(self, scope: Scope, query: str, k: int = 5, status: str | None = "active"):
        """Top-k (record, cosine) by similarity alone - the baseline retrieve.py improves on."""
        recs = self.records(scope, status)
        sims = self.similarity(scope, query, [r.id for r in recs])
        order = np.argsort(-sims, kind="stable")[:k]
        return [(recs[i], float(sims[i])) for i in order]

    def fulltext_search(self, scope: Scope, term: str) -> set[str]:
        return set(self._part(scope).fulltext.get(term.lower(), set()))

    def delete(self, scope: Scope, rid: str) -> dict[str, int]:
        """Remove the record, its vector row and its full-text postings; report what was removed."""
        p = self._part(scope)
        out = {"records": 0, "vectors": 0, "fulltext_postings": 0}
        if p.records.pop(rid, None) is not None:
            out["records"] = 1
        if rid in p.ids:
            i = p.ids.index(rid)
            p.ids.pop(i)
            p.vecs = np.delete(p.vecs, i, axis=0)
            out["vectors"] = 1
        for term in [t for t, ids in p.fulltext.items() if rid in ids]:
            p.fulltext[term].discard(rid)
            out["fulltext_postings"] += 1
            if not p.fulltext[term]:
                del p.fulltext[term]
        return out

    def counts(self, scope: Scope) -> dict[str, int]:
        p = self._part(scope)
        return {"records": len(p.records), "vectors": 0 if p.vecs is None else len(p.vecs),
                "fulltext_postings": sum(len(v) for v in p.fulltext.values())}
