"""deletion.py — a forget is a checklist, not a DELETE: count every copy, then look for the bytes.

The one idea (PRIMER §7): deleting the row does not delete the fact. The same text lives in the
full-text index (FTS5 keeps deleted terms until a merge or ``secure-delete``), in the write-ahead log
(old page images until a checkpoint), in freed database pages (until they are overwritten or the file
is vacuumed), in facts *derived* from it (a consolidated summary), in a prompt prefix an inference
engine still caches, in logs, in eval sets and in backups — and the embedding is itself the data
(embeddings primer §15: vectors can be inverted). ``DeletionReport`` records what a forget touched,
surface by surface, and ``residue()`` proves it by searching the files for the bytes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Every surface a fact can reach, in the order a forget should visit them (PRIMER §7's checklist).
SURFACES = ("records", "vectors", "fts_rows", "derived", "idempotency", "prompt_cache", "audit_log",
            "eval_sets", "backups")


@dataclass
class DeletionReport:
    deletion_key: str
    tenant: str
    counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SURFACES})
    steps: list[str] = field(default_factory=list)       # what ran: DELETE, fts optimize, checkpoint, VACUUM ...
    residue: dict[str, int] = field(default_factory=dict)  # needle -> occurrences still on disk
    not_reachable: list[str] = field(default_factory=list)  # surfaces this process cannot purge (say so)
    deleted_ids: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no searched needle survives anywhere this process can read."""
        return all(v == 0 for v in self.residue.values())

    def table(self) -> str:
        rows = [f"forget {self.deletion_key!r} (tenant {self.tenant})"]
        rows += [f"  {s:13s} {self.counts.get(s, 0):4d}" for s in SURFACES]
        rows.append("  steps: " + " -> ".join(self.steps))
        if self.residue:
            rows.append("  residue on disk: " + ", ".join(f"{k!r}: {v}" for k, v in self.residue.items()))
        if self.not_reachable:
            rows.append("  not reachable from here: " + "; ".join(self.not_reachable))
        return "\n".join(rows)


def db_files(path: str | os.PathLike) -> list[Path]:
    """The database file and its companions that can hold page images: ``-wal`` (and ``-journal``)."""
    p = Path(path)
    return [f for f in (p, Path(str(p) + "-wal"), Path(str(p) + "-journal")) if f.exists()]


def count_bytes(files, needle: bytes) -> int:
    total = 0
    for f in files:
        data = Path(f).read_bytes()
        total += data.count(needle)
    return total


def residue(path: str | os.PathLike, needles=(), blobs=()) -> dict[str, int]:
    """Occurrences of each text needle (as given and lower-cased, since the FTS index stores lower-cased
    terms) and of each raw byte blob (a float32 embedding) in the database file and its WAL."""
    files = db_files(path)
    out: dict[str, int] = {}
    for n in needles:
        forms = {n.encode("utf-8"), n.lower().encode("utf-8")}
        out[n] = sum(count_bytes(files, f) for f in forms)
    for i, b in enumerate(blobs):
        out[f"<vector #{i}, {len(b)} bytes>"] = count_bytes(files, bytes(b))
    return out
