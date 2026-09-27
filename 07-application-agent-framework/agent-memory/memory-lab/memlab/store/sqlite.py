"""sqlite.py — a memory store in one SQLite file: records, float32 vectors, FTS5, and a delete that removes the bytes.

The one idea: everything a memory store needs fits in the standard library — rows for the typed
record (PRIMER §1), a float32 BLOB per row scored with numpy (a flat index: exact, and it can delete,
which 07.4's ``minifaiss`` HNSW cannot), an FTS5 index ranked by ``bm25()``, and reciprocal rank
fusion of the two. The partition is ``(tenant, user_id)``: every query carries it in its WHERE clause,
so scope is a predicate the store enforces, never a filter the caller may forget (vector-databases
primer §9, §11). What the standard library does *not* do by default is forget: ``DELETE`` leaves the
text in the FTS5 index, in the WAL and in freed pages. ``forget(mode="purge")`` runs the steps that
remove it — FTS5 ``secure-delete`` (or ``optimize``), a WAL checkpoint, ``VACUUM`` — and
``deletion.residue()`` checks the file.

Facts this module relies on (SQLite 3.45.1 here; measured 2026-09-26, facts sheet §12):

* The FTS5 tokenizer is ``porter unicode61`` (stemmed); the question's stopwords are dropped from the
  MATCH expression.
* ``bm25()`` returns ``-1 × score`` — negative, lower is better — with k1 = 1.2, b = 0.75 and an IDF
  floored at 1e-6 (ragkit's BM25 uses k1 = 1.5). Its statistics are table-wide: another tenant's rows
  change your ranking (not your result set) unless each tenant has its own FTS table.
* FTS5 ``secure-delete`` (SQLite >= 3.42, verify) removes a deleted row's terms from the index; without
  it they stay until ``INSERT INTO fts(fts) VALUES('optimize')``. It is feature-detected, never assumed.
* In WAL mode the ``-wal`` file keeps old page images until ``PRAGMA wal_checkpoint(TRUNCATE)``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ..deletion import DeletionReport, residue
from ..embedders import HashingEmbedder, tokenize
from ..records import MemoryRecord, count_tokens

RRF_K = 60   # ragkit.reference.reciprocal_rank_fusion's default: score += 1 / (k + rank + 1), rank from 0

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  rowid           INTEGER PRIMARY KEY,
  id              TEXT NOT NULL UNIQUE,
  tenant          TEXT NOT NULL,
  user_id         TEXT NOT NULL,
  scope           TEXT NOT NULL,
  session_id      TEXT,
  agent           TEXT,
  kind            TEXT NOT NULL,
  text            TEXT NOT NULL,
  source          TEXT NOT NULL,
  trust           TEXT NOT NULL,
  slot            TEXT,
  value           TEXT,
  provenance      TEXT NOT NULL,          -- JSON list of turn refs / record ids
  confidence      REAL NOT NULL,
  importance      REAL NOT NULL,
  created_at      REAL NOT NULL,
  valid_from      REAL NOT NULL,
  valid_to        REAL,
  superseded_at   REAL,
  ttl_s           REAL,
  last_accessed   REAL NOT NULL,
  deletion_key    TEXT NOT NULL,
  status          TEXT NOT NULL,
  idempotency_key TEXT UNIQUE,
  dim             INTEGER NOT NULL,
  embedding       BLOB NOT NULL           -- float32, little-endian, L2-normalised
);
CREATE INDEX IF NOT EXISTS memories_partition ON memories (tenant, user_id, status);
CREATE INDEX IF NOT EXISTS memories_deletion  ON memories (tenant, deletion_key);
CREATE INDEX IF NOT EXISTS memories_slot      ON memories (tenant, user_id, slot);
"""
# Porter stemming on top of unicode61: "trips" finds "trip", "moved" finds "move" — lexical, still not semantic.
FTS_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, tokenize='porter unicode61')"
# Words that carry no retrieval signal in a question; dropped from the lexical leg only (the vectors keep them,
# as ragkit's hashing embedder does).
STOPWORDS = frozenset("a an and any are am as at be can could did do does for from have has how i in is it me my "
                      "of on or our please the these those to was were what when where which who why will with "
                      "would you your".split())

_COLS = [f.name for f in fields(MemoryRecord)]


def sqlite_features() -> dict:
    """What this Python's SQLite can do, detected on a throwaway in-memory database (never by version)."""
    out = {"sqlite_version": sqlite3.sqlite_version, "fts5": False, "fts5_secure_delete": False}
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        out["fts5"] = True
        con.execute("INSERT INTO probe(probe, rank) VALUES('secure-delete', 1)")
        out["fts5_secure_delete"] = True
    except sqlite3.OperationalError:
        pass
    finally:
        con.close()
    out["compile_secure_delete"] = any("SECURE_DELETE" in r[0] for r in
                                       sqlite3.connect(":memory:").execute("PRAGMA compile_options"))
    return out


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion as ``ragkit.reference.reciprocal_rank_fusion``: 1 / (k + rank + 1), rank from 0."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def fts_query(text: str) -> str:
    """A safe FTS5 MATCH expression: the query's ``[a-z0-9]+`` tokens minus stopwords, each quoted (so no
    FTS5 syntax can be injected), OR-ed together."""
    toks = [t for t in dict.fromkeys(tokenize(text)) if t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in toks)


@dataclass
class Hit:
    record: MemoryRecord
    score: float                      # the fused (or single-signal) score, higher is better
    cosine: float | None = None
    bm25: float | None = None         # SQLite's value: negative, lower is better
    rank_vector: int | None = None
    rank_fts: int | None = None

    @property
    def tokens(self) -> int:
        return count_tokens(self.record.text)


def pack(hits: list[Hit], budget_tokens: int) -> list[Hit]:
    """Top hits, in order, while they fit a per-turn token budget (a hit that does not fit is skipped,
    a smaller one after it may still fit)."""
    out, used = [], 0
    for h in hits:
        if used + h.tokens <= budget_tokens:
            out.append(h)
            used += h.tokens
    return out


class SQLiteMemoryStore:
    """A memory store in one file (or ``":memory:"``). Thread-safe through one lock; the memory service
    runs it on its event-loop thread while a notebook reads it from the main thread.

    ``journal_mode="wal"`` is the default because a service wants concurrent readers — and because it
    is the mode in which deletion is hardest, which is the lesson.
    """

    def __init__(self, path: str | Path = ":memory:", embedder=None, *, journal_mode: str = "wal",
                 fts_secure_delete: bool | None = None, clock=time.time):
        self.path = str(path)
        self.embedder = embedder or HashingEmbedder()
        self.clock = clock
        self.features = sqlite_features()
        if not self.features["fts5"]:
            raise RuntimeError("this Python's SQLite has no FTS5; install a Python built with it (see README)")
        self._lock = threading.RLock()
        self.con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        # 8 KiB pages keep a 1024-d float32 vector (4 KiB) on the same page as its row instead of spilling
        # to overflow pages; it only takes effect on a new file, before the first table.
        self.con.execute("PRAGMA page_size=8192")
        if self.path != ":memory:":
            self.con.execute(f"PRAGMA journal_mode={journal_mode}")
        self.con.execute("PRAGMA secure_delete=ON")        # zero freed b-tree content (off by default upstream)
        self.con.executescript(SCHEMA)
        existed = self.con.execute("SELECT 1 FROM sqlite_master WHERE name='memories_fts'").fetchone() is not None
        self.con.execute(FTS_SCHEMA)
        want = self.features["fts5_secure_delete"] if fts_secure_delete is None else fts_secure_delete
        if want and not self.features["fts5_secure_delete"]:
            raise RuntimeError(f"FTS5 secure-delete is not available in SQLite {sqlite3.sqlite_version}")
        if want and not existed:     # set it before the first insert (persisted in the FTS config table)
            self.con.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES('secure-delete', 1)")
        self.fts_secure_delete = bool(want) or self._fts_secure_delete_on()

    # ------------------------------------------------------------------ plumbing
    def _fts_secure_delete_on(self) -> bool:
        row = self.con.execute("SELECT v FROM memories_fts_config WHERE k='secure-delete'").fetchone()
        return bool(row and int(row[0]))

    def close(self) -> None:
        with self._lock:
            self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        d = {k: row[k] for k in row.keys()}
        d["user"] = d.pop("user_id")
        d["session"] = d.pop("session_id")
        d["provenance"] = json.loads(d["provenance"])
        return MemoryRecord(**{k: d[k] for k in _COLS})

    def _vector(self, text: str) -> np.ndarray:
        v = np.asarray(self.embedder.encode(text), dtype="<f4")
        return v.reshape(-1)

    # ------------------------------------------------------------------ writes
    def add(self, rec: MemoryRecord, *, idempotency_key: str | None = None,
            embedding: np.ndarray | None = None) -> tuple[str, bool]:
        """Insert a record, its vector and its FTS row in one transaction. Returns ``(id, created)``;
        a repeated ``idempotency_key`` returns the first write's id and ``created=False`` (durable
        primer §3.2: a retried turn writes once)."""
        vec = self._vector(rec.text) if embedding is None else np.asarray(embedding, dtype="<f4").reshape(-1)
        with self._lock:
            if idempotency_key:
                row = self.con.execute("SELECT id FROM memories WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row:
                    return row["id"], False
            self.con.execute("BEGIN IMMEDIATE")
            try:
                cur = self.con.execute(
                    "INSERT INTO memories (id, tenant, user_id, scope, session_id, agent, kind, text, source, trust,"
                    " slot, value, provenance, confidence, importance, created_at, valid_from, valid_to,"
                    " superseded_at, ttl_s, last_accessed, deletion_key, status, idempotency_key, dim, embedding)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rec.id, rec.tenant, rec.user, rec.scope, rec.session, rec.agent, rec.kind, rec.text, rec.source,
                     rec.trust, rec.slot, rec.value, json.dumps(rec.provenance), rec.confidence, rec.importance,
                     rec.created_at, rec.valid_from, rec.valid_to, rec.superseded_at, rec.ttl_s, rec.last_accessed,
                     rec.deletion_key, rec.status, idempotency_key, int(vec.shape[0]), vec.tobytes()))
                self.con.execute("INSERT INTO memories_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, rec.text))
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise
        return rec.id, True

    def set_fields(self, rec_id: str, **changes) -> None:
        """Update metadata columns (status, valid_to, superseded_at, last_accessed, ...). Text is immutable:
        a changed fact is a new record that supersedes the old one."""
        allowed = {"status", "valid_to", "superseded_at", "last_accessed", "ttl_s", "importance", "confidence"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}; write a new record and supersede instead")
        if not changes:
            return
        sets = ", ".join(f"{k}=?" for k in changes)
        with self._lock:
            self.con.execute(f"UPDATE memories SET {sets} WHERE id=?", (*changes.values(), rec_id))

    def supersede(self, old_id: str, at: float, now: float | None = None) -> None:
        """Close a fact's validity at ``at`` (valid time) and stamp when we learned it (system time).
        It stays readable to as-of queries: bi-temporal, as Graphiti keeps ``invalid_at`` / ``expired_at``."""
        self.set_fields(old_id, status="superseded", valid_to=at, superseded_at=self.clock() if now is None else now)

    # ------------------------------------------------------------------ reads
    def get(self, rec_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self.con.execute("SELECT * FROM memories WHERE id=?", (rec_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def records(self, tenant: str, user: str | None = None, *, statuses=None, kind: str | None = None,
                slot: str | None = None) -> list[MemoryRecord]:
        q, args = "SELECT * FROM memories WHERE tenant=?", [tenant]
        if user is not None:
            q += " AND user_id=?"
            args.append(user)
        if statuses:
            q += f" AND status IN ({','.join('?' * len(statuses))})"
            args += list(statuses)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if slot:
            q += " AND slot=?"
            args.append(slot)
        with self._lock:
            rows = self.con.execute(q + " ORDER BY created_at, rowid", args).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _partition_sql(self, tenant, user, statuses, kinds, session, as_of):
        where = ["m.tenant = ?", "m.user_id IN (?, '*')", f"m.status IN ({','.join('?' * len(statuses))})"]
        args: list = [tenant, user, *statuses]
        where.append("(m.scope != 'session' OR m.session_id = ?)")
        args.append(session)
        if kinds:
            where.append(f"m.kind IN ({','.join('?' * len(kinds))})")
            args += list(kinds)
        if as_of is not None:
            where.append("m.valid_from <= ? AND (m.valid_to IS NULL OR m.valid_to > ?)")
            args += [as_of, as_of]
        return " AND ".join(where), args

    def search(self, tenant: str, user: str, query: str, k: int = 5, *, mode: str = "hybrid",
               statuses=None, kinds=None, session: str | None = None, as_of: float | None = None,
               weights: dict | None = None, now: float | None = None, touch: bool = True,
               rrf_k: int = RRF_K) -> list[Hit]:
        """Top-k memories of one (tenant, user) partition for ``query``.

        mode ``"vector"`` (cosine over the partition's vectors, exact), ``"fts"`` (FTS5 ``bm25()``) or
        ``"hybrid"`` (RRF of both, k = 60). ``as_of`` keeps records valid at that time and, by default,
        includes superseded ones (what did we believe then?). ``weights`` re-ranks by relevance, recency
        and importance in the generative-agents paper's form (PRIMER §3; see ``rescore``). Reads are
        writes: returned records get ``last_accessed = now`` unless ``touch=False``.
        """
        if mode not in ("vector", "fts", "hybrid"):
            raise ValueError("mode is 'vector', 'fts' or 'hybrid'")
        statuses = tuple(statuses or (("active", "superseded") if as_of is not None else ("active",)))
        where, args = self._partition_sql(tenant, user, statuses, kinds, session, as_of)
        now = self.clock() if now is None else now
        depth = max(k * 4, 20)
        with self._lock:
            rows = self.con.execute(f"SELECT m.* FROM memories m WHERE {where}", args).fetchall()
            by_id = {r["id"]: r for r in rows}
            vec_rank: list[str] = []
            cos: dict[str, float] = {}
            if mode in ("vector", "hybrid") and rows:
                mat = np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in rows])
                q = self._vector(query)
                if q.shape[0] != mat.shape[1]:
                    raise ValueError(f"query vector has {q.shape[0]} dims, stored vectors {mat.shape[1]}")
                sims = mat @ q
                order = np.argsort(-sims, kind="stable")[:depth]
                vec_rank = [rows[i]["id"] for i in order if sims[i] > 0]
                cos = {rows[i]["id"]: float(sims[i]) for i in order}
            fts_rank: list[str] = []
            bm: dict[str, float] = {}
            match = fts_query(query)
            if mode in ("fts", "hybrid") and match:
                frows = self.con.execute(
                    f"SELECT m.id AS id, bm25(memories_fts) AS s FROM memories_fts JOIN memories m "
                    f"ON m.rowid = memories_fts.rowid WHERE memories_fts MATCH ? AND {where} "
                    f"ORDER BY s LIMIT ?", [match, *args, depth]).fetchall()
                fts_rank = [r["id"] for r in frows]
                bm = {r["id"]: float(r["s"]) for r in frows}
        if mode == "vector":
            fused = [(i, cos[i]) for i in vec_rank]
        elif mode == "fts":
            fused = [(i, -bm[i]) for i in fts_rank]
        else:
            fused = rrf([vec_rank, fts_rank], k=rrf_k)
        hits = [Hit(self._row_to_record(by_id[i]), s, cos.get(i), bm.get(i),
                    vec_rank.index(i) if i in vec_rank else None,
                    fts_rank.index(i) if i in fts_rank else None) for i, s in fused]
        if weights:
            hits = rescore(hits, weights, now)
        hits = hits[:k]
        if touch and hits:
            with self._lock:
                self.con.executemany("UPDATE memories SET last_accessed=? WHERE id=?",
                                     [(now, h.record.id) for h in hits])
        return hits

    # ------------------------------------------------------------------ forgetting
    def forget(self, tenant: str, deletion_key: str, *, user: str | None = None, mode: str = "purge",
               needles=(), include_derived: bool = True) -> DeletionReport:
        """Delete every record with ``deletion_key`` in ``tenant`` — and, with ``include_derived``, every
        record whose provenance cites one of them, transitively (a consolidated fact derived from the
        deleted episode). ``mode="logical"`` stops at DELETE (the naive forget); ``"purge"`` also removes
        the bytes: FTS5 ``optimize`` when ``secure-delete`` is off, ``wal_checkpoint(TRUNCATE)``,
        ``VACUUM``, and a second checkpoint. ``needles`` are searched on disk afterwards (plus each
        deleted row's embedding bytes)."""
        if mode not in ("logical", "purge"):
            raise ValueError("mode is 'logical' or 'purge'")
        rep = DeletionReport(deletion_key, tenant)
        with self._lock:
            q, args = "SELECT rowid, id, embedding FROM memories WHERE tenant=? AND deletion_key=?", [tenant, deletion_key]
            if user is not None:
                q += " AND user_id=?"
                args.append(user)
            targets = {r["id"]: (r["rowid"], bytes(r["embedding"])) for r in self.con.execute(q, args)}
            direct = set(targets)
            if include_derived and targets:
                frontier = set(targets)
                rows = self.con.execute("SELECT rowid, id, provenance, embedding FROM memories WHERE tenant=?",
                                        (tenant,)).fetchall()
                while frontier:
                    nxt = set()
                    for r in rows:
                        if r["id"] not in targets and frontier & set(json.loads(r["provenance"])):
                            targets[r["id"]] = (r["rowid"], bytes(r["embedding"]))
                            nxt.add(r["id"])
                    frontier = nxt
            blobs = [b for _, b in targets.values()]
            self.con.execute("BEGIN IMMEDIATE")
            try:
                for rid, (rowid, _) in targets.items():
                    self.con.execute("DELETE FROM memories_fts WHERE rowid=?", (rowid,))
                    self.con.execute("DELETE FROM memories WHERE id=?", (rid,))
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise
            rep.deleted_ids = sorted(targets)
            rep.counts.update(records=len(direct), vectors=len(targets), fts_rows=len(targets),
                              derived=len(targets) - len(direct))
            rep.steps.append("DELETE rows + FTS5 rows")
            if targets and self.con.execute("SELECT 1 FROM sqlite_master WHERE name='job_checkpoints'").fetchone():
                # an in-flight consolidation run keeps what it extracted in its checkpoints: drop them (the run's
                # steps are idempotent, so a resume redoes them without the deleted episodes)
                users = {r[0] for r in self.con.execute(
                    f"SELECT DISTINCT user_id FROM memories WHERE id IN ({','.join('?' * len(targets))})", list(targets))}
                users |= {user} if user else set()
                n = 0
                for u in users or {"%"}:
                    n += self.con.execute("DELETE FROM job_checkpoints WHERE run_id LIKE ?",
                                          (f"consolidate/{tenant}/{u}/%",)).rowcount
                rep.counts["checkpoints"] = n
            if mode == "purge":
                self._purge(rep)
        if self.path != ":memory:":
            rep.residue = residue(self.path, needles, blobs)
        return rep

    def _purge(self, rep: DeletionReport) -> None:
        if self.fts_secure_delete:
            rep.steps.append("FTS5 secure-delete (on since creation)")
        else:
            self.con.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
            rep.steps.append("FTS5 optimize")
        if self.path != ":memory:":
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.con.execute("VACUUM")
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            rep.steps += ["wal_checkpoint(TRUNCATE)", "VACUUM", "wal_checkpoint(TRUNCATE)"]

    def delete_ids(self, ids, *, mode: str = "logical") -> int:
        """Delete records by id (row, vector and FTS row together); ``mode="purge"`` also removes the bytes."""
        with self._lock:
            n = 0
            for rid in ids:
                row = self.con.execute("SELECT rowid FROM memories WHERE id=?", (rid,)).fetchone()
                if row:
                    self.con.execute("DELETE FROM memories_fts WHERE rowid=?", (row["rowid"],))
                    self.con.execute("DELETE FROM memories WHERE rowid=?", (row["rowid"],))
                    n += 1
            if n and mode == "purge":
                self._purge(DeletionReport("by-id", "*"))
        return n

    def expire(self, now: float | None = None, *, mode: str = "purge") -> int:
        """Forget by TTL: delete every record whose ``created_at + ttl_s`` has passed. Returns the count."""
        now = self.clock() if now is None else now
        with self._lock:
            ids = [r["id"] for r in self.con.execute(
                "SELECT id FROM memories WHERE ttl_s IS NOT NULL AND created_at + ttl_s <= ?", (now,))]
        return self.delete_ids(ids, mode=mode)

    def cap(self, tenant: str, user: str, kind: str, max_records: int, now: float | None = None,
            half_life_days: float = 30.0) -> list[str]:
        """Forget by cap: keep the ``max_records`` active records of one kind with the highest
        importance × decay (half-life on last access); delete the rest. Returns the deleted ids."""
        now = self.clock() if now is None else now
        recs = self.records(tenant, user, statuses=("active",), kind=kind)

        def keep_score(r: MemoryRecord) -> float:
            age_days = max(0.0, now - r.last_accessed) / 86400
            return r.importance * 0.5 ** (age_days / half_life_days)
        drop = sorted(recs, key=keep_score, reverse=True)[max_records:]
        self.delete_ids([r.id for r in drop])
        return [r.id for r in drop]

    def partitions_with_episodes(self, start: float, end: float) -> list[tuple[str, str]]:
        """Every (tenant, user) with active episodes created in [start, end): a consolidation work list."""
        with self._lock:
            rows = self.con.execute("SELECT DISTINCT tenant, user_id FROM memories WHERE kind='episodic' AND "
                                    "status='active' AND created_at >= ? AND created_at < ? ORDER BY 1, 2",
                                    (start, end)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def job_tables(self) -> "SQLiteJobTables":
        return SQLiteJobTables(self)

    def stats(self) -> dict:
        with self._lock:
            n = self.con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by = self.con.execute("SELECT status, COUNT(*) FROM memories GROUP BY status").fetchall()
        size = Path(self.path).stat().st_size if self.path != ":memory:" and Path(self.path).exists() else None
        return {"records": n, "by_status": {r[0]: r[1] for r in by}, "file_bytes": size,
                "fts5_secure_delete": self.fts_secure_delete}


def rescore(hits: list[Hit], weights: dict, now: float, decay_per_hour: float = 0.995) -> list[Hit]:
    """Re-rank candidates by ``w_rel·relevance + w_rec·recency + w_imp·importance``, each term min-max
    normalised over the candidates, recency = ``decay_per_hour ** hours since last access`` — the form
    the generative-agents *paper* states (weights 1, 0.995/hour; verify). Its reference *code* differs
    (rank-based recency that favours the oldest node, weights [0.5, 3, 2], normalised over the whole
    stream: facts sheet §2); memory-core implements both, this lab uses the paper's form only."""
    if not hits:
        return hits

    def norm(xs):
        lo, hi = min(xs), max(xs)
        return [0.5 if hi == lo else (x - lo) / (hi - lo) for x in xs]
    rel = norm([h.score for h in hits])
    rec = norm([decay_per_hour ** (max(0.0, now - h.record.last_accessed) / 3600) for h in hits])
    imp = norm([h.record.importance for h in hits])
    w = {"relevance": 1.0, "recency": 1.0, "importance": 1.0, **weights}
    scored = [(w["relevance"] * a + w["recency"] * b + w["importance"] * c, h) for a, b, c, h in zip(rel, rec, imp, hits)]
    out = []
    for s, h in sorted(scored, key=lambda x: -x[0]):
        h.score = s
        out.append(h)
    return out


# ---------------------------------------------------------------------------------------------- job tables
JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_leases (run_id TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL,
                                       attempts INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS job_checkpoints (run_id TEXT NOT NULL, step TEXT NOT NULL, output TEXT NOT NULL,
                                            done_at REAL NOT NULL, PRIMARY KEY (run_id, step));
CREATE TABLE IF NOT EXISTS job_runs (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, started_at REAL,
                                     finished_at REAL, report TEXT);
"""

# Take the lease if nobody holds it, if it expired, or if we already hold it — in one statement.
ACQUIRE_LEASE = """INSERT INTO job_leases (run_id, holder, expires_at) VALUES (:run_id, :holder, :expires)
ON CONFLICT (run_id) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at,
  attempts = job_leases.attempts + (job_leases.holder != excluded.holder)
WHERE job_leases.expires_at <= :now OR job_leases.holder = excluded.holder"""


class SQLiteJobTables:
    """The durable-run bookkeeping of ``consolidate.ConsolidationJob`` in the store's own file: a lease
    row, a checkpoint per step, a run row. ``PgJobTables`` is the same on Postgres."""

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store
        with store._lock:
            store.con.executescript(JOB_SCHEMA)

    def acquire(self, run: str, holder: str, now: float, ttl_s: float) -> bool:
        with self.store._lock:
            self.store.con.execute(ACQUIRE_LEASE, {"run_id": run, "holder": holder, "expires": now + ttl_s, "now": now})
            row = self.store.con.execute("SELECT holder FROM job_leases WHERE run_id = ?", (run,)).fetchone()
        return row is not None and row[0] == holder

    def lease(self, run: str):
        with self.store._lock:
            row = self.store.con.execute("SELECT holder, expires_at, attempts FROM job_leases WHERE run_id=?",
                                         (run,)).fetchone()
        return tuple(row) if row else None

    def release(self, run: str, holder: str) -> None:
        with self.store._lock:
            self.store.con.execute("DELETE FROM job_leases WHERE run_id=? AND holder=?", (run, holder))

    def checkpoint(self, run: str, step: str):
        with self.store._lock:
            row = self.store.con.execute("SELECT output FROM job_checkpoints WHERE run_id=? AND step=?",
                                         (run, step)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, run: str, step: str, output, now: float) -> None:
        with self.store._lock:
            self.store.con.execute("INSERT OR REPLACE INTO job_checkpoints VALUES (?,?,?,?)",
                                   (run, step, json.dumps(output), now))

    def steps(self, run: str) -> list[str]:
        with self.store._lock:
            return [r[0] for r in self.store.con.execute(
                "SELECT step FROM job_checkpoints WHERE run_id=? ORDER BY done_at, rowid", (run,))]

    def start(self, run: str, now: float) -> str:
        """Record the run (once) and return its status: ``running`` or ``done``."""
        with self.store._lock:
            self.store.con.execute("INSERT OR IGNORE INTO job_runs (run_id, status, started_at) VALUES (?, 'running', ?)",
                                   (run, now))
            return self.store.con.execute("SELECT status FROM job_runs WHERE run_id=?", (run,)).fetchone()[0]

    def finish(self, run: str, report: dict, now: float) -> None:
        with self.store._lock:
            self.store.con.execute("UPDATE job_runs SET status='done', finished_at=?, report=? WHERE run_id=?",
                                   (now, json.dumps(report), run))

    def clear(self, run: str) -> int:
        """Drop a finished run's checkpoints: they hold the text it extracted, and nothing needs them now."""
        with self.store._lock:
            return self.store.con.execute("DELETE FROM job_checkpoints WHERE run_id=?", (run,)).rowcount

