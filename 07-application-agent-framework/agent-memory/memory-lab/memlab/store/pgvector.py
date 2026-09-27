"""pgvector.py — the same memory store on Postgres + pgvector: one schema, the same three queries, a different forget.

The one idea: moving the store from a file to a server changes the operations, not the design. The
record is the same row; the vector is a ``vector(1024)`` column; the full-text index is a generated
``tsvector`` (the ``english`` configuration: stemmed, like the SQLite store's porter tokenizer) with a GIN index; hybrid search is RRF (k = 60) written in SQL; the partition predicate
``tenant = … AND user_id IN (…, '*')`` is in every statement. Two things differ and are the lesson:

* **Filtered ANN.** pgvector applies a WHERE filter *after* the HNSW index scan: "if a condition matches
  10% of rows, with HNSW and the default ``hnsw.ef_search`` of 40, only 4 rows will match on average"
  (pgvector README, 0.8.6). A per-user partition is small, so these queries scan it exactly (the CTE
  filters first); the HNSW index (m = 16, ef_construction = 64) pays only for tenant-wide searches, with
  ``hnsw.iterative_scan = relaxed_order`` (added in 0.8.0) so filtered queries keep scanning.
* **Forgetting.** ``DELETE`` leaves dead tuples; ``VACUUM`` makes the space reusable but Postgres has no
  page-level secure delete (unverified: standard Postgres behaviour, not in pgvector's sources), so the
  bytes can stay in heap and index pages until overwritten, in WAL segments until recycled, and in
  backups until they expire. ``VACUUM FULL`` rewrites the table (and takes an exclusive lock).

psycopg is imported lazily (``pip install "psycopg[binary]"``); vectors are sent as ``'[…]'::vector``
text, so ``pgvector-python`` is optional. Every statement is checked offline by Postgres's own parser
(``pglast``) in the tests. Image: ``pgvector/pgvector:0.8.6-pg17`` (deploy/local/compose.yaml).
"""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np

from ..deletion import DeletionReport
from ..embedders import HashingEmbedder, tokenize
from ..records import MemoryRecord
from .sqlite import RRF_K, STOPWORDS, Hit

DIM = 1024
INDEXABLE_MAX_DIM = 2000        # pgvector: `vector` indexes up to 2,000 dims (halfvec 4,000)


def ddl(dim: int = DIM) -> list[str]:
    if dim > INDEXABLE_MAX_DIM:
        raise ValueError(f"pgvector indexes `vector` up to {INDEXABLE_MAX_DIM} dims; use halfvec for {dim}")
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS memories (
  id              text PRIMARY KEY,
  tenant          text NOT NULL,
  user_id         text NOT NULL,
  scope           text NOT NULL,
  session_id      text,
  agent           text,
  kind            text NOT NULL,
  text            text NOT NULL,
  source          text NOT NULL,
  trust           text NOT NULL,
  slot            text,
  value           text,
  provenance      jsonb NOT NULL DEFAULT '[]',
  confidence      real NOT NULL,
  importance      real NOT NULL,
  created_at      double precision NOT NULL,
  valid_from      double precision NOT NULL,
  valid_to        double precision,
  superseded_at   double precision,
  ttl_s           double precision,
  last_accessed   double precision NOT NULL,
  deletion_key    text NOT NULL,
  status          text NOT NULL,
  idempotency_key text UNIQUE,
  embedding       vector({dim}) NOT NULL,
  fts             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
)""",
        "CREATE INDEX IF NOT EXISTS memories_partition ON memories (tenant, user_id, status)",
        "CREATE INDEX IF NOT EXISTS memories_deletion ON memories (tenant, deletion_key)",
        "CREATE INDEX IF NOT EXISTS memories_fts ON memories USING gin (fts)",
        "CREATE INDEX IF NOT EXISTS memories_hnsw ON memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)",
    ]


INSERT = """INSERT INTO memories (id, tenant, user_id, scope, session_id, agent, kind, text, source, trust, slot, value,
  provenance, confidence, importance, created_at, valid_from, valid_to, superseded_at, ttl_s, last_accessed,
  deletion_key, status, idempotency_key, embedding)
VALUES (%(id)s, %(tenant)s, %(user_id)s, %(scope)s, %(session_id)s, %(agent)s, %(kind)s, %(text)s, %(source)s,
  %(trust)s, %(slot)s, %(value)s, %(provenance)s::jsonb, %(confidence)s, %(importance)s, %(created_at)s,
  %(valid_from)s, %(valid_to)s, %(superseded_at)s, %(ttl_s)s, %(last_accessed)s, %(deletion_key)s, %(status)s,
  %(idempotency_key)s, %(embedding)s::vector)
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING id"""

EXISTING_BY_KEY = "SELECT id FROM memories WHERE idempotency_key = %(idempotency_key)s"

# Hybrid search: the partition first (exact within one user), then cosine and full-text rankings, fused by RRF.
SEARCH = f"""WITH part AS (
  SELECT * FROM memories
  WHERE tenant = %(tenant)s AND user_id IN (%(user_id)s, '*') AND status = ANY(%(statuses)s)
    AND (scope <> 'session' OR session_id = %(session_id)s)
),
vec AS (
  SELECT id, 1 - (embedding <=> %(qvec)s::vector) AS cosine,
         row_number() OVER (ORDER BY embedding <=> %(qvec)s::vector) - 1 AS rank
  FROM part ORDER BY embedding <=> %(qvec)s::vector LIMIT %(depth)s
),
fts AS (
  SELECT id, ts_rank_cd(fts, q) AS text_score,
         row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) - 1 AS rank
  FROM part, to_tsquery('english', %(tsq)s) AS q
  WHERE fts @@ q ORDER BY text_score DESC LIMIT %(depth)s
)
SELECT p.id, p.tenant, p.user_id, p.scope, p.session_id, p.agent, p.kind, p.text, p.source, p.trust, p.slot,
       p.value, p.provenance, p.confidence, p.importance, p.created_at, p.valid_from, p.valid_to,
       p.superseded_at, p.ttl_s, p.last_accessed, p.deletion_key, p.status,
       COALESCE(1.0 / ({RRF_K} + vec.rank + 1), 0) + COALESCE(1.0 / ({RRF_K} + fts.rank + 1), 0) AS score,
       vec.cosine, fts.text_score, vec.rank AS rank_vector, fts.rank AS rank_fts
FROM part p LEFT JOIN vec ON vec.id = p.id LEFT JOIN fts ON fts.id = p.id
WHERE vec.id IS NOT NULL OR fts.id IS NOT NULL
ORDER BY score DESC, p.id
LIMIT %(k)s"""

TOUCH = "UPDATE memories SET last_accessed = %(now)s WHERE tenant = %(tenant)s AND id = ANY(%(ids)s)"

# Forget: the records with the key, then everything whose provenance cites them, transitively.
FORGET = """WITH RECURSIVE doomed AS (
  SELECT id FROM memories WHERE tenant = %(tenant)s AND deletion_key = %(deletion_key)s
  UNION
  SELECT m.id FROM memories m JOIN doomed d ON m.provenance ? d.id WHERE m.tenant = %(tenant)s
)
DELETE FROM memories WHERE tenant = %(tenant)s AND id IN (SELECT id FROM doomed)
RETURNING id, deletion_key = %(deletion_key)s AS direct"""

# What a purge can do on Postgres, in order; none of it reaches WAL archives or backups.
PURGE = [
    "VACUUM (VERBOSE) memories",
    "REINDEX INDEX CONCURRENTLY memories_hnsw",   # the pgvector README's advice when vacuuming HNSW is slow
]
PURGE_FULL = "VACUUM FULL memories"              # rewrites the table; takes an ACCESS EXCLUSIVE lock
ITERATIVE_SCAN = "SET hnsw.iterative_scan = relaxed_order"


RECORD_COLS = ("id, tenant, user_id, scope, session_id, agent, kind, text, source, trust, slot, value, provenance, "
               "confidence, importance, created_at, valid_from, valid_to, superseded_at, ttl_s, last_accessed, "
               "deletion_key, status")
RECORDS = f"""SELECT {RECORD_COLS} FROM memories
WHERE tenant = %(tenant)s
  AND (%(user_id)s::text IS NULL OR user_id = %(user_id)s)
  AND (%(statuses)s::text[] IS NULL OR status = ANY(%(statuses)s))
  AND (%(kind)s::text IS NULL OR kind = %(kind)s)
  AND (%(slot)s::text IS NULL OR slot = %(slot)s)
ORDER BY created_at, id"""
GET = f"SELECT {RECORD_COLS} FROM memories WHERE id = %(id)s"
SET_FIELDS = """UPDATE memories SET status = COALESCE(%(status)s, status), valid_to = COALESCE(%(valid_to)s, valid_to),
  superseded_at = COALESCE(%(superseded_at)s, superseded_at), last_accessed = COALESCE(%(last_accessed)s, last_accessed),
  ttl_s = COALESCE(%(ttl_s)s, ttl_s), importance = COALESCE(%(importance)s, importance),
  confidence = COALESCE(%(confidence)s, confidence)
WHERE id = %(id)s"""
PARTITIONS = """SELECT DISTINCT tenant, user_id FROM memories
WHERE kind = 'episodic' AND status = 'active' AND created_at >= %(start)s AND created_at < %(end)s ORDER BY 1, 2"""

# The consolidation job's durable-run tables (the SQLite versions are in sqlite.py).
JOB_DDL = [
    "CREATE TABLE IF NOT EXISTS job_leases (run_id text PRIMARY KEY, holder text NOT NULL, "
    "expires_at double precision NOT NULL, attempts integer NOT NULL DEFAULT 1)",
    "CREATE TABLE IF NOT EXISTS job_checkpoints (run_id text NOT NULL, step text NOT NULL, output jsonb NOT NULL, "
    "done_at double precision NOT NULL, PRIMARY KEY (run_id, step))",
    "CREATE TABLE IF NOT EXISTS job_runs (run_id text PRIMARY KEY, status text NOT NULL, "
    "started_at double precision, finished_at double precision, report jsonb)",
]
ACQUIRE_LEASE = """INSERT INTO job_leases (run_id, holder, expires_at) VALUES (%(run_id)s, %(holder)s, %(expires)s)
ON CONFLICT (run_id) DO UPDATE SET holder = EXCLUDED.holder, expires_at = EXCLUDED.expires_at,
  attempts = job_leases.attempts + CASE WHEN job_leases.holder <> EXCLUDED.holder THEN 1 ELSE 0 END
WHERE job_leases.expires_at <= %(now)s OR job_leases.holder = EXCLUDED.holder
RETURNING holder"""
LEASE = "SELECT holder, expires_at, attempts FROM job_leases WHERE run_id = %(run_id)s"
RELEASE = "DELETE FROM job_leases WHERE run_id = %(run_id)s AND holder = %(holder)s"
CHECKPOINT_GET = "SELECT output FROM job_checkpoints WHERE run_id = %(run_id)s AND step = %(step)s"
CHECKPOINT_SAVE = """INSERT INTO job_checkpoints (run_id, step, output, done_at) VALUES (%(run_id)s, %(step)s, %(output)s::jsonb, %(now)s)
ON CONFLICT (run_id, step) DO UPDATE SET output = EXCLUDED.output, done_at = EXCLUDED.done_at"""
STEPS = "SELECT step FROM job_checkpoints WHERE run_id = %(run_id)s ORDER BY done_at, step"
RUN_START = """INSERT INTO job_runs (run_id, status, started_at) VALUES (%(run_id)s, 'running', %(now)s)
ON CONFLICT (run_id) DO NOTHING"""
RUN_STATUS = "SELECT status FROM job_runs WHERE run_id = %(run_id)s"
RUN_FINISH = "UPDATE job_runs SET status = 'done', finished_at = %(now)s, report = %(report)s::jsonb WHERE run_id = %(run_id)s"
CHECKPOINT_CLEAR = "DELETE FROM job_checkpoints WHERE run_id = %(run_id)s"


def statements(dim: int = DIM) -> dict[str, str]:
    """Every statement this module sends, by name (the tests parse each one with Postgres's parser)."""
    out = {f"ddl_{i}": s for i, s in enumerate(ddl(dim))}
    out.update(insert=INSERT, existing=EXISTING_BY_KEY, search=SEARCH, touch=TOUCH, forget=FORGET,
               purge_full=PURGE_FULL, iterative_scan=ITERATIVE_SCAN, records=RECORDS, get=GET, set_fields=SET_FIELDS,
               partitions=PARTITIONS, acquire_lease=ACQUIRE_LEASE, lease=LEASE, release=RELEASE,
               checkpoint_get=CHECKPOINT_GET, checkpoint_save=CHECKPOINT_SAVE, steps=STEPS, run_start=RUN_START,
               run_status=RUN_STATUS, run_finish=RUN_FINISH, checkpoint_clear=CHECKPOINT_CLEAR)
    out.update({f"purge_{i}": s for i, s in enumerate(PURGE)})
    out.update({f"job_ddl_{i}": s for i, s in enumerate(JOB_DDL)})
    return out


def to_positional(sql: str) -> tuple[str, list[str]]:
    """psycopg's ``%(name)s`` placeholders as Postgres ``$n`` (for the offline parser): returns the SQL
    and the parameter names in ``$n`` order. The same name reuses its number."""
    names: list[str] = []

    def sub(m):
        n = m.group(1)
        if n not in names:
            names.append(n)
        return f"${names.index(n) + 1}"
    return re.sub(r"%\((\w+)\)s", sub, sql), names


def tsquery(text: str) -> str:
    """``[a-z0-9]+`` tokens minus stopwords, OR-ed for ``to_tsquery('english', …)`` (stemmed, like the SQLite
    store's porter tokenizer) — passed as a parameter, never interpolated."""
    return " | ".join(t for t in dict.fromkeys(tokenize(text)) if t not in STOPWORDS)


def vector_literal(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(v).reshape(-1)) + "]"


def storage_bytes(dim: int) -> int:
    """pgvector's on-disk size of one ``vector(dim)`` value: 4 × dim + 8 (README)."""
    return 4 * dim + 8


class PgVectorStore:
    """The SQLite store's interface on Postgres + pgvector (T0 + Docker). ``dsn`` like
    ``postgresql://memlab:memlab@127.0.0.1:5432/memlab`` (``MEMLAB_PG_DSN``)."""

    def __init__(self, dsn: str, embedder=None, dim: int = DIM, clock=time.time):
        try:
            import psycopg  # noqa: F401  (lazy: T0 never needs it)
        except ImportError as e:
            raise ImportError('Postgres needs psycopg: pip install "psycopg[binary]>=3.1"') from e
        import psycopg
        from psycopg.rows import dict_row
        self.embedder = embedder or HashingEmbedder(dim)
        self.dim = dim
        self.clock = clock
        self._lock = threading.RLock()
        self.con = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

    def init(self) -> None:
        for s in ddl(self.dim) + JOB_DDL:
            self.con.execute(s)

    @staticmethod
    def _rec(r: dict) -> MemoryRecord:
        return MemoryRecord(**{"tenant": r["tenant"], "user": r["user_id"], "session": r["session_id"],
                               **{c: r[c] for c in ("text", "kind", "scope", "source", "trust", "agent", "slot", "value",
                                                    "confidence", "importance", "created_at", "valid_from", "valid_to",
                                                    "superseded_at", "ttl_s", "last_accessed", "deletion_key", "status",
                                                    "id")},
                               "provenance": list(r["provenance"] or [])})

    def records(self, tenant: str, user: str | None = None, *, statuses=None, kind: str | None = None,
                slot: str | None = None) -> list[MemoryRecord]:
        rows = self.con.execute(RECORDS, {"tenant": tenant, "user_id": user,
                                          "statuses": list(statuses) if statuses else None,
                                          "kind": kind, "slot": slot}).fetchall()
        return [self._rec(r) for r in rows]

    def get(self, rec_id: str) -> MemoryRecord | None:
        r = self.con.execute(GET, {"id": rec_id}).fetchone()
        return self._rec(r) if r else None

    def set_fields(self, rec_id: str, **changes) -> None:
        allowed = ("status", "valid_to", "superseded_at", "last_accessed", "ttl_s", "importance", "confidence")
        bad = set(changes) - set(allowed)
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}; write a new record and supersede instead")
        self.con.execute(SET_FIELDS, {"id": rec_id, **{k: changes.get(k) for k in allowed}})

    def supersede(self, old_id: str, at: float, now: float | None = None) -> None:
        self.set_fields(old_id, status="superseded", valid_to=at, superseded_at=self.clock() if now is None else now)

    def partitions_with_episodes(self, start: float, end: float) -> list[tuple[str, str]]:
        return [(r["tenant"], r["user_id"]) for r in self.con.execute(PARTITIONS, {"start": start, "end": end})]

    def job_tables(self) -> "PgJobTables":
        return PgJobTables(self)

    def add(self, rec: MemoryRecord, *, idempotency_key: str | None = None) -> tuple[str, bool]:
        vec = self.embedder.encode(rec.text)
        row = self.con.execute(INSERT, {
            "id": rec.id, "tenant": rec.tenant, "user_id": rec.user, "scope": rec.scope, "session_id": rec.session,
            "agent": rec.agent, "kind": rec.kind, "text": rec.text, "source": rec.source, "trust": rec.trust,
            "slot": rec.slot, "value": rec.value, "provenance": json.dumps(rec.provenance),
            "confidence": rec.confidence, "importance": rec.importance, "created_at": rec.created_at,
            "valid_from": rec.valid_from, "valid_to": rec.valid_to, "superseded_at": rec.superseded_at,
            "ttl_s": rec.ttl_s, "last_accessed": rec.last_accessed, "deletion_key": rec.deletion_key,
            "status": rec.status, "idempotency_key": idempotency_key, "embedding": vector_literal(vec)}).fetchone()
        if row:
            return row["id"], True
        return self.con.execute(EXISTING_BY_KEY, {"idempotency_key": idempotency_key}).fetchone()["id"], False

    def search(self, tenant: str, user: str, query: str, k: int = 5, *, statuses=("active",),
               session: str | None = None, now: float | None = None, touch: bool = True) -> list[Hit]:
        rows = self.con.execute(SEARCH, {
            "tenant": tenant, "user_id": user, "statuses": list(statuses), "session_id": session,
            "qvec": vector_literal(self.embedder.encode(query)), "tsq": tsquery(query) or "nothing",
            "depth": max(4 * k, 20), "k": k}).fetchall()
        hits = [Hit(self._rec(r), float(r["score"]), r["cosine"], None, r["rank_vector"], r["rank_fts"]) for r in rows]
        if touch and hits and now is not None:
            self.con.execute(TOUCH, {"now": now, "tenant": tenant, "ids": [h.record.id for h in hits]})
        return hits

    def forget(self, tenant: str, deletion_key: str, *, full: bool = False) -> DeletionReport:
        rep = DeletionReport(deletion_key, tenant)
        rows = self.con.execute(FORGET, {"tenant": tenant, "deletion_key": deletion_key}).fetchall()
        direct = sum(1 for r in rows if r["direct"])
        rep.deleted_ids = sorted(r["id"] for r in rows)
        rep.counts.update(records=direct, vectors=len(rows), fts_rows=len(rows), derived=len(rows) - direct)
        rep.steps.append("DELETE (recursive over provenance)")
        for s in PURGE:
            self.con.execute(s)
            rep.steps.append(s)
        if full:
            self.con.execute(PURGE_FULL)
            rep.steps.append(PURGE_FULL)
        rep.not_reachable += ["WAL segments until recycled (and WAL archives)", "base backups until they expire",
                              "free space in heap/index pages until overwritten (no page-level secure delete)"]
        return rep

    def close(self) -> None:
        self.con.close()


class PgJobTables:
    """``SQLiteJobTables`` on Postgres: the same lease / checkpoint / run bookkeeping, the same semantics."""

    def __init__(self, store: PgVectorStore):
        self.con = store.con
        for s in JOB_DDL:
            self.con.execute(s)

    def acquire(self, run: str, holder: str, now: float, ttl_s: float) -> bool:
        row = self.con.execute(ACQUIRE_LEASE, {"run_id": run, "holder": holder, "expires": now + ttl_s,
                                               "now": now}).fetchone()
        return row is not None and row["holder"] == holder

    def lease(self, run: str):
        r = self.con.execute(LEASE, {"run_id": run}).fetchone()
        return (r["holder"], r["expires_at"], r["attempts"]) if r else None

    def release(self, run: str, holder: str) -> None:
        self.con.execute(RELEASE, {"run_id": run, "holder": holder})

    def checkpoint(self, run: str, step: str):
        r = self.con.execute(CHECKPOINT_GET, {"run_id": run, "step": step}).fetchone()
        return r["output"] if r else None

    def save(self, run: str, step: str, output, now: float) -> None:
        self.con.execute(CHECKPOINT_SAVE, {"run_id": run, "step": step, "output": json.dumps(output), "now": now})

    def steps(self, run: str) -> list[str]:
        return [r["step"] for r in self.con.execute(STEPS, {"run_id": run})]

    def start(self, run: str, now: float) -> str:
        self.con.execute(RUN_START, {"run_id": run, "now": now})
        return self.con.execute(RUN_STATUS, {"run_id": run}).fetchone()["status"]

    def finish(self, run: str, report: dict, now: float) -> None:
        self.con.execute(RUN_FINISH, {"run_id": run, "now": now, "report": json.dumps(report)})

    def clear(self, run: str) -> int:
        return self.con.execute(CHECKPOINT_CLEAR, {"run_id": run}).rowcount
