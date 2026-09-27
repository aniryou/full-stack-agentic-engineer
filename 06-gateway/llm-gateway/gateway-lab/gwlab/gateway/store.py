"""sqlite3 for the gateway's three tables: virtual keys (hashed), the response cache, the ledger.

The one idea: the gateway's state is small and boring, and it should look boring — three tables you can open
with `sqlite3 gateway.db` and read. A production gateway keeps the keys and the ledger in a database (LiteLLM:
Postgres) and the buckets and cache in Redis; the shape is the same. One connection, guarded by a lock, so the
gateway's event loop and a notebook's thread can both read it.
"""
from __future__ import annotations

import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
  key_id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL, display TEXT NOT NULL, tenant TEXT NOT NULL,
  aliases TEXT, rpm INTEGER, tpm INTEGER, budget_usd REAL, spent_usd REAL NOT NULL DEFAULT 0,
  created REAL NOT NULL, expires_at REAL, revoked_at REAL);
CREATE TABLE IF NOT EXISTS ledger (
  request_id TEXT PRIMARY KEY, ts REAL, tenant TEXT, key_id TEXT, alias TEXT, target TEXT, provider TEXT,
  status INTEGER, stream INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER,
  reasoning_tokens INTEGER, usage_source TEXT, cost_usd REAL, ttft_ms REAL, e2e_ms REAL, cache TEXT,
  attempts INTEGER, error TEXT, finish_reason TEXT, trace_id TEXT);
CREATE TABLE IF NOT EXISTS cache (
  ns TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL, query TEXT, entities TEXT, vec BLOB,
  response TEXT NOT NULL, created REAL NOT NULL, expires_at REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (ns, key, kind));
CREATE INDEX IF NOT EXISTS ledger_tenant ON ledger(tenant);
CREATE INDEX IF NOT EXISTS cache_ns ON cache(ns, kind);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def execute(self, sql: str, params=()) -> int:
        with self.lock:
            return self.db.execute(sql, params).rowcount

    def query(self, sql: str, params=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    def close(self) -> None:
        with self.lock:
            self.db.close()
