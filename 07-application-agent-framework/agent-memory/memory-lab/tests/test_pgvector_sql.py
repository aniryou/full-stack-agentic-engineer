"""The pgvector twin, checked offline: every statement parses with Postgres's own parser, scope is in every
per-user query, placeholders are named, and the pgvector facts it relies on hold."""
import re

import pytest

from memlab.store import pgvector as P

STATEMENTS = P.statements()


@pytest.mark.parametrize("name", sorted(STATEMENTS))
def test_statement_parses_with_postgres_parser(name):
    pglast = pytest.importorskip("pglast")
    sql, params = P.to_positional(STATEMENTS[name])
    tree = pglast.parse_sql(sql)
    assert len(tree) == 1, f"{name}: one statement per call"
    assert "%" not in sql or "%%" in sql, f"{name}: a psycopg placeholder was not converted"


def test_placeholders_are_named_and_nothing_is_interpolated():
    for name, sql in STATEMENTS.items():
        assert "%s" not in sql and "{" not in sql and "}" not in sql, name


@pytest.mark.parametrize("name", ["search", "forget", "records", "touch", "insert", "existing"])
def test_scope_is_in_every_per_user_statement(name):
    sql = STATEMENTS[name]
    assert "%(tenant)s" in sql, name
    if name == "search":
        assert "user_id IN (%(user_id)s, '*')" in sql


def test_hybrid_search_is_rrf_k60_over_cosine_and_full_text():
    s = STATEMENTS["search"]
    assert "embedding <=> %(qvec)s::vector" in s and "to_tsquery('english', %(tsq)s)" in s
    assert s.count(f"1.0 / ({P.RRF_K} + ") == 2 and P.RRF_K == 60
    assert "ORDER BY score DESC" in s


def test_ddl_facts():
    ddl = " ".join(P.ddl(1024))
    assert "vector(1024)" in ddl and "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in ddl
    assert "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in ddl and "USING gin (fts)" in ddl
    with pytest.raises(ValueError, match="2000"):
        P.ddl(3072)
    assert P.storage_bytes(384) == 1544 and P.storage_bytes(1024) == 4104
    assert STATEMENTS["iterative_scan"] == "SET hnsw.iterative_scan = relaxed_order"


def test_idempotency_is_unique_per_partition():
    ddl = " ".join(P.ddl(1024))
    assert "UNIQUE (tenant, user_id, idempotency_key)" in ddl and "idempotency_key text UNIQUE" not in ddl
    assert "ON CONFLICT (tenant, user_id, idempotency_key) DO NOTHING" in STATEMENTS["insert"]
    assert "user_id = %(user_id)s" in STATEMENTS["existing"]


def test_forget_is_recursive_over_provenance():
    f = STATEMENTS["forget"]
    assert f.startswith("WITH RECURSIVE doomed AS") and "m.provenance ? d.id" in f and "RETURNING id" in f


def test_lease_upsert_matches_the_sqlite_semantics():
    a = STATEMENTS["acquire_lease"]
    assert "ON CONFLICT (run_id) DO UPDATE" in a and "WHERE job_leases.expires_at <= %(now)s OR job_leases.holder = EXCLUDED.holder" in a
    assert a.rstrip().endswith("RETURNING holder")


def test_helpers():
    assert P.tsquery("What is my home city?") == "home | city"
    assert P.vector_literal([0.5, 0.25]) == "[0.5,0.25]"
    assert P.to_positional("a = %(x)s AND b = %(y)s OR c = %(x)s") == ("a = $1 AND b = $2 OR c = $1", ["x", "y"])


def test_psycopg_is_lazy():
    try:
        import psycopg  # noqa: F401
        pytest.skip("psycopg is installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="psycopg"):
        P.PgVectorStore("postgresql://nobody@127.0.0.1:1/none")
