"""The SQLite store: partitions, the three rankings, as-of reads, idempotency, and forgetting down to the bytes."""
import math
import sqlite3

import pytest

from memlab.deletion import residue
from memlab.records import MemoryRecord
from memlab.store import SQLiteMemoryStore, pack, rrf, sqlite_features
from memlab.store.sqlite import STOPWORDS, fts_query


def fill(store):
    for tenant, user, text, slot in [("acme", "u1", "Home city: the user lives in Lisbon.", "home_city"),
                                     ("acme", "u1", "Trip: the user took a trip to Kyoto.", "trip"),
                                     ("acme", "u1", "Employer: the user works at Globex.", "employer"),
                                     ("acme", "u2", "Home city: the user lives in Oslo.", "home_city"),
                                     ("globex", "u1", "Home city: the user lives in Lima.", "home_city")]:
        store.add(MemoryRecord(tenant, user, text, slot=slot, created_at=store.clock()))


def test_features_are_detected_not_assumed():
    f = sqlite_features()
    assert f["fts5"] is True and set(f) >= {"sqlite_version", "fts5_secure_delete"}


def test_partition_is_enforced_by_every_query(mem_store):
    fill(mem_store)
    q = "Which city is my home city?"
    for mode in ("vector", "fts", "hybrid"):
        texts = {h.record.text for h in mem_store.search("acme", "u1", q, 5, mode=mode)}
        assert texts and not any("Oslo" in t or "Lima" in t for t in texts), (mode, texts)
    assert [h.record.text for h in mem_store.search("globex", "u1", q, 5)] == ["Home city: the user lives in Lima."]
    assert mem_store.search("nobody", "u1", q, 5) == []


def test_vectors_are_float32_blobs_on_8k_pages(store):
    fill(store)
    row = store.con.execute("SELECT length(embedding), dim FROM memories LIMIT 1").fetchone()
    assert tuple(row) == (4096, 1024)
    assert store.con.execute("PRAGMA page_size").fetchone()[0] == 8192


def test_rrf_pins_and_hybrid_equals_rrf_of_the_two_rankings(mem_store):
    assert rrf([["a", "b"], ["a", "c"]])[0] == ("a", 2 / 61)
    assert math.isclose(dict(rrf([["a", "b"], ["c", "b"]]))["b"], 2 / 62)
    fill(mem_store)
    hits = mem_store.search("acme", "u1", "home city and my trip", 5, mode="hybrid")
    for h in hits:
        want = sum(1 / (60 + r + 1) for r in (h.rank_vector, h.rank_fts) if r is not None)
        assert math.isclose(h.score, want)


def test_bm25_is_negative_and_the_match_expression_is_safe(mem_store):
    fill(mem_store)
    hits = mem_store.search("acme", "u1", "trip Kyoto", 3, mode="fts")
    assert hits[0].record.slot == "trip" and hits[0].bm25 < 0 and hits[0].score == -hits[0].bm25
    assert fts_query('what is my "home" city* NEAR(x)') == '"home" OR "city" OR "near" OR "x"'
    assert "my" in STOPWORDS and fts_query("what is my") == ""
    assert mem_store.search("acme", "u1", "what is my", 3, mode="fts") == []


def test_porter_stemming_finds_plurals(mem_store):
    fill(mem_store)
    assert mem_store.search("acme", "u1", "how many trips", 3, mode="fts")[0].record.slot == "trip"


def test_idempotent_add_and_immutable_text(mem_store):
    r = MemoryRecord("acme", "u1", "Pet: the user has a dog named Rex.", slot="pet")
    rid, created = mem_store.add(r, idempotency_key="k1")
    rid2, created2 = mem_store.add(MemoryRecord("acme", "u1", "Pet: the user has a dog named Rex.", slot="pet"),
                                   idempotency_key="k1")
    assert created and not created2 and rid == rid2 and mem_store.stats()["records"] == 1
    with pytest.raises(ValueError):
        mem_store.set_fields(rid, text="something else")


def test_as_of_reads_superseded_facts(mem_store, clock):
    old = MemoryRecord("acme", "u1", "Home city: the user lives in Lisbon.", slot="home_city", created_at=100)
    new = MemoryRecord("acme", "u1", "Home city: the user lives in Porto.", slot="home_city", created_at=200)
    mem_store.add(old)
    mem_store.add(new)
    mem_store.supersede(old.id, at=200)
    now = {h.record.text for h in mem_store.search("acme", "u1", "home city", 5)}
    then = {h.record.text for h in mem_store.search("acme", "u1", "home city", 5, as_of=150)}
    assert now == {new.text} and then == {old.text}


def test_pack_respects_the_budget(mem_store):
    fill(mem_store)
    hits = mem_store.search("acme", "u1", "home city trip employer", 5)
    assert sum(h.tokens for h in pack(hits, 9)) <= 9 and len(pack(hits, 0)) == 0


def test_rescore_weights_move_importance_up(mem_store):
    a = MemoryRecord("acme", "u1", "Allergy: the user is allergic to peanuts.", importance=9, created_at=1)
    b = MemoryRecord("acme", "u1", "Note: the user is fond of peanuts in salads.", importance=1, created_at=1)
    mem_store.add(a)
    mem_store.add(b)
    top = mem_store.search("acme", "u1", "peanuts", 2, weights={"relevance": 0.1, "recency": 0, "importance": 3})[0]
    assert top.record.id == a.id


@pytest.mark.parametrize("secure", [False, True])
def test_logical_forget_leaves_bytes_and_purge_removes_them(tmp_path, clock, secure):
    if secure and not sqlite_features()["fts5_secure_delete"]:
        pytest.skip("this SQLite has no FTS5 secure-delete")
    s = SQLiteMemoryStore(tmp_path / f"m{secure}.db", clock=clock, fts_secure_delete=secure)
    for i in range(30):
        s.add(MemoryRecord("acme", "u1", f"Note {i}: weather and lunch.", kind="episodic"))
    s.add(MemoryRecord("acme", "u1", "Home address: the user lives at 12 Rua das Zanzibarflores.", slot="address"))
    rep = s.forget("acme", "acme/u1/address", mode="logical", needles=["Zanzibarflores"])
    assert rep.counts["records"] == 1 and rep.residue["Zanzibarflores"] > 0 and not rep.clean
    assert s.search("acme", "u1", "Zanzibarflores", 5) == []
    rep = s.forget("acme", "acme/u1/address", needles=["Zanzibarflores"])
    assert residue(s.path, ["Zanzibarflores"]) == {"Zanzibarflores": 0}
    assert rep.steps[-1] == "wal_checkpoint(TRUNCATE)" and ("FTS5 optimize" in rep.steps) != secure


def test_forget_follows_provenance_and_counts_the_vector(tmp_path, clock):
    s = SQLiteMemoryStore(tmp_path / "p.db", clock=clock)
    ep = MemoryRecord("acme", "u1", "On 2026-09-10 the user said: my address is 12 Rua das Flores.", kind="episodic",
                      deletion_key="acme/u1/address")
    s.add(ep)
    summ = MemoryRecord("acme", "u1", "Summary: the user shared 12 Rua das Flores.", provenance=[ep.id],
                        deletion_key="acme/u1/summary")
    s.add(summ)
    s.add(MemoryRecord("acme", "u1", "Second-order: cites the summary.", provenance=[summ.id], deletion_key="x"))
    blob = bytes(s.con.execute("SELECT embedding FROM memories WHERE id=?", (ep.id,)).fetchone()[0])
    rep = s.forget("acme", "acme/u1/address", needles=["Flores"])
    assert rep.counts["records"] == 1 and rep.counts["derived"] == 2 and rep.clean
    assert residue(s.path, blobs=[blob]) == {"<vector #0, 4096 bytes>": 0}
    assert s.forget("acme", "acme/u1/summary", include_derived=False).counts["records"] == 0


def test_ttl_and_cap_forget(mem_store, clock):
    mem_store.add(MemoryRecord("acme", "u1", "Short-lived note.", kind="episodic", created_at=0, ttl_s=10))
    mem_store.add(MemoryRecord("acme", "u1", "Durable fact.", created_at=0))
    clock.t = 11
    assert mem_store.expire() == 1 and [r.text for r in mem_store.records("acme", "u1")] == ["Durable fact."]
    for i in range(5):
        mem_store.add(MemoryRecord("acme", "u1", f"Episode {i}", kind="episodic", importance=i, created_at=11))
    dropped = mem_store.cap("acme", "u1", "episodic", 2)
    left = sorted(r.importance for r in mem_store.records("acme", "u1", kind="episodic"))
    assert len(dropped) == 3 and left == [3.0, 4.0]


def test_no_fts5_is_a_clear_error(monkeypatch):
    import memlab.store.sqlite as S
    monkeypatch.setattr(S, "sqlite_features", lambda: {"fts5": False, "fts5_secure_delete": False,
                                                       "sqlite_version": "0"})
    with pytest.raises(RuntimeError, match="FTS5"):
        S.SQLiteMemoryStore(":memory:")


def test_raw_fts5_bm25_formula_matches_sqlite():
    import re
    docs = ["the user lives in lisbon", "a dog named rex", "green tea with the dog", "lisbon lisbon trams"]
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='unicode61')")
    con.executemany("INSERT INTO t(x) VALUES (?)", [(d,) for d in docs])
    toks = [re.findall(r"[a-z0-9]+", d) for d in docs]
    avg = sum(map(len, toks)) / len(toks)
    for rowid, got in con.execute("SELECT rowid, bm25(t) FROM t WHERE t MATCH 'dog'"):
        d = toks[rowid - 1]
        n = sum("dog" in x for x in toks)
        idf = max(math.log((len(toks) - n + 0.5) / (n + 0.5)), 1e-6)
        want = -idf * d.count("dog") * 2.2 / (d.count("dog") + 1.2 * (0.25 + 0.75 * len(d) / avg))
        assert math.isclose(got, want, rel_tol=1e-9)
