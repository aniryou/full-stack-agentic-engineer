# %% [markdown]
# # 01 · A memory store on SQLite: records, vectors, full text — and a delete that removes the bytes
#
# **Tier:** T0 — SQLite with FTS5 ships with Python, vectors are numpy arrays in BLOB columns, the embedder
# is a hashing embedder; nothing is downloaded. **T0 + Docker:** the last section runs the same store on
# Postgres + pgvector when `MEMLAB_PG_DSN` points at one (`deploy/local/up.sh --pgvector`); without it the
# section prints the commands and shows sample output in the documented format (illustrative).
#
# ## The one-minute version
#
# A long-term memory store needs four things, and one SQLite file has all of them:
#
# * **the typed record** (PRIMER §1 "What an agent remembers"): kind, scope, source, trust, provenance,
#   confidence, importance, validity, TTL and a deletion key — one row;
# * **a vector per record**, scored exactly with numpy (a flat index: per-user partitions are small, and a
#   flat index can *delete*, which 07.4's `minifaiss` HNSW cannot);
# * **a full-text index** (FTS5, ranked by `bm25()`), fused with the vector ranking by reciprocal rank
#   fusion — 07.4's hybrid search, reused (`ragkit.reference.reciprocal_rank_fusion`, $k = 60$);
# * **the partition** `(tenant, user_id)` in *every* query's WHERE clause, so scope is enforced by the store
#   rather than remembered by the caller (vector-databases primer §9, §11).
#
# What SQLite does not do by default is **forget**. `DELETE` leaves the text in the FTS5 index, in the
# write-ahead log and in freed pages; you will count the copies on disk and then remove them (PRIMER §7
# "Consolidation, forgetting and deletion"). Primer: [`../../PRIMER.md`](../../PRIMER.md).

# %%
import os, sqlite3, tempfile, math, re
import numpy as np
from memlab import env
from memlab.records import MemoryRecord, count_tokens
from memlab.memory import LocalMemory
from memlab.store import SQLiteMemoryStore
from memlab.store.sqlite import SCHEMA, FTS_SCHEMA, STOPWORDS, rrf as lab_rrf, sqlite_features
from memlab.deletion import residue

print(env.banner())
WORK = tempfile.mkdtemp(prefix="memlab-nb01-")
import atexit, shutil
atexit.register(shutil.rmtree, WORK, True)   # removed when the kernel exits, even if a cell stops early
store = SQLiteMemoryStore(os.path.join(WORK, "memory.db"))
print("SQLite features here:", sqlite_features())

# %% [markdown]
# ## Worked example: the schema, and three users' memories in one file
#
# One table holds the record and its vector; one FTS5 table indexes the text (porter-stemmed, so "trips"
# finds "trip"). Two users of tenant `acme` and one of tenant `globex` share the file: isolation is the
# partition predicate, not a separate database.

# %%
print(SCHEMA.strip().splitlines()[0], "...", FTS_SCHEMA, sep="\n")
facts = {
    ("acme", "u1"): ["Home city: the user lives in Lisbon.", "Employer: the user works at Globex.",
                     "Pet: the user has a dog named Rex.", "Trip: the user took a trip to Kyoto.",
                     "Favourite drink: the user's favourite drink is green tea."],
    ("acme", "u2"): ["Home city: the user lives in Oslo.", "Employer: the user works at Initech."],
    ("globex", "u1"): ["Home city: the user lives in Lima.", "Pet: the user has a cat named Tom."],
}
for (tenant, user), texts in facts.items():
    mem = LocalMemory(store, tenant, user)
    for t in texts:
        mem.remember(t, slot=t.split(":")[0].lower().replace(" ", "_"))
row = store.con.execute("SELECT id, tenant, user_id, kind, source, trust, deletion_key, length(embedding) AS vec_bytes "
                        "FROM memories LIMIT 1").fetchone()
print(dict(row))
print(store.stats())

# %% [markdown]
# ## Worked example: three rankings for one question
#
# The vector leg is a dot product over the partition's unit vectors (the hashing embedder is lexical: only
# shared words count). The full-text leg is FTS5's `bm25()` — **negative, lower is better** — over the
# question's words minus stopwords. Hybrid fuses the two *rankings* by RRF, so their different scales
# never meet.

# %%
Q = "Which city is my home city, and where did my trip go?"
for mode in ("vector", "fts", "hybrid"):
    hits = store.search("acme", "u1", Q, k=3, mode=mode, touch=False)
    print(f"{mode:6s}", [(h.record.text[:34], round(h.score, 4), None if h.bm25 is None else round(h.bm25, 3)) for h in hits])
print("acme/u2 asks the same:", [h.record.text for h in store.search("acme", "u2", Q, k=3, touch=False)])

# %% [markdown]
# ## Exercise 1.1 — reciprocal rank fusion
#
# Write `rrf(rankings, k=60)`: each ranking is a list of ids, best first; an id at 0-based position $r$
# earns $1/(k + r + 1)$ from that list; return `(id, score)` pairs sorted by score, highest first. This is
# `ragkit.reference.reciprocal_rank_fusion` (07.4 notebook 03) — the store's hybrid mode uses the same rule.

# %% exercise
def rrf(rankings, k=60):
    ### BEGIN SOLUTION
    scores = {}
    for ranking in rankings:
        for r, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
    ### END SOLUTION

# %% check
assert rrf([["a", "b"], ["a", "c"]])[0] == ("a", 2 / 61)                  # first in both lists
assert math.isclose(dict(rrf([["a", "b"], ["c", "b"]]))["b"], 2 / 62)      # second in both
hits = store.search("acme", "u1", Q, k=5, mode="hybrid", touch=False)
vec = [h.record.id for h in sorted((h for h in hits if h.rank_vector is not None), key=lambda h: h.rank_vector)]
fts = [h.record.id for h in sorted((h for h in hits if h.rank_fts is not None), key=lambda h: h.rank_fts)]
mine = dict(rrf([vec, fts]))
assert all(math.isclose(mine[h.record.id], h.score) for h in hits)
print("✅ rrf reproduces the store's hybrid scores; the best id scores", round(rrf([vec, fts])[0][1], 6))

# %% [markdown]
# ## Exercise 1.2 — `bm25()` by hand, sign and all
#
# Reproduce FTS5's `bm25()` for a query of one or more terms on a small table: for each query term,
#
# $$
# \mathrm{idf} = \ln\frac{N - n + 0.5}{n + 0.5}
# $$
#
# ($N$ rows, $n$ rows containing the term), **floored at 1e-6 when it is not positive**; each row scores
#
# $$
# \sum \mathrm{idf} \cdot
# \frac{\mathrm{tf} \cdot (k_1 + 1)}{\mathrm{tf} + k_1 \cdot (1 - b + b \cdot \lvert d \rvert / \mathrm{avgdl})}
# $$
#
# with $k_1 = 1.2$, $b = 0.75$, $\lvert d \rvert$ the row's token count and $\mathrm{avgdl}$ the mean; `bm25()`
# returns **minus** that. (ragkit's BM25 uses $k_1 = 1.5$ — same idea, different constant.) Return one value per
# document.

# %%
DOCS = ["the user lives in lisbon", "the user works at globex in lisbon", "a dog named rex",
        "green tea every morning with the dog", "the train was late again", "lisbon lisbon lisbon trams"]
toy = sqlite3.connect(":memory:")
toy.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='unicode61')")
toy.executemany("INSERT INTO t(x) VALUES (?)", [(d,) for d in DOCS])
def sqlite_bm25(terms):
    return dict(toy.execute("SELECT rowid - 1, bm25(t) FROM t WHERE t MATCH ?", (" OR ".join(terms),)).fetchall())

# %% exercise
def bm25(terms, docs, k1=1.2, b=0.75):
    ### BEGIN SOLUTION
    toks = [re.findall(r"[a-z0-9]+", d.lower()) for d in docs]
    N, avgdl = len(toks), sum(map(len, toks)) / len(toks)
    out = []
    for d in toks:
        s = 0.0
        for q in terms:
            n = sum(q in x for x in toks)
            idf = math.log((N - n + 0.5) / (n + 0.5))
            idf = idf if idf > 0 else 1e-6
            tf = d.count(q)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        out.append(-s)
    return out
    ### END SOLUTION

# %% check
for terms in (["dog"], ["lisbon"], ["dog", "lisbon"], ["tea", "train"]):
    want, got = sqlite_bm25(terms), bm25(terms, DOCS)
    assert all(math.isclose(got[i], v, rel_tol=1e-9, abs_tol=1e-12) for i, v in want.items()), (terms, want, got)
print("✅ bm25 matches SQLite:", {i: round(v, 4) for i, v in sqlite_bm25(["dog"]).items()},
      "| 'lisbon' is in half the rows, so its IDF is floored:", sqlite_bm25(["lisbon"]))

# %% [markdown]
# The floor matters for memory: a word in half a user's memories ("user", "the") carries no ranking signal,
# and FTS5 statistics are **table-wide** — another tenant's rows move your IDF (your ranking, not your
# result set). A per-tenant FTS table removes that coupling at the cost of more tables.
#
# ## Exercise 1.3 — how big is a memory?
#
# Write `vector_bytes(dim, backend)`: the bytes of one stored vector for `"sqlite"` (float32 BLOB, 4 bytes per
# dimension), `"pgvector"` (`vector`: $4 \cdot \text{dim} + 8$) and `"halfvec"` ($2 \cdot \text{dim} + 8$) —
# pgvector's README sizes. Then `memory_bytes(n, dim, avg_text_chars, backend)`: $n$ records × (vector + text
# bytes), ignoring indexes.

# %% exercise
def vector_bytes(dim, backend="sqlite"):
    ### BEGIN SOLUTION
    return {"sqlite": 4 * dim, "pgvector": 4 * dim + 8, "halfvec": 2 * dim + 8}[backend]
    ### END SOLUTION

def memory_bytes(n, dim, avg_text_chars, backend="sqlite"):
    ### BEGIN SOLUTION
    return n * (vector_bytes(dim, backend) + avg_text_chars)
    ### END SOLUTION

# %% check
assert vector_bytes(1024) == row["vec_bytes"] == 4096
assert vector_bytes(384, "pgvector") == 1544 and vector_bytes(1024, "halfvec") == 2056
per_tenant = memory_bytes(100_000, 1024, 120)
assert per_tenant == 421_600_000
print(f"✅ 100k memories x 1024-d: {per_tenant / 1e6:.0f} MB of vectors and text — the vector is 97% of it; "
      f"a 384-d embedder cuts it to {memory_bytes(100_000, 384, 120) / 1e6:.0f} MB")

# %% [markdown]
# ## Worked example: a forget that is only a DELETE
#
# A user asks us to forget their home address. The naive forget deletes the row and its FTS row. Count the
# address on disk before and after — in the database file *and* its `-wal` file.

# %%
naive = SQLiteMemoryStore(os.path.join(WORK, "naive.db"), fts_secure_delete=False)
for i in range(30):
    naive.add(MemoryRecord("acme", "u1", f"Note {i}: the user mentioned the weather and lunch.", kind="episodic"))
LocalMemory(naive, "acme", "u1").remember("Home address: the user lives at 12 Rua das Flores.", slot="address")
print("before:", residue(naive.path, ["Flores"]))
rep = naive.forget("acme", "acme/u1/address", mode="logical", needles=["Flores"])
print(rep.table())
print("search after the forget:", [h.record.text for h in naive.search("acme", "u1", "Rua das Flores", touch=False)])

# %% [markdown]
# The search finds nothing — and the bytes are still there: the old page images in the WAL and the terms in
# FTS5's index segments. A forget that only stops *retrieval* has not forgotten.
#
# ## Exercise 1.4 — the purge
#
# Write `purge_steps(fts_secure_delete, wal)`: the SQL to run **after** the DELETE so no copy is left, in
# order. Without FTS5 `secure-delete`, merge the index (`INSERT INTO memories_fts(memories_fts)
# VALUES('optimize')`); in WAL mode checkpoint and truncate the log (`PRAGMA wal_checkpoint(TRUNCATE)`)
# before *and* after rewriting the file with `VACUUM` (a VACUUM in WAL mode writes through the log too).

# %% exercise
def purge_steps(fts_secure_delete, wal):
    ### BEGIN SOLUTION
    steps = [] if fts_secure_delete else ["INSERT INTO memories_fts(memories_fts) VALUES('optimize')"]
    ckpt = ["PRAGMA wal_checkpoint(TRUNCATE)"] if wal else []
    return steps + ckpt + ["VACUUM"] + ckpt
    ### END SOLUTION

# %% check
assert purge_steps(True, False) == ["VACUUM"]
assert purge_steps(False, True)[0].endswith("VALUES('optimize')") and purge_steps(False, True)[-1] == "PRAGMA wal_checkpoint(TRUNCATE)"
for step in purge_steps(fts_secure_delete=False, wal=True):
    naive.con.execute(step)
left = residue(naive.path, ["Flores"])
assert left == {"Flores": 0}, left
print("✅ after", len(purge_steps(False, True)), "steps the address is gone from the file and the WAL:", left)
print("   the store's own forget(mode='purge') runs the same steps:", naive.forget("acme", "x/y/z").steps[1:])

# %% [markdown]
# `SQLiteMemoryStore` turns FTS5 `secure-delete` on at creation when this SQLite has it (3.42 or newer,
# verify; Colab's may be older — the store feature-detects on a throwaway table, never by version string),
# which removes a deleted row's terms from the index at DELETE time; `PRAGMA secure_delete=ON` zeroes freed
# page content. Neither reaches a copy of the file somebody else took: backups are PRIMER §7's last line.
#
# ## T0 + Docker: the same store on Postgres + pgvector
#
# `memlab.store.pgvector` has the same schema and the same three queries in SQL (RRF with $k = 60$ written as
# two ranked CTEs), plus the consolidation job's lease and checkpoint tables. With `MEMLAB_PG_DSN` set and
# psycopg installed this cell runs them; otherwise it prints how to start Postgres and checks every
# statement with Postgres's own parser (`pglast`) offline.

# %%
from memlab.store import pgvector as pgv
dsn = env.pg_dsn()
if dsn and env.has_module("psycopg"):
    pg = pgv.PgVectorStore(dsn)
    pg.init()
    for (tenant, user), texts in facts.items():
        for t in texts:
            pg.add(MemoryRecord(tenant, user, t, slot=t.split(":")[0].lower().replace(" ", "_")))
    print("[MEASURED on Postgres]", [(h.record.text, round(h.score, 4)) for h in pg.search("acme", "u1", Q, k=3)])
    print(pg.forget("acme", "acme/u1/home_city").table())
    pg.close()
else:
    print("No Postgres here. To run this section (T0 + Docker):")
    print("   deploy/local/up.sh --pgvector      # pgvector/pgvector:0.8.6-pg17 on 127.0.0.1:5432")
    print('   pip install "psycopg[binary]>=3.1" && export MEMLAB_PG_DSN=postgresql://memlab:memlab-local-only@127.0.0.1:5432/memlab')
    print("\nThe hybrid search it would send:\n" + pgv.SEARCH[:420] + " ...")
    if env.has_module("pglast"):
        import pglast
        for name, sql in pgv.statements().items():
            pglast.parse_sql(pgv.to_positional(sql)[0])
        print(f"\nall {len(pgv.statements())} statements parse with Postgres's parser (pglast)")
    print("\n" + open(os.path.join(os.path.dirname(pgv.__file__), "..", "data", "samples", "pgvector_session.txt")).read())

# %%
import shutil
for s in (store, naive):
    s.close()
shutil.rmtree(WORK, ignore_errors=True)    # this notebook's databases: a lab about forgetting leaves no copies behind
print("removed", WORK)

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "Each memory is one typed row — kind, scope, source, trust, provenance, validity, TTL and a
# deletion key — with its embedding in the same row and its text in a full-text index. Every query carries
# `tenant` and `user_id` in the WHERE clause; a per-user partition is small, so vector search is an exact
# scan and we fuse it with BM25 by reciprocal rank fusion. The flat index can delete, which matters more
# than ANN speed at this size. Forgetting is a purge, not a DELETE: we turn on FTS5 secure-delete, checkpoint
# the WAL, VACUUM, and prove it by searching the file for the deleted bytes. On Postgres the same design
# holds, but a purge can only reach the heap and indexes — WAL archives and backups age out on their own
# schedule, and that window goes into the deletion policy."
#
# **Drill 1.** *Why not HNSW for memory?* — A user's partition holds hundreds to thousands of records: an
# exact scan is milliseconds and never misses. HNSW pays at tenant scale, and pgvector filters *after* the
# index scan (ef_search 40 → about 4 rows at 10% selectivity), so filtered ANN needs partitions or
# `hnsw.iterative_scan`.
#
# **Drill 2.** *A test shows search no longer returns the deleted memory. Are we compliant?* — No: retrieval
# is one surface. The text is still in the FTS index segments and the WAL until a merge/secure-delete and a
# checkpoint, in freed pages until VACUUM, and in derived facts, caches, logs and backups (notebook 05).
#
# **Drill 3.** *Why is `bm25()` negative, and why did "lisbon" score −1e-6?* — FTS5 returns minus the score so
# ascending ORDER BY ranks best first; a term in at least half the rows has a non-positive IDF, floored at
# 1e-6 — it cannot rank anything.
