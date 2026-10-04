# %% [markdown]
# # 01 · A memory store on SQLite: records, vectors, full text — and a delete that removes the bytes
#
# **Tier:** T0. SQLite with FTS5 is part of Python, the vectors are numpy arrays in BLOB columns, and the
# embedder is a hashing embedder. The notebook downloads nothing. **T0 + Docker:** the last section runs the
# same store on Postgres + pgvector when `MEMLAB_PG_DSN` points at one (`deploy/local/up.sh --pgvector`).
# Without it, the section prints the commands and shows sample output in the documented format (illustrative).
#
# ## The one-minute version
#
# A long-term memory store needs four things, and one SQLite file has all of them:
#
# * **the typed record** (PRIMER §1 "What an agent remembers"): kind, scope, source, trust, provenance,
#   confidence, importance, validity, TTL and a deletion key, in one row.
# * **a vector per record**, scored exactly with numpy (a flat index). Per-user partitions are small, and a
#   flat index can *delete*. The `minifaiss` HNSW of 07.4 cannot delete.
# * **a full-text index** (FTS5, ranked by `bm25()`). The store fuses its ranking with the vector ranking by
#   reciprocal rank fusion. This is the hybrid search of 07.4, used again
#   (`ragkit.reference.reciprocal_rank_fusion`, $k = 60$).
# * **the partition** `(tenant, user_id)` in the WHERE clause of *every* query. Thus the store enforces the
#   scope, and the caller does not have to remember it (vector-databases primer §9, §11).
#
# By default, SQLite does not **forget**. `DELETE` leaves the text in the FTS5 index, in the write-ahead log
# and in freed pages. You will count the copies on disk, and then you will remove them (PRIMER §7
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
# One table holds the record and its vector. One FTS5 table indexes the text (porter-stemmed, thus "trips"
# finds "trip"). Two users of tenant `acme` and one user of tenant `globex` share the file. The isolation
# comes from the partition predicate, not from a separate database.

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
# The vector leg is a dot product over the unit vectors of the partition (the hashing embedder is lexical:
# only shared words count). The full-text leg is the `bm25()` of FTS5 over the words of the question, minus
# the stopwords. Its value is **negative, and lower is better**. Hybrid fuses the two *rankings* by RRF, thus
# their different scales never meet.

# %%
Q = "Which city is my home city, and where did my trip go?"
for mode in ("vector", "fts", "hybrid"):
    hits = store.search("acme", "u1", Q, k=3, mode=mode, touch=False)
    print(f"{mode:6s}", [(h.record.text[:34], round(h.score, 4), None if h.bm25 is None else round(h.bm25, 3)) for h in hits])
print("acme/u2 asks the same:", [h.record.text for h in store.search("acme", "u2", Q, k=3, touch=False)])

# %% [markdown]
# ## Exercise 1.1 — reciprocal rank fusion
#
# Write `rrf(rankings, k=60)`. Each ranking is a list of ids, best first. An id at 0-based position $r$ gets
# $1/(k + r + 1)$ from that list. Return `(id, score)` pairs, sorted by score, highest first. This function
# is `ragkit.reference.reciprocal_rank_fusion` (07.4 notebook 03). The hybrid mode of the store uses the same
# rule.

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
# Reproduce the `bm25()` of FTS5 for a query of one or more terms on a small table. For each query term, the
# IDF is:
#
# $$
# \mathrm{idf} = \ln\frac{N - n + 0.5}{n + 0.5}
# $$
#
# Here $N$ is the number of rows, and $n$ is the number of rows that contain the term. **When the IDF is not
# positive, use the floor value 1e-6.** Each row gets this score:
#
# $$
# \sum \mathrm{idf} \cdot
# \frac{\mathrm{tf} \cdot (k_1 + 1)}{\mathrm{tf} + k_1 \cdot (1 - b + b \cdot \lvert d \rvert / \mathrm{avgdl})}
# $$
#
# Here $k_1 = 1.2$, $b = 0.75$, $\lvert d \rvert$ is the token count of the row, and $\mathrm{avgdl}$ is the
# mean. `bm25()` returns **minus** that score. (The BM25 of ragkit uses $k_1 = 1.5$. The idea is the same,
# but the constant is different.) Return one value per document.

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
# The floor is important for memory. A word in half of the memories of a user ("user", "the") gives no signal
# to the ranking. Also, the FTS5 statistics are **table-wide**. Thus the rows of another tenant move your IDF.
# They change your ranking, not your result set. An FTS table per tenant removes that dependency, but the cost
# is more tables.
#
# ## Exercise 1.3 — how big is a memory?
#
# Write `vector_bytes(dim, backend)`. It returns the bytes of one stored vector for `"sqlite"` (float32 BLOB,
# 4 bytes per dimension), `"pgvector"` (`vector`: $4 \cdot \text{dim} + 8$) and `"halfvec"`
# ($2 \cdot \text{dim} + 8$). These are the sizes from the pgvector README. Then write
# `memory_bytes(n, dim, avg_text_chars, backend)`. It returns $n$ records × (vector + text bytes). Do not
# count the indexes.

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
# A user asks us to forget their home address. The simple forget deletes the row and its FTS row. Count the
# address on disk before and after the forget. Count it in the database file *and* in its `-wal` file.

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
# The search finds nothing. But the bytes are still there: the old page images in the WAL, and the terms in
# the index segments of FTS5. A forget that only stops *retrieval* has not forgotten.
#
# ## Exercise 1.4 — the purge
#
# Write `purge_steps(fts_secure_delete, wal)`. It returns, in order, the SQL to run **after** the DELETE, so
# that no copy stays. Without FTS5 `secure-delete`, merge the index
# (`INSERT INTO memories_fts(memories_fts) VALUES('optimize')`). In WAL mode, do a checkpoint and truncate
# the log (`PRAGMA wal_checkpoint(TRUNCATE)`). Do this before *and* after `VACUUM` writes the file again. (In
# WAL mode, a VACUUM also writes through the log.)

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
# When this SQLite has FTS5 `secure-delete` (3.42 or newer, verify), `SQLiteMemoryStore` turns it on at
# creation. It is possible that the SQLite of Colab is older. The store detects the feature on a
# temporary table, never from the version string. With `secure-delete`, a DELETE also removes the terms of
# the deleted row from the index. `PRAGMA secure_delete=ON` writes zeros over the content of freed pages.
# Neither reaches a copy of the file that another person took: backups are the last line of PRIMER §7.
#
# ## T0 + Docker: the same store on Postgres + pgvector
#
# `memlab.store.pgvector` has the same schema and the same three queries in SQL. The SQL writes RRF with
# $k = 60$ as two ranked CTEs. The module also has the lease and checkpoint tables of the consolidation job.
# If `MEMLAB_PG_DSN` has a value and psycopg is available, this cell runs the queries. If not, the cell prints how
# to start Postgres. It also examines every statement offline with the parser of Postgres itself (`pglast`).

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
# **Two minutes.** "Each memory is one typed row with kind, scope, source, trust, provenance, validity, TTL
# and a deletion key. Its embedding is in the same row, and its text is in a full-text index. Every query has
# `tenant` and `user_id` in the WHERE clause. A per-user partition is small. Thus vector search is an exact
# scan, and we fuse it with BM25 by reciprocal rank fusion.
#
# "The flat index can delete, and at this size that is more important than ANN speed. A forget is a purge,
# not a DELETE. We turn on FTS5 secure-delete, do a checkpoint of the WAL and run VACUUM. Then we prove it:
# we search the file for the deleted bytes. On Postgres the same design is correct, but a purge can only reach
# the heap and the indexes. WAL archives and backups expire on their own schedule, and that window goes into
# the deletion policy."
#
# **Drill 1.** *Why not HNSW for memory?* The partition of a user holds hundreds to thousands of records. An
# exact scan takes milliseconds and never misses a record. HNSW is worth its cost at tenant scale. Also,
# pgvector filters *after* the index scan (ef_search 40 gives about 4 rows at 10% selectivity), thus filtered
# ANN needs partitions or `hnsw.iterative_scan`.
#
# **Drill 2.** *A test shows that search no longer returns the deleted memory. Are we compliant?* No,
# retrieval is one surface. The text stays in the FTS index segments and in the WAL until a merge/secure-delete
# and a checkpoint occur. It stays in freed pages until VACUUM, and also in derived facts, caches, logs
# and backups (notebook 05).
#
# **Drill 3.** *Why is `bm25()` negative, and why did "lisbon" score −1e-6?* FTS5 returns minus the score, so
# that an ORDER BY from low to high puts the best row first. A term in at least half of the rows has an IDF
# that is not positive. The floor sets that IDF to 1e-6, thus the term cannot rank anything.
