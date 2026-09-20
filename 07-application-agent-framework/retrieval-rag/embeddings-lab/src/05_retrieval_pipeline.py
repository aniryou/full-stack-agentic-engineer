# %% [markdown]
# # 05 · A hybrid retrieval pipeline, evaluated properly
# **One idea:** lexical and dense retrieval fail differently, so production
# systems fuse them and *measure on their own judged queries*. BM25 from
# scratch + an SVD dense retriever + Reciprocal Rank Fusion, evaluated with
# Recall@k and nDCG on a corpus with planted failure modes. *Primer §5, §6, §15.*

# %%
import json, re
import numpy as np
docs = [json.loads(l) for l in open("../data/docs.jsonl")]
queries = [json.loads(l) for l in open("../data/queries.jsonl")]
ids = [d["id"] for d in docs]
texts = [d["title"] + " " + d["text"] for d in docs]

# with 36 docs, idf cannot bury function words the way it does at scale, so we
# stoplist them. "not" is dropped too — real pipelines do the equivalent, which
# is exactly why the negation query below fails.
STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
        "is", "are", "that", "this", "it", "its", "do", "does", "not", "no",
        "my", "i", "your", "by", "from", "at", "as", "be", "if", "when", "so",
        "because", "only", "should", "which", "how", "what"}

def tok(s):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", s.lower())
            if w not in STOP]

toks = [tok(t) for t in texts]
print(len(docs), "docs |", len(queries), "judged queries")

# %% [markdown]
# ## BM25 in ~20 lines

# %%
from collections import Counter
Ndoc = len(toks)
df = Counter(w for t in toks for w in set(t))
avgdl = np.mean([len(t) for t in toks])
tf = [Counter(t) for t in toks]

def bm25_scores(q, k1=1.5, b=0.75):
    s = np.zeros(Ndoc)
    for w in tok(q):
        if w not in df:
            continue
        idf = np.log(1 + (Ndoc - df[w] + 0.5) / (df[w] + 0.5))
        for i in range(Ndoc):
            f = tf[i][w]
            if f:
                s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks[i]) / avgdl))
    return s

# %% [markdown]
# ## A dense retriever: TF-IDF → SVD (LSA)
# A stand-in for a trained embedding model with identical pipeline mechanics:
# docs become vectors offline, queries are *folded in* at query time
# (`u_q = q · V · S⁻¹`), similarity is cosine. Because SVD links words that
# co-occur, "limescale" and "descale" end up near each other even when a doc
# uses only one of them.

# %%
vocab = sorted(df)
w2i = {w: i for i, w in enumerate(vocab)}
idf = np.log(Ndoc / np.array([df[w] for w in vocab]))

def tfidf(tokens):
    v = np.zeros(len(vocab))
    for w, c in Counter(tokens).items():
        if w in w2i:
            v[w2i[w]] = c * idf[w2i[w]]
    n = np.linalg.norm(v)
    return v / n if n else v

Xt = np.stack([tfidf(t) for t in toks])          # docs × terms
U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
K = 12                                           # low rank = strong smoothing
Udoc = U[:, :K] * S[:K]                          # doc vectors
Udoc = Udoc / np.linalg.norm(Udoc, axis=1, keepdims=True)

def dense_scores(q):
    u = tfidf(tok(q)) @ Vt[:K].T / S[:K]         # fold-in
    n = np.linalg.norm(u)
    return Udoc @ (u / n if n else u)

# %% [markdown]
# ## Metrics: Recall@k and nDCG@10 (binary relevance)

# %%
def rank_ids(scores):
    return [ids[i] for i in np.argsort(-scores)]

def recall_at(ranked, rel, k):
    return len(set(ranked[:k]) & set(rel)) / len(rel)

def ndcg10(ranked, rel):
    gains = [1.0 if r in rel else 0.0 for r in ranked[:10]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg

def evaluate(score_fn):
    out = {}
    for q in queries:
        ranked = rank_ids(score_fn(q["text"]))
        out[q["qid"]] = {"r@3": recall_at(ranked, q["relevant"], 3),
                         "ndcg": ndcg10(ranked, q["relevant"])}
    return out

# %% [markdown]
# ## Fusion: Reciprocal Rank Fusion (RRF)
# Rank-based, so no score-calibration across systems is needed — the reason it
# is everyone's default (`k = 60`).

# %%
def rrf(rankings, k=60):
    sc = Counter()
    for ranked in rankings:
        for r, docid in enumerate(ranked):
            sc[docid] += 1 / (k + r + 1)
    return [d for d, _ in sc.most_common()]

def evaluate_hybrid():
    out = {}
    for q in queries:
        ranked = rrf([rank_ids(bm25_scores(q["text"])), rank_ids(dense_scores(q["text"]))])
        out[q["qid"]] = {"r@3": recall_at(ranked, q["relevant"], 3),
                         "ndcg": ndcg10(ranked, q["relevant"])}
    return out

systems = {"BM25": evaluate(bm25_scores), "dense (LSA)": evaluate(dense_scores),
           "hybrid (RRF)": evaluate_hybrid()}
qtypes = {q["qid"]: q["qtype"] for q in queries}
print(f"{'system':>13} | {'mean nDCG@10':>12} | by query type (nDCG)")
for name, res in systems.items():
    mean = np.mean([r["ndcg"] for r in res.values()])
    by = {}
    for qid, r in res.items():
        by.setdefault(qtypes[qid], []).append(r["ndcg"])
    bys = "  ".join(f"{t}:{np.mean(v):.2f}" for t, v in sorted(by.items()))
    print(f"{name:>13} | {mean:12.3f} | {bys}")

# %% [markdown]
# ## Reading the failure modes (the point of the exercise)

# %%
def show(qid, n=3):
    q = next(x for x in queries if x["qid"] == qid)
    print(f"\nQ: {q['text']!r}   (type: {q['qtype']}, relevant: {q['relevant']})")
    for name, fn in [("BM25", bm25_scores), ("dense", dense_scores)]:
        top = rank_ids(fn(q["text"]))[:n]
        marks = ["✓" if t in q["relevant"] else "✗" for t in top]
        print(f"  {name:>5}: " + ", ".join(f"{m}{t}" for m, t in zip(marks, top)))

show("q01")   # paraphrase: dense finds c03 despite ZERO query-term overlap
show("q02")   # exact id: e-4417 vs e-4471 — BM25 nails it, dense confuses codes
show("q03")   # negation: 'not steep' retrieves the steep doc — both fail

# %% [markdown]
# **What just happened.** The paraphrase query rewards the dense retriever: c03
# shares *zero* content words with the query, yet LSA parked it next to the
# descaling doc that does. The identifier query rewards BM25: the two error docs
# differ in one rare token, so their dense vectors nearly coincide and LSA ranks
# the *wrong* code first. The negation query fools both — "not steep" and
# "steep" are nearly identical bags of words *and* nearly identical vectors (the
# NevIR failure from §15) — which is why rerankers and metadata filters exist.
#
# **Takeaways.** (1) Hybrid by default; RRF is 8 lines — but it is a compromise,
# not a free lunch: on the paraphrase slice it sits *between* BM25 and dense.
# Fusion buys robustness across query types, not dominance on each. (2) Metrics
# on *your* judged queries beat any leaderboard — this whole harness is ~40 lines.
# (3) Keep per-query-type slices; averages hide exactly the failures that hurt.
# → `exercises/ex05.ipynb`.
