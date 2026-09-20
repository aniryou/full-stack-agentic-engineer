# %% [markdown]
# # Exercises 05 · Evaluation & fusion
# The three functions every retrieval system owner ends up writing.
# Solutions: `solutions/ex05_solutions.ipynb`.

# %%
import numpy as np
from collections import Counter

# %% [markdown]
# ## Task 1 — graded nDCG@k
# `ndcg_at_k(gains, ideal_gains, k)` with `DCG = Σ gain_i / log2(i + 2)`.
# Check: ranking with gains [3,2,0,1] vs ideal [3,2,1,0] → 0.98544.

# %%
def dcg(gains, k):
    # >>> SOLUTION
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k]))
    # <<< SOLUTION

def ndcg_at_k(gains, ideal_gains, k=10):
    # >>> SOLUTION
    ide = dcg(sorted(ideal_gains, reverse=True), k)
    return dcg(gains, k) / ide if ide > 0 else 0.0
    # <<< SOLUTION

assert np.isclose(ndcg_at_k([3, 2, 0, 1], [3, 2, 1, 0], k=10), 0.98544, atol=1e-4)
assert np.isclose(ndcg_at_k([3, 2, 1, 0], [3, 2, 1, 0], k=10), 1.0)
print("ndcg_at_k ✓")

# %% [markdown]
# ## Task 2 — Reciprocal Rank Fusion
# `rrf(rankings, k)`: score(d) = Σ over rankings 1/(k + rank(d) + 1), rank
# 0-based; return doc ids sorted by score. Check with k=1:
# [a,b,c] + [c,a,b] → a: 1/2+1/3, c: 1/4+1/2, b: 1/3+1/4 → order a, c, b.

# %%
def rrf(rankings, k=60):
    # >>> SOLUTION
    sc = Counter()
    for ranked in rankings:
        for r, d in enumerate(ranked):
            sc[d] += 1.0 / (k + r + 1)
    return [d for d, _ in sc.most_common()]
    # <<< SOLUTION

assert rrf([["a", "b", "c"], ["c", "a", "b"]], k=1) == ["a", "c", "b"]
print("rrf ✓")

# %% [markdown]
# ## Task 3 — the two halves of a BM25 term score
# `bm25_idf(N, df) = log(1 + (N − df + 0.5)/(df + 0.5))` and
# `bm25_tf_norm(f, dl, avgdl, k1, b) = f·(k1+1) / (f + k1·(1 − b + b·dl/avgdl))`.
# The second is the part worth internalizing: term-frequency **saturates** and
# long documents are **penalized**.

# %%
def bm25_idf(N, df):
    # >>> SOLUTION
    return np.log(1 + (N - df + 0.5) / (df + 0.5))
    # <<< SOLUTION

def bm25_tf_norm(f, dl, avgdl, k1=1.5, b=0.75):
    # >>> SOLUTION
    return f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
    # <<< SOLUTION

assert np.isclose(bm25_idf(100, 10), np.log(1 + 90.5 / 10.5))
assert np.isclose(bm25_tf_norm(2, 100, 100), 10 / 7)
assert bm25_tf_norm(20, 100, 100) < 2.5 * bm25_tf_norm(1, 100, 100)   # saturation
assert bm25_tf_norm(2, 200, 100) < bm25_tf_norm(2, 50, 100)           # length penalty
print("bm25 components ✓")

# %% [markdown]
# ## Task 4 (open) — break the hybrid
# In notebook 05, replace RRF with a weighted score sum
# `α·z(bm25) + (1−α)·z(dense)` (z = standardize scores per query). Sweep α.
# Why does RRF usually win without tuning? (Hint: score scales vs rank scales.)
