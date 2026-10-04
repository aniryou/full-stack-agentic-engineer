# %% [markdown]
# # Exercises 05 · Evaluation & fusion
# These are the three functions that every owner of a retrieval system writes
# eventually. The solutions are in `../solutions/ex05.ipynb`.

# %%
import numpy as np
from collections import Counter

# %% [markdown]
# ## Task 1 — graded nDCG@k
# Implement `ndcg_at_k(gains, ideal_gains, k)` with $\mathrm{DCG} = \sum_i \mathrm{gain}_i / \log_2(i + 2)$.
# Check: a ranked list with gains [3,2,0,1] against the ideal [3,2,1,0] gives 0.98544.

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
# `rrf(rankings, k)` uses this score:
#
# $$
# \mathrm{score}(d) = \sum_{\text{rankings}} \frac{1}{k + \mathrm{rank}(d) + 1},
# $$
#
# The rank is 0-based. Return the doc ids, sorted by score.
#
# Checks with $k = 1$:
#
# - [a,b,c] + [c,a,b] gives a: 1/2+1/3, c: 1/4+1/2, b: 1/3+1/4. The order is a, c, b.
# - [a,b,c,d] + [b,c,d] gives b: 1/3+1/2 = 5/6, c: 1/4+1/3 = 7/12, a: 1/2,
#   d: 1/5+1/4 = 9/20. The order is b, c, a, d. No two scores are equal. Thus the
#   order does not depend on how you break ties, and the constant has an effect:
#   - $1/(k + \mathrm{rank})$ gives the scores b 3/2, a 1, c 5/6, d 7/12, and the order b, a, c, d.
#   - $1/(k + \mathrm{rank} + 2)$ gives the scores b 7/12, c 9/20, d 11/30, a 1/3, and the order b, c, d, a.

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
assert rrf([["a", "b", "c", "d"], ["b", "c", "d"]], k=1) == ["b", "c", "a", "d"], \
    "check the constant: score = 1/(k + rank + 1) with 0-based rank"
print("rrf ✓")

# %% [markdown]
# ## Task 3 — the two halves of a BM25 term score
#
# $$
# \begin{aligned}
# \mathrm{bm25\_idf}(N, \mathrm{df}) &= \log\left(1 + \frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5}\right) \quad \text{and} \\
# \mathrm{bm25\_tf\_norm}(f, \mathrm{dl}, \mathrm{avgdl}, k_1, b) &= \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot (1 - b + b \cdot \mathrm{dl}/\mathrm{avgdl})}.
# \end{aligned}
# $$
#
# The second function is the part to learn well: term-frequency
# **saturates**, and BM25 **penalizes** long documents.

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
# In notebook 05, replace RRF with a weighted sum of scores,
# $\alpha \cdot z(\text{bm25}) + (1 - \alpha) \cdot z(\text{dense})$ ($z$ standardizes the scores for each query). Sweep $\alpha$.
# Why does RRF usually give the better result with no adjustment? (Hint: compare
# score scales against rank scales.)
