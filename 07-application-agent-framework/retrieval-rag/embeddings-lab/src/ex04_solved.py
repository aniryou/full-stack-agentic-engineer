# %% [markdown]
# # Exercises 04 · ANN internals
# Implement the two core moves of IVF-PQ, then demonstrate the filtered-search
# trap yourself. Solutions: `solutions/ex04_solutions.ipynb`.

# %%
import numpy as np
rng = np.random.default_rng(0)

def l2sq(A, B):
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T

def kmeans(X, k, iters=12, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = np.argmin(l2sq(X, C), axis=1)
        for c in range(k):
            if (a == c).any():
                C[c] = X[a == c].mean(0)
    return C, np.argmin(l2sq(X, C), axis=1)

N, D = 4000, 32
X = rng.normal(size=(N, D)) + rng.normal(size=(20, D))[rng.integers(0, 20, N)] * 3
Q = X[rng.integers(0, N, 50)] + 0.3 * rng.normal(size=(50, D))
gold = np.argsort(l2sq(Q, X), axis=1)[:, :10]

# %% [markdown]
# ## Task 1 — asymmetric distance computation (ADC)
# Given per-subspace `codebooks` (M, 256, sub) and `codes` (N, M), return the
# PQ distance from query `q` to every database item. It must equal the exact
# squared distance to each item's *reconstruction* — that's the check.

# %%
M = 4
sub = D // M
codebooks = np.zeros((M, 256, sub))
codes = np.zeros((N, M), dtype=np.uint8)
for m in range(M):
    codebooks[m], codes[:, m] = kmeans(X[:, m * sub:(m + 1) * sub], 256, seed=m)

def adc_dists(q, codebooks, codes):
    # >>> SOLUTION
    M, _, sub = codebooks.shape
    lut = np.stack([((codebooks[m] - q[m * sub:(m + 1) * sub]) ** 2).sum(1)
                    for m in range(M)])
    return lut[np.arange(M)[None, :], codes].sum(1)
    # <<< SOLUTION

q = Q[0]
recon = np.concatenate([codebooks[m][codes[:, m]] for m in range(M)], axis=1)
exact_to_recon = ((recon - q) ** 2).sum(1)
assert np.allclose(adc_dists(q, codebooks, codes), exact_to_recon, atol=1e-8)
print("adc_dists ✓ (equals distance to reconstructions)")

# %% [markdown]
# ## Task 2 — IVF search
# `ivf_search(q, n_probe)`: find the `n_probe` nearest coarse centroids, gather
# their inverted lists, score exactly, return top-10 ids. With
# `n_probe = n_list` it must match brute force; with fewer probes every result
# must come from the probed cells (brute force would pass the first check only).

# %%
NLIST = 32
cents, cell = kmeans(X, NLIST, seed=99)
lists = [np.where(cell == c)[0] for c in range(NLIST)]

def ivf_search(q, n_probe):
    # >>> SOLUTION
    probe = np.argsort(((cents - q) ** 2).sum(1))[:n_probe]
    cand = np.concatenate([lists[c] for c in probe])
    d = ((X[cand] - q) ** 2).sum(1)
    return cand[np.argsort(d)[:10]]
    # <<< SOLUTION

assert np.array_equal(np.sort(ivf_search(Q[0], NLIST)), np.sort(gold[0]))
for q in Q:
    for n_probe in (1, 2):
        probed = np.argsort(((cents - q) ** 2).sum(1))[:n_probe]
        hits = ivf_search(q, n_probe)
        assert np.isin(cell[hits], probed).all(), f"n_probe={n_probe}: a result lies outside the probed cells"
        in_probed = np.where(np.isin(cell, probed))[0]
        best = in_probed[np.argsort(((X[in_probed] - q) ** 2).sum(1))[:10]]
        assert np.array_equal(np.sort(hits), np.sort(best)), f"n_probe={n_probe}: not the exact top-10 of the probed cells"
rec1 = np.mean([len(set(ivf_search(q, 1)) & set(g)) / 10 for q, g in zip(Q, gold)])
rec4 = np.mean([len(set(ivf_search(q, 4)) & set(g)) / 10 for q, g in zip(Q, gold)])
print(f"ivf_search ✓   recall@10 with n_probe=1: {rec1:.2f}, n_probe=4: {rec4:.2f}")

# %% [markdown]
# ## Task 3 — post-filtering collapses on selective filters
# Give every vector a random tag (1% selectivity). Implement `post_filter`:
# search top-`k_search` *ignoring* tags, then keep matches. Measure recall
# against exact filtered search and watch it collapse.

# %%
tag = rng.integers(0, 100, N)
want = tag[np.argmin(l2sq(Q, X), axis=1)]

def filtered_gold(q, t):
    idx = np.where(tag == t)[0]
    return idx[np.argsort(((X[idx] - q) ** 2).sum(1))[:10]]

def post_filter(q, t, k_search=100):
    # >>> SOLUTION
    top = np.argsort(((X - q) ** 2).sum(1))[:k_search]
    return top[tag[top] == t][:10]
    # <<< SOLUTION

recs = [len(set(post_filter(q, t)) & set(filtered_gold(q, t))) / 10
        for q, t in zip(Q, want)]
post_recall = float(np.mean(recs))
print(f"post-filter recall@10 = {post_recall:.2f}   (pre-filter = 1.00 by construction)")
assert post_recall < 0.9, "post-filtering should visibly lose recall at 1% selectivity"
print("filtered-search trap ✓ demonstrated")

# %% [markdown]
# ## Task 4 (open) — the knee of the curve
# For your `ivf_search`, sweep `n_probe ∈ {1..32}` and find the smallest value
# reaching ≥ 0.95 recall@10. How does it change if you double `NLIST`? (Rule of
# thumb: n_list ≈ √N, then tune n_probe on *your* recall target.)
