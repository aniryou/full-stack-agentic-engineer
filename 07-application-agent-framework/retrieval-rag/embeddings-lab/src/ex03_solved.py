# %% [markdown]
# # Exercises 03 · Geometry diagnostics
# Implement three diagnostics you can run on any embedding matrix at work.
# Solutions: `solutions/ex03_solutions.ipynb`.

# %%
import numpy as np
rng = np.random.default_rng(0)

# %% [markdown]
# ## Task 1 — participation ratio (effective dimensionality)
# `PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²` over eigenvalues of the covariance of `X`.
# Isotropic d-dim data → ≈ d; rank-1 data → ≈ 1.

# %%
def participation_ratio(X):
    # >>> SOLUTION
    Xc = X - X.mean(0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())
    # <<< SOLUTION

iso = rng.normal(size=(3000, 10))
r1 = np.outer(rng.normal(size=3000), rng.normal(size=10)) + 1e-3 * rng.normal(size=(3000, 10))
assert 8.5 < participation_ratio(iso) < 10.5
assert participation_ratio(r1) < 1.5
print(f"participation_ratio ✓  (isotropic≈{participation_ratio(iso):.1f}, rank-1≈{participation_ratio(r1):.2f})")

# %% [markdown]
# ## Task 2 — CSLS rescoring (the hubness fix)
# `csls(S, k) = 2·S − r(row) − r(col)`, where `r(x)` is the mean of x's top-k
# similarities (diagonal excluded). Penalizes points that are close to
# *everything*.

# %%
def csls(S, k=10):
    # >>> SOLUTION
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    r = np.sort(S2, axis=1)[:, -k:].mean(1)
    return 2 * S2 - r[None, :] - r[:, None]
    # <<< SOLUTION

St = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
out = csls(St, k=1)          # r = [0.9, 0.9, 0.2]
assert np.isclose(out[0, 1], 2 * 0.9 - 0.9 - 0.9)
assert np.isclose(out[0, 2], 2 * 0.1 - 0.9 - 0.2)
print("csls ✓")

# %% [markdown]
# ## Task 3 — All-but-the-Top (Mu & Viswanath 2018)
# Center `X`, then remove its projection onto the top `n_pc` principal
# components. Returns the corrected matrix.

# %%
def all_but_the_top(X, n_pc=2):
    # >>> SOLUTION
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc - (Xc @ Vt[:n_pc].T) @ Vt[:n_pc]
    # <<< SOLUTION

X = rng.normal(size=(500, 20)) + 5 * np.outer(rng.normal(size=500), np.ones(20))
Y = all_but_the_top(X, 1)
assert np.allclose(Y.mean(0), 0, atol=1e-8)
top_dir = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
assert np.abs(Y @ top_dir).max() < 1e-6      # no variance left along old top PC
print("all_but_the_top ✓")

# %% [markdown]
# ## Task 4 (open) — apply them
# Load `../artifacts/word_vectors.npz` and report: PR of the SGNS vectors, the
# five biggest hubs before/after CSLS, and whether all-but-the-top changes the
# related-vs-unrelated AUC from notebook 03. Two or three sentences of findings.
