# %% [markdown]
# # Exercises 02 · Contrastive training
# Implement the InfoNCE machinery you used in notebook 02, then look at what
# in-batch "hard negatives" actually are. Solutions: `solutions/ex02_solutions.ipynb`.

# %%
import numpy as np
rng = np.random.default_rng(0)

def softmax(S, axis):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# %% [markdown]
# ## Task 1 — symmetric InfoNCE from the similarity matrix
# Given `S = Z_a Z_bᵀ / τ` (B×B, positives on the diagonal), return
# `(loss, dL/dS)` where the loss averages row-wise and column-wise
# cross-entropy toward the diagonal. Hint: for softmax-CE,
# `dL/dS = (softmax(S) − I) / B`, averaged over the two directions.

# %%
def info_nce_from_S(S):
    # >>> SOLUTION
    Bn = len(S)
    Pr, Pc = softmax(S, 1), softmax(S, 0)
    idx = np.arange(Bn)
    loss = -0.5 * (np.log(Pr[idx, idx]).mean() + np.log(Pc[idx, idx]).mean())
    I = np.eye(Bn)
    dS = 0.5 * ((Pr - I) + (Pc - I)) / Bn
    return loss, dS
    # <<< SOLUTION

# self-check: finite differences on a random 5×5 S
S = rng.normal(size=(5, 5))
loss, dS = info_nce_from_S(S)
num = np.zeros_like(S); eps = 1e-6
for i in range(5):
    for j in range(5):
        Sp, Sm = S.copy(), S.copy()
        Sp[i, j] += eps; Sm[i, j] -= eps
        num[i, j] = (info_nce_from_S(Sp)[0] - info_nce_from_S(Sm)[0]) / (2 * eps)
assert np.abs(num - dS).max() < 1e-6
assert loss > 0
print("info_nce_from_S ✓")

# %% [markdown]
# ## Task 2 — alignment & uniformity (Wang & Isola 2020)
# `alignment(Za, Zb)` = mean squared distance between positive pairs (unit
# vectors). `uniformity(Z)` = `log E exp(−2‖z_i − z_j‖²)` over random pairs
# `i ≠ j`. Lower is better for both.

# %%
def alignment(Za, Zb):
    # >>> SOLUTION
    return float((np.linalg.norm(Za - Zb, axis=1) ** 2).mean())
    # <<< SOLUTION

def uniformity(Z, m=4000, seed=0):
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(Z), m), r.integers(0, len(Z), m)
    i, j = i[i != j], j[i != j]
    # >>> SOLUTION
    d2 = np.linalg.norm(Z[i] - Z[j], axis=1) ** 2
    return float(np.log(np.exp(-2 * d2).mean()))
    # <<< SOLUTION

# self-checks: perfect alignment → 0; two antipodal points → uniformity = −8
z = np.array([[1.0, 0.0]])
assert np.isclose(alignment(z, z), 0.0)
Z2 = np.array([[1.0, 0.0], [-1.0, 0.0]])
assert np.isclose(uniformity(Z2), -8.0), uniformity(Z2)
print("alignment/uniformity ✓")

# %% [markdown]
# ## Task 3 — who are the in-batch hard negatives?
# `hardest_negative(S)`: for each row, the index of the highest-scoring
# *off-diagonal* entry. Then inspect: with topic-structured data, hard negatives
# are same-topic paragraphs — which is exactly why they carry the most gradient.

# %%
def hardest_negative(S):
    # >>> SOLUTION
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    return np.argmax(S2, axis=1)
    # <<< SOLUTION

Sx = np.array([[9.0, 0.2, 0.8], [0.1, 9.0, 0.5], [0.9, 0.3, 9.0]])
assert np.array_equal(hardest_negative(Sx), [2, 2, 0])
print("hardest_negative ✓")

# %% [markdown]
# ## Task 4 (stretch) — a Matryoshka loss
# `matryoshka_loss(Ua, Ub, dims, ...)`: sum InfoNCE over *prefixes* of the
# unnormalized embeddings (normalize each prefix before the similarity). With
# `dims=[full]` it must equal the plain loss on normalized vectors.

# %%
def matryoshka_loss(Ua, Ub, dims, tau=0.05):
    # >>> SOLUTION
    total = 0.0
    for k in dims:
        Za = Ua[:, :k] / (np.linalg.norm(Ua[:, :k], axis=1, keepdims=True) + 1e-9)
        Zb = Ub[:, :k] / (np.linalg.norm(Ub[:, :k], axis=1, keepdims=True) + 1e-9)
        total += info_nce_from_S(Za @ Zb.T / tau)[0]
    return total
    # <<< SOLUTION

Ua, Ub = rng.normal(size=(6, 16)), rng.normal(size=(6, 16))
Za = Ua / np.linalg.norm(Ua, axis=1, keepdims=True)
Zb = Ub / np.linalg.norm(Ub, axis=1, keepdims=True)
assert np.isclose(matryoshka_loss(Ua, Ub, [16]),
                  info_nce_from_S(Za @ Zb.T / 0.05)[0])
assert matryoshka_loss(Ua, Ub, [4, 8, 16]) > matryoshka_loss(Ua, Ub, [16])
print("matryoshka_loss ✓ — in notebook 02, swap this into train() and compare "
      "recall@5 after truncating to 8 dims.")
