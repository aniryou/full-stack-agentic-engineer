# %% [markdown]
# # Exercises 01 · Counts to vectors
# Fill in each `YOUR CODE HERE` block. Every task has a self-check cell — run it
# to verify. Solutions: `solutions/ex01_solutions.ipynb`.

# %%
import numpy as np
rng = np.random.default_rng(0)

# %% [markdown]
# ## Task 1 — implement PPMI
# `ppmi(C)[i,j] = max(0, log( p(i,j) / (p(i)·p(j)) ))`, with unseen pairs → 0.

# %%
def ppmi(C):
    # >>> SOLUTION
    total = C.sum()
    pw = C.sum(1, keepdims=True) / total
    pc = C.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / (pw * pc))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)
    # <<< SOLUTION

# self-check: C = [[0,4],[4,0]] → p(0,1)=1/2, p(0)=p(1)=1/2 → PMI = log 2
out = ppmi(np.array([[0.0, 4.0], [4.0, 0.0]]))
assert np.allclose(out, [[0, np.log(2)], [np.log(2), 0]]), out
print("ppmi ✓")

# %% [markdown]
# ## Task 2 — the SGNS gradient
# For one (center `v`, positive `u⁺`, negatives `U⁻` of shape (K,d)) example:
# `L = −log σ(v·u⁺) − Σ_k log σ(−v·u⁻_k)`.
# Return `(dL/dv, dL/du⁺, dL/dU⁻)`. Hint: both gradients are sigmoid residuals
# times the *other* vector.

# %%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sgns_grads(v, upos, Uneg):
    # >>> SOLUTION
    gpos = sigmoid(v @ upos) - 1.0            # scalar
    gneg = sigmoid(Uneg @ v)                  # (K,)
    dv = gpos * upos + gneg @ Uneg
    dupos = gpos * v
    dUneg = gneg[:, None] * v[None, :]
    return dv, dupos, dUneg
    # <<< SOLUTION

# self-check: finite differences
def sgns_loss(v, upos, Uneg):
    return -np.log(sigmoid(v @ upos)) - np.log(sigmoid(-Uneg @ v)).sum()

v, upos, Uneg = rng.normal(size=4), rng.normal(size=4), rng.normal(size=(3, 4))
dv, dupos, dUneg = sgns_grads(v, upos, Uneg)
eps = 1e-6
for arr, g, name in [(v, dv, "dv"), (upos, dupos, "dupos"), (Uneg, dUneg, "dUneg")]:
    num = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        k = it.multi_index
        old = arr[k]
        arr[k] = old + eps; lp = sgns_loss(v, upos, Uneg)
        arr[k] = old - eps; lm = sgns_loss(v, upos, Uneg)
        arr[k] = old
        num[k] = (lp - lm) / (2 * eps)
        it.iternext()
    assert np.abs(num - g).max() < 1e-5, name
print("sgns_grads ✓ (matches finite differences)")

# %% [markdown]
# ## Task 3 — the analogy function
# `analogy(a, b, c)`: nearest word to `a − b + c` by cosine, **excluding** a, b,
# c. Uses the vectors saved by notebook 01 (run it first).

# %%
art = np.load("../artifacts/word_vectors.npz", allow_pickle=True)
W = art["W_svd"]; vocab = list(art["vocab"]); w2i = {w: i for i, w in enumerate(vocab)}

def analogy(a, b, c, W=W):
    # >>> SOLUTION
    Z = W / np.linalg.norm(W, axis=1, keepdims=True)
    t = Z[w2i[a]] - Z[w2i[b]] + Z[w2i[c]]
    t /= np.linalg.norm(t)
    for i in np.argsort(-(Z @ t)):
        if vocab[i] not in {a, b, c}:
            return vocab[i]
    # <<< SOLUTION

assert analogy("king", "man", "woman") == "queen", analogy("king", "man", "woman")
print("analogy ✓ :", "king − man + woman =", analogy("king", "man", "woman"),
      "| prince − boy + girl =", analogy("prince", "boy", "girl"))

# %% [markdown]
# ## Task 4 (open) — window size changes what "similar" means
# Rebuild notebook 01's co-occurrence with `WINDOW = 1` and `WINDOW = 8` and
# compare neighbours of `king`. Small windows → substitutable words (other
# people); large windows → topical associates (palace, throne). No assert —
# write two sentences on what you observe.
