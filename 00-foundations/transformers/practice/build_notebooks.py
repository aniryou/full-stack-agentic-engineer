"""
Generates attention_practice.ipynb (blanks) and attention_solutions.ipynb (filled in) from one spec,
so the two never drift apart. Run:  python practice/build_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = '''import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
rng = np.random.default_rng(0)

# --- provided helpers (no blanks here) ---
def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def init_block(d, h, std=None):
    std = std or 1 / np.sqrt(d)
    W = lambda a, b: rng.standard_normal((a, b)) * std
    return dict(Wq=W(d, d), Wk=W(d, d), Wv=W(d, d), Wo=W(d, d), W1=W(d, 4 * d), W2=W(4 * d, d), h=h)

def mlp(x, p):
    return gelu(x @ p["W1"]) @ p["W2"]

print("ready")'''

# Each exercise: (title/markdown, solution code, practice code, check code)
EXERCISES = [
(
"""## 1. Softmax

Turn a row of scores into a probability distribution: exponentiate, then divide by the row sum.
The max-subtraction is already done for you; it changes nothing mathematically and prevents overflow.""",
'''def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # stability trick (given)
    e = np.exp(x)                              # exponentiate
    return e / e.sum(axis=axis, keepdims=True) # normalise: each row sums to 1''',
'''def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # stability trick (given)
    e = ...                                    # exponentiate
    return ...                                 # normalise: each row sums to 1''',
'''s = softmax(np.array([[2.0, 0.0, 0.0]]))
assert np.allclose(s.sum(axis=-1), 1), f"rows must sum to 1, got {s}"
assert np.allclose(s, [[0.787, 0.107, 0.107]], atol=1e-3), f"expected [0.79 0.11 0.11], got {s}"
assert np.allclose(softmax(np.array([[1000.0, 1000.0]])), [[0.5, 0.5]]), "must not overflow on large inputs"
print("✅ softmax")''',
),
(
"""## 2. Scaled dot-product attention

`Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V`

- `scores[i, j]` = how well query *i* matches key *j*, divided by √d_k
- `weights` = softmax over each row
- `out[i]` = weighted average of the value vectors""",
'''def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)              # (n, m)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # (given) -inf -> weight 0
    weights = softmax(scores)                     # each row is a distribution
    out = weights @ V                             # weighted average of values
    return out, weights''',
'''def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = ...                                  # (n, m): every query against every key, scaled by sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # (given) -inf -> weight 0
    weights = ...                                 # each row is a distribution
    out = ...                                     # weighted average of values
    return out, weights''',
'''Q = np.array([[2.0, 0.0]]) * np.sqrt(2)          # scaled so the scores come out as [2, 0, 0]
K = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
out, w = attention(Q, K, V)
assert np.allclose(w, [[0.787, 0.107, 0.107]], atol=1e-3), f"weights wrong: {w}  (did you scale by sqrt(d_k)?)"
assert np.allclose(out, [[0.894, 0.213]], atol=1e-3), f"output wrong: {out}"
# a sharp query should retrieve one value almost exactly: this is the 'lookup' behaviour
keys = np.eye(4); vals = np.arange(4.0).reshape(4, 1)
out, _ = attention(20 * keys[2:3], keys, vals)
assert np.allclose(out, [[2.0]], atol=1e-2), f"query = key 2 (sharp) should retrieve value 2, got {out}"
print("✅ attention")''',
),
(
"""## 3. The causal mask

Return an `(n, n)` boolean array that is `True` where token *i* is allowed to attend to token *j*:
a token may look at itself and at everything before it, never ahead.""",
'''def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))''',
'''def causal_mask(n):
    return ...   # (n, n) bool: True where j <= i''',
'''m = causal_mask(3)
assert m.dtype == bool and m.shape == (3, 3), "must be an (n, n) boolean array"
assert m.tolist() == [[True, False, False], [True, True, False], [True, True, True]], f"got {m.astype(int)}"
X = rng.standard_normal((5, 4))
_, w = attention(X, X, X, mask=causal_mask(5))
assert np.all(w[np.triu_indices(5, k=1)] == 0), "weights above the diagonal must be exactly 0"
assert np.allclose(w.sum(axis=1), 1), "rows must still sum to 1"
print("✅ causal mask")''',
),
(
"""## 4. Splitting into heads

Heads are just slices of the feature dimension. Reshape `(n, d)` into `(h, n, d/h)` so each head has its own
`(n, d_head)` matrix, and write the inverse that concatenates them back. No arithmetic, just reshapes and a transpose.""",
'''def split_heads(x, h):
    n, d = x.shape
    return x.reshape(n, h, d // h).transpose(1, 0, 2)   # (n, d) -> (n, h, dh) -> (h, n, dh)

def merge_heads(x):
    h, n, dh = x.shape
    return x.transpose(1, 0, 2).reshape(n, h * dh)       # (h, n, dh) -> (n, h, dh) -> (n, d)''',
'''def split_heads(x, h):
    n, d = x.shape
    return ...   # (n, d) -> (h, n, d // h)

def merge_heads(x):
    h, n, dh = x.shape
    return ...   # (h, n, dh) -> (n, h * dh)''',
'''X = rng.standard_normal((5, 8))
S = split_heads(X, 2)
assert S.shape == (2, 5, 4), f"expected (2, 5, 4), got {S.shape}"
assert np.allclose(S[0], X[:, :4]), "head 0 should be the first half of the features"
assert np.allclose(S[1], X[:, 4:]), "head 1 should be the second half of the features"
assert np.allclose(merge_heads(S), X), "merge_heads(split_heads(X)) must give X back"
print("✅ heads")''',
),
(
"""## 5. Multi-head attention

Project the input into Q, K, V, split each into heads, run `attention` per head with the causal mask,
merge the heads, and apply the output projection `Wo`.""",
'''def multi_head_attention(x, p, causal=True):
    n, d = x.shape
    h = p["h"]
    Q, K, V = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
    Q, K, V = split_heads(Q, h), split_heads(K, h), split_heads(V, h)
    mask = causal_mask(n) if causal else None
    heads = [attention(Q[i], K[i], V[i], mask)[0] for i in range(h)]   # one attention per head
    out = merge_heads(np.stack(heads))
    return out @ p["Wo"]''',
'''def multi_head_attention(x, p, causal=True):
    n, d = x.shape
    h = p["h"]
    Q, K, V = ..., ..., ...                                    # project x with Wq, Wk, Wv
    Q, K, V = split_heads(Q, h), split_heads(K, h), split_heads(V, h)
    mask = causal_mask(n) if causal else None
    heads = [attention(Q[i], K[i], V[i], mask)[0] for i in range(h)]   # one attention per head (given)
    out = merge_heads(np.stack(heads))
    return ...                                                 # output projection''',
'''def _reference_mha(x, p, causal=True):
    n, d = x.shape; h = p["h"]; dh = d // h
    Q, K, V = (split_heads(x @ p[k], h) for k in ("Wq", "Wk", "Wv"))
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal: s = np.where(causal_mask(n), s, -np.inf)
    return merge_heads(softmax(s) @ V) @ p["Wo"]

p = init_block(d=16, h=4)
X = rng.standard_normal((6, 16))
out = multi_head_attention(X, p)
assert out.shape == (6, 16), f"expected (6, 16), got {out.shape}"
assert np.allclose(out, _reference_mha(X, p)), "output differs from the reference implementation"
print("✅ multi-head attention")''',
),
(
"""## 6. The block

Two residual updates to the stream, each on a normalised copy of it (pre-norm):

    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))""",
'''def block(x, p):
    x = x + multi_head_attention(layer_norm(x), p)   # mix across positions
    x = x + mlp(layer_norm(x), p)                     # compute within each position
    return x''',
'''def block(x, p):
    x = x + ...     # attention over the normalised stream
    x = x + ...     # MLP over the normalised stream
    return x''',
'''p = init_block(d=16, h=4)
X = rng.standard_normal((6, 16))
Y = block(X, p)
assert Y.shape == X.shape, "a block keeps the shape (n, d)"
u1 = multi_head_attention(layer_norm(X), p)
u2 = mlp(layer_norm(X + u1), p)
assert np.allclose(Y, X + u1 + u2), "expected x + attention_update + mlp_update (pre-norm, residual)"
# causality survives the whole block: nudging token 3 must not change tokens 0-2
X2 = X.copy(); X2[3, 0] += 1.0   # one feature, not a constant across all of them: layer norm would subtract that out
changed = np.where(np.abs(block(X2, p) - Y).max(axis=1) > 1e-9)[0]
assert changed.tolist() == [3, 4, 5], f"tokens changed by nudging token 3: {changed} (expected [3 4 5])"
print("✅ block")''',
),
(
"""## 7. Count the parameters

Per block: attention has four `d × d` matrices, the MLP has `d × 4d` and `4d × d`. Add the token embedding table
(`vocab × d`, counted twice if untied) and the position table (`ctx × d`). Ignore biases and norms.""",
'''def gpt_params(d, L, vocab, ctx, tied=True):
    per_block = 12 * d * d                       # 4 d^2 attention + 8 d^2 MLP
    embeddings = vocab * d * (1 if tied else 2)
    positions = ctx * d
    return L * per_block + embeddings + positions''',
'''def gpt_params(d, L, vocab, ctx, tied=True):
    per_block = ...                              # attention + MLP
    embeddings = ...                             # token table, twice if untied
    positions = ...
    return L * per_block + embeddings + positions''',
'''small = gpt_params(768, 12, 50257, 1024)
assert abs(small - 124.4e6) < 0.5e6, f"GPT-2 small should be ~124M, got {small / 1e6:.1f}M"
xl = gpt_params(1600, 48, 50257, 1024)
assert abs(xl - 1.56e9) < 0.05e9, f"GPT-2 XL should be ~1.56B, got {xl / 1e9:.2f}B"
assert gpt_params(768, 12, 50257, 1024, tied=False) - small == 50257 * 768, "untied adds one more vocab x d table"
print("✅ parameter count")''',
),
(
"""## 8. Prove it to yourself: attention is order-blind

Nothing to fill in (it uses your exercises 1-5). Run it, then answer the question in the next cell.""",
'''X = rng.standard_normal((5, 8))
p = init_block(d=8, h=2)
perm = rng.permutation(5)
out = multi_head_attention(X, p, causal=False)
out_shuffled = multi_head_attention(X[perm], p, causal=False)
print("shuffled input gives shuffled output:", np.allclose(out_shuffled, out[perm]))

pos = rng.standard_normal((5, 8))                     # one vector per slot
out_pos = multi_head_attention(X + pos, p, causal=False)
out_pos_shuffled = multi_head_attention(X[perm] + pos, p, causal=False)
print("with positions added, that stops being true:", not np.allclose(out_pos_shuffled, out_pos[perm]))''',
None,
None,
),
(
"""**Question.** Section 3 of this notebook used a *causal* mask. Would the shuffle test above still pass with `causal=True`? Try it, then explain why in one sentence.

<details><summary>Answer</summary>

No. The causal mask is defined in terms of positions (token *i* may see tokens ≤ *i*), so it already injects order: after shuffling, a token is allowed to see a different set of neighbours. Even without a positional embedding, a causal Transformer is not fully order-blind.
</details>""",
None, None, None,
),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def build(solution):
    title = "# Attention practice" + (" — solutions" if solution else "")
    intro = """Companion to `docs/transformer-primer.md` (Sections 2, 3, 8) and `lessons/01_attention.py`, `lessons/02_block.py`.

Each exercise has a cell with `...` blanks to fill in, followed by a check cell. Run the check; it prints ✅ when your
implementation is right and tells you what's off when it isn't. Later exercises use earlier ones, so go in order.
Solutions are in `attention_solutions.ipynb`; try to get each check to pass before looking."""
    cells = [md(title + "\n\n" + intro), md("## Setup"), code(SETUP)]
    for text, sol, prac, check in EXERCISES:
        cells.append(md(text))
        body = sol if solution else (prac if prac is not None else sol)
        if body is not None:
            cells.append(code(body))
        if check is not None:
            cells.append(code(check))
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    (HERE / "attention_practice.ipynb").write_text(json.dumps(build(solution=False), indent=1))
    (HERE / "attention_solutions.ipynb").write_text(json.dumps(build(solution=True), indent=1))
    print("wrote attention_practice.ipynb and attention_solutions.ipynb")
