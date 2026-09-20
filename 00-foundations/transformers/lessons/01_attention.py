"""
Lesson 1 - Attention from scratch, in numpy.

Read it top to bottom and run it. Every section prints the numbers it computes.
Companion: docs/transformer-primer.md, Section 2.

    python lessons/01_attention.py
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=120)
rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # subtracting a constant changes nothing; it just avoids overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Q: (n, d_k) queries   -- "what am I looking for?"
    K: (m, d_k) keys      -- "what do I contain, for matching?"
    V: (m, d_v) values    -- "what do I hand over if attended to?"
    Returns the (n, d_v) outputs and the (n, m) weight matrix.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)               # (n, m): how well query i matches key j
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # -inf becomes a weight of exactly 0 after softmax
    weights = softmax(scores)                     # every row is a probability distribution
    return weights @ V, weights                   # row i = weighted average of the values


# ---------------------------------------------------------------------------
print("=== 1. One query, three keys and values (the primer's toy example) ===")
q = np.array([[2.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

scores = q @ K.T                        # unscaled here, to match the primer's numbers
weights = softmax(scores)
print("scores  ", scores[0], "          <- the query matches key 1 best")
print("weights ", weights[0], f"   <- sum = {weights.sum():.2f}")
print("output  ", (weights @ V)[0], "     <- mostly v1 = [1 0], plus a little of the others")
print("Attention is an average, never a selection. The output is a blend.\n")

# ---------------------------------------------------------------------------
print("=== 2. Sharpness: softmax is a dial between 'average everything' and 'pick one' ===")
for scale in [0.1, 1.0, 5.0]:
    print(f"scores x {scale:<4}->  weights {softmax(scale * scores)[0]}")
print("Bigger scores -> sharper lookup. The scale of q.k is not a detail; it decides how the model reads.\n")

# ---------------------------------------------------------------------------
print("=== 3. Why divide by sqrt(d_k) ===")
d_k = 64
Q = rng.standard_normal((8, d_k))              # 8 queries, 8 keys, random like an untrained model
K = rng.standard_normal((8, d_k))
raw = Q @ K.T
print(f"typical size of a raw dot product: {raw.std():.1f}   (it grows like sqrt(d_k) = {np.sqrt(d_k):.1f})")
print("largest weight per row WITHOUT scaling:", softmax(raw).max(axis=1)[:5], "... <- near one-hot: gradients vanish")
print("largest weight per row WITH    scaling:", softmax(raw / np.sqrt(d_k)).max(axis=1)[:5], "... <- still soft")
print("Unscaled, softmax saturates before training even starts. sqrt(d_k) keeps the scores O(1).\n")

# ---------------------------------------------------------------------------
print("=== 4. Self-attention on a tiny sequence ===")
n, d = 5, 8                                    # 5 tokens, each an 8-dimensional vector
X = rng.standard_normal((n, d))                # stand-in for token embeddings
Wq, Wk, Wv = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv               # every token makes its own query, key and value
out, weights = attention(Q, K, V)
print("attention weights (row i = where token i reads from):")
print(weights)
print("row sums:", weights.sum(axis=1))
print("output shape:", out.shape, "<- same as the input: n tokens x d features\n")

# ---------------------------------------------------------------------------
print("=== 5. Attention has no idea what order the tokens are in ===")
perm = rng.permutation(n)
Xs = X[perm]                                                        # shuffle the tokens
out_s, _ = attention(Xs @ Wq, Xs @ Wk, Xs @ Wv)
print("shuffle order:", perm)
print("outputs are the same rows, shuffled the same way:", np.allclose(out_s, out[perm]))

pos = rng.standard_normal((n, d)) * 0.5                             # one vector per SLOT, not per token
Xp = X + pos                                                        # add position before attention
out_p, _ = attention(Xp @ Wq, Xp @ Wk, Xp @ Wv)
Xps = X[perm] + pos                                                 # slot 0 stays "first" whoever sits there
out_ps, _ = attention(Xps @ Wq, Xps @ Wk, Xps @ Wv)
print("with positions added, shuffling changes the result:", not np.allclose(out_ps, out_p[perm]))
print("=> The core mechanism is order-blind. Position has to be injected on purpose.\n")

# ---------------------------------------------------------------------------
print("=== 6. The causal mask: a token may only look backwards ===")
mask = np.tril(np.ones((n, n), dtype=bool))                         # True where attention is allowed
print(mask.astype(int))
out_c, weights_c = attention(Q, K, V, mask=mask)
print("masked attention weights (exact zeros above the diagonal):")
print(weights_c)
print("row sums:", weights_c.sum(axis=1), " <- still distributions, just over fewer tokens")
print("Token 0 only sees itself; token 4 sees everything. This is what makes next-token training parallel.")
