"""
Lesson 2 - A complete Transformer block, in numpy.

Builds on lesson 1. The block does two things, and the experiments below make each visible:
  1. attention  - moves information BETWEEN positions
  2. the MLP    - processes information WITHIN a position
Companion: docs/transformer-primer.md, Sections 3 and 8.

    python lessons/02_block.py
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=120)
rng = np.random.default_rng(0)


# --- the pieces ---------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    """Rescale each token's vector to mean 0, variance 1. (Real models also learn a gain and bias.)"""
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def init_block(d, h, std):
    """Every weight in one block. Same shapes as GPT-2; biases dropped for clarity."""
    assert d % h == 0, "d_model must split evenly into heads"
    W = lambda a, b: rng.standard_normal((a, b)) * std
    return dict(Wq=W(d, d), Wk=W(d, d), Wv=W(d, d), Wo=W(d, d),   # attention
                W1=W(d, 4 * d), W2=W(4 * d, d),                    # MLP: expand 4x, then project back
                h=h)


def multi_head_attention(x, p, causal=True):
    """x: (n, d) -> (n, d). Also returns the (h, n, n) attention weights, one pattern per head."""
    n, d = x.shape
    h, dh = p["h"], d // p["h"]
    Q, K, V = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
    split = lambda t: t.reshape(n, h, dh).transpose(1, 0, 2)        # (n, d) -> (h, n, dh): heads are just slices
    Q, K, V = split(Q), split(K), split(V)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)                  # (h, n, n)
    if causal:
        scores = np.where(np.tril(np.ones((n, n), bool)), scores, -np.inf)
    weights = softmax(scores)
    out = (weights @ V).transpose(1, 0, 2).reshape(n, d)             # (h, n, dh) -> (n, d): concatenate heads
    return out @ p["Wo"], weights


def mlp(x, p):
    """Applied to every token separately, with the same weights. No token sees another here."""
    return gelu(x @ p["W1"]) @ p["W2"]


def block(x, p):
    """The whole block: two residual updates to the stream x."""
    x = x + multi_head_attention(layer_norm(x), p)[0]   # mix across positions
    x = x + mlp(layer_norm(x), p)                        # compute within each position
    return x


# --- experiments ---------------------------------------------------------------
n, d, h, L = 6, 16, 4, 3
x = rng.standard_normal((n, d))
layers = [init_block(d, h, std=1 / np.sqrt(d)) for _ in range(L)]   # random weights; GPT-2 uses std=0.02

print("=== 1. Same shape in, same shape out, so blocks stack ===")
y = x
for p in layers:
    y = block(y, p)
print(f"in {x.shape} -> after {L} blocks {y.shape}")
print("Every block is a function (n, d) -> (n, d). Depth is just how many times you apply one.\n")

print("=== 2. The residual stream: blocks ADD to x, they never replace it ===")
p = layers[0]
u1 = multi_head_attention(layer_norm(x), p)[0]
u2 = mlp(layer_norm(x + u1), p)
print("block(x) == x + attention_update + mlp_update :", np.allclose(block(x, p), x + u1 + u2))
print(f"|x| = {np.linalg.norm(x):.1f}   |attention update| = {np.linalg.norm(u1):.1f}   |mlp update| = {np.linalg.norm(u2):.1f}")
print("The input is still in there. Later blocks read the sum of everything earlier blocks wrote.")
print("(With GPT-2's real init, std=0.02, both updates start near zero; training is what grows them.)\n")

print("=== 3. Who is allowed to affect whom? Nudge token 2 and see which rows change ===")
x2 = x.copy()
x2[2] += 1.0
changed = lambda a, b: np.where(np.abs(a - b).max(axis=1) > 1e-9)[0]
print("MLP                   : rows changed =", changed(mlp(x, p), mlp(x2, p)), "        <- only token 2 itself")
print("attention, causal     : rows changed =", changed(multi_head_attention(x, p)[0],
                                                          multi_head_attention(x2, p)[0]), "<- token 2 and everyone AFTER it")
print("attention, no mask    : rows changed =", changed(multi_head_attention(x, p, causal=False)[0],
                                                          multi_head_attention(x2, p, causal=False)[0]), "<- everyone")
print("Attention is the only place tokens interact. The causal mask makes that interaction one-directional.\n")

print("=== 4. Heads: several attention patterns computed at once ===")
_, weights = multi_head_attention(x, p)
print(f"weights shape: {weights.shape}  <- (heads, n, n)")
for i in range(2):
    print(f"head {i}:")
    print(weights[i])
print("Different heads, different patterns, same input. That is the point of having several.\n")

print("=== 5. Where the parameters go ===")
n_params = sum(v.size for v in p.values() if isinstance(v, np.ndarray))
print(f"one block, d={d}: {n_params} parameters = 12 * d^2 = {12 * d * d}")
attn = 4 * d * d
mlp_ = 8 * d * d
print(f"  attention {attn} ({attn / n_params:.0%})   MLP {mlp_} ({mlp_ / n_params:.0%})   <- two thirds live in the MLP")


def gpt_params(d, L, vocab, ctx, tied=True):
    per_block = 12 * d * d
    embeddings = vocab * d * (1 if tied else 2)
    positions = ctx * d
    return L * per_block + embeddings + positions


print(f"GPT-2 small (d=768, L=12, vocab=50257):     {gpt_params(768, 12, 50257, 1024) / 1e6:.1f}M   <- the real number is 124M")
print(f"GPT-2 XL    (d=1600, L=48):                 {gpt_params(1600, 48, 50257, 1024) / 1e6:.0f}M    <- the real number is 1.5B")
print("Read d and L off a config file and you can size any model in your head.\n")

print("=== 6. When does attention's n^2 cost bite? ===")
d_big = 4096
print("per layer, d=4096:   linear-in-n FLOPs (projections + MLP) vs quadratic (the n x n scores)")
for n_ctx in [1_000, 8_000, 24_000, 100_000]:
    linear = 24 * n_ctx * d_big**2          # 2 FLOPs per multiply-add; 4 attention projections + MLP (8 d^2)
    quadratic = 4 * n_ctx**2 * d_big        # Q K^T and weights @ V
    print(f"  n = {n_ctx:>7,}: quadratic share = {quadratic / (linear + quadratic):.0%}")
print("Below ~6*d tokens the model is mostly matmuls that scale linearly. Memory for the n x n matrix bites earlier.")
