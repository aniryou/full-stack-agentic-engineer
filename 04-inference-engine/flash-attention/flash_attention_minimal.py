"""
FlashAttention from scratch, in plain numpy.

Run it:  python flash_attention_minimal.py

Six parts, each building on the last:
  1. Naive attention               -- the thing we're replacing
  2. Why softmax needs the max     -- the numerical constraint
  3. Online softmax                -- the trick, isolated from attention
  4. Merging partial attention     -- the trick, applied to attention
  5. Tiled flash attention         -- the whole algorithm, ~20 lines
  6. What it actually bought us    -- memory accounting

No GPU, no CUDA, no Triton. The point is the algorithm, not the kernel.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


# =====================================================================
# PART 1 -- Naive attention
# =====================================================================
# The textbook three-liner. Correct, and the baseline every other
# version below must match exactly.

def naive_attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)                  # (N, N)  <-- the matrix we want to avoid
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)     # (N, N)  <-- and this one
    return P @ V                              # (N, d)


banner("PART 1 -- Naive attention")

N, d = 8, 4                                   # tiny, so we can print things
Q = np.random.randn(N, d)
K = np.random.randn(N, d)
V = np.random.randn(N, d)

O_naive = naive_attention(Q, K, V)
print(f"Q, K, V shapes : {Q.shape}         ({N * d} numbers each)")
print(f"S, P shapes    : ({N}, {N})        ({N * N} numbers each)  <-- quadratic")
print(f"Output shape   : {O_naive.shape}")
print(f"\nAt N=8 nobody cares. At N=32768 the S matrix alone is "
      f"{32768 ** 2 * 2 / 1e9:.1f} GB in fp16, per head, per sequence.")


# =====================================================================
# PART 2 -- Why softmax needs the running max
# =====================================================================
# Before the tiling trick makes sense, you need to know why every real
# softmax subtracts the max first. It's not cosmetic.

banner("PART 2 -- Why softmax subtracts the max")

big_scores = np.array([100.0, 101.0, 102.0], dtype=np.float32)

with np.errstate(over='ignore', invalid='ignore'):      # the overflow is the point
    naive_exp = np.exp(big_scores)                      # overflows
    naive_softmax = naive_exp / naive_exp.sum()
stable_exp = np.exp(big_scores - big_scores.max())      # fine

print(f"scores                    : {big_scores}")
print(f"exp(scores)               : {naive_exp}          <-- inf, in float32")
print(f"exp(scores - max)         : {stable_exp}")
print(f"\nsoftmax via naive exp     : {naive_softmax}   <-- nan")
print(f"softmax via stable exp    : {stable_exp / stable_exp.sum()}")
print("\nSubtracting the max changes nothing mathematically (the factor cancels")
print("in the ratio) but everything numerically. This is why the algorithm has")
print("to track a running MAX, not just a running sum.")


# =====================================================================
# PART 3 -- Online softmax, isolated
# =====================================================================
# The whole difficulty with tiling attention is that softmax normalizes
# across a full row. Here is the fix, with attention stripped away:
# compute the softmax statistics (max and sum) in a single streaming pass
# over blocks, never holding the full vector.

def online_softmax_stats(x, block_size):
    """Stream over x in blocks, maintaining (running max, running sum-of-exp)."""
    m = -np.inf      # running max
    l = 0.0          # running sum of exp(x - m)

    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]

        m_new = max(m, block.max())
        alpha = np.exp(m - m_new)            # <-- THE correction factor

        # rescale everything accumulated so far, then add this block
        l = alpha * l + np.exp(block - m_new).sum()
        m = m_new

    return m, l


banner("PART 3 -- Online softmax, isolated from attention")

x = np.array([1.0, 3.0, 5.0, 2.0])

# what the one-shot version gives
m_true = x.max()
l_true = np.exp(x - m_true).sum()

print("Streaming over x =", x, "in blocks of 2:\n")

# narrate it by hand so the correction factor is visible
m, l = -np.inf, 0.0
for start in range(0, len(x), 2):
    block = x[start:start + 2]
    m_new = max(m, block.max())
    alpha = np.exp(m - m_new)
    l_new = alpha * l + np.exp(block - m_new).sum()
    print(f"  block {block}:  m {m:>6.2f} -> {m_new:.2f},  "
          f"alpha = exp({m:.2f} - {m_new:.2f}) = {alpha:.4f},  "
          f"l {l:.4f} -> {l_new:.4f}")
    m, l = m_new, l_new

print(f"\n  streaming : m = {m:.4f},  l = {l:.4f}")
print(f"  one-shot  : m = {m_true:.4f},  l = {l_true:.4f}")
print("\nIdentical. When a later block holds a bigger value, everything")
print("accumulated so far was exponentiated against a stale max -- off by")
print("exactly exp(m_old - m_new). Multiply through and the books balance.")

m_chk, l_chk = online_softmax_stats(np.random.randn(1000), block_size=37)
print(f"\n(Same function on 1000 values in ragged blocks of 37: m={m_chk:.4f}, l={l_chk:.4f})")


# =====================================================================
# PART 4 -- Merging partial attention results
# =====================================================================
# Now apply it to attention. Compute attention against a CHUNK of keys
# and values, return the result UNNORMALIZED plus its statistics. Two
# such partial results can then be merged into one.

def attention_chunk(q, K_chunk, V_chunk):
    """Attention of one query vector against a chunk of K/V. Unnormalized."""
    d = q.shape[-1]
    s = q @ K_chunk.T / np.sqrt(d)      # scores against this chunk only
    m = s.max()                         # local max
    p = np.exp(s - m)                   # local unnormalized weights
    l = p.sum()                         # local denominator
    o = p @ V_chunk                     # local unnormalized output
    return o, m, l


def merge(o1, m1, l1, o2, m2, l2):
    """Combine two partial attention results into one. Order doesn't matter."""
    m = max(m1, m2)
    a1, a2 = np.exp(m1 - m), np.exp(m2 - m)   # rescale each to the shared max
    return a1 * o1 + a2 * o2, m, a1 * l1 + a2 * l2


banner("PART 4 -- Merging partial attention results")

q = Q[0]                                # one query row
half = N // 2

o1, m1, l1 = attention_chunk(q, K[:half], V[:half])     # first half of the sequence
o2, m2, l2 = attention_chunk(q, K[half:], V[half:])     # second half
o, m, l = merge(o1, m1, l1, o2, m2, l2)

print(f"chunk 1 stats : m = {m1:.4f},  l = {l1:.4f}")
print(f"chunk 2 stats : m = {m2:.4f},  l = {l2:.4f}")
print(f"merged stats  : m = {m:.4f},  l = {l:.4f}")
print(f"\nmerged / l : {o / l}")
print(f"naive      : {O_naive[0]}")
print(f"max diff   : {np.abs(o / l - O_naive[0]).max():.2e}   <-- float noise only")

print("\nThe two chunks never saw each other. This is the load-bearing fact:")
print("partial attention results are MERGEABLE. Split across tiles in SRAM and")
print("you get FlashAttention; split across GPUs and you get Ring Attention;")
print("split across the KV cache during decode and you get Flash-Decoding.")

# order-independence, since people doubt it
o_rev, _, l_rev = merge(o2, m2, l2, o1, m1, l1)
print(f"\nmerged in the other order, max diff: "
      f"{np.abs(o_rev / l_rev - O_naive[0]).max():.2e}")


# =====================================================================
# PART 5 -- Tiled flash attention, the whole thing
# =====================================================================
# Same idea, now over blocks of queries as well as blocks of keys, with
# the merge fused into the inner loop. S and P never exist in full --
# only one (block_q x block_kv) tile at a time.

def flash_attention(Q, K, V, block_q=4, block_kv=4, causal=False):
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    O = np.zeros((N, d))
    L = np.zeros(N)                      # logsumexp per row; saved for the backward pass

    for i in range(0, N, block_q):                       # outer loop: query blocks
        Qi = Q[i:i + block_q]
        bq = Qi.shape[0]

        Oi = np.zeros((bq, d))           # running unnormalized output
        mi = np.full(bq, -np.inf)        # running row max
        li = np.zeros(bq)                # running row sum

        for j in range(0, N, block_kv):                  # inner loop: key/value blocks
            if causal and j > i + bq - 1:
                continue                 # whole tile is masked out -- skip it entirely

            Kj, Vj = K[j:j + block_kv], V[j:j + block_kv]

            Sij = (Qi @ Kj.T) * scale                    # (bq, bkv) -- the ONLY tile in memory
            if causal:
                rows = np.arange(i, i + bq)[:, None]
                cols = np.arange(j, j + Kj.shape[0])[None, :]
                Sij = np.where(cols <= rows, Sij, -np.inf)

            m_new = np.maximum(mi, Sij.max(axis=1))      # (bq,)
            alpha = np.exp(mi - m_new)                   # correction factor, per row
            Pij = np.exp(Sij - m_new[:, None])

            li = alpha * li + Pij.sum(axis=1)            # rescale + accumulate denominator
            Oi = alpha[:, None] * Oi + Pij @ Vj          # rescale + accumulate output
            mi = m_new

        O[i:i + bq] = Oi / li[:, None]                   # normalize ONCE, at the end
        L[i:i + bq] = mi + np.log(li)                    # logsumexp: m + log(l)

    return O, L


banner("PART 5 -- Tiled flash attention")

O_flash, L_flash = flash_attention(Q, K, V, block_q=4, block_kv=4)
print(f"max |flash - naive| : {np.abs(O_flash - O_naive).max():.2e}")

print("\nBlock size changes nothing about the result -- only what fits in SRAM:")
for bq in (1, 2, 3, 8):
    for bkv in (1, 3, 8):
        O_b, _ = flash_attention(Q, K, V, block_q=bq, block_kv=bkv)
        assert np.allclose(O_b, O_naive, atol=1e-10)
print("  block_q in {1,2,3,8} x block_kv in {1,3,8} -- all match to 1e-10")
print("\nThis is the headline claim, demonstrated: FlashAttention is EXACT.")
print("It is a different schedule for the same arithmetic, not an approximation.")

# causal, checked against a naive masked reference
S_c = Q @ K.T / np.sqrt(d)
S_c = np.where(np.arange(N)[None, :] <= np.arange(N)[:, None], S_c, -np.inf)
P_c = np.exp(S_c - S_c.max(axis=-1, keepdims=True))
O_causal_naive = (P_c / P_c.sum(axis=-1, keepdims=True)) @ V
O_causal, _ = flash_attention(Q, K, V, block_q=2, block_kv=2, causal=True)
print(f"\ncausal masking, max |flash - naive| : "
      f"{np.abs(O_causal - O_causal_naive).max():.2e}")


# =====================================================================
# PART 6 -- What it bought us
# =====================================================================
# The saving isn't FLOPs. It's the size of the largest thing you have to
# hold, and the number of round trips to slow memory.

banner("PART 6 -- Memory accounting")

print(f"{'N':>8} {'naive peak':>14} {'flash peak':>14} {'ratio':>10}")
print("-" * 50)
for N_big in (1024, 4096, 16384, 65536):
    d_big, bq, bkv = 64, 128, 128
    naive_bytes = N_big * N_big * 2                          # the S matrix, fp16
    flash_bytes = bq * bkv * 2 + N_big * d_big * 2 + N_big * 4  # tile + output + logsumexp
    print(f"{N_big:>8} {naive_bytes / 1e6:>11.1f} MB {flash_bytes / 1e6:>11.1f} MB "
          f"{naive_bytes / flash_bytes:>9.0f}x")

print("\nNaive grows with N^2. Flash grows with N. That gap is the entire")
print("reason long-context models are trainable.")

print("\nNote what did NOT change: both do ~4*N^2*d FLOPs. FlashAttention does")
print("not make attention cheaper in arithmetic -- it makes it cheaper in data")
print("movement, which is what the hardware was actually stalling on.")


banner("Summary")
print("""
  1. Attention's cost is the N x N score matrix, not the matmuls.
  2. Softmax seems to block tiling because it normalizes a whole row.
  3. Online softmax removes that block: keep a running (max, sum) and
     rescale earlier work by exp(m_old - m_new) when the max moves.
  4. Partial attention results are therefore mergeable, in any order.
  5. So you can tile: compute one small S tile at a time, in fast memory,
     and accumulate. Normalize once at the end.
  6. Result is bit-comparable to naive, memory is O(N) not O(N^2).

  A real kernel adds: SRAM residency, warp scheduling, async copies,
  fp16/fp8, and a recomputed backward pass. None of that changes the
  five lines in the inner loop above.
""")
