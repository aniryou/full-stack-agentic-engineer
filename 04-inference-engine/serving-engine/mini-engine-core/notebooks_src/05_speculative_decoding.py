# %% [markdown]
# # 05 · Speculative decoding
#
# **Tier:** T0. It needs a CPU only and no network, and it runs in about twenty seconds. The speed-ups on real
# hardware are **SIMULATED** here (roofline model). `vllm-serving-lab` notebook `05_speculation_and_quantization_in_vllm`
# measures the n-gram and EAGLE speculators of vLLM on a GPU (T1).
#
# ## The one-minute version
# Decode is memory-bound. A forward pass over $k + 1$ positions costs approximately the same as a pass over one
# position, because the weight read dominates. Thus let a low-cost **proposer** guess $k$ tokens. Then let the target
# model score all of them in **one** pass. For each draft token $x \sim q$, the rule is:
#
# * Keep the token with probability $\min(1, p(x)/q(x))$.
# * At the first rejection, sample a replacement from the residual $\operatorname{norm}(\max(0, p - q))$, and stop.
# * If all $k$ tokens survive, take a bonus token from the target.
#
# This rule makes the output distribution **exactly** the distribution of the target. Speculation changes the speed,
# never the quality. With the acceptance rate $\alpha = \sum \min(p, q)$, one pass gives
# $(1 - \alpha^{k+1})/(1 - \alpha)$ tokens on average.
#
# If that is a speed-up or not depends on two things: the cost of the draft, and if the verify pass is still
# low-cost. The verify pass is no longer low-cost when the batch is compute-bound.
#
# Primer: §7 *Speculative decoding* (`../../PRIMER.md`).

# %%
import numpy as np

from minengine import EOS, SMALL, VOCAB, TinyLM, decode, encode, perf, spec
from minengine.model import CORPUS, softmax

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — the rule on a four-token vocabulary

# %%
p = np.array([0.5, 0.3, 0.15, 0.05])            # the target's next-token distribution
q = np.array([0.2, 0.2, 0.2, 0.4])              # the draft's
print(f"alpha = sum(min(p, q)) = {spec.acceptance_rate(p, q):.2f} = 1 - TV(p, q) = {1 - 0.5 * np.abs(p - q).sum():.2f}")
print("residual norm(max(0, p - q)) =", np.maximum(p - q, 0) / np.maximum(p - q, 0).sum())
n, emitted, accepted = 20000, np.zeros(4), 0
for _ in range(n):
    x = rng.choice(4, p=q)
    out = spec.verify(np.stack([p, p]), q[None], [x], rng)      # one draft token; keep only the first emitted
    emitted[out[0]] += 1
    accepted += len(out) == 2
print("emitted frequencies:", (emitted / n).round(3), "vs p", p, "| accepted", accepted / n)

# %% [markdown]
# The draft proposes token 3 much too often (0.4 against 0.05). The verification rejects most of those proposals.
# The residual then moves the mass to tokens 0 and 1, which the draft proposed less often than the target. The
# emitted distribution is $p$.
#
# ## Worked example 2 — a real draft/target pair
# The target is `TinyLM()`. The draft is a model that is ten times smaller. It has the same tokenizer (a requirement)
# and similar "training data".

# %%
target, draft = TinyLM(), TinyLM(SMALL, seed=1)
print("params: target", target.num_params(), "draft", draft.num_params())
ids = encode(CORPUS[:400])
alpha = np.minimum(softmax(target.forward_dense(ids)), softmax(draft.forward_dense(ids))).sum(1)
print(f"acceptance rate per position: mean {alpha.mean():.2f}, p10 {np.percentile(alpha, 10):.2f}, "
      f"p90 {np.percentile(alpha, 90):.2f}")
prompt = encode("The engine runs ")
for k in [1, 2, 4]:
    out, st = spec.speculative_generate(spec.lm_probs(target), prompt, 120, k=k,
                                        draft_probs=spec.lm_probs(draft), seed=0)
    print(f"k={k}: {st.passes:3d} target passes for 120 tokens, {st.tokens_per_pass:.2f} tokens/pass "
          f"(formula with alpha={alpha.mean():.2f}: {spec.expected_tokens(alpha.mean(), k):.2f})")
out, st = spec.speculative_generate(spec.lm_probs(target, 0), prompt, 40, k=4, draft_probs=spec.lm_probs(draft, 0))
print("greedy speculation == greedy decoding:", out == target.generate_dense(prompt, 40), f"({st.passes} passes)")

# %% [markdown]
# Look at two things.
#
# First, the measured tokens per pass agree with the formula at $k = 1$ and 2, within the noise of ~50 passes. But at
# $k = 4$, they are less than the formula. This is not only noise. The formula $(1 - \alpha^{k+1})/(1 - \alpha)$
# assumes that the target accepts each position independently, with the same $\alpha$. But the real acceptance
# changes with the position (p10 to p90 in the printed output), and it is correlated. A part of the text that is hard
# for the draft gets early and frequent rejections, thus the formula predicts too much for deep speculation.
#
# Measure the acceptance per position (vLLM exports `vllm:spec_decode_num_accepted_tokens_per_pos`) before you select
# $k$.
#
# Second, greedy speculation is not "close to" greedy decoding. It is identical, token for token. With temperature 0,
# both distributions are one-hot. Thus the rule becomes "accept iff the draft's argmax equals the target's".
#
# ## Worked example 3 — prompt lookup: free drafts when the output copies the context
# N-gram (prompt-lookup) drafts are the tokens that came after the last occurrence of the current n-gram in the
# context. This method uses no draft model at all. It gives the best results when the output repeats the input, for
# example when the output quotes a tool result or edits code. In other cases, it does nothing. `copy_target` is a toy target
# that copies (an "induction head").

# %%
def copy_target(tokens, n=4):
    """One-hot: after the last n tokens, what followed their latest earlier occurrence (EOS if none)."""
    rows = np.zeros((len(tokens), VOCAB))
    for t in range(len(tokens)):
        key, nxt = tokens[max(0, t - n + 1):t + 1], EOS
        for s in range(t - n, -1, -1):
            if tokens[s:s + n] == key:
                nxt = tokens[s + n]
                break
        rows[t, nxt] = 1.0
    return rows


agent = encode("Tool result: order 4411 shipped 2026-09-24 via DHL, tracking DH123456789.\n"
               "Agent: Your order 4411 shipped 2026-09-24 via DHL, tracking ")
out, st = spec.speculative_generate(copy_target, agent, 24, k=6, ngram=3)
print(f"copying target : {decode(out)!r:28} acceptance {st.acceptance:.2f}, {st.tokens_per_pass:.1f} tokens/pass")
out, st = spec.speculative_generate(spec.lm_probs(target), encode("The engine runs. The engine runs"), 24, k=6, ngram=3)
print(f"babbling target: {decode(out)!r:28} acceptance {st.acceptance:.2f}, {st.tokens_per_pass:.1f} tokens/pass")

# %% [markdown]
# ## Worked example 4 — when does it pay? (SIMULATED)
# The setup is a Llama-3.1-8B target and a Llama-3.2-1B draft (same tokenizer) on an H100, with $\alpha = 0.7$ and
# $k = 4$. Plain decode costs one engine step per token. One speculative round is **one** engine step. Thus the round
# pays these costs (`perf.spec_speedup`):
#
# * The constant overhead of the step (2 ms here), one time.
# * $k$ draft forwards. Each forward is a CUDA-graph replay and a draft sample. The model assumes that each forward
#   costs its roofline time plus 0.5 ms of overhead.
# * One target verify pass of $k + 1$ tokens per request, with logits at all $k + 1$ positions.
#
# The output marks a batch with `--` when its KV cache does not fit.

# %%
G, T, D = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"], perf.LLMS["llama-3.2-1b"]
kv_tokens = perf.kv_cache_blocks(G, T) * 16


def spec_speedup(B, ctx, alpha=0.7, k=4, draft_overhead_s=0.0005):
    base = perf.step_time(G, T, [(ctx, 1)] * B)                                  # one token per request
    drafts = sum(perf.step_time(G, D, [(ctx + j, 1)] * B, overhead_s=draft_overhead_s) for j in range(k))
    verify = perf.step_time(G, T, [(ctx, k + 1, k + 1)] * B)                     # (start, tokens, logit rows)
    return spec.expected_tokens(alpha, k) * base / (drafts + verify)


assert abs(spec_speedup(16, 200) - perf.spec_speedup(G, T, D, 16, 200, 0.7, 4)) < 1e-12   # same model as the library
for ctx in [200, 2000, 8000]:
    cells = [f"B={B}: " + (f"{spec_speedup(B, ctx):.2f}x" if B * (ctx + 5) <= kv_tokens else "--")
             for B in [1, 16, 64, 128, 256, 512]]
    print(f"context {ctx:5d}  " + "  ".join(cells), " (SIMULATED)")
for ov in [0.0, 0.0005, 0.002]:
    c = perf.step_time(G, D, [(1000, 1)], overhead_s=ov) / perf.step_time(G, T, [(1000, 1)])
    print(f"per-draft-forward overhead {1e3 * ov:3.1f} ms: draft cost c = {c:.2f} of a target step, "
          f"batch 1 at context 200 -> {spec_speedup(1, 200, draft_overhead_s=ov):.2f}x")
print(f"an EAGLE-like head with c = 0.05 would give {spec.speedup(0.7, 4, 0.05):.2f}x at batch 1")

# %% [markdown]
# At batch 1, the gain is ~1.6× with these assumptions. The cost of the draft is the reason that the gain is not
# more. The draft does four forwards of a 1B model. Each forward costs about 19% of an 8B step
# ($c \approx 0.19$: its weight read is ~17% of the target's, plus the assumed 0.5 ms).
#
# The sensitivity rows show how much that result depends on the overhead assumption. If a draft forward costs only its
# roofline time, the gain is 1.9×. If each forward pays the full 2 ms of an engine step, the gain is 1.1×. Measure the
# overhead before you quote a number.
#
# Draft heads that use the hidden states of the target itself (EAGLE, MTP) cost a few percent of a target step. With
# $c = 0.05$, the same $\alpha$ and $k$ give the 2.31× that the cell printed. This is why these heads, not separate
# draft models, dominate in practice.
#
# The conclusion that holds under all of these assumptions is the shape. At short contexts, the verify pass
# ($B \times 5$ tokens) crosses the compute knee when the batch increases. Past ~100–250 requests (the point depends
# on the overhead), speculation becomes a **slow-down**: the "free" positions are no longer free. At long contexts,
# decode stays memory-bound on the KV read. Thus speculation continues to give a gain, and memory caps the batch before
# compute does. That is why, in production, the speculation settings are usually specific to each workload (or
# speculation is off above a load threshold).
#
# ## Exercise 5.1 — one position of verification
# Write the rule for one draft token $x$ sampled from $q$. If the rule accepts $x$, return `(True, x)`. If not, return
# `(False, y)`, with $y$ sampled from the normalised residual.

# %% exercise
def accept_or_recover(p, q, x, rng):
    ### BEGIN SOLUTION
    if rng.random() < min(1.0, p[x] / q[x]):
        return True, int(x)
    residual = np.maximum(p - q, 0.0)
    return False, int(rng.choice(len(p), p=residual / residual.sum()))
    ### END SOLUTION

# %% check
r, counts, acc = np.random.default_rng(1), np.zeros(4), 0
for _ in range(20000):
    ok, y = accept_or_recover(p, q, r.choice(4, p=q), r)
    counts[y] += 1
    acc += ok
chi2 = ((counts - 20000 * p) ** 2 / (20000 * p)).sum()
assert chi2 < 16.3, chi2                                   # chi-square, 3 dof, p = 0.001
assert abs(acc / 20000 - spec.acceptance_rate(p, q)) < 0.02
print(f"✅ emitted tokens follow p (chi2 = {chi2:.1f}); acceptance {acc / 20000:.3f} ~ alpha {spec.acceptance_rate(p, q):.2f}")

# %% [markdown]
# ## Exercise 5.2 — expected tokens per target pass
# The target accepts each draft token independently, with probability $\alpha$. The first rejection ends the round.
# The round still emits one token (recovered or bonus). Write `expected_tokens(alpha, k)`. Make sure that it also
# handles `alpha == 1`.

# %% exercise
def expected_tokens(alpha, k):
    ### BEGIN SOLUTION
    return k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)
    ### END SOLUTION

# %% check
assert expected_tokens(1.0, 4) == 5 and expected_tokens(0.0, 4) == 1
assert abs(expected_tokens(0.8, 4) - 3.3616) < 1e-9
r = np.random.default_rng(2)
sim = np.mean([1 + np.argmin(np.append(r.random(4) < 0.8, False)) for _ in range(20000)])
assert abs(sim - expected_tokens(0.8, 4)) < 0.05
print(f"✅ alpha 0.8, k 4: {expected_tokens(0.8, 4):.4f} tokens per pass (simulated {sim:.3f})")

# %% [markdown]
# ## Exercise 5.3 — choose k
# If one draft step costs $c$ target steps, a round costs $k\,c + 1$ and emits `expected_tokens(α, k)`. Write
# `best_k(alpha, c)` (search $k = 1, \ldots, 16$). Then set `k_answer` for $\alpha = 0.6$, $c = 0.05$.

# %% exercise
def best_k(alpha, c, k_max=16):
    ### BEGIN SOLUTION
    return max(range(1, k_max + 1), key=lambda k: expected_tokens(alpha, k) / (k * c + 1))
    ### END SOLUTION


### BEGIN SOLUTION
k_answer = best_k(0.6, 0.05)
### END SOLUTION

# %% check
for a in [0.5, 0.7, 0.9]:
    for c in [0.02, 0.1, 0.3]:
        assert best_k(a, c) == spec.best_k(a, c)
assert k_answer == 4
print(f"✅ alpha 0.6, c 0.05 -> k = {k_answer} ({spec.speedup(0.6, 4, 0.05):.2f}x); "
      f"alpha 0.9, c 0.05 -> k = {best_k(0.9, 0.05)}: better drafts earn deeper speculation")

# %% [markdown]
# ## Exercise 5.4 — at what batch size does it stop paying?
# Use `spec_speedup(B, ctx)` from worked example 4 (short 200-token contexts, $\alpha = 0.7$, $k = 4$). Find the
# smallest batch in `[1, 2, 4, ..., 512]` at which speculation makes decode **slower**.

# %% exercise
batches = [2 ** i for i in range(10)]
### BEGIN SOLUTION
b_star = next(B for B in batches if spec_speedup(B, 200) < 1)
### END SOLUTION

# %% check
assert b_star == 128 and spec_speedup(64, 200) > 1
print(f"✅ at context 200 speculation turns into a slow-down at batch {b_star} ({spec_speedup(b_star, 200):.2f}x) - SIMULATED")

# %% [markdown]
# ## Exercise 5.5 — prompt lookup
# Write `lookup(tokens, k, n)`. It finds the most recent **earlier** occurrence of the last $n$ tokens. It returns the
# (up to) $k$ tokens that came after that occurrence. If there is no occurrence, it returns `[]`.

# %% exercise
def lookup(tokens, k, n=3):
    ### BEGIN SOLUTION
    if len(tokens) <= n:
        return []
    tail = list(tokens[-n:])
    for s in range(len(tokens) - n - 1, -1, -1):
        if list(tokens[s:s + n]) == tail:
            return list(tokens[s + n:s + n + k])
    return []
    ### END SOLUTION

# %% check
for text in ["the cat sat. the cat", "abcabcab", "no repeats here", "aaaa"]:
    for k in [1, 3, 5]:
        assert lookup(encode(text), k) == spec.ngram_propose(encode(text), k, 3), (text, k)
print("✅ 'the cat sat. the cat' ->", repr(decode(lookup(encode("the cat sat. the cat"), 4))))

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Decode is memory-bound. Thus the target can examine several guessed tokens in the time
# of one token. A proposer drafts $k$ tokens. It can be a small model with the same tokenizer, or an EAGLE head on the
# hidden states of the target. It can also be the multi-token-prediction heads of the model itself, or plain prompt
# lookup.
#
# "The target scores $k+1$ positions in one pass. Rejection sampling keeps exactly the target distribution. Thus, by
# construction, the quality does not change.
#
# "The gain is $1 + \alpha + \dots + \alpha^k$ tokens per pass, minus the cost of the draft. With $\alpha$ around 0.7,
# that is 2–3x fewer target passes. It helps most at low batch and long context. It can hurt at high batch with short
# contexts, where the verify tokens are no longer free.
#
# "For our agents, prompt lookup is a good option, because they quote tool output frequently. Our plan is to turn it
# on per deployment, and to monitor the acceptance rate and ITL, not to assume them."
#
# **Drill questions**
# 1. *Does speculative decoding change the output distribution?* No. Accept with $\min(1, p/q)$, and resample
#    rejections from $\operatorname{norm}(\max(0, p - q))$. The emitted token has exactly the distribution $p$.
# 2. *$\alpha = 0.8$, $k = 4$: tokens per target pass?* (1 − 0.8⁵)/(1 − 0.8) = 3.36.
# 3. *Why can speculation make a busy server slower?* At large batch, the verify pass is compute-bound. Thus the $k$
#    extra positions per request cost real FLOPs, and the rejected positions are a waste.
