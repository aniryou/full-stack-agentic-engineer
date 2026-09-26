# %% [markdown]
# # 05 · Speculative decoding
#
# **Tier:** T0 — CPU only, no network, about twenty seconds. Speed-ups on real hardware are **SIMULATED** here
# (roofline model); `vllm-serving-lab` notebook `05_speculation_and_quantization_in_vllm` measures vLLM's
# n-gram and EAGLE speculators on a GPU (T1).
#
# ## The one-minute version
# Decode is memory-bound: a forward pass over `k + 1` positions costs about the same as over one, because the
# weight read dominates. So let a cheap **proposer** guess `k` tokens, and have the target model score all of them
# in **one** pass. Keep draft token `x ~ q` with probability `min(1, p(x)/q(x))`; at the first rejection, sample a
# replacement from the residual `norm(max(0, p − q))` and stop; if all `k` survive, take a bonus token from the
# target. This rule makes the output distribution **exactly** the target's — speculation changes speed, never
# quality. With acceptance rate `α = Σ min(p, q)` one pass yields `(1 − α^(k+1)) / (1 − α)` tokens on average.
# Whether that is a speed-up depends on what the draft costs and on whether the verify pass is still cheap —
# which stops being true once the batch is compute-bound.
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
# The draft proposes token 3 far too often (0.4 vs 0.05); verification rejects most of those, and the residual
# re-assigns the mass to tokens 0 and 1, which the draft under-proposed. The emitted distribution is `p`.
#
# ## Worked example 2 — a real draft/target pair
# The target is `TinyLM()`; the draft is a model ten times smaller, with the same tokenizer (a requirement) and
# similar "training data".

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
# Two things to notice. The measured tokens per pass match the formula at k = 1 and 2, within the noise of ~50
# passes, but fall short of it at k = 4. That is not only noise: `(1 − α^(k+1)) / (1 − α)` assumes every position
# is accepted independently with the same α, while real acceptance varies by position (p10 to p90 above) and is
# correlated — a stretch the draft finds hard rejects early and often — so the formula over-predicts deep
# speculation. Measure acceptance per position (vLLM
# exports `vllm:spec_decode_num_accepted_tokens_per_pos`) before choosing k. And greedy speculation is not "close
# to" greedy decoding — it is identical, token for token, because with temperature 0 both distributions are
# one-hot and the rule reduces to "accept iff the draft's argmax equals the target's".
#
# ## Worked example 3 — prompt lookup: free drafts when the output copies the context
# N-gram (prompt-lookup) drafting proposes the tokens that followed the last occurrence of the current n-gram in
# the context. No draft model at all. It shines when the output repeats the input — quoting a tool result, editing
# code — and does nothing otherwise. `copy_target` is a toy target that copies (an "induction head").

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
# Llama-3.1-8B target, Llama-3.2-1B draft (same tokenizer) on an H100, `α = 0.7`, `k = 4`. Plain decoding costs
# one engine step per token. One speculative round is **one** engine step, so it pays the step's fixed overhead
# (2 ms here) once, plus `k` draft forwards — each a CUDA-graph replay and a draft sample, assumed to cost 0.5 ms
# of overhead on top of its roofline time — plus one target verify pass of `k + 1` tokens per request with logits
# at all `k + 1` positions (`perf.spec_speedup`). Batches whose KV cache would not fit are marked `--`.

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
# At batch 1 the gain is ~1.6× with these assumptions, and the draft's cost is why it is not more: four forwards of
# a 1B model, each about 19% of an 8B step (`c ≈ 0.19`: its weight read is ~17% of the target's, plus the assumed
# 0.5 ms). The sensitivity rows show how much that rests on the overhead assumption: 1.9× if a draft forward cost
# only its roofline time, 1.1× if each paid a full engine step's 2 ms — measure it before quoting a number. Draft
# heads that ride on the target's own hidden states (EAGLE, MTP) cost a few percent of a target step: with
# `c = 0.05` the same `α` and `k` give the 2.31× printed above — which is why they, not separate draft models,
# dominate in practice.
#
# The robust conclusion is the shape. At short contexts the verify pass (`B × 5` tokens) crosses the compute knee
# as the batch grows, and speculation becomes a **slow-down** past ~100–250 requests (depending on the overhead):
# the "free" positions are no longer free. At long contexts decode stays memory-bound on the KV read, so
# speculation keeps paying — and memory caps the batch before compute does. In production that is why
# speculation is usually tuned per workload (or switched off above a load threshold).
#
# ## Exercise 5.1 — one position of verification
# Implement the rule for a single draft token `x` drawn from `q`: return `(True, x)` if accepted, else
# `(False, y)` with `y` drawn from the normalised residual.

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
# Each draft token is accepted with probability `α` independently, and the first rejection ends the round, which
# still emits one token (recovered or bonus). Write `expected_tokens(alpha, k)` (handle `alpha == 1`).

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
# If one draft step costs `c` target steps, a round costs `k c + 1` and emits `expected_tokens(α, k)`. Write
# `best_k(alpha, c)` (search k = 1..16), then set `k_answer` for `α = 0.6, c = 0.05`.

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
# Using `spec_speedup(B, ctx)` from worked example 4 (short 200-token contexts, α = 0.7, k = 4), find the
# smallest batch in `[1, 2, 4, ..., 512]` where speculation makes decoding **slower**.

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
# Write `lookup(tokens, k, n)`: find the most recent **earlier** occurrence of the last `n` tokens and return the
# (up to) `k` tokens that followed it; `[]` if there is none.

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
# **The two-minute version.** "Decode is memory-bound, so the target can check several guessed tokens in the time
# of one. A proposer — a small same-tokenizer model, an EAGLE head on the target's hidden states, the model's own
# multi-token-prediction heads, or plain prompt lookup — drafts k tokens; the target scores k+1 positions in one
# pass; rejection sampling keeps exactly the target distribution, so quality is unchanged by construction. Gains
# are 1 + α + … + α^k tokens per pass minus the draft's cost; with α around 0.7 that is 2–3x fewer target passes.
# It helps most at low batch and long context, and it can hurt at high batch with short contexts, where the
# verify tokens are no longer free. For our agents, prompt lookup is attractive because they quote tool output a
# lot; we would enable it per deployment and watch the acceptance rate and ITL, not assume them."
#
# **Drill questions**
# 1. *Does speculative decoding change the output distribution?* — No: accept with min(1, p/q), resample rejections
#    from norm(max(0, p − q)); the emitted token is distributed exactly as p.
# 2. *α = 0.8, k = 4: tokens per target pass?* — (1 − 0.8⁵)/(1 − 0.8) = 3.36.
# 3. *Why can speculation make a busy server slower?* — At large batch the verify pass is compute-bound, so the k
#    extra positions per request cost real FLOPs, and rejected ones are wasted.
