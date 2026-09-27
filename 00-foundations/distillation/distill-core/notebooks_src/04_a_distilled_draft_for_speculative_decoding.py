# %% [markdown]
# # 04 · A distilled draft for speculative decoding
#
# **Tier:** T0 — CPU only, numpy, no network, under a minute. The target and drafts are the tiny next-token
# models of `distillcore.tinylm`; acceptance is computed exactly from their distributions on the target's own
# text. Measuring a distilled draft under vLLM's `--speculative-config` is `distill-lab` notebook
# `04_a_distilled_draft_in_vllm` (T1).
#
# ## The one-minute version
# Speculative decoding (serving-engine primer §7) accepts a drafted token with probability min(1, p/q), which
# makes the per-token acceptance α = Σ_v min(p(v), q(v)) = 1 − TV(p, q), and one target pass yields
# (1 − α^(k+1)) / (1 − α) tokens. So a draft model is a **student whose only metric is acceptance** — measured
# where it matters, on the target's own text. Training it on the target's outputs or distributions
# (distillation from the target) beats an off-the-shelf small model of the same family, whose distribution
# differs wherever the target was fine-tuned. Bigger drafts accept more and cost more per step; the speedup
# E / (k·c + 1) picks the size. And vLLM drafts greedily by default, which caps acceptance at the target's
# top-token probability when the target samples. Primer: `../../PRIMER.md` §7.

# %%
import numpy as np

from distillcore import ModLang, TinyLM, cost as K, draft as Dr, losses as L, seqkd, train
from distillcore.tinylm import fit_language

P = np.array([0.5, 0.3, 0.15, 0.05])     # serving-engine primer §7's target and draft distributions
Q = np.array([0.2, 0.2, 0.2, 0.4])

# %% [markdown]
# ## Worked example 1 — the three formulas, on the serving primer's own numbers

# %%
a = Dr.acceptance_rate(P, Q)
print(f"α = Σ min(p, q) = {a:.2f} = 1 − TV = {1 - 0.5 * np.abs(P - Q).sum():.2f}")
for k in (1, 3, 4, 5):
    v = Dr.vllm_view(a, k)
    print(f"k = {k}: {v['mean_acceptance_length']:.4f} tokens per pass (vLLM's 'mean acceptance length'); "
          f"per-position {np.round(v['per_position'], 4).tolist()}; 'draft acceptance rate' {v['draft_acceptance_rate']:.4f}; "
          f"speedup at c = 0.1: {Dr.speedup(a, k, 0.1):.4f}")
print(f"best k at c = 0.1: {Dr.best_k(a, 0.1)}; greedy drafting accepts p(argmax q) = {Dr.greedy_acceptance(P, Q):.2f}")

# %% [markdown]
# Two reading traps. vLLM's "draft acceptance rate" is accepted ÷ drafted = (E − 1)/k, which falls as k grows
# even at constant α; α itself is the position-0 rate. And vLLM's draft models propose their argmax by default
# (`draft_sample_method="greedy"`, verify), so the acceptance is p(argmax q) — 0.05 here, not 0.6.
#
# ## Worked example 2 — a fine-tuned target and three drafts
# The target was fine-tuned into a dialect of `ModLang`: its noise all goes to the +1 neighbour. Three drafts of
# the same width (16 hidden units): an **off-the-shelf** small model trained on the base language (the same
# family, not the fine-tune), a draft distilled on the **target's distributions** (logit KD), and one trained
# on **24,000 tokens of the target's samples** (SeqKD from the target).

# %%
base, dialect = ModLang(11, 0.2), ModLang(11, 0.2, skew=1.0)
C = base.contexts()
target = fit_language(dialect, 64)
text = target.sample(C[np.random.default_rng(1).integers(0, 121, 500)], 12, np.random.default_rng(2))       # scoring text
corpus = target.sample(C[np.random.default_rng(3).integers(0, 121, 2000)], 12, np.random.default_rng(4))    # draft training data
PT = target.probs(C)


def kd_draft(H):
    d = TinyLM(11, H, 8, seed=3)
    train(d, C, lambda z, i: L.soft_ce(z, PT[i]), 1500)
    return d


drafts = {"off-the-shelf": fit_language(base, 16, seed=3), "KD from target": kd_draft(16)}
sk = TinyLM(11, 16, 8, seed=3)
seqkd.sft(sk, corpus, 1500, batch=512)
drafts["SeqKD from target"] = sk
c16 = Dr.draft_cost(drafts["KD from target"].n_params, target.n_params)
for name, d in drafts.items():
    r = Dr.acceptance_on_text(target, d, text)
    print(f"{name:18s} α {r['alpha']:.3f}  greedy {r['greedy']:.3f}  KL(p‖q) {r['kl']:.3f}  TV {r['tv']:.3f}  "
          f"speedup at k = 4, c = {c16:.3f}: {Dr.speedup(r['alpha'], 4, c16):.2f}×")

# %% [markdown]
# The off-the-shelf draft knows the language but not the dialect: α = 0.890. Distilled on the target's
# distributions it reaches 0.988; from samples alone, 0.933 — samples estimate the target's distribution, soft
# targets hand it over. At k = 4 that is 1.86× against 2.26×. Note the greedy column: when the target samples
# at T = 1, a greedy draft is accepted with probability p(argmax q) ≤ max p = 0.8, however good the draft is.
#
# ## Worked example 3 — acceptance against draft size

# %%
sizes = {}
print(f"{'hidden units':>12} {'c':>6} {'α (KD)':>7} {'speedup k=4':>12} {'best k':>7} {'at best k':>10} {'α off-the-shelf':>16}")
for H in (4, 8, 16, 32):
    d = drafts["KD from target"] if H == 16 else kd_draft(H)
    c = Dr.draft_cost(d.n_params, target.n_params)
    al = Dr.acceptance_on_text(target, d, text)["alpha"]
    off = Dr.acceptance_on_text(target, fit_language(base, H, seed=3), text)["alpha"]
    kb = Dr.best_k(al, c)
    sizes[H] = (c, al)
    print(f"{H:12d} {c:6.3f} {al:7.3f} {Dr.speedup(al, 4, c):12.2f} {kb:7d} {Dr.speedup(al, kb, c):10.2f} {off:16.3f}")

# %% [markdown]
# Acceptance rises with size and saturates once the draft can hold the target (16 units here); cost keeps
# rising, so the speedup turns over (the best k is capped at 16 here). Below that size the capacity gap, not the
# training data, sets α: at 8 units the off-the-shelf draft even edges out the distilled one — which errors a
# too-small draft makes depends on its training run. c here is the ratio of parameters, the
# memory-bound view of a decode step; a real engine adds per-step overheads (serving-engine primer §7).
#
# ## Worked example 4 — the same arithmetic for a real pair
# Qwen3-0.6B drafting for Qwen3-4B (same vocabulary, 151,936 — vLLM's `draft_model` requires equal vocab sizes).
# At batch 1, decode streams the weights, so c ≈ the ratio of weight bytes. A model, not a measurement.

# %%
s06, s4 = K.SHAPES["qwen3-0.6b"], K.SHAPES["qwen3-4b"]
c_real = s06.params() / s4.params()
print(f"c ≈ {s06.params():,} / {s4.params():,} = {c_real:.3f}")
for al in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"α = {al}: speedup at k = 4 {Dr.speedup(al, 4, c_real):.2f}×; best k = {Dr.best_k(al, c_real)} "
          f"→ {Dr.speedup(al, Dr.best_k(al, c_real), c_real):.2f}×")

# %% [markdown]
# ## Exercise 4.1 — score a draft where it will be used
# Return the mean per-position acceptance Σ_v min(p(v), q(v)) of `draft` against `target` over every generated
# position of `seqs` (the target's own samples). Use `target.positions(seqs)` and `.probs(ctx)`.

# %% exercise
def mean_alpha(target, draft, seqs):
    ### BEGIN SOLUTION
    ctx, _ = target.positions(seqs)
    return float(np.minimum(target.probs(ctx), draft.probs(ctx)).sum(1).mean())
    ### END SOLUTION

# %% check
for d in drafts.values():
    assert abs(mean_alpha(target, d, text) - Dr.acceptance_on_text(target, d, text)["alpha"]) < 1e-12
print("✅ α is measured on the target's own text, position by position — not on a benchmark the draft was trained on")

# %% [markdown]
# ## Exercise 4.2 — tokens per pass and the best depth
# Write `tokens_per_pass(alpha, k)` = 1 + α + … + α^k and `best_depth(alpha, c, k_max=16)`, the k that maximises
# tokens_per_pass / (k·c + 1) (the smallest k on ties).

# %% exercise
def tokens_per_pass(alpha, k):
    ### BEGIN SOLUTION
    return float(sum(alpha ** i for i in range(k + 1)))
    ### END SOLUTION


def best_depth(alpha, c, k_max=16):
    ### BEGIN SOLUTION
    return max(range(1, k_max + 1), key=lambda k: tokens_per_pass(alpha, k) / (k * c + 1))
    ### END SOLUTION

# %% check
for al in (0.3, 0.6, 0.8, 0.95):
    for k in (1, 4, 9):
        assert abs(tokens_per_pass(al, k) - Dr.expected_tokens(al, k)) < 1e-12
    for c in (0.05, 0.1, 0.3):
        assert best_depth(al, c) == Dr.best_k(al, c)
assert best_depth(0.8, 0.1) == 6 and abs(tokens_per_pass(0.8, 4) - 3.3616) < 1e-12
print("✅ α = 0.8, k = 4 → 3.36 tokens per pass; at c = 0.1 the best depth is 6 (serving-engine primer §7)")

# %% [markdown]
# ## Exercise 4.3 — greedy against probabilistic drafting
# A *perfect* draft (q = p at every position). The target samples at T = 1. Predict `greedy_alpha`, the mean
# acceptance when the draft proposes its argmax, and `prob_alpha`, when it samples from q. (In this dialect the
# target's top token has probability 0.8 at every context.)

# %% exercise
### BEGIN SOLUTION
greedy_alpha, prob_alpha = 0.8, 1.0
### END SOLUTION

# %% check
perfect = Dr.acceptance_on_text(target, target, text)
assert abs(greedy_alpha - perfect["greedy"]) < 0.005 and abs(prob_alpha - perfect["alpha"]) < 1e-9
print(f"✅ a perfect draft: {perfect['greedy']:.3f} greedy, {perfect['alpha']:.3f} probabilistic — for sampled traffic, "
      "draft probabilistically or the target's own entropy caps the gain")

# %% [markdown]
# ## Exercise 4.4 — read vLLM's counters
# A run reports, over its window: `num_drafts` = 10,000, `num_draft_tokens` = 40,000, `num_accepted_tokens` =
# 17,332 and per-position accepted counts [7,000, 4,900, 3,430, 2,002]. Compute `mean_len` (tokens per pass,
# bonus included), `alpha` (the position-0 rate) and `draft_rate` (accepted ÷ drafted), and say with
# `iid` (True/False) whether the per-position rates are consistent with one α (within 0.01 of α^(i+1)).

# %% exercise
num_drafts, num_draft_tokens, num_accepted = 10_000, 40_000, 17_332
per_pos = np.array([7000, 4900, 3430, 2002])
### BEGIN SOLUTION
mean_len = 1 + num_accepted / num_drafts
alpha = per_pos[0] / num_drafts
draft_rate = num_accepted / num_draft_tokens
iid = bool(np.all(np.abs(per_pos / num_drafts - alpha ** np.arange(1, 5)) < 0.01))
### END SOLUTION

# %% check
v = Dr.vllm_view(0.7, 4)
assert abs(alpha - 0.7) < 1e-12 and abs(mean_len - 2.7332) < 1e-12 and abs(draft_rate - 0.4333) < 1e-12
assert iid is False and abs(v["per_position"][3] - 0.2401) < 1e-12        # position 3 shows 0.2002, not 0.2401
print(f"✅ α = {alpha:.2f}, {mean_len:.4f} tokens per pass, a 'draft acceptance rate' of {draft_rate:.4f} — and the deep "
      "positions fall faster than α^(i+1): acceptance is correlated, so the i.i.d. formula over-predicts deep drafts")

# %% [markdown]
# ## Exercise 4.5 — pick the draft
# From the size sweep (`sizes`: H → (c, α)), choose `best_H`, the width with the highest speedup at k = 4.

# %% exercise
### BEGIN SOLUTION
best_H = max(sizes, key=lambda H: Dr.speedup(sizes[H][1], 4, sizes[H][0]))
### END SOLUTION

# %% check
assert best_H == max(sizes, key=lambda H: Dr.expected_tokens(sizes[H][1], 4) / (4 * sizes[H][0] + 1)) == 16
print("✅ 16 units: the smallest draft that holds the target — more acceptance than a smaller one, less cost than a bigger one")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A draft model is a student whose metric is acceptance, α = Σ min(p, q) = 1 − TV,
# on the target's own traffic. One target pass then yields (1 − α^(k+1))/(1 − α) tokens, and the speedup is that
# over k·c + 1. So we train the draft on the target — on its outputs (SeqKD), or better its distributions —
# rather than take the family's small model off the shelf: a fine-tuned target differs from its base exactly
# where the off-the-shelf draft will be rejected. We size the draft where the speedup turns over, check the
# vocabulary matches (vLLM requires it), draft probabilistically for sampled traffic because greedy drafting is
# capped by the target's top-token probability, and read vLLM's counters correctly: α is the position-0 rate,
# and the 'draft acceptance rate' is (E − 1)/k."
#
# **Drill questions**
# 1. *Our draft's acceptance is 0.55 on chat traffic although it matched the target on benchmarks. Why?* — α is
#    a property of the pair *on this traffic*; if the target is a fine-tune, retrain the draft on the target's
#    own outputs for this traffic (SpecForge's data-regeneration step does exactly this).
# 2. *vLLM reports a 'draft acceptance rate' of 0.33 at k = 4. Is that α?* — No: it is (E − 1)/k. With E = 2.31
#    that is α ≈ 0.6 (per-position[0]). Read `…_accepted_tokens_per_pos` for α.
# 3. *Why not the biggest draft that fits?* — c grows with size and α saturates; the speedup E/(k·c + 1) peaks
#    at the smallest draft that holds the target's behaviour.
