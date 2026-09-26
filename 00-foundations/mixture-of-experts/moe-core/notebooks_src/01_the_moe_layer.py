# %% [markdown]
# # 01 · The MoE layer
#
# **Tier:** T0 — numpy on a laptop or Colab CPU, a few seconds, no network. To watch a real router pick experts
# on a GPU, continue with `../moe-lab/notebooks/02_watch_the_router.ipynb` (T1; it falls back to bundled traces).
#
# ## The one-minute version
# A mixture-of-experts layer replaces the transformer block's one MLP with **E expert MLPs** and a **router**: a
# linear map from the token's hidden state to E scores. Each token keeps its **top-k** experts and its output is
# the router-weighted sum of just those k experts' outputs, plus a **shared expert** that every token uses, in
# models that have one. Memory holds all E experts; a token's FLOPs pay for k. So parameters grow about E/k-fold
# while the compute per token barely moves — the transformer primer's §9 row, made concrete. The families differ
# in details that change numbers: softmax or sigmoid scores, whether the k weights are renormalised, a bias that
# chooses but never weights (DeepSeek-V3), coarse experts (Mixtral: 8, top-2) or fine-grained ones (DeepSeek-V3:
# 256 + 1 shared, top-8). After this notebook you can route a token by hand, run the layer the way fused kernels
# do (sort by expert, one GEMM per expert, scatter back), and count total and active parameters from a config —
# and say which of three "active" conventions a published number uses.
#
# Primer: `../PRIMER.md` §1 *Why sparsity* and §2 *The MoE layer*.

# %%
import math

import numpy as np

from moecore import moe
from moecore.moe import ROUTERS, MoELayer, route, softmax
from moecore.sizing import MODELS

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — route six tokens, then run the layer
# Eight experts, top-2, Mixtral's router: softmax over all eight scores, keep the best two, renormalise the two
# weights to sum to 1.

# %%
layer = MoELayer.init(rng, d=16, ff=32, n_experts=8, k=2, **ROUTERS["mixtral"])
x = rng.standard_normal((6, 16))
r = layer.route(x)
for t in range(6):
    print(f"token {t}: experts {r.idx[t].tolist()}  weights {np.round(r.weights[t], 3).tolist()}  "
          f"(softmax mass on them before renormalising: {r.probs[t, r.idx[t]].sum():.2f})")
print("rows per expert:", np.bincount(r.idx.ravel(), minlength=8).tolist(), "- T x k = 12 rows in all")
print("sparse forward == dense reference:", np.allclose(layer.forward(x), layer.forward_dense(x)))

# %% [markdown]
# `forward()` is written the way a fused MoE kernel runs: flatten the T × k assignments, **sort them by expert**,
# run each expert once over its contiguous slice (a *grouped GEMM*), then scatter the weighted rows back and sum
# each token's k copies. An expert nobody chose does no work. `forward_dense()` runs all eight experts on every
# token and multiplies by a gate that is zero off the top-k — the same answer at E/k times the FLOPs.

# %%
flat = r.idx.ravel()
order = np.argsort(flat, kind="stable")
print("assignment slots sorted by expert:", order.tolist())
print("their experts                    :", flat[order].tolist())
print("token of each slot (slot // k)   :", (order // 2).tolist())

# %% [markdown]
# ## Worked example 2 — five routers on the same scores
# One token's router logits, pushed through each family's rules (`moe.ROUTERS`). Watch the weights: Mixtral
# renormalises to 1; Qwen3 and OLMoE (HF default `norm_topk_prob=False`) keep the raw softmax mass; gpt-oss picks
# on the logits and softmaxes just those k — which is *exactly* Mixtral's renormalised softmax (the exponent ratios
# are the same); DeepSeek-V3 uses sigmoid scores, renormalises them and multiplies by 2.5; Llama 4 keeps one expert
# and its sigmoid score scales the expert's *input*.

# %%
logits = np.array([[2.0, 1.2, 0.4, 0.3, -0.5, -1.0, -1.1, -2.0]])
for fam, opts in ROUTERS.items():
    k = 1 if fam == "llama4" else 2
    rr = route(logits, k, **opts)
    print(f"{fam:12s} experts {rr.idx[0].tolist()}  weights {np.round(rr.weights[0], 3).tolist()}  sum {rr.weights[0].sum():.3f}")
bias = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])            # DeepSeek's balancing bias: it changes WHO, not HOW MUCH
rb = route(logits, 2, select_bias=bias, **ROUTERS["deepseek-v3"])
print(f"deepseek + bias on expert 7: experts {rb.idx[0].tolist()} weights {np.round(rb.weights[0], 3).tolist()} "
      "- expert 7 gets in on its biased score, weighted by its unbiased one")

# %% [markdown]
# ## Worked example 3 — shared and fine-grained experts
# A **shared expert** runs on every token with weight 1 (DeepSeek-V3: 1 shared + 256 routed; Llama 4: 1 shared;
# Qwen1.5-MoE: one shared expert four routed experts wide, scaled by `sigmoid(x · g)`). It carries what every token
# needs, so the routed experts can specialise. **Fine-grained** experts split the same parameters into more,
# smaller experts and route to more of them: Mixtral has 8 experts of width 14,336 (top-2); DeepSeek-V3 has 256 of
# width 2,048 (top-8). Same idea, very different serving behaviour (notebook 03).

# %%
print(f"{'model':24s} {'E':>4s} {'k':>3s} {'shared':>6s} {'expert I':>8s} {'d':>5s} {'total':>9s} {'active':>8s} {'x':>5s}")
for key in ("mixtral-8x7b", "qwen3-30b-a3b", "qwen1.5-moe-a2.7b", "deepseek-v3", "gpt-oss-120b", "llama4-maverick",
            "olmoe-1b-7b", "llama-3.1-8b"):
    c = MODELS[key]
    print(f"{c.name:24s} {c.n_experts:4d} {c.top_k:3d} {c.n_shared:6d} {c.expert_ff:8d} {c.d_model:5d} "
          f"{c.total() / 1e9:8.2f}B {c.active() / 1e9:7.2f}B {c.total() / c.active():5.1f}")
q15 = MODELS["qwen1.5-moe-a2.7b"]
print(f"\nQwen1.5-MoE: shared width {q15.shared_ff} = {q15.shared_ff // q15.expert_ff} x {q15.expert_ff}: "
      f"a token uses {q15.top_k} routed + {q15.shared_ff // q15.expert_ff} shared-sized units of "
      f"{q15.n_experts + q15.shared_ff // q15.expert_ff}")

# %% [markdown]
# ## Exercise 1.1 — the router
# Write `topk_route(logits, k, renormalise)` → `(idx, weights)` for a softmax router: softmax over all E scores,
# take the k largest (largest first), and renormalise the k weights to sum to 1 only if `renormalise`.

# %% exercise
def topk_route(logits, k, renormalise=True):
    ### BEGIN SOLUTION
    p = softmax(logits)
    idx = np.argsort(-p, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(p, idx, axis=1)
    if renormalise:
        w = w / w.sum(axis=1, keepdims=True)
    return idx, w
    ### END SOLUTION

# %% check
L = rng.standard_normal((50, 16))
for renorm, fam in ((True, "mixtral"), (False, "qwen3")):
    idx, w = topk_route(L, 4, renorm)
    ref = route(L, 4, **ROUTERS[fam])
    assert (idx == ref.idx).all() and np.allclose(w, ref.weights), fam
print(f"✅ softmax -> top-k -> (renormalise): Mixtral's weights sum to 1; Qwen3's to "
      f"{route(L, 4, **ROUTERS['qwen3']).weights.sum(1).mean():.2f} on average here")

# %% [markdown]
# ## Exercise 1.2 — the sparse forward pass
# Write `moe_forward(x, w_router, experts, k)` for the Mixtral router: route with your `topk_route`, then for each
# expert gather the tokens routed to it, run the expert **once** on that slice, and add `weight × output` back into
# each token's row. No expert may run on a token that did not choose it.

# %% exercise
def moe_forward(x, w_router, experts, k):
    ### BEGIN SOLUTION
    idx, w = topk_route(x @ w_router, k)
    out = np.zeros_like(x)
    for e, expert in enumerate(experts):
        tok, slot = np.nonzero(idx == e)
        if len(tok):
            out[tok] += w[tok, slot][:, None] * expert(x[tok])
    return out
    ### END SOLUTION

# %% check
lay = MoELayer.init(rng, d=12, ff=24, n_experts=6, k=2, **ROUTERS["mixtral"])
xx = rng.standard_normal((40, 12))
assert np.allclose(moe_forward(xx, lay.w_router, lay.experts, 2), lay.forward_dense(xx))
one = MoELayer.init(rng, d=12, ff=24, n_experts=1, k=1, **ROUTERS["mixtral"])
assert np.allclose(moe_forward(xx, one.w_router, one.experts, 1), one.experts[0](xx))
print("✅ gather -> expert -> weighted scatter-add equals every-expert-on-every-token; with E = k = 1 it is the dense MLP")

# %% [markdown]
# ## Exercise 1.3 — count Mixtral by hand
# From its config: 32 layers, d = 4,096, 32 query heads and 8 KV heads of 128, SwiGLU experts of width 14,336,
# E = 8, k = 2, vocabulary 32,000, untied embedding and LM head, a bias-free router (d × E). Compute `total` and
# `active` (both embedding tables counted, the convention `roofline.llm.active_params()` uses).

# %% exercise
### BEGIN SOLUTION
d, L_, E_, k_, ff, V = 4096, 32, 8, 2, 14_336, 32_000
attn = 2 * d * 32 * 128 + 2 * d * 8 * 128
expert = 3 * d * ff
router = d * E_
total = L_ * (attn + E_ * expert + router) + 2 * V * d
active = L_ * (attn + k_ * expert + router) + 2 * V * d
### END SOLUTION

# %% check
assert total == MODELS["mixtral-8x7b"].total() == 46_702_526_464
assert active == MODELS["mixtral-8x7b"].active() == 12_879_659_008
print(f"✅ Mixtral: {total / 1e9:.2f}B total, {active / 1e9:.2f}B active ({total / active:.1f}x) - "
      f"the name '8x7B' double-counts: attention and embeddings are shared, so 8 x 7B would be ~56B")

# %% [markdown]
# ## Exercise 1.4 — which "active"?
# OpenAI reports gpt-oss-120b as 5.1B active; DeepSeek reports 37B for V3; `roofline.llm` counts both embedding
# tables. Using `MoEConfig.active(embeddings=...)` with `"both"`, `"head"` and `"none"`, set `oss` and `ds` to dicts of
# the three counts (in billions, 2 decimals), then set `oss_convention` and `ds_convention` to the one that
# reproduces each published number (5.13 and 37.55 in our arithmetic).

# %% exercise
### BEGIN SOLUTION
oss = {c: round(MODELS["gpt-oss-120b"].active(c) / 1e9, 2) for c in ("both", "head", "none")}
ds = {c: round(MODELS["deepseek-v3"].active(c) / 1e9, 2) for c in ("both", "head", "none")}
oss_convention, ds_convention = "head", "both"
### END SOLUTION

# %% check
assert oss[oss_convention] == 5.13 and ds[ds_convention] == 37.55
assert oss == {"both": 5.71, "head": 5.13, "none": 4.55}
print(f"✅ gpt-oss-120b {oss}, DeepSeek-V3 {ds}: a 201K-token vocabulary at d = 2,880 is 0.58B per table - "
      "11% of gpt-oss's active count. Name the convention before comparing active parameters")

# %% [markdown]
# ## Exercise 1.5 — fine-grained experts at equal cost
# Keep Mixtral's expert parameters per layer fixed (8 experts × width 14,336) but split each expert into 8
# narrower ones (64 experts of width 1,792) and route each token to 16 of them. Compute, per layer:
# `flops_ratio` = FLOPs per token (fine ÷ coarse), `params_ratio` = expert params (fine ÷ coarse), and
# `combos_coarse`, `combos_fine` = the number of distinct expert sets a token can pick (`math.comb`).

# %% exercise
### BEGIN SOLUTION
coarse_flops, fine_flops = 2 * 3 * 4096 * 14_336, 16 * 3 * 4096 * 1792
flops_ratio = fine_flops / coarse_flops
params_ratio = (64 * 3 * 4096 * 1792) / (8 * 3 * 4096 * 14_336)
combos_coarse, combos_fine = math.comb(8, 2), math.comb(64, 16)
### END SOLUTION

# %% check
assert flops_ratio == 1.0 and params_ratio == 1.0
assert combos_coarse == 28 and combos_fine > 4e14
print(f"✅ same FLOPs, same parameters; {combos_coarse} expert sets vs {combos_fine:.2e} - the DeepSeekMoE argument "
      "for fine granularity. The serving price comes in notebook 03: more, smaller experts per token touch more of them")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "An MoE layer swaps the block's MLP for E expert MLPs and a linear router. Each token
# scores all E experts, keeps its top-k, and sums those k outputs weighted by the router — plus a shared expert
# in models that have one. All E experts sit in memory; each token computes only k of them, so a model like
# Mixtral holds 46.7B parameters but runs 12.9B per token, and DeepSeek-V3 holds 671B and runs about 37B. The
# routers differ — softmax or sigmoid, renormalised or not, DeepSeek's selection-only bias — and so do the
# 'active' numbers: gpt-oss counts only the LM head, DeepSeek both embedding tables. Kernels run the layer by
# sorting tokens by expert and doing one grouped GEMM, so an unchosen expert costs no FLOPs — but it still costs
# HBM, and at inference that is the whole story of the next notebooks."
#
# **Drill questions**
# 1. *Why is Mixtral 46.7B and not 8 × 7B = 56B?* — Only the MLPs are replicated; attention, norms and the
#    embedding tables are shared by all experts.
# 2. *Two routers pick the same experts; why can their outputs differ?* — The combine weights: renormalised or not
#    (Qwen3's default is not), sigmoid × 2.5 (DeepSeek), or applied to the input (Llama 4).
# 3. *What does a shared expert buy?* — Common knowledge lives in one always-on MLP, so the routed experts can
#    specialise; it also adds a fixed chunk of active parameters every token pays for.
