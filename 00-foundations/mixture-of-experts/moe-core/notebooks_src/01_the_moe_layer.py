# %% [markdown]
# # 01 · The MoE layer
#
# **Tier:** T0. It needs numpy on a laptop or Colab CPU, a few seconds and no network. To see a real router select
# experts on a GPU, continue with `../moe-lab/notebooks/02_watch_the_router.ipynb` (T1, with a fallback to bundled
# traces).
#
# ## The one-minute version
# A mixture-of-experts layer replaces the one MLP of the transformer block with **$E$ expert MLPs** and a **router**.
# The router is a linear map from the hidden state of the token to $E$ scores. Each token keeps its **top-k** experts.
# Its output is the sum of the outputs of only those $k$ experts, with the router weights. Models that have a **shared
# expert** also add the output of that expert, and every token uses it.
#
# Memory holds all $E$ experts, but the FLOPs of a token pay for $k$ only. Thus the parameters increase approximately
# $E/k$ times, and the compute per token almost does not change. This notebook shows the §9 row of the transformer
# primer in concrete numbers.
#
# The families differ in details that change the numbers:
# - softmax scores or sigmoid scores
# - a router that renormalises the $k$ weights, or a router that does not
# - a bias that selects experts but never weights them (DeepSeek-V3)
# - coarse experts (Mixtral: 8, top-2) or fine-grained experts (DeepSeek-V3: 256 + 1 shared, top-8)
#
# After this notebook, you can do these things:
# - Route a token by hand.
# - Run the layer as fused kernels do: sort by expert, do one GEMM per expert, and scatter the rows back.
# - Count the total and active parameters from a config.
# - Tell which of three "active" conventions a published number uses.
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
# The layer has eight experts and uses the router of Mixtral with top-2 routing. This router does a softmax over
# all eight scores and keeps the best two. Then it renormalises the two weights to a sum of 1.

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
# `forward()` operates as a fused MoE kernel does. It flattens the $T \times k$ assignments and **sorts them by
# expert**. Then it runs each expert one time over its contiguous slice (a *grouped GEMM*). After that, it scatters the
# weighted rows back and adds the $k$ copies of each token. An expert that no token selected does no work.
#
# `forward_dense()` runs all eight experts on every token and multiplies by a gate that is zero outside the top-k. It
# gives the same answer at $E/k$ times the FLOPs.

# %%
flat = r.idx.ravel()
order = np.argsort(flat, kind="stable")
print("assignment slots sorted by expert:", order.tolist())
print("their experts                    :", flat[order].tolist())
print("token of each slot (slot // k)   :", (order // 2).tolist())

# %% [markdown]
# ## Worked example 2 — the routers on the same scores
# The cell puts the router logits of one token through the rules of each family (`moe.ROUTERS`). Look at the weights:
# - Mixtral renormalises them to 1.
# - OLMoE (`norm_topk_prob=False`, also the default of transformers for Qwen2/Qwen3-MoE) keeps the raw softmax mass.
#   But reports say that the released Qwen3 MoE configs set it true (verify: read the config of the checkpoint). Thus
#   those configs give weights as Mixtral does.
# - gpt-oss selects on the logits and does a softmax over only those $k$. This is *exactly* the renormalised softmax of
#   Mixtral (the exponent ratios are the same).
# - DeepSeek-V3 uses sigmoid scores, renormalises them and multiplies them by 2.5.
# - Llama 4 keeps one expert, and its sigmoid score scales the *input* of the expert.

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
# A **shared expert** runs on every token with weight 1. Examples:
# - DeepSeek-V3: 1 shared + 256 routed.
# - Llama 4: 1 shared.
# - Qwen1.5-MoE: one shared expert four routed experts wide, scaled by $\operatorname{sigmoid}(x \cdot g)$.
#
# The shared expert holds what every token needs, so the routed experts can specialise. **Fine-grained** experts divide
# the same parameters into more, smaller experts, and the router sends each token to more of them. Mixtral has 8
# experts of width 14,336 (top-2). DeepSeek-V3 has 256 of width 2,048 (top-8). The idea is the same, but the serving
# behaviour has large differences (notebook 03).

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
# Write `topk_route(logits, k, renormalise)` for a softmax router. It returns `(idx, weights)`. Do a softmax over all
# $E$ scores. Take the $k$ largest (largest first). Renormalise the $k$ weights to a sum of 1 only if `renormalise` is
# true.

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
for renorm, fam in ((True, "mixtral"), (False, "olmoe")):
    idx, w = topk_route(L, 4, renorm)
    ref = route(L, 4, **ROUTERS[fam])
    assert (idx == ref.idx).all() and np.allclose(w, ref.weights), fam
print(f"✅ softmax -> top-k -> (renormalise): Mixtral's weights sum to 1; OLMoE's to "
      f"{route(L, 4, **ROUTERS['olmoe']).weights.sum(1).mean():.2f} on average here")

# %% [markdown]
# ## Exercise 1.2 — the sparse forward pass
# Write `moe_forward(x, w_router, experts, k)` for the Mixtral router. Route with your `topk_route`. Then, for each
# expert, gather the tokens routed to it. Run the expert **one time** on that slice. Add `weight × output` back into
# the row of each token. An expert must not run on a token that did not select it.

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
# The config of Mixtral gives these values:
# - 32 layers, $d = 4{,}096$.
# - 32 query heads and 8 KV heads of 128.
# - SwiGLU experts of width 14,336, $E = 8$, $k = 2$.
# - vocabulary 32,000, an untied embedding and LM head.
# - a bias-free router ($d \times E$).
#
# Calculate `total` and `active`. Count both embedding tables. This is the convention that
# `roofline.llm.active_params()` uses.

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
# OpenAI reports 5.1B active parameters for gpt-oss-120b. DeepSeek reports 37B for V3. `roofline.llm` counts both
# embedding tables. Use `MoEConfig.active(embeddings=...)` with `"both"`, `"head"` and `"none"`. Set `oss` and `ds` to
# dicts of the three counts (in billions, 2 decimals). Then set `oss_convention` and `ds_convention` to the convention
# that agrees with each published number (5.13 and 37.55 in our arithmetic).

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
# Keep the expert parameters per layer of Mixtral constant (8 experts × width 14,336). But divide each expert into 8
# narrower experts (64 experts of width 1,792), and route each token to 16 of them. Calculate these values per layer:
# - `flops_ratio` = FLOPs per token (fine ÷ coarse).
# - `params_ratio` = expert params (fine ÷ coarse).
# - `combos_coarse`, `combos_fine` = the number of different expert sets that a token can select (`math.comb`).

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
# **The two-minute version.** "An MoE layer replaces the MLP of the block with $E$ expert MLPs and a linear router.
# Each token gives a score to all $E$ experts and keeps its top-k. It adds the outputs of those $k$ experts with the
# router weights. Models that have a shared expert also add that expert. All $E$ experts stay in memory, but each token
# calculates only $k$ of them.
#
# "Thus a model like Mixtral holds 46.7B parameters but runs 12.9B per token, and
# DeepSeek-V3 holds 671B and runs approximately 37B.
#
# "The routers differ: softmax or sigmoid, renormalised or not, and the selection-only bias of DeepSeek. The 'active'
# numbers also differ: gpt-oss counts only the LM head, and DeepSeek counts both embedding tables. Kernels sort the
# tokens by expert and do one grouped GEMM. Thus an expert that no token selected costs no FLOPs. But it still costs
# HBM, and at inference that is the whole story of the next notebooks."
#
# **Drill questions**
# 1. *Why is Mixtral 46.7B and not 8 × 7B = 56B?* The model replicates only the MLPs. All experts share the attention,
#    the norms and the embedding tables.
# 2. *Two routers select the same experts. Why can their outputs be different?* The combine weights differ. A router
#    renormalises them or does not (the default of Qwen3 is not). DeepSeek uses sigmoid × 2.5. Llama 4 applies them to the
#    input.
# 3. *What does a shared expert give you?* Common knowledge lives in one MLP that is always on, so the routed experts
#    can specialise. It also adds a constant quantity of active parameters that every token pays for.
