# %% [markdown]
# # 01 · A tiny MoE in torch: the router, the experts, collapse and balance
#
# **Tier:** T0 — torch on a laptop or Colab CPU, a few seconds per training run. Without torch the
# same cells read curves this code recorded on a CPU (`fixtures/tinymoe_curves.json`, labelled);
# every exercise is numpy and runs either way.
#
# ## The one-minute version
#
# * An **MoE layer** replaces one MLP with E expert MLPs and a **router**: a linear layer scores the
#   experts, a softmax (or sigmoid) turns scores into probabilities, the top-k experts run, and their
#   outputs are summed with the probabilities as weights. Parameters grow ~E times; FLOPs per token
#   grow ~k times.
# * Nothing in next-token loss asks the router to spread the tokens. With top-1 routing the experts
#   chosen early get the gradient, get better, get chosen more — **router collapse**: a few experts
#   carry the traffic, others starve. Their parameters still cost memory, and in serving the GPU that
#   holds a hot expert is the slowest one.
# * Two fixes: the **Switch auxiliary loss** `α · E · Σ f_e P_e` (pushes the router's probabilities
#   toward uniform), and **DeepSeek-V3's auxiliary-loss-free bias** (a per-expert bias that only
#   *chooses* experts, nudged each step by the sign of its load error — the gate weights never see it).
# * Experts do specialise, but along the lines the data make cheap: in this toy the token in front
#   of the router explains most of the choice and the domain almost none. Measure; do not assume.
#
# Concepts: PRIMER §2 "The MoE layer", §3 "Routing and load balance" and §4 "Training MoE in brief"
# ([`PRIMER.md`](../../PRIMER.md)). The numpy layer and trainer of this topic's `moe-core` teach the
# same ideas with manual gradients; this notebook is the torch version, shaped like the real models.

# %%
import numpy as np
from moelab import env, hooks
from moelab.tinymoe import ToyTask, load_bundled, load_stats, specialisation, table

print(env.describe())
HAS_TORCH = env.has_torch()
task = ToyTask()
x, dom = task.batch(np.random.default_rng(0), 3)
print(f"vocab {task.vocab} = {task.n_domains} domain tags + {task.n_content} content tokens; "
      f"lowest possible loss {task.bayes_loss():.3f} nats/token")
for row, d in zip(x, dom):
    print(f"domain {d}:", " ".join(map(str, row)))

# %% [markdown]
# Each sequence starts with its domain tag (0–7), then follows that domain's own rule for the next
# content token (a permutation), with 10% noise. The rule depends on the tag at position 0, so the
# model needs attention to carry the tag forward — after attention the router's input *could* say
# which domain it is in.
#
# ## Worked example: total versus active parameters of one MoE layer
#
# The layer below has E = 8 SwiGLU experts of width 16 on a 32-wide residual stream and routes each
# token to k = 1 of them (Switch-style). Every expert sits in memory; one runs per token.

# %%
D, E, K, FF = 32, 8, 1, 16
expert = 3 * D * FF                       # gate, up, down
router = D * E
total, active = router + E * expert, router + K * expert
print(f"one expert {expert:,} params; router {router:,}; layer total {total:,}; active per token {active:,} "
      f"({total / active:.1f}x)")
if HAS_TORCH:
    import torch
    from moelab.tinymoe.model import TinyMoE, TinyMoETransformer
    layer = TinyMoE(D, E, K, FF)
    assert layer.params() == (total, active)
    print("TinyMoE.params() agrees:", layer.params())

# %% [markdown]
# ## Exercise 1.1 — the router
#
# Write `route(logits, k, score="softmax", renormalize=True)` for logits `[N, E]`: turn scores into
# probabilities (softmax over experts, or an element-wise sigmoid as DeepSeek-V3 does), pick the k
# highest per token, and return `(indices [N, k], weights [N, k])` — the chosen probabilities,
# divided by their sum when `renormalize` is on (Mixtral always does; Qwen3 and OLMoE only with
# `norm_topk_prob`). Order the k picks from highest to lowest.

# %% exercise
def route(logits, k, score="softmax", renormalize=True):
    ### BEGIN SOLUTION
    logits = np.asarray(logits, dtype=float)
    if score == "softmax":
        z = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = z / z.sum(axis=-1, keepdims=True)
    else:
        probs = 1 / (1 + np.exp(-logits))
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(probs, idx, axis=-1)
    if renormalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return idx, w
    ### END SOLUTION

# %% check
idx, w = route([[2.0, 1.0, 0.0, -1.0]], 2)
assert idx.tolist() == [[0, 1]] and np.isclose(w[0, 0], 1 / (1 + np.exp(-1)))      # e^2 / (e^2 + e^1)
_, w_raw = route([[2.0, 1.0, 0.0, -1.0]], 2, renormalize=False)
assert np.isclose(w_raw.sum(), (np.exp(2) + np.exp(1)) / np.exp([2, 1, 0, -1]).sum())
idx_s, w_s = route([[0.0, 3.0, -3.0]], 1, score="sigmoid", renormalize=False)
assert idx_s.tolist() == [[1]] and np.isclose(w_s[0, 0], 1 / (1 + np.exp(-3)))
if HAS_TORCH:                                          # the same numbers as the torch router
    torch.manual_seed(0)
    r = TinyMoE(D, E, 2, FF).router
    xs = torch.randn(64, D)
    logits_t, w_t, i_t = r(xs)
    i_np, w_np = route(logits_t.detach().numpy(), 2)
    assert (i_np == i_t.numpy()).all() and np.allclose(w_np, w_t.detach().numpy(), atol=1e-6)
print("✅ softmax -> top-k -> (re)normalise; the sigmoid router scores each expert on its own")

# %% [markdown]
# ## Exercise 1.2 — the layer: gather, one matmul per expert, scatter
#
# Write `moe_forward(x, router_w, gate_up, down, k)`: route `x @ router_w.T` with your `route`
# (softmax, renormalised when k > 1, raw probability when k = 1), then for each expert take the rows
# of the tokens routed to it, run its SwiGLU `(silu(g) * u) @ down[e].T` where
# `g, u = split(x @ gate_up[e].T)`, multiply by the gate weight and add into the output. Shapes are
# those of transformers v5 (and `TinyMoE`): `gate_up [E, 2I, d]`, `down [E, d, I]`. This
# gather → per-expert GEMM → scatter is what fused MoE kernels do in one launch.

# %% exercise
def silu(v):
    return v / (1 + np.exp(-v))


def moe_forward(x, router_w, gate_up, down, k):
    ### BEGIN SOLUTION
    idx, w = route(x @ router_w.T, k, renormalize=k > 1)
    out = np.zeros_like(x)
    for e in range(router_w.shape[0]):
        tok, slot = np.nonzero(idx == e)
        if len(tok) == 0:
            continue
        g, u = np.split(x[tok] @ gate_up[e].T, 2, axis=-1)
        out[tok] += ((silu(g) * u) @ down[e].T) * w[tok, slot][:, None]
    return out
    ### END SOLUTION

# %% check
rng = np.random.default_rng(1)
N, I = 40, FF
xs = rng.normal(size=(N, D))
rw, gu, dn = rng.normal(size=(E, D)), rng.normal(size=(E, 2 * I, D)) / D ** 0.5, rng.normal(size=(E, D, I)) / I ** 0.5
# 1) one expert, top-1: the router's softmax is 1.0, so the layer is exactly the dense MLP
g1, u1 = np.split(xs @ gu[0].T, 2, axis=-1)
assert np.allclose(moe_forward(xs, rw[:1], gu[:1], dn[:1], 1), (silu(g1) * u1) @ dn[0].T)
# 2) brute force: run every expert on every token, keep the routed ones
for k in (1, 2):
    ids, ws = route(xs @ rw.T, k, renormalize=k > 1)
    dense = np.zeros_like(xs)
    for e in range(E):
        g, u = np.split(xs @ gu[e].T, 2, axis=-1)
        dense += ((silu(g) * u) @ dn[e].T) * ((ids == e) * ws).sum(1, keepdims=True)
    assert np.allclose(moe_forward(xs, rw, gu, dn, k), dense)
if HAS_TORCH:                                        # 3) the torch layer, same weights
    layer = TinyMoE(D, E, 2, I)
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor(rw)); layer.gate_up.copy_(torch.tensor(gu)); layer.down.copy_(torch.tensor(dn))
        y = layer(torch.tensor(xs, dtype=torch.float32)).numpy()
    assert np.allclose(moe_forward(xs, rw, gu, dn, 2), y, atol=1e-4)
print("✅ E = 1 is the dense MLP; sparse dispatch gives what computing every expert and masking gives, at k/E of the expert FLOPs")

# %% [markdown]
# ## Exercise 1.3 — the Switch auxiliary loss
#
# For router probabilities `probs [N, E]` and chosen `idx [N, k]`, write
# `switch_aux_loss(probs, idx)` = `E · Σ_e f_e · P_e`, where `f_e` = (assignments to e) / N — so
# Σ f_e = k — and `P_e` = mean probability of e over the tokens. This is transformers'
# `load_balancing_loss_func` normalisation: perfectly uniform routing gives **k**, not 1 (Megatron
# and MegaBlocks divide by k, so theirs is 1 at balance — say which one you mean). Only P carries a
# gradient; f comes out of a top-k.

# %% exercise
def switch_aux_loss(probs, idx):
    ### BEGIN SOLUTION
    probs = np.asarray(probs, float)
    n, e = probs.shape
    f = np.bincount(np.asarray(idx).ravel(), minlength=e) / n
    return float(e * (f * probs.mean(axis=0)).sum())
    ### END SOLUTION

# %% check
uni = np.full((8, 4), 0.25)
assert np.isclose(switch_aux_loss(uni, np.array([[0], [1], [2], [3]] * 2)), 1.0)                   # k = 1
assert np.isclose(switch_aux_loss(uni, np.array([[0, 1], [2, 3]] * 4)), 2.0)                       # k = 2
one_hot = np.eye(4)[[0] * 8]
assert np.isclose(switch_aux_loss(one_hot, np.zeros((8, 1), int)), 4.0)                             # collapse: E
if HAS_TORCH:
    from moelab.tinymoe.model import switch_aux_loss as torch_aux
    lg = torch.randn(256, 8)
    i2 = lg.softmax(-1).topk(2, -1).indices
    assert np.isclose(switch_aux_loss(lg.softmax(-1).numpy(), i2.numpy()), float(torch_aux(lg, i2, 8)), atol=1e-5)
print("✅ uniform -> k (HF normalisation), full collapse -> E; Megatron's version divides by k")

# %% [markdown]
# ## Worked example: train it three ways
#
# Same model, same data, same seed; only the balancing differs: `none`, `aux` (the loss above with
# α = 0.1 — larger than the usual 0.01 because a 400-step run has little time) and `bias` (the
# auxiliary-loss-free rule of Exercise 1.4, step 0.01 per update). With torch this trains here
# (~5 s a run on one CPU thread); without, it reads the runs this code recorded.

# %%
if HAS_TORCH:
    from moelab.tinymoe.train import TrainConfig, train
    runs, models = {}, {}
    for b in ("none", "aux", "bias"):
        run, models[b] = train(TrainConfig(balance=b, seed=0, steps=400))
        runs[b] = run.__dict__
        print(run.summary())
    SOURCE = "trained here (torch, CPU)"
else:
    bundled = load_bundled()
    runs = {r["config"]["balance"]: r for r in bundled["runs"] if r["config"]["seed"] == 0}
    SOURCE = "bundled: " + bundled["_label"]
print(SOURCE)
print(table(list(runs.values())))


def spark(load):
    bars = " ▁▂▃▄▅▆▇█"
    top = max(load)
    return "".join(bars[min(8, int(round(8 * v / top)))] for v in load)


for b in ("none", "bias"):
    print(f"\n{b}: expert load share every 50 steps (bar height = share; 8 experts)")
    for step, load in list(zip(runs[b]["steps"], runs[b]["load"]))[4::5]:
        print(f"  step {step:4d}  {spark(load)}  max/mean {max(load) * len(load):.2f}")

# %% [markdown]
# Read the table: without balancing the busiest expert carries about twice its fair share (and in
# some seeds an expert dies); both fixes bring max/mean close to 1 within a few hundred steps. The
# loss columns are close — this toy has spare capacity; a collapsed model spends parameters it
# never uses, which shows up in quality only when capacity is tight (run `python -m moelab.tinymoe
# --steps 1000 --seeds 0 1 2` and compare the final losses). The three seeds bundled for the numpy
# path tell the same story about load:

# %%
print(table(load_bundled()["runs"]))

# %% [markdown]
# ## Exercise 1.4 — balancing without a loss: the bias rule
#
# DeepSeek-V3 (and Megatron's `moe_router_enable_expert_bias`) keep a bias per expert that is added
# to the scores **only to choose** the top-k; the gate weights still come from the unbiased scores.
# After each step: `bias += rate · sign(mean load − load)` — under-loaded experts rise, over-loaded
# ones fall, by a fixed step (the sign, not the size of the error). Write
# `bias_update(bias, counts, rate)`. The check runs your rule against a stubborn router whose
# scores never change.

# %% exercise
def bias_update(bias, counts, rate):
    ### BEGIN SOLUTION
    counts = np.asarray(counts, float)
    return np.asarray(bias, float) + rate * np.sign(counts.mean() - counts)
    ### END SOLUTION

# %% check
assert np.allclose(bias_update([0, 0, 0, 0], [10, 2, 2, 2], 0.01), [-0.01, 0.01, 0.01, 0.01])
assert np.allclose(bias_update([0.1, 0.0], [5, 5], 0.01), [0.1, 0.0])                  # balanced: no change
rng = np.random.default_rng(0)
fixed = rng.normal(size=(512, 8)) + np.array([1.5, 1.0, 0.5, 0, 0, 0, 0, -1.0])      # a router that prefers 0-2
probs = np.exp(fixed) / np.exp(fixed).sum(1, keepdims=True)
bias = np.zeros(8)
first = None
for step in range(300):
    choice = np.argsort(-(probs + bias), axis=1)[:, :1]
    counts = np.bincount(choice.ravel(), minlength=8)
    first = counts if first is None else first
    bias = bias_update(bias, counts, 0.01)
before, after = load_stats(first / first.sum()), load_stats(counts / counts.sum())
assert before["max_over_mean"] > 2.5 and after["max_over_mean"] < 1.3
moved = choice[:, 0] != probs.argmax(1)                  # tokens the bias sent to a lower-rated expert
gate = np.take_along_axis(probs, choice, axis=1)[:, 0]   # their gate weight: that expert's own probability
assert moved.mean() > 0.2 and (gate[moved] < probs.max(1)[moved]).all()
print(f"✅ max/mean load {before['max_over_mean']:.2f} -> {after['max_over_mean']:.2f} with no loss term; "
      f"the bias moved the choice, not the gate weights")

# %% [markdown]
# ## Exercise 1.5 — what imbalance costs at serving time
#
# Serve the model with expert parallelism over `ep` GPUs and vLLM's default "linear" placement
# (GPU r holds experts `[r·E/ep, (r+1)·E/ep)`). Every GPU must finish its experts before the layer
# ends, so the step waits for the busiest one. Write `ep_slowdown(load, ep)`: the busiest GPU's
# share of the assignments divided by the balanced share `1/ep` (1.0 = no waste).

# %% exercise
def ep_slowdown(load, ep):
    ### BEGIN SOLUTION
    load = np.asarray(load, float) / np.sum(load)
    per_gpu = load.reshape(ep, -1).sum(axis=1)
    return float(per_gpu.max() * ep)
    ### END SOLUTION

# %% check
assert np.isclose(ep_slowdown([1] * 8, 4), 1.0)
assert np.isclose(ep_slowdown([3, 1, 1, 1, 0, 0, 1, 1], 2), 1.5)          # GPUs get 6 and 2 of 8
assert np.isclose(ep_slowdown([3, 1, 1, 1, 0, 0, 1, 1], 4), 2.0)          # pairs 4, 2, 0, 2
assert np.isclose(ep_slowdown([3, 1, 1, 1, 0, 0, 1, 1], 8), 3.0)
rows = {b: [ep_slowdown(r["load"][-1], ep) for ep in (2, 4, 8)] for b, r in runs.items()}
for b, v in rows.items():
    print(f"   {b:5s} EP=2 {v[0]:.2f}x  EP=4 {v[1]:.2f}x  EP=8 {v[2]:.2f}x")
assert rows["none"][2] > rows["bias"][2] and rows["none"][2] > rows["aux"][2]
assert rows["none"][2] >= rows["none"][0]                                 # more ranks, less averaging
print("✅ the collapsed router's hottest expert sets the pace of every EP step; more ranks average less")

# %% [markdown]
# ## Exercise 1.6 — do experts specialise, and on what?
#
# From co-occurrence counts `[values of X, experts]` write `specialisation(counts)` = I(X; expert) /
# H(expert): the share of the router's choice that property X explains (0 = unrelated, 1 = X
# decides). Then ask it twice: X = the sequence's domain, and X = the current token.

# %% exercise
def my_specialisation(counts):
    ### BEGIN SOLUTION
    c = np.asarray(counts, float)
    pj = c / c.sum()
    px, pe = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = (pj[nz] * np.log2(pj[nz] / (px @ pe)[nz])).sum()
    he = -(pe[pe > 0] * np.log2(pe[pe > 0])).sum()
    return float(mi / he) if he else 0.0
    ### END SOLUTION

# %% check
assert np.isclose(my_specialisation(np.eye(4) * 10), 1.0)
assert np.isclose(my_specialisation(np.ones((4, 4))), 0.0, atol=1e-12)
for b, r in runs.items():
    assert np.isclose(my_specialisation(r["domain_expert"]), specialisation(r["domain_expert"]))
    print(f"   {b:5s}: expert explained by domain {my_specialisation(r['domain_expert']):.2f}, "
          f"by current token {my_specialisation(r['token_expert']):.2f}")
assert all(my_specialisation(r["token_expert"]) > 5 * my_specialisation(r["domain_expert"]) for r in runs.values())
print("✅ the router keys on the token in front of it far more than on the domain a person would name")

# %% [markdown]
# That is the practical lesson for serving: which experts are hot depends on *the tokens* your
# traffic contains, so hot sets shift with the workload mix (notebook 02 measures a real router).
#
# ## Worked example: record the router with hooks (a preview of notebook 02)
#
# `TinyTopKRouter` returns `(logits, weights, indices)` like the Hugging Face v5 routers, so
# `moelab.hooks.RouterRecorder` — written for OLMoE, Mixtral, Qwen-MoE — records it unchanged.

# %%
if HAS_TORCH:
    m = models["bias"]
    rec = hooks.RouterRecorder(m)
    xb, _ = task.batch(np.random.default_rng(9), 4)
    with torch.no_grad():
        m(torch.from_numpy(xb))
    ids = rec.pop()
    rec.remove()
    print("recorded", rec.names, "->", ids.shape, "[tokens, layers, k]")
    assert (ids[:, 0, :] == m.moes[0].last["indices"].numpy()).all()
    print("per-expert assignments in this batch:", hooks.utilisation(ids, 8)[0].tolist())
else:
    print("T0 without torch: notebook 02 reads bundled router traces instead")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "An MoE layer is a router plus E ordinary MLPs; each token runs k of them, so we
# pay memory for all E and compute for k. The router needs help to spread the load: left alone, top-1
# routing collapses — in our toy the busiest expert carried about 2x its share within 400 steps.
# Training adds either the Switch loss, which nudges the router's probabilities toward uniform at
# some cost to the language-model objective, or DeepSeek-V3's bias, which only changes which experts
# are chosen and so leaves the gate weights alone. The same imbalance matters at inference: with
# expert parallelism the GPU holding the hottest expert sets the step time, and the fewer experts per
# GPU, the less it averages out. Experts specialise, but by token far more than by topic — so the hot
# set follows the traffic mix, and we measure it rather than guess."
#
# **Drill 1.** *Your aux loss reads 2.0. Is the router collapsing?* — Depends on the normalisation:
# transformers' `load_balancing_loss_func` is k at perfect balance (2.0 for top-2 is ideal); Megatron's
# and MegaBlocks' are 1. Look at max/mean load and dead experts, not the raw loss.
#
# **Drill 2.** *Why does DeepSeek-V3's bias not distort outputs the way the aux loss can?* — It is
# added to the scores only for the top-k choice; the gate weights come from the unbiased scores, and
# there is no gradient term competing with the language-model loss.
#
# **Drill 3.** *Can we assign experts to "the code GPU" and "the prose GPU"?* — Not on the evidence:
# routers key mostly on local token features (here the current token explains ~0.8 of the choice, the
# domain ~0.03). Balance by measured load (EPLB, redundant experts), not by topic.
