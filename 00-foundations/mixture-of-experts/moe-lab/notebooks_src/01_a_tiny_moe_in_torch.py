# %% [markdown]
# # 01 · A tiny MoE in torch: the router, the experts, collapse and balance
#
# **Tier:** T0: torch on a laptop or Colab CPU, a few seconds for each training run. Without torch, the same
# cells read curves that this code recorded on a CPU (`fixtures/tinymoe_curves.json`, labelled). Every
# exercise is numpy and runs with or without torch.
#
# ## The one-minute version
#
# * An **MoE layer** replaces one MLP with $E$ expert MLPs and a **router**. A linear layer gives a score to
#   each expert. A softmax (or a sigmoid) changes the scores into probabilities. The top-k experts run, and
#   the layer adds their outputs with the probabilities as weights. The parameters increase approximately $E$
#   times. The FLOPs per token increase approximately $k$ times.
# * Nothing in the next-token loss asks the router to spread the tokens. With top-1 routing, the experts that
#   the router selects early get the gradient, become better and get more selections. This is **router
#   collapse**: a few experts carry the traffic, and the other experts get almost no tokens. At the start, the
#   router sees almost the same input for every token, because real hidden states share a large common
#   direction. Thus the same few experts win from step 0. Dead experts still cost memory, and the model loses
#   the capacity that these experts were there to add. In serving, the GPU that holds a hot expert does the
#   most work.
# * There are two solutions. The first is the **Switch auxiliary loss** $\alpha \cdot E \cdot \sum f_e P_e$,
#   which pushes the probabilities of the router toward uniform. The second is the **auxiliary-loss-free bias
#   of DeepSeek-V3**. It is a per-expert bias that only *selects* experts. Each step moves it by a small
#   constant quantity, in the direction of the sign of its load error. The gate weights never see it.
# * Experts do specialise, but along the lines that the data make low-cost. In this toy, the token in front of
#   the router explains most of the selection, and the domain explains almost none. Measure. Do not assume.
#
# Concepts: PRIMER §2 "The MoE layer", §3 "Routing and load balance" and §4 "Training MoE in brief"
# ([`PRIMER.md`](../../PRIMER.md)). The numpy layer and the trainer of the `moe-core` of this topic teach the
# same ideas with manual gradients. This notebook is the torch version, with the shape of the real models.

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
# Each sequence starts with its domain tag (0–7). Then it obeys the rule of that domain for the next content
# token (a permutation), with 10% noise. The rule depends on the tag at position 0. Thus the model needs
# attention to carry the tag forward. After attention, the input of the router *can* tell which domain the
# sequence is in.
#
# ## Worked example: total versus active parameters of one MoE layer
#
# The layer in the next cell has $E = 8$ SwiGLU experts of width 16 on a 32-wide residual stream. It routes
# each token to $k = 1$ of them (Switch-style). Every expert is in memory. One expert runs per token.

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
# Write `route(logits, k, score="softmax", renormalize=True)` for logits `[N, E]`. Do these steps:
#
# 1. Change the scores into probabilities. Use a softmax over the experts, or an element-wise sigmoid as
#    DeepSeek-V3 does.
# 2. Select the $k$ highest probabilities for each token.
# 3. Return `(indices [N, k], weights [N, k])`.
#
# The weights are the selected probabilities. When `renormalize` is on, divide them by their sum. Mixtral
# always does this. Qwen3 and OLMoE do it only with `norm_topk_prob`. Put the $k$ selections in order from
# highest to lowest.

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
# Write `moe_forward(x, router_w, gate_up, down, k)`. Do these steps:
#
# 1. Route `x @ router_w.T` with your `route`. Use the softmax. Renormalise when $k > 1$, and use the raw
#    probability when $k = 1$.
# 2. For each expert, take the rows of the tokens that the router sends to it.
# 3. Run the SwiGLU of the expert, `(silu(g) * u) @ down[e].T`, where `g, u = split(x @ gate_up[e].T)`.
# 4. Multiply the result by the gate weight. Then add it into the output.
#
# The shapes are those of transformers v5 (and `TinyMoE`): `gate_up [E, 2I, d]`, `down [E, d, I]`. Fused MoE
# kernels do this sequence in one launch: a gather, then one GEMM for each expert, then a scatter.

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
# For router probabilities `probs [N, E]` and the selected `idx [N, k]`, write `switch_aux_loss(probs, idx)` =
# $E \cdot \sum_e f_e \cdot P_e$. Here $f_e = (\text{assignments to } e) / N$, thus $\sum f_e = k$. $P_e$ is
# the mean probability of $e$ over the tokens.
#
# This is the normalisation of the transformers `load_balancing_loss_func`. Fully uniform routing gives
# **$k$**, not 1. Megatron and MegaBlocks divide by $k$, thus their loss is 1 at balance. Say which one you
# mean. Only $P$ carries a gradient. $f$ comes out of a top-k.

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
# The three runs have the same model, the same data and the same seed. Only the balance method is different:
#
# * `none`: no balance method.
# * `aux`: the loss of Exercise 1.3 with $\alpha = 0.1$. This is larger than the usual 0.01, because a
#   400-step run has only a short time.
# * `bias`: the auxiliary-loss-free rule of Exercise 1.4, with a step of 0.01 for each update.
#
# Every token embedding shares one large direction (`TrainConfig.common = 12`), as the hidden states of a real
# transformer do. Thus, at step 0, the router gives almost the same scores to every token. With torch, this
# cell trains the model here (approximately 5 s for each run on one CPU thread). Without torch, it reads the
# runs that this code recorded.

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
# Read the table. Without a balance method, the router collapses. The busiest expert carries 6–7× its fair
# share, and five or six of the eight experts are dead (each gets under 2% of the tokens). Thus the model is
# in effect a two- or three-expert MoE that carries the memory of eight experts. Its loss is 0.1–0.3 nats
# worse than the loss of each balanced run. Both solutions keep every expert alive, at max/mean 1.04–1.17.
#
# Remove the shared direction (`python -m moelab.tinymoe --common 0`). Then the same toy only moves slowly to
# about 2× on the busiest expert, with one dead expert at the most. Collapse needs router inputs that look
# alike, and real hidden states do look alike. The three bundled seeds for the numpy path tell the same story:

# %%
print(table(load_bundled()["runs"]))

# %% [markdown]
# ## Exercise 1.4 — balancing without a loss: the bias rule
#
# DeepSeek-V3 (and the Megatron `moe_router_enable_expert_bias`) keep a bias for each expert. They add this
# bias to the scores **only to select** the top-k. The gate weights still come from the unbiased scores.
#
# After each step, the rule is `bias += rate · sign(mean load − load)`. The bias of an under-loaded expert
# increases, and the bias of an over-loaded expert decreases. Each change is a constant step, because the rule
# uses the sign of the error, not its size. Write `bias_update(bias, counts, rate)`. The check runs your rule
# against a router with scores that never change.

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
# Serve the model with expert parallelism over `ep` GPUs and the default "linear" placement of vLLM. In this
# placement, GPU $r$ holds experts `[r·E/ep, (r+1)·E/ep)`. Every GPU must complete its experts before the
# layer ends. Thus the step waits for the busiest GPU. Write `ep_slowdown(load, ep)`. It is the share of the
# assignments on the busiest GPU, divided by the balanced share `1/ep` (1.0 = no waste).

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
# From co-occurrence counts `[values of X, experts]`, write `specialisation(counts)` =
# $I(X; \text{expert}) / H(\text{expert})$. This is the share of the selection of the router that the property
# $X$ explains (0 = unrelated, 1 = $X$ decides). Then ask it two times: once with $X$ = the domain of the
# sequence, and once with $X$ = the current token.

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
balanced = [runs[b] for b in ("aux", "bias")]         # a collapsed router has little choice left to explain
assert all(my_specialisation(r["token_expert"]) > 5 * my_specialisation(r["domain_expert"]) for r in balanced)
print("✅ a healthy router keys on the token in front of it far more than on the domain a person would name")

# %% [markdown]
# This is the practical lesson for serving. Which experts are hot depends on *the tokens* that your traffic
# contains. Thus the hot sets change with the workload mix (notebook 02 measures a real router).
#
# ## Worked example: record the router with hooks (a preview of notebook 02)
#
# `TinyTopKRouter` returns `(logits, weights, indices)`, as the Hugging Face v5 routers do.
# `moelab.hooks.RouterRecorder` is for OLMoE, Mixtral and Qwen-MoE. Because `TinyTopKRouter` returns the same
# tuple, the recorder records it with no change.

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
# **Two minutes:** "An MoE layer is a router plus $E$ ordinary MLPs. Each token runs $k$ of them, thus we pay
# memory for all $E$ and compute for $k$. The router needs help to spread the load. With no help, top-1
# routing collapses. In our toy, the busiest expert carried 6-7x its share within 400 steps, most experts were
# dead, and the loss was worse.
#
# "Training adds one of two things. The first is the Switch loss, which pushes the probabilities of the router
# a small step toward uniform, at some cost to the language-model objective. The second is the DeepSeek-V3
# bias, which only changes which experts the router selects. Thus it does not change the gate weights.
#
# "The same imbalance is important at inference. With expert parallelism, the GPU that holds the hottest
# expert sets the step time. The fewer experts there are on each GPU, the less the load averages out. Experts
# specialise, but by token much more than by topic. Thus the hot set changes with the traffic mix, and we
# measure it. We do not guess it."
#
# **Drill 1.** *Your aux loss reads 2.0. Is the router in collapse now?* It depends on the normalisation. The
# transformers `load_balancing_loss_func` is $k$ at perfect balance (2.0 for top-2 is ideal), and the loss of
# Megatron and of MegaBlocks is 1. Look at the max/mean load and the dead experts, not at the raw loss.
#
# **Drill 2.** *Why does the DeepSeek-V3 bias not make the outputs worse in the way that the aux loss can?* The
# rule adds the bias to the scores only for the top-k selection. The gate weights come from the unbiased
# scores. Also, no gradient term competes with the language-model loss.
#
# **Drill 3.** *Can we assign experts to "the code GPU" and "the prose GPU"?* Not on the evidence. Routers key
# mostly on local token features. Here, in the balanced runs, the current token explains ~0.8 of the
# selection, and the domain explains ~0.04. Balance by measured load (EPLB, redundant experts), not by topic.
