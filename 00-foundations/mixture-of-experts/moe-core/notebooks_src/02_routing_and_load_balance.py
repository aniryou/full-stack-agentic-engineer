# %% [markdown]
# # 02 · Routing and load balance
#
# **Tier:** T0. It needs numpy on a laptop or Colab CPU. The training runs take about 10 seconds in all. The torch
# version of the same experiment is `../moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb` (T0 with torch installed).
#
# ## The one-minute version
# The same loss trains the router and the experts. That loss gives a reward when the router sends a token to the
# expert that is *already* good at it. The expert that wins early gets the gradient, becomes better, and wins more. An
# expert that gets no tokens never trains and never gets a chance. If nothing stops it, a layer **collapses** onto a
# few experts. The other experts are dead weight in HBM.
#
# You must force the balance. These are the methods:
# - an **auxiliary loss** $E \cdot \sum f_e \cdot P_e$ (Switch/GShard). It is smallest when the shares of tokens $f$ and
#   the mean probabilities $P$ are both uniform.
# - a **capacity** per expert with the overflow dropped, or **dropless** routing that pads the rows (MegaBlocks, and
#   every inference engine).
# - a **selection-only bias** that moves in steps against the load (the auxiliary-loss-free balance method of
#   DeepSeek-V3).
# - **expert-choice** routing, where the experts select the tokens.
#
# The **z-loss** keeps the router logits small for bf16.
#
# After this notebook, you can calculate each of these by hand. You can also see a router collapse and recover, and
# tell what each solution costs.
#
# Primer: `../PRIMER.md` §3 *Routing and load balance* and §4 *Training MoE in brief*.

# %%
import numpy as np

from moecore import routing as R
from moecore import train as T
from moecore.moe import ROUTERS, route, softmax

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — the Switch loss, in both normalisations
# $f_e$ is the share of the assignments of the batch that went to expert $e$. $P_e$ is the mean router probability of
# $e$. The `load_balancing_loss_func` of transformers lets $\sum f = k$. Thus a perfectly uniform router gets a score
# of **$k$** (2 for Mixtral). Megatron and MegaBlocks divide by $k$, so a uniform router gets **1**. Both are $E$ at
# the worst ($k = 1$, one expert takes all).
#
# Only $P$ has a gradient: $f$ comes out of a top-k.

# %%
t, e, k = 64, 8, 2
uniform_p = np.full((t, e), 1 / e)
spread = np.array([[(i * k + j) % e for j in range(k)] for i in range(t)])
lumped = np.tile([0, 1], (t, 1))
lumped_p = np.zeros((t, e)); lumped_p[:, :2] = 0.5
print(f"{'':28s} {'hf (Σf = k)':>12s} {'megatron (Σf = 1)':>18s}")
for name, p, idx in (("uniform", uniform_p, spread), ("two experts take everything", lumped_p, lumped)):
    print(f"{name:28s} {R.switch_aux_loss(p, idx, 'hf'):12.2f} {R.switch_aux_loss(p, idx, 'megatron'):18.2f}")
print(f"z-loss of all-zero logits over 8 experts: (ln 8)^2 = {R.z_loss(np.zeros((4, 8))):.3f}; "
      f"the same logits + 20: {R.z_loss(np.full((4, 8), 20.0)):.0f} (softmax unchanged, z-loss not)")

# %% [markdown]
# ## Worked example 2 — capacity, dropping, and dropless padding
# With a capacity factor, expert $e$ takes at most `int(factor · k · T / E)` rows. The layer **drops** the other rows.
# Those tokens skip the expert and continue on the residual connection.
#
# Dropless routing keeps every assignment. Instead, it pads the rows of each expert to the block size of the kernel.
# The `moe_align_block_size` of vLLM sorts the $T \cdot k$ slots by expert. It pads each segment to a multiple of
# `BLOCK_SIZE_M` with a pad id ($T \cdot k$), so no block GEMM mixes experts.

# %%
logits = np.random.default_rng(1).standard_normal((256, 8)) + np.array([1.5, 0.8, 0, 0, 0, 0, 0, 0])  # 0, 1 popular
rr = route(logits, 2, **ROUTERS["mixtral"])
print("assignments per expert:", R.load(rr.idx, 8).tolist())
for factor in (1.0, 1.25, 2.0):
    cap = R.capacity(256, 8, 2, factor)
    kept = R.apply_capacity(rr.idx, rr.weights, cap)
    print(f"capacity factor {factor:4.2f}: {cap:3d} rows/expert, dropped {1 - kept.mean():6.1%} of assignments")
ids, blocks, padded = R.align_block_size(rr.idx, 16, 8)
print(f"dropless: {rr.idx.size} assignments padded to {padded} rows in {len(blocks)} blocks of 16 "
      f"({padded / rr.idx.size - 1:.1%} padding)")

# %% [markdown]
# ## Worked example 3 — collapse, and two ways out
# `moecore.train` trains a small MoE with gradients written by hand. It has these parts:
# - four clusters of tokens, each with its own target map
# - four linear experts
# - a linear top-1 router with Switch-style weights
#
# The hidden states share a large common direction (as transformer hidden states do). Thus at step 0, two experts are
# at the top for most tokens. Look at the share of tokens per expert.

# %%
task = T.make_task(seed=6)
runs = {b: T.train(task, balance=b) for b in ("none", "aux", "bias")}
for b, h in runs.items():
    print(f"\n{b}: task loss {h.loss[0]:.3f} -> {h.loss[-1]:.3f}")
    for step in (0, 25, 50, 100, 299):
        print(f"  step {step:3d}: share per expert {np.round(h.share[step], 2).tolist()}")
print("\nwhere each cluster's tokens went at the end (rows: clusters, cols: experts):")
for b in ("none", "aux", "bias"):
    print(f"{b:5s}", runs[b].placement.tolist())

# %% [markdown]
# With no balance method, the router starts at 55/45 on two experts. At the end, one expert takes 95%, and the other
# three never learn anything. With the aux loss ($\alpha = 0.1$) or the DeepSeek-style bias, all four experts hold
# approximately a quarter. Also, the task loss is three to five times lower.
#
# Look at the placements. The bias run gives each cluster its own expert, almost cleanly. The aux run also balances.
# But its gradient demands equal counts, and the clusters are not equal (124, 117, 135, 136 tokens). Thus it
# divides two clusters across experts. This is the quality cost of a balance that comes through the loss.
#
# This is a toy, so the numbers depend on the seed, but the direction does not. The pattern holds across ten task
# seeds:

# %%
print(f"{'seed':>4s} | {'none: experts used, loss':>26s} | {'aux':>18s} | {'bias':>18s}")
for seed in range(10):
    tk = T.make_task(seed=seed)
    cells = []
    for b in ("none", "aux", "bias"):
        h = T.train(tk, balance=b)
        cells.append(f"{(h.final_share > 0.05).sum()} used, {h.loss[-1]:.3f}")
    print(f"{seed:4d} | {cells[0]:>26s} | {cells[1]:>18s} | {cells[2]:>18s}")

# %% [markdown]
# ## Exercise 2.1 — the Switch loss
# Write `switch_loss(probs, idx)` in the convention of transformers: $E \cdot \sum_e f_e \cdot P_e$. Here, $f_e$ =
# assignments to $e$ divided by $T$ (so $\sum f = k$), and $P_e$ = mean of `probs[:, e]`.

# %% exercise
def switch_loss(probs, idx):
    ### BEGIN SOLUTION
    t, e = probs.shape
    f = np.bincount(idx.ravel(), minlength=e) / t
    return e * np.sum(f * probs.mean(axis=0))
    ### END SOLUTION

# %% check
for _ in range(5):
    p = softmax(rng.standard_normal((40, 16)))
    idx = np.argsort(-p, axis=1)[:, :4]
    assert np.isclose(switch_loss(p, idx), R.switch_aux_loss(p, idx, "hf"))
assert np.isclose(switch_loss(uniform_p, spread), 2.0)
print("✅ E·Σ f·P: k when uniform (hf), E when one expert takes all; divide by k for Megatron's convention")

# %% [markdown]
# ## Exercise 2.2 — what does a capacity factor drop?
# Use the skewed batch `rr` from worked example 2 (256 tokens, 8 experts, top-2). Set `drop_100` and `drop_125` to the
# fraction of assignments dropped at capacity factor 1.0 and 1.25 (position policy). Set `factor_for_zero` to the
# smallest factor, in steps of 0.25, that drops nothing. Use `R.capacity` and `R.apply_capacity`.

# %% exercise
### BEGIN SOLUTION
def dropped(factor):
    return 1 - R.apply_capacity(rr.idx, rr.weights, R.capacity(256, 8, 2, factor)).mean()
drop_100, drop_125 = dropped(1.0), dropped(1.25)
factor_for_zero = next(f for f in np.arange(1.0, 8.01, 0.25) if dropped(f) == 0)
### END SOLUTION

# %% check
loads = R.load(rr.idx, 8)
assert np.isclose(drop_100, np.maximum(loads - 64, 0).sum() / 512)
assert drop_125 < drop_100 and dropped(factor_for_zero) == 0 and dropped(factor_for_zero - 0.25) > 0
assert R.capacity(256, 8, 2, factor_for_zero) >= loads.max()
print(f"✅ factor 1.0 drops {drop_100:.1%}, 1.25 drops {drop_125:.1%}; nothing is dropped until {factor_for_zero:.2f}, "
      f"the first step above the hottest expert's load over the mean ({loads.max()} / {loads.mean():.0f} = "
      f"{loads.max() / loads.mean():.2f}). Inference engines are dropless: a dropped token changes the answer")

# %% [markdown]
# ## Exercise 2.3 — balance with a bias alone
# Freeze a skewed router (`scores` in the next cell: experts 0 and 1 favoured). Balance it with the rule of DeepSeek.
# Select with `scores + bias`. After each batch, move the bias of every expert by `rate` toward the mean load:
# `bias += rate · sign(total − load · E)`. Write `balance_by_bias(scores, k, rate, steps)`. It returns the final
# bias and the load of the last step.
#
# When the router calculates weights, they still come from `scores`. The bias only selects.

# %%
scores = softmax(np.random.default_rng(2).standard_normal((512, 8)) + np.array([2.0, 1.0, 0, 0, 0, 0, 0, 0]))
print("load with no bias:", R.load(np.argsort(-scores, axis=1)[:, :2], 8).tolist())

# %% exercise
def balance_by_bias(scores, k, rate, steps):
    ### BEGIN SOLUTION
    bias = np.zeros(scores.shape[1])
    for _ in range(steps):
        idx = np.argsort(-(scores + bias), axis=1)[:, :k]
        counts = np.bincount(idx.ravel(), minlength=scores.shape[1])
        bias = bias + rate * np.sign(counts.sum() - counts * scores.shape[1])
    return bias, counts
    ### END SOLUTION

# %% check
bias, counts = balance_by_bias(scores, 2, 0.002, 400)
ref = np.zeros(8)
for _ in range(400):
    c = R.load(np.argsort(-(scores + ref), axis=1)[:, :2], 8)
    ref = R.update_bias(ref, c, 0.002)
assert np.allclose(bias, ref)
assert R.stats(counts)["max_over_mean"] < 1.2 and bias[0] < 0 < bias[-1]
print(f"✅ load after 400 bias steps: {counts.tolist()} (max/mean {R.stats(counts)['max_over_mean']:.2f}); "
      f"bias {np.round(bias, 3).tolist()} - no gradient touched the router")

# %% [markdown]
# ## Exercise 2.4 — how strong must the aux loss be?
# On task seed 4, train with the aux loss at $\alpha = 0.01$ and $\alpha = 0.1$. Before you run it, predict which one
# still collapses. Set `collapsed` to the $\alpha$ that collapses. Set `balanced` to the $\alpha$ that does not.

# %% exercise
### BEGIN SOLUTION
tk4 = T.make_task(seed=4)
res = {a: T.train(tk4, balance="aux", alpha=a).final_share.max() for a in (0.01, 0.1)}
collapsed, balanced = 0.01, 0.1
### END SOLUTION

# %% check
tk4 = T.make_task(seed=4)
assert T.train(tk4, balance="aux", alpha=collapsed).final_share.max() > 0.9
assert T.train(tk4, balance="aux", alpha=balanced).final_share.max() < 0.35
print("✅ α = 0.01 is too weak to pull tokens off the winning expert here; 0.1 spreads them. Too strong trades "
      "quality for balance (it forces equal counts even when the data is not balanced) - DeepSeek-V3's reason to move "
      "balancing out of the loss and into a bias")

# %% [markdown]
# ## Exercise 2.5 — balanced on average, collapsed per sequence
# Make `probs` and `idx` for 4 sequences of 16 tokens and 4 experts, top-1. In them, **sequence $i$ sends every token
# to expert $i$** with probability 1. Then set `batch_loss` (Megatron convention over the whole batch) and `seq_loss`
# (`R.sequence_aux_loss`).

# %% exercise
### BEGIN SOLUTION
idx5 = np.repeat(np.arange(4), 16)[:, None]
probs5 = np.eye(4)[idx5[:, 0]]
batch_loss = R.switch_aux_loss(probs5, idx5, "megatron")
seq_loss = R.sequence_aux_loss(probs5, idx5, 16)
### END SOLUTION

# %% check
assert np.isclose(batch_loss, 1.0) and np.isclose(seq_loss, 4.0)
print("✅ the batch-level loss says perfect (1.0), the sequence-level one says collapsed (4.0 = E). DeepSeek-V3 keeps "
      "a small sequence-wise term for this; at inference, a long sequence that hammers one expert is a hot expert")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A learned router is a feedback loop. The expert that wins tokens early trains on them
# and continues to win. Without a counter-force, the layer collapses onto a few experts, and the other experts are dead
# HBM.
#
# "Training adds a counter-force: the Switch auxiliary loss $E \cdot \sum f \cdot P$. Its minimum is $k$ in the
# convention of transformers and 1 in the convention of Megatron. Thus, find which convention applies before you
# compare coefficients. Training also adds a small z-loss for bf16 stability.
#
# "Capacity factors put a limit on the work of each expert, because they drop the overflow tokens. A drop of
# tokens is satisfactory for training throughput, but not acceptable at inference. Thus engines are dropless and
# pad the rows instead.
#
# "DeepSeek-V3 moved the balance out of the loss. A per-expert bias changes in steps of a constant rate against the
# load. It changes which experts the router selects, but never the weights, so the task gradient stays clean. Balance
# at training time holds on average. At inference, a skewed workload still makes hot experts, and that is an
# expert-parallel problem."
#
# **Drill questions**
# 1. *A paper says 'aux loss coefficient 0.01'. The loss of your framework reads 2.0 on a uniform router. Why?* The
#    convention of transformers counts all $k$ assignments ($\sum f = k$). The convention of Megatron divides by $k$.
#    The coefficient is the same, but the gradient is $k$ times different.
# 2. *Why can DeepSeek not also add its bias to the combine weights?* A bias in the combine weights changes the output
#    of every token, and the task gradient fights it. As a selection-only term, the bias controls the load and does
#    not put a bias into the outputs.
# 3. *Expert-choice routing is perfectly balanced. Why do decode engines not use it?* Each expert selects from the
#    whole batch. Thus the experts of a token depend on the other tokens in the step. The token can then get
#    zero experts. That is not causal and not deterministic per request.
