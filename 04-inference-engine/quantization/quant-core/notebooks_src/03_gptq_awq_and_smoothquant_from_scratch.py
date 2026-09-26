# %% [markdown]
# # 03 · GPTQ, AWQ and SmoothQuant from scratch
#
# **Tier:** T0 — numpy, under a minute on a laptop. The model is `quantcore.TinyModel`: a residual MLP stack
# whose RMSNorm gains make four activation channels 25–40× larger than the rest (like the "massive
# activations" of real LLMs), trained in closed form on a synthetic 16-class task. No download. Running the same
# algorithms on a real 0.5B checkpoint with llm-compressor is the lab's notebook 01 (T1).
#
# ## The one-minute version
# Round-to-nearest (RTN) treats every weight alone. Calibration methods use a few hundred sample inputs to
# decide *which* errors matter, and still produce an ordinary quantized checkpoint:
# - **GPTQ** minimises the layer's output error `‖XWᵀ − XQᵀ‖²`. It rounds one input column at a time and
#   pushes each column's rounding error onto the columns not yet rounded, weighted by the inverse Hessian
#   `H = 2/n XᵀX` — the Optimal Brain Surgeon update. It wins where inputs are **correlated**.
# - **AWQ** scales up the weight columns that meet **large activations** before rounding (and divides the
#   activations, folding 1/s into the previous norm), with `s = mean|x|^α` and α found by a 20-point search.
#   It wins where a few input channels dominate.
# - **SmoothQuant** is for W8A8: it moves activation outliers into the weights, `X Wᵀ = (X/s)(W s)ᵀ` with
#   `s = max|X|^α / max|W|^(1−α)`, so a per-token INT8 activation scale is no longer set by one channel.
# All three are exact reparameterisations or better-chosen codes: the served format and kernels do not change.
# After this notebook you can implement each, say which layer each helps, and say what calibration data is for.
#
# Primer: `../PRIMER.md` §4 *Weight-only post-training quantization* and §5 *Weight-and-activation quantization*.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import TinyModel, awq, eval as E, granularity as G, gptq, quantize_model, smoothquant as S

m = TinyModel()
X, y = m.sample(4000, "test")                 # held-out evaluation set
Xc, _ = m.sample(256, "calib")                # calibration set: different points, same distribution
ref = m.forward(X)
cap = m.calibration_inputs(Xc)                # each linear's input activations over the calibration set
print(f"full-precision accuracy {np.mean(ref.argmax(1) == y):.1%} on {len(y)} held-out points; linears: {m.linears()}")

# %% [markdown]
# ## Worked example 1 — RTN, the baseline that needs no data

# %%
def report(label, model):
    r = E.compare(ref, model.forward(X), y)
    print(f"{label:24} acc {r['acc']:.1%} (±{r['stderr']:.1%})  KL {r['kl']:.4f}  top-1 agreement {r['top1']:.1%}  "
          f"flips: {r['lost']} lost, {r['gained']} gained")
    return r


for bits in (8, 4, 3):
    report(f"RTN INT{bits} g32", quantize_model(m, "rtn", bits, 32))

# %% [markdown]
# INT8 is free; INT4 costs 6.6 points and INT3 22 on this small model (larger models tolerate RTN better — the
# GPTQ paper's LLaMA-7B loses 0.6 perplexity at 4-bit RTN and 20 at 3-bit, verify). Note the flips: accuracy is a
# net of right answers lost and wrong answers gained, so it understates the churn — KL and top-1 agreement see it.
#
# ## Worked example 2 — GPTQ, one layer at a time
# `H = 2/n XᵀX` is the curvature of the output error. Its spectrum says how correlated a layer's inputs are.

# %%
capt = m.calibration_inputs(X[:1000])          # the same layers' inputs on held-out data
for name in m.linears():
    Xn, W = cap[name], m.weights[name]
    ev = np.sort(np.linalg.eigvalsh(gptq.hessian(Xn)))[::-1]
    k99 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99)) + 1
    r, g = gptq.rtn(W, 4, 32).w_hat, gptq.gptq(W, Xn, bits=4, group_size=32).w_hat
    print(f"{name:14} inputs {W.shape[1]:3d}-dim, 99% of H's trace in {k99:2d} directions | output error RTN "
          f"{G.output_error(Xn, W, r):.4f} -> GPTQ {G.output_error(Xn, W, g):.4f} (held-out {G.output_error(capt[name], W, r):.4f} "
          f"-> {G.output_error(capt[name], W, g):.4f})")

# %% [markdown]
# The down-projections read a ReLU of a 64-dim stream spread over 256 dims, so their inputs live in 13–37
# directions: rounding error along the other directions never reaches the output, and GPTQ steers it there —
# 5–8× less output error, on held-out data too. The up-projections' inputs are dominated by the four outlier
# channels, and GPTQ gains less there: 40% on the second block, almost nothing held-out on the first. Real LLM
# layers are less redundant than this toy's, so expect a smaller but real gain.

# %%
for bits in (4, 3):
    report(f"GPTQ INT{bits} g32", quantize_model(m, "gptq", bits, 32, calib=Xc))
W, Xn = m.weights["blocks.0.down"], cap["blocks.0.down"]
print(f"act-order on blocks.0.down: output error {G.output_error(Xn, W, gptq.gptq(W, Xn, bits=4, group_size=32).w_hat):.4f} "
      f"-> {G.output_error(Xn, W, gptq.gptq(W, Xn, bits=4, group_size=32, actorder=True).w_hat):.4f}")

# %% [markdown]
# GPTQ recovers INT4 to within a point and INT3 to 6 points. **Act-order** (visit the most-used inputs
# first, while the most columns remain to absorb their error) helps a little more; with static groups the
# checkpoint layout is unchanged.
#
# ## Worked example 3 — AWQ: a scale search, then RTN
# The loss curve of the search on the first up-projection, relative to α = 0 (which is RTN):

# %%
Wu, Xu = m.weights["blocks.0.up"], cap["blocks.0.up"]
s, alpha, losses = awq.search_scale(Wu, Xu, bits=4, group_size=32, symmetric=True)
print("alpha: " + " ".join(f"{k / 20:.2f}" for k in range(0, 20, 2)))
print("loss:  " + " ".join(f"{v:.2f}" for v in (losses / losses[0])[::2]))
print(f"best alpha {alpha:.2f}; the largest scale goes to channel {int(np.argmax(s))} "
      f"(mean |x| {np.abs(Xu).mean(0)[np.argmax(s)]:.1f} vs median {np.median(np.abs(Xu).mean(0)):.2f})")
for name in m.linears():
    W, Xn = m.weights[name], cap[name]
    q, s_ = awq.awq(W, Xn, bits=4, group_size=32, symmetric=True)
    print(f"{name:14} output error RTN {G.output_error(Xn, W, gptq.rtn(W, 4, 32).w_hat):.4f} -> AWQ "
          f"{G.output_error(Xn / s_, W * s_, q.w_hat):.4f}")

# %% [markdown]
# A U-shaped curve: too little scaling leaves the salient channels coarse, too much makes them set the group
# scale for everyone. AWQ cuts the up-projections' error by ~25% and does little or nothing for the
# down-projections, whose inputs have no dominant channels. On the model:

# %%
ups = ["blocks.0.up", "blocks.1.up"]
for bits in (4, 3):
    report(f"AWQ INT{bits} g32", quantize_model(m, "awq", bits, 32, calib=Xc))
for bits in (4, 3):
    report(f"AWQ then GPTQ INT{bits} g32", quantize_model(m, "awq+gptq", bits, 32, calib=Xc))
print("up-projections only, INT3 g32:")
for method in ("rtn", "awq", "gptq"):
    report(f"  {method}", quantize_model(m, method, 3, 32, calib=Xc, targets=ups))

# %% [markdown]
# On the layers with outlier channels AWQ beats GPTQ; on the model as a whole GPTQ's down-projection gains
# dominate; and the two compose — AWQ's scales first, then GPTQ's rounding (llm-compressor's `AWQModifier`
# followed by a `GPTQModifier`) — for the lowest KL of all (accuracy moves within its error bars at INT4 and
# gains a point at INT3).
#
# ## Worked example 4 — SmoothQuant for W8A8
# W8A8 quantizes the up-projection's input per token; its four outlier channels set every token's scale. Sweep
# α and fold the result into the RMSNorm gain.

# %%
print("output error by alpha:", {a: round(v, 4) for a, v in S.alpha_sweep(Xu, Wu).items()}, "| no smoothing:",
      round(S.w8a8_error(Xu, Wu), 4))


def w8a8_logits(model, fmt="int8"):
    """Every hidden linear in W8A8: per-channel weights, dynamic per-token activations; the head in 16-bit."""
    wq = {n: G.fake_quant(model.weights[n], fmt=fmt, granularity="channel") for n in model.linears()}
    aq = lambda name, x: x if name == "head" else G.quantize_activations(x, fmt, "token")
    return model.forward(X, act_quant=aq, weights=wq)


smoothed = m
for name in ups:
    A = smoothed.calibration_inputs(Xc)[name]
    s_ = S.smooth_scales(np.abs(A).max(0), np.abs(smoothed.weights[name]).max(0), 0.5)
    smoothed = smoothed.with_weights(smoothed.fold(name, s_))
print(f"folding is exact: max logit change {np.abs(smoothed.forward(X) - ref).max():.1e}")
for label, logits in (("INT8 W8A8", w8a8_logits(m)), ("INT8 W8A8 + SmoothQuant 0.5", w8a8_logits(smoothed)),
                      ("FP8 W8A8 (no smoothing)", w8a8_logits(m, "fp8"))):
    r = E.compare(ref, logits, y)
    print(f"{label:28} KL {r['kl']:.5f}  top-1 {r['top1']:.1%}  acc {r['acc']:.1%}")

# %% [markdown]
# Smoothing cuts the up-projection's output error 2.7× and the model's KL 2.7×, for free at run time. Accuracy
# barely moves either way — W8A8 is gentle on this model — which is exactly why the eval has to look at KL too
# (notebook 04 and primer §8). FP8 without smoothing keeps accuracy but has 6× the INT8 KL here: its per-token
# scale handles the outliers, but 3 mantissa bits are coarser than INT8's 7 for well-scaled values.
#
# ## Worked example 5 — what calibration data does, and does not do

# %%
for n in (16, 64, 256, 1024):
    Xn, _ = m.sample(n, "calib")
    r = E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=Xn).forward(X), y)
    print(f"GPTQ INT3 g32, {n:4d} calibration samples: acc {r['acc']:.1%}, KL {r['kl']:.3f}")
Xa, ya = m.sample(4096, "calib")
for label, C in (("only 2 of the 16 classes", Xa[ya < 2][:256]),
                 ("pure noise", np.random.default_rng(1).standard_normal((256, 64)) * 1.6)):
    r = E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=C).forward(X), y)
    print(f"GPTQ INT3 g32, 256 samples of {label:25}: acc {r['acc']:.1%}, KL {r['kl']:.3f}")
draws = m.sample(768, "calib")[0].reshape(3, 256, -1)      # three more 256-sample draws, disjoint
accs = [E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=D).forward(X), y)["acc"] for D in draws]
print("GPTQ INT3 g32, three other 256-sample draws: acc " + ", ".join(f"{a:.1%}" for a in accs))

# %% [markdown]
# More samples help until H is well estimated (a few hundred here; recipes use 128–512 sequences of 512–2,048
# tokens). Past that, *which* samples you drew matters about as much as how many: three other 256-sample draws
# spread over about a point, more than the 0.6 points between 256 and 1,024 samples. A narrow or even random calibration set does surprisingly well on this toy, because what GPTQ and AWQ
# need — which channels are large, how inputs correlate — comes mostly from the model's own weights and norm gains,
# which any input reveals. Calibration data does not teach the model anything and cannot fix a format that is too
# coarse. On real models the text still matters (chat templates, languages, long contexts shift activation
# statistics) — calibrate on data that looks like your traffic, and measure (§8).
#
# ## Exercise 3.1 — the GPTQ loop
# Write `gptq_per_channel(W, X, bits=4)`: symmetric full-convention INT`bits` with one scale per row computed
# from `W` up front (`amax / ((2^bits − 1) / 2)`), `H = 2/n XᵀX`, dampen the diagonal by 1% of its mean, take
# `U = cholesky(inv(H)).T` (upper, `H⁻¹ = UᵀU`), then for each column j: quantize it, compute
# `err = (w_j − q_j) / U[j, j]` and subtract `outer(err, U[j, j+1:])` from the columns after it. Return the
# dequantized weight.

# %% exercise
def gptq_per_channel(W, X, bits=4):
    ### BEGIN SOLUTION
    W = W.astype(float).copy()
    H = 2 / len(X) * X.T @ X
    H[np.diag_indices_from(H)] += 0.01 * np.mean(np.diag(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    scale = np.abs(W).max(1) / ((2 ** bits - 1) / 2)
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        q = np.clip(np.round(W[:, j] / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * scale
        err = (W[:, j] - q) / U[j, j]
        W[:, j + 1:] -= np.outer(err, U[j, j + 1:])
        Q[:, j] = q
    return Q
    ### END SOLUTION

# %% check
Wd, Xd = m.weights["blocks.1.down"], cap["blocks.1.down"]
mine = gptq_per_channel(Wd, Xd)
assert np.allclose(mine, gptq.gptq(Wd, Xd, bits=4, group_size=None).w_hat, atol=1e-9)
print(f"✅ your GPTQ matches quantcore's: output error {G.output_error(Xd, Wd, mine):.4f} vs RTN "
      f"{G.output_error(Xd, Wd, gptq.rtn(Wd, 4, None).w_hat):.4f} (per-channel INT4)")

# %% [markdown]
# ## Exercise 3.2 — the AWQ search
# Write `awq_alpha(W, X, bits=4, g=32, n_grid=20)`: for α = k/n_grid, `s = mean|x|^α` (per input channel),
# normalised by `sqrt(s.max() · s.min())`; quantize `W · s` with `gptq.rtn(·, bits, g)`; the loss is the mean
# squared difference between `(X / s) Q(W s)ᵀ` and `X Wᵀ`. Return the best α.

# %% exercise
def awq_alpha(W, X, bits=4, g=32, n_grid=20):
    ### BEGIN SOLUTION
    x_mean, target = np.abs(X).mean(0), X @ W.T
    best = (np.inf, None)
    for k in range(n_grid):
        s = np.maximum(x_mean ** (k / n_grid), 1e-4)
        s = s / np.sqrt(s.max() * s.min())
        loss = np.mean(((X / s) @ gptq.rtn(W * s, bits, g).w_hat.T - target) ** 2)
        best = min(best, (loss, k / n_grid))
    return best[1]
    ### END SOLUTION

# %% check
for name in ups:
    ref_alpha = awq.search_scale(m.weights[name], cap[name], 4, 32, symmetric=True)[1]
    assert awq_alpha(m.weights[name], cap[name]) == ref_alpha, name
print(f"✅ same alpha as quantcore on both up-projections ({ref_alpha:.2f} on the second)")

# %% [markdown]
# ## Exercise 3.3 — smooth and fold
# SmoothQuant the **second** block's up-projection at α = 0.85 on the calibration inputs, and fold the scale into
# the model with `m.fold(...)` (it divides the RMSNorm gain by `s` and multiplies the weight's columns by `s`).
# Set `sm` to the new model.

# %% exercise
### BEGIN SOLUTION
A1 = cap["blocks.1.up"]
s85 = S.smooth_scales(np.abs(A1).max(0), np.abs(m.weights["blocks.1.up"]).max(0), 0.85)
sm = m.with_weights(m.fold("blocks.1.up", s85))
### END SOLUTION

# %% check
assert np.allclose(sm.forward(X), ref, atol=1e-8)
A_s = sm.calibration_inputs(Xc)["blocks.1.up"]
assert np.abs(A_s).max(0).max() / np.median(np.abs(A_s).max(0)) < 0.3 * (np.abs(A1).max(0).max() / np.median(np.abs(A1).max(0)))
print(f"✅ same function; the up-projection's input max/median channel ratio falls from "
      f"{np.abs(A1).max(0).max() / np.median(np.abs(A1).max(0)):.0f}x to {np.abs(A_s).max(0).max() / np.median(np.abs(A_s).max(0)):.1f}x")

# %% [markdown]
# ## Exercise 3.4 — which layers does AWQ help?
# Without running `awq` in the answer cell, predict for each linear whether AWQ at INT4 g32 cuts its calibration
# output error by more than 10% versus RTN. Fill `helps` (name → bool) from what you know about each layer's
# inputs (worked examples 2 and 3), then let the check measure it.

# %% exercise
helps = {name: None for name in m.linears()}
### BEGIN SOLUTION
helps = {name: name.endswith(".up") for name in m.linears()}     # only the ups have dominant input channels
### END SOLUTION

# %% check
for name in m.linears():
    W, Xn = m.weights[name], cap[name]
    q, s_ = awq.awq(W, Xn, bits=4, group_size=32, symmetric=True)
    gain = 1 - G.output_error(Xn / s_, W * s_, q.w_hat) / G.output_error(Xn, W, gptq.rtn(W, 4, 32).w_hat)
    assert (gain > 0.10) == helps[name], (name, gain)
print("✅ AWQ helps exactly the layers whose inputs have outlier channels: it rescales, it does not decorrelate")

# %% [markdown]
# ## Exercise 3.5 — how much calibration data?
# "Enough" means: more would not make a difference this eval can see. Find the smallest calibration size among
# 32, 64, 128, 256, 512 (drawn with `m.sample(n, "calib")`) whose GPTQ INT3 g32 accuracy is below the 1,024-sample
# result by less than two standard errors of the *difference* between two such runs: `2 * E.diff_stderr(p, p, n)`
# with `p` the 1,024-sample accuracy and `n` the number of test points. Set `n_enough`.

# %% exercise
### BEGIN SOLUTION
acc = lambda n: E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=m.sample(n, "calib")[0]).forward(X), y)["acc"]
p = acc(1024)
n_enough = next(n for n in (32, 64, 128, 256, 512) if acc(n) > p - 2 * E.diff_stderr(p, p, len(y)))
### END SOLUTION

# %% check
res = {n: E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=m.sample(n, "calib")[0]).forward(X), y)["acc"]
       for n in (32, 64, 128, 256, 512, 1024)}
bar = 2 * E.diff_stderr(res[1024], res[1024], len(y))
assert n_enough == next(n for n in (32, 64, 128, 256, 512) if res[n] > res[1024] - bar) and n_enough in (128, 256), n_enough
print(f"✅ {n_enough} samples ({res[n_enough]:.1%} vs {res[1024]:.1%} at 1,024, noise bar {bar:.1%}): once H for a 256-dim "
      "input is estimated from a few hundred rows, more data moves accuracy less than the eval can see (recipes use "
      "128-512 sequences of 512-2,048 tokens, i.e. far more token rows than this)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We ship INT4 weights with GPTQ or AWQ, never plain round-to-nearest. Both use a
# few hundred calibration samples and produce the same checkpoint format and kernels as RTN; they only choose
# better codes. GPTQ minimises each layer's output error: it rounds column by column and lets the unrounded
# columns compensate through the inverse Hessian of the calibration inputs, so it is strongest where inputs are
# correlated. AWQ protects the weights that meet large activations by scaling them up before rounding and folding
# the inverse into the previous norm; it is strongest where a few channels dominate. They compose — AWQ scales,
# then GPTQ rounding — and on our toy that gives the lowest KL. For W8A8 INT8 we add SmoothQuant, which moves
# activation outliers into the weights at no run-time cost. Calibration data should look like traffic, but it
# does not teach the model anything: if a format is too coarse no calibration saves it."
#
# **Drills**
# 1. *What does GPTQ need from calibration data, and why a few hundred samples?* — Only H = 2/n XᵀX per layer;
#    it must be well conditioned in the input dimension, and damping (1% of the mean diagonal) covers the rest.
# 2. *AWQ multiplies weight columns by s. Why is the model unchanged?* — The activations are divided by s, and
#    that division is folded into the preceding RMSNorm gain (or the previous linear's rows): (X/s)(Ws)ᵀ = XWᵀ.
# 3. *SmoothQuant α = 0 or 1?* — Neither: 0 leaves the activation outliers, 1 moves them all into the weights;
#    the output error is lowest in between (0.5 here; 0.8–0.9 tuned for Llama-class models).
