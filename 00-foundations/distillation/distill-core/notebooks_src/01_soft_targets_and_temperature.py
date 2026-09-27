# %% [markdown]
# # 01 · Soft targets and temperature
#
# **Tier:** T0 — CPU only, numpy, no network, about a minute. The models are tiny next-token networks with
# manual gradients (`distillcore.tinylm`), trained in seconds on `ModLang`, a toy language whose true
# distribution is known exactly — so "how much did the student learn?" has an exact answer. The same losses on a
# real transformer are `distill-lab` notebook `01_kd_on_a_tiny_transformer`.
#
# ## The one-minute version
# A teacher's output is a whole distribution, not just its top token. Hinton's **soft targets**
# $p_T = \operatorname{softmax}(z / T)$ expose how the teacher ranks the wrong answers — the **dark knowledge** — and
# a temperature $T$ > 1 turns the small logits up.
#
# The classic loss is $\alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}(y, q)$: its soft term's
# gradient on the student's logits is $T\,(q_T - p_T)$, and the $T^2$ keeps it from fading as $T$ grows; as
# $T \to \infty$ it becomes matching centred logits. A hard label is the same loss with a one-hot target, and sampling
# it adds $1 - \sum p^2$ of noise per example that the teacher's distribution does not — which is why a student learns
# more per example from soft targets than a same-size model trained from scratch on the same tokens.
#
# And there are three routes to a small model: train it small, prune a big one, or distil — pruning then distilling
# gets there in fewer steps. After this notebook you can compute every one of those numbers by hand. Primer:
# `../../PRIMER.md` §1–§2 (and §6 for pruning).

# %%
import numpy as np

from distillcore import ModLang, TinyLM, eval as E, losses as L, train
from distillcore.tinylm import fit_language

Z = np.array([[4.0, 3.0, 1.0, 0.0, -1.0]])      # a teacher's logits over five tokens
V = np.array([[3.0, 3.5, 0.0, 0.5, -1.0]])      # a student's


def show(name, p):
    print(f"{name:28s}", " ".join(f"{x:6.4f}" for x in np.ravel(p)))

# %% [markdown]
# ## Worked example 1 — soft targets on five tokens
# The teacher is sure token 0 is right, but it also says token 1 is a far better second choice than tokens 3
# or 4. That ranking is invisible in a hard label (`[1, 0, 0, 0, 0]`). Raising $T$ flattens the distribution and
# makes the ranking of the unlikely tokens carry weight.

# %%
for T in (1, 2, 4):
    show(f"teacher p_T, T = {T}", L.softmax(Z, T))
show("student q, T = 1", L.softmax(V))
print(f"KL(p ‖ q) at T = 1: {L.kl(L.softmax(Z), L.softmax(V))[0]:.5f} nats")

# %% [markdown]
# ## Worked example 2 — the loss, its gradient and the T² factor
# $\mathrm{KL}(p_T \,\|\, q_T)$ shrinks roughly as $1/T^2$ when $T$ grows, and so does its gradient $(q_T - p_T)/T$ —
# so without a correction, a soft term at $T$ = 4 is drowned by the hard-label term. Multiplying by $T^2$ restores the
# scale; the limit is logit matching,
#
# $$
# \frac{1}{2N}\,\lVert (v - \bar{v}) - (z - \bar{z}) \rVert^2.
# $$

# %%
print(f"{'T':>6} {'KL(p_T ‖ q_T)':>14} {'T²·KL':>8}")
for T in (1, 2, 4, 10, 100, 1000):
    print(f"{T:6d} {L.kd(V, Z, T, scale=False)[0]:14.5f} {L.kd(V, Z, T)[0]:8.5f}")
print(f"logit matching (T → ∞): {L.logit_mse(V, Z)[0]:.5f}")
print("gradient of KL at T = 2, (q − p)/T:", np.round(L.kd(V, Z, 2.0, scale=False)[1][0], 6))

# %% [markdown]
# ## Worked example 3 — the toy language, a teacher, and the capacity gap
# `ModLang`: after tokens ($a$, $b$) the next token is $(a + b) \bmod 11$ with probability 0.8 and each neighbour with
# 0.1. The rule is a 121-entry lookup table with no smooth structure, so width is capacity. The teacher (64 hidden
# units) learns it; narrower models cannot.

# %%
lang = ModLang(11, 0.2)
C = lang.contexts()
teacher = fit_language(lang, 64)
print(f"{'hidden units':>12} {'params':>7} {'KL(truth ‖ model)':>18} {'rule accuracy':>14}")
for H in (64, 16, 8, 4):
    m = teacher if H == 64 else fit_language(lang, H)
    r = E.vs_truth(m, lang)
    print(f"{H:12d} {m.n_params:7d} {r['kl']:18.4f} {r['rule_acc']:14.3f}")
print(f"label noise of one hard example, 1 − Σp² = {L.label_noise(lang.true_probs(C[:1]))[0]:.2f}")

# %% [markdown]
# Everything below uses a 16-unit student: big enough to hold the language, so what it learns depends only on
# the training signal. (Eight units is the capacity gap: 93.4% of contexts at best.)
#
# ## Worked example 4 — soft targets carry more per example
# Same student, same contexts, same number of steps. One trains on the sampled next token (hard labels — a
# same-size model trained from scratch on the same tokens); the other on the teacher's distribution at those
# contexts (KD at $T$ = 1). $N$ = 242 is two examples per context on average; 605 is five.

# %%
results = {}
for N in (242, 605):
    rng = np.random.default_rng(1)
    ctx = C[rng.integers(0, 121, N)]
    y, zt = lang.sample_next(ctx, rng), teacher.logits(ctx)
    hard = TinyLM(11, 16, 8, seed=1)
    train(hard, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
    soft = TinyLM(11, 16, 8, seed=1)
    train(soft, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 400)
    results[N] = (ctx, y, zt)
    for name, m in (("hard labels", hard), ("soft targets", soft)):
        r = E.vs_truth(m, lang)
        print(f"N = {N:3d}  {name:12s}  KL(truth ‖ student) {r['kl']:6.3f}   rule accuracy {r['rule_acc']:.3f}")

# %% [markdown]
# With two examples per context, soft targets give 87.6% rule accuracy against 67.8%; with five, 99.2%
# against 91.7%. The hard-label student is also badly calibrated: it never saw most neighbours, so it gives
# them almost no probability, and its KL to the truth is large. A soft target tells it about all eleven tokens
# at once. This is the whole case for distilling rather than training small from scratch on the same data.
#
# ## Worked example 5 — three routes to a small model
# Train small from scratch (hard labels), distil into a fresh small model, or **prune** the teacher to 16 hidden
# units by activation magnitude (Minitron's width pruning) and distil into what is left. One run proves little
# here, so compare over five draws of the 242 training contexts (draw 1 is worked example 4's), with four
# initialisations of the fresh student per draw.

# %%
def prune_vs_fresh(steps, draws=range(1, 6), inits=range(1, 5)):
    """Rule accuracy of the pruned teacher and of fresh 16-unit students after `steps` KD steps from the parent."""
    pruned, fresh = [], []
    for d in draws:
        rng = np.random.default_rng(d)
        c = C[rng.integers(0, 121, 242)]
        lang.sample_next(c, rng)
        zc = teacher.logits(c)
        for m, out in [(teacher.prune_width(c, 16), pruned)] + [(TinyLM(11, 16, 8, seed=s), fresh) for s in inits]:
            if steps:
                train(m, c, lambda z, i: L.kd(z, zc[i], 1.0), steps)
            out.append(E.vs_truth(m, lang)["rule_acc"])
    return np.array(pruned), np.array(fresh)


for steps in (0, 20, 100):
    pr, fr = prune_vs_fresh(steps)
    print(f"{steps:3d} KD steps: pruned {pr.mean():.3f} ({pr.min():.3f}–{pr.max():.3f})   "
          f"fresh {fr.mean():.3f} ({fr.min():.3f}–{fr.max():.3f})")

# %% [markdown]
# Pruning alone breaks the model (0.317 on average), but not back to nothing: a fresh student scores 0.097. KD
# from the parent repairs it fast. After 20 steps the worst pruned student (0.612) beats the best fresh one
# (0.331). By 100 steps the two are within seed noise, and a single draw could show either one ahead. (The
# numbers here are one CPU's: numpy's matrix kernel differs by microarchitecture, and a few hundred Adam steps
# turn last-bit differences into a slightly different model, so your run may differ in the last digits — the
# core's tests hold each number to the spread measured across kernels, `tests/test_primer_numbers.py`.) Pruning
# buys a head start in training steps, not a better student: Minitron's argument (prune by importance, then
# distil from the parent, with far fewer training tokens per model) in miniature.
#
# ## Exercise 1.1 — temperature
# Write `softmax_T(z, T)`: the softmax of $z\,/\,T$ along the last axis, numerically stable (subtract the max).

# %% exercise
def softmax_T(z, T):
    ### BEGIN SOLUTION
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)
    ### END SOLUTION

# %% check
rng = np.random.default_rng(0)
for T in (0.5, 1.0, 2.0, 7.0):
    zz = rng.normal(size=(4, 9)) * 3
    assert np.allclose(softmax_T(zz, T), L.softmax(zz, T))
assert np.allclose(softmax_T(np.array([1000.0, 999.0]), 1.0), [0.7310586, 0.2689414])
print("✅ p_T = softmax(z/T): T > 1 flattens, T < 1 sharpens, and a large logit does not overflow")

# %% [markdown]
# ## Exercise 1.2 — the gradient Hinton's loss hands the student
# Return the gradient of $T^2\,\mathrm{KL}(p_T \,\|\, q_T)$ with respect to the student's logits `v` (one row), where
# $p_T = \operatorname{softmax}(z/T)$ and $q_T = \operatorname{softmax}(v/T)$. Derive it; do not use finite
# differences.

# %% exercise
def kd_grad(v, z, T):
    ### BEGIN SOLUTION
    return T * (softmax_T(v, T) - softmax_T(z, T))
    ### END SOLUTION

# %% check
for T in (1.0, 2.0, 4.0):
    assert np.allclose(kd_grad(V, Z, T), L.kd(V, Z, T)[1])
eps, T = 1e-6, 3.0
num = np.array([(L.kd(V + eps * np.eye(5)[j], Z, T)[0] - L.kd(V - eps * np.eye(5)[j], Z, T)[0]) / (2 * eps) for j in range(5)])
assert np.allclose(kd_grad(V, Z, T)[0], num, atol=1e-7)
print("✅ ∂(T²·KL)/∂v = T·(q_T − p_T): without the T² it would be (q_T − p_T)/T and fade as T grows")

# %% [markdown]
# ## Exercise 1.3 — predict the high-temperature limit
# As $T \to \infty$, $T^2\,\mathrm{KL}(p_T \,\|\, q_T)$ tends to
#
# $$
# \frac{1}{2N} \sum_i \bigl((v_i - \bar{v}) - (z_i - \bar{z})\bigr)^2.
# $$
#
# Compute `limit` for the five-token example by hand (numpy arithmetic on `Z` and `V`, not a library loss).

# %% exercise
### BEGIN SOLUTION
cz, cv = Z[0] - Z[0].mean(), V[0] - V[0].mean()
limit = float(((cv - cz) ** 2).sum() / (2 * 5))
### END SOLUTION

# %% check
assert abs(limit - 0.23) < 1e-12 and abs(L.kd(V, Z, 1e5)[0] - limit) < 1e-5
print(f"✅ the limit is {limit:.2f}: at very high T distillation is logit matching on centred logits (Caruana's recipe)")

# %% [markdown]
# ## Exercise 1.4 — Hinton's full loss
# Write `hinton_loss(v, z, y, T, alpha)` returning `(loss, grad)` averaged over rows:
#
# $$
# \alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}(y, q_1).
# $$
#
# You may call `L.kd` and `L.hard_ce`.

# %% exercise
def hinton_loss(v, z, y, T, alpha):
    ### BEGIN SOLUTION
    ls, gs = L.kd(v, z, T)
    lh, gh = L.hard_ce(v, y)
    return alpha * ls + (1 - alpha) * lh, alpha * gs + (1 - alpha) * gh
    ### END SOLUTION

# %% check
ctx, y, zt = results[242]
vv = TinyLM(11, 16, 8, seed=5).logits(ctx)
for T, a in ((2.0, 0.5), (4.0, 0.9), (1.0, 0.0)):
    mine, ref = hinton_loss(vv, zt, y, T, a), L.hinton(vv, zt, y, T, a)
    assert abs(mine[0] - ref[0]) < 1e-12 and np.allclose(mine[1], ref[1])
print("✅ α mixes the teacher's soft targets with the hard labels; α = 0 is plain training from scratch")

# %% [markdown]
# ## Exercise 1.5 — Minitron's importance score
# Width pruning keeps the hidden units that matter on calibration data. Return the indices (sorted ascending) of the
# `keep` hidden units with the largest mean $\lvert \text{activation} \rvert$ over the contexts `ctx`. Use
# `model.forward(ctx)`, whose cache is `(contexts, x, h)`.

# %% exercise
def keep_units(model, ctx, keep):
    ### BEGIN SOLUTION
    _, (_, _, h) = model.forward(ctx)
    return np.sort(np.argsort(-np.abs(h).mean(0))[:keep])
    ### END SOLUTION

# %% check
idx = keep_units(teacher, C, 16)
assert len(idx) == 16 and np.all(np.diff(idx) > 0)
assert np.allclose(teacher.p["W2"][idx], teacher.prune_width(C, 16).p["W2"])
print("✅ keep the most active units, slice W1, b1 and W2 — a narrower model that starts from its parent's busiest units")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We distil rather than train the small model from scratch because the teacher's full
# distribution is worth far more per example than a sampled token: it ranks the wrong answers, and a sampled label
# adds $1 - \sum p^2$ of noise per example that the distribution does not. The loss is
# $\alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}$; the soft term's gradient on the student's
# logits is $T\,(q_T - p_T)$, the $T^2$ keeps its scale as $T$ grows, and at very high $T$ it is logit matching.
#
# "$T$ and $\alpha$ are knobs to sweep, not constants; TRL applies no $T^2$ (its GKD trainer runs the divergence at
# $T$ = 1). Width is capacity: below the size that can hold what the teacher does, no loss closes the gap. If we have
# the teacher's weights, we prune it to the student's shape by activation importance and distil into that — it gets
# there in fewer steps than a fresh network, not further."
#
# **Drill questions**
# 1. *Why multiply the soft term by $T^2$?* — Its gradient is $(q_T - p_T)/T$ and the difference itself shrinks
#    $\sim 1/T$, so the soft gradient falls $\sim 1/T^2$; $T^2$ restores it, keeping $\alpha$ meaningful across
#    temperatures.
# 2. *Same data budget: fine-tune the small model on the labels, or distil?* — Distil if a teacher exists: at two
#    examples per context the toy student reached 87.6% with soft targets and 67.8% with labels.
# 3. *What cannot soft targets fix?* — Capacity: an 8-unit student tops out at 93.4% of contexts even with the
#    exact distribution; and anything the teacher does not know.
