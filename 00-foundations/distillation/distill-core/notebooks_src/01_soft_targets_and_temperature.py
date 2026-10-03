# %% [markdown]
# # 01 · Soft targets and temperature
#
# **Tier:** T0. It uses only a CPU and numpy, it needs no network, and it takes about a minute. The models are small
# next-token networks with manual gradients (`distillcore.tinylm`). They train in seconds on `ModLang`, a toy language,
# and we know the true distribution of this language exactly. Thus the question "how much did the student learn?" has an
# exact answer. The same losses on a real transformer are in `distill-lab` notebook `01_kd_on_a_tiny_transformer`.
#
# ## The one-minute version
# The output of a teacher is a full distribution, not only its top token. Hinton's **soft targets**
# $p_T = \operatorname{softmax}(z / T)$ show how the teacher ranks the incorrect answers. This ranking is the
# **dark knowledge**. A temperature $T$ > 1 gives more weight to the small logits.
#
# The classic loss is $\alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}(y, q)$. The gradient of its
# soft term on the student's logits is $T\,(q_T - p_T)$. The $T^2$ prevents a decrease of this gradient as $T$
# increases. As $T \to \infty$, the soft term becomes a match of the centred logits.
#
# A hard label is the same loss with
# a one-hot target. A label that you sample adds $1 - \sum p^2$ of noise per example, and the teacher's distribution does
# not add this noise. That is why a student learns more per example from soft targets than a same-size model that
# trains from scratch on the same tokens.
#
# There are three routes to a small model: train it small, prune a large one, or distil. If you prune and then distil,
# you get there in fewer steps. After this notebook, you can calculate each of those numbers by hand. Primer:
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
# The teacher is sure that token 0 is correct. But it also says that token 1 is a much better second choice than tokens
# 3 or 4. A hard label (`[1, 0, 0, 0, 0]`) does not show that ranking. When you increase $T$, the distribution becomes
# flatter. Thus the ranking of the unlikely tokens gets weight.

# %%
for T in (1, 2, 4):
    show(f"teacher p_T, T = {T}", L.softmax(Z, T))
show("student q, T = 1", L.softmax(V))
print(f"KL(p ‖ q) at T = 1: {L.kl(L.softmax(Z), L.softmax(V))[0]:.5f} nats")

# %% [markdown]
# ## Worked example 2 — the loss, its gradient and the T² factor
# When $T$ increases, $\mathrm{KL}(p_T \,\|\, q_T)$ decreases approximately as $1/T^2$. Its gradient $(q_T - p_T)/T$
# also decreases approximately as $1/T^2$. Thus, without a correction, the hard-label term is much larger than a soft
# term at $T$ = 4, and the soft term has almost no effect. A multiplication by $T^2$ puts the scale back. The limit is
# logit matching,
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
# `ModLang`: after the tokens ($a$, $b$), the next token is $(a + b) \bmod 11$ with probability 0.8, and each neighbour
# has the probability 0.1. The rule is a lookup table with 121 entries and no smooth structure. Thus width is capacity.
# The teacher (64 hidden units) learns the rule. Narrower models cannot learn it.

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
# All the worked examples after this cell use a 16-unit student. This student is sufficiently large to hold the
# language. Thus what it learns depends only on the training signal. An 8-unit student shows the capacity gap: it
# is correct on 93.4% of contexts at best.
#
# ## Worked example 4 — soft targets carry more per example
# The two runs use the same student, the same contexts and the same number of steps. One student trains on the sampled
# next token. These are hard labels: the student is a same-size model that trains from scratch on the same tokens. The
# other student trains on the teacher's distribution at those contexts (KD at $T$ = 1). $N$ = 242 is two examples per
# context on average. $N$ = 605 is five examples per context on average.

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
# With two examples per context, soft targets give 87.6% rule accuracy against 67.8%. With five, they give 99.2%
# against 91.7%. The hard-label student also has a bad calibration. It never saw most neighbours, so it gives them
# almost no probability, and its KL to the truth is large. A soft target tells it about all eleven tokens at the same
# time. This is the full case for distillation, instead of a small model that trains from scratch on the same data.
#
# ## Worked example 5 — three routes to a small model
# Compare three routes to a small model:
#
# - train it small from scratch (hard labels),
# - distil into a new small model,
# - **prune** the teacher to 16 hidden units by activation magnitude (Minitron's width pruning), and distil into the
#   model that stays.
#
# One run does not prove much here. Thus compare the routes over five draws of the 242 training contexts. Draw 1 is the draw
# of worked example 4. Each draw has four initialisations of the new student.

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
# Pruning alone damages the model (0.317 on average), but the pruned model is better than nothing: a new student
# scores 0.097. KD from the parent repairs the pruned model fast. After 20 steps, the worst pruned student (0.612) is
# better than the best new student (0.331). At 100 steps, the difference between the two is within seed noise, and one
# draw can show either one ahead.
#
# The numbers here come from one CPU. The matrix kernel of numpy changes with the microarchitecture. A few hundred
# Adam steps change last-bit differences into a slightly different model. Thus your run can differ in the last digits.
# The core's tests hold each number to the spread that we measured across kernels, `tests/test_primer_numbers.py`.
#
# Pruning gives a head start in training steps, not a better student. This is Minitron's argument at a small scale:
# prune by importance, then distil from the parent, with many fewer training tokens per model.
#
# ## Exercise 1.1 — temperature
# Write `softmax_T(z, T)`. Return the softmax of $z\,/\,T$ along the last axis. Make it numerically stable:
# subtract the max.

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
# Return the gradient of $T^2\,\mathrm{KL}(p_T \,\|\, q_T)$ with respect to the student's logits `v` (one row). Here
# $p_T = \operatorname{softmax}(z/T)$ and $q_T = \operatorname{softmax}(v/T)$. Derive the gradient. Do not use finite
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
# As $T \to \infty$, $T^2\,\mathrm{KL}(p_T \,\|\, q_T)$ goes to
#
# $$
# \frac{1}{2N} \sum_i \bigl((v_i - \bar{v}) - (z_i - \bar{z})\bigr)^2.
# $$
#
# Calculate `limit` for the five-token example by hand. Use numpy arithmetic on `Z` and `V`, not a library loss.

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
# Write `hinton_loss(v, z, y, T, alpha)`. Return `(loss, grad)` as the mean over the rows:
#
# $$
# \alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}(y, q_1).
# $$
#
# You can call `L.kd` and `L.hard_ce`.

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
# Width pruning keeps the hidden units that are important on calibration data. Return the indices of the `keep` hidden
# units with the largest mean $\lvert \text{activation} \rvert$ over the contexts `ctx`. Sort the indices from the smallest to the largest. Use
# `model.forward(ctx)`. Its cache is `(contexts, x, h)`.

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
# **The two-minute version.** "We distil because the teacher's full distribution is worth much more per example
# than a sampled token. We do not train the small model from scratch. The teacher's distribution ranks the
# incorrect answers, and a sampled label adds $1 - \sum p^2$ of noise per example that the distribution does not
# add.
#
# "The loss is
# $\alpha\,T^2\,\mathrm{KL}(p_T \,\|\, q_T) + (1 - \alpha)\,\mathrm{CE}$. The gradient of the soft term on the student's
# logits is $T\,(q_T - p_T)$. The $T^2$ keeps the scale of the gradient as $T$ increases. At the limit of high $T$,
# the soft term is logit matching.
#
# "$T$ and $\alpha$ are settings to sweep, not constants. TRL applies no $T^2$ (its GKD trainer calculates the
# divergence at $T$ = 1). Width is capacity. If the student is smaller than the size that can hold what the teacher
# does, no loss closes the gap. If we have the teacher's weights, we prune the teacher to the student's shape by
# activation importance, and we distil into that model. It gets there in fewer steps than a new network, not further."
#
# **Drill questions**
# 1. *Why multiply the soft term by $T^2$?* Its gradient is $(q_T - p_T)/T$, and the difference itself decreases as
#    $\sim 1/T$. Thus the soft gradient decreases as $\sim 1/T^2$. The $T^2$ puts it back, and keeps $\alpha$
#    meaningful across temperatures.
# 2. *Same data budget: fine-tune the small model on the labels, or distil?* If a teacher exists, distil. At two
#    examples per context, the toy student got to 87.6% with soft targets and 67.8% with labels.
# 3. *What cannot soft targets repair?* Capacity: an 8-unit student gets no higher than 93.4% of contexts, even with the
#    exact distribution. Also, soft targets cannot repair anything that the teacher does not know.
