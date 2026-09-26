# %% [markdown]
# # 02 · Preferences, reward models and DPO
#
# **Tier:** T0 — CPU only, numpy, no network, well under a minute. The bracket task of notebook 01 and a
# catalogue of 200 synthetic answers stand in for a model and a preference dataset; every optimum is computed
# exactly. Real DPO and reward-model training use TRL's `DPOTrainer` and `RewardTrainer` (primer §3).
#
# ## The one-minute version
# When no program can check an answer, people compare two: "A is better than B". **Bradley–Terry** reads that as
# a noisy comparison of rewards, P(A ≻ B) = σ(r(A) − r(B)), so a **reward model** is logistic regression on pairs.
# **RLHF** then maximises E[r] − β·KL(π‖π_ref) with PPO — a clipped policy-gradient step plus a **value model**
# for the baseline (GAE). **DPO** notices that the KL-regularised optimum has a closed form, π* ∝ π_ref·exp(r/β),
# inverts it, r = β·log(π*/π_ref) + const, and substitutes into Bradley–Terry: the constant cancels and the policy
# trains directly on pairs with a classification loss. Its **implicit reward** β·log(π/π_ref) is what TRL logs as
# `rewards/chosen`. What DPO gives up: it only sees the pairs (no exploration), and it optimises the *margin*, so
# the chosen answer's likelihood can fall. And any bias in the annotators — say, a taste for long answers —
# becomes the reward, and optimising harder turns it into padding. Primer: `../PRIMER.md` §3.

# %%
import math

import numpy as np

from rlcore import Policy, SeqTask, pg, pref

task = SeqTask("brackets", 8)
seqs = task.all_sequences()
demos = [task.as_trajectory(s) for s in seqs if task.verify(s)]
ref = Policy.for_task(task)
for _ in range(2):                                   # the same weak SFT reference as notebook 01 (29.7% right)
    pg.sft_step(ref, demos, lr=1.0)
ref_p = ref.sequence_probs(task, seqs)
ok = np.array([task.verify(s) for s in seqs])

# %% [markdown]
# ## Worked example 1 — Bradley–Terry, and why a reward model has no zero point
# Hidden linear reward r = 1.5·x₁ − 0.5·x₂ over two features. Annotators compare random pairs with
# P(A ≻ B) = σ(r_A − r_B). Fitting the same logistic model to "chosen minus rejected" recovers the weights.

# %%
rng = np.random.default_rng(0)
xa, xb = rng.standard_normal((4000, 2)), rng.standard_normal((4000, 2))
w_true = np.array([1.5, -0.5])
a_wins = rng.random(4000) < pref.bt_prob(xa @ w_true, xb @ w_true)
chosen, rejected = np.where(a_wins[:, None], xa, xb), np.where(a_wins[:, None], xb, xa)
w = pref.fit_bradley_terry(chosen, rejected, l2=0.0, steps=1500)
print("fitted weights", w.round(2), "true", w_true)
print(f"training loss {pref.bt_nll(w, chosen, rejected):.3f} (chance: ln 2 = {math.log(2):.3f})")
print("σ(2 − 1) =", round(float(pref.bt_prob(2, 1)), 4), "= σ(12 − 11) =", round(float(pref.bt_prob(12, 11)), 4))

# %% [markdown]
# Only differences are identified: add 10 to every reward and every probability is unchanged. That is why TRL's
# `RewardTrainer` offers `center_rewards_coefficient` — to pin the free constant near zero — and why reward-model
# scores from different runs are not comparable.
#
# ## Worked example 2 — RLHF in two stages: fit a reward model, then RL against it with a KL penalty
# Preferences over bracket strings from a true reward r = 3·balanced, pairs drawn from the reference (as a real
# dataset is collected). Stage 1 fits a one-feature reward model; stage 2 runs REINFORCE with β = 1 against it.
# The target is the closed form π* ∝ π_ref·exp(r/β).

# %%
R = 3.0 * ok
pistar, er_star, kl_star = pg.kl_optimal(ref_p, R, beta=1.0)
rng = np.random.default_rng(0)
ia, ib = rng.choice(256, 4096, p=ref_p), rng.choice(256, 4096, p=ref_p)
wins = rng.random(4096) < pref.bt_prob(R[ia], R[ib])
pairs = [(seqs[i], seqs[j]) if w_ else (seqs[j], seqs[i]) for i, j, w_ in zip(ia, ib, wins)]
print(f"{np.mean(ok[ia] == ok[ib]):.0%} of the pairs are ties in the true reward (their labels are coin flips)")

feat = lambda s: [task.verify(s)]
w_rm = pref.fit_bradley_terry([feat(c) for c, _ in pairs], [feat(r) for _, r in pairs], l2=0.0, steps=3000, lr=3.0)
print(f"stage 1, reward model: r̂ = {w_rm[0]:.2f} · balanced   (true 3)")
rm_task = SeqTask("brackets", 8, reward_fn=lambda s: w_rm[0] * task.verify(s))    # score with the reward model
rlhf = ref.copy()
pg.train_reinforce(rlhf, rm_task, np.random.default_rng(1), steps=400, batch=16, lr=0.3, ref=ref, beta=1.0)
print(f"stage 2, RL with β = 1: P(balanced) {pg.expected(rlhf, task, task.verify):.3f}; "
      f"closed form π*: {pistar @ ok:.3f} (reference {ref_p @ ok:.3f})")

# %% [markdown]
# ## Worked example 3 — DPO: skip the reward model
# The same 4,096 pairs, no reward model and no sampling: one classification loss on the policy's own
# log-probability ratios. TRL's metric names are printed as it trains.

# %%
data = pref.encode_pairs(task, pairs)
dpo = ref.copy()
for epoch in range(151):
    m = pref.dpo_step(dpo, ref, data, beta=1.0, lr=5.0)
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v:+.3f}" for k, v in m.items()))
p_dpo = dpo.sequence_probs(task, seqs)
print(f"DPO: P(balanced) {p_dpo @ ok:.3f} vs π* {pistar @ ok:.3f};  KL(π_DPO‖π*) = {np.sum(p_dpo * np.log(p_dpo / pistar)):.4f}")
ir = np.array([pref.implicit_reward(dpo, ref, task, s, 1.0) for s in seqs])
print(f"implicit reward β·log(π/π_ref): balanced minus unbalanced = {ir[ok > 0].mean() - ir[ok == 0].mean():.2f} (true gap 3)")

# %% [markdown]
# Three things to read off. DPO lands on the same policy as RM + RL — it *is* the same objective, solved in
# closed form. The implicit reward recovers the true reward gap (up to the constant Bradley–Terry cannot see).
# And `rewards/chosen` is **negative**: the chosen strings' log-ratios fell. DPO pushes the *margin*; when the
# two answers of a pair share most of their tokens (here: the same states), lowering both — the rejected one
# faster — also wins. Watch that metric in real runs: a falling chosen log-probability is the usual first sign
# of over-training.
#
# ## Worked example 4 — PPO's value side, in brief: GAE
# PPO gives every token its own advantage from a learned value model V(s): δ_t = r_t + γV(s_{t+1}) − V(s_t),
# A_t = δ_t + γλ·A_{t+1}. In RLHF the reward-model score sits on the last token and −β·log(π/π_ref) on every
# token. The value model is typically as large as the policy — the memory GRPO saves (notebook 03).

# %%
r_tok, v_tok = [0.0, 0.0, 1.0], [0.5, 0.6, 0.8]
for lam in (0.0, 0.5, 1.0):
    print(f"λ = {lam}: advantages {pref.gae(r_tok, v_tok, gamma=1.0, lam=lam).round(3).tolist()}")

# %% [markdown]
# λ = 0 trusts the value model (one-step TD error: low variance, biased if V is wrong); λ = 1 trusts the sampled
# return (Monte Carlo minus V: unbiased, noisy). PPO then applies the same clipped ratio GRPO uses (notebook 03).
#
# ## Worked example 5 — an annotator who likes long answers
# 40 answers to one prompt, each in five versions padded with 0–800 filler tokens; filler lowers true quality by
# 0.3 per 100 tokens. The reference rarely pads. Annotators prefer higher quality **and** longer answers:
# P(A ≻ B) = σ(Δquality + 0.6·Δlength/100). Fit a reward model on 3,000 such pairs, then optimise it harder and
# harder (smaller β, via the closed form) and watch the true quality.

# %%
cat = pref.response_catalogue()
ref_c = np.exp(-cat["padding"] / 150)
ref_c /= ref_c.sum()
ch, rj = pref.length_biased_prefs(cat, 3000, length_weight=0.6, ref_probs=ref_c)
X = np.stack([cat["quality"], cat["length"] / 100], axis=1)
w_len = pref.fit_bradley_terry(X[ch], X[rj], steps=2000)
print(f"reward model: r̂ = {w_len[0]:.2f}·quality + {w_len[1]:.2f}·length/100  — it learned the annotators, faithfully")
print(f"{'beta':>5} {'KL':>6} {'RM score':>9} {'true quality':>13} {'mean length':>12}")
print(f"{'ref':>5} {0:6.2f} {ref_c @ (X @ w_len):9.2f} {ref_c @ cat['quality']:13.3f} {ref_c @ cat['length']:12.0f}")
for beta in (10, 3, 1, 0.5, 0.3, 0.1):
    pi, er, kl = pg.kl_optimal(ref_c, X @ w_len, beta)
    print(f"{beta:5} {kl:6.2f} {er:9.2f} {pi @ cat['quality']:13.3f} {pi @ cat['length']:12.0f}")
pig, _, klg = pg.kl_optimal(ref_c, cat["quality"], 0.1)
print(f"optimising true quality instead (β = 0.1): quality {pig @ cat['quality']:.3f}, length {pig @ cat['length']:.0f}")

# %% [markdown]
# The reward-model score rises monotonically; true quality rises, peaks around a KL of 0.4–1.7 nats, and then
# falls below the reference while answers get four times longer. That shape — proxy up, gold up then down — is
# reward-model **over-optimisation**, and length is its most common form in practice. Defences: cap the KL
# budget (stop early), control for length when collecting or fitting preferences, prefer verifiable rewards where
# they exist (notebook 03), and evaluate on the true objective.
#
# ## Exercise 2.1 — the Bradley–Terry gradient
# For weights `w` and feature differences `d = x_chosen − x_rejected` (one row per pair), return the gradient of
# the mean log-likelihood Σ log σ(w·d) / n. (Hint: ∂ log σ(z)/∂z = 1 − σ(z).)

# %% exercise
def bt_grad(w, d):
    ### BEGIN SOLUTION
    z = d @ w
    return (d * (1 - pref.sigmoid(z))[:, None]).mean(axis=0)
    ### END SOLUTION

# %% check
d = chosen - rejected
eps = 1e-6
num = [(-pref.bt_nll(w + eps * e, chosen, rejected) + pref.bt_nll(w - eps * e, chosen, rejected)) / (2 * eps) for e in np.eye(2)]
assert np.allclose(bt_grad(w, d), num, atol=1e-7)
assert np.allclose(bt_grad(np.zeros(2), d), d.mean(axis=0) / 2)
print("✅ the BT gradient: each pair pulls w along its feature difference, weighted by how wrong the model is")

# %% [markdown]
# ## Exercise 2.2 — DPO's loss by hand
# The policy raised the chosen answer's log-ratio log(π/π_ref) by +1 and lowered the rejected one's by −1.
# With β = 0.1 compute the DPO loss `loss_a`. Then compute `loss_b` for a pair where both log-ratios are 0.3.

# %% exercise
### BEGIN SOLUTION
loss_a = -math.log(1 / (1 + math.exp(-0.1 * (1.0 - (-1.0)))))
loss_b = -math.log(1 / (1 + math.exp(-0.1 * (0.3 - 0.3))))
### END SOLUTION

# %% check
assert round(loss_a, 6) == 0.598139 and abs(loss_b - math.log(2)) < 1e-12
assert np.isclose(loss_a, pref.dpo_loss(1.0, -1.0, 0.1))
print(f"✅ margin 2, β 0.1: loss {loss_a:.6f}; zero margin: ln 2 = {loss_b:.6f} — where every DPO run starts")

# %% [markdown]
# ## Exercise 2.3 — which pairs teach DPO the most?
# DPO's gradient on a pair is β·σ(−m)·(∇log π(y⁺) − ∇log π(y⁻)) with m = β·(Δ⁺ − Δ⁻). Write `pair_weight(m, beta)`
# = β·σ(−m). Then set `most_informative` to the index of the pair in `margins` that gets the largest update.

# %% exercise
margins = [2.0, 0.0, -1.5, 0.5]                      # m for four pairs
def pair_weight(m, beta):
    ### BEGIN SOLUTION
    return beta / (1 + math.exp(m))
    ### END SOLUTION


### BEGIN SOLUTION
most_informative = int(np.argmax([pair_weight(m, 0.1) for m in margins]))
### END SOLUTION

# %% check
assert np.isclose(pair_weight(0.0, 0.1), 0.05) and most_informative == 2
assert pair_weight(10, 0.1) < 1e-5
print("✅ mis-ranked pairs (m < 0) get up to β; pairs already ranked confidently get almost nothing — DPO is "
      "self-paced, like logistic regression")

# %% [markdown]
# ## Exercise 2.4 — generalised advantage estimation
# Implement `my_gae(rewards, values, gamma, lam)` with V after the last token = 0.

# %% exercise
def my_gae(rewards, values, gamma, lam):
    ### BEGIN SOLUTION
    adv, run = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        run = rewards[t] + gamma * nxt - values[t] + gamma * lam * run
        adv[t] = run
    return np.array(adv)
    ### END SOLUTION

# %% check
for lam in (0.0, 0.5, 0.95, 1.0):
    assert np.allclose(my_gae(r_tok, v_tok, 1.0, lam), pref.gae(r_tok, v_tok, 1.0, lam))
assert np.allclose(my_gae(r_tok, v_tok, 1.0, 1.0), [0.5, 0.4, 0.2])
print("✅ GAE: λ interpolates between the value model's one-step view and the sampled return")

# %% [markdown]
# ## Exercise 2.5 — choose the KL budget
# From the length-biased experiment, pick `best_beta` in `betas` that maximises **true** quality, and report the
# KL it spends (`kl_budget`). This is the early-stopping point a gold eval would give you.

# %% exercise
betas = [10, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1]
### BEGIN SOLUTION
gold = {b: pg.kl_optimal(ref_c, X @ w_len, b)[0] @ cat["quality"] for b in betas}
best_beta = max(gold, key=gold.get)
kl_budget = pg.kl_optimal(ref_c, X @ w_len, best_beta)[2]
### END SOLUTION

# %% check
assert best_beta == 0.7 and 0.4 < kl_budget < 1.2
print(f"✅ β = {best_beta}: true quality peaks at a KL of {kl_budget:.2f} nats; past it the RM is being gamed")

# %% [markdown]
# ## Exercise 2.6 — IPO's fixed margin
# IPO replaces −log σ(β·margin) with (margin − 1/(2β))², where margin = Δ⁺ − Δ⁻. For β = 0.1, what margin does
# IPO drive a pair toward (`ipo_target`)? And what is the DPO loss's gradient with respect to the margin as the
# margin → ∞ (`dpo_grad_at_inf`)?

# %% exercise
### BEGIN SOLUTION
ipo_target = 1 / (2 * 0.1)
dpo_grad_at_inf = 0.0
### END SOLUTION

# %% check
assert ipo_target == 5.0 and pref.ipo_loss(5.0, 0.0, 0.1) == 0.0 and dpo_grad_at_inf == 0.0
assert pref.dpo_loss(50.0, 0.0, 0.1) < pref.dpo_loss(40.0, 0.0, 0.1)          # DPO keeps rewarding a wider margin
print("✅ IPO stops at a margin of 1/(2β) = 5; DPO's loss keeps (slowly) paying for more, which on deterministic "
      "preferences pushes log-ratios without bound")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Where no program can grade the output we collect pairwise preferences and model
# them with Bradley–Terry, P(A ≻ B) = σ(r_A − r_B). The classic path fits a reward model and runs PPO against it
# with a KL penalty to the SFT model — four networks in memory: policy, reference, reward model and value model.
# DPO gets the same KL-regularised optimum in closed form: π* ∝ π_ref·exp(r/β), so r is β·log(π/π_ref) up to a
# constant that cancels in the pairwise likelihood, and we train the policy directly on pairs with a
# classification loss — two networks, no sampling. We watch rewards/margins and rewards/chosen; a falling chosen
# log-ratio is expected but a collapsing one means over-training. The main risk on either path is the preference
# data itself: annotators' biases, length above all, become the reward, and optimising harder turns them into
# the behaviour — so we control for length, cap the KL budget, and evaluate on held-out human or verifiable
# judgements. Where answers can be checked, we use verifiable rewards instead."
#
# **Drill questions**
# 1. *Why does DPO need no reward model?* — The KL-regularised objective's optimum gives r = β·log(π*/π_ref) +
#    β·log Z(x); in the Bradley–Terry difference r(y⁺) − r(y⁻) the Z term cancels, so the likelihood can be written
#    in terms of the policy alone.
# 2. *rewards/chosen is falling during DPO. Bug?* — Not necessarily: DPO optimises the margin, and lowering both
#    log-ratios (the rejected one faster) wins it. Worry when the chosen log-probability collapses or generations
#    degrade; lower the learning rate or epochs, raise β.
# 3. *After RLHF, answers are 3× longer and win-rate against the old model is up. Ship?* — Not on that evidence:
#    length is the classic reward-model exploit and judges share the bias. Compare at matched length or with a
#    length-controlled judge, and check task accuracy.
