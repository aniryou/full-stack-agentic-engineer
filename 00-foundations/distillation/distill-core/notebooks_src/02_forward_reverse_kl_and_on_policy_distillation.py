# %% [markdown]
# # 02 · Forward vs reverse KL, and on-policy distillation
#
# **Tier:** T0 — CPU only, numpy, no network, about a minute. Every divergence here is computed exactly; the
# sequence-level identities are checked by enumerating all 125 continuations of a small language. The same
# trainer on a real transformer (and TRL's `GKDTrainer`) is `distill-lab` notebooks `01_kd_on_a_tiny_transformer`
# and `02_teacher_data_and_a_real_student`.
#
# ## The one-minute version
# When the student cannot copy the teacher, the divergence decides what it gets wrong. **Forward KL**
# KL(p ‖ q) — what SFT and classic KD minimise — makes a too-small student cover every mode and put mass
# *between* them; **reverse KL** KL(q ‖ p) makes it commit to the modes it can fit; the generalised **JSD(β)**
# interpolates (TRL: β = 0 forward, β = 1 reverse). **Sequence-level distillation** is SFT on text the teacher
# wrote — the recipe behind the R1 distills — and it pays for tokens the verifier throws away. Its flaw is
# **exposure bias**: the student learns on the teacher's prefixes and decodes on its own, so one slip puts it
# where no training example was. **On-policy distillation** (GKD) samples from the student and asks the teacher
# about every token it produced, so it trains exactly where it will be used; as RL it is REINFORCE with a dense
# reward r_t = log π_T − log π_S per token — `rlcore.pg.reinforce_grad` with the teacher as the reference.
# Primer: `../../PRIMER.md` §2–§4.

# %%
import numpy as np

from distillcore import ModLang, TinyLM, divergences as D, eval as E, losses as L, onpolicy as op, seqkd
from distillcore.tinylm import fit_language

lang = ModLang(11, 0.2)
C = lang.contexts()
teacher = fit_language(lang, 64)


def bar(p, width=30):
    return " ".join(f"{x:4.2f}" for x in p)

# %% [markdown]
# ## Worked example 1 — a bimodal teacher, a one-bump student
# The teacher has two good continuations (tokens 2 and 8, say two ways to phrase an answer). The student family
# can only make one bump. Fit it three ways.

# %%
p = D.bimodal()
print("teacher             ", bar(p))
for name, div, beta in (("forward KL(p‖q)", "forward", 0.5), ("JSD β = 0.5", "jsd", 0.5), ("reverse KL(q‖p)", "reverse", 0.5)):
    f = D.fit_bump(p, div, beta)
    print(f"{name:20s}", bar(f["q"]), f"  μ = {f['mu']:.1f}, s = {f['s']:.1f}; mass where the teacher has < 1%: "
          f"{f['mass_where_teacher_is_empty']:.3f}")

# %% [markdown]
# Forward KL is paid wherever the teacher has mass and the student has none, so the best one-bump student is
# nearly flat (s = 8.6) and puts 45.4% of its mass on tokens the teacher gives under 1% — in generation, text the
# teacher would never write. Reverse KL is paid wherever the *student* has mass and the teacher has none, so the
# student commits to one mode (KL = ln 2: it drops half the teacher's mass) and almost never produces a token the
# teacher would reject. JSD(0.5) still covers here; the flip comes between β = 0.6 and 0.7 for this family.

# %%
for b in (0.0, 0.5, 0.6, 0.7, 0.9, 1.0):
    f = D.fit_bump(p, "jsd", b)
    print(f"β = {b:3.1f}: μ = {f['mu']:.1f}, s = {f['s']:4.1f}, uncovered teacher mass {f['teacher_mass_uncovered']:.2f}")

# %% [markdown]
# ## Worked example 2 — sequence-level distillation: what the data costs
# Twenty prompts (the contexts of two cycles of the rule), 8 samples each, 12 tokens. At T = 0 every sample of a
# prompt is identical; at T = 1 most samples break the rule somewhere and the verifier throws them away — but
# every generated token was paid for.

# %%
o = lang.orbits()
prompts = np.array(o[0] + o[1])
for T in (0.0, 1.0):
    _, st = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=T)
    print(f"T = {T}: {st}")
print(f"a perfect teacher at T = 1 passes (0.8)^12 = {0.8 ** 12:.3f} of 12-token samples")

# %% [markdown]
# ## Worked example 3 — exposure bias
# Train three 16-unit students on the teacher's greedy continuations of those prompts: plain SFT on the tokens,
# and supervised KD (the teacher's full distribution at every position of *its* text: GKD with λ = 0). Then
# compare the student's accuracy (is its top token the rule's?) on the teacher's prefixes and on its own,
# sampled at T = 1.

# %%
greedy = seqkd.teacher_data(teacher, prompts, 1, 12, np.random.default_rng(1), T=0.0)
sft = TinyLM(11, 16, 8, seed=1)
seqkd.sft(sft, greedy, 400)
kd = TinyLM(11, 16, 8, seed=1)
op.gkd_train(kd, teacher, prompts, 12, 300, lam=0.0, beta=0.0, data=greedy)
ctx_g, _ = teacher.positions(greedy)


def entropy(m):
    q = m.probs(ctx_g)
    return float(-(q * np.log(q)).sum(1).mean())


def report(name, m):
    eb = seqkd.exposure_bias(m, lang, greedy, prompts, np.random.default_rng(5))
    own = eb["own_prefixes"]
    print(f"{name:26s} teacher prefixes {eb['teacher_prefixes'].mean():.3f} | own prefixes: position 1 {own[0]:.3f}, "
          f"4 {own[3]:.3f}, 12 {own[11]:.3f} | all contexts {E.vs_truth(m, lang)['rule_acc']:.3f} | entropy {entropy(m):.3f}")


print(f"teacher entropy {entropy(teacher):.3f}")
report("SFT on greedy text", sft)
report("supervised KD (λ = 0)", kd)

# %% [markdown]
# The SFT student never slips — because it never varies: its entropy is near zero where the teacher's is 0.64.
# It has copied the teacher's greedy output, not the teacher. The KD student matches the teacher's
# distribution on the teacher's text, so at T = 1 it wanders off the rule's cycle about as often as the teacher
# would — into contexts no training example covered — and its accuracy falls from 1.000 on the teacher's
# prefixes to 0.788 by position 4. Per-token errors now compound with length. That is exposure bias.
#
# ## Worked example 4 — on-policy distillation removes it
# Continue from the KD student with GKD at λ = 1: the student samples its own continuations; the teacher scores
# every position; minimise JSD(β) there.

# %%
runs = {}
for lam, beta in ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 0.5)):
    s = kd.copy()
    op.gkd_train(s, teacher, prompts, 12, 300, lam=lam, beta=beta, data=greedy, seed=1)
    runs[(lam, beta)] = s
    report(f"+ GKD λ = {lam}, β = {beta}", s)

# %% [markdown]
# Training on its own samples, the student learns the contexts it actually visits: at β = 0 (forward KL) its
# accuracy stays at 0.994 at position 4 and 0.984 at position 12. The reverse end (β = 1) improves more slowly
# here: where the student is confidently wrong, reverse KL's gradient q·(log q − log p − KL) nearly vanishes
# (exercise 2.3 shows why), so in this toy — a student far from the teacher off the cycle — the forward end wins.
# TRL's docs say the same thing in general form: on-policy data helps, and the best β depends on the task.
#
# ## Worked example 5 — on-policy distillation is policy gradient with a dense reward
# Reverse KL over whole sequences, KL(π_S ‖ π_T) = E_{y~π_S}[log π_S(y) − log π_T(y)], has the REINFORCE gradient
# E[R(y)·∇log π_S(y)] with R(y) = Σ_t r_t, r_t = log π_T(y_t | ·) − log π_S(y_t | ·). Check it exactly on a
# 5-token language with 3-token continuations (125 of them).

# %%
small = ModLang(5, 0.2)
t5, s5, prompt = fit_language(small, 16, steps=500), TinyLM(5, 4, 3, seed=1), np.array([1, 2])
kl5, ps, seqs5 = op.exact_seq_kl(s5, t5, prompt, 3)
g_exact = op.exact_pg_grad(s5, t5, prompt, 3)
dirs = {k: np.random.default_rng(0).normal(size=a.shape) for k, a in s5.p.items()}
sp, sm = s5.copy(), s5.copy()
for k in dirs:
    sp.p[k] += 1e-5 * dirs[k]
    sm.p[k] -= 1e-5 * dirs[k]
fd = -(op.exact_seq_kl(sp, t5, prompt, 3)[0] - op.exact_seq_kl(sm, t5, prompt, 3)[0]) / 2e-5
print(f"KL(π_S ‖ π_T) = {kl5:.4f} nats over {len(seqs5)} continuations")
print(f"along a random direction: −∇KL by finite differences {fd:.6f}; E[R·∇log π] exactly {sum((g_exact[k] * dirs[k]).sum() for k in dirs):.6f}")
print("the per-token rewards of one sample:", np.round(op.token_rewards(s5, t5, seqs5[:1])[0], 3))

# %% [markdown]
# GRPO's verifier gives one number per sequence; here every token gets its own. That is the variance argument
# for on-policy distillation, and the reason its compute per prompt is small: one teacher forward pass over the
# student's tokens (prefill-shaped, 2·N_T per token), against G generated rollouts for GRPO.

# %%
g16 = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False)
d4 = op.flops_per_prompt(8e9, 32e9, 4096, samples=4)
print(f"8B student, 4K-token completions: GRPO with G = 16 {g16['total']:.2e} FLOPs per prompt; "
      f"on-policy distillation with 4 samples scored by a 32B teacher {d4['total']:.2e} ({g16['total'] / d4['total']:.1f}× less)")

# %% [markdown]
# ## Exercise 2.1 — the reverse-KL fit
# Write `reverse_kl(q, p)` = Σ q·log(q/p) over entries with q > 0, then return the (μ, s) of the one-bump student
# (`D.bump(mu, s, 11)`) that minimises it over the grids `mus` and `ss`. Break ties toward the smaller μ.

# %% exercise
mus, ss = np.round(np.linspace(0, 10, 201), 3), np.round(np.arange(0.3, 12.01, 0.1), 2)


def reverse_kl(q, p):
    ### BEGIN SOLUTION
    m = q > 0
    return float((q[m] * (np.log(q[m]) - np.log(p[m]))).sum())
    ### END SOLUTION


def best_bump(p):
    ### BEGIN SOLUTION
    best = min(((round(reverse_kl(D.bump(m, s, 11), p), 9), m, s) for m in mus for s in ss))
    return best[1], best[2]
    ### END SOLUTION

# %% check
ref = D.fit_bump(p, "reverse")
assert best_bump(p) == (ref["mu"], ref["s"]) == (2.0, 0.7)
assert abs(reverse_kl(D.bump(2.0, 0.7), p) - np.log(2)) < 1e-3
print("✅ reverse KL picks one mode (μ = 2, s = 0.7) and pays ln 2 for the half of the teacher it leaves out")

# %% [markdown]
# ## Exercise 2.2 — the generalised JSD, TRL's convention
# `jsd(p, q, beta)` with p the teacher and q the student: β·KL(p ‖ m) + (1 − β)·KL(q ‖ m), m = β·p + (1 − β)·q;
# at β = 0 return KL(p ‖ q) and at β = 1 KL(q ‖ p), as TRL's `generalized_jsd_loss` does.

# %% exercise
def jsd(p, q, beta):
    kl = lambda a, b: float((a[a > 0] * (np.log(a[a > 0]) - np.log(b[a > 0]))).sum())
    ### BEGIN SOLUTION
    if beta == 0:
        return kl(p, q)
    if beta == 1:
        return kl(q, p)
    m = beta * p + (1 - beta) * q
    return beta * kl(p, m) + (1 - beta) * kl(q, m)
    ### END SOLUTION

# %% check
pp, qq = L.softmax(np.array([4.0, 3, 1, 0, -1])), L.softmax(np.array([3.0, 3.5, 0, 0.5, -1]))
for b in (0.0, 0.01, 0.3, 0.5, 0.9, 1.0):
    assert abs(jsd(pp, qq, b) - D.jsd(pp, qq, b)) < 1e-12
assert jsd(pp, qq, 0.5) <= np.log(2) and abs(jsd(pp, qq, 0.01) / 0.01 - jsd(pp, qq, 0.0)) < 0.005
print(f"✅ JSD(β): {jsd(pp, qq, 0.0):.4f} at β = 0 (forward), {jsd(pp, qq, 0.5):.4f} at 0.5, {jsd(pp, qq, 1.0):.4f} at β = 1 (reverse) — "
      "the endpoints are exact KLs, so the loss scale jumps there")

# %% [markdown]
# ## Exercise 2.3 — the reverse-KL gradient, and why it stalls
# For one position with teacher probabilities `p` and student logits `v`, return ∂KL(q ‖ p)/∂v, q = softmax(v):
# q_i·(log q_i − log p_i − KL(q ‖ p)). Then compute `ratio`: the norm of that gradient divided by the norm of the
# forward-KL gradient (q − p) for a student that is confidently wrong, `v_wrong`.

# %% exercise
def reverse_kl_grad(p, v):
    ### BEGIN SOLUTION
    q = L.softmax(v)
    lq, lp = np.log(q), np.log(p)
    kl = float((q * (lq - lp)).sum())
    return q * (lq - lp - kl)
    ### END SOLUTION


p_t = np.array([0.8, 0.1, 0.0999, 0.0001])
v_wrong = np.array([0.0, 0.0, 0.0, 9.0])              # all its mass on the token the teacher rejects
### BEGIN SOLUTION
ratio = float(np.linalg.norm(reverse_kl_grad(p_t, v_wrong)) / np.linalg.norm(L.softmax(v_wrong) - p_t))
### END SOLUTION

# %% check
rng = np.random.default_rng(3)
for _ in range(5):
    pr, vr = L.softmax(rng.normal(size=6)), rng.normal(size=6)
    assert np.allclose(reverse_kl_grad(pr, vr), L.gkd(vr[None], pr[None], 1.0)[1][0])
assert abs(ratio - np.linalg.norm(L.gkd(v_wrong[None], p_t[None], 1.0)[1]) / np.linalg.norm(L.softmax(v_wrong) - p_t)) < 1e-9
assert ratio < 0.02
print(f"✅ confidently wrong, the reverse-KL gradient is {ratio:.4f} of the forward one: the softmax is saturated, "
      "so start reverse-KL distillation from a student that is already close (SFT or KD first), or mix in forward KL")

# %% [markdown]
# ## Exercise 2.4 — advantages for on-policy distillation
# Given per-token log-probabilities of a batch of the student's own samples under the student (`s_lp`, shape
# (N, n)) and the teacher (`t_lp`), return the per-token weights on ∇log π_S that rlcore's REINFORCE uses: every
# token of sequence i gets (R_i − mean R) / N with R_i = Σ_t (t_lp − s_lp).

# %% exercise
def my_advantages(s_lp, t_lp):
    ### BEGIN SOLUTION
    R = (t_lp - s_lp).sum(1)
    A = (R - R.mean()) / len(R)
    return np.repeat(A[:, None], s_lp.shape[1], 1)
    ### END SOLUTION

# %% check
batch = s5.sample(np.tile(prompt, (32, 1)), 3, np.random.default_rng(7))
s_lp, t_lp = s5.token_logprobs(batch), t5.token_logprobs(batch)
assert np.allclose(my_advantages(s_lp, t_lp), op.advantages(s_lp, t_lp))
g_mine = s5.grad_logprob(batch, my_advantages(s_lp, t_lp))
g_lib = op.pg_grad(s5, t5, batch)
assert all(np.allclose(g_mine[k], g_lib[k]) for k in g_lib)
print("✅ reward = teacher log-prob − student log-prob, baseline = batch mean: rlcore's reinforce_grad with ref = teacher, β = 1")

# %% [markdown]
# ## Exercise 2.5 — predict the verifier's yield
# The teacher is (almost) the true language, which follows the rule with probability 0.8 per token. Predict
# `yield_12`, the fraction of 12-token samples at T = 1 that pass the verifier, and `tokens_per_kept`, the
# teacher tokens paid per kept sample (12 tokens per sample).

# %% exercise
### BEGIN SOLUTION
yield_12 = 0.8 ** 12
tokens_per_kept = 12 / yield_12
### END SOLUTION

# %% check
big = seqkd.teacher_data(teacher, C[np.random.default_rng(2).integers(0, 121, 4000)], 1, 12, np.random.default_rng(3), T=1.0)
measured = lang.verify(big).mean()
assert abs(yield_12 - measured) < 0.02 and abs(tokens_per_kept - 12 / 0.8 ** 12) < 1e-9
print(f"✅ predicted {yield_12:.3f}, measured {measured:.3f} over 4,000 samples: about {tokens_per_kept:.0f} teacher tokens per kept "
      "sample — rejection sampling multiplies the data bill by 1/yield")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Sequence-level distillation is SFT on text the teacher wrote; it needs only the
# teacher's samples, so it works through any API, and it is how the R1 distills were made. We budget it by
# teacher tokens paid, not tokens kept — the verifier discards most long samples — and we sample at the
# temperature we want the student to have: SFT on greedy outputs gives a student that has copied one output,
# not the teacher. Its weakness is exposure bias: trained on the teacher's prefixes, the student decodes on its
# own. On-policy distillation fixes that — the student samples, the teacher scores every token, and we minimise
# a divergence there; as RL it is REINFORCE with a dense per-token reward, log π_T − log π_S, and it costs one
# teacher forward pass per student token. We choose the divergence knowingly: forward KL makes a small student
# cover everything, including the gaps between the teacher's modes; reverse KL makes it commit, and its gradient
# stalls where the student is confidently wrong, so we start it from an SFT or KD student and sweep β."
#
# **Drill questions**
# 1. *Why can on-policy distillation not be done through a text-only API?* — It needs the teacher's
#    log-probability of every token the *student* generated (a forward pass over given text); SeqKD needs only
#    samples. vLLM's `prompt_logprobs` gives it for a served teacher (verify the limits).
# 2. *Our distilled model produces odd blends of two valid answers. Which loss did we use?* — Forward KL (SFT/KD):
#    a student that cannot hold both modes spreads mass between them. Try reverse KL or a JSD with β near 1.
# 3. *Offline KD looked great on teacher-forced loss; generations degrade after a few sentences. Why?* — Exposure
#    bias: teacher-forced metrics are measured on the teacher's prefixes. Measure on the student's own samples,
#    and add on-policy data (λ > 0).
