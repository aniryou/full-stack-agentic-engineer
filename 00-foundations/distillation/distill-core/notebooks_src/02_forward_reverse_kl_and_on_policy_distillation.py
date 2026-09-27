# %% [markdown]
# # 02 · Forward vs reverse KL, and on-policy distillation
#
# **Tier:** T0 — CPU only, numpy, no network, about a minute. Every divergence here is computed exactly; the
# sequence-level identities are checked by enumerating all 125 continuations of a small language. The same
# trainer on a real transformer (and TRL's `GKDTrainer`) is `distill-lab` notebooks `01_kd_on_a_tiny_transformer`
# and `02_teacher_data_and_a_real_student`.
#
# ## The one-minute version
# When the student cannot copy the teacher, the divergence decides what it gets wrong.
#
# - **Forward KL** $\mathrm{KL}(p \,\|\, q)$ — what SFT and classic KD minimise — makes a too-small student cover
#   every mode and put mass *between* them;
# - **reverse KL** $\mathrm{KL}(q \,\|\, p)$ makes it commit to the modes it can fit;
# - the generalised **JSD($\beta$)** interpolates (TRL: $\beta = 0$ forward, $\beta = 1$ reverse).
# - **Sequence-level distillation** is SFT on text the teacher wrote — the recipe behind the R1 distills — and it pays
#   for tokens the verifier throws away.
# - Its flaw is **exposure bias**: the student learns on the teacher's prefixes and decodes on its own, so one slip
#   puts it where no training example was.
# - **On-policy distillation** (GKD) samples from the student and asks the teacher about every token it produced, so
#   it trains exactly where it will be used; as RL it is REINFORCE with a dense reward $r_t = \log \pi_T - \log \pi_S$
#   per token — `rlcore.pg.reinforce_grad` with the teacher as the reference.
#
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
# Forward KL is paid wherever the teacher has mass and the student has none, so the best one-bump student is nearly
# flat ($s$ = 8.6) and puts 45.4% of its mass on tokens the teacher gives under 1% — in generation, text the teacher
# would never write. Reverse KL is paid wherever the *student* has mass and the teacher has none, so the student
# commits to one mode ($\mathrm{KL} = \ln 2$: it drops half the teacher's mass) and almost never produces a token the
# teacher would reject. $\mathrm{JSD}(0.5)$ still covers here; the flip comes between $\beta = 0.6$ and 0.7 for this
# family.

# %%
for b in (0.0, 0.5, 0.6, 0.7, 0.9, 1.0):
    f = D.fit_bump(p, "jsd", b)
    print(f"β = {b:3.1f}: μ = {f['mu']:.1f}, s = {f['s']:4.1f}, uncovered teacher mass {f['teacher_mass_uncovered']:.2f}")

# %% [markdown]
# ## Worked example 2 — sequence-level distillation: what the data costs
# Twenty prompts (the contexts of two cycles of the rule), 8 samples each, 12 tokens. At $T$ = 0 every sample of a
# prompt is identical; at $T$ = 1 most samples break the rule somewhere and the verifier throws them away — but every
# generated token was paid for.

# %%
o = lang.orbits()
prompts = np.array(o[0] + o[1])
for T in (0.0, 1.0):
    _, st = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=T)
    print(f"T = {T}: {st}")
print(f"a perfect teacher at T = 1 passes (0.8)^12 = {0.8 ** 12:.3f} of 12-token samples")

# %% [markdown]
# ## Worked example 3 — exposure bias
# Train two 16-unit students on the teacher's greedy continuations of those prompts: plain SFT on the tokens, and
# supervised KD (the teacher's full distribution at every position of *its* text: GKD with $\lambda = 0$). Then
# compare each one's accuracy (is its top token the rule's?) on the teacher's prefixes and on its own, sampled at
# $T$ = 1: per position, and over its whole output so far (right at every position up to this one).

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
    own, allr = eb["own_prefixes"], eb["all_right_so_far"]
    print(f"{name:26s} teacher prefixes {eb['teacher_prefixes'].mean():.3f} | own prefixes: position 1 {own[0]:.3f}, "
          f"4 {own[3]:.3f}, 12 {own[11]:.3f} | whole output right through 4 {allr[3]:.3f}, through 12 {allr[11]:.3f} | "
          f"all contexts {E.vs_truth(m, lang)['rule_acc']:.3f} | entropy {entropy(m):.3f}")
    return eb


print(f"teacher entropy {entropy(teacher):.3f}")
eb_t = report("teacher", teacher)
report("SFT on greedy text", sft)
eb_kd = report("supervised KD (λ = 0)", kd)
print("share of sampled tokens that follow the rule, by position:")
print("  teacher", " ".join(f"{x:.2f}" for x in eb_t["sampled_on_rule"]))
print("  KD     ", " ".join(f"{x:.2f}" for x in eb_kd["sampled_on_rule"]))

# %% [markdown]
# The SFT student never slips — because it never varies: its entropy is near zero where the teacher's is 0.64. It has
# copied the teacher's greedy output, not the teacher. The KD student matches the teacher's distribution on the
# teacher's text, so at $T$ = 1 it samples off the rule's cycle: at position 1 as often as the teacher, then more
# often (0.64 of its samples follow the rule from position 4, against the teacher's 0.8).
#
# Each slip lands it in contexts no training example covered, where its top token is wrong about a quarter of the
# time: per position its accuracy falls from 1.000 on the teacher's prefixes to 0.788 by position 4 and then stays
# there. What compounds with length is the whole output: the chance it has been right at every position falls to 0.650
# by position 4 and 0.190 by position 12, while the teacher's stays at 1.000. That is exposure bias.
#
# ## Worked example 4 — on-policy distillation removes it
# Continue from the KD student with GKD at $\lambda = 1$: the student samples its own continuations; the teacher
# scores every position; minimise $\mathrm{JSD}(\beta)$ there.

# %%
runs = {}
for lam, beta in ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 0.5)):
    s = kd.copy()
    op.gkd_train(s, teacher, prompts, 12, 300, lam=lam, beta=beta, data=greedy, seed=1)
    runs[(lam, beta)] = s
    report(f"+ GKD λ = {lam}, β = {beta}", s)

# %% [markdown]
# Training on its own samples, the student learns the contexts it actually visits: at $\beta = 0$ (forward KL) its
# accuracy stays at 0.994 at position 4 and 0.984 at position 12. The reverse end ($\beta = 1$) improves more slowly.
# Is that the starting point? Run both ends from three starts: the KD student, the SFT student, and a fresh one.

# %%
for name, start in (("KD student", kd), ("SFT student", sft), ("fresh student", TinyLM(11, 16, 8, seed=1))):
    v, out = E.vs_truth(start, lang), []
    for beta in (0.0, 1.0):
        s = start.copy()
        op.gkd_train(s, teacher, prompts, 12, 300, lam=1.0, beta=beta, data=greedy, seed=1)
        w = E.vs_truth(s, lang)
        out.append(f"β = {beta:.0f}: {w['rule_acc']:.3f} (where wrong: {w['wrong_top_q']:.2f} on its pick, "
                   f"{w['wrong_right_q']:.3f} on the right token)")
    print(f"{name:14s} start {v['rule_acc']:.3f} (where wrong, {v['wrong_right_q']:.3f} on the right token) → " + "; ".join(out))

# %% [markdown]
# Reverse KL is slow from every start. Its gradient on a logit is $q_i\,(\log q_i - \log p_i - \mathrm{KL})$: it lifts
# a token in proportion to the student's *own* probability of it, so it sharpens what the student already proposes and
# barely finds what it does not. Where the KD and SFT students are wrong, they give the right token 0.029 and 0.007,
# so warming up from them does not help; a fresh student commits early under $\beta = 1$ and ends up in the same
# state. Exercise 2.3 shows the extreme case.
#
# The published recipes do start on-policy distillation from an SFT checkpoint (primer §4; verify): on a real model
# that checkpoint writes in the teacher's format and puts mass near its modes, which this lookup-table toy cannot
# show. TRL's docs say it in general form: on-policy data helps, and the best $\beta$ depends on the task.
#
# ## Worked example 5 — on-policy distillation is policy gradient with a dense reward
# Reverse KL over whole sequences,
#
# $$
# \mathrm{KL}(\pi_S \,\|\, \pi_T) = \mathbb{E}_{y \sim \pi_S}\bigl[\log \pi_S(y) - \log \pi_T(y)\bigr],
# $$
#
# has the REINFORCE gradient $\mathbb{E}[R(y)\,\nabla \log \pi_S(y)]$ with $R(y) = \sum_t r_t$,
#
# $$
# r_t = \log \pi_T(y_t \mid \cdot) - \log \pi_S(y_t \mid \cdot).
# $$
#
# Check it exactly on a 5-token language with 3-token continuations (125 of them).

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
# GRPO's verifier gives one number per sequence; here every token gets its own. That is the variance argument for
# on-policy distillation: the dense signal needs far fewer steps. Each step is not cheaper, though. The teacher scores
# every student token in one prefill-shaped forward pass, $2N_T$ FLOPs per token, on top of the student's $2N_S$ to
# generate and $6N_S$ to train; GRPO pays $8N_S$, plus $2N_{\mathrm{ref}}$ if it keeps a KL reference model.

# %%
for label, kw in (("on-policy distillation, 32B teacher", {}), ("GRPO", dict(teacher_scores=False)),
                  ("GRPO with an 8B KL reference", dict(teacher_scores=False, reference_params=8e9))):
    f = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, **kw)
    print(f"8B student, 16 samples × 4K tokens, {label:36s}: {f['per_token'] / 1e9:4.0f} GFLOP per token, "
          f"{f['total']:.2e} per prompt")

# %% [markdown]
# ## Exercise 2.1 — the reverse-KL fit
# Write `reverse_kl(q, p)` $= \sum q \log(q/p)$ over entries with $q$ > 0, then return the $(\mu, s)$ of the one-bump
# student (`D.bump(mu, s, 11)`) that minimises it over the grids `mus` and `ss`. Break ties toward the smaller $\mu$.

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
# `jsd(p, q, beta)` with $p$ the teacher and $q$ the student:
#
# $$
# \beta\,\mathrm{KL}(p \,\|\, m) + (1 - \beta)\,\mathrm{KL}(q \,\|\, m), \quad m = \beta\,p + (1 - \beta)\,q;
# $$
#
# at $\beta = 0$ return $\mathrm{KL}(p \,\|\, q)$ and at $\beta = 1$ $\mathrm{KL}(q \,\|\, p)$, as TRL's
# `generalized_jsd_loss` does.

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
# For one position with teacher probabilities `p` and student logits `v`, return
# $\partial \mathrm{KL}(q \,\|\, p) / \partial v$, $q = \operatorname{softmax}(v)$:
#
# $$
# q_i \bigl(\log q_i - \log p_i - \mathrm{KL}(q \,\|\, p)\bigr).
# $$
#
# Then compute `ratio`: the norm of that gradient divided by the norm of the forward-KL gradient $\left(q - p\right)$
# for a student that is confidently wrong, `v_wrong`.

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
print(f"✅ confidently wrong, the reverse-KL gradient is {ratio:.4f} of the forward one: it lifts a token only in "
      "proportion to the student's own probability of it. A warm start helps only if it already puts mass on the "
      "teacher's modes (the KD and SFT students above do not, off the cycle); otherwise sweep β or mix in forward KL")

# %% [markdown]
# ## Exercise 2.4 — advantages for on-policy distillation
# Given per-token log-probabilities of a batch of the student's own samples under the student (`s_lp`, shape ($N$,
# $n$)) and the teacher (`t_lp`), return the per-token weights on $\nabla \log \pi_S$ that rlcore's REINFORCE uses:
# every token of sequence $i$ gets $(R_i - \operatorname{mean} R) / N$ with
# $R_i = \sum_t (t_{\mathrm{lp}} - s_{\mathrm{lp}})$.

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
# `yield_12`, the fraction of 12-token samples at $T$ = 1 that pass the verifier, and `tokens_per_kept`, the
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
# teacher's samples, so it works through any API, and it is how the R1 distills were made. We budget it by teacher
# tokens paid, not tokens kept — the verifier discards most long samples — and we sample at the temperature we want
# the student to have: SFT on greedy outputs gives a student that has copied one output, not the teacher.
#
# "Its weakness is exposure bias: trained on the teacher's prefixes, the student decodes on its own. On-policy
# distillation fixes that — the student samples, the teacher scores every token, and we minimise a divergence there;
# as RL it is REINFORCE with a dense per-token reward, $\log \pi_T - \log \pi_S$, and it costs one teacher forward
# pass per student token, which with a big teacher makes each step dearer than GRPO's; it wins on the number of steps.
#
# "We choose the divergence knowingly: forward KL makes a small student cover everything, including the gaps between
# the teacher's modes; reverse KL makes it commit, and it barely lifts a token the student gives little probability,
# so it is slow where the student is confidently wrong. The recipes start it from an SFT student; we sweep $\beta$,
# and in this toy forward KL won from every start."
#
# **Drill questions**
# 1. *Why can on-policy distillation not be done through a text-only API?* — It needs the teacher's
#    log-probability of every token the *student* generated (a forward pass over given text); SeqKD needs only
#    samples. vLLM's `prompt_logprobs` gives it for a served teacher (verify the limits).
# 2. *Our distilled model produces odd blends of two valid answers. Which loss did we use?* — Forward KL (SFT/KD):
#    a student that cannot hold both modes spreads mass between them. Try reverse KL or a JSD with $\beta$ near 1.
# 3. *Offline KD looked great on teacher-forced loss; generations degrade after a few sentences. Why?* — Exposure
#    bias: teacher-forced metrics are measured on the teacher's prefixes. Measure on the student's own samples,
#    and add on-policy data ($\lambda > 0$).
