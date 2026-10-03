# %% [markdown]
# # 02 · Forward vs reverse KL, and on-policy distillation
#
# **Tier:** T0. It uses only a CPU and numpy, it needs no network, and it takes about a minute. This notebook calculates
# each divergence exactly. It lists all 125 continuations of a small language, and does a check of the sequence-level
# identities on them. The same trainer on a real transformer (and TRL's `GKDTrainer`) is in `distill-lab` notebooks
# `01_kd_on_a_tiny_transformer` and `02_teacher_data_and_a_real_student`.
#
# ## The one-minute version
# When the student cannot copy the teacher, the divergence decides which errors the student makes.
#
# - **Forward KL** $\mathrm{KL}(p \,\|\, q)$ is what SFT and classic KD minimise. It makes a student that is too small
#   cover each mode and put mass *between* the modes.
# - **Reverse KL** $\mathrm{KL}(q \,\|\, p)$ makes the student commit to the modes that it can fit.
# - The generalised **JSD($\beta$)** interpolates between the two (TRL: $\beta = 0$ forward, $\beta = 1$ reverse).
# - **Sequence-level distillation** is SFT on text that the teacher wrote. It is the recipe behind the R1 distills.
#   Sequence-level distillation pays for tokens that the verifier discards.
# - Its problem is **exposure bias**. The student learns on the teacher's prefixes and decodes on its own prefixes.
#   Thus one error puts the student where no training example was.
# - **On-policy distillation** (GKD) samples from the student and asks the teacher about each token that the student
#   made. Thus it trains exactly where you will use it. As RL, it is REINFORCE with a dense reward
#   $r_t = \log \pi_T - \log \pi_S$ per token. This is `rlcore.pg.reinforce_grad` with the teacher as the reference.
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
# The teacher has two good continuations (tokens 2 and 8, for example two ways to write an answer). The student family
# can make only one bump. Fit the student in three ways.

# %%
p = D.bimodal()
print("teacher             ", bar(p))
for name, div, beta in (("forward KL(p‖q)", "forward", 0.5), ("JSD β = 0.5", "jsd", 0.5), ("reverse KL(q‖p)", "reverse", 0.5)):
    f = D.fit_bump(p, div, beta)
    print(f"{name:20s}", bar(f["q"]), f"  μ = {f['mu']:.1f}, s = {f['s']:.1f}; mass where the teacher has < 1%: "
          f"{f['mass_where_teacher_is_empty']:.3f}")

# %% [markdown]
# Forward KL has a cost wherever the teacher has mass and the student has none. Thus the best one-bump student is
# almost flat ($s$ = 8.6). It puts 45.4% of its mass on tokens to which the teacher gives less than 1%. In generation,
# this is text that the teacher never writes.
#
# Reverse KL has a cost wherever the *student* has mass and the teacher has none. Thus the student commits to one mode
# ($\mathrm{KL} = \ln 2$: it drops half of the teacher's mass). It almost never makes a token that the teacher rejects.
# $\mathrm{JSD}(0.5)$ still covers here. For this family, the change from cover to commit occurs between $\beta = 0.6$ and 0.7.

# %%
for b in (0.0, 0.5, 0.6, 0.7, 0.9, 1.0):
    f = D.fit_bump(p, "jsd", b)
    print(f"β = {b:3.1f}: μ = {f['mu']:.1f}, s = {f['s']:4.1f}, uncovered teacher mass {f['teacher_mass_uncovered']:.2f}")

# %% [markdown]
# ## Worked example 2 — sequence-level distillation: what the data costs
# There are twenty prompts (the contexts of two cycles of the rule), 8 samples for each prompt, and samples of 12
# tokens. At $T$ = 0, all samples of a prompt are identical. At $T$ = 1, most samples break the rule at some position,
# and the verifier discards them. But you paid for each generated token.

# %%
o = lang.orbits()
prompts = np.array(o[0] + o[1])
for T in (0.0, 1.0):
    _, st = seqkd.pipeline(teacher, lang, prompts, 8, 12, np.random.default_rng(1), T=T)
    print(f"T = {T}: {st}")
print(f"a perfect teacher at T = 1 passes (0.8)^12 = {0.8 ** 12:.3f} of 12-token samples")

# %% [markdown]
# ## Worked example 3 — exposure bias
# Train two 16-unit students on the teacher's greedy continuations of those prompts:
#
# - plain SFT on the tokens,
# - supervised KD (the teacher's full distribution at each position of *its* text: GKD with $\lambda = 0$).
#
# Then compare each student's accuracy on the teacher's prefixes and on its own prefixes, sampled at $T$ = 1. The
# accuracy asks one question: is the student's top token the token of the rule? Measure it per position. Also measure
# it over the full output so far, that is, correct at each position up to this one.

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
# The SFT student never makes an error, because it never varies. Its entropy is near zero, where the teacher's entropy
# is 0.64. It copied the teacher's greedy output, not the teacher.
#
# The KD student matches the teacher's distribution on the teacher's text. Thus at $T$ = 1, it samples off the cycle of
# the rule. At position 1, it does this as often as the teacher. After that, it does this more often (0.64 of its
# samples obey the rule from position 4, against the teacher's 0.8).
#
# Each error puts the student in contexts that no training example covered. In these contexts, its top token is
# incorrect approximately one quarter of the time. Per position, its accuracy decreases from 1.000 on the teacher's
# prefixes to 0.788 by position 4, and then stays there.
#
# What becomes worse with length is the full output. The chance that the student was correct at each position
# decreases to 0.650 by position 4 and to 0.190 by position 12. The teacher's chance stays at 1.000. That is exposure
# bias.
#
# Your numbers can differ in the last digits. A seeded run is the run of one CPU. The on-policy rows of the next example
# move by up to ±0.05 across CPU kernels. The first note of the primer and `tests/test_primer_numbers.py` say how much.
#
# ## Worked example 4 — on-policy distillation removes it
# Continue from the KD student with GKD at $\lambda = 1$. The student samples its own continuations. The teacher scores
# each position. Minimise $\mathrm{JSD}(\beta)$ at those positions.

# %%
runs = {}
for lam, beta in ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 0.5)):
    s = kd.copy()
    op.gkd_train(s, teacher, prompts, 12, 300, lam=lam, beta=beta, data=greedy, seed=1)
    runs[(lam, beta)] = s
    report(f"+ GKD λ = {lam}, β = {beta}", s)

# %% [markdown]
# When the student trains on its own samples, it learns the contexts that it actually visits. At $\beta = 0$ (forward
# KL), its accuracy stays at 0.994 at position 4 and at 0.984 at position 12. The reverse end ($\beta = 1$) improves
# more slowly. Is the start point the cause? Run both ends from three starts: the KD student, the SFT student, and a new
# student.

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
# Reverse KL is slow from each start. Its gradient on a logit is $q_i\,(\log q_i - \log p_i - \mathrm{KL})$. It
# increases the probability of a token in proportion to the student's *own* probability of that token. Thus it sharpens
# what the student already proposes, and it almost never finds what the student does not propose.
#
# Where the KD and SFT students are incorrect, they give the correct token 0.029 and 0.007. Thus a warm start from them
# does not help. A new student commits early under $\beta = 1$ and gets to the same state. Exercise 2.3 shows the
# extreme case.
#
# The published recipes do start on-policy distillation from an SFT checkpoint (primer §4, verify). On a real model,
# that checkpoint writes in the teacher's format and puts mass near the teacher's modes. This lookup-table toy cannot
# show that. The documentation of TRL says it in general form: on-policy data helps, and the best $\beta$ depends on the
# task.
#
# ## Worked example 5 — on-policy distillation is policy gradient with a dense reward
# Reverse KL over full sequences,
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
# Do an exact check of this on a 5-token language with 3-token continuations (125 of them).

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
# GRPO's verifier gives one number per sequence. Here, each token gets its own number. That is the variance argument for
# on-policy distillation: the dense signal needs many fewer steps. But each step does not cost less.
#
# The teacher scores each student token in one forward pass with the shape of a prefill, $2N_T$ FLOPs per token. This
# adds to the student's $2N_S$ to generate and $6N_S$ to train. GRPO pays $8N_S$, plus $2N_{\mathrm{ref}}$ if it keeps a
# KL reference model.

# %%
for label, kw in (("on-policy distillation, 32B teacher", {}), ("GRPO", dict(teacher_scores=False)),
                  ("GRPO with an 8B KL reference", dict(teacher_scores=False, reference_params=8e9))):
    f = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, **kw)
    print(f"8B student, 16 samples × 4K tokens, {label:36s}: {f['per_token'] / 1e9:4.0f} GFLOP per token, "
          f"{f['total']:.2e} per prompt")

# %% [markdown]
# ## Exercise 2.1 — the reverse-KL fit
# Write `reverse_kl(q, p)` $= \sum q \log(q/p)$ over the entries with $q$ > 0. Then return the $(\mu, s)$ of the
# one-bump student (`D.bump(mu, s, 11)`) that minimises it over the grids `mus` and `ss`. If there is a tie, select the
# smaller $\mu$.

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
# Write `jsd(p, q, beta)`, with $p$ the teacher and $q$ the student:
#
# $$
# \beta\,\mathrm{KL}(p \,\|\, m) + (1 - \beta)\,\mathrm{KL}(q \,\|\, m), \quad m = \beta\,p + (1 - \beta)\,q;
# $$
#
# At $\beta = 0$, return $\mathrm{KL}(p \,\|\, q)$. At $\beta = 1$, return $\mathrm{KL}(q \,\|\, p)$. TRL's
# `generalized_jsd_loss` does the same.

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
# $\partial \mathrm{KL}(q \,\|\, p) / \partial v$, with $q = \operatorname{softmax}(v)$:
#
# $$
# q_i \bigl(\log q_i - \log p_i - \mathrm{KL}(q \,\|\, p)\bigr).
# $$
#
# Then calculate `ratio`. This is the norm of that gradient divided by the norm of the forward-KL gradient
# $\left(q - p\right)$, for a student that is confidently incorrect, `v_wrong`.

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
# The inputs are the per-token log-probabilities of a batch of the student's own samples. `s_lp` holds them under the
# student, with shape ($N$, $n$), and `t_lp` holds them under the teacher. Return the per-token weights on
# $\nabla \log \pi_S$ that rlcore's REINFORCE uses. Each token of sequence $i$ gets $(R_i - \operatorname{mean} R) / N$,
# with $R_i = \sum_t (t_{\mathrm{lp}} - s_{\mathrm{lp}})$.

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
# The teacher is (almost) the true language. The true language obeys the rule with probability 0.8 per token. Predict
# `yield_12`, the fraction of 12-token samples at $T$ = 1 that pass the verifier. Also predict `tokens_per_kept`, the
# teacher tokens that you pay per kept sample (12 tokens per sample).

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
# **The two-minute version.** "Sequence-level distillation is SFT on text that the teacher wrote. It needs only the
# teacher's samples. Thus it works through any API, and the R1 distills came from it.
#
# "We set its budget by the teacher tokens that we pay for, not the tokens that we keep, because the verifier
# discards most long samples. Also, we sample at the temperature that we want the student to have. SFT on greedy
# outputs gives a student that copied one output, not the teacher.
#
# "Its weak point is exposure bias. The student trains on the teacher's prefixes, but it decodes on its own prefixes.
# On-policy distillation repairs that. The student samples, the teacher scores each token, and we minimise a divergence
# there.
#
# "As RL, on-policy distillation is REINFORCE with a dense per-token reward, $\log \pi_T - \log \pi_S$. It costs one
# teacher forward pass per student token. With a large teacher, this makes each step more costly than a GRPO step.
# But on-policy distillation wins on the number of steps.
#
# "We select the divergence and we know what it does. Forward KL makes a small student cover everything, the gaps
# between the teacher's modes included. Reverse KL makes the student commit. It increases the probability of a token
# only by a small quantity when the student gives that token a low probability. Thus it is slow where the student is
# confidently incorrect.
#
# "The recipes start on-policy distillation from an SFT student. We sweep $\beta$, and in this toy, forward KL won from every start."
#
# **Drill questions**
# 1. *Why can you not do on-policy distillation through a text-only API?* It needs the teacher's log-probability of
#    each token that the *student* generated (a forward pass over given text). SeqKD needs only samples. vLLM's
#    `prompt_logprobs` gives it for a served teacher (verify the limits).
# 2. *Our distilled model makes strange mixtures of two valid answers. Which loss did we use?* Forward KL (SFT/KD). A
#    student that cannot hold both modes spreads mass between them. Try reverse KL or a JSD with $\beta$ near 1.
# 3. *Offline KD had a low teacher-forced loss, but the generations become worse after a few sentences. Why?*
#    Exposure bias. We measure teacher-forced metrics on the teacher's prefixes. Measure on the student's own samples,
#    and add on-policy data ($\lambda > 0$).
