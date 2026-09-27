# %% [markdown]
# # 01 · Distillation on a tiny transformer: hard labels against logit KD, SeqKD and GKD
#
# **Tier:** T0 with torch on a laptop CPU. The teacher and four students train in about 90 seconds and one more
# student in about 30 (measured on a shared 4-core container, two torch threads), so the notebook takes about two
# minutes. T1 (any GPU) runs the same code faster, which these models do not need. Without torch the notebook still
# runs: the training cells show a recorded run, labelled illustrative, and every exercise is numpy.
#
# ## The one-minute version
#
# * **Distillation trains a small model (the student) to match what a large one (the teacher) does**, not only
#   what the labels say. The three families differ in what you ask the teacher for: its *logits* on your data
#   (logit KD), its *samples* (sequence-level KD, which is SFT on the teacher's outputs), or its *scores of the
#   student's own samples* (on-policy distillation, GKD) (PRIMER §1 "Why distil").
# * **The setup.** A teacher of 101K parameters has learned to write running sums before it answers, and that is
#   what makes it right. A student of 26K parameters has a labelled set in which 80% of the answers skip the
#   working, like an answer key. Every student gets the same architecture, initial weights, 1,500 steps of 64
#   sequences and 1,000 prompts. Only what it is asked to match changes.
# * **What you should see** (the recorded run did; a cell below checks your run). Hard labels teach the student to
#   answer directly, and it scores far below every distilled student (0.24 in the recorded run, where a guess
#   scores 0.2). The teacher's logits on the *same* labelled sequences teach it to think first, because the soft
#   target at the branch says "think" 70% of the time (PRIMER §2 "Soft targets, temperature and the choice of
#   divergence"). SeqKD on verifier-filtered teacher samples beats the teacher itself. GKD copies the teacher most
#   closely (PRIMER §3 "Sequence-level distillation: learning from the teacher's outputs", §4 "On-policy
#   distillation").
# * **What you will build:** Hinton's loss and its gradient, GKD's divergence in TRL's convention, the effect of a
#   verifier filter on what a student inherits, and why reverse KL can make a student give up thinking altogether.

# %%
import math, statistics
from dataclasses import replace
import numpy as np
from distillab import env
from distillab.report import plot, table
from distillab.tinylm.task import EQ, SumTask, data_mix, render, teacher_mix, verify
from distillab.tinylm.curves import COLS, label, load_experiments, load_recorded, show, summary_rows

print(env.describe())
HAVE_TORCH = env.has_torch()
print("torch available:", HAVE_TORCH, "-> the runs below are measured now" if HAVE_TORCH else "-> recorded run (illustrative)")

# %% [markdown]
# ## Worked example: the task, and the two sets of demonstrations
#
# Six base-5 digits, then `=`. The answer is the last digit of their sum. A completion opens `<think>`, may write
# running sums, closes `</think>`, gives the answer and `<eos>`. The verifier checks the format and the final
# answer, never the scratchpad. A two-layer transformer cannot add six digits in the one token it spends on the
# answer, but it can add one digit per scratchpad token.

# %%
task = SumTask(k=6, base=5)
import random
p = task.sample(random.Random(3))
print("prompt       ", render(p.prompt), "   answer:", p.answer)
for j in (0, 3, 6):
    print(f"scratchpad {j}  ", render(task.demo(p, j)))
print(table([{"scratchpad length": j, "teacher's demos": round(teacher_mix(6)[j], 3), "labelled set": round(data_mix(6)[j], 3)}
             for j in range(7)], title="Share of demonstrations by scratchpad length"))

# %% [markdown]
# The teacher was trained on the left column, for many steps and on fresh problems each step, the way a large
# model sees far more data than you have. The student has the right column: 1,000 labelled problems, 80% of them
# answered directly. A partial scratchpad (1–5 running sums) leaves the rest of the sum to be done "in the head",
# so it helps little. Only the full one reliably gives the right answer.
#
# ## Exercise 1.1 — Hinton's soft-target loss and its gradient, in numpy
#
# Write `kd_soft(z_teacher, z_student, T)` for one position: return `(loss, grad)` where `loss`
# $= T^2\,\mathrm{KL}(p_T \,\|\, q_T)$, with $p_T = \operatorname{softmax}(z_{\text{teacher}} / T)$ and
# $q_T = \operatorname{softmax}(z_{\text{student}} / T)$, and `grad` is its gradient with respect to `z_student`.
# Derive the gradient rather than differentiating numerically. The KL is $\sum p\,(\log p - \log q)$, and
# $\partial(-\sum p \log q) / \partial z_{\text{student}} = (q - p) / T$. The check uses the fact sheet's five-token
# example.

# %% exercise
def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def kd_soft(z_teacher, z_student, T: float):
    ### BEGIN SOLUTION
    p, q = softmax(z_teacher, T), softmax(z_student, T)
    loss = T * T * float((p * (np.log(p) - np.log(q))).sum())
    grad = T * (q - p)                              # T² × (q − p) / T
    return loss, grad
    ### END SOLUTION

# %% check
z, v = [4, 3, 1, 0, -1], [3, 3.5, 0, 0.5, -1]
loss1, _ = kd_soft(z, v, 1.0)
loss2, g2 = kd_soft(z, v, 2.0)
assert abs(loss1 - 0.25650) < 1e-5 and abs(loss2 - 0.26558) < 1e-5
assert np.allclose(g2 / 4.0, [-0.073543, 0.071047, -0.016410, 0.015853, 0.003053], atol=1e-5)   # (q - p)/T = grad / T²
eps = 1e-6
num = [(kd_soft(z, np.add(v, eps * np.eye(5)[i]), 2.0)[0] - kd_soft(z, np.add(v, -eps * np.eye(5)[i]), 2.0)[0]) / (2 * eps) for i in range(5)]
assert np.allclose(num, g2, atol=1e-6)
for temp in (10, 100, 1000):
    print(f"T = {temp:>4}: T² x KL = {kd_soft(z, v, temp)[0]:.5f}")
limit = ((np.subtract(v, np.mean(v)) - np.subtract(z, np.mean(z))) ** 2).sum() / (2 * 5)
assert abs(kd_soft(z, v, 1000)[0] - limit) < 2e-4
if HAVE_TORCH:
    import torch
    from distillab.losses import kd_loss
    vt = torch.tensor([v], dtype=torch.float64, requires_grad=True)
    lt = kd_loss(vt, torch.tensor([z], dtype=torch.float64), alpha=1.0, temperature=2.0)
    lt.backward()
    assert abs(lt.item() - loss2) < 1e-9 and np.allclose(vt.grad.numpy()[0], g2)
print(f"✅ loss and gradient right; as T grows, T² x KL tends to {limit:.3f}: matching centred logits "
      "(the T² keeps the soft term's gradient the same size at any temperature)")

# %% [markdown]
# ## Worked example: the teacher and four students, trained now
#
# `train.run()` trains the teacher, samples its completions for the student's 1,000 prompts (SeqKD's data, kept only
# when the verifier accepts them), then trains four students from the same initial weights. Logit KD uses $T$ = 2 and
# $\alpha = 0.9$ on the soft term. GKD uses $\lambda = 0.5$ (half the batches are the student's own samples) and
# $\beta = 0.5$, which are TRL's `GKDConfig` defaults. Every 100 steps each student is measured on 256 held-out
# problems.

# %%
if HAVE_TORCH:
    from distillab.tinylm import train as T
    CFG = T.DistillConfig()
    RUN = T.run(CFG, log=lambda *a: None)             # ~90 s on a laptop CPU
else:
    RUN = load_recorded()
LABEL = label(RUN)
print(show(RUN, curves=False))
plot({m: s["curve"] for m, s in RUN["students"].items()}, "step", ["accuracy", "full", "agree", "kl"], title=LABEL)

# %% [markdown]
# Did *this* run separate the students the way the one-minute version says? The next cell checks and says so
# either way. Nothing below assumes the answer.

# %%
S = {m: s["final"] for m, s in RUN["students"].items()}
TE = RUN["teacher"]["eval"]
claims = {
    "hard labels answer directly (full scratchpad < 40%)": S["hard"]["full"] < 0.4,
    "hard labels score below every distilled student": S["hard"]["accuracy"] < min(S[m]["accuracy"] for m in ("kd", "seqkd", "gkd")),
    "every distilled student copies the teacher better than hard labels (agree, kl)":
        all(S[m]["agree"] > S["hard"]["agree"] and S[m]["kl"] < S["hard"]["kl"] for m in ("kd", "seqkd", "gkd")),
    "logit KD on the same sequences learns to think (full > hard + 0.2)": S["kd"]["full"] > S["hard"]["full"] + 0.2,
    "SeqKD with the verifier filter beats hard labels by 0.3 in accuracy": S["seqkd"]["accuracy"] > S["hard"]["accuracy"] + 0.3,
    "GKD has the lowest KL to the teacher": S["gkd"]["kl"] == min(s["kl"] for s in S.values()),
}
for c, ok in claims.items():
    print(f"{'yes' if ok else 'NO ':>4}  {c}")
SEPARATED = all(claims.values())
print(f"[{LABEL}] " + ("the four students separated as expected." if SEPARATED else
      "WARNING: this run did not separate as expected; read the curves above, and compare the recorded run "
      "(DISTILLAB_NO_TORCH=1) or another seed."))

# %% [markdown]
# How to read the table:
#
# * **hard** learned the answer key's habit: it rarely writes the scratchpad, and without it the answer is a guess
#   (1 in 5). Its 20% of full-working examples were too few to learn the running sums in this budget.
# * **kd** saw the *same* 1,000 sequences, but at every position it was also shown the teacher's whole
#   distribution. Right after `<think>`, where the label usually says `</think>`, the teacher says "write the
#   first running sum" 70% of the time. That soft target carries the teacher's *behaviour*, and the student
#   comes to think about as often as the teacher does. It then needs its scratchpad to be right, which it
#   learns later in the run, and its accuracy climbs.
# * **seqkd** trained on the teacher's completions after the verifier removed the wrong ones. The filter removed
#   mostly the short ones, so this student thinks *more* than its teacher and scores *higher*, while agreeing
#   with it less than GKD does. Exercise 1.2 turns this into arithmetic.
# * **gkd** has the lowest KL to the teacher on the teacher's own samples *and* on its own samples (`rkl`).
#   On-policy batches train it at the prefixes it actually produces. By agreement it is the best copy of the
#   teacher.
#
# ## Exercise 1.2 — what the verifier filter does to the length distribution
#
# The teacher writes the full scratchpad in a share `f` of its samples and is right with probability `acc_full`
# when it does and `acc_other` when it does not. Keeping only the verified samples leaves a share `f_after` with
# the full scratchpad. Write `filtered_full_share(f, acc_full, acc_other)`. The check reads the three inputs from
# this run's SeqKD statistics and compares with the share the run kept.

# %% exercise
def filtered_full_share(f: float, acc_full: float, acc_other: float) -> float:
    ### BEGIN SOLUTION
    return f * acc_full / (f * acc_full + (1 - f) * acc_other)
    ### END SOLUTION

# %% check
sq = RUN["data"]["seqkd"]
pred = filtered_full_share(sq["full_before"], sq["acc_full"], sq["acc_other"])
assert abs(filtered_full_share(0.5, 1.0, 0.2) - 1 / 1.2) < 1e-12 and filtered_full_share(1.0, 0.9, 0.1) == 1.0
assert abs(pred - sq["full_after"]) < 2e-3, (pred, sq["full_after"])
print(f"✅ [{LABEL}] the teacher's samples: {sq['full_before']:.0%} full scratchpad, right {sq['acc_full']:.0%} of the time "
      f"with it and {sq['acc_other']:.0%} without; after the filter {pred:.0%} are full "
      f"(the SeqKD student then writes it {S['seqkd']['full']:.0%} of the time, the teacher {TE['full']:.0%})")

# %% [markdown]
# That is PRIMER §5's point in miniature: a student inherits the length distribution of *the traces you keep*,
# not of the teacher. The verifier kept the long traces because they were more often right. A length filter
# pointing the other way (keep short traces, for cheaper serving) would cost accuracy, which notebook 03 measures.
# Without any filter the SeqKD student should think like its teacher (about 70%). The recorded experiment
# `seqkd_all` below shows what it did.

# %%
EXP = load_experiments()
e = EXP["seqkd_all"]
print(table([{"student": "seqkd (verified only)", **{c: S["seqkd"][c] for c in ("accuracy", "full", "agree", "kl", "accept")}},
             {"student": "seqkd_all (every sample)", **{c: e["final"][c] for c in ("accuracy", "full", "agree", "kl", "accept")}},
             {"student": "teacher", **{c: TE[c] for c in ("accuracy", "full", "agree", "kl", "accept")}}],
            title=f"[{e['source']}] SeqKD with and without the verifier filter"))

# %% [markdown]
# ## Exercise 1.3 — GKD's divergence, in TRL's convention
#
# TRL's `generalized_jsd_loss` takes `beta`. $\beta = 0$ is the forward
# $\mathrm{KL}(\text{teacher} \,\|\, \text{student})$, $\beta = 1$ is the reverse
# $\mathrm{KL}(\text{student} \,\|\, \text{teacher})$, and in between it is
#
# $$
# \beta\,\mathrm{KL}(p \,\|\, m) + (1 - \beta)\,\mathrm{KL}(q \,\|\, m)
# $$
#
# with $m = \beta\,p + (1 - \beta)\,q$, where $p$ is the teacher and $q$ the student. Write it for two probability
# vectors. The check pins the fact sheet's values on the five-token example ($p = \operatorname{softmax}(z)$,
# $q = \operatorname{softmax}(v)$), and the `distillab.losses` torch version when torch is here.

# %% exercise
def kl(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a > 0
    return float((a[m] * np.log(a[m] / b[m])).sum())

def my_gjsd(p, q, beta: float) -> float:
    ### BEGIN SOLUTION
    if beta == 0:
        return kl(p, q)
    if beta == 1:
        return kl(q, p)
    m = beta * np.asarray(p) + (1 - beta) * np.asarray(q)
    return beta * kl(p, m) + (1 - beta) * kl(q, m)
    ### END SOLUTION

# %% check
pz, qv = softmax(z), softmax(v)
want = {0.0: 0.256501, 0.01: 0.002539, 0.1: 0.023026, 0.5: 0.064431, 0.9: 0.024067, 0.99: 0.002683, 1.0: 0.271424}
for beta, w in want.items():
    assert abs(my_gjsd(pz, qv, beta) - w) < 2e-6, (beta, my_gjsd(pz, qv, beta), w)
if HAVE_TORCH:
    from distillab.losses import generalized_jsd
    for beta in want:
        t = generalized_jsd(torch.tensor([v], dtype=torch.float64), torch.tensor([z], dtype=torch.float64), beta).item()
        assert abs(t - my_gjsd(pz, qv, beta)) < 1e-9
print("✅ TRL's endpoints are the exact KLs (0.2565 forward, 0.2714 reverse); in between the JSD is at most ln 2, "
      "~beta x KL(p||q) near beta = 0 and ~(1 - beta) x KL(q||p) near beta = 1, so the loss scale jumps ~1/beta at one "
      "end and ~1/(1 - beta) at the other: re-tune the learning rate")

# %% [markdown]
# ## Exercise 1.4 — mode covering and mode seeking at the branch token
#
# Look at the position right after `<think>`. The teacher puts 0.896 on the correct first running sum (which is
# the first digit), 0.1 on `</think>` (answer directly), and 0.001 on each other digit. Early in training the
# student cannot yet tell *which* digit comes next. Compare two students it could become:
#
# * **A (think, unsure which digit):** 0.18 on each of the five digits, 0.1 on `</think>`;
# * **B (answer directly):** 0.96 on `</think>`, 0.008 on each digit.
#
# Compute the forward $\mathrm{KL}(p \,\|\, q)$ and the reverse $\mathrm{KL}(q \,\|\, p)$ of each, and set
# `forward_prefers` and `reverse_prefers` to `"A"` or `"B"`, the student each divergence scores lower.

# %%
V_BRANCH = ["0", "1", "2", "3", "4", "</think>"]
correct = 2                                                   # say the first digit is 2
p_branch = np.array([0.896 if i == correct else 0.001 for i in range(5)] + [0.1])
q_A = np.array([0.18] * 5 + [0.1])
q_B = np.array([0.008] * 5 + [0.96])

# %% exercise
forward_prefers = reverse_prefers = None
### BEGIN SOLUTION
fA, fB = kl(p_branch, q_A), kl(p_branch, q_B)
rA, rB = kl(q_A, p_branch), kl(q_B, p_branch)
forward_prefers = "A" if fA < fB else "B"
reverse_prefers = "A" if rA < rB else "B"
### END SOLUTION

# %% check
nums = {"forward A": kl(p_branch, q_A), "forward B": kl(p_branch, q_B), "reverse A": kl(q_A, p_branch), "reverse B": kl(q_B, p_branch)}
assert forward_prefers == "A" and reverse_prefers == "B", nums
print(table([{k: round(v, 3) for k, v in nums.items()}], title="KL at the branch position (nats)"))
print("✅ forward KL pays wherever the teacher has mass the student lacks, so it keeps the student thinking; reverse "
      "KL pays wherever the student has mass the teacher lacks, so a student that cannot yet pick the right digit "
      "retreats to the one mode it can fit: answering directly")

# %% [markdown]
# Does that happen in training? The next cell trains GKD with $\beta = 1$ (reverse KL) from the same teacher: about 30
# seconds, since the teacher is reused. Without torch it shows the recorded experiments.

# %%
if HAVE_TORCH:
    BETA = {b: T.run(replace(CFG, methods=("gkd",), gkd_beta=b), log=lambda *a: None)["students"]["gkd"]["final"]
            for b in (1.0,)}
    BETA[0.5] = S["gkd"]
    src = LABEL
else:
    BETA = {1.0: EXP["gkd_beta1"]["final"], 0.0: EXP["gkd_beta0"]["final"], 0.5: S["gkd"]}
    src = EXP["gkd_beta1"]["source"]
print(table([{"beta": b, **{c: BETA[b][c] for c in ("accuracy", "full", "agree", "kl", "rkl")}} for b in sorted(BETA)],
            title=f"[{src}] GKD students by beta (lambda = 0.5)"))
collapsed = BETA[1.0]["full"] < 0.1
print("reverse KL collapsed to answering directly in this run" if collapsed else
      "this run did not collapse with beta = 1: the student learned the first digit before reverse KL pushed it "
      "away; the pull toward the easy mode is a tendency, not a certainty")

# %% [markdown]
# Reverse KL is not wrong. MiniLLM and TRL's `DistillationTrainer` (default `beta = 1.0`) use it on purpose,
# because mode seeking keeps a small generator from spreading mass over outputs its teacher would never produce.
# But it lifts a token only in proportion to the student's own probability of it, so it needs a student that
# already puts mass near the teacher's modes. That is why the published recipes start on-policy distillation from
# an SFT'd student (PRIMER §4, verify); this notebook's GKD starts from scratch. A warm start helps only if it does
# put mass there: in distill-core's toy, students that were confidently wrong off their training data gained
# nothing from it. $\beta = 0.5$ hedges.
#
# ## Worked example: exposure bias, measured
#
# A student trained on someone else's prefixes (the labelled set, or the teacher's samples) decodes on its own.
# `T.exposure` measures each student's top-1 agreement with the teacher position by position, once with the
# teacher's samples as the prefix and once with the student's own samples as the prefix.

# %%
for m in ("kd", "gkd"):
    rows = [{"position": r["position"], "on teacher prefixes": r["on teacher prefixes"], "on own prefixes": r["on own prefixes"],
             "own sequences": r["n own"]} for r in RUN["students"][m]["exposure"]]
    print(table(rows, title=f"[{LABEL}] {m}: agreement with the teacher by completion position"), "\n")

# %% [markdown]
# ## Exercise 1.5 — the exposure gap
#
# Write `exposure_gap(rows)`: the mean over positions 1 onward of (agreement on teacher prefixes − agreement on
# own prefixes), weighted by how many of the student's own sequences reach each position (`"n own"`). A positive
# gap means the student does worse on its own prefixes than on the ones it was trained on. Then predict which
# of `"kd"` and `"gkd"` should have the smaller gap, and set `smaller_gap`.

# %% exercise
def exposure_gap(rows: list) -> float:
    ### BEGIN SOLUTION
    rs = [r for r in rows if r["position"] >= 1 and r["n own"] > 0]
    w = sum(r["n own"] for r in rs)
    return sum((r["on teacher prefixes"] - r["on own prefixes"]) * r["n own"] for r in rs) / w
    ### END SOLUTION

smaller_gap = None
### BEGIN SOLUTION
smaller_gap = "gkd"          # half its batches are its own samples: it is trained where it is used
### END SOLUTION

# %% check
ref = {}
for m, s in RUN["students"].items():
    arr = np.array([[r["on teacher prefixes"], r["on own prefixes"], r["n own"]] for r in s["exposure"] if r["position"] >= 1], float)
    ref[m] = float(np.average(arr[:, 0] - arr[:, 1], weights=arr[:, 2]))
    assert abs(exposure_gap(s["exposure"]) - ref[m]) < 1e-9
assert smaller_gap == "gkd"
print(table([{"student": m, "exposure gap": round(g, 4)} for m, g in ref.items()], title=f"[{LABEL}] exposure gap"))
print("✅ " + ("this run agrees: GKD's gap is smaller than KD's" if ref["gkd"] < ref["kd"] else
               "the prediction is the theory's; in this run KD's gap came out smaller, so look at the rows above"))

# %% [markdown]
# SeqKD's gap can be *negative*. Its training prefixes were the teacher's verified samples, which are mostly full
# scratchpads, and the held-out teacher samples include partial ones it rarely saw. Exposure bias is about the
# distance between the training distribution and the student's own, not about teacher versus student as such.
#
# ## On a real GPU (T1)
#
# `DistillConfig(device="auto")` moves everything to CUDA when one is visible (`python -m distillab tinylm` from a
# terminal). At this size a GPU mostly buys room to scale the toy. The knobs worth a second run:

# %%
print(table([{"knob": "kd_temperature / kd_alpha", "try": "1 / 0.5", "what changes": "less of the teacher's branch preference reaches the student"},
             {"knob": "seqkd_filter", "try": "False", "what changes": "the student thinks like the teacher (~70%), not more"},
             {"knob": "gkd_lmbda", "try": "1.0", "what changes": "fully on-policy: slower start from scratch, no exposure gap"},
             {"knob": "gkd_beta", "try": "0.0 / 1.0", "what changes": "forward (covering) vs reverse (seeking) KL"},
             {"knob": "data_full", "try": "0.5", "what changes": "labels that show the working: hard labels catch up"},
             {"knob": "student_d", "try": "16", "what changes": "a smaller student: the capacity gap shows up in accuracy"}],
            title="Experiments for a second run (what to look for, not results)"))

# %% [markdown]
# Notebook 02 does the same with real models: a served teacher generates the data, then SFT and logit KD train a
# 0.5B student on a T4.
#
# ## In a design review
#
# **Two minutes:** "We distilled with three kinds of teacher signal and compared them on the same budget. Hard labels
# from our answer key taught the student to skip the working, and it scored far below the distilled students, close to
# guessing. The teacher's logits on the *same* sequences taught it to think as often as the teacher does. That is the
# extra information soft targets carry: the teacher's behaviour at every position, not only the next token.
#
# "Sequence-level distillation on the teacher's verified outputs gave the most accurate student, more accurate than
# the teacher, because the verifier kept mostly the long traces. That student also inherits their length, so it costs
# more to serve. On-policy GKD gave the closest copy of the teacher. Reverse KL alone can pull a weak student onto the
# easy mode, because it lifts a token only as far as the student already proposes it, so we start on-policy
# distillation from an SFT'd student, as the published recipes do, and check $\beta$ on our own data."
#
# **Drill 1.** *Why does logit KD beat SFT on exactly the same sequences?* Every position gets the teacher's full
# distribution instead of one token. Where the label and the teacher disagree, as at the think-or-answer branch
# here, the soft target teaches the teacher's behaviour. The $T^2$ factor keeps that term's gradient from vanishing
# at high temperature.
#
# **Drill 2.** *The SeqKD student beats its teacher. Should we expect that at scale?* It can happen on a narrow,
# verifiable task when the filter keeps better-than-average behaviour: here, and in distill-core's toy (PRIMER §8,
# 0.975 against the teacher's 0.950). Do not expect it in general. The R1 distills show something else: SFT on the
# big model's traces beat RL on the same small base (72.6 against 47.0 on AIME 2024 for Qwen-32B), while staying
# below DeepSeek-R1's own 79.8 (rl-and-thinking-models PRIMER §5). A student that beats its teacher will also agree
# with it less, so do not grade it by agreement alone (PRIMER §8 "Measuring a student").
#
# **Drill 3.** *On-policy distillation from scratch produced a student that never thinks. What happened?* Reverse
# KL ($\beta$ near 1) is mode-seeking. A student that cannot yet fit the teacher's main mode moves to one it can fit.
# Warm-start with SFT or SeqKD so that the student already proposes the thinking mode (the recipes' practice; it
# helps only if the warm start does put mass there), or use $\beta \le 0.5$ at first.
