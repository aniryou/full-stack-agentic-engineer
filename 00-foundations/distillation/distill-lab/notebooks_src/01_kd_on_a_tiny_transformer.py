# %% [markdown]
# # 01 · Distillation on a tiny transformer: hard labels against logit KD, SeqKD and GKD
#
# **Tier:** T0 with torch on a laptop CPU. The teacher and four students train in about 90 seconds, and one more
# student trains in about 30 seconds. We measured these times on a shared 4-core container with two torch threads. Thus the
# notebook takes about two minutes. T1 (any GPU) runs the same code faster, but these models do not need that speed.
# Without torch, the notebook still runs: the training cells show a recorded run, labelled illustrative, and every
# exercise is numpy.
#
# ## The one-minute version
#
# * **Distillation trains a small model (the student) to match what a large one (the teacher) does**, not only
#   what the labels say. The three families differ in what you ask the teacher for (PRIMER §1 "Why distil"):
#   * its *logits* on your data (logit KD),
#   * its *samples* (sequence-level KD, which is SFT on the teacher's outputs),
#   * or its *scores of the student's own samples* (on-policy distillation, GKD).
# * **The setup.** A teacher of 101K parameters has learned to write running sums before it answers. These sums
#   make it correct. A student of 26K parameters has a labelled set in which 80% of the answers do not show the
#   steps, like an answer key. Every student gets the same architecture, the same initial weights, 1,500 steps of 64
#   sequences and 1,000 prompts. Only the thing that we ask it to match changes.
# * **The expected result** (the recorded run showed it, and a check cell later in this notebook examines your
#   run). Hard labels teach the student to answer directly. It scores far below every distilled student (0.24 in the recorded run, where a
#   guess scores 0.2). The teacher's logits on the *same* labelled sequences teach it to think first. This is because
#   the soft target says "think" 70% of the time at the branch (PRIMER §2 "Soft targets, temperature and the
#   choice of divergence"). SeqKD on verifier-filtered teacher samples beats the teacher itself. GKD copies the
#   teacher most closely (PRIMER §3 "Sequence-level distillation: learning from the teacher's outputs", §4
#   "On-policy distillation").
# * **What you will build:**
#   * Hinton's loss and its gradient,
#   * GKD's divergence in TRL's convention,
#   * the effect of a verifier filter on what a student inherits,
#   * and why reverse KL can cause a student to give up the thinking mode completely.

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
# A prompt is six base-5 digits, then `=`. The answer is the last digit of their sum. A completion opens `<think>`
# and can write running sums. Then it closes `</think>` and gives the answer and `<eos>`. The verifier examines the
# format and the final answer, never the scratchpad. A two-layer transformer cannot add six digits in the one token
# of its answer, but it can add one digit per scratchpad token.

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
# The teacher trained on the left column, for many steps and on new problems at each step. In the same way, a
# large model sees much more data than you have. The student has the right column: 1,000 labelled problems, and 80% of
# them have a direct answer. A partial scratchpad (1–5 running sums) leaves the rest of the sum for the model to do
# "in the head", so its help is small. Only the full scratchpad reliably gives the correct answer.
#
# ## Exercise 1.1 — Hinton's soft-target loss and its gradient, in numpy
#
# Write `kd_soft(z_teacher, z_student, T)` for one position. Return `(loss, grad)`. The value `loss`
# $= T^2\,\mathrm{KL}(p_T \,\|\, q_T)$, with $p_T = \operatorname{softmax}(z_{\text{teacher}} / T)$ and
# $q_T = \operatorname{softmax}(z_{\text{student}} / T)$. The value `grad` is its gradient with respect to `z_student`.
#
# Derive the gradient by hand. Do not calculate it numerically. The KL is $\sum p\,(\log p - \log q)$. Also,
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
# `train.run()` trains the teacher. Then it samples the teacher's completions for the student's 1,000 prompts. These
# completions are SeqKD's data, and the run keeps a completion only when the verifier accepts it. Then the run trains
# four students from the same initial weights.
#
# Logit KD uses $T$ = 2 and $\alpha = 0.9$ on the soft term. GKD uses $\lambda = 0.5$ (half the batches are the
# student's own samples) and $\beta = 0.5$. These values are TRL's `GKDConfig` defaults. Every 100 steps, the run
# measures each student on 256 held-out problems.

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
# Did *this* run separate the students as the one-minute version says? The next cell does a check and tells you the
# result in each case. No cell after it assumes the answer.

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
# * **hard** learned the habit of the answer key. It rarely writes the scratchpad, and without the scratchpad the
#   answer is a guess (1 in 5). Its 20% of examples with the full scratchpad were too few to learn the running sums
#   in this budget.
# * **kd** saw the *same* 1,000 sequences, but at every position it also saw the teacher's full distribution.
#   Immediately after `<think>`, where the label usually says `</think>`, the teacher says "write the first running
#   sum" 70% of the time. That soft target carries the teacher's *behaviour*, so the student comes to think
#   approximately as often as the teacher does. After that, the student needs a correct scratchpad. It learns to
#   write a correct scratchpad later in the run, and its accuracy increases.
# * **seqkd** trained on the teacher's completions after the verifier removed the incorrect ones. The filter removed
#   mostly the short ones. Thus this student thinks *more* than its teacher and scores *higher*. But it agrees with
#   the teacher less than GKD does. Exercise 1.2 turns this into arithmetic.
# * **gkd** has the lowest KL to the teacher on the teacher's own samples *and* on its own samples (`rkl`).
#   On-policy batches train it at the prefixes that it actually produces. By agreement, it is the best copy of the
#   teacher.
#
# ## Exercise 1.2 — what the verifier filter does to the length distribution
#
# The teacher writes the full scratchpad in a share `f` of its samples. When it writes the full scratchpad, it is
# correct with probability `acc_full`. When it does not, it is correct with probability `acc_other`. If you keep only
# the samples that the verifier accepts, a share `f_after` of them has the full scratchpad. Write
# `filtered_full_share(f, acc_full, acc_other)`. The check reads the three inputs from the SeqKD statistics of this
# run and compares your result with the share that the run kept.

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
# This is the point of PRIMER §5 on a small scale. A student inherits the length distribution of *the traces that
# you keep*, not of the teacher. The verifier kept the long traces because they were more often correct. If a
# length filter points in the opposite direction (keep short traces, for lower-cost serving), it costs accuracy,
# and notebook 03 measures this cost. Without a filter, we expect the SeqKD student to think like its teacher
# (approximately 70%). The recorded experiment `seqkd_all` in the next cell shows what it did.

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
# TRL's `generalized_jsd_loss` takes `beta`. The value $\beta = 0$ gives the forward
# $\mathrm{KL}(\text{teacher} \,\|\, \text{student})$. The value $\beta = 1$ gives the reverse
# $\mathrm{KL}(\text{student} \,\|\, \text{teacher})$. Between these values, the loss is
#
# $$
# \beta\,\mathrm{KL}(p \,\|\, m) + (1 - \beta)\,\mathrm{KL}(q \,\|\, m)
# $$
#
# with $m = \beta\,p + (1 - \beta)\,q$, where $p$ is the teacher and $q$ is the student. Write it for two probability
# vectors. The check compares your result with the fact sheet's values on the five-token example
# ($p = \operatorname{softmax}(z)$, $q = \operatorname{softmax}(v)$). When torch is available, the check also
# compares it with the `distillab.losses` torch version.

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
# Look at the position immediately after `<think>`. The teacher puts 0.896 on the correct first running sum (which
# is the first digit) and 0.1 on `</think>` (answer directly). It puts 0.001 on each other digit. Early in training,
# the student cannot yet tell *which* digit comes next. Compare two students that it can become:
#
# * **A (think, unsure which digit):** 0.18 on each of the five digits, 0.1 on `</think>`.
# * **B (answer directly):** 0.96 on `</think>`, 0.008 on each digit.
#
# Calculate the forward $\mathrm{KL}(p \,\|\, q)$ and the reverse $\mathrm{KL}(q \,\|\, p)$ of each student. Then set
# `forward_prefers` and `reverse_prefers` to `"A"` or `"B"`. Each value is the student that the divergence gives
# the lower score.

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
# Does that occur in training? The next cell trains GKD with $\beta = 1$ (reverse KL) from the same teacher. It
# takes about 30 seconds, because the cell uses the same teacher again. Without torch, it shows the recorded
# experiments.

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
# Reverse KL is not incorrect. MiniLLM and TRL's `DistillationTrainer` (default `beta = 1.0`) use it intentionally.
# The reason is that with mode seeking, a small generator does not spread its mass over outputs that its teacher
# never produces.
#
# But reverse KL increases the probability of a token only in proportion to the student's own probability of that
# token. Thus it needs a student that already puts mass near the teacher's modes. That is why the published recipes
# start on-policy distillation from a student trained with SFT (PRIMER §4, verify). The GKD of this notebook starts
# from scratch.
#
# A warm start helps only if it does put mass there. In distill-core's toy, students that were confidently incorrect
# outside their training data got nothing from it. The value $\beta = 0.5$ is a hedge.
#
# ## Worked example: exposure bias, measured
#
# A student that trains on prefixes from another source (the labelled set, or the teacher's samples) decodes on its
# own. `T.exposure` measures the top-1 agreement of each student with the teacher at each position. It does this two
# times: once with the teacher's samples as the prefix, and once with the student's own samples as the prefix.

# %%
for m in ("kd", "gkd"):
    rows = [{"position": r["position"], "on teacher prefixes": r["on teacher prefixes"], "on own prefixes": r["on own prefixes"],
             "own sequences": r["n own"]} for r in RUN["students"][m]["exposure"]]
    print(table(rows, title=f"[{LABEL}] {m}: agreement with the teacher by completion position"), "\n")

# %% [markdown]
# ## Exercise 1.5 — the exposure gap
#
# Write `exposure_gap(rows)`. It is the mean of (agreement on teacher prefixes − agreement on own prefixes) over
# positions 1 and higher. The weight of each position is the number of the student's own sequences that reach it
# (`"n own"`). A positive gap means that the student does worse on its own prefixes than on the prefixes that it
# trained on. Then predict which of `"kd"` and `"gkd"` gets the smaller gap. Set `smaller_gap` to your prediction.

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
# The gap of SeqKD can be *negative*. Its training prefixes were the teacher's samples that the verifier accepted,
# and most of them are full scratchpads. The held-out teacher samples include partial scratchpads, which the SeqKD
# student rarely saw. Exposure bias is about the distance between the training distribution and the student's own distribution. It
# is not about teacher against student as such.
#
# ## On a real GPU (T1)
#
# `DistillConfig(device="auto")` moves everything to CUDA when a CUDA device is visible (`python -m distillab tinylm`
# from a terminal). At this size, a GPU mostly gives you room to scale the toy. These knobs are worth a second run:

# %%
print(table([{"knob": "kd_temperature / kd_alpha", "try": "1 / 0.5", "what changes": "less of the teacher's branch preference reaches the student"},
             {"knob": "seqkd_filter", "try": "False", "what changes": "the student thinks like the teacher (~70%), not more"},
             {"knob": "gkd_lmbda", "try": "1.0", "what changes": "fully on-policy: slower start from scratch, no exposure gap"},
             {"knob": "gkd_beta", "try": "0.0 / 1.0", "what changes": "forward (covering) vs reverse (seeking) KL"},
             {"knob": "data_full", "try": "0.5", "what changes": "labels that show the working: hard labels catch up"},
             {"knob": "student_d", "try": "16", "what changes": "a smaller student: the capacity gap shows up in accuracy"}],
            title="Experiments for a second run (what to look for, not results)"))

# %% [markdown]
# Notebook 02 does the same with real models. A served teacher generates the data. Then SFT and logit KD train a
# 0.5B student on a T4.
#
# ## In a design review
#
# **Two minutes:** "We distilled with three kinds of teacher signal and compared them on the same budget. Hard labels
# from our answer key taught the student to skip the steps. It scored far below the distilled students, near the
# score of a guess. The teacher's logits on the *same* sequences taught it to think as often as the teacher does.
# That is the extra information that soft targets carry: the teacher's behaviour at every position, not only the next
# token.
#
# "Sequence-level distillation on the outputs of the teacher that the verifier accepted gave the most accurate
# student, more accurate than the teacher. The reason is that the verifier kept mostly the long traces. That student
# also inherits their length, so it costs more to serve. On-policy GKD gave the closest copy of the teacher.
#
# "Reverse KL alone can pull a weak student onto the easy mode. The reason is that it increases the probability of
# a token only as far as the student already proposes that token. Thus we start on-policy distillation from a student trained with SFT, as the published
# recipes do. We also examine $\beta$ on our own data."
#
# **Drill 1.** *Why does logit KD beat SFT on exactly the same sequences?* Every position gets the teacher's full
# distribution instead of one token. Where the label and the teacher disagree, as at the think-or-answer branch
# here, the soft target teaches the teacher's behaviour. Without the $T^2$ factor, the gradient of that term
# decreases to almost zero at high temperature.
#
# **Drill 2.** *The SeqKD student beats its teacher. Is that likely at scale?* It can occur on a narrow, verifiable
# task when the filter keeps better-than-average behaviour. It occurred here, and in distill-core's toy (PRIMER §8,
# 0.975 against the teacher's 0.950). Do not expect it in general.
#
# The R1 distills show something different. SFT on the large model's traces beat RL on the same small base (72.6
# against 47.0 on AIME 2024 for Qwen-32B). But it stayed below DeepSeek-R1's own 79.8 (rl-and-thinking-models
# PRIMER §5). A student that beats its teacher also agrees with it less. Thus, do not grade it by agreement alone
# (PRIMER §8 "Measuring a student").
#
# **Drill 3.** *On-policy distillation from scratch produced a student that never thinks. What occurred?* Reverse
# KL ($\beta$ near 1) is mode-seeking. A student that cannot yet fit the teacher's main mode moves to a mode that it
# can fit.
#
# Give the student a warm start with SFT or SeqKD, so that it already proposes the thinking mode. This is
# the practice of the recipes, and it helps only if the warm start does put mass there. Or, use $\beta \le 0.5$ at
# first.
