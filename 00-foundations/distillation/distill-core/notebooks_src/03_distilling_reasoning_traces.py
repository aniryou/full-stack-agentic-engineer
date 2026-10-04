# %% [markdown]
# # 03 · Distilling reasoning traces
#
# **Tier:** T0, with only a CPU and numpy, no network, and a few seconds of run time. The reasoning toy is the
# ThinkTask formula of `rlcore`. The model thinks for $L$ tokens and then answers, and the answer is correct with
# probability $1 - e_0\,(1 - q)^L$. A policy is a stopping rule: the chance that the think block ends at each length.
# Thus each accuracy and length in this notebook is exact. Real traces from Qwen3-1.7B or DeepSeek-R1-Distill-Qwen-1.5B
# into a 0.5–0.6B student are in `distill-lab` notebook `03_distilling_reasoning_traces_for_real` (T1).
#
# ## The one-minute version
# The traces of a thinking model teach a student a procedure. This procedure includes **when to stop thinking**. SFT on
# traces is maximum likelihood. Thus the student gets the teacher's thinking-length distribution, with its tail. With
# that distribution, the student also gets the serving workload of the teacher (rl-and-thinking-models §7).
#
# A **filter** on the traces changes the shape of that distribution. If you keep only the correct traces (rejection
# sampling), the student thinks longer. If you put a cap on the length (budget-aware distillation), the student thinks
# for a shorter time, and the accuracy decreases.
#
# A trace carries the full behaviour, but an RL reward carries one bit. Thus a thousand traces from a strong teacher are
# better than a thousand RL rollouts on the student. This is the toy version of R1's result.
#
# What does not transfer is **knowledge**. Accuracy depends on the student's own $q$. Thus a weaker student that copies
# the teacher's length is less accurate. The correct choice for that student is to think longer, not for the same
# length. Primer: `../../PRIMER.md` §5, and §3 for the filters and the data budget.

# %%
import numpy as np

from distillcore import ThinkToy, reasoning as R

task = ThinkToy(e0=0.8, q=0.1, max_think=32)      # rlcore.tasks.ThinkTask's defaults


def row(name, e):
    print(f"{name:34s} accuracy {e['accuracy']:.3f}  mean length {e['length']:5.2f}  p90 {e['p90']:2d}  "
          f"tokens per correct {e['tokens_per_correct']:5.2f}")

# %% [markdown]
# ## Worked example 1 — the teacher, made by RL
# An untrained student answers at once half of the time. The teacher is the result of 16,000 REINFORCE rollouts on the
# same task (as in rl-and-thinking-models notebook 01). It learned to think for approximately 20 tokens.

# %%
row("untrained student", R.LengthPolicy(32).expected(task))
teacher = R.reinforce(task, np.random.default_rng(0), 16000)
row("teacher (RL, 16,000 rollouts)", teacher.expected(task))
d = teacher.length_distribution()
print("teacher's length distribution, L = 12..27:", " ".join(f"{x:.2f}" for x in d[12:28]))

# %% [markdown]
# ## Worked example 2 — distil it: SFT on its traces
# Sample 1,000 traces (a length, and if the answer was correct). Then fit the student by maximum likelihood. For a
# stopping-rule table, the maximum-likelihood fit has a closed form. It is the fraction of the traces that stopped at
# each length, among the traces that got to that length.

# %%
L, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
print(f"1,000 traces: mean length {L.mean():.2f}, {ok.mean():.1%} correct; correct traces average "
      f"{L[ok].mean():.2f} tokens, wrong ones {L[~ok].mean():.2f}")
student = R.distil(L, ok)
row("student (SFT on 1,000 traces)", student.expected(task))

# %% [markdown]
# The student reproduces the teacher: 0.892 against 0.893, the same mean length and almost the same tail. Nothing told
# the student how long to think. It copied that from the traces. Also, a fleet that serves the student will see the
# teacher's thinking-length distribution.
#
# ## Worked example 3 — how many traces?

# %%
for n in (10, 30, 100, 300, 1000):
    Ln, okn = R.traces(teacher, task, np.random.default_rng(2), n)
    e = R.distil(Ln, okn).expected(task)
    print(f"{n:5d} traces: accuracy {e['accuracy']:.3f}, mean length {e['length']:5.2f}")

# %% [markdown]
# With few traces, the student saw few long think blocks. Its untrained prior (stop half of the time) fills the gaps, so
# it stops early. The student learns the length from the tail of the data.
#
# ## Worked example 4 — filtering sets the accuracy–length trade

# %%
for name, keep, ml in (("all traces (plain SeqKD)", "all", None), ("correct only (rejection sampling)", "correct", None),
                       ("correct and L ≤ 16", "correct", 16), ("correct and L ≤ 12", "correct", 12)):
    kept = (ok if keep == "correct" else np.ones_like(ok)) & (L <= (ml if ml is not None else 99)) & (L < 32)
    row(f"{name} ({kept.sum()} kept)", R.distil(L, ok, keep=keep, max_len=ml).expected(task))

# %% [markdown]
# Correct traces are longer on average (it helps to think). Thus, with rejection sampling, the student thinks slightly
# longer. A length cap at $L \le 16$ decreases the tokens per correct answer from 22.3 to 16.6. It also decreases the
# accuracy from 0.892 to 0.769. The budget is a product decision, and the traces that you keep make that decision for
# you. The cap also discards 91% of the traces that you paid the teacher to write.
#
# ## Worked example 5 — distillation against RL, at equal samples
# The same 1,000 samples have two uses: 1,000 teacher traces for SFT, or 1,000 of the student's own rollouts for
# REINFORCE with a correct/incorrect reward.

# %%
for n in (1000, 4000, 16000):
    e = R.reinforce(task, np.random.default_rng(3), n).expected(task)
    print(f"RL on the student, {n:6,d} rollouts: accuracy {e['accuracy']:.3f}, mean length {e['length']:5.2f}")
print(f"distilled from 1,000 traces:        accuracy {student.expected(task)['accuracy']:.3f}")

# %% [markdown]
# A trace shows the student *how long* to think. The reward of a rollout says only correct or incorrect. RL on the
# student needs the teacher's own budget to get to where the traces put the student at once. R1's comparison had the
# same shape at scale (the distilled 32B was better than RL on the 32B base, primer §5). R1's comparison also had a
# capability ceiling that the toy does not have.
#
# ## Worked example 6 — what does not transfer
# The answer is knowledge. Give the student a smaller $q$ (it solves the problem half as often per thinking token). It
# copies the teacher's length exactly, and that length is incorrect for it.

# %%
weak = student.expected(task, q=0.05)
row("student with q = 0.05 (copied length)", weak)
print(f"best length at a cost of 0.01 per token: teacher (q = 0.1) {task.optimal_length(0.01):.1f}, "
      f"student (q = 0.05) {task.optimal_length(0.01, q=0.05):.1f}")

# %% [markdown]
# ## Exercise 3.1 — SFT on traces, in closed form
# Write `fit_hazard(lengths, max_think, prior)`. For each $t$ < `max_think`, calculate the stopping rate
#
# $$
# h_t = \frac{\#\text{traces with } L = t + \text{prior}/2}{\#\text{traces with } L \ge t + \text{prior}}.
# $$
#
# Return the array of $h_t$.

# %% exercise
def fit_hazard(lengths, max_think, prior=1.0):
    ### BEGIN SOLUTION
    L_ = np.asarray(lengths)
    reached = np.array([(L_ >= t).sum() for t in range(max_think)], float)
    stops = np.array([(L_ == t).sum() for t in range(max_think)], float)
    return (stops + 0.5 * prior) / (reached + prior)
    ### END SOLUTION

# %% check
for seed in range(3):
    Ls = np.random.default_rng(seed).integers(0, 20, 300)
    ref = 1 / (1 + np.exp(-R.LengthPolicy.fit(Ls, 32, 1.0).theta))
    assert np.allclose(fit_hazard(Ls, 32, 1.0), ref)
assert np.allclose(fit_hazard([0, 1, 1, 2, 2, 2], 4, 1.0), [1.5 / 7, 2.5 / 6, 3.5 / 4, 0.5])
print("✅ maximum likelihood on traces = the empirical stopping rate; the prior speaks only where the traces are silent")

# %% [markdown]
# ## Exercise 3.2 — exact accuracy and length of a stopping rule
# From the stopping rates `h` (length `max_think`), calculate
#
# $$
# P(L = t) = h_t \prod_{s<t} (1 - h_s) \quad \text{and} \quad P(\text{truncated}) = \prod (1 - h_s);
# $$
#
# Then calculate `acc` $= \sum_t P(L = t)\,\bigl(1 - e_0\,(1 - q)^t\bigr)$ (truncated traces score 0). Also calculate
# `mean_len` (truncated traces count `max_think` tokens).

# %% exercise
def exact(h, e0=0.8, q=0.1):
    ### BEGIN SOLUTION
    h = np.asarray(h, float)
    survive = np.concatenate([[1.0], np.cumprod(1 - h)])
    pL, trunc = survive[:-1] * h, survive[-1]
    t = np.arange(len(h))
    acc = float(pL @ (1 - e0 * (1 - q) ** t))
    mean_len = float(pL @ t + trunc * len(h))
    return acc, mean_len
    ### END SOLUTION

# %% check
for pol in (teacher, student, R.LengthPolicy(32), R.LengthPolicy.geometric(8)):
    e = pol.expected(task)
    acc, ml = exact(1 / (1 + np.exp(-pol.theta)))
    assert abs(acc - e["accuracy"]) < 1e-12 and abs(ml - e["length"]) < 1e-12
print("✅ accuracy = Σ P(L)·(1 − e0(1 − q)^L): exact, because the policy is a table")

# %% [markdown]
# ## Exercise 3.3 — the REINFORCE gradient of a stopping rule
# For one trace of length $L$ ($L$ < `max_think`: it stopped), return $\nabla_\theta \log \pi(L)$, where
#
# $$
# \log \pi(L) = \sum_{t<L} \log\bigl(1 - \sigma(\theta_t)\bigr) + \log \sigma(\theta_L).
# $$
#
# The derivative of $\log \sigma$ is $1 - \sigma$. The derivative of $\log(1 - \sigma)$ is $-\sigma$.

# %% exercise
def grad_log_pi(theta, L_):
    ### BEGIN SOLUTION
    s = 1 / (1 + np.exp(-theta))
    g = np.zeros_like(theta)
    g[:L_] = -s[:L_]
    g[L_] = 1 - s[L_]
    return g
    ### END SOLUTION

# %% check
th = np.random.default_rng(4).normal(size=32)
logpi = lambda th_, L_: float(np.log(1 - 1 / (1 + np.exp(-th_[:L_]))).sum() + np.log(1 / (1 + np.exp(-th_[L_]))))
for L_ in (0, 3, 17):
    num = np.array([(logpi(th + 1e-6 * np.eye(32)[j], L_) - logpi(th - 1e-6 * np.eye(32)[j], L_)) / 2e-6 for j in range(32)])
    assert np.allclose(grad_log_pi(th, L_), num, atol=1e-7)
print("✅ one rollout pushes every 'keep thinking' decision it made and its one 'stop' — by a single reward")

# %% [markdown]
# ## Exercise 3.4 — pick the cheapest filter that keeps accuracy
# Use the length caps `caps`, with the filter that keeps only correct traces. Select `cap`: the smallest cap whose
# distilled student gets accuracy ≥ 0.85 on `task` with the 1,000 traces `L, ok`. Then give its `tokens_per_correct`.

# %% exercise
caps = [12, 14, 16, 18, 20, 22, 24, 31]
### BEGIN SOLUTION
cap = min(c for c in caps if R.distil(L, ok, keep="correct", max_len=c).expected(task)["accuracy"] >= 0.85)
tokens_per_correct = R.distil(L, ok, keep="correct", max_len=cap).expected(task)["tokens_per_correct"]
### END SOLUTION

# %% check
accs = {c: R.distil(L, ok, keep="correct", max_len=c).expected(task)["accuracy"] for c in caps}
assert accs[cap] >= 0.85 and all(accs[c] < 0.85 for c in caps if c < cap)
assert abs(tokens_per_correct - R.distil(L, ok, keep="correct", max_len=cap).expected(task)["tokens_per_correct"]) < 1e-12
print(f"✅ cap {cap}: accuracy {accs[cap]:.3f} at {tokens_per_correct:.1f} tokens per correct answer "
      f"(uncapped: {student.expected(task)['tokens_per_correct']:.1f}) — the budget is chosen by the data you keep")

# %% [markdown]
# ## Exercise 3.5 — how long should the weaker student think?
# The student with $q$ = 0.05 pays 0.01 per thinking token ($\text{reward} = \text{correct} - 0.01\,L$). Predict
# `L_star`, the real-valued optimum, from
#
# $$
# e_0\,\bigl(-\ln(1 - q)\bigr)\,(1 - q)^L = c,
# $$
#
# Also predict `gain`: its expected reward at `round(L_star)` minus its expected reward at the teacher's mean length,
# which it copied. Use $\text{accuracy}(L) - 0.01\,L$ at the two lengths.

# %% exercise
### BEGIN SOLUTION
e0, qs, c = 0.8, 0.05, 0.01
L_star = np.log(c / (e0 * -np.log(1 - qs))) / np.log(1 - qs)
reward = lambda L_: 1 - e0 * (1 - qs) ** L_ - c * L_
gain = float(reward(round(L_star)) - reward(student.expected(task)["length"]))
### END SOLUTION

# %% check
ref_reward = lambda L_: float(task.accuracy(L_, q=0.05)) - 0.01 * L_
ref_gain = ref_reward(round(task.optimal_length(0.01, q=0.05))) - ref_reward(student.expected(task)["length"])
assert abs(L_star - task.optimal_length(0.01, q=0.05)) < 1e-9 and abs(gain - ref_gain) < 1e-9 and gain > 0
print(f"✅ L* = {L_star:.1f} tokens for the weaker student against the {student.expected(task)['length']:.1f} it copied (+{gain:.3f} reward): "
      "distillation copies the teacher's procedure, not the one the student needs")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We distil reasoning by SFT on the traces of a thinking teacher. That copies the
# procedure: the format, the steps and how long to think. Thus the student arrives with the teacher's thinking-length
# distribution. We calculate the serving capacity of the student for that tail, not for the tail of a chat model.
#
# "We apply filters to the traces on purpose. We keep only the traces that the verifier accepts as correct (rejection sampling). If we need a
# lower-cost student, we also apply a length cap. The cap decreases the tokens per correct answer, but it also decreases
# the accuracy, and it discards teacher tokens that we paid for.
#
# "At equal samples, distillation is better than RL on a small model. The reason is that a trace carries the full
# behaviour, and a reward carries one bit. But distillation cannot transfer knowledge that the student does not have. A
# weaker student copies a length that is incorrect for it. Thus we evaluate at the student's own best budget, and we
# think about a short RL stage after distillation."
#
# **Drill questions**
# 1. *Our distilled 1.5B thinks as long as the 32B teacher, and the GPU bill doubled. Why?* SFT on traces copies the
#    length distribution of the teacher. Put a cap on the trace length in the data. Or add a budget and a length
#    penalty in a short RL stage.
# 2. *Keep only correct traces: is there a side effect?* When it helps to think, correct traces are longer. Thus the
#    student thinks longer. Also, the data loses the hard problems that have no correct trace (a coverage bias).
# 3. *When is RL on the student better than distillation?* When no stronger teacher exists for the task. Or after the
#    student has the teacher's behaviour and needs its own behaviour. RL on the student fits its own $q$ (here $L^*$ =
#    27.5, not the copied 19.9). R1's authors note that it is possible for an RL stage after distillation to add much
#    more.
