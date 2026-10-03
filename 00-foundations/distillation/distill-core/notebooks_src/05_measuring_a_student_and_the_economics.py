# %% [markdown]
# # 05 · Measuring a student, and whether it is worth it
#
# **Tier:** T0, with only a CPU and numpy, no network, and less than a minute of run time. The notebook measures
# agreement and accuracy on the toy models. The serving costs come from a roofline decode step, not from hardware. This
# step is an ideal bound (`distillcore.cost`, the same arithmetic as `roofline.llm` and `roofline.cost`). Measured
# throughput and a real agreement report are in `distill-lab` notebook `05_is_the_student_worth_it` (T1).
#
# ## The one-minute version
# Measure a student in two ways.
#
# - **Agreement** with the teacher needs no labels. Its measures are the mean
#   $\mathrm{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$, the top-1 agreement and the top-k overlap. These are
#   the same definitions that quantization §8 uses for a quantized model.
# - **Task accuracy** is what users feel, and it needs an interval (Wilson).
# - Report the two **per slice**. The gap of a student hides in rare inputs and long outputs. Also, a student can be
#   better than its teacher on the task, but agree with it less.
#
# Then the economics. A 1.5B student of a 32B teacher streams a twentieth of the weights and a ninth of the KV per
# token. Thus, under the same ITL budget, it runs 16× the batch per GPU of the teacher on two H100s. On the roofline,
# its cost per token is ~16× lower. Against a teacher squeezed onto one H100, the ratio is 96×, but that is not a fair
# baseline.
#
# Against this decrease in cost stands a one-off bill, and the teacher's tokens are most of that bill. Break-even is
# days at a large daily volume and months at a small one. A **cascade** is an option between the teacher alone and the
# student alone. Primer: `../../PRIMER.md` §8, §9 (and §1 for the roofline argument).

# %%
import numpy as np

from distillcore import TinyLM, ModLang, cost as K, eval as E, losses as L, onpolicy as op, seqkd, train
from distillcore.tinylm import fit_language

lang = ModLang(11, 0.2)
C = lang.contexts()
teacher = fit_language(lang, 64)
o = lang.orbits()
prompts = np.array(o[0] + o[1])
greedy = seqkd.teacher_data(teacher, prompts, 1, 12, np.random.default_rng(1), T=0.0)
kd = TinyLM(11, 16, 8, seed=1)
op.gkd_train(kd, teacher, prompts, 12, 300, lam=0.0, beta=0.0, data=greedy)       # notebook 02's KD student
onp = kd.copy()
op.gkd_train(onp, teacher, prompts, 12, 300, lam=1.0, beta=0.0, data=greedy, seed=1)   # + on-policy
students = {"KD on teacher text": kd, "+ on-policy": onp, "8 units, KD on everything": fit_language(lang, 8)}

# %% [markdown]
# ## Worked example 1 — agreement with the teacher, over every context

# %%
zt = teacher.logits(C)
for name, s in students.items():
    zs = s.logits(C)
    print(f"{name:26s} KL(teacher ‖ student) {E.kl(zt, zs):6.3f}  top-1 agreement {E.argmax_agreement(zt, zs):.3f}  "
          f"top-3 overlap {E.topk_overlap(zt, zs, 3):.3f}")

# %% [markdown]
# ## Worked example 2 — the gap by slice, with intervals
# Each item: continue a prompt for $n$ tokens with greedy decode. The item is correct if every token obeys the rule.
# The "common" prompts are the 20 that the students trained from. The "rare" prompts are the other 101. The outputs
# are short ($n$ = 2) and long ($n$ = 8).

# %%
common = set(map(tuple, prompts))
rare = np.array([c for c in C if tuple(c) not in common])


def solved(m, ctx, n):
    return lang.verify(m.sample(ctx, n, None, 0.0))


for name, s in students.items():
    rows = E.capability_gap({f"{k}, n = {n}": (solved(teacher, x, n), solved(s, x, n))
                             for k, x in (("common", prompts), ("rare", rare)) for n in (2, 8)})
    print(name)
    for r in rows:
        lo, hi = r["student_ci"]
        print(f"   {r['slice']:14s} n = {r['n']:3d}  teacher {r['teacher']:.3f}  student {r['student']:.3f} "
              f"(95% CI {lo:.3f}–{hi:.3f})  gap {r['gap']:.3f}")

# %% [markdown]
# Every student is perfect on the common prompts. This is the slice that a fast eval uses. The gap is in the rare
# prompts, and it increases with the output length. The 8-unit student solves 85.1% of the rare short items and 52.5%
# of the rare long items. It is incorrect on 6.6% of the contexts, and a long output goes through many contexts.
#
# Twenty items give an interval 16 points wide, even at 20/20. A slice needs more items than its share of traffic
# suggests.
#
# ## Worked example 3 — a student that beats its teacher
# A weak teacher (64 units, trained on only 605 sampled tokens) is correct on 95.0% of contexts. One student trains on
# the samples of this teacher after the verifier filters them. Another student trains on the unfiltered samples of the
# same teacher.

# %%
ctx = C[np.random.default_rng(7).integers(0, 121, 605)]
y = lang.sample_next(ctx, np.random.default_rng(8))
weak = TinyLM(11, 64, 8, seed=2)
train(weak, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
data = seqkd.teacher_data(weak, C, 20, 1, np.random.default_rng(9))
filt, raw = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
seqkd.sft(filt, seqkd.keep_verified(lang, data), 600)
seqkd.sft(raw, data, 600)
zw = weak.logits(C)
for name, m in (("weak teacher", weak), ("student, verified samples", filt), ("student, all samples", raw)):
    agree = "" if m is weak else f"  KL(teacher ‖ student) {E.kl(zw, m.logits(C)):.3f}  top-1 agreement {E.argmax_agreement(zw, m.logits(C)):.3f}"
    print(f"{name:26s} rule accuracy {E.vs_truth(m, lang)['rule_acc']:.3f}{agree}")

# %% [markdown]
# The verifier kept the correct answers of the teacher and removed its mistakes. Thus that student is correct on 97.5%
# of the contexts, which is more than its teacher. But it agrees with the teacher *less* (KL 1.59 against 0.16). If the
# release gate is "agree with the teacher", the worse student passes the gate.
#
# ## Worked example 4 — what the student saves in serving
# This example decodes on H100s at 2K context under a 30 ms ITL budget. For each model, it finds the largest batch that
# meets the budget and fits in HBM. Then it gives the step time and the throughput of that batch, and the \$/M output
# tokens at \$11/GPU-hour on demand (verify). The example calculates the cost of the 32B two times. The first time is on
# one H100, where its 65.5 GB of weights leave 6.5 GB for KV. The second time, the example divides the 32B over two GPUs
# (`K.tp_group`: ideal tensor parallelism, with the all-reduces not in the count).
#
# Each cost is a roofline bound, thus the costs are lower bounds. Carry the like-for-like ratio.

# %%
H100 = K.GPUS["h100"]
serve = {}
for name, key, n in (("qwen2.5-32b", "qwen2.5-32b", 1), ("qwen2.5-32b", "qwen2.5-32b-tp2", 2),
                     ("qwen2.5-1.5b", "qwen2.5-1.5b", 1), ("qwen2.5-0.5b", "qwen2.5-0.5b", 1)):
    m = K.SHAPES[name]
    s = serve[key] = K.serving(m, H100, 11, 2048, 0.030, n_gpus=n)
    print(f"{key:16s} on {n} GPU{'s' if n > 1 else ' '}: {m.params() / 1e9:5.2f} B params, {m.kv_bytes_per_token():7,.0f} B KV/token: "
          f"batch {s['batch']:5d}, step {s['step_s'] * 1e3:5.2f} ms, {s['tok_s']:9,.0f} tok/s, ${s['usd_per_m']:.4f}/M")
T_PER_M = serve["qwen2.5-32b-tp2"]["usd_per_m"]                                     # the teacher as you would run it
for key in ("qwen2.5-32b", "qwen2.5-32b-tp2"):
    print(f"{key} / 1.5B student, cost per token: {serve[key]['usd_per_m'] / serve['qwen2.5-1.5b']['usd_per_m']:.0f}×")
print(f"at a 10 ms ITL the 32B cannot serve on one H100: batch 1 takes {K.decode_step(K.SHAPES['qwen2.5-32b'], H100, 1, 2048) * 1e3:.1f} ms")

# %% [markdown]
# On one GPU the teacher runs 12 sequences, and the student looks 96× lower in cost. But that compares the student with
# a deployment that nobody selects. On two GPUs the teacher runs 146, and the ratio is 16×. This is the number to carry.
# All the cells after this one use the cost of the teacher on two H100s.

# %% [markdown]
# ## Worked example 5 — the fixed cost, and break-even
# 100,000 prompts × one 2,000-token teacher completion = 2 × 10⁸ tokens. Buy them from an API at $9/M output, or
# generate them on the self-hosted 32B at its roofline cost. The API price is the Gemini 3.5 Flash price of the 06
# scaling lab, dated there (verify). Then train the 1.5B student with SFT for one epoch, at 6·N·D FLOPs on an H100 at
# 40% MFU.

# %%
for label, price in (("API teacher", 9.00), ("self-hosted 32B", T_PER_M)):
    f = K.fixed_cost(100_000, 1, 2000, price, K.SHAPES["qwen2.5-1.5b"].params(), H100, 11, mfu=0.4)
    print(f"{label:16s} teacher tokens ${f['generation_usd']:8,.2f} + training {f['gpu_hours']:.3f} GPU-h = ${f['train_usd']:.2f} "
          f"→ ${f['total_usd']:,.2f}")
    for per_day in (50e6, 5e6):
        b = K.break_even(f["total_usd"], T_PER_M, serve["qwen2.5-1.5b"]["usd_per_m"], per_day)
        print(f"    {per_day / 1e6:4.0f}M output tokens/day: saves ${b['saving_per_day']:,.2f}/day, pays back in {b['days']:.1f} days")
fixed = K.fixed_cost(100_000, 1, 2000, T_PER_M, K.SHAPES["qwen2.5-1.5b"].params(), H100, 11)["total_usd"]

# %% [markdown]
# The teacher's tokens are 93% of the bill when self-hosted and 99% through the API. The training of the student is an
# hour of one GPU. That is why the data budget (§3) is the number to argue about. The data budget is prompts × samples ×
# tokens, and the number of samples that the verifier discards.
#
# With self-hosted data, break-even comes at roughly the volume that the teacher wrote for the student (2 × 10⁸ tokens
# here). This is true at any cost of the teacher. API-bought data takes ten times longer. Also, the break-even omits the
# work of the engineers, the evals and the continuous cost to maintain a second model.
#
# ## Worked example 6 — the cascade
# Route by difficulty instead of a replacement of the teacher. Each request has 500 output tokens at the costs of worked
# example 4. The accuracies are illustrative:
#
# - Easy requests (70%): student 0.95, teacher 0.97.
# - Hard requests (30%): student 0.30, teacher 0.85.
#
# This is the split that rl-and-thinking-models §7 uses. Compare the cost per correct answer.

# %%
c_t, c_s = 500 * T_PER_M / 1e6, 500 * serve["qwen2.5-1.5b"]["usd_per_m"] / 1e6
acc_s, acc_t = (0.95, 0.30), (0.97, 0.85)
options = {"teacher only": K.cascade(c_s, c_t, acc_s, acc_t, 0.3, catch=1.0, false_alarm=1.0, student_first=False),
           "student only": K.cascade(c_s, c_t, acc_s, acc_t, 0.3, catch=0.0, false_alarm=0.0),
           "student first, gate 0.8 / 0.1": K.cascade(c_s, c_t, acc_s, acc_t, 0.3, catch=0.8, false_alarm=0.1),
           "perfect router up front": K.cascade(c_s, c_t, acc_s, acc_t, 0.3, catch=1.0, false_alarm=0.0, student_first=False)}
for name, r in options.items():
    print(f"{name:30s} to teacher {r['to_teacher']:.2f}  accuracy {r['accuracy']:.3f}  ${r['cost'] * 1e3:.3f} per 1,000 requests  "
          f"${r['cost_per_correct'] * 1e3:.3f} per 1,000 correct answers")

# %% [markdown]
# ## Exercise 5.1 — the Wilson interval
# Write `wilson(k, n, z=1.96)`:
#
# $$
# \text{centre } \frac{p + z^2/2n}{1 + z^2/n}, \qquad
# \text{half-width } \frac{z\,\sqrt{p(1 - p)/n + z^2/4n^2}}{1 + z^2/n},
# $$
#
# Clip the interval to [0, 1]. Return (0, 1) when $n$ = 0.

# %% exercise
def wilson(k, n, z=1.96):
    ### BEGIN SOLUTION
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
    ### END SOLUTION

# %% check
for k, n in ((30, 60), (170, 200), (0, 20), (20, 20), (9, 10), (0, 0)):
    assert np.allclose(wilson(k, n), E.wilson_interval(k, n))
print(f"✅ 20/20 proves only ≥ {wilson(20, 20)[0]:.3f}; 0/20 allows up to {wilson(0, 20)[1]:.3f} — p ± 1.96·SE would say [1, 1] and [0, 0]")

# %% [markdown]
# ## Exercise 5.2 — top-k overlap
# Calculate the mean over positions of $\lvert \text{top-}k(\text{ref}) \cap \text{top-}k(\text{test}) \rvert / k$.
# This is the agreement on the plausible set, not only on the most probable token.

# %% exercise
def topk(ref, test, k):
    ### BEGIN SOLUTION
    a, b = np.argsort(-ref, -1)[:, :k], np.argsort(-test, -1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))
    ### END SOLUTION

# %% check
for s in students.values():
    for k in (1, 3, 5):
        assert abs(topk(zt, s.logits(C), k) - E.topk_overlap(zt, s.logits(C), k)) < 1e-12
assert topk(np.array([[3.0, 2, 1, 0]]), np.array([[2.0, 3, 0, 1]]), 2) == 1.0
print("✅ top-1 can disagree while the top-3 sets agree — and vice versa; report the one that matches how you sample")

# %% [markdown]
# ## Exercise 5.3 — a student's cost per token from the roofline
# Use `m = K.SHAPES["qwen2.5-0.5b"]` on the H100 at 2K context and a 30 ms ITL. Find the largest batch that meets these
# two conditions:
#
# - Its decode step (`K.decode_step`) is ≤ 30 ms.
# - It fits in memory (`K.max_batch`).
#
# Then calculate its tokens/s and \$/M at \$11/GPU-hour. Set `batch`, `tok_s`, `usd_per_m`.

# %% exercise
m = K.SHAPES["qwen2.5-0.5b"]
### BEGIN SOLUTION
batch = max(b for b in range(1, K.max_batch(m, H100, 2048) + 1) if K.decode_step(m, H100, b, 2048) <= 0.030)
tok_s = batch / K.decode_step(m, H100, batch, 2048)
usd_per_m = 11 / (tok_s * 3600) * 1e6
### END SOLUTION

# %% check
ref = K.serving(m, H100, 11, 2048, 0.030)
assert batch == ref["batch"] and abs(tok_s - ref["tok_s"]) < 1e-6 and abs(usd_per_m - ref["usd_per_m"]) < 1e-12
print(f"✅ batch {batch} (memory-bound, not ITL-bound), {tok_s:,.0f} tok/s, ${usd_per_m:.4f}/M — "
      f"{T_PER_M / usd_per_m:.0f}× cheaper than the 32B teacher on two H100s on this bound")

# %% [markdown]
# ## Exercise 5.4 — break-even
# The distillation of a 0.5B student from the 32B teacher costs the same one-off `fixed` as in worked example 5. The
# volume is 20M output tokens a day. After how many `days` does the student pay for itself, against the cost to serve
# the 32B on two H100s? (Use `T_PER_M` and your `usd_per_m`.)

# %% exercise
### BEGIN SOLUTION
days = fixed / ((T_PER_M - usd_per_m) * 20e6 / 1e6)
### END SOLUTION

# %% check
assert abs(days - K.break_even(fixed, T_PER_M, usd_per_m, 20e6)["days"]) < 1e-9
print(f"✅ {days:.1f} days at 20M tokens/day — before the cost of evals, of a second model to keep current, and of any quality gap")

# %% [markdown]
# ## Exercise 5.5 — choose the gate
# Use a student-first cascade with the costs and accuracies of worked example 6. Each gate in `gates` is a pair: the
# catch rate on hard requests and the false-alarm rate on easy requests. Set `best_gate` to the gate with the lowest
# cost per correct answer. Select only from the gates whose overall accuracy is at least 0.90.

# %% exercise
gates = [(0.6, 0.05), (0.8, 0.1), (0.9, 0.2), (0.95, 0.4), (1.0, 1.0)]
### BEGIN SOLUTION
ok_gates = [g for g in gates if K.cascade(c_s, c_t, acc_s, acc_t, 0.3, *g)["accuracy"] >= 0.90]
best_gate = min(ok_gates, key=lambda g: K.cascade(c_s, c_t, acc_s, acc_t, 0.3, *g)["cost_per_correct"])
### END SOLUTION

# %% check
res = {g: K.cascade(c_s, c_t, acc_s, acc_t, 0.3, *g) for g in gates}
feasible = [g for g in gates if res[g]["accuracy"] >= 0.90]
assert best_gate in feasible and all(res[best_gate]["cost_per_correct"] <= res[g]["cost_per_correct"] for g in feasible)
print(f"✅ gate {best_gate}: accuracy {res[best_gate]['accuracy']:.3f} at ${res[best_gate]['cost_per_correct']:.6f} per correct answer, "
      f"{res[best_gate]['to_teacher']:.0%} of requests escalated — the gate's recall on hard requests is the product")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Our gate for the student has two measurements. The first, agreement with the teacher
# (KL, top-1 and top-k on the same positions), is low-cost, needs no labels and finds regressions early. The second is
# task accuracy with Wilson intervals, per slice, and it is the release gate. The reason is that the gap hides in rare
# inputs and long outputs, and a fast eval of common cases will say 100%. Agreement alone is not our gate, because a
# verifier-filtered student can be better than its teacher but agree with it less.
#
# "The case for the student is the roofline. At a 2K context and a 30 ms ITL, a 1.5B serves 16× the batch per GPU of a
# 32B on two H100s. On the bound, its cost per token is ~16× lower. The 96× against a 32B squeezed onto one H100 is not
# a fair baseline. The one-off cost is mostly teacher tokens. With self-hosted data, break-even is days at 50M tokens a
# day and weeks at 5M.
#
# "If the student is only sufficiently good on easy requests, we use a cascade. The student goes first, and a gate sends
# the hard requests on to the teacher. We adjust the recall of the gate to get the lowest cost per correct answer."
#
# **Drill questions**
# 1. *The student agrees with the teacher on 99% of tokens. Ship it?* Not on that alone. The agreement is an average
#    over common positions. Examine the task accuracy with intervals on the rare and long slices. That is where it fails.
# 2. *Where does a distillation budget go?* To the teacher's tokens (here \$178 of \$192 self-hosted, \$1,800 of
#    \$1,814 through an API). The training of the student was 1.3 GPU-hours. Before you generate more samples, remove
#    the samples that the verifier will reject.
# 3. *Distil, or route easy traffic to a lower-cost off-the-shelf model?* If an off-the-shelf model meets the easy
#    slice, the routing costs no training. Distil when no such model exists for your task. Also distil when the volume
#    makes the decrease in cost per token dominate (primer §9's decision table).
