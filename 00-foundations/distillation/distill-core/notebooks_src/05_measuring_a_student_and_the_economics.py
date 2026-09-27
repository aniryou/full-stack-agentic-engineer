# %% [markdown]
# # 05 · Measuring a student, and whether it is worth it
#
# **Tier:** T0 — CPU only, numpy, no network, under a minute. Agreement and accuracy are measured on the toy
# models; serving costs come from a roofline decode step (an ideal bound, `distillcore.cost`, the same
# arithmetic as `roofline.llm` and `roofline.cost`), not from hardware. Measured throughput and a real
# agreement report are `distill-lab` notebook `05_is_the_student_worth_it` (T1).
#
# ## The one-minute version
# Measure a student two ways.
#
# - **Agreement** with the teacher — mean $\mathrm{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$, top-1
#   agreement, top-k overlap — needs no labels (the same definitions quantization §8 uses for a quantized model).
# - **Task accuracy** is what users feel, and it needs an interval (Wilson).
# - Report both **per slice**: a student's gap hides in rare inputs and long outputs; and a student can beat its
#   teacher on the task while agreeing with it less.
#
# Then the economics: a 1.5B student of a 32B teacher streams a twentieth of the weights and a ninth of the KV per
# token, so under the same ITL budget it runs 16× the batch per GPU of the teacher on two H100s, at ~16× lower cost
# per token on the roofline (96× against a teacher squeezed onto one H100, which is not a fair baseline). Against that
# saving stands a one-off bill dominated by the teacher's tokens; break-even is days at a large daily volume and
# months at a small one; a **cascade** sits in between. Primer: `../../PRIMER.md` §8, §9 (and §1 for the roofline
# argument).

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
# Items: continue a prompt greedily for $n$ tokens; correct if every token follows the rule. "Common" prompts are
# the 20 the students were trained from; "rare" are the other 101. Short ($n$ = 2) and long ($n$ = 8) outputs.

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
# Every student is perfect on the common prompts — the slice a quick eval would use. The gap is in the rare
# ones, and it grows with output length: the 8-unit student solves 85.1% of rare short items and 52.5% of rare
# long ones (it is wrong on 6.6% of contexts, and a long output visits many). Twenty items give an interval
# 16 points wide even at 20/20; a slice needs more items than its share of traffic suggests.
#
# ## Worked example 3 — a student that beats its teacher
# A weak teacher (64 units, trained on only 605 sampled tokens) is right on 95.0% of contexts. Its samples,
# filtered by the verifier, train one student; its unfiltered samples another.

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
# The verifier kept the teacher's right answers and dropped its mistakes, so that student is right on 97.5% of
# contexts — more than its teacher — while agreeing with it *less* (KL 1.59 against 0.16). If the release gate
# had been "agree with the teacher", it would have shipped the worse student.
#
# ## Worked example 4 — what the student saves in serving
# Decode on H100s at 2K context under a 30 ms ITL budget: the largest batch that meets it and fits in HBM, its
# step time and throughput, and \$/M output tokens at \$11/GPU-hour on demand (verify). The 32B is costed twice: on
# one H100, where its 65.5 GB of weights leave 6.5 GB for KV, and split over two (`K.tp_group`: ideal tensor
# parallelism, all-reduces not counted). A roofline bound — the costs are lower bounds; carry the like-for-like
# ratio.

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
# On one GPU the teacher runs 12 sequences and the student looks 96× cheaper; that compares the student with a
# deployment nobody would choose. On two GPUs the teacher runs 146 and the ratio is 16×: the number to carry.
# Everything below prices the teacher on two H100s.

# %% [markdown]
# ## Worked example 5 — the fixed cost, and break-even
# 100,000 prompts × one 2,000-token teacher completion = 2 × 10⁸ tokens. Buy them from an API at $9/M output
# (the 06 scaling lab's Gemini 3.5 Flash price, dated there, verify) or generate them on the self-hosted 32B at
# its roofline cost; then SFT the 1.5B student for one epoch at 6·N·D FLOPs on an H100 at 40% MFU.

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
# The teacher's tokens are 93% of the bill self-hosted and 99% through the API; the student's training is an hour
# of one GPU. That is why the data budget (§3) — prompts × samples × tokens, and how many the verifier discards —
# is the number to argue about. With self-hosted data, break-even comes at roughly the volume the teacher wrote
# for the student (2 × 10⁸ tokens here), whatever the teacher costs; API-bought data takes ten times longer. And
# the break-even omits the engineering, the evals and the ongoing cost of a second model to maintain.
#
# ## Worked example 6 — the cascade
# Route by difficulty instead of replacing the teacher. Per request (500 output tokens at the costs above) and
# with illustrative accuracies — easy requests (70%) student 0.95, teacher 0.97; hard ones (30%) student 0.30,
# teacher 0.85, the split rl-and-thinking-models §7 uses — compare cost per correct answer.

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
# clipped to [0, 1]; return (0, 1) when $n$ = 0.

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
# Mean over positions of $\lvert \text{top-}k(\text{ref}) \cap \text{top-}k(\text{test}) \rvert / k$ — agreement on
# the plausible set, not just the favourite.

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
# For `m = K.SHAPES["qwen2.5-0.5b"]` on the H100 at 2K context and a 30 ms ITL: find the largest batch whose
# decode step (`K.decode_step`) is ≤ 30 ms and that fits in memory (`K.max_batch`), then its tokens/s and
# \$/M at \$11/GPU-hour. Set `batch`, `tok_s`, `usd_per_m`.

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
# Distilling a 0.5B student from the 32B teacher costs the same one-off `fixed` as above. At 20M output tokens
# a day, after how many `days` does it pay for itself against serving the 32B on two H100s? (Use `T_PER_M` and
# your `usd_per_m`.)

# %% exercise
### BEGIN SOLUTION
days = fixed / ((T_PER_M - usd_per_m) * 20e6 / 1e6)
### END SOLUTION

# %% check
assert abs(days - K.break_even(fixed, T_PER_M, usd_per_m, 20e6)["days"]) < 1e-9
print(f"✅ {days:.1f} days at 20M tokens/day — before the cost of evals, of a second model to keep current, and of any quality gap")

# %% [markdown]
# ## Exercise 5.5 — choose the gate
# A student-first cascade with the costs and accuracies of worked example 6. Among the gates `gates`
# (catch rate on hard requests, false-alarm rate on easy ones), pick `best_gate`: the cheapest per correct
# answer among those whose overall accuracy is at least 0.90.

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
# **The two-minute version.** "We gate the student on two measurements. Agreement with the teacher — KL, top-1, top-k
# on the same positions — is cheap, needs no labels and catches regressions early; task accuracy with Wilson
# intervals, per slice, is the release gate, because the gap hides in rare inputs and long outputs and a quick eval of
# common cases will say 100%. We do not gate on agreement alone: a verifier-filtered student can beat its teacher
# while agreeing less.
#
# "The case for the student is the roofline: at a 2K context and a 30 ms ITL a 1.5B serves 16× the batch per GPU of a
# 32B on two H100s, ~16× cheaper per token on the bound (the 96× against a 32B squeezed onto one H100 is not a fair
# baseline). The one-off cost is mostly teacher tokens; with self-hosted data, break-even is days at 50M tokens a day
# and weeks at 5M.
#
# "If the student is only good enough on easy requests, we cascade: student first, a gate escalates the hard ones, and
# we tune the gate's recall on cost per correct answer."
#
# **Drill questions**
# 1. *The student agrees with the teacher on 99% of tokens. Ship it?* — Not on that alone: agreement is averaged
#    over common positions. Check task accuracy with intervals on the rare and long slices; that is where it fails.
# 2. *Where does a distillation budget go?* — The teacher's tokens (here \$178 of \$192 self-hosted, \$1,800 of \$1,814
#    via an API); training the student was 1.3 GPU-hours. Cut samples the verifier will reject before generating more.
# 3. *Distil, or route easy traffic to a cheaper off-the-shelf model?* — If an off-the-shelf model meets the easy
#    slice, routing costs no training; distil when no such model exists for your task or the volume makes the
#    per-token saving dominate (primer §9's decision table).
