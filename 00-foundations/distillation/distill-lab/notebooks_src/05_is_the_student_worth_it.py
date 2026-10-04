# %% [markdown]
# # 05 · Is the student worth it? — agreement, the capability gap, cost per correct answer and break-even
#
# **Tier:** T0 (default). The serving costs come from a roofline model (**predicted**: bounds at ideal bandwidth, 01
# PRIMER §3 and §8). The accuracy comes from this lab's fake teacher and fake student (**simulated**). The agreement
# comes from the recorded tiny run of notebook 01 (illustrative).
#
# T1: set `DISTILLAB_URL` to a vLLM server. Then the throughput cell measures output tokens per second from its
# `/metrics`. The lm-eval tool gives real accuracies (the commands are in the section "Worked example: measured
# throughput at T1").
#
# ## The one-minute version
#
# * **Two kinds of number, two questions** (PRIMER §8 "Measuring a student"). *Agreement* (KL, top-1 agreement)
#   asks if the student copies the teacher. *Task accuracy with an interval* asks if the student solves the task.
#   The two numbers can disagree. A student can beat its teacher and agree less with it.
# * **The capability gap hides in the tail.** The teacher and the student can look near on an average over easy and
#   hard problems, while they are far apart on the hardest slice. Report the accuracy per difficulty with Wilson
#   intervals. Use paired comparisons.
# * **The economics** (PRIMER §9 "The economics of a student"). The student costs less per token, because each GPU
#   holds many more of its sequences. This is because its weights and its KV per sequence are both smaller. Calculate
#   the cost of the teacher on the GPUs that you really give it (two H100s for a 32B). Do not use one card that it
#   barely fits on. Distillation has a fixed cost: the teacher's tokens plus $6 \cdot N \cdot D$ of training. Break-even is that fixed
#   cost divided by the decrease in cost per token.
# * **The cascade:** send easy queries to the student and hard queries to the teacher. Measure the cascade by the
#   cost per *correct* answer, not by the cost per token.

# %%
import math, statistics
from distillab import agreement as A, cost as C, data as D, env, metrics as M, teacher as TE
from distillab.client import Client
from distillab.fakeserver import FakeTeacher
from distillab.report import table
from distillab.tinylm.curves import COLS, load_recorded, summary_rows

print(env.describe())
REC = load_recorded()
print(table(summary_rows(REC), ["model", "accuracy", "agree", "kl", "rkl"], f"[{REC['source']}] notebook 01's students"))

# %% [markdown]
# The most accurate tiny student (`seqkd`) is not the one that agrees most with the teacher (`gkd`). Its training
# used the teacher's *verified* outputs. Thus it learned a better-than-teacher habit, and it diverges where the
# teacher is incorrect. Agreement is the correct metric for a speculative draft (notebook 04). It is also the correct
# metric for a compressed replacement that must behave identically, as with quantization's accuracy checks
# (quantization PRIMER §8). It is the incorrect metric when the goal is the task.
#
# ## Worked example: teacher and student on the same eval problems
#
# The fake teacher is a substitute for `Qwen/Qwen2.5-1.5B-Instruct`. The fake student is a substitute for
# `Qwen/Qwen2.5-0.5B-Instruct`. Both answer the same 200 held-out problems with greedy decoding. Both are
# **simulated**: by construction, the student makes more errors on harder problems. Here, the method is the important
# part. At T1, run lm-eval or this cell against the real models.

# %%
EVAL = D.make_set(200, seed=11)
servers = {name: FakeTeacher(name) for name in ("teacher", "student")}
clients = {name: Client(s.start(), s.model) for name, s in servers.items()}
OUT = {name: [c[0] for c in cl.chat_many([p.messages() for p in EVAL], temperature=0.0, max_tokens=512)] for name, cl in clients.items()}
for s in servers.values():
    s.stop()
RECORDS = [{"difficulty": p.difficulty, "kind": p.kind, "teacher": D.verify(p, t.content), "student": D.verify(p, s.content),
            "teacher tokens": t.completion_tokens, "student tokens": s.completion_tokens}
           for p, t, s in zip(EVAL, OUT["teacher"], OUT["student"])]
PAIRED = A.paired([r["teacher"] for r in RECORDS], [r["student"] for r in RECORDS])
print(table([{k: round(v, 3) if isinstance(v, float) else v for k, v in PAIRED.items()}], title="SIMULATED: paired comparison on 200 problems"))

# %% [markdown]
# ## Exercise 5.1 — a Wilson interval
#
# Write `wilson(passes, n, z=1.96)`. The centre is $(p + z^2/2n) / (1 + z^2/n)$. The half-width is
# $z\,\sqrt{p(1 - p)/n + z^2/4n^2} / (1 + z^2/n)$. Clip the interval to [0, 1]. Return `(0, 1)` for $n$ = 0.
#
# The interval stays useful at 0/$n$ and $n$/$n$, where $p \pm 1.96\,\mathrm{SE}$ becomes a single point. `memory-core`
# and the eval gates of the 07 agent lab use this interval.

# %% exercise
def wilson(passes: int, n: int, z: float = 1.96) -> tuple:
    ### BEGIN SOLUTION
    if n == 0:
        return (0.0, 1.0)
    p, z2 = passes / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
    ### END SOLUTION

# %% check
for (k, n), want in {(30, 60): (0.3773, 0.6227), (170, 200): (0.7939, 0.8929), (0, 20): (0.0, 0.1611)}.items():
    assert all(abs(a - b) < 1e-4 for a, b in zip(wilson(k, n), want)), (k, n, wilson(k, n))
for k, n in ((0, 0), (5, 5), (3, 10), (47, 50)):
    assert all(abs(a - b) < 1e-12 for a, b in zip(wilson(k, n), A.wilson_interval(k, n)))
lo, hi = wilson(sum(r["student"] for r in RECORDS), len(RECORDS))
print(f"✅ student accuracy {PAIRED['student']:.3f}, 95% interval {lo:.3f}-{hi:.3f} on {len(RECORDS)} problems")

# %% [markdown]
# ## Exercise 5.2 — where the gap hides
#
# Write `gap_by(records, key)`. It returns a dict from each value of `key` (such as `"difficulty"`) to the gap
# $\text{teacher accuracy} - \text{student accuracy}$ on those records. Then compare the overall gap with the gap on
# the hardest slice.

# %% exercise
def gap_by(records: list, key: str) -> dict:
    ### BEGIN SOLUTION
    out = {}
    for v in sorted({r[key] for r in records}):
        rs = [r for r in records if r[key] == v]
        out[v] = (sum(r["teacher"] for r in rs) - sum(r["student"] for r in rs)) / len(rs)
    return out
    ### END SOLUTION

# %% check
g = gap_by(RECORDS, "difficulty")
ref = {r["difficulty"]: r["gap"] for r in A.gap_table(RECORDS, "difficulty")}
assert all(abs(g[d] - ref[d]) < 1e-3 for d in ref)
overall = PAIRED["teacher"] - PAIRED["student"]
print(table(A.gap_table(RECORDS, "difficulty"), title="SIMULATED: accuracy by difficulty, with 95% Wilson intervals"))
print(f"✅ overall gap {overall:.3f}; at difficulty 4 it is {g[4]:.3f}: " +
      ("the average understates the tail" if g[4] > overall else "in this sample the tail is not worse than the average"))

# %% [markdown]
# That is where a distilled student fails in production: the hard, rare requests. An offline check of agreement or
# of average accuracy passes, and the complaints come from the tail. Put a gate on the hard slice, with its own
# interval. Expect wide intervals there: 50 problems at 60% is ±13 points.
#
# ## Worked example: what each model costs to serve
#
# This example calculates the roofline bound for two pairs. The first is the fact sheet's pair: a 32B teacher and a
# 1.5B student at 2K context, under a 30 ms inter-token latency. The second is the T4-sized pair. `C.serving` selects
# the largest batch whose decode step fits the latency budget and the memory. Then it calculates the price with
# `roofline.cost`'s formula (the tests compare both against layer 01's core). The cell calculates the cost of the 32B
# on one H100 and on two (`n_gpus=2`: ideal tensor parallelism, with no count for the all-reduces).

# %%
H100, T4 = C.GPUS["H100"], C.GPUS["T4"]
T32, S15 = C.shape("qwen2.5-32b-instruct"), C.shape("qwen2.5-1.5b-instruct")
TEACHER_1GPU = C.serving(T32, H100, context=2048, itl_s=0.030)
TEACHER = C.serving(T32, H100, context=2048, itl_s=0.030, n_gpus=2)          # the teacher as you would run it
STUDENT = C.serving(S15, H100, context=2048, itl_s=0.030)
T4_TEACHER = C.serving(S15, T4, context=2048, itl_s=0.030, precision="fp16")
T4_STUDENT = C.serving(C.shape("qwen2.5-0.5b-instruct"), T4, context=2048, itl_s=0.030, precision="fp16")
print(table([TEACHER_1GPU, TEACHER, STUDENT, T4_TEACHER, T4_STUDENT], ["model", "gpu", "batch", "step ms", "tok/s", "$/M"],
            "PREDICTED (roofline bound, 100% utilisation; H100 $11/GPU-h, T4 $0.35/GPU-h, COMPUTE.md 2026-09-26, verify)"))
print(f"32B -> 1.5B: {TEACHER['$/M'] / STUDENT['$/M']:.0f}x cheaper per output token against the teacher on two H100s, "
      f"{TEACHER_1GPU['$/M'] / STUDENT['$/M']:.0f}x against one (weights {T32.params() / S15.params():.0f}x smaller, "
      f"KV per token {T32.kv_bytes_per_token() / S15.kv_bytes_per_token():.0f}x smaller)")

# %% [markdown]
# On one H100, the 32B's weights leave space for only 12 sequences of 2K context. Thus it cannot batch, and the cost
# of the student looks 96× lower. In practice, nobody serves a 32B that way. On two H100s, it runs 146 sequences, and
# the cost of the student is 16× lower. This is the number to use.
#
# Both runs fill HBM. Thus the cost per token depends on how many sequences each GPU holds. The student's KV per
# sequence is 9× smaller, and its weights leave more of each card free. The rest of this notebook calculates the
# price of the teacher on two H100s.
#
# ## Exercise 5.3 — the fixed cost and break-even
#
# The fact sheet's worked case starts with 100k prompts × 2,000 teacher tokens at \$9.00 per million output tokens.
# Then, SFT of the 1.5B student on those 2e8 tokens (6·N·D FLOPs) runs on an H100 at 40% MFU and \$11/GPU-h. The
# bundled config gives the exact parameter count of the student.
#
# Write `break_even_days(fixed, teacher_per_m, student_per_m, tokens_per_day)`. It returns the days of serving after
# which the student has paid for itself. Each million tokens that the student serves instead of the teacher saves
# `teacher_per_m − student_per_m` dollars. The table also shows the result when the self-hosted teacher generates the
# same data.

# %% exercise
def break_even_days(fixed: float, teacher_per_m: float, student_per_m: float, tokens_per_day: float) -> float:
    ### BEGIN SOLUTION
    saving = teacher_per_m - student_per_m
    return math.inf if saving <= 0 else fixed / saving * 1e6 / tokens_per_day
    ### END SOLUTION

# %% check
FIXED = C.fixed_cost(teacher_tokens=2e8, teacher_price_per_m=9.0, student_params=S15.params(), train_tokens=2e8, gpu=H100, mfu=0.4)
OWN = C.fixed_cost(teacher_tokens=2e8, teacher_price_per_m=TEACHER["$/M"], student_params=S15.params(), train_tokens=2e8, gpu=H100, mfu=0.4)
assert abs(FIXED["teacher generation $"] - 1800) < 1e-9 and abs(FIXED["training $"] - 14.30) < 5e-3   # PRIMER §3: $14.30
VOLUMES = (1e6, 1e7, 1e8, 1e9)
for tpd in VOLUMES:
    want = C.break_even(FIXED["total $"], TEACHER["$/M"], STUDENT["$/M"], tpd)["days"]
    assert abs(break_even_days(FIXED["total $"], TEACHER["$/M"], STUDENT["$/M"], tpd) - want) < 1e-9
assert break_even_days(10, 1, 2, 1e6) == math.inf
DAYS = {tpd: break_even_days(FIXED["total $"], TEACHER["$/M"], STUDENT["$/M"], tpd) for tpd in VOLUMES}
print(table([{"tokens per day": f"{tpd:,.0f}", "break-even days, API data": round(DAYS[tpd], 2),
              "break-even days, self-hosted data": round(break_even_days(OWN["total $"], TEACHER["$/M"], STUDENT["$/M"], tpd), 2)}
             for tpd in VOLUMES],
            title=f"PREDICTED: fixed cost ${FIXED['total $']:,.2f} with API data (teacher tokens ${FIXED['teacher generation $']:,.0f}, "
                  f"training ${FIXED['training $']:.2f}); ${OWN['total $']:,.2f} with data from the self-hosted teacher"))
print(f"✅ the teacher's tokens, not the student's training, dominate the fixed cost. With API-bought data, break-even "
      f"against the 32B on two H100s is {DAYS[1e9]:.1f} days at a billion tokens a day, {DAYS[1e8]:.0f} at 100 million, "
      f"{DAYS[1e7]:.0f} at 10 million and {DAYS[1e6]:,.0f} at a million, longer than most models stay in service; "
      f"generating the data on the self-hosted teacher cuts every row {FIXED['total $'] / OWN['total $']:.1f}x")

# %% [markdown]
# ## Exercise 5.4 — a cascade by difficulty
#
# A router sends each query of difficulty `>= route_from` to the teacher, and the other queries to the student. The
# router decides before either model runs. Use the per-query costs in the next cell (the simulated output lengths,
# priced at the T4 rows' $/M). Also use the accuracy per difficulty of each model from `RECORDS`. Write
# `cascade(records, route_from)`. It returns `(cost_per_query, accuracy)`.
#
# Then select `BEST`. Examine the `route_from` values in 1…5 (5 = student only) whose accuracy is at least `FLOOR`.
# `BEST` is the value among them with the lowest cost per correct answer.

# %%
PRICE = {"teacher": T4_TEACHER["$/M"] / 1e6, "student": T4_STUDENT["$/M"] / 1e6}  # $ per output token (T4 rows)
FLOOR = 0.90                                                                        # the accuracy the product needs
print({k: f"${v * 1e6:.4f}/M" for k, v in PRICE.items()}, "accuracy floor", FLOOR)

# %% exercise
def cascade(records: list, route_from: int) -> tuple:
    ### BEGIN SOLUTION
    cost = correct = 0.0
    for r in records:
        who = "teacher" if r["difficulty"] >= route_from else "student"
        cost += r[f"{who} tokens"] * PRICE[who]
        correct += r[who]
    return cost / len(records), correct / len(records)
    ### END SOLUTION

BEST = None
### BEGIN SOLUTION
ok = [d for d in range(1, 6) if cascade(RECORDS, d)[1] >= FLOOR]
BEST = min(ok, key=lambda d: cascade(RECORDS, d)[0] / cascade(RECORDS, d)[1]) if ok else 1
### END SOLUTION

# %% check
rows = []
for d in range(1, 6):
    cst, acc = cascade(RECORDS, d)
    ref_cost = statistics.fmean(r["teacher tokens"] * PRICE["teacher"] if r["difficulty"] >= d else r["student tokens"] * PRICE["student"] for r in RECORDS)
    ref_acc = statistics.fmean(r["teacher"] if r["difficulty"] >= d else r["student"] for r in RECORDS)
    assert abs(cst - ref_cost) < 1e-15 and abs(acc - ref_acc) < 1e-12
    rows.append({"teacher for difficulty >=": d if d < 5 else "never", "accuracy": round(acc, 3),
                 "$ per 1M queries": round(cst * 1e6, 2), "$ per 1M correct": round(cst / acc * 1e6, 2)})
meets = [d for d in range(1, 6) if cascade(RECORDS, d)[1] >= FLOOR]
assert BEST == (min(meets, key=lambda d: cascade(RECORDS, d)[0] / cascade(RECORDS, d)[1]) if meets else 1)
cheapest = min(range(1, 6), key=lambda d: cascade(RECORDS, d)[0] / cascade(RECORDS, d)[1])
print(table(rows, title="SIMULATED accuracy x PREDICTED cost: cascade by difficulty"))
print(f"✅ with an accuracy floor of {FLOOR}: teacher from difficulty {BEST if BEST < 5 else 'never'}. Without the floor the "
      f"cheapest per correct answer is {'student only' if cheapest == 5 else f'teacher from {cheapest}'}: when failures cost "
      "nothing extra, cost per correct answer alone favours the cheap model, so state the accuracy the product needs first")

# %% [markdown]
# The router needs a difficulty signal or a confidence signal that it can calculate before a model answers (06
# scaling-admission-cost: routing by cost). In real traffic, the router does not know the difficulty. It has one of
# these proxies:
#
# * the prompt length
# * a classifier
# * the student's own confidence (`C.cascade(..., student_first=True)`)
# * the user's tier
#
# With the student's confidence, the router sends the query to the teacher after the student answers. This pays for
# both answers on the routed share.
#
# A difficulty classifier that is 90% accurate changes this table. Thus, measure the router, not the oracle.
#
# ## Worked example: measured throughput at T1
#
# When `DISTILLAB_URL` points at a real vLLM, the next cell sends a burst of the eval problems. Then it calculates the
# price of the output tokens that the engine counted (`vllm:generation_tokens_total`) between two scrapes. It uses a
# GPU price that you give. The fake teacher answers immediately. Thus its throughput tells you nothing, and the
# cell skips it.

# %%
if env.server_url() and not env.is_simulated(env.server_url(), env.auth_headers()):
    import time
    tgt = env.connect()
    cl = Client(tgt.url, tgt.model, tgt.headers)
    before = M.scrape(tgt.url, tgt.headers)
    t0 = time.time()
    cl.chat_many([p.messages() for p in EVAL[:64]], workers=32, temperature=0.7, max_tokens=512)
    after = M.scrape(tgt.url, tgt.headers)
    live = C.from_metrics(before, after, price_per_gpu_hour=0.35, seconds=time.time() - t0)
    print(table([{k: round(v, 4) if isinstance(v, float) else v for k, v in live.items()}], title=f"MEASURED: {tgt.model}"))
else:
    print("T0: no real server in DISTILLAB_URL; the costs above are roofline predictions. At T1:\n"
          "  vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --port 8000 &   then   export DISTILLAB_URL=http://127.0.0.1:8000\n"
          "  and for accuracy:  " + " ".join(A.lm_eval_command("Qwen/Qwen2.5-0.5B-Instruct", "gsm8k")))

# %% [markdown]
# ## The decision, in one table
#
# Distillation is one of five ways to get a lower-cost model that does your task. It is the correct way when you have
# volume, a verifiable or judgeable task, and a teacher that you have permission to train on. PRIMER §1 has a dated
# paragraph about licences and terms of service, with a verify mark. That paragraph is not legal advice.

# %%
print(table([
    {"option": "prompt the big model better", "fixed cost": "hours", "per-token cost": "unchanged", "when": "low volume; the task is still moving"},
    {"option": "fine-tune a small model on labels", "fixed cost": "labels + GPU-hours", "per-token cost": "small model's", "when": "labels exist and show the behaviour you want"},
    {"option": "distil (SeqKD / KD / GKD)", "fixed cost": "teacher tokens + GPU-hours", "per-token cost": "small model's", "when": "volume, a verifier or judge, a teacher you may train on"},
    {"option": "quantize the big model", "fixed cost": "calibration + eval", "per-token cost": "~2-4x lower", "when": "same model, less memory; quantization PRIMER §10"},
    {"option": "a smaller off-the-shelf model", "fixed cost": "eval only", "per-token cost": "small model's", "when": "its accuracy on your slice is already enough"}],
    title="Cheaper inference: the options (PRIMER §9)"))

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We measured the student in two ways. Agreement tells us how much it copies the teacher.
# Task accuracy with Wilson intervals, per difficulty, tells us if it solves our problems. The averages were near, and
# the gap was on the hardest slice. Thus we put a gate on that slice.
#
# "Two H100s are the smallest deployment that leaves the 32B space to batch. Against the 32B on two H100s, the 1.5B
# student costs approximately a sixteenth as much per output token, by the roofline bound. On one H100, the cost of
# the student looks like a ninety-sixth. But this is only because the teacher's KV fills that card at a dozen
# sequences.
#
# "The teacher's generated tokens (\$1,800 for 2e8 at \$9 per million) are most of the fixed cost, not the training
# (\$14). Thus, with bought data, the break-even is approximately two days at a billion tokens a day. It is
# approximately three weeks at a hundred million, and years at a million. When our own teacher generates the data, the
# break-even decreases approximately ninefold.
#
# "We serve a cascade. The student answers by default, and the teacher answers the requests that the router marks as
# hard. We measure the cascade by the cost per correct answer."
#
# **Drill 1.** *Offline agreement with the teacher is 97%, but users say that the student became worse on hard
# tickets. What do you examine?* Examine the task accuracy on a hard slice with its own interval, paired against the
# teacher on the same items. Agreement is an average over easy positions. Then decide if you route that
# slice to the teacher.
#
# **Drill 2.** *The student is 21× smaller. Against the teacher on one H100, its cost per token is 96× lower. Against
# the teacher on two H100s, it is 16× lower. Which number do you quote, and where does the gap come from?* Quote 16×.
#
# Decode streams the weights *and* the KV, and both runs fill HBM. Thus the cost per token depends on the number of
# sequences that each GPU holds. One H100 leaves the 32B 6.5 GB for KV, that is 12 sequences. Two H100s give it 146.
# The student's KV per sequence is 9× smaller, and its weights leave more of the card free. Thus it holds 1,173
# sequences per GPU against the teacher's 73.
#
# **Drill 3.** *When does distillation not pay?* It does not pay at low volume, because break-even is the fixed cost
# divided by the decrease in cost per token. With bought data, break-even is years at a million tokens a day.
#
# It also does not pay when the task needs knowledge that the student does not have. That gap in knowledge shows as a
# tail that the cascade must cover.
# It also does not pay when a quantized teacher or a smaller off-the-shelf model already meets the accuracy bar.
