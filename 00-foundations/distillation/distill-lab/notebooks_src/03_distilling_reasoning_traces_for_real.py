# %% [markdown]
# # 03 · Distilling reasoning traces: what a student inherits, and what a length cap costs
#
# **Tier:** T1: traces from `Qwen/Qwen3-1.7B` (thinking on) or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, served
# by vLLM with `--reasoning-parser`, filtered, then SFT of `Qwen/Qwen3-0.6B` (or `Qwen/Qwen2.5-0.5B-Instruct`) on
# a T4 (model ids verify). T0 (default): 160 bundled traces from this lab's fake thinking teacher (**simulated**,
# illustrative, in vLLM's response fields), and a tiny budget-aware distillation run with torch on a CPU (about
# 40 seconds; without torch that cell is skipped).
#
# ## The one-minute version
#
# * **A student trained on reasoning traces inherits the procedure and the length distribution of the traces you
#   keep.** The serving workload of a thinking model (rl-and-thinking-models PRIMER §7) arrives with them: long,
#   heavy-tailed outputs (PRIMER §5 "Distilling reasoning").
# * **Two filters set it.** The *verifier* keeps correct traces, which raises the accuracy the student imitates and
#   shifts the kept lengths toward whatever lengths were more often right. A *length cap* makes the student
#   cheaper to serve, but the problems that need long traces lose their examples first.
# * **Distillation beats RL on a small model when a strong teacher exists.** DeepSeek-R1-Distill-Qwen-32B scores
#   72.6 on AIME 2024 against 47.0 for RL on the same base; Qwen3's on-policy distillation used about a tenth of
#   RL's GPU hours (1,800 against 17,920, Table 21). Those numbers are rl-and-thinking-models PRIMER §5's and the
#   fact sheet's (verify); this notebook does not repeat them. What distillation cannot add is knowledge the
#   student lacks.
# * **Thinking tokens are output tokens,** so a trace dataset is billed at the output price.

# %%
import math, statistics
from distillab import agreement as A, data as D, env, traces as TR, teacher as TE
from distillab.client import Client
from distillab.fakeserver import FakeTeacher
from distillab.report import histogram, table

print(env.describe())
if env.server_url():
    target = env.connect()
    TRACES = TR.collect(Client(target.url, target.model, target.headers), D.make_set(40, seed=0, split="train"), n=4)
    LABEL = target.label
    target.stop()
else:
    TRACES = TR.load_bundled()
    LABEL = "ILLUSTRATIVE (bundled, simulated by the fake thinking teacher)"
print(f"[{LABEL}] {len(TRACES)} traces for {len({t.problem_id for t in TRACES})} problems")
print(TRACES[3].reasoning[:400], "\n...\n", TRACES[3].content)

# %% [markdown]
# ## Worked example: how long the teacher thinks
#
# Reasoning lengths come from `usage.completion_tokens_details.reasoning_tokens` (one request per trace, so the
# count is per trace). They are heavy-tailed: a few long traces hold a large share of the tokens.

# %%
print(table([TR.length_stats(TRACES, "all"), TR.length_stats([t for t in TRACES if t.correct], "correct"),
             TR.length_stats([t for t in TRACES if not t.correct], "wrong")], title=f"[{LABEL}] reasoning tokens per trace"))
print(histogram([t.reasoning_tokens for t in TRACES], bins=10, label="reasoning tokens (count of traces)"))
print(table(TR.accuracy_by_length(TRACES, [0, 50, 100, 200, 400, 1000]), title="Teacher accuracy by trace length"))

# %% [markdown]
# In the fake teacher, a longer trace holds more re-checks, and each re-check may catch a slip, so longer is
# more often right. In real models hard problems also take longer, which can make long traces *less* often right
# on average. Read accuracy by length within a difficulty before concluding that longer thinking helps.
#
# ## Exercise 3.1 — how much of the bill the tail holds
#
# Write `tail_share(traces, q)`: the share of all reasoning tokens that sits in the longest $\left(1 - q\right)$
# fraction of traces (at least one trace). For $q$ = 0.9 that is the top 10%. The bill, the student's KV cache and the
# request timeout are all set by this tail, not by the mean.

# %% exercise
def tail_share(traces: list, q: float = 0.9) -> float:
    ### BEGIN SOLUTION
    lens = sorted((t.reasoning_tokens for t in traces), reverse=True)
    k = max(1, round(len(lens) * (1 - q)))
    return sum(lens[:k]) / sum(lens)
    ### END SOLUTION

# %% check
toy = [TR.Trace("p", "arith", 1, "q", "q", "r", "c", True, n, n) for n in (10,) * 9 + (110,)]
assert abs(tail_share(toy, 0.9) - 0.55) < 1e-12 and abs(tail_share(toy, 0.0) - 1.0) < 1e-12
lens = sorted((t.reasoning_tokens for t in TRACES), reverse=True)
ref = sum(lens[: max(1, round(len(lens) * 0.1))]) / sum(lens)
assert abs(tail_share(TRACES, 0.9) - ref) < 1e-12
print(f"✅ [{LABEL}] the longest 10% of traces hold {tail_share(TRACES):.0%} of the reasoning tokens; "
      f"the longest 1% hold {tail_share(TRACES, 0.99):.0%}")

# %% [markdown]
# ## Worked example: the verifier and a length cap
#
# For each cap: the correct traces that survive, their mean length (what the student will inherit), and how many
# problems keep at least one example, over all problems and over the hardest difficulty.

# %%
CAPS = [None, 400, 200, 100, 60]
TRADE = TR.trade(TRACES, CAPS)
print(table(TRADE, title=f"[{LABEL}] correct traces kept under a reasoning-length cap"))

# %% [markdown]
# ## Exercise 3.2 — choose a cap
#
# You want the student to think as little as possible, provided at least 70% of the hardest problems keep an
# example. Write `choose_cap(rows, min_hard)`: among the rows of `TR.trade` whose `"hardest covered"` is at least
# `min_hard`, return the `"cap"` with the lowest `"mean reasoning tokens"`.

# %% exercise
def choose_cap(rows: list, min_hard: float = 0.7):
    ### BEGIN SOLUTION
    ok = [r for r in rows if r["hardest covered"] >= min_hard]
    return min(ok, key=lambda r: r["mean reasoning tokens"])["cap"]
    ### END SOLUTION

# %% check
toy_rows = [{"cap": "none", "hardest covered": 1.0, "mean reasoning tokens": 300}, {"cap": 200, "hardest covered": 0.8, "mean reasoning tokens": 120},
            {"cap": 100, "hardest covered": 0.3, "mean reasoning tokens": 60}]
assert choose_cap(toy_rows, 0.7) == 200 and choose_cap(toy_rows, 0.9) == "none" and choose_cap(toy_rows, 0.2) == 100
pick = choose_cap(TRADE, 0.7)
row = next(r for r in TRADE if r["cap"] == pick)
full = TRADE[0]
print(f"✅ [{LABEL}] cap {pick}: the student would inherit {row['mean reasoning tokens']:.0f} reasoning tokens on average "
      f"instead of {full['mean reasoning tokens']:.0f} ({row['mean reasoning tokens'] / full['mean reasoning tokens']:.0%} of the "
      f"decode cost per answer), keeping {row['share of correct']:.0%} of the correct traces and {row['hardest covered']:.0%} of the hardest problems")

# %% [markdown]
# ## Worked example: the SFT rows and the bill
#
# TRL's Qwen3 training template takes the reasoning from the assistant message's `reasoning_content` field, or
# splits it off `</think>` inside `content`. It always renders a think block, which is empty when there is no
# reasoning (TRL 1.14.0, verify). Both forms follow. Every trace is billed at the output price, the kept ones and
# the rejected ones.

# %%
KEPT = TR.filter_traces(TRACES, max_reasoning_tokens=None if pick == "none" else pick)
ROWS = TR.to_sft_rows(KEPT)
assert TE.check_rows(ROWS) == []
m = ROWS[0]["messages"][1]
print({"role": m["role"], "content": m["content"], "reasoning_content": m["reasoning_content"][:120] + " ..."})
print(TR.to_sft_rows(KEPT[:1], "inline")[0]["messages"][1]["content"][:160], "...")
print(table([{"set": "all traces (paid)", **TR.reasoning_bill(TRACES, 9.0)}, {"set": "kept (trained on)", **TR.reasoning_bill(KEPT, 9.0)}],
            title="The bill at $9.00 per million output tokens (the 06 scaling lab's gemini-3.5-flash price, verify)"))

# %% [markdown]
# ## Exercise 3.3 — the student's output budget
#
# If the student reproduces the kept traces' length distribution, a serving `max_tokens` of $M$ cuts off every answer
# that would be longer: `finish_reason: "length"`, and often no answer at all. Write `cut_share(traces, M)` (the share
# of traces with more than $M$ completion tokens) and `budget_for(traces, max_cut)`: the smallest $M$ among the
# traces' own completion lengths for which at most `max_cut` of them are cut.

# %% exercise
def cut_share(traces: list, M: int) -> float:
    ### BEGIN SOLUTION
    return sum(t.completion_tokens > M for t in traces) / len(traces)
    ### END SOLUTION

def budget_for(traces: list, max_cut: float = 0.01) -> int:
    ### BEGIN SOLUTION
    for M in sorted({t.completion_tokens for t in traces}):
        if cut_share(traces, M) <= max_cut:
            return M
    return max(t.completion_tokens for t in traces)
    ### END SOLUTION

# %% check
assert cut_share(toy, 10) == 0.1 and budget_for(toy, 0.1) == 10 and budget_for(toy, 0.0) == 110
for tr in (KEPT, TR.filter_traces(TRACES)):
    b = budget_for(tr, 0.01)
    assert cut_share(tr, b) <= 0.01 and all(cut_share(tr, M) > 0.01 for M in {t.completion_tokens for t in tr} if M < b)
b_all, b_kept = budget_for(TR.filter_traces(TRACES)), budget_for(KEPT)
print(f"✅ [{LABEL}] to cut at most 1% of answers: max_tokens {b_all} for a student of every correct trace, {b_kept} for "
      f"the capped set; set max_model_len to prompt + that, and size the KV pool for it (rl-and-thinking-models PRIMER §7)")

# %% [markdown]
# ## Worked example: budget-aware distillation, measured on the tiny task
#
# The tiny teacher of notebook 01 answers 1,000 fresh problems. Two students train by SFT on its *verified*
# answers, one with no cap and one capped at 9 tokens, one short of the full scratchpad ($K$ + 4 = 10). The capped
# set keeps every correct answer that is short, which leaves only the lucky guesses and the partial working. About
# 40 seconds with torch.

# %%
if env.has_torch():
    from distillab.tinylm import train as T
    CAP = T.length_cap_experiment(T.DistillConfig(), caps=(None, 9), log=lambda *a: None)
    print(table([CAP["teacher"]], ["generated", "kept", "acc_full", "acc_other", "full_before", "full_after"],
                "MEASURED: the tiny teacher's answers before and after the verifier"))
    print(table(CAP["rows"], title="MEASURED: SFT students on verified answers, by length cap"))
    r0, r9 = CAP["rows"]
    print(f"the cap cut the student's mean length from {r0['student length']} to {r9['student length']} tokens and its "
          f"accuracy from {r0['student accuracy']:.2f} to {r9['student accuracy']:.2f}")
else:
    print("torch is missing: the tiny budget-aware run is skipped (pip install torch; the CPU build is enough)")

# %% [markdown]
# Every trace the capped student saw was *correct*, yet it scores near chance. What it never saw was the
# procedure that makes answers correct. Verified short traces of a problem that needs long reasoning are mostly
# guesses that happened to land. That is the budget trade of PRIMER §5 in its sharpest form. With real models
# the curve is smoother: moderate caps cost a few points of accuracy and save a large share of the tokens.
# Measure it on your eval set before shipping a capped student.
#
# ## Worked example: before and after, measured the same way
#
# At T1 the student before SFT, the student after it and the teacher are three vLLM servers, and `TR.compare`
# measures each one the same way: accuracy on held-out problems with a Wilson interval, and the reasoning length
# it produces. Here the fake thinking teacher and the fake student (both **simulated**) stand in, to show the table.

# %%
with FakeTeacher("thinker") as tu, FakeTeacher("student") as su:
    print(table(TR.compare({"teacher (simulated)": Client(tu, "Qwen/Qwen3-1.7B"), "student (simulated)": Client(su, "Qwen/Qwen3-0.6B")},
                           D.make_set(40, seed=21), n=2), title="SIMULATED: the before/after table"))

# %% [markdown]
# Read the intervals before the point estimates. With 80 traces per model near 90% accuracy each interval is about
# ±6 points wide, so a difference of a few points is noise. Size the eval set before you compare. Resolving a
# 4-point difference at these accuracies takes several hundred problems per model, and paired comparisons on the
# same problems help (notebook 05).
#
# ## On a real GPU (T1)
#
# ```bash
# vllm serve Qwen/Qwen3-1.7B --dtype half --max-model-len 8192 --reasoning-parser qwen3 --port 8000 &   # T4: fp16
# export DISTILLAB_URL=http://127.0.0.1:8000      # this notebook now collects real traces (Qwen3 thinking sampling)
# # after filtering: SFT the student on the reasoning_content rows, then serve it and measure it the same way
# python -m distillab.hf.sft --model Qwen/Qwen3-0.6B --data _run_outputs/traces_sft.jsonl --out _run_outputs/student-traces
# vllm serve _run_outputs/student-traces --dtype half --reasoning-parser qwen3 --port 8001 &
# ```
#
# For the R1 distill as the teacher use `--reasoning-parser deepseek_r1`, temperature 0.6 and no system prompt.
# Its template (`<｜User｜>`/`<｜Assistant｜>`) is not Qwen's, so SeqKD into a Qwen student is fine and logit KD is not.
# The student's before and after is `TR.compare({"before": Client(url_base), "after": Client(url_sft), "teacher":
# Client(url_teacher)}, D.make_set(200, seed=21))`: accuracy with a Wilson interval and the reasoning length, the
# same measurement for all three.

# %%
if not env.server_url():
    TE.write_jsonl(ROWS, "_run_outputs/traces_sft.jsonl")
    print(f"wrote _run_outputs/traces_sft.jsonl: {len(ROWS)} rows ({LABEL}: the format, not a dataset to train on)")
print("lm-eval for a thinking student:", " ".join(A.lm_eval_command("Qwen/Qwen3-0.6B", "gsm8k", thinking=True)))

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We distilled reasoning by SFT on a thinking teacher's verified traces. The student inherits the
# procedure *and* the length distribution of the traces we keep. That makes the filters a serving decision as
# well as a quality one. The verifier alone shifts the kept traces toward longer ones. A length cap makes the
# student cheaper, but it removes the hard problems' examples first; on our toy, a cap below the length the
# procedure needs left only correct guesses, and the student scored at chance. We chose the cap from a trade
# table (kept traces, mean length, coverage of the hardest problems), set `max_tokens` from the kept
# distribution's p99, and checked the student's accuracy with intervals on a hard slice. Distillation beats
# small-model RL when a strong teacher exists (R1's 32B distill against RL on the same base), but it does not add
# knowledge the student lacks."
#
# **Drill 1.** *We capped traces at 1,000 tokens to halve serving cost and accuracy on hard problems collapsed.
# Why?* The hard problems' traces were the long ones, so the cap removed their examples and kept the lucky short
# ones. Check coverage of the hardest problems per cap, or keep long traces for hard problems and cap only the
# easy ones.
#
# **Drill 2.** *Should the student's traces be as long as the teacher's?* Only if the length buys accuracy on your
# workload. The student inherits whatever you keep, so shorter correct traces (a budget-aware filter, or a
# teacher run with a thinking budget) give a cheaper student. Measure the accuracy cost.
#
# **Drill 3.** *Distil or RL for a 1.5B model?* With a stronger teacher available: distil first. It is cheaper and
# better at small scale, as R1's and Qwen3's reports found. RL on top can help (Qwen3 Table 21), but it cannot add
# what neither the teacher's traces nor the base model contain.
