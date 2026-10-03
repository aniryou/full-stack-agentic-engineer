# %% [markdown]
# # 03 · Test-time compute for real: pass@k, majority vote, best-of-n, and the price of a correct answer
#
# **Tier:** T1. Sample every problem $n$ times from a real thinking model. Point `THINKLAB_URL` at
# `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`. 40 problems × (8 + 8 + 16) samples is a few
# thousand requests: tens of minutes on a T4 (verify on yours). Set `COLLECT = True`. T0 (default):
# the bundled table of the **simulated** model of the lab, on the same problems, with illustrative
# outcomes that `python -m thinklab.thinking.recorded` generates again.
#
# ## The one-minute version
#
# * At inference time, there are two ways to use more compute on a question. One way is to think
#   **longer** (sequential: more reasoning tokens, or a larger budget). The other way is to sample
#   **more** answers and select one (parallel: majority vote, best-of-n with a verifier or reward
#   model). See PRIMER §6 "Test-time compute".
# * Measure them correctly. **pass@k**, the chance that at least one of $k$ samples is correct, needs
#   the unbiased estimator `1 − C(n−c, k)/C(n, k)` from $n \ge k$ samples. The shortcut
#   `1 − (1 − c/n)^k` is biased. Another measure is **pass^k**, the chance that *all* $k$ are
#   correct. It is the reliability that an agent needs, and it *decreases* with $k$. **maj@k** needs no verifier, only answers that you can
#   compare.
# * pass@k is an upper bound, and only a perfect verifier gets to it. Majority vote and a noisy
#   reward model stay below it. Majority vote can *lose* accuracy when an incorrect answer is the
#   most common one.
# * The only fair comparison is **cost per correct answer**: tokens (all samples, all reasoning) ×
#   price ÷ accuracy. The bill counts a thinking token as an output token.

# %%
import math, random, statistics
from thinklab import env
from thinklab.report import table
from thinklab.thinking import recorded
from thinklab.thinking.evalset import make_evalset
from thinklab.thinking.ttc import (best_of_n_accuracy, compute_optimal, maj_at_k, majority_vote as ref_vote,
                                   pass_at_k as ref_pass_at_k, pass_hat_k as ref_pass_hat_k)

print(env.describe())
COLLECT = False                                  # T1: set True with THINKLAB_URL pointing at a real server
if COLLECT and env.server_url() and not env.is_simulated(env.server_url(), env.auth_headers()):
    from thinklab.thinking.client import ThinkingClient
    client = ThinkingClient(env.server_url(), headers=env.auth_headers())
    RECORDS = recorded.collect(client, make_evalset(40, seed=0), samples=8, budgets=(256, 1024), budget_samples=8)
    LABEL = f"MEASURED on {client.model}"
else:
    RECORDS = recorded.load()
    LABEL = "SIMULATED model, bundled records (illustrative)"
print(LABEL, "|", len(RECORDS), "records")

def by(mode, budget=None):
    return [r for r in RECORDS if r["mode"] == mode and r["budget"] == budget]

rows = []
for mode, b in (("off", None), ("on", None), ("budget", 128), ("budget", 256), ("budget", 512), ("budget", 1024)):
    sel = by(mode, b)
    if not sel:
        continue
    rows.append({"mode": mode if b is None else f"budget {b}", "problems": len(sel),
                 "samples each": len(sel[0]["correct"]),
                 "accuracy (pass@1)": round(statistics.fmean(c for r in sel for c in r["correct"]), 3),
                 "output tokens": round(statistics.fmean(a + t for r in sel for a, t in zip(r["reasoning_tokens"], r["answer_tokens"])))})
print(table(rows, title=f"[{LABEL}] one sample, by mode"))

# %% [markdown]
# ## Exercise 3.1 — pass@k from recorded samples, and what the shortcut would have said
#
# The rl-core notebook 04 (exercise 4.1) derives and implements the unbiased estimator
# `1 − C(n−c, k)/C(n, k)`. Here, its name is `ref_pass_at_k`. Apply it to the records.
# `pass_at_k_both(sel, k)` returns `(unbiased, plugin)`. These are two means over the problems in
# `sel`: the mean of `ref_pass_at_k(n, c, k)` and the mean of the plug-in `1 − (1 − c/n)^k`. Each
# problem has $n$ samples, and $c$ of them are correct.
#
# Then set `gap_comes_from` to the kind of problem that produces all of the difference between the
# two values: `"all right"`, `"all wrong"` or `"some right"`.

# %% exercise
def pass_at_k_both(sel: list, k: int) -> tuple:
    ### BEGIN SOLUTION
    nc = [(len(r["correct"]), sum(r["correct"])) for r in sel]
    return (statistics.fmean(ref_pass_at_k(n, c, k) for n, c in nc),
            statistics.fmean(1 - (1 - c / n) ** k for n, c in nc))
    ### END SOLUTION

gap_comes_from = None
### BEGIN SOLUTION
gap_comes_from = "some right"      # c = 0: both are 0; c = n: both are 1
### END SOLUTION

# %% check
toy = [{"correct": [1, 0, 0, 0]}, {"correct": [1, 1, 1, 1]}, {"correct": [0, 0, 0, 0]}]
ub, pl = pass_at_k_both(toy, 2)
assert abs(ub - (0.5 + 1 + 0) / 3) < 1e-12 and abs(pl - (1 - 0.75 ** 2 + 1) / 3) < 1e-12
assert gap_comes_from == "some right"
rows = []
for mode in ("off", "on"):
    sel = by(mode)
    assert abs(pass_at_k_both(sel, 1)[0] - pass_at_k_both(sel, 1)[1]) < 1e-12        # k = 1: the same
    few = sum(0 < sum(r["correct"]) <= 2 for r in sel)
    for k in (2, 4, 8):
        ub, pl = pass_at_k_both(sel, k)
        assert ub >= pl - 1e-12                              # drawing without replacement misses less
        rows.append({"thinking": mode, "k": k, "pass@k unbiased": round(ub, 3), "plug-in": round(pl, 3),
                     "understated by": round(ub - pl, 3), "problems with 1-2 of 8 right": few})
print(table(rows, title=f"[{LABEL}] pass@k two ways, from the same samples"))
print("✅ the plug-in understates pass@k, and only through problems the model sometimes gets right; "
      "it is worst where few samples are right, the slice where extra samples matter most")

# %% [markdown]
# ## Exercise 3.2 — majority vote
#
# Return the most common answer. Ignore `None`: it is no answer, for example because the output
# ended during the thinking. If two answers have the same count, select the answer that appeared
# *first*. If there are no answers, return `None`.

# %% exercise
def my_vote(answers: list):
    ### BEGIN SOLUTION
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = {}
    for a in votes:
        counts[a] = counts.get(a, 0) + 1
    top = max(counts.values())
    return next(a for a in votes if counts[a] == top)
    ### END SOLUTION

# %% check
assert my_vote(["4", "5", "4", None, None, None]) == "4"
assert my_vote(["5", "4", "4", "5"]) == "5" and my_vote([None, None]) is None and my_vote([]) is None
rng = random.Random(0)
for _ in range(300):
    xs = [rng.choice(["a", "b", "c", None]) for _ in range(rng.randrange(1, 9))]
    assert my_vote(xs) == ref_vote(xs)
print("✅ majority vote with None ignored and first-seen tie-breaking")

# %% [markdown]
# ## Worked example: three ways to use k samples
#
# For each problem with 8 thinking-mode samples:
#
# * **pass@k** (perfect verifier, oracle best-of-n): the ceiling.
# * **maj@k**: the majority vote over $k$ of the 8 samples, as a mean over random subsets.
# * **best-of-k by a reward model**: the sample with the top score from a *noisy* scorer. In the
#   simulated records, the scorer gives correct answers +0.8 on average, with unit noise. At T1, use
#   a real reward model.

# %%
for mode in ("off", "on"):
    sel = by(mode)
    out = []
    for k in (1, 2, 4, 8):
        out.append({"k": k,
                    "pass@k": round(statistics.fmean(ref_pass_at_k(len(r["correct"]), sum(r["correct"]), k) for r in sel), 3),
                    "maj@k": round(statistics.fmean(maj_at_k(r["answers"], r["truth"], k) for r in sel), 3),
                    **({"best-of-k (RM)": round(statistics.fmean(best_of_n_accuracy(r["correct"], r["rm_scores"], k) for r in sel), 3)}
                       if sel[0]["rm_scores"] else {}),
                    "pass^k": round(statistics.fmean(ref_pass_hat_k(len(r["correct"]), sum(r["correct"]), k) for r in sel), 3)})
    print(table(out, title=f"[{LABEL}] thinking {mode}"), "\n")

# %% [markdown]
# Read across a row. pass@k increases fastest, because it assumes that you can recognise the correct
# answer. maj@k increases more slowly. It stops at the share of problems where the correct answer is
# the *most common* one. On problems where an incorrect answer is the most common one, more votes
# make it *more* certain.
#
# Best-of-k with the noisy reward model increases at first, and then it can *decrease*. With more
# samples, an incorrect sample has more chances to get the top score. That is reward
# over-optimisation at inference time. It is also the reason to trust a verifier more than a reward
# model, where a verifier exists. The pass^k column decreases: an agent that runs the same step many
# times sees the chance that every one of $k$ attempts succeeds. Thinking increases every column,
# and each thinking sample costs approximately ten times the tokens.
#
# ## Exercise 3.3 — pass^k on the records, and the independence shortcut
#
# pass^k, the chance that $k$ samples of the *same* problem are all correct, is `C(c, k)/C(n, k)`
# (τ-bench). See rl-core notebook 04, exercise 4.2. Here, its name is `ref_pass_hat_k`. A common
# shortcut raises pass@1 to the $k$-th power. It treats each attempt as an independent coin with the
# average accuracy of the eval set.
#
# `reliability(sel, k)` returns `(pass_hat, shortcut)`: the mean over problems of
# `ref_pass_hat_k(n, c, k)`, and $(\text{mean accuracy over all samples})^k$. Then set `shortcut_is`
# to `"optimistic"` or `"pessimistic"` for these records. Write the reason in a comment.

# %% exercise
def reliability(sel: list, k: int) -> tuple:
    ### BEGIN SOLUTION
    nc = [(len(r["correct"]), sum(r["correct"])) for r in sel]
    acc = sum(c for _, c in nc) / sum(n for n, _ in nc)
    return statistics.fmean(ref_pass_hat_k(n, c, k) for n, c in nc), acc ** k
    ### END SOLUTION

shortcut_is = None
### BEGIN SOLUTION
shortcut_is = "pessimistic"   # failures concentrate on the hard problems: easy ones succeed every time
### END SOLUTION

# %% check
ph, sc = reliability([{"correct": [1, 1, 1, 1]}, {"correct": [0, 0, 0, 0]}], 2)
assert ph == 0.5 and sc == 0.25                            # half the problems always work: pass^2 = 0.5, not 0.25
out = []
for mode in ("off", "on"):
    for k in (2, 4, 8):
        ph, sc = reliability(by(mode), k)
        out.append({"thinking": mode, "k": k, "pass^k": round(ph, 3), "accuracy^k": round(sc, 3)})
        if not LABEL.startswith("MEASURED"):
            assert (ph > sc) == (shortcut_is == "pessimistic")
print(table(out, title=f"[{LABEL}] repeating the same problem k times"))
print("✅ averaged over problems of mixed difficulty, pass^k is higher than accuracy^k: the easy problems "
      "succeed every time and the hard ones fail every time. For one hard problem it is far lower, so report "
      "pass^k per slice, not one number")

# %% [markdown]
# ## Exercise 3.4 — cost per correct answer
#
# Return the dollars per correct answer under these conditions:
#
# * The model answers each question with `samples` samples.
# * Each sample has `output_tokens` output tokens (reasoning included), at `price_out` dollars per
#   million.
# * The selected answer is correct with the probability `accuracy`.
#
# Then compare three strategies with the numbers from the table that the first code cell prints, at
# an output price of $2.00 per million tokens. This is an illustrative price. The 06 gateway lab's
# `cost_per_call` has dated prices. The three strategies are thinking off × maj@8, thinking on × 1
# sample, and a 512-token budget × 1 sample.

# %% exercise
def dollars_per_correct(output_tokens: float, accuracy: float, price_out: float, samples: int = 1) -> float:
    ### BEGIN SOLUTION
    if accuracy <= 0:
        return math.inf
    return samples * output_tokens * price_out / 1e6 / accuracy
    ### END SOLUTION

# %% check
assert dollars_per_correct(1000, 0.5, 2.0) == 0.004 and dollars_per_correct(100, 0.0, 2.0) == math.inf
assert abs(dollars_per_correct(100, 0.4, 2.0, samples=8) - 0.004) < 1e-12
def tok(sel):
    return statistics.fmean(a + t for r in sel for a, t in zip(r["reasoning_tokens"], r["answer_tokens"]))
PRICE = 2.00
strategies = [
    ("thinking off, maj@8", tok(by("off")), statistics.fmean(maj_at_k(r["answers"], r["truth"], 8) for r in by("off")), 8),
    ("thinking on, 1 sample", tok(by("on")), statistics.fmean(c for r in by("on") for c in r["correct"]), 1)]
if by("budget", 512):
    strategies.append(("budget 512, 1 sample", tok(by("budget", 512)),
                       statistics.fmean(c for r in by("budget", 512) for c in r["correct"]), 1))
print(table([{"strategy": s, "tokens/question": round(t * n), "accuracy": round(a, 3),
              "$ per 1K correct": round(1000 * dollars_per_correct(t, a, PRICE, n), 3)} for s, t, a, n in strategies],
            title=f"[{LABEL}] at ${PRICE:.2f} per 1M output tokens"))
print("✅ cost per correct answer = all tokens of all samples x price / accuracy")

# %% [markdown]
# ## Exercise 3.5 — compute-optimal: the best accuracy for a token budget per question
#
# Build the option list. It has every (mode, $k$) with two values:
#
# * its expected tokens per question (`k` × mean output tokens of the mode)
# * its accuracy (maj@k for ${k > 1}$, pass@1 for $k$ = 1)
#
# For each budget of tokens per question, select the most accurate option that fits. If two options
# have the same accuracy, the lower-cost option wins.
#
# This is the small-scale version of "compute-optimal test-time scaling". The best way to use a
# given number of tokens depends on how many tokens there are. With other models and tasks, the
# order can be different. Sometimes many short samples are better than one long sample. Measure it
# for your model and task.

# %% exercise
def options_from_records() -> list:
    ### BEGIN SOLUTION
    opts = []
    for mode, b in (("off", None), ("on", None), ("budget", 256), ("budget", 512), ("budget", 1024)):
        sel = by(mode, b)
        if not sel:
            continue
        n = len(sel[0]["correct"])
        for k in (1, 2, 4, 8):
            if k > n:
                continue
            acc = (statistics.fmean(c for r in sel for c in r["correct"]) if k == 1 else
                   statistics.fmean(maj_at_k(r["answers"], r["truth"], k) for r in sel))
            opts.append({"option": f"{mode if b is None else f'budget {b}'} x{k}", "tokens": k * tok(sel), "accuracy": acc})
    return opts
    ### END SOLUTION

# %% check
opts = options_from_records()
assert len(opts) >= 8 and all({"option", "tokens", "accuracy"} <= set(o) for o in opts)
picks = [{"budget/question": B, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in (compute_optimal(opts, B) or {"option": "nothing fits"}).items()}}
         for B in (150, 400, 800, 1500, 3000, 10000)]
print(table(picks, title=f"[{LABEL}] most accurate option within a token budget per question"))
accs = [p.get("accuracy", 0) for p in picks]
assert accs == sorted(accs), "a larger budget can never pick a less accurate option"
print("✅ the best option changes with the budget: direct answers, then budgeted thinking, then full thinking, then votes over thinking samples")

# %% [markdown]
# ## On a real GPU (T1)
#
# Point `THINKLAB_URL` at a real server. Set `COLLECT = True`, and run the notebook again.
# `recorded.collect` asks for `n=8` choices per request. Thus vLLM prefills the prompt one time and
# decodes eight sequences in the same batch. Parallel sampling has a low cost in prefill, and it
# costs the same KV and decode as eight requests. The T1 path has no reward model.
#
# Add a reward model, or use only the verifier column. Any sequence-classification reward model that
# vLLM serves with `--task reward` is applicable. Make sure that your version has that flag.

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We can buy accuracy at inference time in two ways. We let the model think
# longer, or we sample several answers and select one. We compare them on cost per *correct* answer,
# and we count every reasoning token of every sample as output. We can reach pass@k only with a
# verifier that we trust. We have one for math and code, but not for open-ended answers.
#
# "Majority vote needs no verifier, but it stops at a limit, and it can make a common incorrect
# answer stronger. For agents, we report pass^k, because a workflow that runs a step many times
# needs all of the runs to succeed. On our eval, the best option depends on the token budget per
# question. Thus the router, not the model, selects it (notebook 04, exercise 4.5)."
#
# **Drill 1.** *Why not estimate pass@8 as $1 - (1 - \text{pass@1})^8$?* That formula assumes
# independent samples with the same probability of success on every problem. The estimator from
# $n \ge k$ real samples per problem is unbiased. When you calculate its mean over problems, easy
# and hard problems stay separate.
#
# **Drill 2.** *maj@16 is worse than maj@4 on our hardest slice. Bug?* No. When an incorrect answer
# is the modal answer on a problem, more votes make the incorrect answer win more reliably. Examine
# the share of problems where the correct answer is modal.
#
# **Drill 3.** *pass@1 is 90%, so the five-step agent succeeds 90% of the time?* The calculation
# applies only if you run each step one time and the steps are independent. It gives $0.9^5$ ≈ 59%.
# If a step must succeed every time that the agent retries or repeats it, measure pass^k.
