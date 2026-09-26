# %% [markdown]
# # 03 · Measure the accuracy cost: distribution distance, task accuracy, and whether a drop is real
#
# **Tier:** T0 — an offline mini-eval on the bundled tiny model (1,000 held-out problems, exact
# answers) for nine schemes, plus lm-evaluation-harness commands and **sample output in the documented
# format (illustrative)** parsed and compared. T1 — the same commands run `lm_eval` against a real
# checkpoint on a GPU, or against a running `vllm serve` (`QUANTLAB_URL`, `local-completions`).
#
# ## The one-minute version
#
# "Did quantization hurt?" is three measurements that can disagree:
#
# | Measurement | Cost | Sees | Misses |
# |---|---|---|---|
# | **Logit KL / argmax agreement** vs the unquantized model | one forward pass per example, no labels | every shift in the distribution, even ones that do not change an answer yet | whether a shift matters to the task |
# | **Perplexity** | cheap, no generation | average confidence | rare, answer-changing errors |
# | **Task accuracy** (exact match) | generation + labels | what users feel | small effects: ±2.7 points of noise on 250 gsm8k questions |
#
# Compare on the **same questions** (paired flips, McNemar) rather than two accuracies with error bars;
# look at *where* the errors land; and remember long generations compound per-token damage — a
# thinking model that writes 5,000 tokens turns a 1-in-10,000 per-token flip into a 39% chance of at
# least one divergence. Concepts: PRIMER §8 "Measuring the accuracy you pay" ([`PRIMER.md`](../../PRIMER.md)).

# %%
import math
from quantlab import compress as C, env, evalharness as E, tinymodel as tm
import numpy as np

print(env.describe())
ref = tm.load()
calib = C.calibration_inputs(ref, 128)
RECIPES = {"FP8_DYNAMIC": C.Recipe("FP8_DYNAMIC"), "W8A8 (rtn)": C.Recipe("W8A8"),
           "W8A8 + SmoothQuant": C.Recipe("W8A8", ("smoothquant",)), "W4A16 g128 (rtn)": C.Recipe("W4A16"),
           "W4A16 g32 (rtn)": C.Recipe("W4A16", group_size=32), "W4A16 g128 (gptq)": C.Recipe("W4A16", ("gptq",)),
           "W4A16 g128 (awq)": C.Recipe("W4A16", ("awq",)), "NVFP4A16": C.Recipe("NVFP4A16"), "NVFP4 W4A4": C.Recipe("NVFP4")}
BITS = {"FP8_DYNAMIC": 8, "W8A8 (rtn)": 8, "W8A8 + SmoothQuant": 8, "W4A16 g128 (rtn)": 4.125, "W4A16 g32 (rtn)": 4.5,
        "W4A16 g128 (gptq)": 4.125, "W4A16 g128 (awq)": 4.125, "NVFP4A16": 4.5, "NVFP4 W4A4": 4.5}
cache, results, quantized = {}, {"bf16 (reference)": E.mini_eval(ref, ref, n=1000)}, {}
for name, r in RECIPES.items():
    quantized[name] = C.quantize_model(ref, r, calib)
    results[name] = E.mini_eval(quantized[name].model(), ref, n=1000, act_quant=quantized[name].act_quant(), ref_cache=cache)
print("measured on the bundled tiny model (T0), 1,000 held-out problems per task")
print(E.table(results))

# %% [markdown]
# Read the table three ways. The distance metric sees what accuracy does not: the two W8A8 rows tie on
# accuracy (100%), but SmoothQuant cuts KL by orders of magnitude. RTN INT4 loses points, and GPTQ and
# AWQ get them back at the same 4.125 bits; RTN at g32 is *worse* than at g128, which notebook 01 traced
# to the two outlier-meeting columns, not to group size. And W4A4 with FP4 *activations* collapses: the
# two massive-activation channels set every block scale of their token (notebook 05).
#
# One warning before reading a *ranking* off it. How sure is the reference of its answers?

# %%
for task in tm.TASKS:
    p, a = tm.make_task(task, 1000, 1000)
    z = ref.answer_logits(p, a)
    z = z - z.max(-1, keepdims=True)
    top = (np.exp(z) / np.exp(z).sum(-1, keepdims=True)).max(-1)
    print(f"{task:8s} reference top-1 probability: median {np.median(top):.6f}; above 0.999 at "
          f"{np.mean(top > 0.999):.1%} of answer positions")

# %% [markdown]
# Saturated: the reference is all but certain at every position, so a small logit error barely moves
# the softmax, and KL values of 1e-8 against 1e-9 mean nothing. The model is also easy to compensate:
# its inputs come from a 16-token vocabulary through 128-wide layers, so they are low-rank and GPTQ can
# move nearly all of each column's rounding error onto the others. That is why INT4 GPTQ looks as close
# to BF16 as FP8 or W8A8 here. On a real LLM, expect FP8 W8A8 and INT8 W8A8 with SmoothQuant within a
# fraction of a point, INT4 GPTQ/AWQ behind them and INT4 RTN last — the starting order of PRIMER §10's
# `cost.choose()` — and measure it on your model. What transfers from this table is the mechanisms
# (RTN's outlier loss, what calibration recovers, FP4 activations collapsing), not the order of the
# schemes that pass.
#
# ## Exercise 3.1 — the mean KL between two models
#
# Write `mean_kl(ref_logits, test_logits)`: the mean over positions of `KL(p_ref || p_test)` in nats,
# with a numerically safe log-softmax (subtract the max first).

# %% exercise
def mean_kl(ref_logits, test_logits):
    ### BEGIN SOLUTION
    def logsm(z):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
    lp, lq = logsm(ref_logits), logsm(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())
    ### END SOLUTION

# %% check
p, a = tm.make_task("add", 500, 1000)
r_logits = ref.answer_logits(p, a)
q = quantized["W4A16 g128 (rtn)"]
t_logits = q.model().answer_logits(p, a)
assert math.isclose(mean_kl(r_logits, t_logits), E.logit_kl(r_logits, t_logits), rel_tol=1e-9)
assert mean_kl(r_logits, r_logits) == 0.0 and mean_kl(r_logits, t_logits) > 0
print(f"✅ KL(bf16 || W4A16 RTN) = {mean_kl(r_logits, t_logits):.3e} nats per answer token")

# %% [markdown]
# ## Worked example: where the errors land
#
# An aggregate hides *which* skill broke. Per answer position, how often does RTN INT4 pick a
# different token than the reference (teacher-forced), and at which position does a wrong greedy
# answer first go wrong?

# %%
for task in tm.TASKS:
    p, a = tm.make_task(task, 1000, 1000)
    dis = (ref.answer_logits(p, a).argmax(-1) != q.model().answer_logits(p, a).argmax(-1)).mean(0)
    g = q.model().generate(p, a.shape[1])
    first = np.bincount(np.argmax(g != a, 1)[(g != a).any(1)], minlength=a.shape[1])
    print(f"{task:8s} disagreement by position {np.round(dis, 3)}   first wrong position counts {first}")

# %% [markdown]
# For `add` the damage sits almost entirely on the tens digit — the first digit that depends on a
# carry. The same pattern at scale: quantization damage concentrates on specific capabilities
# (arithmetic, rare languages, tool-call formats, long-range retrieval), which is why a task-level eval
# on *your* workload is the gate, not perplexity.
#
# ## Exercise 3.2 — how many questions does it take to see a drop?
#
# An unpaired comparison of two accuracies near `p` has a standard error of about
# `sqrt(2 p (1 - p) / n)`. Write `n_needed(p, delta, z=2)`: the number of questions per model at
# which a drop of `delta` is `z` standard errors. Then use it on the lm-eval sample below.

# %% exercise
def n_needed(p, delta, z=2.0):
    ### BEGIN SOLUTION
    return math.ceil(2 * z * z * p * (1 - p) / delta ** 2)
    ### END SOLUTION

# %% check
base, test = E.sample_results("bf16"), E.sample_results("w4a16")
cmp = E.compare(E.parse_results(base), E.parse_results(test))
for c in cmp:
    print(f"  [sample output in the documented format (illustrative)] gsm8k {c['filter']:16s} "
          f"{c['base']:.3f} -> {c['test']:.3f}  delta {c['delta']:+.3f} +- {c['se']:.3f}  ({c['verdict']})")
need = n_needed(0.30, 0.028)
assert 2100 <= need <= 2200 and need > 1319                  # gsm8k's test split has 1,319 questions
assert all(c["verdict"] == "within noise" for c in cmp)
print(f"✅ a 2.8-point drop at 30% needs ~{need:,} questions per model to see unpaired — more than gsm8k has. "
      "Pair the questions instead.")

# %% [markdown]
# ## Exercise 3.3 — paired flips: McNemar's exact test
#
# Run both models on the same questions and count only the questions they disagree on: `b` = the
# reference got it right and the quantized model wrong, `c` = the reverse. Under "no difference" each
# disagreement is a fair coin, so the two-sided p-value is `2 x P(Binomial(b + c, 1/2) <= min(b, c))`,
# capped at 1. Write `mcnemar(b, c)`.

# %% exercise
def mcnemar(b, c):
    ### BEGIN SOLUTION
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    ### END SOLUTION

# %% check
for name in ("W4A16 g128 (rtn)", "W4A16 g128 (gptq)", "W8A8 (rtn)"):
    for task in tm.TASKS:
        row = results[name][task]
        assert math.isclose(mcnemar(row["right_to_wrong"], row["wrong_to_right"]), row["mcnemar_p"])
rtn = results["W4A16 g128 (rtn)"]["add"]
print(f"✅ RTN INT4 on add: {rtn['right_to_wrong']} right->wrong, {rtn['wrong_to_right']} wrong->right, "
      f"p = {mcnemar(rtn['right_to_wrong'], rtn['wrong_to_right']):.1e} — a real regression, from "
      f"{rtn['right_to_wrong'] + rtn['wrong_to_right']} disagreements out of 1,000")

# %% [markdown]
# ## Exercise 3.4 — set a budget, then pick the cheapest scheme that meets it
#
# A budget has two parts: a task part (accuracy may drop at most `max_drop` on every task) and a
# distribution part (mean KL at most `max_kl` on every task — headroom for inputs your eval did not
# cover). Write `passes(row, ref_row, max_drop, max_kl)` for one scheme's `mini_eval` result, then
# `cheapest(results, bits, ...)`: the passing scheme with the fewest bits per weight (ties: any).

# %% exercise
def passes(row, ref_row, max_drop=0.005, max_kl=1e-3):
    ### BEGIN SOLUTION
    return all(ref_row[t]["accuracy"] - row[t]["accuracy"] <= max_drop and row[t]["kl"] <= max_kl for t in row)
    ### END SOLUTION

def cheapest(results, bits, max_drop=0.005, max_kl=1e-3):
    ### BEGIN SOLUTION
    ok = [k for k in bits if passes(results[k], results["bf16 (reference)"], max_drop, max_kl)]
    return min(ok, key=lambda k: bits[k])
    ### END SOLUTION

# %% check
pick = cheapest(results, BITS)
assert pick in ("W4A16 g128 (gptq)", "W4A16 g128 (awq)"), pick
assert not passes(results["W4A16 g128 (rtn)"], results["bf16 (reference)"])
assert not passes(results["NVFP4 W4A4"], results["bf16 (reference)"])
assert passes(results["FP8_DYNAMIC"], results["bf16 (reference)"])
print(f"✅ cheapest scheme within 0.5 points and 1e-3 nats on every task: {pick} ({BITS[pick]} bits per weight); "
      "round-to-nearest at the same bits fails the budget (on this saturated toy: run the same gate on your eval)")

# %% [markdown]
# ## Worked example: one decision, four kinds of numbers — and a label on each
#
# A recommendation mixes numbers of very different standing: bytes counted from files (exact), step
# times from the emulator (simulated), task accuracy on the tiny model (a T0 measurement, not your
# model) and lm-eval figures copied from a documented format (sample). `report.Report` refuses a
# section without a source and prints the label next to every table.

# %%
import pathlib, tempfile
from quantlab import bench as B, report
rep = report.Report("W4A16 GPTQ vs FP8 vs BF16 for an 8B model on L4s")
rep.add("Checkpoint size", "exact", [{"scheme": k, "bits per weight": BITS[k]} for k in ("FP8_DYNAMIC", "W4A16 g128 (gptq)")])
rep.add("Serving speed, Llama-3.1-8B on an L4, 8 users", "simulated",
        [{k: r[k] for k in ("scheme", "ttft_ms_mean", "tpot_ms_mean", "output_tok_s")}
         for r in B.compare(["bf16", "fp8", "w4a16"], users=8, n_requests=16)])
rep.add("Accuracy on the tiny model", "t0-eval",
        [{"scheme": k, "add": results[k]["add"]["accuracy"], "add KL": results[k]["add"]["kl"]} for k in
         ("FP8_DYNAMIC", "W4A16 g128 (rtn)", "W4A16 g128 (gptq)")])
rep.add("gsm8k (lm-eval)", "sample", [{k: c[k] for k in ("filter", "base", "test", "delta", "verdict")} for c in cmp])
tmp = pathlib.Path(tempfile.mkdtemp(prefix="quantlab-03-"))
md, js = rep.save(tmp / "decision")
print(md.read_text())
C.clean(tmp)

# %% [markdown]
# ## Worked example: long generations compound small damage
#
# If each generated token independently diverges with probability `e`, a generation of `L` tokens
# stays identical with probability `(1 - e)^L`. The tiny model's answers are 4-6 tokens; a reasoning
# trace is thousands. Per-token argmax disagreement measured above is the `e` to plug in.

# %%
e_rtn = float(np.mean([results["W4A16 g128 (rtn)"][t]["argmax_agree"] for t in tm.TASKS]))
for e in (1 - e_rtn, 1e-3, 1e-4):
    print(f"per-token divergence {e:.1e}: " + "  ".join(f"L={L:>6,}: {1 - (1 - e) ** L:6.1%}" for L in (6, 500, 5000, 20000)))

# %% [markdown]
# Divergence is not failure — a paraphrase can still be right — but it is why quantized *reasoning*
# models are evaluated on long-generation tasks (math, code) with the chat template and the model's
# thinking tokens, and why a KL budget matters more for them than for short classification.
#
# ## lm-evaluation-harness (T1)
#
# The command lines for the three backends (lm-eval 0.4.13, verify): `vllm` runs the engine in
# process; `local-completions` evaluates a server you already started (the same one you benchmark);
# `hf` uses transformers. `add_bos_token=True` because quantized models can be sensitive to it (vLLM's
# docs); `--limit` is for smoke tests and makes results "for testing only".

# %%
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print(E.shell(E.lm_eval_command(MODEL, tasks=["gsm8k"], num_fewshot=5, limit=250)))
print(E.shell(E.lm_eval_command("./Qwen2.5-0.5B-Instruct-W4A16-gptq", tasks=["gsm8k"], num_fewshot=5, limit=250)))
print(E.shell(E.lm_eval_command(MODEL, backend="local-completions", tasks=["gsm8k"],
                                base_url="http://127.0.0.1:8000/v1/completions")))
if env.t1_allowed() and env.has_cli("lm_eval"):
    out = E.run_lm_eval(E.lm_eval_command(MODEL, tasks=["gsm8k"], num_fewshot=5, limit=50, output_path="lm_eval_out"))
    print("MEASURED:\n", out.stdout[-2000:])
else:
    print("T0: lm_eval not run here (GPU + `pip install \"lm_eval[vllm]==0.4.13\"` + QUANTLAB_RUN_T1=1). "
          "The table it prints parses with E.parse_table():")
    print((E.SAMPLES / "lm_eval_table_bf16.txt").read_text())

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We gate every quantized checkpoint on two numbers against the unquantized model:
# task accuracy on our own eval set, compared question by question, and mean KL of the next-token
# distributions on held-out traffic. Accuracy is what users see, but it is noisy — a 3-point drop on
# 250 gsm8k questions is inside the noise, and seeing it unpaired would take more questions than
# gsm8k has — so we compare paired flips. KL is sensitive long before accuracy moves; it tells us how
# much headroom we have for inputs our eval does not cover, and it matters most for long generations,
# where small per-token divergences compound. On the lab's model, round-to-nearest INT4 fails that
# budget, GPTQ at the same 4.125 bits passes it, and FP4 activations fail badly — though that toy is
# saturated and easy to compensate, so on our model we expect INT4 GPTQ to cost more than FP8, and we
# measure it."
#
# **Drill 1.** *Perplexity went from 6.14 to 6.26 after W8A8 (the SmoothQuant README's Llama-3-8B numbers).
# Ship?* — Not on perplexity alone: run the
# task evals that matter (and the long-generation ones for a reasoning model), compare paired, check
# the capabilities quantization tends to break first (math, code, rare languages, tool-call formats).
#
# **Drill 2.** *Accuracy is identical; why do you still care about KL?* — Identical accuracy on N questions
# bounds the damage only on that distribution; KL measures the shift everywhere, and large KL predicts
# failures on inputs the eval did not contain.
#
# **Drill 3.** *lm-eval says 27.2 vs 30.0 on gsm8k with ±2.9 stderr each. Regression?* — Unpaired, it is
# within noise (z ≈ -0.7). Re-run with `--log_samples` and count paired flips; or run the full split.
