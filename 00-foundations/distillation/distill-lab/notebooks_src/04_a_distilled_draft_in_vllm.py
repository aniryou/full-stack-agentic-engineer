# %% [markdown]
# # 04 · A distilled draft for speculative decoding: acceptance is the student's metric
#
# **Tier:** T1 on a rented 24 GB card (L4 or RTX 4090). The target is `Qwen/Qwen3-4B`. The drafts are `Qwen/Qwen3-0.6B`
# off the shelf and a 0.6B distilled on the target's own outputs. The notebook compares them through vLLM's
# `--speculative-config` and its spec-decode counters (model ids, fits and flags verify).
#
# T0 (default): the tiny teacher of notebook 01 is the target, and its students are the drafts. Torch on a CPU
# calculates their acceptance from the models' own distributions. This takes about 50 seconds. Without torch, the
# notebook uses the recorded run. A bundled pair of `/metrics` scrapes, **synthetic** at $\alpha = 0.7$, teaches vLLM's
# counters.
#
# ## The one-minute version
#
# * **A draft model is a student graded on one number:** how often the target accepts what the draft proposes. At each
#   position, $\alpha = \sum_v \min(p(v), q(v)) = 1 - \mathrm{TV}(p, q)$. Then one target pass gives
#   $(1 - \alpha^{k+1}) / (1 - \alpha)$ tokens. The speedup is that number divided by $(k\,c + 1)$, where $c$ is the
#   cost of a draft step. See serving-engine PRIMER §7, whose formulas `minengine.spec` implements, and PRIMER §7 "A
#   distilled draft for speculative decoding".
# * **Distillation of the draft on the target's outputs raises $\alpha$.** The makers of an off-the-shelf small model
#   of the same family trained it on other data. That mismatch of distributions lowers $\alpha$. For a draft, do *not*
#   filter by a verifier. The reason is that the goal is to match the target, also where the target makes mistakes.
# * **Read vLLM's numbers carefully.** By default, vLLM 0.30.0 makes *greedy* drafts. Thus the acceptance is
#   $p(\operatorname{argmax} q)$, not $\sum \min(p, q)$. Its "acceptance rate" is
#   $\text{accepted} / \text{drafted} = (E - 1)/k$, not $\alpha$. $\alpha$ is the per-position rate at position 0. The
#   mean acceptance length $1 + \text{accepted}/\text{drafts}$ is $E$.
# * **Same vocabulary or nothing:** vLLM's `draft_model` method compares `vocab_size`. A Qwen3-0.6B draft for Qwen3-4B
#   passes (151,936 both). A Qwen2.5-0.5B draft for Qwen2.5-7B fails (151,936 against 152,064).

# %%
import math, shlex, statistics
from distillab import draft as DR, env, metrics as M
from distillab.cost import load_config, shape
from distillab.report import table
from distillab.tinylm.curves import label, load_experiments, load_recorded

print(env.describe())
P_PRIMER, Q_PRIMER = [0.5, 0.3, 0.15, 0.05], [0.2, 0.2, 0.2, 0.4]
a = DR.acceptance_rate(P_PRIMER, Q_PRIMER)
print(f"serving-engine PRIMER §7's example: alpha = {a:.2f}; a greedy draft (argmax q = token 3) is accepted with p(3) = "
      f"{DR.greedy_acceptance(P_PRIMER, Q_PRIMER):.2f}")
print(table([{"k": k, **{f"E at alpha {al}": round(DR.expected_tokens(al, k), 3) for al in (0.6, 0.8, 0.95)}} for k in (1, 2, 4, 6, 8)],
            title="Tokens per target pass, (1 - alpha^(k+1)) / (1 - alpha)"))

# %% [markdown]
# ## Worked example: vLLM's spec-decode counters
#
# vLLM exports four counters: drafts (the verify steps that had drafts), draft tokens, accepted tokens, and
# accepted tokens per position (label `position` = $0 \ldots k - 1$). A position $i$ counts only when the target
# accepted all of the positions $0 \ldots i$. The bundled scrapes are **synthetic**. `DR.simulate_counters` made 20,000
# drafts of $k$ = 4 at $\alpha = 0.7$. The scrapes show how to read the counters, not what any draft gets.

# %%
from importlib import resources
text = lambda n: resources.files("distillab.assets").joinpath("samples", n).read_text()  # noqa: E731
BEFORE, AFTER = M.parse(text("spec_decode_metrics_before.txt")), M.parse(text("spec_decode_metrics_after.txt"))
print(text("spec_decode_metrics_after.txt"))

# %% [markdown]
# ## Exercise 4.1 — from counters to α
#
# Write `read_counters(before, after)`. It returns `(alpha, mean_length, rate)`. `alpha` is the per-position rate at
# position 0 (the accepted tokens at position 0 divided by the drafts). `mean_length` is
# $1 + \text{accepted} / \text{drafts}$ (vLLM counts the target's bonus token). `rate` is the accepted tokens divided by
# the draft tokens. vLLM logs this number as "Avg Draft acceptance rate".
#
# Use `after.value(name)` and `before.value(name)` with the names in `distillab.metrics`. Use `after.by_label(M.SPEC_ACCEPTED_PER_POS, "position")`
# for the per-position counts.

# %% exercise
def read_counters(before, after) -> tuple:
    ### BEGIN SOLUTION
    d = lambda n: after.value(n) - before.value(n)  # noqa: E731
    drafts, draft_tokens, accepted = d(M.SPEC_DRAFTS), d(M.SPEC_DRAFT_TOKENS), d(M.SPEC_ACCEPTED)
    pos0 = after.by_label(M.SPEC_ACCEPTED_PER_POS, "position")["0"] - before.by_label(M.SPEC_ACCEPTED_PER_POS, "position").get("0", 0.0)
    return pos0 / drafts, 1 + accepted / drafts, accepted / draft_tokens
    ### END SOLUTION

# %% check
alpha, mean_len, rate = read_counters(BEFORE, AFTER)
ref = M.spec_decode(BEFORE, AFTER)
assert abs(alpha - ref["alpha_pos0"]) < 1e-12 and abs(mean_len - ref["mean_acceptance_length"]) < 1e-12
assert abs(rate - ref["acceptance_rate"]) < 1e-12
v = DR.vllm_views(alpha, 4)
assert abs(mean_len - v["mean_acceptance_length"]) < 0.05 and abs(rate - (mean_len - 1) / 4) < 1e-12
print(f"✅ [synthetic counters] alpha {alpha:.3f}, mean acceptance length {mean_len:.3f} (formula at that alpha, k = 4: "
      f"{v['mean_acceptance_length']:.3f}), vLLM's 'acceptance rate' {rate:.3f}, which is (E - 1)/k, not alpha")

# %% [markdown]
# ## Worked example: the tiny drafts
#
# The target is the tiny teacher of notebook 01. Three students are the drafts:
#
# * **off-the-shelf** (`hard`): its training used the labelled set, not the target's outputs.
# * **distilled** (`seqkd_all`): SFT on the target's own samples, with no filter.
# * **distilled and verified** (`seqkd`): the same, with the verifier filter.
#
# For each position of the target's own samples, the cell calculates $\alpha = \sum \min(p, q)$ and the greedy
# acceptance $p(\operatorname{argmax} q)$. With torch, this takes about 50 seconds.

# %%
if env.has_torch():
    from distillab.tinylm import train as T
    RUN = T.run(T.DistillConfig(methods=("hard", "seqkd_all", "seqkd")), log=lambda *a: None, keep_models=True)
    MODELS = RUN["models"]
    POS = {m: DR.tiny_acceptance(MODELS[m], MODELS["teacher"], RUN["teacher_seqs"]) for m in ("hard", "seqkd_all", "seqkd")}
    ALPHA = {m: RUN["students"][m]["final"]["accept"] for m in POS}
    GREEDY = {m: RUN["students"][m]["final"]["accept_greedy"] for m in POS}
    ALPHA["target itself"] = 1.0
    GREEDY["target itself"] = RUN["teacher"]["eval"]["accept_greedy"]
    LABEL = label(RUN)
    print(table([{"position": i, **{m: POS[m][i]["alpha"] for m in POS}} for i in range(len(POS["hard"]))],
                title=f"[{LABEL}] alpha by completion position"))
else:
    RUN, EXP = load_recorded(), load_experiments()
    LABEL = RUN["source"]
    finals = {"hard": RUN["students"]["hard"]["final"], "seqkd_all": EXP["seqkd_all"]["final"], "seqkd": RUN["students"]["seqkd"]["final"]}
    ALPHA = {m: f["accept"] for m, f in finals.items()}
    GREEDY = {m: f["accept_greedy"] for m, f in finals.items()}
    ALPHA["target itself"], GREEDY["target itself"] = 1.0, RUN["teacher"]["eval"]["accept_greedy"]
C_TINY = RUN["params"]["student"] / RUN["params"]["teacher"]
print(table([{"draft": m, "alpha (sampled draft)": ALPHA[m], "greedy draft": GREEDY[m]} for m in ALPHA],
            title=f"[{LABEL}] mean acceptance over the target's samples; draft/target size c = {C_TINY:.3f}"))

# %% [markdown]
# ## Exercise 4.2 — the best draft and k
#
# You have the $\alpha$ and the cost ratio $c$ of each draft. Write `best_draft(alphas, c, k_max=8)`. It returns the
# `(name, k, speedup)` with the highest `DR.speedup(alpha, k, c)`. The search covers every draft and every $k$ from 1
# to `k_max`. Leave out `"target itself"`, because the target is not a draft that you can afford.

# %% exercise
def best_draft(alphas: dict, c: float, k_max: int = 8) -> tuple:
    ### BEGIN SOLUTION
    cands = [(name, k, DR.speedup(al, k, c)) for name, al in alphas.items() if name != "target itself"
             for k in range(1, k_max + 1)]
    return max(cands, key=lambda x: x[2])
    ### END SOLUTION

# %% check
name, k, sp = best_draft(ALPHA, C_TINY)
brute = max(((n, kk, DR.speedup(al, kk, C_TINY)) for n, al in ALPHA.items() if n != "target itself" for kk in range(1, 9)),
            key=lambda x: x[2])
assert (name, k) == brute[:2] and abs(sp - brute[2]) < 1e-12
assert best_draft({"x": 0.9, "y": 0.5}, 0.1)[0] == "x"
rows = [{"draft": n, "alpha": ALPHA[n], "best k": DR.best_k(ALPHA[n], C_TINY, 8),
         "speedup": round(DR.speedup(ALPHA[n], DR.best_k(ALPHA[n], C_TINY, 8), C_TINY), 3)} for n in ALPHA if n != "target itself"]
print(table(rows, title=f"[{LABEL}] speedup at the best k (c = {C_TINY:.3f})"))
print(f"✅ best: {name} at k = {k}, {sp:.2f}x. " + ("The drafts distilled on the target's outputs beat the off-the-shelf one."
      if min(ALPHA["seqkd_all"], ALPHA["seqkd"]) > ALPHA["hard"] else "In this run the off-the-shelf draft was not worst: read the alpha table."))

# %% [markdown]
# Note two things. In the recorded run, the **unfiltered** distilled draft has the higher $\alpha$. This is one run,
# so examine the alpha table before this cell for your run. A higher $\alpha$ for the unfiltered draft is the expected
# direction. The reason is that the verifier filter trained the other draft toward *better* answers than the target
# gives. Better answers are the incorrect goal for a draft.
#
# Also, the $c$ of the tiny models (a quarter of the target) makes a long $k$ high-cost. A real 0.6B draft for a 4B
# target has $c \approx 0.15$ by weight bytes. Thus its best $k$ is larger (Exercise 4.4).
#
# ## Exercise 4.3 — greedy drafting
#
# The default `draft_sample_method` of vLLM 0.30.0 is `"greedy"`. The draft proposes $\operatorname{argmax} q$, and the
# target keeps it with the probability $p(\operatorname{argmax} q)$. Write `greedy_accept(p, q)`. Then calculate
# `perfect`. It is the greedy acceptance of a *perfect* draft ($q$ = $p$) for the primer's target $p$. The check
# prints the two columns of the tiny models side by side.

# %% exercise
def greedy_accept(p, q) -> float:
    ### BEGIN SOLUTION
    return float(p[max(range(len(q)), key=lambda i: q[i])])
    ### END SOLUTION

perfect = None
### BEGIN SOLUTION
perfect = greedy_accept(P_PRIMER, P_PRIMER)
### END SOLUTION

# %% check
assert greedy_accept(P_PRIMER, Q_PRIMER) == 0.05 and perfect == 0.5
assert greedy_accept([0, 1, 0, 0], [0.1, 0.6, 0.2, 0.1]) == 1.0          # a greedy target (T = 0): accept iff argmaxes match
print(table([{"draft": m, "alpha, sampled draft": ALPHA[m], "greedy draft": GREEDY[m], "lost to greedy": round(ALPHA[m] - GREEDY[m], 4)}
             for m in ALPHA], title=f"[{LABEL}] what greedy drafting costs when the target samples at T = 1"))
print("✅ with a sampling target even a perfect draft is accepted only p(argmax p) of the time under greedy drafting; "
      "'draft_sample_method': 'probabilistic' uses Sigma min(p, q) (vLLM v0.30.0; verify)")

# %% [markdown]
# ## Exercise 4.4 — can vLLM run this pair, and at what k?
#
# Write `plan_draft(target_cfg, draft_cfg, alpha)`. If `DR.check_vocab` rejects the pair, return `None`. If not,
# return `(k, speedup)` at the best $k$. For $c$, use the ratio of the parameter counts of the two models
# (`shape(name).params()`). The reason is that a memory-bound decode step streams the weights. The check runs the
# function for two pairs at $\alpha = 0.7$.

# %% exercise
def plan_draft(target: str, draft: str, alpha: float):
    ### BEGIN SOLUTION
    ok, _ = DR.check_vocab(load_config(target), load_config(draft))
    if not ok:
        return None
    c = DR.cost_ratio(shape(draft).params(), shape(target).params())
    k = DR.best_k(alpha, c)
    return k, DR.speedup(alpha, k, c)
    ### END SOLUTION

# %% check
assert plan_draft("qwen2.5-7b-instruct", "qwen2.5-0.5b-instruct", 0.7) is None
k4, sp4 = plan_draft("qwen3-4b", "qwen3-0.6b", 0.7)
c = shape("qwen3-0.6b").params() / shape("qwen3-4b").params()
assert (k4, sp4) == (DR.best_k(0.7, c), DR.speedup(0.7, DR.best_k(0.7, c), c))
rows = [{"alpha": al, "best k": plan_draft("qwen3-4b", "qwen3-0.6b", al)[0],
         "speedup (predicted)": round(plan_draft("qwen3-4b", "qwen3-0.6b", al)[1], 2)} for al in (0.5, 0.6, 0.7, 0.8, 0.9)]
print(table(rows, title=f"PREDICTED: Qwen3-0.6B drafting for Qwen3-4B, c = {c:.3f} (memory-bound, batch 1)"))
print("✅ Qwen2.5-0.5B cannot draft for Qwen2.5-7B (vocab_size 151,936 vs 152,064); Qwen3-0.6B -> 4B can. Each 0.1 of alpha "
      "a distilled draft gains moves the speedup more than any k does")

# %% [markdown]
# The speedups are an upper bound for batch 1. The formula assumes a memory-bound target step. The verification of
# $k$ + 1 tokens costs approximately the same as one token only while this is true. Larger batches make the target
# compute-bound, and the gain decreases. Notebook 05 of the 04 serving lab measures where this occurs.
#
# ## On a real GPU (T1)
#
# On a 24 GB card, serve the target with each draft in turn. Run the same load. Read the counters before and after
# with `read_counters`. `DR.speculative_config` builds the JSON and examines it:

# %%
for draft in ("Qwen/Qwen3-0.6B", "_run_outputs/draft-distilled"):
    spec = DR.speculative_config(draft, 4, draft_sample_method="probabilistic")
    print(shlex.join(DR.serve_args("Qwen/Qwen3-4B", spec, max_model_len=8192, gpu_memory_utilization=0.9)))
if env.server_url():
    live = M.spec_decode(M.parse(""), M.scrape(env.server_url(), headers=env.auth_headers()))
    print("DISTILLAB_URL:", "spec-decode counters since start:" if live.get("drafts") else "no spec-decode counters (no draft configured)",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in live.items() if k != "per_position"})

# %% [markdown]
# To make the distilled draft, serve the *target* first. Collect its answers with no verifier filter and no
# deduplication. Keep every finished answer, correct or incorrect:
#
#     python -m distillab teacher-data --url http://127.0.0.1:8000 --keep all --out _run_outputs/target
#
# Here, the lab's generated problems are a substitute for your traffic. Give a real draft prompts like the prompts that
# it will serve. Then do SFT of Qwen3-0.6B on them. Serve the result as the draft:
#
#     python -m distillab.hf.sft --model Qwen/Qwen3-0.6B --data _run_outputs/target_pc.jsonl --out _run_outputs/draft-distilled
#     DRAFT=_run_outputs/draft-distilled deploy/any-gpu/serve_with_draft.sh
#
# EAGLE-3 heads are the same idea at the feature level. An EAGLE-3 head is a small head. Its training uses a
# soft-target cross-entropy to the frozen target's distribution, from the target's hidden states. The training code is
# in the `eagle` repository. SpecForge trains these heads for vLLM and SGLang. The quoted speedups of EAGLE-3 heads are
# for 13B targets on 2× RTX 3090 (verify before you quote them for your setup).
#
# ## In a design review
#
# **Two minutes:** "A draft model is a student whose only metric is acceptance: $\alpha = \sum \min(p, q)$. The
# speedup comes from $\alpha$, $k$ and the relative cost of the draft. We distil the draft on the target's own outputs.
# We do not filter them for correctness, because the draft's job is to predict the target, also where the target
# makes mistakes. On our toy, this distillation raised $\alpha$ over an off-the-shelf model that its makers trained on
# other data.
#
# "We examine vLLM's counters the correct way. The mean acceptance length is the numerator of the speedup. The logged
# 'acceptance rate' is $\left(E - 1\right)/k$, and $\alpha$ is the rate of position 0. Greedy drafts, the default method
# of vLLM, limit acceptance to $p(\operatorname{argmax} q)$ for a target that samples. The pair must share `vocab_size`.
# A Qwen3-0.6B draft for Qwen3-4B works, and a Qwen2.5-0.5B draft for Qwen2.5-7B does not."
#
# **Drill 1.** *vLLM says that the acceptance rate is 40%. Does speculation work?* Possibly it works well. With $k$ =
# 4, that rate is a mean acceptance length of 2.6, thus 2.6 tokens per target pass. Make the decision from the measured
# inter-token latency at your batch size, not from that rate.
#
# **Drill 2.** *Is it correct to train the draft on verified outputs?* No. Verification makes it a better model and
# a worse predictor of the target. Train it on the target's own samples, at the temperature that you use when you serve.
#
# **Drill 3.** *The distilled draft doubled $\alpha$ at batch 1, but throughput at batch 64 decreased. Why?* At batch
# 64, the target step is compute-bound. The verification of $k$ + 1 tokens per sequence costs up to $k$ + 1 times as
# much, and the extra tokens do not pay for it.
#
# Speculation is a latency tool. Thus, make it depend on the load, as `num_speculative_tokens_per_batch_size` permits
# in vLLM (verify).
