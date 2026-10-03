# %% [markdown]
# # 02 · Teacher data and a real student: from a served teacher to a 0.5B student on a T4
#
# **Tier:** T1: `vllm serve Qwen/Qwen2.5-1.5B-Instruct` generates the data on a Colab or Kaggle T4. Then SFT and
# logit KD train `Qwen/Qwen2.5-0.5B-Instruct` on the same card (model ids and fits: verify). T0 (default): the same
# pipeline runs against this lab's fake teacher, whose answers and log-probabilities are **simulated**. A small
# student trains on its answers with torch on a CPU in about 15 seconds. The memory plans are **predicted** values
# from a calculator.
#
# ## The one-minute version
#
# * **Sequence-level distillation's dataset is a pipeline with a yield at every stage:**
#   1. prompts × $n$ samples,
#   2. then the verifier,
#   3. then deduplication,
#   4. then a length cap,
#   5. then JSONL.
#
#   You pay for every token that the teacher generated, not only for the kept tokens (PRIMER §3 "Sequence-level
#   distillation: learning from the teacher's outputs").
# * **A served teacher gives you three things:**
#   * samples,
#   * the top-k log-probabilities of each sampled token (at most 20 by default in vLLM),
#   * and its log-probability of *any* text that you send (`prompt_logprobs`), which is the per-token reward of
#     on-policy distillation (PRIMER §4 "On-policy distillation").
#
#   It does not give full distributions, so logit KD runs the teacher in-process.
# * **The verifier is not optional.** A student that trains on unfiltered teacher data learns the teacher's mistakes
#   at the teacher's rate.
# * **On a T4:** there is no bf16, and the trainable weights are in fp32 with fp16 autocast. At a 151,936-token
#   vocabulary, the logits are the largest activation. Full logit KD of a 0.5B student from a 1.5B teacher does not
#   fit without a chunked loss. With LoRA or a chunked loss, it fits (PRIMER §10 "Where to run it").

# %%
import json, math, os, random, statistics
from distillab import data as D, env, teacher as TE
from distillab.client import QWEN25_SYSTEM, Client
from distillab.cost import load_config
from distillab.fakeserver import FakeTeacher
from distillab.hf import memory as H
from distillab.report import table

print(env.describe())
target = env.connect()
teacher = Client(target.url, target.model, target.headers)
LABEL = target.label
print(target)

# %% [markdown]
# ## Worked example: one teacher sample, with its top-k log-probabilities
#
# `top_logprobs: 5` asks for the five most likely tokens at every position of the sample. That is the most logit
# information that an OpenAI-compatible API returns. vLLM rejects a request for more than `--max-logprobs` (20 by
# default). If you increase the limit to the full vocabulary (`-1`), there is a risk that the server runs out of
# memory.

# %%
p = D.make_set(8, seed=5, split="train")[5]
c = teacher.chat(p.messages(), temperature=0.7, top_logprobs=5, seed=1)[0]
print(f"[{LABEL}] {p.question}\n{c.content}\n")
for (tok, lp), top in list(zip(c.logprobs, c.top_logprobs))[:4]:
    print(f"{tok!r:>12}  p = {math.exp(lp):.3f}   top-5: " + ", ".join(f"{t!r} {math.exp(l):.3f}" for t, l in top))
err = teacher.chat(p.messages(), top_logprobs=21)[0].error
print("\nasking for 21:", err[:160])

# %% [markdown]
# ## Exercise 2.1 — a logit target from top-k log-probabilities
#
# Logit KD through an API has only the top k. Write `topk_target(top)` for one position. Take the `(token,
# logprob)` pairs and renormalise their probabilities to a sum of 1. Return `(target, missing)`, where `target`
# maps token to probability and `missing` is the mass that the API did not return. A renormalisation gives that mass
# to the top k. Thus the student learns from a sharper teacher than the real one.

# %% exercise
def topk_target(top: list) -> tuple:
    ### BEGIN SOLUTION
    probs = {t: math.exp(lp) for t, lp in top}
    total = sum(probs.values())
    return {t: v / total for t, v in probs.items()}, max(0.0, 1.0 - total)
    ### END SOLUTION

# %% check
tgt, miss = topk_target([("a", math.log(0.5)), ("b", math.log(0.3))])
assert abs(tgt["a"] - 0.625) < 1e-12 and abs(miss - 0.2) < 1e-12
for top in c.top_logprobs:
    mine, lib = topk_target(top), TE.topk_target(top)
    assert all(abs(mine[0][t] - lib[0][t]) < 1e-12 for t in lib[0]) and abs(mine[1] - lib[1]) < 1e-12
missing = [topk_target(top)[1] for top in c.top_logprobs]
print(f"✅ [{LABEL}] top-5 leaves out {statistics.fmean(missing):.1%} of the mass on average (max {max(missing):.1%}); "
      "a truncated KD target is an approximation: say so, or run the teacher in-process (distillab.hf.kd)")

# %% [markdown]
# ## Worked example: the teacher-data pipeline
#
# The training problems come from `distillab.data`, and the pipeline decontaminates them against the eval split. The
# teacher answers each problem four times at temperature 0.7. The bill includes everything that the teacher
# generates. The verifier, deduplication and a length cap decide what the pipeline keeps. The per-kind table shows
# where the verifier discards the most samples.

# %%
TRAIN, dropped = D.decontaminate(D.make_set(200, seed=0, split="train"), D.make_set(200, seed=0))
SAMPLES = TE.generate(teacher, TRAIN, n=4, temperature=0.7, max_tokens=512)
KEPT, funnel = TE.funnel(SAMPLES, max_completion_tokens=48)
print(f"{len(dropped)} training problems dropped: their question is also in the eval set")
print(table(funnel, title=f"[{LABEL}] {len(TRAIN)} problems x 4 samples from {target.model}"))
print(table([TE.token_bill(SAMPLES)], title="What the teacher generated (all of it is billed)"))
print(table([r for r in TE.by_kind(SAMPLES) if r["difficulty"] >= 3], title="Teacher accuracy on the harder problems"))

# %% [markdown]
# ## Exercise 2.2 — the bill, per kept sample
#
# Use the 06 scaling lab's price for `gemini-3.5-flash` (\$1.50 per million input tokens and \$9.00 per million
# output, checked 5 Sep 2026, verify). At this price, what did this dataset cost, and what did each *kept* sample
# cost? Write `data_cost(samples, kept, model)`, which returns `(total_dollars, dollars_per_kept)`. You pay for every
# generated sample, kept or not.

# %% exercise
def data_cost(samples: list, kept: list, model: str = "gemini-3.5-flash") -> tuple:
    ### BEGIN SOLUTION
    total = sum(TE.api_cost(model, s.prompt_tokens, s.completion_tokens) for s in samples)
    return total, total / max(1, len(kept))
    ### END SOLUTION

# %% check
total, per_kept = data_cost(SAMPLES, KEPT)
inp, out = sum(s.prompt_tokens for s in SAMPLES), sum(s.completion_tokens for s in SAMPLES)
assert abs(total - (inp * 1.50 + out * 9.00) / 1e6) < 1e-9 and abs(per_kept - total / len(KEPT)) < 1e-12
assert abs(TE.api_cost("gemini-3.5-flash", 0, 2e8) - 1800) < 1e-6
print(f"✅ [{LABEL}] ${total:.4f} for {len(SAMPLES)} samples, ${per_kept * 1e3:.3f} per 1,000 kept. At the scale of a "
      "real run, 100k prompts x 2,000 output tokens is $1,800 at this price; training a 1.5B student on those 2e8 "
      "tokens is ~1.3 H100-hours ($14 at $11/h): the teacher's tokens dominate (notebook 05)")

# %% [markdown]
# ## Worked example: the rows TRL reads
#
# TRL reads two formats (TRL 1.14.0's `dataset_formats.md`, verify):
#
# * *prompt-completion* for `SFTTrainer`, where the loss applies to the completion only,
# * *conversational* `messages` for `GKDTrainer` and `distillab.hf.kd`.

# %%
PC, MSG = TE.to_prompt_completion(KEPT), TE.to_messages(KEPT)
assert TE.check_rows(PC) == [] and TE.check_rows(MSG) == []
print(json.dumps(PC[0], indent=1)[:600])
PATH_PC = TE.write_jsonl(PC, "_run_outputs/teacher_pc.jsonl")
PATH_MSG = TE.write_jsonl(MSG, "_run_outputs/teacher_msgs.jsonl")
print(f"\nwrote {PATH_PC} ({len(PC)} rows) and {PATH_MSG}")

# %% [markdown]
# ## Worked example: the teacher scores a student's answer
#
# On-policy distillation samples from the *student* and asks the teacher for its log-probability of every sampled
# token. Through an API, that is `/v1/completions` with the chat-templated prompt plus the student's text,
# `echo: true`, `max_tokens: 0` and `prompt_logprobs: 0` (`Client.score`). At T0, a second fake server acts as a
# weaker student (more slips). At T1, set `DISTILLAB_STUDENT_URL` to a second server with the student model (same
# tokenizer). The two token lists must align, and that is why on-policy distillation needs a shared tokenizer.

# %%
student_srv = None
if os.environ.get("DISTILLAB_STUDENT_URL"):
    su = os.environ["DISTILLAB_STUDENT_URL"]
    student = Client(su, env.first_model(su, env.auth_headers()), env.auth_headers())
else:
    student_srv = FakeTeacher("student")
    student = Client(student_srv.start(), student_srv.model)
hard = [q for q in D.make_set(80, seed=7) if q.kind == "arith" and q.difficulty == 4]
for i in range(40):
    q = hard[i % len(hard)]
    s = student.chat(q.messages(), temperature=1.0, seed=i, top_logprobs=0)[0]
    if not D.verify(q, s.content):
        break
SCORED = teacher.score(q.messages(), s.content, default_system=None if "Qwen3" in (target.model or "") else QWEN25_SYSTEM)
ALIGNED = [t for t, _ in SCORED] == [t for t, _ in s.logprobs]
print(f"[{LABEL}] {q.question}  (answer {q.answer})\nstudent ({student.model}): {s.content!r}")
print("token lists line up" if ALIGNED else "token lists do NOT line up: teacher and student tokenize differently")

# %% [markdown]
# ## Exercise 2.3 — the per-token reward
#
# You have the student's own log-probabilities `s.logprobs` and the teacher's `SCORED`. Both are lists of
# `(token, logprob)` pairs. Write `rewards(scored, own)`. It returns the per-token reward
# $r_t = \log \pi_{\text{teacher}} - \log \pi_{\text{student}}$ as a list. Then set `worst` to the index of the most
# negative reward. The total reward of the sequence is minus its log-ratio, and its expectation over the student's
# samples is minus the reverse KL (PRIMER §4).

# %% exercise
def rewards(scored: list, own: list) -> list:
    ### BEGIN SOLUTION
    return [lt - ls for (_, lt), (_, ls) in zip(scored, own)]
    ### END SOLUTION

### BEGIN SOLUTION
R = rewards(SCORED, s.logprobs)
worst = min(range(len(R)), key=R.__getitem__)
### END SOLUTION

# %% check
ref = [a[1] - b[1] for a, b in zip(SCORED, s.logprobs)]
assert R == ref and worst == ref.index(min(ref))
if ALIGNED:
    assert abs(sum(R) - (sum(v for _, v in SCORED) - sum(v for _, v in s.logprobs))) < 1e-9
else:
    print("(the rewards pair tokens by position only; with two tokenizers they are not the same tokens)")
print(table([{"t": i, "token": repr(SCORED[i][0]), "log p_teacher": round(SCORED[i][1], 2), "log p_student": round(s.logprobs[i][1], 2),
              "reward": round(R[i], 2)} for i in range(max(0, worst - 3), min(len(R), worst + 3))],
            title=f"[{LABEL}] rewards around the worst token"))
print(f"✅ the most negative reward is at token {worst} ({SCORED[worst][0]!r}): where the student left the teacher's "
      "working. GRPO's verifier would give this whole answer a single 0; the teacher's scores point at the step")

# %% [markdown]
# ## Worked example: a tiny student trained on the teacher's answers
#
# The `modsum` problems at difficulty 2 are the task of notebook 01: six base-5 digits. The teacher's answers to 600
# of them convert to the tokens of the tiny transformer of notebook 01 (`from_teacher_text`). This cell always uses the fake teacher,
# because a real model writes its steps in its own way. Two students train with torch. One trains on everything that
# the teacher said, and one trains only on the answers that the verifier accepted. Without torch, the cell prints the
# counts of the pipeline.

# %%
from distillab.tinylm.task import from_teacher_text
rng = random.Random(0)
MOD = [D.make_problem("modsum", 2, rng, i, "train") for i in range(600)]
FAKE = FakeTeacher()                     # the tiny run needs answers in the tiny task's format: the fake teacher's
fake_teacher = Client(FAKE.start(), FAKE.model)
MS = TE.generate(fake_teacher, MOD, n=1, temperature=1.0)
FAKE.stop()
PAIRS = {"every answer": [x for s in MS if (x := from_teacher_text(s.question, s.content))],
         "verified only": [x for s in MS if s.correct and (x := from_teacher_text(s.question, s.content))]}
print(f"[SIMULATED] {sum(s.correct for s in MS)} of {len(MS)} fake-teacher answers verified")
TINY = {}
if env.has_torch():
    from distillab.tinylm import train as T
    for name, pairs in PAIRS.items():
        TINY[name] = T.sft_student(pairs, T.DistillConfig(), steps=800)
    print(table([{"trained on": k, **{c: v[c] for c in ("pairs", "accuracy", "full", "length", "seconds")}} for k, v in TINY.items()],
                title="MEASURED: tiny students, SFT on the teacher's answers"))
else:
    print("torch is missing: no tiny students (pip install torch, the CPU build is enough)")

# %% [markdown]
# The unfiltered student makes the teacher's mistakes at about the teacher's rate. It learned *how* to slip from
# examples of slips. The student that trained on accepted answers makes almost no mistakes. A filter costs you the
# tokens of the rejected samples, and it gives you correctness. On real data, the verifier is a test suite, an
# exact-match checker or a judge. Also, the errors of a judge go into the data in the same way as the errors of the
# teacher.
#
# ## Exercise 2.4 — what fits on a T4
#
# `H.plan(student, gpu=..., regime=..., batch=..., seq=..., teacher=..., chunk=...)` predicts the training memory
# (`distillab.hf.memory`: 16 bytes per parameter for full fine-tuning with AdamW, 2 for a frozen 16-bit model, fp32
# logits). Write `max_batch(student, teacher, regime, chunk, seq=512, gpu="T4")`. It returns the largest batch from 1
# to 64 whose plan fits, or 0 if no batch fits. The check compares SFT, logit KD with and without a chunked loss, and
# KD with LoRA.

# %% exercise
def max_batch(student: dict, teacher, regime: str, chunk, seq: int = 512, gpu: str = "T4") -> int:
    ### BEGIN SOLUTION
    best = 0
    for b in range(1, 65):
        if not H.plan(student, gpu=gpu, regime=regime, batch=b, seq=seq, teacher=teacher, chunk=chunk)["fits"]:
            break
        best = b
    return best
    ### END SOLUTION

# %% check
S05, T15 = load_config("qwen2.5-0.5b-instruct"), load_config("qwen2.5-1.5b-instruct")
cases = {"SFT, full fine-tuning": (None, "full", None), "logit KD, full, unchunked": (T15, "full", None),
         "logit KD, full, chunk 256": (T15, "full", 256), "logit KD, LoRA r=16, chunk 256": (T15, "lora", 256)}
fits = {k: max_batch(S05, t, r, ch) for k, (t, r, ch) in cases.items()}
for k, (t, r, ch) in cases.items():
    ref = max([b for b in range(1, 65) if H.plan(S05, gpu="T4", regime=r, batch=b, seq=512, teacher=t, chunk=ch)["fits"]] or [0])
    assert fits[k] == ref, (k, fits[k], ref)
assert fits["logit KD, full, unchunked"] < fits["SFT, full fine-tuning"] <= fits["logit KD, full, chunk 256"] + 64
assert fits["logit KD, full, chunk 256"] > fits["logit KD, full, unchunked"] and fits["logit KD, LoRA r=16, chunk 256"] >= fits["logit KD, full, chunk 256"]
print(table([{"run (0.5B student, 512 tokens, T4)": k, "largest batch (predicted)": v} for k, v in fits.items()]))
print("✅ the teacher's weights cost 3.1 GB, but its unchunked logits cost more per sequence: chunk the loss, then "
      "use LoRA for room (predictions; verify on the card)")

# %% [markdown]
# ## On a real GPU (T1)
#
# On a Colab or Kaggle T4, run these commands. The recipe is in `deploy/any-gpu/` (model ids and fits: verify):
#
# ```bash
# # 1. the teacher, with logprobs allowed (vLLM's default cap is 20)
# vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 --max-logprobs 20 --port 8000 &
# export DISTILLAB_URL=http://127.0.0.1:8000          # this notebook now measures the real teacher
# python -m distillab teacher-data --problems 2000 -n 4 --out _run_outputs/teacher
# # 2. stop vLLM (the T4 has room for one of the two), then the student
# python -m distillab.hf.sft --model Qwen/Qwen2.5-0.5B-Instruct --data _run_outputs/teacher_pc.jsonl --out _run_outputs/student-sft
# python -m distillab.hf.kd --data _run_outputs/teacher_msgs.jsonl --out _run_outputs/student-kd --lora-r 16 --chunk 256
# ```
#
# Notebook 05 compares the accuracy of the student against the accuracy of the teacher. The training of Qwen2.5 and
# Qwen3 used bf16, and a T4 runs fp16. Thus, examine the first losses for `nan` before you trust a run (verify per
# model).

# %%
from distillab.hf import kd as KD, sft as SFT
print("SFTConfig on a T4:", SFT.sft_kwargs(out="_run_outputs/student-sft", gpu="T4"))
print("pair check:", KD.check_pair(T15, S05) or "same vocab_size and family: logit KD is possible")
print("a pair that is not:", KD.check_pair(load_config("qwen2.5-7b-instruct"), S05))
if student_srv is not None:
    student_srv.stop()
target.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Our distillation data is a pipeline with a yield at each stage. We take four samples per prompt
# from a 1.5B teacher. Then the pipeline applies a verifier, deduplication and a length cap. We report the yield and
# the bill at every stage, because we pay for every generated token. The verifier is necessary. A student that trains
# on the teacher's unfiltered answers learns its mistakes at the teacher's rate.
#
# "For on-policy distillation, the API teacher scores the student's own samples with `prompt_logprobs`. This gives a
# per-token reward that points at the step where the student made an error. Logit KD needs full distributions, so
# there the teacher runs in-process. On a T4, that means a chunked loss and LoRA."
#
# **Drill 1.** *Why not do logit KD through the API?* It returns at most the top 20 log-probabilities per token by
# default. A renormalisation of them gives the mass of all other tokens to the top k, and it teaches an over-confident target.
# Use samples and scores through the API. Run the teacher in-process for logits.
#
# **Drill 2.** *Our SFT data yield is 60%. Is that bad?* It is a cost, not a defect: you paid for 100% of the tokens.
# Find where the rejects are (by kind and difficulty). If the hard problems lose all their samples, increase $n$ for
# them, or the student never sees them (PRIMER §5).
#
# **Drill 3.** *Can a Qwen2.5-7B teacher logit-distil into Qwen2.5-0.5B?* Not as it is. The `vocab_size` is 152,064
# against 151,936. TRL's GKD and vLLM's draft check both reject the pair, although the token ids agree. Use SeqKD
# (text is tokenizer-agnostic), a same-size-class teacher, or a cross-tokenizer method such as TRL's GOLD (PRIMER §6
# "Feature distillation, pruning and vocabulary mismatch").
