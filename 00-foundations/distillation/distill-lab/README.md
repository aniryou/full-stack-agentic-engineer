# distill-lab — distil a student four ways, from a served teacher, with reasoning traces and as a speculative draft, and price it

After this lab you can train a small model to copy a large one with hard labels, logit KD, sequence-level KD and
on-policy GKD, and say which one to use. You can also build teacher data from a served model, distil reasoning
traces without breaking the serving budget, measure a distilled draft's acceptance, and work out whether a student
pays for itself. Everything runs on a laptop first and on a free T4 second.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m distillab tinylm`. With torch (the CPU build is enough)
   this takes about 90 seconds: a 101K-parameter teacher and four 26K-parameter students trained on the same
   budget, and a table of what each learned. Without torch it prints a recorded run.
2. Open [`notebooks/01_kd_on_a_tiny_transformer.ipynb`](notebooks/01_kd_on_a_tiny_transformer.ipynb) for the same
   run explained: Hinton's loss and GKD's divergence written by you, what a verifier filter does to what a student
   inherits, and why reverse KL can make a student give up thinking.
3. On any GPU (a free Colab or Kaggle T4 is enough), start a teacher with
   [`deploy/any-gpu/serve_teacher.sh`](deploy/any-gpu/serve_teacher.sh), `export DISTILLAB_URL=http://127.0.0.1:8000`,
   and notebooks 02–05 use the real teacher instead of the simulated one.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU
box, rented for an hour; T3 = the Google Cloud deployment, optional.* Each notebook opens with *the one-minute
version*, works examples against the library, then 3–6 exercises (implement the key function, predict a number,
pick a setting) each followed by a check that prints ✅, and closes with *in a design review*. Answers are in
[`solutions/`](solutions/). Concepts are in the topic's [`PRIMER.md`](../PRIMER.md).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`kd_on_a_tiny_transformer`](notebooks/01_kd_on_a_tiny_transformer.ipynb) | T0 with torch (~2 min on a CPU) | why logit KD beats SFT on the *same* sequences (the soft target carries the teacher's behaviour); the T² factor and the high-temperature limit; how a verifier filter makes a SeqKD student beat its teacher while agreeing less; GKD's β in TRL's convention; mode covering against mode seeking, and a reverse-KL student that stops thinking; exposure bias, measured | [§1 Why distil](../PRIMER.md#1-why-distil); [§2 Soft targets, temperature and the choice of divergence](../PRIMER.md#2-soft-targets-temperature-and-the-choice-of-divergence); [§3 Sequence-level distillation](../PRIMER.md#3-sequence-level-distillation-learning-from-the-teachers-outputs); [§4 On-policy distillation](../PRIMER.md#4-on-policy-distillation) | ~2.5 h |
| 02 | [`teacher_data_and_a_real_student`](notebooks/02_teacher_data_and_a_real_student.ipynb) | T1 (T0: fake teacher, simulated; a tiny student with torch) | the teacher-data pipeline and its yield and bill; what an API teacher gives (samples, top-k log-probs, `prompt_logprobs` scores) and why logit KD runs in-process; the per-token on-policy reward; a student trained on unverified data learning the teacher's mistakes; what fits on a T4 and why the logits decide it | [§3](../PRIMER.md#3-sequence-level-distillation-learning-from-the-teachers-outputs); [§4](../PRIMER.md#4-on-policy-distillation); [§10 Where to run it](../PRIMER.md#10-where-to-run-it) | ~2 h |
| 03 | [`distilling_reasoning_traces_for_real`](notebooks/03_distilling_reasoning_traces_for_real.ipynb) | T1 (T0: bundled traces, illustrative; a tiny run with torch) | what a student inherits from traces (procedure and length); how the verifier and a length cap trade accuracy, coverage of hard problems and serving cost; choosing a cap and a `max_tokens`; why a capped student can score at chance on correct-only data | [§5 Distilling reasoning](../PRIMER.md#5-distilling-reasoning) | ~1.5 h |
| 04 | [`a_distilled_draft_in_vllm`](notebooks/04_a_distilled_draft_in_vllm.ipynb) | T1, 24 GB (T0: tiny models with torch; synthetic counters) | acceptance as a draft's metric; why a draft distilled on the target's unfiltered outputs beats an off-the-shelf one; reading vLLM's spec-decode counters (α, mean acceptance length, the "acceptance rate" that is not α); greedy drafting's ceiling; which pairs vLLM accepts and the best k | [§7 A distilled draft for speculative decoding](../PRIMER.md#7-a-distilled-draft-for-speculative-decoding) | ~2 h |
| 05 | [`is_the_student_worth_it`](notebooks/05_is_the_student_worth_it.ipynb) | T0 calculators, simulated accuracy (T1: measured throughput, lm-eval) | agreement against task accuracy; the capability gap per difficulty with Wilson intervals; serving cost per million tokens from a roofline decode step; the fixed cost and break-even; a cascade judged by cost per correct answer under an accuracy floor; the five options for a cheaper model | [§8 Measuring a student](../PRIMER.md#8-measuring-a-student); [§9 The economics of a student](../PRIMER.md#9-the-economics-of-a-student) | ~2 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | tiny torch teachers and students on a CPU; a fake vLLM teacher (simulated); bundled traces and counters (illustrative); roofline and memory calculators (predicted) | every notebook and test |
| **T1** | one GPU: Colab/Kaggle T4 (free), a 24 GB card | `vllm serve Qwen/Qwen2.5-1.5B-Instruct` as the teacher; SFT, logit KD and GKD of `Qwen/Qwen2.5-0.5B-Instruct` (TRL 1.14.0); traces from `Qwen/Qwen3-1.7B`; `Qwen/Qwen3-4B` with a `Qwen/Qwen3-0.6B` draft on 24 GB | `DISTILLAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | a multi-GPU box | not needed: every model here fits one GPU | — |
| **T3** | GCP | a larger teacher served by the 04 serving lab's Cloud Run or GKE deploy | [`deploy/gcp/`](deploy/gcp/) |

## Run it

```bash
cd distill-lab
python3 -m pip install -e ".[dev]"           # numpy, aiohttp; dev: pytest, jupyter, matplotlib
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: the tiny models train for real
python3 -m pytest -q                         # 201 tests, offline, no GPU: ~2 min with torch (one full tiny run; -m 'not slow' skips it: ~20 s), ~15 s without torch (the torch and TRL ones skip)
python3 -m distillab tinylm                  # teacher + four students on the tiny task (~90 s on a CPU)
python3 -m distillab fake --port 8000 &      # a fake vLLM teacher (every answer simulated)
DISTILLAB_URL=http://127.0.0.1:8000 python3 -m distillab teacher-data --problems 100 -n 4   # --keep all: a draft's data
python3 -m distillab memory --teacher qwen2.5-1.5b-instruct   # does logit KD of a 0.5B fit a T4? (predicted)
python3 -m distillab cost                    # a 32B teacher (two H100s) against a 1.5B student: $/M and break-even (predicted)
python3 -m distillab spec-config             # the vllm serve command for Qwen3-4B with a Qwen3-0.6B draft
python3 -m jupyterlab notebooks              # the exercises; answers in solutions/
```

Measure a real teacher instead (T1/T3): start one ([`deploy/any-gpu/`](deploy/any-gpu/) has the Colab/Kaggle T4
recipe), then `export DISTILLAB_URL=http://127.0.0.1:8000` (plus `DISTILLAB_API_KEY`, or
`DISTILLAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run service). A `distillab fake` server
in `DISTILLAB_URL` is recognised by its `/version` and stays labelled simulated. `DISTILLAB_NO_TORCH=1` shows the
recorded-run path even when torch is installed. The T1 trainers are `python -m distillab.hf.sft`, `.kd` and `.gkd`
(`--dry-run` prints their configuration anywhere).

## The library (`distillab/`)

| Module | The idea |
|---|---|
| `tinylm/` | the verifiable sum task with an optional scratchpad and two demonstration mixes (pure Python); a tiny decoder-only transformer; a teacher and four students on one budget (hard labels, logit KD, SeqKD, GKD), exposure bias, a budget-aware trace experiment; a recorded run for machines without torch |
| `losses.py` | Hinton's KD loss with T² and α (chunked), GKD's generalised JSD in TRL's convention, the per-token on-policy reward — torch, shared by the tiny models and `hf/kd.py` |
| `data.py` | generated arithmetic and logic problems with canonical scratchpads, a verifier, train/eval splits and decontamination |
| `fakeserver.py`, `client.py` | a fake vLLM teacher (chat with `logprobs`, thinking with `reasoning`, `/v1/completions` scoring with `prompt_logprobs`, vLLM-named metrics; simulated) and a small client for any OpenAI-compatible teacher |
| `teacher.py`, `traces.py` | the teacher-data pipeline (n samples, verifier, dedup, length cap, the bill, TRL-format JSONL); reasoning traces (length statistics, the cap trade, SFT rows) |
| `draft.py` | acceptance, expected tokens and speedup (as `minengine.spec`), vLLM's spec-decode counters, a checked `--speculative-config`, the vocabulary check, the tiny models' acceptance |
| `agreement.py`, `cost.py` | KL, top-1 agreement, top-k overlap, Wilson intervals, paired flips, the gap per difficulty; roofline serving cost, the fixed cost of distilling, break-even, the cascade, $/M from `/metrics` |
| `hf/` | T1: `sft.py` (TRL `SFTTrainer`, LoRA option), `kd.py` (logit KD with the teacher in-process), `gkd.py` (TRL `GKDTrainer` and `DistillationTrainer`), `memory.py` (training memory from `config.json`) |
| `metrics.py`, `report.py`, `env.py`, `__main__.py` | vLLM metric names and a parser; labelled tables and reports (MEASURED, SIMULATED, PREDICTED, ILLUSTRATIVE); tier detection; the CLI |

Its siblings: the topic's minimal core, [`../distill-core/`](../distill-core/), which this lab never imports. Formulas
that already have a home in the repo are re-implemented and checked against it in
[`tests/test_repo_numbers.py`](tests/test_repo_numbers.py): `minengine.spec` (the serving primer's α = 0.6 example),
`roofline.llm` and `roofline.cost` (01 PRIMER §3 and §8), `capacity.py`, `quantcore.eval`, the Wilson interval of
`memory-core` and the 07 agent lab, `servelab.sizing`'s parameter counts and the 06 scaling lab's prices.

## Deploy

[`deploy/`](deploy/): [`any-gpu/`](deploy/any-gpu/) (`serve_teacher.sh`: vLLM with `--max-logprobs` and a reasoning
parser for thinking teachers, `--dtype half` on a T4; `train_student.sh`: teacher data then SFT, logit KD or GKD;
`serve_with_draft.sh`: a target with a draft model; Colab/Kaggle and 24 GB recipes; RunPod/Vast notes) and
[`gcp/`](deploy/gcp/) (the 04 lab's Cloud Run Terraform with a teacher's settings; no new Terraform). Each has a
README with cost and cleanup.

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network): ~4 min with torch, ~8 s without
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Measured, simulated, predicted, illustrative.** The tiny models' curves are measured on your machine (the
  recorded fallback says where it came from). The fake teacher's answers, slips and log-probabilities are
  simulated: it knows the answers to the generated problems and slips with a probability set by difficulty and
  temperature. Serving costs and training memory are calculator predictions at ideal bandwidth. Bundled traces
  and spec-decode counters are illustrative (the counters are synthetic at α = 0.7). Every table says which.
- **The toy is not a language model.** A 26K-parameter student on a sum task shows the mechanisms: soft targets
  carry behaviour, filters shape what is inherited, reverse KL can collapse, a distilled draft is accepted more
  often. It does not predict the size of any effect on a real model. Measure yours.
- **Checked by construction.** The T1 paths (vLLM teacher and draft, TRL SFT and GKD, the in-process KD trainer)
  are written against vLLM v0.30.0 and TRL 1.14.0 and checked with `bash -n`, `DRY_RUN=1`, `--dry-run`, config
  builders and tests; the trainer configs were also built against TRL 1.14.0 with transformers 5.17.0 on a CPU
  (`tests/test_memory_hf.py` does it whenever TRL is installed). They were not run on a GPU here.

## Verify list (facts dated 2026-09-27 that move)

Checked against source on 2026-09-27: vLLM v0.30.0 (`--speculative-config` fields `method`, `model`,
`num_speculative_tokens`, `draft_sample_method` defaulting to `"greedy"`, `draft_tensor_parallel_size`; the
`draft_model` vocabulary check; the spec-decode counter names and their PromQL definitions; `--max-logprobs` 20;
`prompt_logprobs` with `echo` and `max_tokens: 0`); TRL 1.14.0 (`trl.experimental.gkd.GKDConfig` defaults
`temperature=0.9`, `lmbda=0.5`, `beta=0.5`, `max_new_tokens=128`, `seq_kd=False`; the loss at T = 1 without T²;
`DistillationTrainer` stable with `beta=1.0`; `bf16` on unless `fp16` is set). Still to verify on hardware:

* the T4 fits: full SFT of Qwen2.5-0.5B, logit KD with LoRA and a chunked loss, Qwen3 in fp16 (no NaNs);
* that PEFT keeps LoRA adapters in fp32 over an fp16 base (`hf/sft.py` relies on it; `hf/kd.py` casts them);
* the speedup of a Qwen3-0.6B draft for Qwen3-4B on an L4, off the shelf and distilled;
* model ids and shapes: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
  `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; the
  bundled configs marked verify;
* prices and GPU availability in [`COMPUTE.md`](../../../COMPUTE.md); the 06 scaling lab's API prices (5 Sep 2026).

MIT licensed.
