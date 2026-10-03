# distill-lab — distil a student four ways, from a served teacher, with reasoning traces and as a speculative draft, and price it

After this lab, you can train a small model to copy a large one. You can do it with hard labels, logit KD,
sequence-level KD and on-policy GKD, and you can say which one to use. You can also build teacher data from a served
model. You can distil reasoning traces and stay inside the serving budget. You can measure the acceptance of a
distilled draft, and you can calculate if a student pays for itself. Everything runs on a laptop first and on a free
T4 second.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m distillab tinylm`. With torch (the CPU build is
   sufficient), this takes about 90 seconds. It trains a 101K-parameter teacher and four 26K-parameter students on
   the same budget. Then it prints a table of what each one learned. Without torch, it prints a recorded run.
2. Open [`notebooks/01_kd_on_a_tiny_transformer.ipynb`](notebooks/01_kd_on_a_tiny_transformer.ipynb). It explains
   the same run. You write Hinton's loss and the divergence of GKD yourself. The notebook also shows what a verifier
   filter does to what a student inherits. Then it shows why reverse KL can make a student give up thinking.
3. On any GPU, start a teacher with [`deploy/any-gpu/serve_teacher.sh`](deploy/any-gpu/serve_teacher.sh). A free
   Colab or Kaggle T4 is sufficient. Then run `export DISTILLAB_URL=http://127.0.0.1:8000`. After that, notebooks
   02–05 use the real teacher instead of the simulated one.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, and it is free. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2
is a multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, and it is optional.*

Each notebook starts with *the one-minute version*. Then it shows worked examples that use the library. Then it
gives 3–6 exercises (implement the key function, predict a number, select a setting), and a check that prints ✅
comes after each exercise. The notebook ends with *in a design review*. The answers are in
[`solutions/`](solutions/). The concepts are in the [`PRIMER.md`](../PRIMER.md) of the topic.

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`kd_on_a_tiny_transformer`](notebooks/01_kd_on_a_tiny_transformer.ipynb) | T0 with torch (~2 min on a CPU) | Why logit KD does better than SFT on the *same* sequences (the soft target carries the behaviour of the teacher). The T² factor and the high-temperature limit. How a verifier filter makes a SeqKD student do better than its teacher, but agree with it less. The β of GKD in the convention of TRL. Mode covering against mode seeking, and a reverse-KL student that stops thinking. Exposure bias, measured. | [§1 Why distil](../PRIMER.md#1-why-distil), [§2 Soft targets, temperature and the choice of divergence](../PRIMER.md#2-soft-targets-temperature-and-the-choice-of-divergence), [§3 Sequence-level distillation](../PRIMER.md#3-sequence-level-distillation-learning-from-the-teachers-outputs) and [§4 On-policy distillation](../PRIMER.md#4-on-policy-distillation) | ~2.5 h |
| 02 | [`teacher_data_and_a_real_student`](notebooks/02_teacher_data_and_a_real_student.ipynb) | T1 (T0: the simulated fake teacher, and a small student with torch) | The teacher-data pipeline, with its yield and its bill. What an API teacher gives (samples, top-k log-probs, `prompt_logprobs` scores), and why logit KD runs in-process. The per-token on-policy reward. How a student trained on unverified data learns the mistakes of the teacher. What fits on a T4, and why the logits decide it. | [§3](../PRIMER.md#3-sequence-level-distillation-learning-from-the-teachers-outputs), [§4](../PRIMER.md#4-on-policy-distillation) and [§10 Where to run it](../PRIMER.md#10-where-to-run-it) | ~2 h |
| 03 | [`distilling_reasoning_traces_for_real`](notebooks/03_distilling_reasoning_traces_for_real.ipynb) | T1 (T0: bundled traces, which are illustrative, and a small run with torch) | What a student inherits from traces (procedure and length). How the verifier and a length cap change the balance of accuracy, coverage of hard problems and serving cost. How to select a cap and a `max_tokens`. Why a capped student can score at chance on correct-only data. | [§5 Distilling reasoning](../PRIMER.md#5-distilling-reasoning) | ~1.5 h |
| 04 | [`a_distilled_draft_in_vllm`](notebooks/04_a_distilled_draft_in_vllm.ipynb) | T1, 24 GB (T0: small models with torch, and synthetic counters) | Acceptance as the metric of a draft. Why a draft distilled on the unfiltered outputs of the target does better than an off-the-shelf one. How to read the spec-decode counters of vLLM (α, mean acceptance length, the "acceptance rate" that is not α). The upper limit of a greedy draft. Which pairs vLLM accepts, and the best k. | [§7 A distilled draft for speculative decoding](../PRIMER.md#7-a-distilled-draft-for-speculative-decoding) | ~2 h |
| 05 | [`is_the_student_worth_it`](notebooks/05_is_the_student_worth_it.ipynb) | T0 calculators, simulated accuracy (T1: measured throughput, lm-eval) | Agreement against task accuracy. The capability gap per difficulty, with Wilson intervals. Serving cost per million tokens from a roofline decode step. The fixed cost and break-even. A cascade judged by cost per correct answer under an accuracy floor. The five options for a lower-cost model. | [§8 Measuring a student](../PRIMER.md#8-measuring-a-student) and [§9 The economics of a student](../PRIMER.md#9-the-economics-of-a-student) | ~2 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | small torch teachers and students on a CPU, a fake vLLM teacher (simulated), bundled traces and counters (illustrative), roofline and memory calculators (predicted) | every notebook and test |
| **T1** | one GPU: Colab/Kaggle T4 (free), a 24 GB card | `vllm serve Qwen/Qwen2.5-1.5B-Instruct` as the teacher. SFT, logit KD and GKD of `Qwen/Qwen2.5-0.5B-Instruct` (TRL 1.14.0). Traces from `Qwen/Qwen3-1.7B`. `Qwen/Qwen3-4B` with a `Qwen/Qwen3-0.6B` draft on 24 GB. | `DISTILLAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | a multi-GPU box | not necessary: every model here fits one GPU | — |
| **T3** | GCP | a larger teacher that the Cloud Run or GKE deploy of the 04 serving lab serves | [`deploy/gcp/`](deploy/gcp/) |

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

To measure a real teacher instead (T1/T3), start one. [`deploy/any-gpu/`](deploy/any-gpu/) has the Colab/Kaggle T4
recipe. Then run `export DISTILLAB_URL=http://127.0.0.1:8000`. Also set `DISTILLAB_API_KEY`, or set
`DISTILLAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run service.

The lab recognises a `distillab fake` server in `DISTILLAB_URL` by its `/version`, and keeps the label simulated on
it. `DISTILLAB_NO_TORCH=1` shows the recorded-run path, even on a machine with torch. The T1 trainers are
`python -m distillab.hf.sft`, `.kd` and `.gkd`. Their `--dry-run` option prints their configuration on any machine.

## The library (`distillab/`)

| Module | The idea |
|---|---|
| `tinylm/` | The verifiable sum task with an optional scratchpad and two demonstration mixes (pure Python). A small decoder-only transformer. A teacher and four students on one budget (hard labels, logit KD, SeqKD, GKD), exposure bias and a budget-aware trace experiment. A recorded run for machines without torch. |
| `losses.py` | Hinton's KD loss with T² and α (chunked), the generalised JSD of GKD in the convention of TRL, and the per-token on-policy reward. All of them are in torch, and the small models and `hf/kd.py` share them. |
| `data.py` | Generated arithmetic and logic problems with canonical scratchpads, a verifier, train/eval splits and decontamination. |
| `fakeserver.py`, `client.py` | A fake vLLM teacher (simulated): chat with `logprobs`, thinking with `reasoning`, scores from `/v1/completions` with `prompt_logprobs`, and vLLM-named metrics. Also a small client for any OpenAI-compatible teacher. |
| `teacher.py`, `traces.py` | The teacher-data pipeline (n samples, verifier, dedup, length cap, the bill, TRL-format JSONL). Reasoning traces (length statistics, the trade-off of a cap, SFT rows). |
| `draft.py` | Acceptance, expected tokens and speedup (as `minengine.spec`), the spec-decode counters of vLLM, a checked `--speculative-config`, the vocabulary check, and the acceptance of the small models. |
| `agreement.py`, `cost.py` | KL, top-1 agreement, top-k overlap, Wilson intervals, paired flips and the gap per difficulty. Roofline serving cost, the fixed cost of distillation, break-even, the cascade and $/M from `/metrics`. |
| `hf/` | T1: `sft.py` (TRL `SFTTrainer`, LoRA option), `kd.py` (logit KD with the teacher in-process), `gkd.py` (TRL `GKDTrainer` and `DistillationTrainer`), `memory.py` (training memory from `config.json`). |
| `metrics.py`, `report.py`, `env.py`, `__main__.py` | vLLM metric names and a parser, labelled tables and reports (MEASURED, SIMULATED, PREDICTED, ILLUSTRATIVE), tier detection, and the CLI. |

Its siblings: the minimal core of the topic, [`../distill-core/`](../distill-core/). This lab never imports it. Some
formulas already have a home in the repo. This lab implements them again, and
[`tests/test_repo_numbers.py`](tests/test_repo_numbers.py) compares them with that home. These formulas are:

- `minengine.spec` (the α = 0.6 example of the serving primer)
- `roofline.llm` and `roofline.cost` (01 PRIMER §3 and §8)
- `capacity.py`
- `quantcore.eval`
- the Wilson interval of `memory-core` and the 07 agent lab
- the parameter counts of `servelab.sizing`
- the prices of the 06 scaling lab

## Deploy

[`deploy/`](deploy/) has two targets:

- [`any-gpu/`](deploy/any-gpu/). `serve_teacher.sh` runs vLLM with `--max-logprobs` and a reasoning parser for
  thinking teachers, and with `--dtype half` on a T4. `train_student.sh` makes teacher data, then does SFT, logit KD
  or GKD. `serve_with_draft.sh` runs a target with a draft model. The folder also has Colab/Kaggle and 24 GB
  recipes, and RunPod/Vast notes.
- [`gcp/`](deploy/gcp/). It has the Cloud Run Terraform of the 04 lab with the settings of a teacher. It adds no new
  Terraform.

Each target has a README with cost and cleanup.

## Regenerating notebooks

The builder makes `notebooks/` (exercises) and `solutions/` from `notebooks_src/*.py`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network): ~4 min with torch, ~8 s without
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Measured, simulated, predicted, illustrative.** The lab measures the curves of the small models on your machine
  (the recorded fallback says where it came from). The answers, slips and log-probabilities of the fake teacher come
  from a simulation. The fake teacher knows the answers to the generated problems, and it makes a slip with a
  probability that difficulty and temperature set. Serving costs and training memory are calculator predictions at
  ideal bandwidth. Bundled traces and spec-decode counters are illustrative (the counters are synthetic at α = 0.7).
  Every table says which.
- **The toy is not a language model.** A 26K-parameter student on a sum task shows four mechanisms. Soft targets
  carry behaviour, and filters shape what the student inherits. Reverse KL can collapse, and the target accepts a
  distilled draft more often. The toy does not predict the size of any effect on a real model. Measure yours.
- **Checked by construction.** The T1 paths (vLLM teacher and draft, TRL SFT and GKD, the in-process KD trainer) use the
  APIs of vLLM v0.30.0 and TRL 1.14.0. `bash -n`, `DRY_RUN=1`, `--dry-run`, config builders and
  tests examine these paths. The lab also built the trainer configs against TRL 1.14.0 with transformers 5.17.0
  on a CPU. `tests/test_memory_hf.py` does this whenever TRL is on the machine. The paths did not run on a GPU here.

## Verify list (facts dated 2026-09-27 that move)

Checked against the source on 2026-09-27:

- vLLM v0.30.0: the `--speculative-config` fields `method`, `model`, `num_speculative_tokens`,
  `draft_sample_method` (its default is `"greedy"`) and `draft_tensor_parallel_size`. The `draft_model` vocabulary
  check. The spec-decode counter names and their PromQL definitions. `--max-logprobs` 20. `prompt_logprobs` with
  `echo` and `max_tokens: 0`.
- TRL 1.14.0: the `trl.experimental.gkd.GKDConfig` defaults `temperature=0.9`, `lmbda=0.5`, `beta=0.5`,
  `max_new_tokens=128` and `seq_kd=False`. The loss at T = 1 without T². `DistillationTrainer` stable with
  `beta=1.0`. `bf16` on, unless you set `fp16`.

Still to examine on hardware:

* the T4 fits: full SFT of Qwen2.5-0.5B, logit KD with LoRA and a chunked loss, and Qwen3 in fp16 (no NaNs).
* that PEFT keeps LoRA adapters in fp32 over an fp16 base (`hf/sft.py` needs this, and `hf/kd.py` casts them).
* the speedup of a Qwen3-0.6B draft for Qwen3-4B on an L4, off the shelf and distilled.
* model ids and shapes: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
  `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, and the
  bundled configs marked verify.
* prices and GPU availability in [`COMPUTE.md`](../../../COMPUTE.md), and the API prices of the 06 scaling lab (5 Sep
  2026).

MIT licensed.
