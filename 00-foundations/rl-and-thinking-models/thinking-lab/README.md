# thinking-lab — watch RL teach a model to think, then serve a thinking model and price what thinking costs

After this lab you can train a tiny transformer with GRPO, serve a real thinking model with the right
switches, parsers and budgets, measure what extra samples and extra thinking buy, and say what long
outputs do to ITL, KV capacity and cost, first on a laptop and then on a free T4.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m thinklab tinyrl`. With torch (the CPU build is
   enough) this runs in about a minute: a 100K-parameter transformer goes from thinking half the time
   to always thinking, and its reward and completion length rise together. Without torch it prints a
   recorded run.
2. Open [`notebooks/01_grpo_on_a_tiny_transformer.ipynb`](notebooks/01_grpo_on_a_tiny_transformer.ipynb)
   on your laptop for the same run explained: the verifier, group-relative advantages, and the curves your
   machine produced.
3. On any GPU (a free Colab T4 is enough), start a real thinking model with
   [`deploy/any-gpu/serve.sh`](deploy/any-gpu/serve.sh), `export THINKLAB_URL=http://127.0.0.1:8000`, and
   notebooks 02–04 measure it instead of the simulator.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a
multi-GPU box, rented for an hour; T3 = the Google Cloud deployment, optional.* Each notebook opens with
*the one-minute version*, works examples against the library, then 3–6 exercises (implement the key
function, predict a number, pick a setting) each followed by a check that prints ✅, and closes with
*in a design review*. Answers are in [`solutions/`](solutions/). Concepts are in the topic's
[`PRIMER.md`](../PRIMER.md).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`grpo_on_a_tiny_transformer`](notebooks/01_grpo_on_a_tiny_transformer.ipynb) | T0 with torch (T1 faster) | why a scratchpad adds serial computation; an outcome verifier; group-relative advantages exactly as TRL computes them; the reward-and-length curves of a real (tiny) GRPO run; the `grpo` / `dr_grpo` / `dapo` normalisations and their length bias; the k3 KL estimator; why `clip_frac` is 0 at `num_iterations=1` | §2, §4, §5 | ~2 h |
| 02 | [`a_thinking_model_on_one_gpu`](notebooks/02_a_thinking_model_on_one_gpu.ipynb) | T1 (T0: bundled samples + fake server) | sizing Qwen3 on a T4 (an 8K trace is 0.94 GB of KV); `reasoning` vs `reasoning_content`; parsing R1-style output; when the answer starts in a stream; the three switches (`enable_thinking`, `thinking_token_budget`, `reasoning_effort`); the `max_tokens` trap; choosing a budget from an accuracy curve | §5, §7 | ~2 h |
| 03 | [`test_time_compute_for_real`](notebooks/03_test_time_compute_for_real.ipynb) | T1 (T0: bundled simulated records) | unbiased pass@k; pass^k; majority vote and when it hurts; best-of-n with a verifier vs a noisy reward model; cost per correct answer; the most accurate option for a token budget | §6 | ~1.5 h |
| 04 | [`serving_thinking_models`](notebooks/04_serving_thinking_models.ipynb) | T0 fake server + emulator / T1 real vLLM | heavy-tailed output lengths; KV × time growing as P·L + L²/2; the capacity primer's numbers with 10× outputs (≈18× the GPUs for KV); the batch the KV pool allows and the ITL it gives; budgets vs `max_model_len`; why the prefix cache stops at the last assistant header; routing by effort | §7 | ~2.5 h |
| 05 | [`rl_rollouts_with_an_engine`](notebooks/05_rl_rollouts_with_an_engine.ipynb) | T1 (T0: the tiny model's rollouts) | the rollout as an inference workload; advantages and dynamic sampling; the train–inference log-prob mismatch and TRL's `sequence_mask` correction; DAPO's overlong penalty; straggler idle time in a synchronous rollout batch; weight-sync bytes | §4, §8 | ~2 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the tiny-transformer GRPO (torch CPU), a fake vLLM with a simulated thinking model, the engine emulator, bundled samples | every notebook and test |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3` (1.7B, 4B, R1-Distill-1.5B also fit); one GRPO step with vLLM rollouts | `THINKLAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | a multi-GPU box | not needed here: TRL's server mode puts vLLM on its own GPUs (layer 02 covers the collectives that sync weights) | — |
| **T3** | GCP | the 04 serving lab's Cloud Run or GKE deploy with Qwen3-4B and a reasoning parser | [`deploy/gcp/`](deploy/gcp/) |

## Run it

```bash
cd thinking-lab
python3 -m pip install -e ".[dev]"             # aiohttp; dev: pytest, jupyter, pyyaml, matplotlib
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: notebooks 01/05 train for real
python3 -m pytest -q                           # 87 tests, ~10 s, offline, no GPU
python3 -m thinklab tinyrl                     # SFT + GRPO on the tiny transformer (~1 min on a CPU)
python3 -m thinklab fake --port 8000 &         # a fake vLLM serving a simulated Qwen3-0.6B on a T4
THINKLAB_URL=http://127.0.0.1:8000 python3 -m thinklab ask "What is 47 * 23 - 318?"
python3 -m thinklab eval -n 20                 # thinking off / on / budgeted on the generated eval set
python3 -m thinklab shape                      # the serving shape of thinking vs not (virtual time; simulated)
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

Measure a real engine instead (T1/T3): start one ([`deploy/any-gpu/`](deploy/any-gpu/) has the Colab/Kaggle
T4 recipe), then `export THINKLAB_URL=http://127.0.0.1:8000` (plus `THINKLAB_API_KEY`, or
`THINKLAB_BEARER=$(gcloud auth print-identity-token)` for a private Cloud Run service). The notebooks
detect it and measure it. A `thinklab fake` server in `THINKLAB_URL` is recognised by its `/version`
and stays labelled simulated. `THINKLAB_NO_TORCH=1` shows the recorded-run path even when torch is
installed.

## The library (`thinklab/`, ~3,600 lines)

| Module | Lines | The idea |
|---|---:|---|
| `tinyrl/` | ~540 | the scratchpad task and its verifier (pure Python); a tiny decoder-only transformer mirroring `00-foundations/transformers/lessons/03_tiny_gpt.py`; SFT warm-up then GRPO with TRL's names and defaults; rollouts from a bf16 "engine" copy re-scored by an fp32 "trainer" copy; a recorded run for machines without torch |
| `thinking/` | ~680 | an OpenAI-compatible client that reads `reasoning` *and* `reasoning_content` and times the answer's start; request bodies for the thinking switch, budgets and effort; budget strategies (truncate, native, Qwen's two-call recipe); pass@k, pass^k, majority vote, best-of-n, cost per correct answer; a generated eval set of verifiable problems; recorded simulated outcomes |
| `parsers.py` | ~180 | vLLM's `deepseek_r1` and `qwen3` semantics, gpt-oss Harmony channels, a streaming splitter that holds back split tags, answer extraction |
| `templates.py` | ~110 | Qwen3's history rendering (reasoning dropped before the last user message) and the cached-prefix arithmetic that follows |
| `fakemodel.py` | ~160 | the simulated thinking model: log-normal thinking lengths by difficulty, accuracy rising with thinking, distractor answers |
| `engine.py` | ~270 | the timing emulator: FCFS continuous batching, KV blocks, recompute preemption, a roofline step time; Qwen3 profiles whose numbers reproduce `servelab.sizing` |
| `fakeserver.py` | ~460 | the fake vLLM over HTTP: chat completions (streaming, `n`, `continue_final_message`), the reasoning switches and budgets, `usage.completion_tokens_details.reasoning_tokens`, cached tokens, `/metrics` with vLLM's names |
| `workload.py` | ~370 | length statistics, the steady-state serving shape, the capacity primer's arithmetic, open-loop load with gauge polling, modes compared in virtual time |
| `rollout.py` | ~250 | the RL step's bookkeeping (advantages, dynamic sampling, IS corrections, DAPO shaping, aggregations, straggler time, weight sync) and the T1 vLLM + transformers step |
| `metrics.py`, `report.py`, `env.py`, `__main__.py` | ~570 | Prometheus writer/parser and PromQL quantiles; labelled tables and reports; tier detection; the CLI |

Its siblings: the topic's minimal core, [`../rl-core/`](../rl-core/), which this lab never imports; the
04 serving lab ([`vllm-serving-lab`](../../../04-inference-engine/serving-engine/vllm-serving-lab/)), whose
fake server, load generator and metric names this lab's copies follow; and the capacity primer
([`00-foundations/gpu-capacity-planning`](../../gpu-capacity-planning/PRIMER.md)), whose arithmetic
`workload.capacity_primer_view` reproduces (`tests/test_reuse.py` checks it against `capacity.py`,
`servelab.sizing` and the 07 agent lab's Wilson interval).

## Deploy

[`deploy/`](deploy/): [`any-gpu/`](deploy/any-gpu/) (`serve.sh`: docker or pip `vllm serve` with the right
reasoning parser, budget config and `--dtype half` on a T4; `rl_step.sh`: one GRPO step with vLLM
rollouts; Colab/Kaggle and 24 GB recipes; RunPod/Vast notes) and [`gcp/`](deploy/gcp/) (the 04 lab's
Cloud Run Terraform and GKE manifests with a thinking model's settings; no new Terraform). Each has a
README with cost and cleanup.

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Measured, simulated, illustrative.** The tiny-transformer curves are measured on your machine
  (the recorded fallback says where it came from). The fake server's model and timing are
  simulated: it knows the eval answers and is right with a probability that grows with its
  thinking, and its latencies come from a roofline step model of Qwen3-0.6B on a T4. The bundled
  JSON, SSE and raw-text files are sample output in the documented format (illustrative). Every
  table says which.
- **The simulated model is not Qwen3.** Its accuracy curve, length distribution and distractors are
  parameters chosen to behave plausibly. Conclusions such as "where thinking pays per token"
  (notebook 04) are about the method; measure your own model before acting on the numbers.
- **Checked by construction.** The T1 paths (vLLM serving, the GRPO step) and the GCP deploys are
  written against vLLM v0.30.0, TRL 1.14.0 and the 04 lab's Terraform, and checked with `bash -n`,
  `DRY_RUN=1`, schema validation and tests. They were not run on a GPU here.

## Verify list (facts dated 2026-09-26 that move)

Checked against source on 2026-09-26: vLLM v0.30.0 (`--reasoning-parser`, parser names `qwen3` and
`deepseek_r1`, the response field `reasoning` (formerly `reasoning_content`), `thinking_token_budget` with
`--reasoning-config`, `include_reasoning`, `reasoning_effort` → `enable_thinking`,
`usage.completion_tokens_details.reasoning_tokens`, no reasoning-specific Prometheus metric,
`--enable-reasoning` removed); TRL 1.14.0 `GRPOConfig` defaults (`beta=0.0`, `loss_type="dapo"`,
`num_iterations=1`, `scale_rewards="group"`, `vllm_importance_sampling_mode="sequence_mask"` with
`clip_max=3.0`, `bf16` on unless `fp16` is set); Qwen3 sampling guidance and template history logic;
DAPO's hyper-parameters. Still to verify on real hardware:

* Qwen3 in fp16 on a T4 (trained in bf16): outputs sane, no NaNs.
* The T1 GRPO step's memory fit on a 15 GB T4 (vLLM at 0.3 + an fp32 0.5B trainer with SGD).
* Whether `thinking_token_budget` counts correctly for templates that open `<think>` in the prompt
  (R1 distills, Qwen3 Thinking-2507).
* Cloud Run's maximum request timeout and L4 regions; Colab and Kaggle GPU quotas; prices in
  [`COMPUTE.md`](../../../COMPUTE.md).
* Model ids: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507`,
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen2.5-0.5B-Instruct`.

The RL-training material also connects to agentic RL, where rollouts are multi-turn tool use in a
sandbox. That is the [sandboxed-execution topic](../../../07-application-agent-framework/sandboxed-execution/README.md).
Reward design and release gates for those agents are the evals of the 07 agent lab's
[notebook 08](../../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py).

MIT licensed.
