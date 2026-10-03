# thinking-lab — watch RL teach a model to think, then serve a thinking model and price what thinking costs

After this lab, you can do these things, first on a laptop and then on a free T4:

- Train a small transformer with GRPO.
- Serve a real thinking model with the correct switches, parsers and budgets.
- Measure what more samples and more thinking give you.
- Tell what long outputs do to ITL, KV capacity and cost.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m thinklab tinyrl`. With torch, the command runs in about
   a minute. The CPU build of torch is sufficient. At the start, a 100K-parameter transformer thinks half of the
   time. At the end, it always thinks. Its reward and its completion length increase together. Without torch, the
   command prints a recorded run.
2. Open [`notebooks/01_grpo_on_a_tiny_transformer.ipynb`](notebooks/01_grpo_on_a_tiny_transformer.ipynb) on your
   laptop. The notebook explains the same run. It shows the verifier and the GRPO loss that you write in torch and
   then use for training. It also shows the curves that your machine produced. A check tells you if this run
   learned.
3. On any GPU, start a real thinking model with [`deploy/any-gpu/serve.sh`](deploy/any-gpu/serve.sh). A free Colab
   T4 is sufficient. Then run `export THINKLAB_URL=http://127.0.0.1:8000`. Notebooks 02–04 then measure the real
   model instead of the simulator.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, and it is optional.*

Each notebook starts with *the one-minute version*. Then it shows worked examples that use the library. Then it gives
3–6 exercises: you write the key function, predict a number or select a setting. A check that prints ✅ comes after
each exercise. The notebook ends with *in a design review*.

The answers are in [`solutions/`](solutions/). The concepts are in the topic's [`PRIMER.md`](../PRIMER.md).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`grpo_on_a_tiny_transformer`](notebooks/01_grpo_on_a_tiny_transformer.ipynb) | T0 with torch (T1 faster) | Why a scratchpad adds serial computation. An outcome verifier. The reward-and-length curves of a real GRPO run on a small model, and if the run learned. The GRPO loss in torch (clip, β·k3, the `grpo` / `dr_grpo` / `dapo` aggregations), and training with it. Why `clip_frac` is 0 at `num_iterations=1`, and why the clip has an effect at 2. Why the logged k3 KL has spikes. | [§2 Policy gradients over token sequences](../PRIMER.md#2-policy-gradients-over-token-sequences), [§4 RL with verifiable rewards and GRPO](../PRIMER.md#4-rl-with-verifiable-rewards-and-grpo), [§5 Thinking models](../PRIMER.md#5-thinking-models) | ~2 h |
| 02 | [`a_thinking_model_on_one_gpu`](notebooks/02_a_thinking_model_on_one_gpu.ipynb) | T1 (T0: bundled samples + fake server) | How Qwen3 fits on a T4 (an 8K trace is 0.94 GB of KV). `reasoning` against `reasoning_content`. How to parse R1-style output. When the answer starts in a stream. The three switches (`enable_thinking`, `thinking_token_budget`, `reasoning_effort`). The `max_tokens` trap. How to select a budget from an accuracy curve. | [§5 Thinking models](../PRIMER.md#5-thinking-models), [§7 What thinking does to serving](../PRIMER.md#7-what-thinking-does-to-serving) | ~2 h |
| 03 | [`test_time_compute_for_real`](notebooks/03_test_time_compute_for_real.ipynb) | T1 (T0: bundled simulated records) | pass@k from recorded samples, and how much the plug-in shortcut makes it too low. pass^k against the independence shortcut. Majority vote, and when it makes the result worse. Best-of-n with a verifier against a noisy reward model. The cost per correct answer. The most accurate option for a token budget. | [§6 Test-time compute](../PRIMER.md#6-test-time-compute) | ~1.5 h |
| 04 | [`serving_thinking_models`](notebooks/04_serving_thinking_models.ipynb) | T0 fake server + emulator / T1 real vLLM | Heavy-tailed output lengths. KV × time from measured lengths, and how much of it the tail holds. The capacity primer's numbers with 10× outputs (≈18× the GPUs for KV). The batch that the KV pool permits, and the ITL that it gives. Budgets against `max_model_len`. Why the prefix cache stops at the last assistant header. Routing by effort. | [§7 What thinking does to serving](../PRIMER.md#7-what-thinking-does-to-serving) | ~2.5 h |
| 05 | [`rl_rollouts_with_an_engine`](notebooks/05_rl_rollouts_with_an_engine.ipynb) | T1 (T0: the small model's rollouts) | The rollout as an inference workload. Advantages and dynamic sampling. The train–inference log-prob mismatch, and TRL's `sequence_mask` correction. How to select a generation cap under DAPO's overlong penalty. Straggler idle time in a synchronous rollout batch. Weight-sync bytes. | [§4 RL with verifiable rewards and GRPO](../PRIMER.md#4-rl-with-verifiable-rewards-and-grpo), [§8 The RL training stack in brief](../PRIMER.md#8-the-rl-training-stack-in-brief) | ~2 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the GRPO on the small transformer (torch CPU), a fake vLLM with a simulated thinking model, the engine emulator, bundled samples | every notebook and test |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3` (1.7B, 4B and R1-Distill-1.5B also fit), and one GRPO step with vLLM rollouts | `THINKLAB_URL=...`, [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | a multi-GPU box | This lab does not need it. TRL's server mode puts vLLM on its own GPUs. Layer 02 covers the collectives that sync weights. | — |
| **T3** | GCP | the 04 serving lab's Cloud Run or GKE deploy with Qwen3-4B and a reasoning parser | [`deploy/gcp/`](deploy/gcp/) |

## Run it

```bash
cd thinking-lab
python3 -m pip install -e ".[dev]"             # aiohttp; dev: pytest, jupyter, pyyaml, matplotlib
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: notebooks 01/05 train for real
python3 -m pytest -q                           # 99 tests, offline, no GPU: ~80 s with torch (one full GRPO run; -m 'not slow' skips it), ~40 s without
python3 -m thinklab tinyrl                     # SFT + GRPO on the tiny transformer (~1 min on a CPU)
python3 -m thinklab fake --port 8000 &         # a fake vLLM serving a simulated Qwen3-0.6B on a T4
THINKLAB_URL=http://127.0.0.1:8000 python3 -m thinklab ask "What is 47 * 23 - 318?"
python3 -m thinklab eval -n 20                 # thinking off / on / budgeted on the generated eval set
python3 -m thinklab shape                      # the serving shape of thinking vs not (virtual time; simulated)
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

To measure a real engine instead (T1/T3), start one. [`deploy/any-gpu/`](deploy/any-gpu/) has the Colab/Kaggle T4
recipe. Then run `export THINKLAB_URL=http://127.0.0.1:8000`. Also set `THINKLAB_API_KEY`. For a private Cloud Run
service, set `THINKLAB_BEARER=$(gcloud auth print-identity-token)` instead. The notebooks find the engine and measure
it.

The lab identifies a `thinklab fake` server in `THINKLAB_URL` by its `/version`, and the server stays labelled
simulated. `THINKLAB_NO_TORCH=1` shows the recorded-run path, also when torch is installed.

## The library (`thinklab/`, ~3,600 lines)

| Module | Lines | The idea |
|---|---:|---|
| `tinyrl/` | ~540 | The scratchpad task and its verifier (pure Python). A small decoder-only transformer with the same structure as `00-foundations/transformers/lessons/03_tiny_gpt.py`. An SFT warm-up, then GRPO with TRL's names and defaults, a pluggable loss and `grpo_from` for experiments from one SFT model. Rollouts from a bf16 "engine" copy, which an fp32 "trainer" copy scores again. A recorded run for machines without torch. |
| `thinking/` | ~680 | An OpenAI-compatible client that reads `reasoning` *and* `reasoning_content`, and measures the time when the answer starts. Request bodies for the thinking switch, budgets and effort. Budget strategies (truncate, native, Qwen's two-call recipe). pass@k, pass^k, majority vote, best-of-n, cost per correct answer. A generated eval set of verifiable problems. Recorded simulated outcomes. |
| `parsers.py` | ~180 | vLLM's `deepseek_r1` and `qwen3` semantics, gpt-oss Harmony channels, a streaming splitter that holds back split tags, and answer extraction. |
| `templates.py` | ~110 | How Qwen3 renders the history (it drops the reasoning before the last user message), and the cached-prefix arithmetic that comes from it. |
| `fakemodel.py` | ~160 | The simulated thinking model: log-normal thinking lengths by difficulty, accuracy that increases with thinking, and distractor answers. |
| `engine.py` | ~270 | The emulator of time: FCFS continuous batching, KV blocks, recompute preemption and a roofline step time. Qwen3 profiles whose numbers reproduce `servelab.sizing`. |
| `fakeserver.py` | ~460 | The fake vLLM over HTTP: chat completions (streaming, `n`, `continue_final_message`), the reasoning switches and budgets, `usage.completion_tokens_details.reasoning_tokens`, cached tokens, and `/metrics` with vLLM's names. |
| `workload.py` | ~370 | Length statistics, the steady-state serving shape, the capacity primer's arithmetic, open-loop load with gauge polls, and modes that it compares in virtual time. |
| `rollout.py` | ~250 | The records of the RL step (advantages, dynamic sampling, IS corrections, DAPO shaping, aggregations, straggler time, weight sync), and the T1 vLLM + transformers step. |
| `metrics.py`, `report.py`, `env.py`, `__main__.py` | ~570 | A Prometheus writer and parser, and PromQL quantiles. Labelled tables and reports. Tier detection. The CLI. |

The lab has three siblings:

- The topic's minimal core, [`../rl-core/`](../rl-core/). This lab never imports it.
- The 04 serving lab ([`vllm-serving-lab`](../../../04-inference-engine/serving-engine/vllm-serving-lab/)). The fake
  server, the load generator and the metric names of this lab are copies of the ones in the 04 lab.
- The capacity primer ([`00-foundations/gpu-capacity-planning`](../../gpu-capacity-planning/PRIMER.md)).
  `workload.capacity_primer_view` reproduces its arithmetic. `tests/test_reuse.py` compares it with `capacity.py`,
  `servelab.sizing` and the 07 agent lab's Wilson interval.

## Deploy

[`deploy/`](deploy/) has two targets:

- [`any-gpu/`](deploy/any-gpu/). `serve.sh` runs `vllm serve` through docker or pip. It sets the correct reasoning
  parser, the budget config and `--dtype half` on a T4. `rl_step.sh` does one GRPO step with vLLM rollouts. The
  folder also has Colab/Kaggle and 24 GB recipes, and RunPod/Vast notes.
- [`gcp/`](deploy/gcp/). It uses the 04 lab's Cloud Run Terraform and GKE manifests with the settings of a thinking
  model. It adds no new Terraform.

Each target has a README with the cost and the cleanup.

## Regenerating notebooks

The builder generates `notebooks/` (the exercises) and `solutions/` from `notebooks_src/*.py`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests + bash -n
```

## Caveats

- **Measured, simulated, illustrative.** The curves of the small transformer come from a measurement on your machine
  (the recorded fallback tells where it came from). The model and the latencies of the fake server come from a simulation. The
  server knows the eval answers, and it gives the correct answer with a probability that increases with its thinking.
  Its latencies come from a roofline step model of Qwen3-0.6B on a T4. The bundled JSON, SSE and raw-text files are
  sample output in the documented format (illustrative). Each table tells which kind it shows.
- **The simulated model is not Qwen3.** Its accuracy curve, length distribution and distractors are parameters. Their
  values make the model behave in a plausible way. Conclusions such as "where thinking pays per token" (notebook 04)
  are about the method. Measure your own model before you act on the numbers.
- **Checked by construction.** The T1 paths (vLLM serving, the GRPO step) and the GCP deploys use the interfaces of
  vLLM v0.30.0, TRL 1.14.0 and the 04 lab's Terraform. The checks on them are `bash -n`, `DRY_RUN=1`, schema
  validation and tests. They did not run on a GPU here.

## Verify list (facts dated 2026-09-26 that move)

On 2026-09-26, a comparison with the source confirmed these facts:

* vLLM v0.30.0: `--reasoning-parser` and the parser names `qwen3` and `deepseek_r1`. The response field `reasoning`
  (its old name is `reasoning_content`). `thinking_token_budget` with `--reasoning-config`, and `include_reasoning`.
  vLLM maps `reasoning_effort` to `enable_thinking`. `usage.completion_tokens_details.reasoning_tokens`. No
  reasoning-specific Prometheus metric. `--enable-reasoning` is no longer in vLLM.
* TRL 1.14.0 `GRPOConfig` defaults: `beta=0.0`, `loss_type="dapo"`, `num_iterations=1`, `scale_rewards="group"`,
  `vllm_importance_sampling_mode="sequence_mask"` with `clip_max=3.0`, and `bf16` on if you do not set `fp16`.
* Qwen3 sampling guidance and the history logic of its template.
* DAPO's hyper-parameters.

These items still need a check on real hardware:

* Qwen3 in fp16 on a T4 (its training used bf16): the outputs are sane, with no NaNs.
* The memory fit of the T1 GRPO step on a 15 GB T4 (vLLM at 0.3 + an fp32 0.5B trainer with SGD).
* If `thinking_token_budget` counts correctly for templates that open `<think>` in the prompt (R1 distills, Qwen3
  Thinking-2507).
* Cloud Run's maximum request timeout and its L4 regions. The Colab and Kaggle GPU quotas. The prices in
  [`COMPUTE.md`](../../../COMPUTE.md).
* Model ids: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507`,
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen2.5-0.5B-Instruct`.

The RL-training material also connects to agentic RL. In agentic RL, the rollouts are multi-turn tool use in a
sandbox. The [sandboxed-execution topic](../../../07-application-agent-framework/sandboxed-execution/README.md) covers
that subject. The reward design and the release gates for those agents are the evals of the 07 agent lab's
[notebook 08](../../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks_src/08_evals_trajectory_judge_gates.py).

MIT licensed.
