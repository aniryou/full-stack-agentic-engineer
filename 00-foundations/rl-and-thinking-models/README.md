# rl-and-thinking-models — how post-training teaches a model to reason, and what thinking does to serving

After this topic, you can do these things:

- Explain what an RL post-training step does to a model: REINFORCE, the KL penalty, reward models, DPO, and GRPO
  and its corrections.
- Predict how the step fails: reward hacking, length bias, over-optimisation.
- Select between longer thinking and more samples.
- Calculate the size of a serving fleet for a thinking model, and operate the fleet.

## Start here

1. Read "The one-minute version" in [PRIMER.md](PRIMER.md). Then read §1 From pretraining to post-training and §2
   Policy gradients over token sequences (40 min).
2. Run `cd rl-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 65 tests run in
   approximately 50 s. Then open
   [`01_policy_gradients_on_a_toy_task`](rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb). See how RL
   exploits a verifier that has a bug.
3. Use any GPU. A free Colab T4 is sufficient. Serve a real thinking model. Then turn its thinking on and off:
   [`thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb`](thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb).

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box, rented for an hour. T3 is the Google Cloud deployment, and it is optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain post-training and thinking models in nine sections. The first three are §1 from pretraining to post-training, §2 policy gradients over token sequences and §3 learning from preferences. Then come §4 RL with verifiable rewards and GRPO, §5 thinking models and §6 test-time compute. The last three are §7 what thinking does to serving, §8 the RL training stack in brief and §9 where to run it. Each formula has a worked number and the core function that calculates it. After the sections come "In a design review", a glossary, sources and a dated Verify list. | ~2.5 h, read together with the core | — |
| [`rl-core/`](rl-core/) | Build it yourself in `rlcore` (standard library + numpy, ~1,000 lines). It has verifiable toy tasks, a table-of-softmaxes policy, REINFORCE and the KL closed form, and Bradley–Terry and DPO. It also has GRPO with TRL's options and DAPO's corrections, pass@k and budget allocation, and the serving workload model built on the capacity primer. There are five fill-in notebooks. | ~10 h | T0 |
| [`thinking-lab/`](thinking-lab/) | Train a small transformer with SFT, then with GRPO, in torch. Serve Qwen3 in vLLM with `--reasoning-parser`, with thinking on and off, and with budgets. Run best-of-n and majority vote on a real model. Measure ITL, KV usage and preemptions when outputs are long. Run one GRPO step in which vLLM generates the rollouts. The package is `thinklab`, and its fake server and bundled outputs let every notebook run at T0. | ~10 h | T0 to T1 (T3 through the deploys of the 04 lab) |

### Work it in this order

Read the primer sections. Then do the core notebook (T0). Then do the lab notebook, first at T0, and then on a GPU
if you have one.

| Step | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| Policy gradients | §1 From pretraining to post-training · §2 Policy gradients over token sequences | [`01_policy_gradients_on_a_toy_task`](rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb) | [`01_grpo_on_a_tiny_transformer`](thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb) | T0 (T1 faster) |
| Preferences and DPO | §3 Learning from preferences | [`02_preferences_reward_models_and_dpo`](rl-core/notebooks/02_preferences_reward_models_and_dpo.ipynb) | — | T0 |
| GRPO and DAPO | §4 RL with verifiable rewards and GRPO · §8 The RL training stack in brief | [`03_grpo_with_verifiable_rewards`](rl-core/notebooks/03_grpo_with_verifiable_rewards.ipynb) | [`01_grpo_on_a_tiny_transformer`](thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb), [`05_rl_rollouts_with_an_engine`](thinking-lab/notebooks/05_rl_rollouts_with_an_engine.ipynb) | T0 to T1 |
| Test-time compute | §6 Test-time compute | [`04_test_time_compute`](rl-core/notebooks/04_test_time_compute.ipynb) | [`03_test_time_compute_for_real`](thinking-lab/notebooks/03_test_time_compute_for_real.ipynb) | T0 to T1 |
| Thinking models and serving | §5 Thinking models · §7 What thinking does to serving · §9 Where to run it | [`05_thinking_models_and_the_serving_workload`](rl-core/notebooks/05_thinking_models_and_the_serving_workload.ipynb) | [`02_a_thinking_model_on_one_gpu`](thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb), [`04_serving_thinking_models`](thinking-lab/notebooks/04_serving_thinking_models.ipynb) | T0 to T1 |

Each notebook ends with "In a design review". This section gives the two-minute explanation and its drills. The
design-review section of the primer covers the whole topic.

## Run it

```bash
cd rl-core
python3 -m pip install -r requirements.txt     # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 65 tests, ~50 s
python3 -m jupyterlab notebooks                # the exercises; finished versions are in solutions/

cd ../thinking-lab
python3 -m pip install -e ".[dev]"
python3 -m pytest -q                           # offline; ~50 s with torch (a full GRPO run), ~10 s without
python3 -m jupyterlab notebooks
```

On Colab, the first cell of each notebook clones the repo and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | All core notebooks. From the lab: the GRPO on a small transformer on CPU (torch), the fake OpenAI-compatible server (labelled simulated) and the bundled model outputs (labelled illustrative). The fake server sends reasoning with heavy-tailed lengths. | laptop, Colab CPU or CI, $0 |
| **T1** | Qwen3-0.6B or DeepSeek-R1-Distill-Qwen-1.5B in vLLM on a T4 with `--dtype half`. Qwen3-4B on a 24 GB card. Best-of-n and votes on real outputs. One GRPO step in which vLLM generates the rollouts. | Colab/Kaggle T4 (free, fp16 only), any 24 GB GPU (~$0.3–0.7/hr, verify) |
| **T3** | A thinking model behind the Cloud Run GPU deploy or the GKE deploy of the 04 serving lab. This topic adds no new Terraform. | GCP, pay per use. For cleanup, see the `deploy/` READMEs of that lab. |

For prices, free tiers and how to get GPUs on GCP and on other platforms, see [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`transformers`](../transformers/) (primer §6 training, §7 inference) and [`gpu-capacity-planning`](../gpu-capacity-planning/PRIMER.md) | The training objective and what post-training is. Weights, KV bytes, TTFT and TPOT, which §7 uses again and reproduces. |
| beside | [`04 serving-engine`](../../04-inference-engine/serving-engine/README.md) primer §5, §6, §7, §11, and [`vllm-internals`](../../04-inference-engine/vllm-internals/README.md) §9 | Prefix caching, sampling, speculative decoding and measurement. These sections describe the engine that a rollout generator and a thinking model both run on. They also cover LoRA adapters in flight. |
| after | [`distillation`](../distillation/README.md) (primer §3, §4, §5) | How to copy a teacher into a small student. This covers SFT on the traces of the teacher and on-policy distillation as RL with a dense per-token reward. It also covers what a distilled thinking model gets from the teacher. |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | Cost per conversation, output pricing and routing by effort at the gateway. |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/): the evals notebook of the platform lab, and [`sandboxed-execution`](../../07-application-agent-framework/sandboxed-execution/README.md) | Evals with intervals for the true objective. How to run model-generated code and tool calls for agentic RL rollouts. |

## Caveats

- **Toys and models, labelled.** The policies of the core are tables of softmaxes. The direction of each effect
  does not change across seeds, and tests pin it. But the magnitudes are those of the toy. The serving numbers come
  from the formulas of the capacity primer plus a roofline step, not from hardware.
- **Measured only in the lab.** Real reasoning outputs, ITL under long outputs and GRPO on a transformer come from
  the lab. The lab gets them in its T0 runs (torch on CPU) and its T1 runs. The lab labels its fake server and its
  bundled outputs simulated or illustrative.
- **Dated facts.** The reasoning flags and field names of vLLM 0.30.0 (`reasoning`, not `reasoning_content`) are
  as of September 2026. The `GRPOConfig` defaults of TRL 1.14.0, the Qwen3 and DeepSeek-R1 facts and all prices are
  also as of September 2026. All of these facts have the mark `(verify)`. The [Verify list](PRIMER.md#verify-list) of
  the primer collects them.
