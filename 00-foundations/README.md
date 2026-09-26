# 00 · Foundations

Understand the model the whole stack serves: after this layer you can build a transformer from nothing, size a
model's memory, bandwidth and latency before paying for a GPU, place a model family in the open-weight landscape,
say what a mixture-of-experts router does to memory, batching and serving cost, and explain how RL post-training
produces thinking models and what their long outputs do to a serving fleet.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer sits beneath the stack: it is the model every layer above stores, moves and serves.

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box
(Kaggle's free 2×T4, or rented for an hour); T3 = the Google Cloud deployment, optional.* Times are rough, include
the exercises, and match the repo's curriculum ([`CURRICULUM.md`](../CURRICULUM.md), modules 00.1–00.5).

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`transformers/`](transformers/README.md) | build attention, a transformer block and a tiny GPT from nothing; count parameters from a config; say what the KV cache stores and why decoding is sequential — a [primer](transformers/docs/transformer-primer.md), three runnable lessons, practice and walkthrough notebooks | ~5 h | T0 (lesson 3 uses CPU PyTorch) |
| [`gpu-capacity-planning/`](gpu-capacity-planning/README.md) | size weights and KV cache against HBM, estimate TTFT from prefill FLOPs and TPOT from bandwidth, and take a GPU count from the binding constraint plus headroom — a [primer](gpu-capacity-planning/PRIMER.md), `capacity.py` and a practice notebook | ~2 h | T0 |
| [`model-landscape/`](model-landscape/open-weight-llms-primer.md) | say what "open weight" grants, check a licence, and place a model family by size, architecture and deployment tier — the [open-weight primer](model-landscape/open-weight-llms-primer.md) and the [Mistral exercises](model-landscape/mistral-primer-exercises.md) | ~1 h | read |
| [`mixture-of-experts/`](mixture-of-experts/README.md) | explain how an MoE layer routes tokens and why routers must be balanced; count total and active parameters from a config; predict which experts a decode batch reads and when it turns compute-bound; price expert parallelism's all-to-alls; size an MoE deployment against a dense one — a [PRIMER](mixture-of-experts/PRIMER.md), [`moe-core`](mixture-of-experts/moe-core/README.md) (numpy, 5 notebooks) and [`moe-lab`](mixture-of-experts/moe-lab/README.md) (a tiny MoE in torch, router hooks, decode step time vs batch in vLLM, expert parallelism on two GPUs, offload and 4-bit experts; 5 notebooks) | ~7 h primer + core; ~8.5 h lab | T0 → T2 (T3 optional) |
| [`rl-and-thinking-models/`](rl-and-thinking-models/README.md) | explain what an RL post-training step does — REINFORCE, the KL penalty, reward models, DPO, GRPO and its fixes; predict reward hacking, length bias and over-optimisation; choose between thinking longer and sampling more; size and operate a serving fleet for a thinking model — a [PRIMER](rl-and-thinking-models/PRIMER.md), [`rl-core`](rl-and-thinking-models/rl-core/README.md) (numpy, 5 notebooks) and [`thinking-lab`](rl-and-thinking-models/thinking-lab/README.md) (GRPO on a tiny transformer in torch, Qwen3 in vLLM with a reasoning parser, best-of-n and voting, one GRPO step with vLLM rollouts; 5 notebooks) | ~12 h primer + core; ~10 h lab | T0 → T1 (T3 optional) |

## Start here

1. Read the [transformer primer](transformers/docs/transformer-primer.md) §2–8 and run the three
   [lessons](transformers/lessons/) — skip to step 2 if attention and the KV cache are already familiar.
2. `cd gpu-capacity-planning && python3 worked_example.py` — under a second, standard library only: it prints every
   number in the capacity primer, from a 24B dense model on one H100 to Mistral Large 3's 675B MoE.
3. Work the two primer + core + lab topics by their module tables:
   [`mixture-of-experts/`](mixture-of-experts/README.md) once you have layer 01's
   [roofline](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) (§3), and
   [`rl-and-thinking-models/`](rl-and-thinking-models/README.md) once you have read the
   [serving-engine primer](../04-inference-engine/serving-engine/PRIMER.md).

## Run it

```bash
cd transformers && python3 -m pip install -r requirements.txt && python3 lessons/01_attention.py
cd ../gpu-capacity-planning && python3 worked_example.py
cd ../mixture-of-experts/moe-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q     # 67 tests, ~7 s
cd ../moe-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q   # 118 tests; torch tests skip without torch
cd ../../rl-and-thinking-models/rl-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q     # 57 tests, ~30 s
cd ../thinking-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q   # 91 tests, offline; ~50 s with torch
```

Then `python3 -m jupyterlab notebooks` in any core or lab directory, or the Colab links below. The two labs run
every notebook at T0 (torch on a CPU, a fake vLLM, bundled outputs labelled illustrative) and measure on a GPU when
you point them at one.

## How it fits

**Builds on** Python and numpy (CPU PyTorch for the tiny GPT) and nothing else: this is the first stop in the
[curriculum's spiral](../CURRICULUM.md#31-why-this-order) (00 → 04 → 01 → 02 → 04 → 03 → 05 → 06 → 07). Its two
newer topics come later in that spiral, where their prerequisites are: mixture-of-experts after layer 01's roofline,
RL and thinking models after the serving-engine primer (step 3 above).

Everything above builds on this layer's numbers: parameters and KV bytes per token (transformers, capacity
planning), total vs active parameters (mixture-of-experts) and output length (rl-and-thinking-models). Layer 01's
[`roofline-and-fabric`](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) turns them into step times; layer
04's [`serving-engine`](../04-inference-engine/serving-engine/README.md) runs them, and its
[`quantization`](../04-inference-engine/quantization/README.md) topic changes the bytes per parameter. The MoE topic
leans on layer 02's [all-to-all](../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#5-collectives) and feeds layer 05's
[wide-EP](../05-orchestrator/serving-orchestration/PRIMER.md#8-large-moe-topologies-wide-ep-in-brief) fleets; the
thinking-model workload reshapes the engine's KV budget, the router and the gateway's cost per conversation
([`06-gateway`](../06-gateway/README.md)).

## Caveats

- The transformer lessons and the capacity formulas are exact on their own terms; step times, all-to-alls, costs and
  serving numbers in the MoE and RL cores are models (a roofline, an α-β link, the capacity primer's formulas) and
  labelled simulated. The labs measure only on real GPUs, or on the tiny torch models they train on a CPU.
- The toy trainers (a tiny MoE, a table-of-softmaxes policy, a tiny transformer under GRPO) show a mechanism's
  direction across seeds, not a real model's magnitude.
- Model configs, vLLM v0.30.0 flags, TRL defaults and prices are a September 2026 snapshot marked `(verify)`; each
  primer ends with a dated Verify list.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`gpu-capacity-planning/`** — [01_capacity_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice_solved.ipynb)
- **`mixture-of-experts/moe-core/`** — [01_the_moe_layer](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/notebooks/01_the_moe_layer.ipynb) · [02_routing_and_load_balance](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/notebooks/02_routing_and_load_balance.ipynb) · [03_which_experts_a_batch_touches](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/notebooks/03_which_experts_a_batch_touches.ipynb) · [04_expert_parallelism_and_all_to_all](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/notebooks/04_expert_parallelism_and_all_to_all.ipynb) · [05_sizing_and_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/notebooks/05_sizing_and_cost.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/solutions/01_the_moe_layer.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/solutions/02_routing_and_load_balance.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/solutions/03_which_experts_a_batch_touches.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/solutions/04_expert_parallelism_and_all_to_all.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-core/solutions/05_sizing_and_cost.ipynb)
- **`mixture-of-experts/moe-lab/`** — [01_a_tiny_moe_in_torch](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb) · [02_watch_the_router](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/notebooks/02_watch_the_router.ipynb) · [03_batch_vs_weight_stream](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/notebooks/03_batch_vs_weight_stream.ipynb) · [04_expert_parallelism_on_two_gpus](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb) · [05_moe_on_a_small_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/solutions/01_a_tiny_moe_in_torch.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/solutions/02_watch_the_router.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/solutions/03_batch_vs_weight_stream.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/solutions/04_expert_parallelism_on_two_gpus.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/mixture-of-experts/moe-lab/solutions/05_moe_on_a_small_gpu.ipynb)
- **`rl-and-thinking-models/rl-core/`** — [01_policy_gradients_on_a_toy_task](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb) · [02_preferences_reward_models_and_dpo](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/notebooks/02_preferences_reward_models_and_dpo.ipynb) · [03_grpo_with_verifiable_rewards](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/notebooks/03_grpo_with_verifiable_rewards.ipynb) · [04_test_time_compute](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/notebooks/04_test_time_compute.ipynb) · [05_thinking_models_and_the_serving_workload](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/notebooks/05_thinking_models_and_the_serving_workload.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/solutions/01_policy_gradients_on_a_toy_task.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/solutions/02_preferences_reward_models_and_dpo.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/solutions/03_grpo_with_verifiable_rewards.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/solutions/04_test_time_compute.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/rl-core/solutions/05_thinking_models_and_the_serving_workload.ipynb)
- **`rl-and-thinking-models/thinking-lab/`** — [01_grpo_on_a_tiny_transformer](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb) · [02_a_thinking_model_on_one_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb) · [03_test_time_compute_for_real](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/notebooks/03_test_time_compute_for_real.ipynb) · [04_serving_thinking_models](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/notebooks/04_serving_thinking_models.ipynb) · [05_rl_rollouts_with_an_engine](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/notebooks/05_rl_rollouts_with_an_engine.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/solutions/01_grpo_on_a_tiny_transformer.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/solutions/02_a_thinking_model_on_one_gpu.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/solutions/03_test_time_compute_for_real.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/solutions/04_serving_thinking_models.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/rl-and-thinking-models/thinking-lab/solutions/05_rl_rollouts_with_an_engine.ipynb)
- **`transformers/`** — [01_transformer_walkthrough](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/notebooks/01_transformer_walkthrough.ipynb) · [02_transformer_exercises](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/notebooks/02_transformer_exercises.ipynb) · [practice/attention_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/practice/attention_practice.ipynb) — *answers:* [practice/attention_solutions](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/00-foundations/transformers/practice/attention_solutions.ipynb)
<!-- colab-links:end -->
