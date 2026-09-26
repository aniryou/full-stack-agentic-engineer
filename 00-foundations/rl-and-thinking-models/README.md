# rl-and-thinking-models — how post-training teaches a model to reason, and what thinking does to serving

After this topic you can explain what an RL post-training step does to a model — REINFORCE, the KL penalty, reward
models, DPO, GRPO and its fixes — predict how it goes wrong (reward hacking, length bias, over-optimisation),
choose between thinking longer and sampling more, and size and operate a serving fleet for a thinking model.

## Start here

1. Read [PRIMER.md](PRIMER.md): "The one-minute version", then §1 From pretraining to post-training and §2 Policy
   gradients over token sequences (40 min).
2. `cd rl-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 65 tests in about 50 s; then
   open [`01_policy_gradients_on_a_toy_task`](rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb) and watch
   RL exploit a buggy verifier.
3. With any GPU (a free Colab T4 is enough), serve a real thinking model and switch its thinking on and off:
   [`thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb`](thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain post-training and thinking models in nine sections — §1 from pretraining to post-training · §2 policy gradients over token sequences · §3 learning from preferences · §4 RL with verifiable rewards and GRPO · §5 thinking models · §6 test-time compute · §7 what thinking does to serving · §8 the RL training stack in brief · §9 where to run it — each formula with a worked number and the core function that computes it; then "In a design review", a glossary, sources and a dated Verify list | ~2.5 h, read alongside the core | — |
| [`rl-core/`](rl-core/) | build it yourself in `rlcore` (standard library + numpy, ~1,000 lines): verifiable toy tasks, a table-of-softmaxes policy, REINFORCE and the KL closed form, Bradley–Terry and DPO, GRPO with TRL's options and DAPO's fixes, pass@k and budget allocation, and the serving workload model built on the capacity primer; five fill-in notebooks | ~10 h | T0 |
| [`thinking-lab/`](thinking-lab/) | train a tiny transformer with SFT then GRPO in torch; serve Qwen3 in vLLM with `--reasoning-parser`, thinking on and off, with budgets; run best-of-n and majority vote on a real model; measure ITL, KV usage and preemptions under long outputs; drive one GRPO step with vLLM generating the rollouts (`thinklab`; a fake server and bundled outputs make every notebook run at T0) | ~10 h | T0 → T1 (T3 via the 04 lab's deploys) |

### Work it in this order

Read the primer sections, do the core notebook (T0), then the lab notebook — at T0 first, then on a GPU if you
have one.

| Step | Primer | Core notebook (T0) | Lab notebook | Tier |
|---|---|---|---|---|
| Policy gradients | §1 From pretraining to post-training · §2 Policy gradients over token sequences | [`01_policy_gradients_on_a_toy_task`](rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb) | [`01_grpo_on_a_tiny_transformer`](thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb) | T0 (T1 faster) |
| Preferences and DPO | §3 Learning from preferences | [`02_preferences_reward_models_and_dpo`](rl-core/notebooks/02_preferences_reward_models_and_dpo.ipynb) | — | T0 |
| GRPO and DAPO | §4 RL with verifiable rewards and GRPO · §8 The RL training stack in brief | [`03_grpo_with_verifiable_rewards`](rl-core/notebooks/03_grpo_with_verifiable_rewards.ipynb) | [`01_grpo_on_a_tiny_transformer`](thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb), [`05_rl_rollouts_with_an_engine`](thinking-lab/notebooks/05_rl_rollouts_with_an_engine.ipynb) | T0 → T1 |
| Test-time compute | §6 Test-time compute | [`04_test_time_compute`](rl-core/notebooks/04_test_time_compute.ipynb) | [`03_test_time_compute_for_real`](thinking-lab/notebooks/03_test_time_compute_for_real.ipynb) | T0 → T1 |
| Thinking models and serving | §5 Thinking models · §7 What thinking does to serving · §9 Where to run it | [`05_thinking_models_and_the_serving_workload`](rl-core/notebooks/05_thinking_models_and_the_serving_workload.ipynb) | [`02_a_thinking_model_on_one_gpu`](thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb), [`04_serving_thinking_models`](thinking-lab/notebooks/04_serving_thinking_models.ipynb) | T0 → T1 |

Each notebook ends with "In a design review" — the two-minute explanation and its drills. The primer's own
design-review section covers the whole topic.

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

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | What you run in this topic | Hardware and cost |
|---|---|---|
| **T0** | every core notebook; the lab's tiny-transformer GRPO on CPU (torch), its fake OpenAI-compatible server emitting reasoning with heavy-tailed lengths (labelled simulated), and its bundled model outputs (labelled illustrative) | laptop, Colab CPU or CI — $0 |
| **T1** | Qwen3-0.6B or DeepSeek-R1-Distill-Qwen-1.5B in vLLM on a T4 with `--dtype half`; Qwen3-4B on a 24 GB card; best-of-n and voting on real outputs; one GRPO step with vLLM generating the rollouts | Colab/Kaggle T4 (free; fp16 only), any 24 GB GPU (~$0.3–0.7/hr, verify) |
| **T3** | a thinking model behind the 04 serving lab's Cloud Run GPU or GKE deploy (no new Terraform here) | GCP, pay per use; see that lab's `deploy/` READMEs for cleanup |

Prices, free tiers and how to obtain GPUs on GCP and elsewhere: [`COMPUTE.md`](../../COMPUTE.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`transformers`](../transformers/) (primer §6 training, §7 inference); [`gpu-capacity-planning`](../gpu-capacity-planning/PRIMER.md) | the training objective and what post-training is; weights, KV bytes, TTFT and TPOT, which §7 reuses and reproduces |
| beside | [`04 serving-engine`](../../04-inference-engine/serving-engine/README.md) primer §5, §6, §7, §11; [`vllm-internals`](../../04-inference-engine/vllm-internals/README.md) §9 | prefix caching, sampling, speculative decoding and measurement — the engine a rollout generator and a thinking model both run on; LoRA adapters in flight |
| after | [`06 agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | cost per conversation, output pricing and routing by effort at the gateway |
| after | [`07-application-agent-framework`](../../07-application-agent-framework/) — the platform lab's evals notebook, and [`sandboxed-execution`](../../07-application-agent-framework/sandboxed-execution/README.md) | evals with intervals for the true objective; running model-generated code and tool calls for agentic RL rollouts |

## Caveats

- **Toys and models, labelled.** The core's policies are tables of softmaxes; every effect's direction is robust
  across seeds and pinned by tests, but magnitudes are the toy's. Serving numbers come from the capacity primer's
  formulas plus a roofline step, not from hardware.
- **Measured only in the lab.** Real reasoning outputs, ITL under long outputs and GRPO on a transformer are the lab's
  T0 (torch on CPU) and T1 runs; its fake server and bundled outputs are labelled simulated or illustrative.
- **Dated facts.** vLLM 0.30.0's reasoning flags and field names (`reasoning`, not `reasoning_content`), TRL 1.14.0's
  `GRPOConfig` defaults, the Qwen3 and DeepSeek-R1 facts and all prices are as of September 2026 and marked
  `(verify)`; the primer's [Verify list](PRIMER.md#verify-list) collects them.
