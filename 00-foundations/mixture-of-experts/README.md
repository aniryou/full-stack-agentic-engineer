# mixture-of-experts — understand the router and the experts, and predict what sparsity does to serving

After this topic you can explain how an MoE layer routes tokens and why routers must be balanced, count a model's
total and active parameters from its config, predict how many experts a decode batch reads and when it turns
compute-bound, price expert parallelism's all-to-alls, and size an MoE deployment against a dense one — numbers you
can defend in a design review.

## Start here

1. Read [PRIMER.md](PRIMER.md) "The one-minute version", then §1–§2 (30 min): why sparsity, the MoE layer.
2. `cd moe-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 67 tests, ~7 s; then open
   [`01_the_moe_layer`](moe-core/notebooks/01_the_moe_layer.ipynb).
3. With torch installed (CPU is enough), train a tiny MoE in the lab, [`moe-lab/`](moe-lab/README.md): notebook
   [`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb); with any GPU,
   [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb) records a real router's choices.

The fastest win, with nothing but numpy:

```python
from moecore import touched            # run from moe-core/
print(touched.experts_touched(8, 2, 1), round(touched.experts_touched(8, 2, 16), 2))   # 2.0 7.92
# Mixtral reads 2 of its 8 experts per layer for one token, and nearly all 8 for a batch of 16
```

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box
(Kaggle's free 2×T4, or rented for an hour); T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the concepts, §1–9: why sparsity, the MoE layer, routing and load balance, training in brief, which experts a step touches, running MoE on GPUs (fused kernels, EP, wide-EP, offload, quantized experts), sizing and cost, failure modes, where to run it; then a design-review walkthrough and six drills. Every computed number comes from the core | ~2 h, read alongside the core | — |
| [`moe-core/`](moe-core/README.md) | **predict**: package `moecore`, six numpy modules (`moe`, `routing`, `train`, `touched`, `ep`, `sizing`) and five fill-in notebooks; reproduces layer 01's MoE table and layer 02's all-to-all numbers | ~7 h with the primer | T0 |
| [`moe-lab/`](moe-lab/README.md) | **measure**: package `moelab` — a tiny MoE in torch, router hooks on real MoE models, decode step time vs batch in vLLM, `--enable-expert-parallel` on two GPUs, offload and 4-bit experts on a 16–24 GB GPU; `deploy/any-gpu/` and GKE manifests for the 02 lab's `l4x2` pool. Every notebook falls back to a labelled T0 path | ~6 h | T0 → T3 |

### Work it in this order

Each step pairs a primer section with a core notebook (predict) and a lab notebook (measure).

| Step | Read | Predict (core, T0) | Measure (lab) |
|---|---|---|---|
| 1 | [PRIMER §1–2](PRIMER.md#1-why-sparsity) why sparsity, the layer, counting parameters | [`01_the_moe_layer`](moe-core/notebooks/01_the_moe_layer.ipynb) | [`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb) — T0 with torch |
| 2 | [PRIMER §3–4](PRIMER.md#3-routing-and-load-balance) balance, collapse, training | [`02_routing_and_load_balance`](moe-core/notebooks/02_routing_and_load_balance.ipynb) | [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb) — T1 (bundled traces at T0) |
| 3 | [PRIMER §5](PRIMER.md#5-moe-at-inference-which-experts-a-step-touches) experts touched, the crossover, big batches | [`03_which_experts_a_batch_touches`](moe-core/notebooks/03_which_experts_a_batch_touches.ipynb) | [`03_batch_vs_weight_stream`](moe-lab/notebooks/03_batch_vs_weight_stream.ipynb) — T1 (simulated at T0) |
| 4 | [PRIMER §6](PRIMER.md#6-running-moe-on-gpus) fused kernels, EP, wide-EP, offload | [`04_expert_parallelism_and_all_to_all`](moe-core/notebooks/04_expert_parallelism_and_all_to_all.ipynb) | [`04_expert_parallelism_on_two_gpus`](moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb) — T2 on Kaggle 2×T4 (simulated at T0) |
| 5 | [PRIMER §7–9](PRIMER.md#7-sizing-and-cost) sizing, cost, failure modes, where to run | [`05_sizing_and_cost`](moe-core/notebooks/05_sizing_and_cost.ipynb) | [`05_moe_on_a_small_gpu`](moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb) — T1 (sizing at T0) |

Finish with the primer's [design-review walkthrough and drills](PRIMER.md#in-a-design-review).

## Run it

```bash
cd moe-core
python3 -m pip install -r requirements.txt   # numpy + notebook/test tooling
python3 -m pytest -q                          # 67 tests, ~7 s
python3 -m jupyterlab notebooks

cd ../moe-lab                                 # see its README for the GPU paths
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q
```

On Colab, every notebook's first cell clones the repo and installs its package; the per-notebook links are in the
[layer README](../README.md).

| Tier | Where | What you do here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | all five core notebooks; the lab's tiny torch MoE (CPU) and its labelled fallbacks | $0 |
| **T1** | Colab/Kaggle T4 (free), a rented 24 GB GPU (L4, RTX 4090) | real router traces, decode step time vs batch in vLLM, CPU offload and 4-bit experts | free – ~$0.7/hr |
| **T2** | Kaggle 2×T4 (free, PCIe), a rented NVLink box | `--enable-expert-parallel` vs tensor parallel on two GPUs | ~$0–25 per session |
| **T3** | GCP: the cuda-and-nccl lab's `l4x2` pool (2 × L4) | a vLLM Deployment with EP = 2 on GKE, from the lab's manifests (no new Terraform) | pay per use |

Prices and obtainability move monthly: see [`COMPUTE.md`](../../COMPUTE.md); the whole learning path is in
[`CURRICULUM.md`](../../CURRICULUM.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [transformer primer §9](../transformers/docs/transformer-primer.md#9-modern-variants-and-why-each-exists) | the block MoE modifies, and the MoE / MLA rows |
| before | [capacity primer](../gpu-capacity-planning/PRIMER.md) | memory vs bandwidth, TTFT/TPOT, the Mistral Large 3 MoE section |
| before | [roofline-and-fabric §3.6](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#36-moe-which-weights-a-step-streams) | the decode roofline this topic extends |
| alongside | [cuda-and-nccl §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#5-collectives) | the all-to-all collective behind EP |
| after | [serving-engine](../../04-inference-engine/serving-engine/README.md) §9 and §12 | EP inside an engine, which engines serve MoE |
| after | [serving-orchestration §8](../../05-orchestrator/serving-orchestration/PRIMER.md#8-large-moe-topologies-wide-ep-in-brief) | wide-EP deployments in a fleet |
| after | [open-weight primer §4](../model-landscape/open-weight-llms-primer.md#4-the-families-vendor-by-vendor) | the 2026 MoE families |

Quantizing experts connects to the [quantization primer](../../04-inference-engine/quantization/PRIMER.md): its
[§2](../../04-inference-engine/quantization/PRIMER.md#2-number-formats) (formats, MXFP4), [§5](../../04-inference-engine/quantization/PRIMER.md#5-weight-and-activation-quantization) (why routers stay 16-bit)
and [§8](../../04-inference-engine/quantization/PRIMER.md#8-measuring-the-accuracy-you-pay) (calibrating rarely routed experts).

## Caveats

- **Predicted vs measured.** Every step time, all-to-all and cost in the core is a model (roofline or α-β) and
  labelled simulated; the lab measures what real kernels do, on the GPUs you have.
- **The toy trainers are toys.** The core's four linear experts on a clustered regression and the lab's tiny torch
  transformer both collapse without balancing and balance with it, across seeds; the exact numbers do not transfer.
- **Dated facts.** Model configs, vLLM v0.30.0 flags, DeepEP requirements and all prices are a September 2026
  snapshot marked (verify) — see the primer's Verify list.
