# mixture-of-experts — understand the router and the experts, and predict what sparsity does to serving

After this topic, you can do these things:

- Explain how an MoE layer routes tokens, and why a router must balance the load across its experts.
- Count the total and active parameters of a model from its config.
- Predict how many experts a decode batch reads, and when the batch becomes compute-bound.
- Give the cost of the all-to-alls of expert parallelism.
- Size an MoE deployment against a dense one.

You can defend these numbers in a design review.

## Start here

1. Read "The one-minute version" in [PRIMER.md](PRIMER.md). Then read §1–§2, why sparsity and the MoE layer (30 min).
2. Run `cd moe-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 75 tests run in ~30 s.
   Then open [`01_the_moe_layer`](moe-core/notebooks/01_the_moe_layer.ipynb).
3. If torch is installed, train a small MoE in the lab, [`moe-lab/`](moe-lab/README.md). A CPU is sufficient for
   this step. Use notebook [`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb). If you have
   a GPU of any type, run [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb). It records the
   choices of a real router.

The fastest result needs only numpy:

```python
from moecore import touched            # run from moe-core/
print(touched.experts_touched(8, 2, 1), round(touched.experts_touched(8, 2, 16), 2))   # 2.0 7.92
# Mixtral reads 2 of its 8 experts per layer for one token, and nearly all 8 for a batch of 16
```

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box (the free 2×T4 of Kaggle, or a box rented for an hour). T3 is the Google Cloud deployment, and it is
optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain the concepts of §1–9. The first four sections are about why sparsity, the MoE layer, routing and load balance, and training in brief. The next two are about which experts a step touches, and how MoE runs on GPUs (fused kernels, EP, wide-EP, offload, quantized experts). The last three are about sizing and cost, failure modes, and where to run it. After them come a design-review walkthrough and six drills. Every computed number comes from the core. | ~2 h. Read it together with the core. | — |
| [`moe-core/`](moe-core/README.md) | **predict**: the package `moecore` has six numpy modules (`moe`, `routing`, `train`, `touched`, `ep`, `sizing`) and five fill-in notebooks. It reproduces the MoE table of layer 01 and the all-to-all numbers of layer 02. | ~7 h with the primer | T0 |
| [`moe-lab/`](moe-lab/README.md) | **measure**: the package is `moelab`. It has a small MoE in torch, router hooks on real MoE models, and a measurement of decode step time against batch in vLLM. It also has `--enable-expert-parallel` on two GPUs, and offload and 4-bit experts on a 16–24 GB GPU. It has `deploy/any-gpu/` and GKE manifests for the `l4x2` pool of the 02 lab. Every notebook falls back to a labelled T0 path. | ~8.5 h | T0 to T3 |

### Work it in this order

Each step has a primer section, a core notebook (predict) and a lab notebook (measure).

| Step | Read | Predict (core, T0) | Measure (lab) |
|---|---|---|---|
| 1 | [PRIMER §1–2](PRIMER.md#1-why-sparsity) why sparsity, the layer, how to count parameters | [`01_the_moe_layer`](moe-core/notebooks/01_the_moe_layer.ipynb) | [`01_a_tiny_moe_in_torch`](moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb), T0 with torch |
| 2 | [PRIMER §3–4](PRIMER.md#3-routing-and-load-balance) balance, collapse, training | [`02_routing_and_load_balance`](moe-core/notebooks/02_routing_and_load_balance.ipynb) | [`02_watch_the_router`](moe-lab/notebooks/02_watch_the_router.ipynb), T1 (bundled traces at T0) |
| 3 | [PRIMER §5](PRIMER.md#5-moe-at-inference-which-experts-a-step-touches) experts touched, the crossover, large batches | [`03_which_experts_a_batch_touches`](moe-core/notebooks/03_which_experts_a_batch_touches.ipynb) | [`03_batch_vs_weight_stream`](moe-lab/notebooks/03_batch_vs_weight_stream.ipynb), T1 (simulated at T0) |
| 4 | [PRIMER §6](PRIMER.md#6-running-moe-on-gpus) fused kernels, EP, wide-EP, offload | [`04_expert_parallelism_and_all_to_all`](moe-core/notebooks/04_expert_parallelism_and_all_to_all.ipynb) | [`04_expert_parallelism_on_two_gpus`](moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb), T2 on Kaggle 2×T4 (simulated at T0) |
| 5 | [PRIMER §7–9](PRIMER.md#7-sizing-and-cost) sizing, cost, failure modes, where to run | [`05_sizing_and_cost`](moe-core/notebooks/05_sizing_and_cost.ipynb) | [`05_moe_on_a_small_gpu`](moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb), T1 (sizing at T0) |

Finish with the [design-review walkthrough and drills](PRIMER.md#in-a-design-review) of the primer.

## Run it

```bash
cd moe-core
python3 -m pip install -r requirements.txt   # numpy + notebook/test tooling
python3 -m pytest -q                          # 75 tests, ~30 s
python3 -m jupyterlab notebooks

cd ../moe-lab                                 # see its README for the GPU paths
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q
```

On Colab, the first cell of every notebook clones the repo and installs its package. The per-notebook links are in
the [layer README](../README.md).

| Tier | Where | What you do here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | all five core notebooks, the small torch MoE of the lab (CPU) and its labelled fallbacks | $0 |
| **T1** | Colab/Kaggle T4 (free), a rented 24 GB GPU (L4, RTX 4090) | real router traces, decode step time against batch in vLLM, CPU offload and 4-bit experts | free to ~$0.7/hr |
| **T2** | Kaggle 2×T4 (free, PCIe), a rented NVLink box | `--enable-expert-parallel` against tensor parallel on two GPUs | ~$0–25 per session |
| **T3** | GCP: the cuda-and-nccl lab's `l4x2` pool (2 × L4) | a vLLM Deployment with EP = 2 on GKE, from the lab's manifests (no new Terraform) | pay per use |

Prices and obtainability change monthly. See [`COMPUTE.md`](../../COMPUTE.md). The whole learning path is in
[`CURRICULUM.md`](../../CURRICULUM.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [transformer primer §9](../transformers/docs/transformer-primer.md#9-modern-variants-and-why-each-exists) | the block that MoE changes, and the MoE / MLA rows |
| before | [capacity primer](../gpu-capacity-planning/PRIMER.md) | memory against bandwidth, TTFT/TPOT, the Mistral Large 3 MoE section |
| before | [roofline-and-fabric §3.6](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#36-moe-which-weights-a-step-streams) | the decode roofline that this topic extends |
| alongside | [cuda-and-nccl §5](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md#5-collectives) | the all-to-all collective behind EP |
| after | [serving-engine](../../04-inference-engine/serving-engine/README.md) §9 and §12 | EP inside an engine, which engines serve MoE |
| after | [serving-orchestration §8](../../05-orchestrator/serving-orchestration/PRIMER.md#8-large-moe-topologies-wide-ep-in-brief) | wide-EP deployments in a fleet |
| after | [open-weight primer §4](../model-landscape/open-weight-llms-primer.md#4-the-families-vendor-by-vendor) | the 2026 MoE families |

The quantization of experts connects to the [quantization primer](../../04-inference-engine/quantization/PRIMER.md).
Its [§2](../../04-inference-engine/quantization/PRIMER.md#2-number-formats) covers the formats and MXFP4. Its
[§5](../../04-inference-engine/quantization/PRIMER.md#5-weight-and-activation-quantization) tells why routers stay
16-bit. Its [§8](../../04-inference-engine/quantization/PRIMER.md#8-measuring-the-accuracy-you-pay) covers the
calibration of rarely routed experts.

## Caveats

- **Predicted against measured.** Every step time, all-to-all and cost in the core is a model (roofline or α-β),
  and each has the label "simulated". The lab measures what real kernels do, on the GPUs that you have.
- **The toy trainers are toys.** The core has four linear experts on a clustered regression. The lab has a small
  torch transformer. Across seeds, both collapse without load balance, and both become balanced with it.
  The exact numbers do not transfer.
- **Dated facts.** Model configs, the flags of vLLM v0.30.0, DeepEP requirements and all prices are a September 2026
  snapshot. The snapshot has the mark (verify). See the Verify list of the primer.
