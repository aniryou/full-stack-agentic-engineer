# roofline-and-fabric — reading the machine

*Rooflines, the memory hierarchy, fabrics, and the cost of a token.* This topic turns datasheets and
network diagrams into numbers you can defend in a design review: which resource bounds an LLM step, what
a collective costs on each link, how long a replica takes to load, how often a big job fails, and what a
token costs. Every concept runs at tier **T0** on a laptop; real hardware is the optional second half.

## What's here

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](PRIMER.md) | The concepts, sections 1–10: spec sheets, roofline, LLM inference on the roofline, memory hierarchy, fabrics, storage and cold start, reliability, cost, the September 2026 accelerator landscape, getting hardware. Every number is computed by the core and pinned by its tests. | read |
| [`roofline-core/`](roofline-core/README.md) | The **minimal** implementation: package `roofline`, seven standard-library modules (`specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability`, `cost`) and four fill-in notebooks. | T0 |
| [`gpu-bench-lab/`](gpu-bench-lab/) | The **detailed** lab: package `gpubench`, "measure the machine you have" — numpy (CPU) and torch (CUDA) backends for GEMM throughput, memory bandwidth, host↔device and GPU↔GPU transfers, weight loading; `nvidia-smi` topology and inventory parsers; Docker and GCP Terraform deploys. | T0 → T3 |

## How to work this topic

Each step pairs a primer section with a core notebook (predict) and a lab notebook (measure):

| Step | Read | Predict (core, T0) | Measure (lab) |
|---|---|---|---|
| 1 | [PRIMER §1–2](PRIMER.md#1-spec-sheet-literacy) spec sheets, the roofline | `01_spec_sheets_and_the_roofline` | `01_measure_your_roofline` — T0 on your CPU, T1 on a GPU |
| 2 | [PRIMER §3–4](PRIMER.md#3-llm-inference-on-the-roofline) LLM steps, KV reads, quantization, MoE; tiling and fusion | `02_llm_inference_on_the_roofline` | `02_memory_bandwidth_and_transfers` — T0 / T1 |
| 3 | [PRIMER §5](PRIMER.md#5-fabrics-quantitatively) α-β, collectives, TP cost, rails, topology | `03_fabrics_and_collective_cost` | `03_multi_gpu_topology_and_p2p` — T2 (T0 falls back to sample topology output) |
| 4 | [PRIMER §6–8](PRIMER.md#6-storage-and-cold-start) cold start, reliability, $/M tokens | `04_loading_reliability_and_cost` | `04_weights_loading_and_cold_start` — T0 / T1 |
| 5 | [PRIMER §9–10](PRIMER.md#9-the-accelerator-landscape-september-2026-snapshot) the landscape, getting hardware | — | lab `deploy/any-gpu/` (T1/T2) or `deploy/gcp/terraform/` (T3) |

Finish with the primer's [design-review walkthrough and drills](PRIMER.md#in-a-design-review).

## Tiers

| Tier | Where | What you do here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | all four core notebooks; the lab's numpy backend measures your CPU's roofline and disk throughput | $0 |
| **T1** | Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | a real GPU roofline by dtype, HBM bandwidth, pinned vs pageable copies, weight loading | free – ~$0.7/hr |
| **T2** | Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | P2P bandwidth over PCIe vs NVLink, `nvidia-smi topo -m` on real machines | ~$0–25 per session |
| **T3** | GCP via the lab's Terraform | the suite on a Spot L4 VM with auto-stop, results uploaded to a bucket | pay per use |

Prices and obtainability move monthly: see [`COMPUTE.md`](../../COMPUTE.md) (repo root) and the primer's
§10; the whole learning path is in [`CURRICULUM.md`](../../CURRICULUM.md).

## Quick start

```bash
cd roofline-core
python3 -m pip install -r requirements.txt   # only for notebooks and tests; the library needs nothing
python3 -m pytest -q
python3 -m jupyterlab notebooks
```

```python
from roofline import llm, specs
step = llm.decode(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), batch=1, context=1024)
print(step.bound, f"{step.time * 1e3:.2f} ms")          # memory 4.52 ms
```

## Where this sits

- **Builds on:** [`gpu-primer`](../gpu-primer/gpu-primer.md) (why a GPU is shaped the way it is),
  [`gpu-deployment`](../gpu-deployment/gpu-deployment-primer.md) (scale-up vs scale-out, the parallelism menu),
  [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) (sizing, TTFT/TPOT budgets).
- **Leads to:** layer 02 ([`02-cuda-nccl-runtime`](../../02-cuda-nccl-runtime/README.md)) for the execution
  model, memory access patterns and NCCL collectives measured for real; layer 04
  ([`04-inference-engine`](../../04-inference-engine/README.md)) for the engine techniques — batching, paged
  KV, quantization — whose payoff this topic computes; layers 03 and 05 for the cold-start and failure-domain
  consequences at cluster scale.
