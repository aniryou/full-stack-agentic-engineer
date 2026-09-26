# roofline-and-fabric — read a GPU spec sheet and predict what a model will do on it

After this topic you can look at a GPU, a model and a fabric and say which resource bounds an LLM step, what a
collective costs on each link, how long a replica takes to load, how often a big job fails and what a token costs —
numbers you can defend in a design review.

## Start here

1. Read [PRIMER.md](PRIMER.md) "The one-minute version", then §1–§2: spec-sheet literacy and the roofline.
2. `cd roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 58 tests, well under a
   second; then open [`01_spec_sheets_and_the_roofline`](roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb).
3. Measure the real thing with [`gpu-bench-lab/notebooks/01_measure_your_roofline`](gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb):
   on a laptop it measures your CPU's roofline; on any GPU (even a free Colab T4) it measures the GPU's.

The fastest win, with nothing installed:

```python
from roofline import llm, specs    # run from roofline-core/
step = llm.decode(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), batch=1, context=1024)
print(step.bound, f"{step.time * 1e3:.2f} ms")          # memory 4.52 ms
```

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the concepts, sections 1–10: spec sheets, the roofline, LLM inference on the roofline, the memory hierarchy, fabrics, storage and cold start, reliability, cost, the September 2026 accelerator landscape, getting hardware; then a design-review walkthrough and drills. Every computed number comes from the core and is pinned by its tests | read alongside the core | — |
| [`roofline-core/`](roofline-core/README.md) | **predict** step times, collective costs, cold starts, failure rates and $/M tokens with the minimal implementation: package `roofline`, seven standard-library modules (`specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability`, `cost`) and four fill-in notebooks | ~8 h with the primer | T0 |
| [`gpu-bench-lab/`](gpu-bench-lab/README.md) | **measure** the machine you have with the detailed lab, package `gpubench`: numpy (CPU) and torch (CUDA) backends for GEMM throughput, memory bandwidth, host↔device and GPU↔GPU transfers, weight loading; `nvidia-smi` topology and inventory parsers; Docker and GCP Terraform deploys | ~5 h | T0 → T3 |

Times are rough and come from the repo's curriculum (`CURRICULUM.md` at the repo root, modules 01.1–01.5).

### Work it in this order

Each step pairs a primer section with a core notebook (predict) and a lab notebook (measure).

| Step | Read | Predict (core, T0) | Measure (lab) |
|---|---|---|---|
| 1 | [PRIMER §1–2](PRIMER.md#1-spec-sheet-literacy) spec sheets, the roofline | `01_spec_sheets_and_the_roofline` | `01_measure_your_roofline` — T0 on your CPU, T1 on a GPU |
| 2 | [PRIMER §3–4](PRIMER.md#3-llm-inference-on-the-roofline) LLM steps per step and per kernel, KV reads, quantization, MoE; tiling and fusion | `02_llm_inference_on_the_roofline` | `02_memory_bandwidth_and_transfers` — T0 / T1 |
| 3 | [PRIMER §5](PRIMER.md#5-fabrics-quantitatively) α-β, collectives, TP cost, rails, topology | `03_fabrics_and_collective_cost` | `03_multi_gpu_topology_and_p2p` — T2 (T0 falls back to sample topology output) |
| 4 | [PRIMER §6–8](PRIMER.md#6-storage-and-cold-start) cold start, reliability, $/M tokens | `04_loading_reliability_and_cost` | `04_weights_loading_and_cold_start` — T0 / T1 |
| 5 | [PRIMER §9–10](PRIMER.md#9-the-accelerator-landscape-september-2026-snapshot) the landscape, getting hardware | — | lab `deploy/any-gpu/` (T1/T2) or `deploy/gcp/terraform/` (T3) |

Finish with the primer's [design-review walkthrough and drills](PRIMER.md#in-a-design-review).

## Run it

```bash
cd roofline-core
python3 -m pip install -r requirements.txt   # only for notebooks and tests; the library needs nothing
python3 -m pytest -q
python3 -m jupyterlab notebooks

cd ../gpu-bench-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q
python3 -m gpubench run --out results        # the measurement suite on this machine
```

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | Where | What you do here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | all four core notebooks; the lab's numpy backend measures your CPU's roofline and disk throughput | $0 |
| **T1** | Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | a real GPU roofline by dtype, HBM bandwidth, pinned vs pageable copies, weight loading | free – ~$0.7/hr |
| **T2** | Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | P2P bandwidth over PCIe vs NVLink, `nvidia-smi topo -m` on real machines | ~$0–25 per session |
| **T3** | GCP via the lab's Terraform | the suite on a Spot L4 VM with auto-stop, results uploaded to a bucket | pay per use |

Prices and obtainability move monthly: see `COMPUTE.md` (repo root) and the primer's §10; the whole learning path is
in `CURRICULUM.md` (repo root).

## How it fits

| | Read | For |
|---|---|---|
| before | [`gpu-primer`](../gpu-primer/gpu-primer.md) | why a GPU is shaped the way it is |
| before | [`gpu-deployment`](../gpu-deployment/gpu-deployment-primer.md) | scale-up vs scale-out, the parallelism menu |
| before | [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) | sizing, TTFT/TPOT budgets |
| after | layer 02 ([`02-cuda-nccl-runtime`](../../02-cuda-nccl-runtime/README.md)) | the execution model, memory access patterns and NCCL collectives measured for real |
| after | layer 04 ([`04-inference-engine`](../../04-inference-engine/README.md)) | the engine techniques — batching, paged KV, quantization — whose payoff this topic computes |
| after | layers 03 and 05 ([`03-kubernetes-gpu`](../../03-kubernetes-gpu/README.md), [`05-orchestrator`](../../05-orchestrator/README.md)) | the cold-start and failure-domain consequences at cluster scale |

## Caveats

- **Predicted vs measured.** Every time the core prints is a roofline bound; real kernels land below it, and the lab
  measures the gap on your hardware.
- **Dated facts.** Accelerator specs, prices and obtainability are a September 2026 snapshot marked (verify).
