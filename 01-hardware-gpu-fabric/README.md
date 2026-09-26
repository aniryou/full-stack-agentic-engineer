# 01 · Hardware and fabric

Read the machine: after this layer you can take a GPU spec sheet, a model and a network diagram and say which
resource bounds an LLM step, what a collective costs on each link, how long a replica takes to load, how often a
big job fails and what a token costs — and then measure the machine you have to check the prediction.

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

This layer is the machine every step is measured against: the limits the runtime (02) drives and the engine (04)
runs into.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`gpu-primer/`](gpu-primer/gpu-primer.md) | explain why a GPU is shaped the way it is, from first principles (primer + [exercises](gpu-primer/gpu-primer-exercises.md)) | ~4 h with `gpu-deployment/` | read |
| [`gpu-deployment/`](gpu-deployment/gpu-deployment-primer.md) | draw the scale-up (NVLink) vs scale-out (InfiniBand/RoCE) boundary and say what it means for LLM serving (primer + [exercises](gpu-deployment/gpu-deployment-exercises.md)) | (above) | read |
| [`roofline-and-fabric/`](roofline-and-fabric/README.md) | predict whether an LLM step is compute- or memory-bound from a spec sheet; price a collective on NVLink vs InfiniBand; budget a cold start, a failure rate and $/M tokens — then measure your own CPU, a GPU, or a Spot L4 on Google Cloud | ~8 h primer + core; ~5 h lab | T0 → T3 |

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* Times are rough and come from the repo's curriculum
([`CURRICULUM.md`](../CURRICULUM.md), modules 01.0–01.5).

## Start here

1. Read [`roofline-and-fabric/PRIMER.md`](roofline-and-fabric/PRIMER.md) §1–§2: spec-sheet literacy and the roofline.
2. `cd roofline-and-fabric/roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` —
   66 tests, ~30 s; then open
   [`01_spec_sheets_and_the_roofline`](roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb).
3. When you have any GPU (even a free Colab T4), measure the real thing with
   [`gpu-bench-lab/notebooks/01_measure_your_roofline`](roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb);
   without one it measures your CPU's roofline instead.

## What is inside `roofline-and-fabric/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](roofline-and-fabric/PRIMER.md) | ten sections: spec-sheet literacy, the roofline model, LLM inference on the roofline (prefill vs decode, KV reads, quantization, MoE), the memory hierarchy, fabrics quantitatively (α-β, tensor-parallel all-reduce cost, rails and bisection, GPUDirect, `nvidia-smi topo -m`), storage and cold start, reliability at scale, the cost of a token, the September 2026 accelerator landscape, getting hardware — then a design-review walkthrough and drills. Every computed number comes from `roofline-core` | read |
| [`roofline-core/`](roofline-and-fabric/roofline-core/README.md) | the minimal implementation, package `roofline`, standard library only: `specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability`, `cost`, and four fill-in notebooks that **predict** what the hardware should do | T0 |
| [`gpu-bench-lab/`](roofline-and-fabric/gpu-bench-lab/README.md) | "measure the machine you have", package `gpubench`: a numpy backend for your own CPU's roofline, memory bandwidth and disk throughput (T0); a PyTorch backend for GEMM by dtype, HBM bandwidth, pinned vs pageable copies and weight loading on one GPU (T1) and the P2P bandwidth matrix on several (T2); `nvidia-smi` topology and inventory parsers with bundled sample output; deploys for any GPU box (Docker, Colab, Kaggle, rented GPUs) and for Google Cloud — Terraform for one Spot L4 VM that runs the suite, uploads the report to a bucket and powers off (T3). Four notebooks that **measure** what the core predicts | T0 → T3 |

Work it a section at a time — read, predict, measure. Each primer section pairs with one core and one lab notebook:
§1–2 → `01`, §3–4 → `02`, §5 → `03`, §6–8 → `04`. The topic [README](roofline-and-fabric/README.md) has the step table.

## Run it

```bash
cd roofline-and-fabric/roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../gpu-bench-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
python3 -m gpubench info            # what is this machine?
```

Then `python3 -m jupyterlab notebooks` in either directory, or the Colab links below.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the primer, all four core notebooks, the lab's numpy backend (your CPU's roofline, disk throughput), `nvidia-smi` parsing on sample output | $0 |
| **T1** | one GPU: Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | a real GPU roofline by dtype, HBM bandwidth, pinned vs pageable copies, weight loading | free – ~$0.7/hr |
| **T2** | ≥ 2 GPUs: Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | P2P bandwidth over PCIe vs NVLink, `nvidia-smi topo -m` on a real machine | ~$0–25 per session |
| **T3** | Google Cloud, via the lab's Terraform | the suite on a Spot L4 VM with auto-stop, report uploaded to a bucket | pay per use (~$0.1–0.3/hr on Spot) |

## How it fits

**Needed first:** [`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md)
(sizing, TTFT/TPOT budgets) and this layer's `gpu-primer/` and `gpu-deployment/`. The
[curriculum's spiral](../CURRICULUM.md#31-why-this-order) visits layer 04's concepts before this layer on purpose
(00 → 04 → 01 → 02 → 04 again with a GPU): with the engine's step loop in mind, each hardware number has a use; the
roofline itself needs only 00. **Leads to** layer 02
([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md): the execution model, memory access patterns and NCCL
collectives behind the fabric numbers here), layer 03 ([`03-kubernetes-gpu`](../03-kubernetes-gpu/README.md): cold
start, failure domains and topology at cluster scale) and layer 04
([`04-inference-engine`](../04-inference-engine/README.md): batching, the KV cache,
[paged attention](../04-inference-engine/paged-attention/paged-attention-primer.md),
[FlashAttention](../04-inference-engine/flash-attention/flash-attention-primer.md) and quantization — the
techniques whose payoff the roofline computes).

## Caveats

- Every time the core prints is a **roofline bound** (ideal overlap, compulsory traffic, peak clocks); real kernels
  land below it, and the lab measures the gap. Lab output is labelled *measured*, *model*, *assumed*, *spec* or
  *sample output (illustrative)*.
- GPU specs, prices and obtainability are a September 2026 snapshot marked (verify); they move monthly.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`roofline-and-fabric/gpu-bench-lab/`** — [01_measure_your_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb) · [02_memory_bandwidth_and_transfers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/02_memory_bandwidth_and_transfers.ipynb) · [03_multi_gpu_topology_and_p2p](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/03_multi_gpu_topology_and_p2p.ipynb) · [04_weights_loading_and_cold_start](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/04_weights_loading_and_cold_start.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/01_measure_your_roofline.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/02_memory_bandwidth_and_transfers.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/03_multi_gpu_topology_and_p2p.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/04_weights_loading_and_cold_start.ipynb)
- **`roofline-and-fabric/roofline-core/`** — [01_spec_sheets_and_the_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb) · [02_llm_inference_on_the_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb) · [03_fabrics_and_collective_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb) · [04_loading_reliability_and_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/04_loading_reliability_and_cost.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/01_spec_sheets_and_the_roofline.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/02_llm_inference_on_the_roofline.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/03_fabrics_and_collective_cost.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/04_loading_reliability_and_cost.ipynb)
<!-- colab-links:end -->
