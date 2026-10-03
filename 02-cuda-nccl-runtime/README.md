# 02 · CUDA, NCCL and runtime

After this layer, you can explain these things:

- Why a GPU kernel is fast or slow.
- What an all-reduce costs a tensor-parallel decode step.
- Why a container sees its GPU, or does not see it.
- Which GPU metrics you can trust.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, memory, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, model routing, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the software that changes raw GPUs into a multi-GPU compute substrate that you can use. The hardware
(01) is below this layer. Any scheduler (03) or engine (04) is above it.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`cuda-and-nccl/`](cuda-and-nccl/README.md) | explain CUDA's execution model and memory access patterns, launch overhead and CUDA Graphs. Explain collectives and NCCL, and driver/CUDA/GPU compatibility. Explain how a container gets a GPU, and the differences between MIG, MPS and time-slicing. Explain DCGM and XIDs. The topic has a primer, a numpy core and a hands-on lab. | ~9 h primer + core, ~9 h lab | T0 to T3 |

## Start here

1. Read the one-minute version of [`cuda-and-nccl/PRIMER.md`](cuda-and-nccl/PRIMER.md#the-one-minute-version). It is one page.
2. Run `cd cuda-and-nccl/cuda-nccl-core && python3 -m pytest -q`. It runs 141 tests in about 30 s, with numpy only.
3. Open [`01_simt_warps_and_memory`](cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb)
   on your computer, or through its Colab link in "Run in Colab". The [topic README](cuda-and-nccl/README.md) gives the full order.

## What you get

**Tiers:** T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab or Kaggle T4, or a rented card).
T2 is a multi-GPU box that you rent for an hour. Kaggle's 2 × T4 is a multi-GPU box at no cost. T3 is the Google Cloud
deployment, and it is optional.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`cuda-and-nccl/PRIMER.md`](cuda-and-nccl/PRIMER.md) | explain the layer in a design review. The primer has nine sections from the driver stack to DCGM, worked numbers, drills with answers and a dated verify list. | ~9 h with the core | read |
| [`cuda-and-nccl/cuda-nccl-core/`](cuda-and-nccl/cuda-nccl-core/README.md) | predict sectors, bank conflicts, occupancy, and GEMM and softmax traffic. Also predict collective costs and busbw, compatibility errors, MIG layouts and XID owners. The core makes these predictions with seven numpy simulators (5 notebooks). | (with the primer) | T0 |
| [`cuda-and-nccl/cuda-nccl-lab/`](cuda-and-nccl/cuda-nccl-lab/README.md) | examine the layer in practice (6 notebooks). Run Numba CUDA kernels in the simulator, then on a GPU. Measure collectives as nccl-tests does, with an α-β fit. Find what a container sees of its GPU. See DCGM on GKE. The lab also has deploy assets for any GPU box, GKE and Terraform. | ~7 h at T0 (01–05), ~2 h for 06 | T0 to T3 |

## Run it

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 141 tests, ~30 s
cd ../cuda-nccl-lab
python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 130 pass, 3 skip; ~50 s
python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M          # a real ring all-reduce, no GPU
```

You can also open any notebook in Colab through its link in the "Run in Colab" section. Its first cell clones the repo and installs the lab.

## How it fits

- **Builds on** layer 01 ([`01-hardware-gpu-fabric`](../01-hardware-gpu-fabric/README.md)). You must know these
  topics first: the roofline, the memory hierarchy, link rates and the α-β model in
  [roofline-and-fabric](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §2, §4 and §5. The
  [curriculum's spiral](../CURRICULUM.md#31-why-this-order) comes to this layer after layer 00, the concepts of
  layer 04, and layer 01. Thus you learn tiling, fusion and CUDA Graphs on an engine that you know already. Nothing in 04 is a
  prerequisite for this layer.
- **Leads to** layers 03, 04 and 05. Layer 03 ([`03-kubernetes-gpu/`](../03-kubernetes-gpu/)) has the device plugin,
  MIG and time-sharing per node pool. Layer 04 ([`04-inference-engine`](../04-inference-engine/README.md)) has
  kernels, CUDA Graphs and TP all-reduces inside an engine. Layer 05 ([`05-orchestrator`](../05-orchestrator/README.md))
  has KV transfer over the same fabrics.

## Going further / caveats

- The core is a simulation. It is a model of documented NVIDIA behaviour. The lab marks its T0 model output as
  predictions, and its sample tool outputs as illustrative. For real measurements, a GPU is necessary. You can use a GPU at
  no cost on Colab or Kaggle (2 × T4, PCIe only). A rented NVLink box costs approximately $2–25 for an hour (verify).
- The GKE deployment is optional. GPU pools scale from zero. The idle cluster costs a few dollars a day (verify).
  `terraform destroy` ends it. For the prices, see [`COMPUTE.md`](../COMPUTE.md).
- The driver tables, MIG profiles, DCGM field lists and versions have the date 2026-09-26 and the tag (verify).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`cuda-and-nccl/cuda-nccl-core/`** — [01_simt_warps_and_memory](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) · [02_tiling_fusion_and_occupancy](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) · [03_collectives_from_scratch](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) · [04_compatibility_and_containers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) · [05_sharing_and_health](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/01_simt_warps_and_memory.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/02_tiling_fusion_and_occupancy.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/03_collectives_from_scratch.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/04_compatibility_and_containers.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/05_sharing_and_health.ipynb)
- **`cuda-and-nccl/cuda-nccl-lab/`** — [01_kernels_in_the_simulator](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb) · [02_memory_bound_kernels_on_a_real_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) · [03_collectives_with_torch_distributed](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/03_collectives_with_torch_distributed.ipynb) · [04_busbw_and_the_alpha_beta_fit](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) · [05_how_a_container_sees_a_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/05_how_a_container_sees_a_gpu.ipynb) · [06_gpu_sharing_and_dcgm_on_gke](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/01_kernels_in_the_simulator.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/02_memory_bound_kernels_on_a_real_gpu.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/03_collectives_with_torch_distributed.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/04_busbw_and_the_alpha_beta_fit.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/05_how_a_container_sees_a_gpu.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/06_gpu_sharing_and_dcgm_on_gke.ipynb)
<!-- colab-links:end -->
