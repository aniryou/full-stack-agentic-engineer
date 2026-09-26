# 02 · Runtime — CUDA, NCCL, the container runtime and the driver

After this layer you can explain why a GPU kernel is fast or slow, what an all-reduce costs a tensor-parallel
decode step, why a container does or does not see its GPU, and which GPU metrics to trust.

## Where this layer sits

```
  05-orchestrator           routes and scales engine replicas; moves KV caches between them
  04-inference-engine       runs the kernels, captures CUDA Graphs, all-reduces across GPUs
  03-kubernetes-gpu         the device plugin hands a pod its GPUs
▶ 02-cuda-nccl-runtime      driver, CUDA runtime, kernels, NCCL, container runtime, GPU sharing, health
  01-hardware-gpu-fabric    SMs, HBM, NVLink, PCIe, NICs: the limits everything above works within
```

The software that turns raw GPUs into a usable, multi-GPU compute substrate: it sits between the hardware
and any scheduler or engine above it.

| Topic | What you learn | Tier |
|---|---|---|
| [`cuda-and-nccl/`](cuda-and-nccl/README.md) | CUDA's execution model and memory access patterns, launch overhead and CUDA Graphs, collectives and NCCL, driver/CUDA/GPU compatibility, how a container gets a GPU, MIG vs MPS vs time-slicing, DCGM and XIDs — a primer, a numpy core and a hands-on lab | T0 → T3 |

## Start here

1. Read the one-minute version of [`cuda-and-nccl/PRIMER.md`](cuda-and-nccl/PRIMER.md#the-one-minute-version) (a page).
2. `cd cuda-and-nccl/cuda-nccl-core && python3 -m pytest -q` — 128 tests in under a second, numpy only.
3. Open [`01_simt_warps_and_memory`](cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb)
   locally, or through its Colab badge below. The [topic README](cuda-and-nccl/README.md) has the full order.

## What you get

**Tiers:** T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a
multi-GPU box, rented for an hour (Kaggle's 2 × T4 is a free one); T3 = the Google Cloud deployment, optional.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`cuda-and-nccl/PRIMER.md`](cuda-and-nccl/PRIMER.md) | explain the layer in a design review: nine sections from the driver stack to DCGM, worked numbers, drills with answers, a dated verify list | ~9 h with the core | read |
| [`cuda-and-nccl/cuda-nccl-core/`](cuda-and-nccl/cuda-nccl-core/README.md) | predict sectors, bank conflicts, occupancy, GEMM and softmax traffic, collective costs and busbw, compatibility errors, MIG layouts and XID owners with seven numpy simulators (5 notebooks) | (with the primer) | T0 |
| [`cuda-and-nccl/cuda-nccl-lab/`](cuda-and-nccl/cuda-nccl-lab/README.md) | check it for real: Numba CUDA kernels (simulator, then GPU), collectives measured like nccl-tests with an α-β fit, what a container sees of its GPU, DCGM on GKE (6 notebooks; deploy assets for any GPU box, GKE and Terraform) | ~7 h at T0 (01–05), ~2 h for 06 | T0 → T3 |

## Run it

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core
python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 128 tests, under a second
cd ../cuda-nccl-lab
python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 122 pass, 2 skip; ~20 s
python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M          # a real ring all-reduce, no GPU
```

Or open any notebook below in Colab: its first cell clones the repo and installs the lab.

## How it fits

- **Builds on** layer 01 ([`01-hardware-gpu-fabric`](../01-hardware-gpu-fabric/README.md)): the roofline,
  the memory hierarchy, link rates and the α-β model in
  [roofline-and-fabric](../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §2, §4 and §5.
- **Leads to** layer 03 (`03-kubernetes-gpu/`: the device plugin, MIG and time-sharing per node pool),
  layer 04 ([`04-inference-engine`](../04-inference-engine/README.md): kernels, CUDA Graphs and TP
  all-reduces inside an engine) and layer 05 (`05-orchestrator/`: KV transfer over the same fabrics).

## Going further / caveats

- The core is simulated (a model of documented NVIDIA behaviour); the lab labels its T0 model output as
  predictions and its sample tool outputs as illustrative. Real measurements need a GPU: free on Colab or
  Kaggle (2 × T4, PCIe only), roughly $2–25 for an hour on a rented NVLink box (verify).
- The GKE deployment is optional: GPU pools scale from zero, the idle cluster costs a few dollars a day
  (verify), and `terraform destroy` ends it. Prices: `COMPUTE.md` at the repo root.
- Driver tables, MIG profiles, DCGM field lists and versions are dated 2026-09-26 and marked (verify).

## Scope of this layer

**Covers:** NVIDIA driver & CUDA toolkit versioning and compatibility, cuDNN/cuBLAS,
NCCL collectives (all-reduce, all-gather) and topology awareness, the container
runtime (containerd, NVIDIA Container Toolkit), device plugins, MIG partitioning,
CUDA graphs, kernels & memory model.

**Signal keywords:** CUDA, cuDNN, NCCL, all-reduce, collective, driver, MIG,
nvidia-container-toolkit, containerd, PTX, kernel, CUDA graph, compute capability.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`cuda-and-nccl/cuda-nccl-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) `01_simt_warps_and_memory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) `02_tiling_fusion_and_occupancy.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) `03_collectives_from_scratch.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) `04_compatibility_and_containers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) `05_sharing_and_health.ipynb`

**`cuda-and-nccl/cuda-nccl-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/01_simt_warps_and_memory.ipynb) `01_simt_warps_and_memory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/02_tiling_fusion_and_occupancy.ipynb) `02_tiling_fusion_and_occupancy.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/03_collectives_from_scratch.ipynb) `03_collectives_from_scratch.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/04_compatibility_and_containers.ipynb) `04_compatibility_and_containers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/solutions/05_sharing_and_health.ipynb) `05_sharing_and_health.ipynb`

**`cuda-and-nccl/cuda-nccl-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb) `01_kernels_in_the_simulator.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) `02_memory_bound_kernels_on_a_real_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/03_collectives_with_torch_distributed.ipynb) `03_collectives_with_torch_distributed.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) `04_busbw_and_the_alpha_beta_fit.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/05_how_a_container_sees_a_gpu.ipynb) `05_how_a_container_sees_a_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) `06_gpu_sharing_and_dcgm_on_gke.ipynb`

**`cuda-and-nccl/cuda-nccl-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/01_kernels_in_the_simulator.ipynb) `01_kernels_in_the_simulator.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/02_memory_bound_kernels_on_a_real_gpu.ipynb) `02_memory_bound_kernels_on_a_real_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/03_collectives_with_torch_distributed.ipynb) `03_collectives_with_torch_distributed.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/04_busbw_and_the_alpha_beta_fit.ipynb) `04_busbw_and_the_alpha_beta_fit.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/05_how_a_container_sees_a_gpu.ipynb) `05_how_a_container_sees_a_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/solutions/06_gpu_sharing_and_dcgm_on_gke.ipynb) `06_gpu_sharing_and_dcgm_on_gke.ipynb`
<!-- colab-links:end -->
