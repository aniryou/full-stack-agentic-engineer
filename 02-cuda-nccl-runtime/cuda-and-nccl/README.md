# cuda-and-nccl: the GPU software substrate

*Layer 02. CUDA's execution model, collectives, and how a container gets a GPU.*

The software between the silicon and everything above it: how a driver, a CUDA runtime and a compiled
kernel agree to run; how 32-thread warps turn memory accesses into 32-byte sectors; why tiling, fusion and
CUDA Graphs matter for inference; how GPUs all-reduce, all-gather and all-to-all, and what that costs a
tensor-parallel decode step; how a container gets a GPU; how one GPU is shared; and how to read its health.
Every concept is learnable on a laptop (T0). Real GPUs and GCP are optional steps up.

## What's here

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](PRIMER.md) | the concept primer: 9 sections, a design-review walkthrough with 6 drills, glossary, sources, dated verify list | read |
| [`cuda-nccl-core/`](cuda-nccl-core/) | **minimal** implementation, package `gpusim`: seven small numpy simulators (SIMT and memory, occupancy, tiling and launches, collectives, compatibility, sharing, health) and 5 fill-in notebooks. Every formula and worked number in the primer is computed here | T0 |
| [`cuda-nccl-lab/`](cuda-nccl-lab/) | **detailed** implementation, package `gpurt`: CUDA kernels written in Numba (run in the CUDA simulator on CPU, or on a GPU), collectives over OS pipes, gloo (T0) or NCCL (2+ GPUs) with algbw/busbw exactly as nccl-tests computes them and an α-β fit, nccl-tests output parsing, CUDA Graphs vs eager, what a container sees of its GPU, DCGM parsing; deploy assets for any GPU box, GKE and Terraform | T0 fallbacks, then T1, T2, T3 |

## Order to work it

Read a primer section, do the core notebook (T0), then the matching lab notebook, which falls back to T0 when
there is no GPU.

| # | Primer | Core notebook (T0) | Lab notebook | What you can then explain |
|---|---|---|---|---|
| 1 | §2 The execution model · §3 Memory access patterns | [`01_simt_warps_and_memory`](cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) | `01_kernels_in_the_simulator` | warps, coalescing, bank conflicts, divergence |
| 2 | §3 · §4 Streams, launch overhead and CUDA Graphs | [`02_tiling_fusion_and_occupancy`](cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) | `02_memory_bound_kernels_on_a_real_gpu` | GEMM and softmax traffic, occupancy limits, why engines capture decode graphs |
| 3 | §5 Collectives | [`03_collectives_from_scratch`](cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) | `03_collectives_with_torch_distributed`, `04_busbw_and_the_alpha_beta_fit` | all-reduce = reduce-scatter + all-gather, α-β, busbw, TP/EP traffic, hangs |
| 4 | §1 The stack from driver to framework · §6 How a container gets a GPU | [`04_compatibility_and_containers`](cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) | `05_how_a_container_sees_a_gpu` | driver/runtime/CC rules, "no kernel image", what the toolkit injects |
| 5 | §7 Sharing a GPU · §8 Health and observability · §9 On GCP and elsewhere | [`05_sharing_and_health`](cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) | `06_gpu_sharing_and_dcgm_on_gke` | MIG vs MPS vs time-slicing, GPU util vs SM active, XID triage |

Finish with the primer's *In a design review* drills. The curriculum's modules 02.1–02.5 follow this order
([CURRICULUM.md](../../CURRICULUM.md)).

## Tiers

| Tier | Where | What runs here |
|---|---|---|
| **T0** | laptop, Colab CPU, CI ($0) | all of `cuda-nccl-core`; in the lab, Numba kernels in the CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`), collectives over OS pipes (a real multi-process ring, no torch) or gloo, and parsing sample nccl-tests, nvidia-smi and DCGM output |
| **T1** | one GPU: Colab or Kaggle T4, an L4, any rented 24 GB GPU | real kernel timings against the roofline, CUDA Graphs vs eager, what a container sees |
| **T2** | 2+ GPUs: Kaggle 2 × T4 (free, PCIe), a rented NVLink box for an hour | NCCL collectives, nccl-tests, busbw and α-β fits, P2P |
| **T3** | GCP: GKE through Terraform | an L4 Spot pool that scales from zero, a 2-GPU nccl-tests Job, time-sharing and MIG pools, DCGM metrics |

Costs, free options and cleanup habits are in [COMPUTE.md](../../COMPUTE.md).

## Where this sits in the stack

Below it, [layer 01](../../01-hardware-gpu-fabric/roofline-and-fabric/) supplies the roofline, the link rates and
the α-β model this topic uses. Above it, [layer 03](../../03-kubernetes-gpu/gpu-scheduling/) schedules GPUs that
the device plugin advertises, and shares them with MIG and time-sharing at cluster level.
[Layer 04](../../04-inference-engine/serving-engine/) runs the kernels, CUDA graphs and tensor-parallel
all-reduces described here inside an engine. [Layer 05](../../05-orchestrator/serving-orchestration/) moves KV
caches between prefill and decode workers over the same fabrics.
