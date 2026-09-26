# cuda-and-nccl — the GPU software substrate, from driver to collective

After this topic you can explain what makes a GPU kernel fast or slow, what an all-reduce costs a
tensor-parallel decode step, why a container does or does not see its GPU, and which GPU metrics to trust.

## Start here

1. Read [PRIMER.md](PRIMER.md): *The one-minute version*, then §2–§3 (the execution model, memory access patterns).
2. `cd cuda-nccl-core && python3 -m pytest -q` — 128 tests in under a second, numpy only; then open
   [`notebooks/01_simt_warps_and_memory.ipynb`](cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb).
3. `cd ../cuda-nccl-lab && python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M` — a real ring
   all-reduce between two processes, printed like nccl-tests, in about a second. Then
   [`notebooks/01_kernels_in_the_simulator.ipynb`](cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb)
   runs real CUDA kernels on your CPU.

## What you get

**Tiers:** T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a
multi-GPU box, rented for an hour (Kaggle's 2 × T4 is a free one); T3 = the Google Cloud deployment, optional.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the layer from first principles: 9 sections with worked numbers, a design-review walkthrough and 6 drills, a glossary, sources and a dated verify list | ~9 h with the core | read |
| [`cuda-nccl-core/`](cuda-nccl-core/) | predict sectors per warp request, bank conflicts, occupancy, GEMM and softmax traffic, collective costs and busbw, compatibility errors, MIG layouts and XID owners — seven numpy simulators (`gpusim`) that compute every worked number in the primer, and 5 fill-in notebooks | (with the primer) | T0 |
| [`cuda-nccl-lab/`](cuda-nccl-lab/) | check it for real — Numba CUDA kernels (simulator, then GPU), collectives measured like nccl-tests with an α-β fit, CUDA Graphs vs eager, what a container sees of its GPU, DCGM fields and alert rules (`gpurt`, 6 notebooks) | ~7 h at T0 (01–05), ~2 h for 06 | T0 → T3 |
| [`cuda-nccl-lab/deploy/`](cuda-nccl-lab/deploy/) | run the lab on a real GPU: Colab, Kaggle 2 × T4, rented GPUs, Docker ([`any-gpu`](cuda-nccl-lab/deploy/any-gpu/README.md)); a GKE cluster ([`gcp`](cuda-nccl-lab/deploy/gcp/README.md)) and its Jobs ([`gke`](cuda-nccl-lab/deploy/gke/README.md)) | per session | T1–T3 |

What runs at each tier:

| Tier | Where | What runs here |
|---|---|---|
| **T0** | laptop, Colab CPU, CI ($0) | all of `cuda-nccl-core`; in the lab, Numba kernels in the CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`), collectives over OS pipes (a real multi-process ring, no torch) or gloo, and parsing sample nccl-tests, probe and DCGM output |
| **T1** | one GPU: Colab or Kaggle T4, an L4, any rented 24 GB GPU | real kernel timings against the roofline, CUDA Graphs vs eager, what a container sees |
| **T2** | 2+ GPUs: Kaggle 2 × T4 (free, PCIe), a rented NVLink box for an hour | NCCL collectives, nccl-tests, busbw and α-β fits, P2P |
| **T3** | GCP: GKE through Terraform | an L4 Spot pool that scales from zero, a 2-GPU nccl-tests Job, time-sharing and MIG pools, DCGM metrics |

### The order to work it

Read a primer section, do the core notebook (T0), then the matching lab notebook, which falls back to a
labelled T0 path when there is no GPU. Hours are the course plan's estimates for all three together.

| # | Primer | Core notebook (T0) | Lab notebook | What you can then explain | Hours |
|---|---|---|---|---|---|
| 1 | §2 The execution model · §3 Memory access patterns | [`01_simt_warps_and_memory`](cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) | [`01_kernels_in_the_simulator`](cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb) (T0) | warps, coalescing, bank conflicts, divergence | 3 |
| 2 | §3 · §4 Streams, launch overhead and CUDA Graphs | [`02_tiling_fusion_and_occupancy`](cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) | [`02_memory_bound_kernels_on_a_real_gpu`](cuda-nccl-lab/notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) (T1; T0 model path) | GEMM and softmax traffic, occupancy limits, why engines capture decode graphs | 3 |
| 3 | §5 Collectives | [`03_collectives_from_scratch`](cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) | [`03_collectives_with_torch_distributed`](cuda-nccl-lab/notebooks/03_collectives_with_torch_distributed.ipynb), [`04_busbw_and_the_alpha_beta_fit`](cuda-nccl-lab/notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) (T0; NCCL at T2) | all-reduce = reduce-scatter + all-gather, α-β, busbw, TP/EP traffic, hangs | 4 |
| 4 | §1 The stack from driver to framework · §6 How a container gets a GPU | [`04_compatibility_and_containers`](cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) | [`05_how_a_container_sees_a_gpu`](cuda-nccl-lab/notebooks/05_how_a_container_sees_a_gpu.ipynb) (T0; your T1/T3 logs) | driver/runtime/CC rules, "no kernel image", what the toolkit injects | 3 |
| 5 | §7 Sharing a GPU · §8 Health and observability · §9 On GCP and elsewhere | [`05_sharing_and_health`](cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) | [`06_gpu_sharing_and_dcgm_on_gke`](cuda-nccl-lab/notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) (T0; T1/T2 on a GPU VM; T3) | MIG vs MPS vs time-slicing, GPU util vs SM active, XID triage | 3 (+2 on GKE) |

Finish with the primer's [*In a design review*](PRIMER.md#in-a-design-review) drills. These five steps are the course plan's modules
02.1–02.5 ([`CURRICULUM.md`](../../CURRICULUM.md)).

## Run it

```bash
# T0 — the core: numpy only
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core
python3 -m pip install -r requirements.txt
python3 -m pytest -q                            # 128 tests, under a second
python3 -m jupyterlab notebooks                 # do the exercises

# T0 — the lab: numpy, numba (its CUDA simulator runs on any CPU), pyyaml
cd ../cuda-nccl-lab
python3 -m pip install -r requirements.txt
python3 -m pytest -q                            # 122 pass, 2 skip without torch / numba-cuda; ~20 s
python3 -m jupyterlab notebooks
```

On Colab, each notebook's first cell clones the repo and installs its lab; the badges are in the
[layer README](../README.md). For a GPU (T1/T2) or GKE (T3), follow the lab's
[Run it](cuda-nccl-lab/README.md#run-it) section.

## How it fits

- **Builds on** layer 01's [roofline-and-fabric](../../01-hardware-gpu-fabric/roofline-and-fabric/README.md)
  primer — the roofline (§2), the memory hierarchy (§4), link rates and the α-β model (§5) — and the
  [GPU primer](../../01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) for the anatomy of an SM.
- **Leads to** layer 03 ([`03-kubernetes-gpu/gpu-scheduling/`](../../03-kubernetes-gpu/gpu-scheduling/): the device plugin, MIG and time-sharing at
  cluster level), layer 04 ([`04-inference-engine/serving-engine/`](../../04-inference-engine/serving-engine/): these kernels, CUDA Graphs and TP
  all-reduces inside an engine; also the [FlashAttention](../../04-inference-engine/flash-attention/flash-attention-primer.md)
  and [PagedAttention](../../04-inference-engine/paged-attention/paged-attention-primer.md) primers) and
  layer 05 ([`serving-orchestration`](../../05-orchestrator/serving-orchestration/README.md): KV transfer between prefill and decode workers over
  the same fabrics).

## Going further / caveats

- **Simulated vs measured.** Everything in `cuda-nccl-core` is simulated: a model of documented NVIDIA
  behaviour, not a measurement. The lab labels its T0 model output as predictions; its pipes and gloo sweeps
  are real timings of a CPU backend, not of a GPU fabric; its bundled nccl-tests, probe and DCGM outputs are
  illustrative samples in the tools' documented formats.
- **Real hardware.** T1 is free on Colab or Kaggle. Kaggle's 2 × T4 is a free T2 box, but PCIe only: no
  NVLink numbers. A rented 24 GB GPU is roughly $0.3–0.7/hr, an hour on a multi-GPU NVLink box $2–25
  (verify). Prices and obtainability: [`COMPUTE.md`](../../COMPUTE.md).
- **Google Cloud (T3).** Idle, the lab's GKE cluster costs the management fee (one zonal cluster is covered
  by GKE's free-tier credit, verify) and one e2-standard-4 system node, a few dollars a day (verify). GPU
  pools cost nothing until a pod asks for a GPU; the MIG pool (A100) is off by default and needs quota. Run
  `terraform destroy` after each session ([`deploy/gcp`](cuda-nccl-lab/deploy/gcp/README.md)).
- **Dated facts.** Driver tables, MIG profiles, DCGM field lists, NCCL versions, image tags and prices are
  dated 2026-09-26 and marked (verify); the primer's [Verify list](PRIMER.md#verify-list) collects them.
