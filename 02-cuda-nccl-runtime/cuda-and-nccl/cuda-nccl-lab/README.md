# cuda-nccl-lab — the GPU software substrate, hands-on

After this lab, you can do these things:

- Write a CUDA kernel and do a check of it on a laptop. Then measure its time on a GPU.
- Measure an all-reduce and read its busbw the way nccl-tests does.
- Calculate α and β from a sweep.
- From inside a container, find why it does or does not see its GPU.
- Read DCGM metrics correctly, also when "GPU util" gives an incorrect picture.

## Start here

1. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. 130 tests pass and 3 skip (they
   need torch or numba-cuda). The run takes 15–30 s and uses only the simulator.
2. Run `python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M`. This is a real ring all-reduce across two
   OS processes. It prints its result in the layout of nccl-tests, in about a second.
3. Open [`notebooks/01_kernels_in_the_simulator.ipynb`](notebooks/01_kernels_in_the_simulator.ipynb). It runs real
   CUDA kernels in the simulator of Numba. With any GPU (a free Colab T4 is sufficient),
   [`02_memory_bound_kernels_on_a_real_gpu`](notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) measures the
   time of the same source.

## What you get

**Tiers:** T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2
is a multi-GPU box that you rent for an hour. Kaggle's 2 × T4 is a free T2 box. T3 is the Google Cloud deployment,
and it is optional.

Every notebook runs at T0. With a GPU, two GPUs or a GKE cluster, the same notebooks measure real hardware.

Each notebook has worked examples, 4–6 exercises with a ✅ check after each, and an *In a design review* section
with drill questions. The exercises are in `notebooks/`, and the worked answers are in `solutions/`. The times are
for the T0 path. The course plan gives about 7 hours to notebooks 01–05.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`01_kernels_in_the_simulator`](notebooks/01_kernels_in_the_simulator.ipynb) | Index threads and do bounds checks. Write grid-stride loops and a shared-memory reduction. Measure coalescing and bank conflicts from the transpose kernels. See the reuse that tiling gives. | ~1½ h | T0 |
| [`02_memory_bound_kernels_on_a_real_gpu`](notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) | Calculate the effective bandwidth and the latency floor. Measure the time of transpose and fusion on real hardware. Think about atomics and determinism, launch overhead and CUDA Graphs. | ~1½ h | T1, with a T0 model path |
| [`03_collectives_with_torch_distributed`](notebooks/03_collectives_with_torch_distributed.ipynb) | Give the semantics of each collective. Show all-reduce = reduce-scatter + all-gather. Write and run the ring schedule. Derive the busbw factor that comes from the schedule. Calculate the size of TP messages. | ~1½ h | T0 (pipes or gloo), T2 NCCL |
| [`04_busbw_and_the_alpha_beta_fit`](notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) | Calculate the nccl-tests numbers again. Fit α-β. Find S½ in two ways. Calculate the cost of TP for a decode step. Find if the plateau is at the link. | ~1½ h | T0 on samples, also your T1/T2 logs |
| [`05_how_a_container_sees_a_gpu`](notebooks/05_how_a_container_sees_a_gpu.ipynb) | Read mountinfo. Tell the difference between the injection of the toolkit and the device plugin of GKE. Apply the driver/runtime and kernel-image gates. Find what error numbers mean. | ~1½ h | T0 live and on samples, also your T1/T3 logs |
| [`06_gpu_sharing_and_dcgm_on_gke`](notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) | Trace the path from node pools to advertised GPUs to selectors. Explain the utilisation paradox. Read clock-event bits in PromQL. Route alerts by owner. Say which rules can fire with the fields of an exporter. | ~1½ h, +2 h on GKE | T0, T1/T2 on a GPU VM, T3 |
| [`deploy/any-gpu/`](deploy/any-gpu/README.md) | Run the T1/T2 paths anywhere: Colab, Kaggle 2 × T4, RunPod/Vast/Lambda and Docker. Run nccl-tests, dcgm-exporter, MIG and MPS by hand. | per session | T1, T2 |
| [`deploy/gcp/`](deploy/gcp/README.md), [`deploy/gke/`](deploy/gke/README.md) | Build a zonal GKE cluster with Spot L4 pools from zero. Then run the smoke, CUDA sample, 2-GPU nccl-tests, time-sharing and MIG Jobs, and the DCGM alert rules. | ~2 h per session | T3 |

What each tier adds:

| Tier | Where | What it runs |
|---|---|---|
| **T0** | laptop, Colab CPU, CI ($0) | Kernels in the CUDA simulator (correctness, thread indices, shared memory, barriers, and per-warp memory requests that the lab traces from the kernel itself). Collectives over a real multi-process ring (`pipes`) or gloo. nccl-tests, probe and DCGM samples. Every model prediction has a label that says so. |
| **T1** | one GPU (Colab/Kaggle T4, L4, rented 24 GB GPU) | kernel timings with CUDA events, effective bandwidth against peak, launch overhead, CUDA Graphs against eager mode (torch), the live container probe with the driver API |
| **T2** | 2+ GPUs (Kaggle 2×T4 free, PCIe only, or a rented NVLink box) | NCCL sweeps through `torchrun`, nccl-tests (pinned v2.20.0) against the same NCCL, and busbw against the link. On a GPU VM: dcgm-exporter in Docker, and MIG and MPS by hand. |
| **T3** | GKE through Terraform (optional) | A smoke and probe Job, a CUDA sample, a 2-GPU nccl-tests Job on `g2-standard-24`, and time-sharing and MIG pools. DCGM metrics in Managed Prometheus. Alert rules as `ClusterRules`, which its Alertmanager routes. |

## Run it

**T0, on any laptop:**

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
python3 -m pip install -r requirements.txt      # numpy, numba, pyyaml + notebook/test tooling
python3 -m pytest -q                            # 130 pass, 3 skip (2 without torch / numba-cuda); ~50 s, simulator only
python3 -m gpurt.env                            # tier, GPUs, numba mode
python3 -m gpurt.container                      # how this process sees a GPU (on a laptop: it doesn't — and why)
python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M   # a real ring all-reduce over pipes
python3 -m gpurt.nccltests gpurt/fixtures/nccl_all_reduce_8gpu_sample.txt --op all_reduce
python3 -m jupyterlab notebooks
```

**On a GPU box (T1/T2)**, run `pip install -e ".[gpu]"` (NVIDIA's `numba-cuda`). Also install a CUDA build of
PyTorch. Then run `python -m gpurt.kernels.bench` and `torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl`.
[`deploy/any-gpu`](deploy/any-gpu/README.md) has the recipes for Colab, Kaggle 2×T4, RunPod/Vast/Lambda and Docker.

**On GKE (T3)**: [`deploy/gcp`](deploy/gcp/README.md) creates the cluster with Terraform. Then `run.sh` in
[`deploy/gke`](deploy/gke/README.md) runs each Job and saves its log under `out/`. With `DRY_RUN=1`, it only prints
the commands.

**Regenerating and checking** (the builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py`):

```bash
python3 tools/build_notebooks.py                        # notebooks_src/*.py -> notebooks/ + solutions/
python3 tools/run_notebooks.py solutions                # solutions must run clean (CPU, offline)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
python3 tools/make_fixtures.py                          # the illustrative nccl-tests samples
python3 tools/check_ptx.py                              # PTX for sm_75/sm_89 (needs numba-cuda; exit 2 = skipped)
kubernetes-validate --strict -k 1.34.0 deploy/gke/0*.yaml
make check                                              # tests + both notebook runs
```

`gpurt.dcgm.rules_manifest()` generates `deploy/gke/06-dcgm-alert-rules.yaml`. A test keeps the two in sync. In
the manifests and in Terraform, a `VERIFY` mark shows a product detail. Examine each of these details again before
you use it.

## How it fits

This is the **detailed** implementation for [`cuda-and-nccl`](../README.md). The minimal numpy simulators are in
[`../cuda-nccl-core`](../cuda-nccl-core/), and the concepts are in [the primer](../PRIMER.md). Each module in the
section *The library* names the primer section that it measures. Each notebook links its sections at the top.

## Going further / caveats

* **Measured**: anything that the T1/T2 path of a notebook prints, or that the pipes/gloo sweeps print. These
  are real timings of the backend that the output names next to them.
* **Simulated / model prediction**: the T0 paths. They print byte counts, and α-β and launch models with their
  assumptions.
* **Illustrative samples**: `gpurt/fixtures/*`. These are nccl-tests, dcgm-exporter and probe outputs in the
  documented formats of the tools. The first line of each sample gives its label. `tools/make_fixtures.py`
  generates the nccl-tests samples from a stated α-β model. Replace the samples with your own logs.
* **(verify)**: datasheet peaks, driver-branch tables, image tags, GKE labels. These are dated facts. Examine them
  again.
* **Cost.** T1 is free on Colab or Kaggle, and an hour on a rented NVLink box costs approximately $2–25 (verify).
  When the GKE cluster is idle, it costs one e2-standard-4 system node, a few dollars a day. GKE's free-tier credit
  covers its management fee for one zonal cluster (verify). Its GPU pools cost nothing until a pod asks for a GPU.
  Run `terraform destroy` after each session. [`COMPUTE.md`](../../../COMPUTE.md) gives the prices and tells how
  easy it is to get the hardware.

## The library

| Module | The one idea | Primer |
|---|---|---|
| `gpurt/env.py` | Numba selects the simulator or the GPU one time, at import. Decide first, and do not initialise CUDA. When a GPU is visible but Numba cannot use it, fall back to the simulator and say why. | §1 |
| `gpurt/kernels/elementwise.py` | A kernel is a loop body. Bounds checks. Grid-stride loops. | §2 |
| `gpurt/kernels/reduction.py` | Shared memory and barriers. Two-pass (deterministic) against atomic. | §2, §3 |
| `gpurt/kernels/matmul.py` | tiling: `tile`× fewer global loads through shared memory | §3 |
| `gpurt/kernels/transpose.py` | coalescing (4 against 32 sectors per warp request) and bank-conflict padding | §3 |
| `gpurt/kernels/softmax.py` | fusion: online softmax moves half the bytes of four separate kernels | §3 |
| `gpurt/kernels/trace.py` | Record every access that a simulated warp makes. From these accesses, get the sectors per request, from the kernel itself. | §3 |
| `gpurt/kernels/traffic.py` | the bytes/FLOPs that each kernel must move, datasheet peaks (verify) | §3 |
| `gpurt/kernels/bench.py` | Measure the kernel, not the infrastructure code: device data, CUDA events, warm-up (T1). | §2–§4 |
| `gpurt/kernels/triton_kernels.py` | the same ideas in Triton, block-level programming (optional, T1) | §1, §3 |
| `gpurt/launch.py` | launch-bound steps and CUDA Graphs: model (T0) and measurement (T1, torch) | §4 |
| `gpurt/dist/busbw.py` | algbw against busbw, and the buffer size rule (16-byte per-rank chunks), exactly as nccl-tests v2.20.0 defines them | §5 |
| `gpurt/dist/alphabeta.py` | `t = α + S/B`, the half-bandwidth size, ring costs | §5 |
| `gpurt/dist/semantics.py` | What each collective computes. The ring schedule (with the step numbers of primer §5.2), and a NumPy executor that does a check of any schedule. | §5 |
| `gpurt/dist/sweep.py` | one benchmark loop for every backend: size, check, warm up, time, average, report | §5 |
| `gpurt/dist/pipes.py` | a real ring all-reduce across OS processes (T0, no torch) | §5 |
| `gpurt/dist/bench.py` | torch.distributed: gloo on CPU and NCCL on GPUs, with spawn or `torchrun` | §5 |
| `gpurt/nccltests.py` | Parse `*_perf` output (v2.20.0 and older layouts, per-iteration columns). Calculate its numbers again, and fit it. | §5 |
| `gpurt/container.py` | Device nodes and injected driver files (toolkit against GKE). Versions give compat verdicts (from the driver table of the core). | §1, §6 |
| `gpurt/dcgm.py` | From DCGM text: SM active against GPU util, clock events, XID owners and alert rules. Also, which rules can fire with the fields of your exporter. | §8 |

The same kernel source serves both tiers. The tests run it in the simulator. `tools/check_ptx.py` compiles every
kernel to PTX for `sm_75` and `sm_89`, and it rejects float64 arithmetic. `make ptx-check` runs it.
`tests/test_ptx.py` also runs it when numba-cuda is available, and skips without numba-cuda. No GPU is necessary for
that, but the repository leaves the kernel runs on a GPU to you.

One detail of the simulator is important. Numba's simulator creates the shared array of a block only at its first
use, and without a lock. This can give two threads different arrays, but only in rare cases. When `gpurt.kernels`
runs in simulator mode, it serialises that allocation.

MIT licensed.
