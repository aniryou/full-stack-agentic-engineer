# cuda-nccl-lab — the GPU software substrate, hands-on

After this lab you can write a CUDA kernel and check it on a laptop, then time it on a GPU; measure an
all-reduce and read its busbw the way nccl-tests does; turn a sweep into α and β; tell from inside a
container why it does or does not see its GPU; and read DCGM metrics without being fooled by "GPU util".

## Start here

1. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 122 tests pass and 2 skip (they need
   torch or numba-cuda) in about 20 s, simulator only.
2. `python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M` — a real ring all-reduce across two OS
   processes, printed in nccl-tests' layout, in about a second.
3. Open [`notebooks/01_kernels_in_the_simulator.ipynb`](notebooks/01_kernels_in_the_simulator.ipynb): real
   CUDA kernels in Numba's simulator. With any GPU (a free Colab T4 will do),
   [`02_memory_bound_kernels_on_a_real_gpu`](notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) times the
   same source.

## What you get

**Tiers:** T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a
multi-GPU box, rented for an hour (Kaggle's 2 × T4 is a free one); T3 = the Google Cloud deployment,
optional. Every notebook runs at T0; a GPU, two GPUs or a GKE cluster make the same notebooks measure real
hardware.

Each notebook has worked examples, 4–6 exercises with a ✅ check after each, and an *In a design review*
section with drill questions; exercises are in `notebooks/`, worked answers in `solutions/`. Times are for
the T0 path (the course plan budgets about 7 hours for notebooks 01–05).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`01_kernels_in_the_simulator`](notebooks/01_kernels_in_the_simulator.ipynb) | index threads, bounds-check, write grid-stride loops and a shared-memory reduction, measure coalescing and bank conflicts from the transpose kernels, see tiling's reuse | ~1½ h | T0 |
| [`02_memory_bound_kernels_on_a_real_gpu`](notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) | compute effective bandwidth and the latency floor, time transpose and fusion on real hardware, reason about atomics and determinism, launch overhead and CUDA Graphs | ~1½ h | T1; T0 model path |
| [`03_collectives_with_torch_distributed`](notebooks/03_collectives_with_torch_distributed.ipynb) | state collective semantics, show all-reduce = reduce-scatter + all-gather, write and run the ring schedule, derive the busbw factor it implies, size TP messages | ~1½ h | T0 (pipes or gloo); T2 NCCL |
| [`04_busbw_and_the_alpha_beta_fit`](notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) | recompute nccl-tests, fit α-β, find S½ two ways, price TP for a decode step, tell whether the plateau is at the link | ~1½ h | T0 on samples; your T1/T2 logs |
| [`05_how_a_container_sees_a_gpu`](notebooks/05_how_a_container_sees_a_gpu.ipynb) | read mountinfo, tell the toolkit's injection from GKE's device plugin, apply the driver/runtime and kernel-image gates, decode error numbers | ~1½ h | T0 live + samples; your T1/T3 logs |
| [`06_gpu_sharing_and_dcgm_on_gke`](notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) | trace node pools → advertised GPUs → selectors, explain the utilisation paradox, read clock-event bits in PromQL, route alerts by owner, and say which rules an exporter's fields let fire | ~1½ h, +2 h on GKE | T0; T1/T2 on a GPU VM; T3 |
| [`deploy/any-gpu/`](deploy/any-gpu/README.md) | run the T1/T2 paths anywhere: Colab, Kaggle 2 × T4, RunPod/Vast/Lambda, Docker; nccl-tests, dcgm-exporter, MIG and MPS by hand | per session | T1, T2 |
| [`deploy/gcp/`](deploy/gcp/README.md), [`deploy/gke/`](deploy/gke/README.md) | build a zonal GKE cluster with Spot L4 pools from zero, then run the smoke, CUDA sample, 2-GPU nccl-tests, time-sharing and MIG Jobs and the DCGM alert rules | ~2 h per session | T3 |

What each tier adds:

| Tier | Where | What it runs |
|---|---|---|
| **T0** | laptop, Colab CPU, CI ($0) | kernels in the CUDA simulator (correctness, thread indexing, shared memory, barriers, and per-warp memory requests traced from the kernel itself); collectives over a real multi-process ring (`pipes`) or gloo; nccl-tests, probe and DCGM samples; every model prediction labelled as such |
| **T1** | one GPU (Colab/Kaggle T4, L4, rented 24 GB GPU) | kernel timings with CUDA events, effective bandwidth vs peak, launch overhead, CUDA Graphs vs eager (torch), the live container probe with the driver API |
| **T2** | 2+ GPUs (Kaggle 2×T4 free, PCIe only; a rented NVLink box) | NCCL sweeps via `torchrun`, nccl-tests (pinned v2.20.0) against the same NCCL, busbw vs the link; on a GPU VM: dcgm-exporter in Docker, MIG and MPS by hand |
| **T3** | GKE via Terraform (optional) | smoke + probe Job, CUDA sample, 2-GPU nccl-tests Job on `g2-standard-24`, time-sharing and MIG pools, DCGM metrics in Managed Prometheus, alert rules as `ClusterRules` routed by its Alertmanager |

## Run it

**T0, on any laptop:**

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
python3 -m pip install -r requirements.txt      # numpy, numba, pyyaml + notebook/test tooling
python3 -m pytest -q                            # 122 pass, 2 skip without torch / numba-cuda; ~20 s, simulator only
python3 -m gpurt.env                            # tier, GPUs, numba mode
python3 -m gpurt.container                      # how this process sees a GPU (on a laptop: it doesn't — and why)
python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M   # a real ring all-reduce over pipes
python3 -m gpurt.nccltests gpurt/fixtures/nccl_all_reduce_8gpu_sample.txt --op all_reduce
python3 -m jupyterlab notebooks
```

**On a GPU box (T1/T2)**, add `pip install -e ".[gpu]"` (NVIDIA's `numba-cuda`) and a CUDA build of PyTorch,
then `python -m gpurt.kernels.bench` and `torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl`.
Recipes for Colab, Kaggle 2×T4, RunPod/Vast/Lambda and Docker: [`deploy/any-gpu`](deploy/any-gpu/README.md).

**On GKE (T3)**: [`deploy/gcp`](deploy/gcp/README.md) creates the cluster with Terraform, then
[`deploy/gke`](deploy/gke/README.md)'s `run.sh` runs each Job and saves its log under `out/`
(`DRY_RUN=1` prints the commands only).

**Regenerating and checking** (`notebooks/` and `solutions/` are built from `notebooks_src/*.py`):

```bash
python3 tools/build_notebooks.py                        # notebooks_src/*.py -> notebooks/ + solutions/
python3 tools/run_notebooks.py solutions                # solutions must run clean (CPU, offline)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
python3 tools/make_fixtures.py                          # the illustrative nccl-tests samples
python3 tools/check_ptx.py                              # PTX for sm_75/sm_89 (needs numba-cuda; exit 2 = skipped)
kubernetes-validate --strict -k 1.34.0 deploy/gke/0*.yaml
make check                                              # tests + both notebook runs
```

`deploy/gke/06-dcgm-alert-rules.yaml` is generated by `gpurt.dcgm.rules_manifest()`; a test keeps them in
sync. Things marked `VERIFY` in the manifests and Terraform are product details to re-check before use.

## How it fits

This is the **detailed** implementation for [`cuda-and-nccl`](../README.md): the minimal numpy simulators
are in [`../cuda-nccl-core`](../cuda-nccl-core/), and the concepts in [the primer](../PRIMER.md). Each module
below names the primer section it measures; each notebook links its sections at the top.

## Going further / caveats

* **Measured** — anything printed by a notebook's T1/T2 path or by the pipes/gloo sweeps: real timings of
  the backend named next to them.
* **Simulated / model prediction** — the T0 paths: byte counts, α-β and launch models with their
  assumptions printed.
* **Illustrative samples** — `gpurt/fixtures/*`: nccl-tests, dcgm-exporter and probe outputs in the tools'
  documented formats, each labelled in its first line. The nccl-tests samples are generated by
  `tools/make_fixtures.py` from a stated α-β model. Replace them with your own logs.
* **(verify)** — datasheet peaks, driver-branch tables, image tags, GKE labels: dated facts to re-check.
* **Cost.** T1 is free on Colab or Kaggle; an hour on a rented NVLink box is roughly $2–25 (verify). Idle,
  the GKE cluster costs one e2-standard-4 system node, a few dollars a day (its management fee is covered for
  one zonal cluster by GKE's free-tier credit; verify), and its GPU pools cost nothing until a pod asks for a
  GPU; `terraform destroy` after each session. Prices and obtainability: [`COMPUTE.md`](../../../COMPUTE.md).

## The library

| Module | The one idea | Primer |
|---|---|---|
| `gpurt/env.py` | Numba chooses simulator vs GPU once, at import: decide first, without initialising CUDA — and fall back to the simulator, saying why, when a GPU is visible but Numba cannot use it | §1 |
| `gpurt/kernels/elementwise.py` | a kernel is a loop body; bounds checks; grid-stride loops | §2 |
| `gpurt/kernels/reduction.py` | shared memory + barriers; two-pass (deterministic) vs atomic | §2, §3 |
| `gpurt/kernels/matmul.py` | tiling: `tile`× fewer global loads through shared memory | §3 |
| `gpurt/kernels/transpose.py` | coalescing (4 vs 32 sectors per warp request) and bank-conflict padding | §3 |
| `gpurt/kernels/softmax.py` | fusion: online softmax moves half the bytes of four separate kernels | §3 |
| `gpurt/kernels/trace.py` | record every access a simulated warp makes → sectors per request, from the kernel itself | §3 |
| `gpurt/kernels/traffic.py` | the bytes/FLOPs each kernel must move; datasheet peaks (verify) | §3 |
| `gpurt/kernels/bench.py` | measure the kernel, not the plumbing: device data, CUDA events, warm-up (T1) | §2–§4 |
| `gpurt/kernels/triton_kernels.py` | the same ideas in Triton, block-level programming (optional, T1) | §1, §3 |
| `gpurt/launch.py` | launch-bound steps and CUDA Graphs: model (T0) and measurement (T1, torch) | §4 |
| `gpurt/dist/busbw.py` | algbw vs busbw and buffer sizing (16-byte per-rank chunks) exactly as nccl-tests v2.20.0 defines them | §5 |
| `gpurt/dist/alphabeta.py` | `t = α + S/B`, the half-bandwidth size, ring costs | §5 |
| `gpurt/dist/semantics.py` | what each collective computes; the ring schedule (numbered as in primer §5.2) and a NumPy executor that checks any schedule | §5 |
| `gpurt/dist/sweep.py` | one benchmark loop for every backend: size, check, warm up, time, average, report | §5 |
| `gpurt/dist/pipes.py` | a real ring all-reduce across OS processes (T0, no torch) | §5 |
| `gpurt/dist/bench.py` | torch.distributed: gloo on CPU, NCCL on GPUs; spawn or `torchrun` | §5 |
| `gpurt/nccltests.py` | parse `*_perf` output (v2.20.0 and older layouts, per-iteration columns), re-check it, fit it | §5 |
| `gpurt/container.py` | device nodes, injected driver files (toolkit vs GKE), versions → compat verdicts (the core's driver table) | §1, §6 |
| `gpurt/dcgm.py` | DCGM text → SM active vs GPU util, clock events, XID owners, alert rules — and which rules your exporter's fields let fire | §8 |

The same kernel source serves both tiers: the tests run it in the simulator, and `tools/check_ptx.py`
(`make ptx-check`, and `tests/test_ptx.py` when numba-cuda is installed — skipped otherwise) compiles every
kernel to PTX for `sm_75` and `sm_89` and rejects float64 arithmetic; no GPU is needed for that, but
running the kernels on one is left to you. One simulator detail worth knowing: Numba's simulator creates a
block's shared array lazily without a lock, which can very occasionally hand two threads different arrays;
`gpurt.kernels` serialises that allocation when it runs in simulator mode.

MIT licensed.
