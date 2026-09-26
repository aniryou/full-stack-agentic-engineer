# cuda-nccl-lab

**The GPU software substrate, hands-on.** Real CUDA kernels written in Numba that run in Numba's CUDA
simulator on any CPU and, unchanged, on a GPU; collectives measured the way nccl-tests measures them, over
a ring of OS pipes, gloo or NCCL; the α-β fit that turns a sweep into two numbers; and the plumbing that
decides whether a container can use its GPU at all — device nodes, injected driver files, version
gates — plus DCGM metrics read the right way. Package `gpurt`, six notebooks, deploy assets for any GPU
box, GKE and Terraform.

This is the **detailed** implementation for [`cuda-and-nccl`](../README.md); the minimal numpy simulators
are in [`../cuda-nccl-core`](../cuda-nccl-core/), and the concepts in [the primer](../PRIMER.md). Every
notebook runs on a laptop (T0); a GPU, two GPUs or a GKE cluster make the same notebooks measure real hardware.

## Tiers in this lab

| Tier | Where | What it runs |
|---|---|---|
| **T0** | laptop, Colab CPU, CI ($0) | kernels in the CUDA simulator (correctness, thread indexing, shared memory, barriers, and per-warp memory requests traced from the kernel itself); collectives over a real multi-process ring (`pipes`) or gloo; nccl-tests, probe and DCGM samples; every model prediction labelled as such |
| **T1** | one GPU (Colab/Kaggle T4, L4, rented 24 GB GPU) | kernel timings with CUDA events, effective bandwidth vs peak, launch overhead, CUDA Graphs vs eager (torch), the live container probe with the driver API |
| **T2** | 2+ GPUs (Kaggle 2×T4 free, PCIe only; a rented NVLink box) | NCCL sweeps via `torchrun`, nccl-tests (pinned v2.20.0) against the same NCCL, busbw vs the link; on a GPU VM: dcgm-exporter in Docker, MIG and MPS by hand |
| **T3** | GKE via Terraform (optional) | smoke + probe Job, CUDA sample, 2-GPU nccl-tests Job on `g2-standard-24`, time-sharing and MIG pools, DCGM metrics in Managed Prometheus, alert rules as `ClusterRules` routed by its Alertmanager |

Prices and obtainability: [COMPUTE.md](../../../COMPUTE.md).

## Quick start (T0)

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
python3 -m pip install -r requirements.txt      # numpy, numba, pyyaml + notebook/test tooling
python3 -m pytest -q                            # ~125 tests, ~30 s, simulator only
python3 -m gpurt.env                            # tier, GPUs, numba mode
python3 -m gpurt.container                      # how this process sees a GPU (on a laptop: it doesn't — and why)
python3 -m gpurt.dist.bench --backend pipes --nranks 2 -e 4M   # a real ring all-reduce over pipes
python3 -m gpurt.nccltests gpurt/fixtures/nccl_all_reduce_8gpu_sample.txt --op all_reduce
python3 -m jupyterlab notebooks
```

On a GPU box (T1/T2) add `pip install -e ".[gpu]"` (NVIDIA's `numba-cuda`) and a CUDA build of PyTorch,
then `python -m gpurt.kernels.bench` and `torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl`.
Recipes for Colab, Kaggle 2×T4, RunPod/Vast/Lambda and Docker: [`deploy/any-gpu`](deploy/any-gpu/README.md).
GKE: [`deploy/gcp`](deploy/gcp/README.md) then [`deploy/gke`](deploy/gke/README.md).

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
running the kernels on one is left to you.
One simulator detail worth knowing: Numba's simulator creates a block's shared array lazily without a lock,
which can very occasionally hand two threads different arrays; `gpurt.kernels` serialises that allocation
when it runs in simulator mode.

## The notebooks

Each has worked examples, 4–6 exercises with a ✅ check after each, and an *In a design review* section
with drill questions. Exercises are in `notebooks/`, worked answers in `solutions/`.

1. **`01_kernels_in_the_simulator`** (T0) — indexing, bounds checks, grid-stride loops, a shared-memory
   reduction you write, coalescing and bank conflicts measured from the transpose kernels, tiling's reuse.
2. **`02_memory_bound_kernels_on_a_real_gpu`** (T1; T0 model path) — effective bandwidth, the latency
   floor, transpose and fusion on real hardware, atomics and determinism, launch overhead and CUDA Graphs.
3. **`03_collectives_with_torch_distributed`** (T0 gloo/pipes; T2 NCCL) — semantics, all-reduce =
   reduce-scatter + all-gather, the ring schedule you write and run, the busbw factor it implies, TP
   message sizes.
4. **`04_busbw_and_the_alpha_beta_fit`** (T0 on samples; your T1/T2 logs) — recompute nccl-tests, fit α-β,
   S½ two ways, what TP costs a decode step, whether the plateau is at the link.
5. **`05_how_a_container_sees_a_gpu`** (T0 live + samples; T1/T3 logs) — mountinfo, the toolkit vs GKE's
   device plugin, the driver/runtime and kernel-image gates, error numbers.
6. **`06_gpu_sharing_and_dcgm_on_gke`** (T0; T1/T2 on a GPU VM; T3) — node pools → advertised GPUs →
   selectors, the utilisation paradox, clock-event bits in PromQL, routing alerts by owner, and which
   rules an exporter's fields let fire.

## Numbers in this lab

* **Measured** — anything printed by a notebook's T1/T2 path or by the pipes/gloo sweeps: real timings of
  the backend named next to them.
* **Simulated / model prediction** — the T0 paths: byte counts, α-β and launch models with their
  assumptions printed.
* **Illustrative samples** — `gpurt/fixtures/*`: nccl-tests, dcgm-exporter and probe outputs in the tools'
  documented formats, each labelled in its first line. The nccl-tests samples are generated by
  `tools/make_fixtures.py` from a stated α-β model. Replace them with your own logs.
* **(verify)** — datasheet peaks, driver-branch tables, image tags, GKE labels: dated facts to re-check.

## Regenerating and checking

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

MIT licensed.
