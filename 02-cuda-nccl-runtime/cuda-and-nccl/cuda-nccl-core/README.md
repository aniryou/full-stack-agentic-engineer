# cuda-nccl-core — the GPU software substrate, simulated on a CPU

After these five notebooks you can predict, on a laptop and without a GPU, how many memory transactions a
warp costs, what limits occupancy, what tiling, fusion and CUDA Graphs save, what a collective costs and what
nccl-tests will report, whether a CUDA build runs under a given driver and GPU, how to share a GPU, and when
"GPU util" misleads.

## Start here

1. `python3 -m pytest -q` in this folder — 141 tests, ~30 s.
2. Paste the tour below into a `python3` prompt: a column walk costs 32 sectors, a ring all-reduce prints its
   six steps, busbw reads 447 GB/s, and a CUDA 12.4 build fails on driver 535 with error 222, with the reason.
3. Open [`notebooks/01_simt_warps_and_memory.ipynb`](notebooks/01_simt_warps_and_memory.ipynb) with the
   [primer](../PRIMER.md) §2–§3 beside it.

```python
import numpy as np
from gpusim import collectives, compat, simt

simt.coalescing(simt.warp_addresses(stride=32)).sectors        # 32: a column walk, 12.5% of bytes used
r = collectives.all_reduce([np.arange(8) * k for k in range(4)], algo="ring")
print(r.trace.table())                                          # 6 steps: reduce-scatter, then all-gather
collectives.busbw("all_reduce", 2**30, 4.2e-3, p=8) / 1e9       # 447 GB/s, as nccl-tests reports it
print(compat.check("12.4", "535.183.01", gpu="H100", targets="8.0+PTX"))  # fails: error 222, and why
```

## What you get

Everything here is **T0** — laptop or Colab CPU, free: no GPU, no network, no Docker. Each notebook has a
`**Tier:**` line, *The one-minute version*, worked examples, exercises (`# YOUR CODE HERE`) each followed by a
check cell that prints ✅ when you are right, and *In a design review* drills; worked answers are in
`solutions/`. Times include the matching primer sections; the course plan budgets about 9 hours for all five.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_simt_warps_and_memory`](notebooks/01_simt_warps_and_memory.ipynb) | count sectors, predict coalescing, derive the gcd(s, 32) bank-conflict rule, pick the padding that fixes a transpose, and measure the SIMT efficiency of ragged loops | §2, §3 | 1½–2 h | T0 |
| [`02_tiling_fusion_and_occupancy`](notebooks/02_tiling_fusion_and_occupancy.ipynb) | derive the tiled-GEMM traffic formula, work the register limit by hand, choose a GEMM tile for an L4, write online softmax, and tell when a decode step is launch-bound (CUDA Graphs) | §2–§4 | 1½–2 h | T0 |
| [`03_collectives_from_scratch`](notebooks/03_collectives_from_scratch.ipynb) | write ring reduce-scatter and all-gather yourself (the check replays your message schedule), compute the α-β time and crossover and busbw like nccl-tests, size the tensor-parallel decode all-reduce you would ship, and find a hang | §5 | 2–2½ h | T0 |
| [`04_compatibility_and_containers`](notebooks/04_compatibility_and_containers.ipynb) | apply the SASS/PTX rules, predict error codes, choose a fleet's CUDA version, and explain five container failure stories | §1, §6 | 1½–2 h | T0 |
| [`05_sharing_and_health`](notebooks/05_sharing_and_health.ipynb) | lay out MIG instances, model time-slicing latency, choose a sharing mode, read GPU util against SM active, and triage a night of XIDs | §7, §8 | 1½–2 h | T0 |

## Run it

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core
python3 -m pip install -r requirements.txt   # numpy + the notebook/test tools
python3 -m pytest -q                          # 141 tests, ~30 s
python3 -m jupyterlab notebooks               # do the exercises
```

On Colab, each notebook's first cell clones the repo and installs this package; the Colab links are in the
[layer README](../../README.md). `notebooks/` and `solutions/` are generated from `notebooks_src/*.py`
(percent format with `### BEGIN SOLUTION` blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests.

## How it fits

This is the minimal core of the [`cuda-and-nccl`](../README.md) topic: every formula and worked number in its
[primer](../PRIMER.md) is computed here. The detailed lab, [`cuda-nccl-lab`](../cuda-nccl-lab/), runs the same
ideas for real: the same kernels in Numba (the CUDA simulator on a CPU, then any GPU), collectives over OS
pipes, gloo and NCCL, nccl-tests output (bundled samples, or your own from two or more GPUs) parsed into
algbw/busbw and an α-β fit, what a container actually sees of its GPU, and MIG, time-sharing and DCGM on
GKE (optional).

## Going further / caveats

- Every output is **simulated**: a model of documented NVIDIA behaviour, not a measurement. The lab measures.
- Version tables, per-SM limits and MIG profiles are dated September 2026 and marked *verify* in the code and in
  the primer's [Verify list](../PRIMER.md#verify-list).
- Where to run the GPU parts and what they cost: [`COMPUTE.md`](../../../COMPUTE.md).

## The library

Seven files (package `gpusim`), about 900 lines of code; the rest is docstrings and comments. Each module
opens with the one idea it teaches:

| File | Lines | What it teaches |
|------|-------|-----------------|
| `gpusim/simt.py` | ~120 | a warp is the unit: sectors per request (coalescing), bank conflicts, divergence cost |
| `gpusim/occupancy.py` | ~130 | resident warps per SM and what limits them (the `cuda_occupancy.h` rules); waves; Little's law |
| `gpusim/tiling.py` | ~120 | bytes and launches: tiled GEMM traffic (with a simulated tiled kernel), fused and online softmax, CUDA Graphs vs eager |
| `gpusim/collectives.py` | ~420 | ring, tree, one-/two-shot and in-switch all-reduce, reduce-scatter, all-gather, broadcast and all-to-all on simulated ranks, with step traces; α-β costs; algbw/busbw; TP and EP message sizes; finding the call that hangs |
| `gpusim/compat.py` | ~270 | driver ↔ CUDA runtime ↔ compute capability: SASS vs PTX, minor-version and forward compatibility (with the kernel-driver branches each `cuda-compat` supports), the error each failure produces, what a container gets from the host |
| `gpusim/sharing.py` | ~150 | MIG profile placement (a packer and a first-fit that fragments), time-slicing latency, MPS vs MIG vs turns |
| `gpusim/health.py` | ~130 | GPU util vs SM active, clock-event (throttle) bits, XID triage by owner, alert severities |

MIT licensed.
