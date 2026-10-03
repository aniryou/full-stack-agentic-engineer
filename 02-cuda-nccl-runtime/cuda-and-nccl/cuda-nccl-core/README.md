# cuda-nccl-core — the GPU software substrate, simulated on a CPU

After these five notebooks, you can predict these things on a laptop, without a GPU:

- How many memory transactions a warp costs.
- What limits occupancy.
- What tiling, fusion and CUDA Graphs save.
- What a collective costs, and what nccl-tests will report.
- If a CUDA build runs under a given driver and GPU.
- How to share a GPU.
- When "GPU util" gives an incorrect picture.

## Start here

1. Run `python3 -m pytest -q` in this folder. The 141 tests run in ~30 s.
2. Paste the tour in the code block that comes next into a `python3` prompt. The tour shows these results:
   - A column walk costs 32 sectors.
   - A ring all-reduce prints its six steps.
   - busbw reads 447 GB/s.
   - A CUDA 12.4 build fails on driver 535 with error 222. The tour also gives the reason.
3. Open [`notebooks/01_simt_warps_and_memory.ipynb`](notebooks/01_simt_warps_and_memory.ipynb). Keep §2–§3 of the
   [primer](../PRIMER.md) open beside it.

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

Everything here is **T0**: a laptop or a Colab CPU, at no cost, with no GPU, no network and no Docker. Each
notebook has these parts:

- A `**Tier:**` line.
- *The one-minute version*.
- Worked examples.
- Exercises (`# YOUR CODE HERE`). After each exercise, a check cell prints ✅ when your answer is correct.
- *In a design review* drills.

The worked answers are in `solutions/`. The times include the primer sections of each notebook. The course plan
gives about 9 hours to all five notebooks.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_simt_warps_and_memory`](notebooks/01_simt_warps_and_memory.ipynb) | Count sectors and predict coalescing. Derive the gcd(s, 32) bank-conflict rule. Select the padding that removes the bank conflicts of a transpose. Measure the SIMT efficiency of ragged loops. | §2, §3 | 1½–2 h | T0 |
| [`02_tiling_fusion_and_occupancy`](notebooks/02_tiling_fusion_and_occupancy.ipynb) | Derive the tiled-GEMM traffic formula. Calculate the register limit by hand. Select a GEMM tile for an L4. Write online softmax. Find when a decode step is launch-bound (CUDA Graphs). | §2–§4 | 1½–2 h | T0 |
| [`03_collectives_from_scratch`](notebooks/03_collectives_from_scratch.ipynb) | Write ring reduce-scatter and all-gather yourself. The check replays your message schedule. Calculate the α-β time and the crossover, and calculate busbw like nccl-tests. Calculate the size of the tensor-parallel decode all-reduce for a production deployment. Find a hang. | §5 | 2–2½ h | T0 |
| [`04_compatibility_and_containers`](notebooks/04_compatibility_and_containers.ipynb) | Apply the SASS/PTX rules and predict error codes. Select the CUDA version of a fleet. Explain five container failure stories. | §1, §6 | 1½–2 h | T0 |
| [`05_sharing_and_health`](notebooks/05_sharing_and_health.ipynb) | Lay out MIG instances. Calculate the time-slicing latency with a model. Select a mode to share a GPU. Read GPU util against SM active. Do the triage of a night of XIDs. | §7, §8 | 1½–2 h | T0 |

## Run it

```bash
cd 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core
python3 -m pip install -r requirements.txt   # numpy + the notebook/test tools
python3 -m pytest -q                          # 141 tests, ~30 s
python3 -m jupyterlab notebooks               # do the exercises
```

On Colab, the first cell of each notebook clones the repo and installs this package. The Colab links are in the
[layer README](../../README.md). The builder generates `notebooks/` and `solutions/` from
`notebooks_src/*.py` (percent format with `### BEGIN SOLUTION` blocks). Edit the sources, then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three commands and the tests.

## How it fits

This is the minimal core of the [`cuda-and-nccl`](../README.md) topic. It computes every formula and worked number
in the [primer](../PRIMER.md) of the topic. The detailed lab, [`cuda-nccl-lab`](../cuda-nccl-lab/), runs the same
ideas for real:

- The same kernels in Numba (the CUDA simulator on a CPU, then any GPU).
- Collectives over OS pipes, gloo and NCCL.
- nccl-tests output (bundled samples, or your own from two or more GPUs). The lab parses it into algbw/busbw and an
  α-β fit.
- What a container really sees of its GPU.
- MIG, time-sharing and DCGM on GKE (optional).

## Going further / caveats

- Every output is **simulated**. It is a model of documented NVIDIA behaviour, not a measurement. The lab measures.
- The version tables, per-SM limits and MIG profiles are dated September 2026. They have the *verify* mark in the
  code and in the [Verify list](../PRIMER.md#verify-list) of the primer.
- [`COMPUTE.md`](../../../COMPUTE.md) tells where to run the GPU parts and what they cost.

## The library

The library has seven files (package `gpusim`) and about 900 lines of code. The rest is docstrings and comments.
Each module starts with the one idea that it teaches:

| File | Lines | What it teaches |
|------|-------|-----------------|
| `gpusim/simt.py` | ~120 | a warp is the unit: sectors per request (coalescing), bank conflicts, divergence cost |
| `gpusim/occupancy.py` | ~130 | resident warps per SM and what limits them (the `cuda_occupancy.h` rules), waves, Little's law |
| `gpusim/tiling.py` | ~120 | bytes and launches: tiled GEMM traffic (with a simulated tiled kernel), fused and online softmax, CUDA Graphs against eager mode |
| `gpusim/collectives.py` | ~420 | ring, tree, one-/two-shot and in-switch all-reduce, reduce-scatter, all-gather, broadcast and all-to-all on simulated ranks, with step traces. The module also calculates α-β costs, algbw/busbw and TP and EP message sizes, and it finds the call that hangs. |
| `gpusim/compat.py` | ~270 | the relation of driver, CUDA runtime and compute capability: SASS against PTX, and minor-version and forward compatibility (with the kernel-driver branches that each `cuda-compat` supports). The module also gives the error that each failure causes, and what a container gets from the host. |
| `gpusim/sharing.py` | ~150 | MIG profile placement (a packer and a first-fit that fragments), time-slicing latency, MPS against MIG against turns |
| `gpusim/health.py` | ~130 | GPU util against SM active, clock-event (throttle) bits, XID triage by owner, alert severities |

MIT licensed.
