# cuda-nccl-core

The GPU software substrate, **simulated on a CPU**. Seven small numpy modules show how warps turn
addresses into 32-byte sectors and bank conflicts, what limits occupancy, why tiling, fusion and CUDA Graphs
win, how ring, tree, two-shot and in-switch collectives move bytes (with their α-β costs and nccl-tests'
busbw), whether a CUDA binary will run under a given driver and GPU, how MIG, MPS and time-slicing share a
GPU, and why "GPU util" misleads. Five fill-in notebooks. **Tier T0**: no GPU, no network, no Docker.

This is the minimal core of the [`cuda-and-nccl`](../README.md) topic, and every formula and worked number in its
[PRIMER](../PRIMER.md) is computed here (product facts are dated and marked *verify*). The detailed lab, [`cuda-nccl-lab`](../cuda-nccl-lab/), runs the same
ideas on real hardware: Numba kernels, torch.distributed and NCCL, nccl-tests, containers and DCGM on GKE.

## Quick start

```bash
cd cuda-nccl-core
python3 -m pip install -r requirements.txt   # numpy + the notebook/test tools
python3 -m pytest -q                          # 128 tests, well under a second
python3 -m jupyterlab notebooks               # do the exercises
```

The library needs only numpy:

```python
import numpy as np
from gpusim import collectives, compat, simt

simt.coalescing(simt.warp_addresses(stride=32)).sectors        # 32: a column walk, 12.5% of bytes used
r = collectives.all_reduce([np.arange(8) * k for k in range(4)], algo="ring")
print(r.trace.table())                                          # 6 steps: reduce-scatter, then all-gather
collectives.busbw("all_reduce", 2**30, 4.2e-3, p=8) / 1e9       # 447 GB/s, as nccl-tests reports it
print(compat.check("12.4", "535.183.01", gpu="H100", targets="8.0+PTX"))  # fails: error 222, and why
```

## The whole library (seven files)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `gpusim/simt.py` | ~120 | a warp is the unit: sectors per request (coalescing), bank conflicts, divergence cost |
| `gpusim/occupancy.py` | ~130 | resident warps per SM and what limits them (the `cuda_occupancy.h` rules); waves; Little's law |
| `gpusim/tiling.py` | ~120 | bytes and launches: tiled GEMM traffic (with a simulated tiled kernel), fused and online softmax, CUDA Graphs vs eager |
| `gpusim/collectives.py` | ~420 | ring, tree, one-/two-shot and in-switch all-reduce, reduce-scatter, all-gather, broadcast and all-to-all on simulated ranks, with step traces; α-β costs; algbw/busbw; TP and EP message sizes; finding the call that hangs |
| `gpusim/compat.py` | ~270 | driver ↔ CUDA runtime ↔ compute capability: SASS vs PTX, minor-version and forward compatibility (with the kernel-driver branches each `cuda-compat` supports), the error each failure produces, what a container gets from the host |
| `gpusim/sharing.py` | ~150 | MIG profile placement (a packer and a first-fit that fragments), time-slicing latency, MPS vs MIG vs turns |
| `gpusim/health.py` | ~130 | GPU util vs SM active, clock-event (throttle) bits, XID triage by owner, alert severities |

About 900 lines of code; the rest is docstrings and comments. Each module opens with the one idea it teaches. Version
tables, per-SM limits and MIG profiles are dated September 2026 and marked *verify* in the code and in the
primer's Verify list. Every output is **simulated**: a model of documented NVIDIA behaviour, not a measurement.

## The notebooks

Each has a `**Tier:**` line, *The one-minute version*, worked examples, exercises with `# YOUR CODE HERE`
followed by a check cell that prints ✅ when you are right, and *In a design review* drills at the end.
Solutions are in `solutions/`.

1. **`01_simt_warps_and_memory`**: count sectors, predict coalescing, derive the gcd(s, 32) bank-conflict rule,
   pick the padding that fixes a transpose, and measure SIMT efficiency of ragged loops.
2. **`02_tiling_fusion_and_occupancy`**: the tiled-GEMM traffic formula, the register limit by hand, choosing a
   GEMM tile for an L4, online softmax, and when a decode step is launch-bound (CUDA Graphs).
3. **`03_collectives_from_scratch`**: write ring reduce-scatter and all-gather yourself (the check replays your
   message schedule), the α-β time and
   crossover, busbw like nccl-tests, the tensor-parallel decode all-reduce you would ship, and finding a hang.
4. **`04_compatibility_and_containers`**: SASS/PTX rules, predicting error codes, choosing a fleet's CUDA
   version, and five container failure stories.
5. **`05_sharing_and_health`**: MIG layouts, time-slicing latency, choosing a sharing mode, GPU util vs SM active,
   and triaging a night of XIDs.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs
this package.

## When you outgrow this

Go to [`cuda-nccl-lab`](../cuda-nccl-lab/) when you want real measurements: the same kernels in Numba (the CUDA
simulator at T0, a GPU at T1), torch.distributed collectives with gloo and NCCL, nccl-tests output parsed into
algbw/busbw and an α-β fit (T2), what a container actually sees of its GPU, and MIG, time-sharing and DCGM on
GKE (T3). Costs and where to run each tier are in [COMPUTE.md](../../../COMPUTE.md). MIT licensed.
