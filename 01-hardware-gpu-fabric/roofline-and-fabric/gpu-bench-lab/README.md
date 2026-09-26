# gpu-bench-lab — measure the machine you have

The detailed lab for **roofline-and-fabric**. The primer ([`../PRIMER.md`](../PRIMER.md)) derives the
roofline, the memory hierarchy, the α-β model of links and the cold-start budget from first
principles; the minimal core ([`../roofline-core/`](../roofline-core/)) turns them into calculators.
This lab **measures** them: package `gpubench` benchmarks GEMM throughput, memory bandwidth,
host↔device and GPU↔GPU transfers and weights loading, then builds the roofline those numbers imply
— with the same code on a laptop CPU (numpy) and on a CUDA GPU (PyTorch).

Measuring your own CPU's roofline is not a toy version of the GPU exercise: it has the same two roofs,
the same ridge point, the same cache ladder and the same α-β copies, only 10–100× lower. Every concept
below is learnable with no GPU at all; the GPU steps are optional and progressive.

## Tiers

| Tier | Where | What runs | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | numpy backend: GEMM by size and dtype, STREAM with 1…N threads, fusion, the cache ladder, a cache-cold `memcpy` α-β fit, safetensors loading cold and warm; `nvidia-smi` parsing on bundled sample output | $0 |
| **T1** | one GPU: Colab/Kaggle T4, an L4, any rented GPU | torch backend: fp32/TF32/fp16/bf16 (+FP8 on sm_89+) GEMMs, STREAM on HBM/GDDR, the GPU cache ladder, pinned vs pageable host↔device sweeps, disk → pinned → GPU streaming | free – ~$0.7/hr |
| **T2** | ≥ 2 GPUs: Kaggle 2×T4 (free, PCIe), an NVLink pod, GCP `a2-highgpu-2g` | the P2P bandwidth matrix (uni- and bidirectional), α-β of the link, measured vs topology-predicted | ~$0–25 per session |
| **T3** | GCP via Terraform ([`deploy/gcp/`](deploy/gcp/)) | the whole suite on a fresh Spot L4 VM; JSON + Markdown report uploaded to a bucket; auto-stop | ~$0.1–0.3/hr on Spot (verify) |

The notebooks detect what they have: no GPU means the T0 path runs and the cell prints what to run
on real hardware. Prices and where to get GPUs: [`../../../COMPUTE.md`](../../../COMPUTE.md).

## Quick start (T0)

```bash
cd 01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                      # ~60 tests, a few seconds, no GPU
python3 -m gpubench info                  # what is this machine?
python3 -m gpubench run --out results     # the suite: results/gpubench-<host>-<backend>-<time>.{json,md}
python3 -m jupyterlab notebooks           # the four fill-in notebooks (answers in solutions/)
```

On a GPU the same commands pick the torch backend automatically (`--backend numpy|torch` to force one).
Colab and Kaggle already have PyTorch; elsewhere `pip install -e '.[gpu]'`, or use Docker:
[`deploy/any-gpu/`](deploy/any-gpu/). The notebooks' first cell also works on Colab: it clones the repo,
changes into this directory and installs the lab.

## What a number from this lab means

| Label | Meaning | Where |
|---|---|---|
| *measured* | produced by this run on this machine; every table the suite and the notebooks print | `Measurement` records, reports |
| *model* / *theoretical* | computed from a formula: P2P from topology, PCIe line rate, cold-start budgets | `p2p.predict`, `specs.pcie_gbs`, `loading.load_time` |
| *spec* | a vendor datasheet's **dense** peak, dated 2026-09 — verify before relying on it | `gpubench/specs.py` |
| *sample output (illustrative)* | bundled `nvidia-smi` output in the documented format, for parsing practice — not a measurement | `gpubench/fixtures/` |

Byte counts follow one convention per kind of operation, pinned by tests (primer §2):

* **GEMM** `2mnk` FLOPs over the compulsory traffic `(mk + kn + mn)·b` — the roofline's intensity.
* **STREAM** kernels count each named array once, no write-allocate read. Where an implementation makes
  more passes than STREAM assumes (numpy's triad needs two), the report shows both *moved* and
  *STREAM convention* rates.
* **Transfers** (host↔device, P2P, disk→RAM) count the bytes delivered, once.

Every timing reports **best** (what the machine can do) and **median** (what you typically see) over
several samples, after warm-up; GPU work is timed with CUDA events or with every device synchronised.

## The library

| Module | The one idea | Primer |
|---|---|---|
| `accounting.py` | FLOPs and bytes per operation — the numerator of every rate | §2 |
| `timing.py` | warm up, amortise, report best and median; the α-β fit `t = α + n/β` | §5 |
| `measure.py` | a measurement = counted cost ÷ measured time, recorded with both | — |
| `backends/numpy_backend.py` | the CPU has the same two roofs; threads fill the memory bus | §1, §4 |
| `backends/torch_backend.py` | async GPUs, precision modes (TF32), guarded FP8, pinned memory, P2P streams | §1, §5 |
| `gemm.py` | size, shape and dtype decide how close a GEMM gets to the flat roof | §2, §3 |
| `membw.py` | STREAM, Little's law, fusion, the cache ladder | §4 |
| `transfer.py` | pinned vs pageable, PCIe arithmetic, α-β of a copy | §5 |
| `p2p.py` | measure GPU↔GPU, predict it from topology, price a TP all-reduce | §5 |
| `topo.py` | read `nvidia-smi topo -m`: TP groups, NIC per GPU, NUMA affinity | §5 |
| `inventory.py` | read `nvidia-smi --query-gpu`; flag degraded links, power caps, busy GPUs | §1 |
| `loading.py` | safetensors from scratch; cold vs warm reads; pipelined cold-start model | §6 |
| `roofline.py` | build the roofline from measurements; ASCII log-log chart | §2 |
| `specs.py` | a small dated spec table (dense peaks), a CPU peak from its ISA, link rates | §1, §9 |
| `report.py`, `suite.py`, `cli.py` | one JSON + Markdown report per run, so machines line up | — |

## The notebooks

Each has worked examples, exercises with `# YOUR CODE HERE` and a check that prints ✅, and ends with
"In a design review" (a two-minute explanation and drill questions). Answers are in `solutions/`.

1. **`01_measure_your_roofline`** (T0 → T1) — count a GEMM, time it honestly, sweep sizes and dtypes,
   build your roofline, compare with the spec, and compute the decode batch at which an H100 turns
   compute-bound (≈ 320 at d = 8192).
2. **`02_memory_bandwidth_and_transfers`** (T0 → T1) — STREAM's rules, threads and Little's law,
   write-allocate, fusion, the cache ladder, α-β of a copy, pinned vs pageable over PCIe.
3. **`03_multi_gpu_topology_and_p2p`** (T0 → T2) — read the inventory and topology, choose TP groups,
   NICs and cores, predict then measure P2P, and price the TP all-reduce per token.
4. **`04_weights_loading_and_cold_start`** (T0 → T1/T3) — safetensors from scratch, cold vs warm reads,
   Little's law for storage, pipelined loading to the GPU, the cold-start budget.

## Command line

```bash
python3 -m gpubench info                          # CPU + GPUs + backend
python3 -m gpubench run [--backend torch] [--full] [--suite gemm,stream,transfer,p2p,load,inventory] [--out DIR]
python3 -m gpubench run --tiny                    # plumbing check for CI (sizes too small to mean anything)
python3 -m gpubench topo [FILE] [--tp 4]          # analyse `nvidia-smi topo -m` (runs it if FILE is omitted)
python3 -m gpubench inventory [FILE]              # parse `nvidia-smi --query-gpu=... --format=csv` + health findings
python3 -m gpubench show results/<run>.json       # re-render a saved report
```

## Where to run it

| | Colab / Kaggle | Any GPU box (RunPod, Vast, Lambda, your own) | GCP (T3) |
|---|---|---|---|
| How | open the notebook from the layer README's Colab badge; T4 runtime; Kaggle for 2×T4 | `pip install -e '.[gpu]'` or [`deploy/any-gpu/run.sh`](deploy/any-gpu/) (Docker + NVIDIA Container Toolkit) | [`deploy/gcp/`](deploy/gcp/): Terraform, one Spot `g2-standard-4` (L4), report to GCS |
| Good for | T1 for free; P2P over PCIe | NVLink P2P on SXM machines; FP8 on H100 | a clean, repeatable cloud baseline; the cloud disk's cold reads |
| Clean up | nothing | stop or terminate the pod/VM | `terraform destroy` (or `bench-on-gcp.sh`, which destroys for you) |

## Regenerating notebooks, and the tests

`notebooks/` and `solutions/` are generated from the percent-format sources in `notebooks_src/`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (CPU, offline)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above plus pytest
```

Tests (`tests/`) run offline on the numpy backend and bundled fixtures; they pin every FLOP and byte
formula to hand-computed values. The torch backend is imported lazily and is not needed to install,
test or run T0.

## Verify list (as of 2026-09)

* GPU spec figures in `gpubench/specs.py` (dense peaks, bandwidths, TDPs) — vendor datasheets.
* PyTorch APIs used on GPUs: `torch.set_float32_matmul_precision`, `torch._scaled_mm` (private; its
  signature has changed between releases), `torch.cuda.can_device_access_peer`.
* GCP: Deep Learning VM image family names, `install-nvidia-driver` metadata, disk types per machine
  series, L4/A100 Spot availability per zone — see [`deploy/gcp/README.md`](deploy/gcp/README.md).
* Docker base image tag `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` and its driver requirement.

MIT licensed.
