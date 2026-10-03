# gpu-bench-lab — measure the machine you have

After this lab, you can measure these values of a machine:

- Its roofline.
- Its memory bandwidth.
- Its transfers between host and device, and between GPUs.
- How fast it loads weights.

Then you can compare them with the predictions of the spec sheet and the roofline, and you can explain the gap.

This is the detailed lab for **roofline-and-fabric**. The primer ([`../PRIMER.md`](../PRIMER.md)) derives the
roofline, the memory hierarchy, the α-β model of links and the cold-start budget from first principles. The minimal
core ([`../roofline-core/`](../roofline-core/)) makes calculators from them. The package `gpubench` **measures**
them. It uses the same code on a laptop CPU (numpy) and on a CUDA GPU (PyTorch).

When you measure the roofline of your own CPU, you do not do a toy version of the GPU exercise. The CPU has the same
two roofs, the same ridge point, the same cache ladder and the same α-β copies. Only the values are 10–100× lower.
You can learn every concept on this page with no GPU. The GPU steps are optional, and each one adds to the step
before it.

## Start here

1. Install the lab and run the tests, as "Run it (T0)" shows. You do not need a GPU.
2. Run `python3 -m gpubench info`. It tells you what this machine is. Then run
   `python3 -m gpubench run --out results`. It writes a JSON + Markdown report of what the machine can do.
3. Open [`notebooks/01_measure_your_roofline.ipynb`](notebooks/01_measure_your_roofline.ipynb). It names the
   roofline-core notebook that *predicted* the values that it measures.

## What you get: tiers

| Tier | Where | What runs | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | The numpy backend measures GEMM by size and dtype, STREAM with 1…N threads, fusion, the cache ladder and a cache-cold `memcpy` α-β fit. It also loads safetensors cold and warm. The lab parses bundled `nvidia-smi` sample output. | $0 |
| **T1** | one GPU: Colab/Kaggle T4, an L4, any rented GPU | The torch backend measures fp32/TF32/fp16/bf16 (+FP8 on sm_89+) GEMMs, STREAM on HBM/GDDR and the GPU cache ladder. It does sweeps of copies between host and device, pinned against pageable. It also measures streaming from disk to pinned memory to the GPU. | free to ~$0.7/hr |
| **T2** | ≥ 2 GPUs: Kaggle 2×T4 (free, PCIe), an NVLink pod, GCP `a2-highgpu-2g` | The P2P bandwidth matrix (uni- and bidirectional) and the α-β of the link. The lab compares the measured values with the values that it predicts from the topology. | ~$0–25 per session |
| **T3** | GCP via Terraform ([`deploy/gcp/`](deploy/gcp/)) | The full suite runs on a new Spot L4 VM. The VM uploads a JSON + Markdown report to a bucket. Auto-stop is on. | ~$0.1–0.3/hr on Spot (verify) |

The notebooks find the hardware that they have. If there is no GPU, the T0 path runs, and the cell prints what to
run on real hardware. For prices and where to get GPUs, see [`COMPUTE.md`](../../../COMPUTE.md).

## Run it (T0)

```bash
cd 01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                      # 100 tests (97 pass, 3 skip), ~35 s, no GPU
python3 -m gpubench info                  # what is this machine?
python3 -m gpubench run --out results     # the suite: results/gpubench-<host>-<backend>-<time>.{json,md}
python3 -m jupyterlab notebooks           # the four fill-in notebooks (answers in solutions/)
```

On a GPU, the same commands select the torch backend automatically. To force one backend, use
`--backend numpy|torch`. Colab and Kaggle already have PyTorch. On other machines, run `pip install -e '.[gpu]'`, or
use Docker: [`deploy/any-gpu/`](deploy/any-gpu/). The first cell of the notebooks also works on Colab. It clones the
repository, changes into this directory and installs the lab.

## What a number from this lab means

| Label | Meaning | Where |
|---|---|---|
| *measured* | This run made the number on this machine. Every table that the suite and the notebooks print has measured numbers. | `Measurement` records, reports |
| *model* / *theoretical* | The lab calculates the number from a formula. The formulas give P2P from topology, the PCIe line rate and cold-start budgets. A P2P number has the label `direct`, `staged` or `upper-bound`, from what it assumes about peer access. | `p2p.predict`, `specs.pcie_gbs`, `loading.load_time` |
| *assumed* | It is an input that the lab cannot measure here, and the lab names it as an assumption. Examples are a cold-start stage time, and a PCIe link when `nvidia-smi` is absent. | notebook 04, `transfer.host_link` |
| *spec* | It is the **dense** peak from a vendor datasheet, with the date 2026-09. Make sure that it is correct before you depend on it. | `gpubench/specs.py` |
| *sample output (illustrative)* | It is `nvidia-smi` output that comes with the lab, in the documented format, for practice with the parsers. It is not a measurement. | `gpubench/fixtures/` |

The byte counts obey one convention for each kind of operation, and tests pin each convention (primer §2):

* **GEMM** counts `2mnk` FLOPs over the compulsory traffic `(mk + kn + mn)·b`. This ratio is the intensity of the
  roofline.
* **STREAM** kernels count each named array one time, with no write-allocate read. If an implementation makes more
  passes than STREAM assumes (numpy's triad needs two), the report shows both *moved* and *STREAM convention* rates.
* **Transfers** (between host and device, P2P, from disk to RAM) count the delivered bytes one time.

Each measurement of time reports the **best** value (what the machine can do) and the **median** value (what you
usually see). It takes them over several samples, after a warm-up. The lab times GPU work with CUDA events, or with
all devices synchronised.

On a GPU, the lab times copies in two ways. The first way is back to back. Then the α of the α-β fit is a per-copy
*issue* cost, because asynchronous copies overlap. The second way is one synchronised copy per sample. Then α is the
*latency* that a dependent step pays.

The run gives a warning in two cases. In the first case, the samples behind a roof scatter (coefficient of variation
above 15%). In the second case, the float32 peak of a CPU comes out below its float64 peak. Then the run prints a
**noisy measurement (shared CPU?)** warning (`roofline.noise_warnings`), and the report starts with it. That roofline
describes the other load on the machine. Thus, run the suite again on an idle machine.

The report shows "cold" disk reads only when the lab can really drop the page cache. This is not possible for a file
on tmpfs. A `--tiny` report says in bold that it is a "plumbing check".

## The library

| Module | The one idea | Primer |
|---|---|---|
| `accounting.py` | FLOPs and bytes per operation: the numerator of every rate | §2 |
| `timing.py` | Warm up, amortise, and report the best and the median. Make the α-β fit `t = α + n/β`. | §5 |
| `measure.py` | A measurement = counted cost ÷ measured time. The record holds both values. | — |
| `backends/numpy_backend.py` | The CPU has the same two roofs. Threads fill the memory bus. | §1, §4 |
| `backends/torch_backend.py` | async GPUs, precision modes (TF32), guarded FP8, pinned memory, P2P streams | §1, §5 |
| `gemm.py` | Size, shape and dtype decide how near a GEMM gets to the flat roof. | §2, §3 |
| `membw.py` | STREAM, Little's law, fusion, the cache ladder | §4 |
| `transfer.py` | Pinned against pageable, PCIe arithmetic, the α-β of a copy. | §5 |
| `p2p.py` | Measure transfers between GPUs. Predict them from topology. Calculate the cost of a TP all-reduce. | §5 |
| `topo.py` | Read `nvidia-smi topo -m`: TP groups, the NIC for each GPU, NUMA affinity. | §5 |
| `inventory.py` | Read `nvidia-smi --query-gpu`. Show degraded links, power caps and busy GPUs. | §1 |
| `loading.py` | Safetensors from scratch, cold against warm reads, the pipelined cold-start model. | §6 |
| `roofline.py` | Build the roofline from measurements. Draw an ASCII log-log chart. | §2 |
| `specs.py` | A small dated spec table (dense peaks), a CPU peak from its ISA, link rates. | §1, §9 |
| `report.py`, `suite.py`, `cli.py` | One JSON + Markdown report for each run, so that you can compare machines. | — |

## The notebooks

Each notebook starts with the name of the roofline-core notebook that *predicted* what it measures. It has worked
examples, and exercises with `# YOUR CODE HERE` and a check that prints ✅. Most of the exercises read or predict the
measurements of this machine. Each notebook ends with "In a design review" (a two-minute explanation and drill
questions). The answers are in `solutions/`.

1. **`01_measure_your_roofline`** (T0 to T1). Count a GEMM, time it honestly, and do a sweep of sizes and dtypes.
   Build your roofline from the measurements, and find what your peak tells you about the hardware. Then predict
   and measure a decode projection across batch sizes. The H100 crossover for d = 8192 comes out as 296 with
   activations on chip (primer §3.4). It is 319 for a standalone GEMM that also moves its activations, and the
   notebook shows why both are correct.
2. **`02_memory_bandwidth_and_transfers`** (T0 to T1). The notebook covers the rules of STREAM and Little's law read
   from your thread-scaling curve. It also covers write-allocate, and fusion with its measured gap. After that, it
   covers the cache ladder and the α-β of a copy (and which α). Then you predict a copy from your fit and measure it. At the end, you
   compare pinned and pageable copies over PCIe.
3. **`03_multi_gpu_topology_and_p2p`** (T0 to T2). Read the inventory and the topology. Select TP groups, NICs and
   cores. Predict P2P, and say what the prediction assumes about peer access. Explain a measured P2P number.
   Calculate the cost of the TP all-reduce with the correct measured α.
4. **`04_weights_loading_and_cold_start`** (T0 to T1/T3). The notebook covers safetensors from scratch, and cold
   against warm reads. It shows if a "cold" number is a disk number, and which measured rate belongs in the
   pipelined model. You select a loader for this disk. Then you make the cold-start budget, with a label on each
   assumption.

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
| How | Open the notebook from its Colab link in the layer README. Use the T4 runtime. Use Kaggle for 2×T4. | Run `pip install -e '.[gpu]'` or [`deploy/any-gpu/run.sh`](deploy/any-gpu/) (Docker + NVIDIA Container Toolkit). | [`deploy/gcp/`](deploy/gcp/): Terraform makes one Spot `g2-standard-4` (L4). The report goes to GCS. |
| Good for | T1 at no cost, P2P over PCIe | NVLink P2P on SXM machines, FP8 on H100 | a clean cloud baseline that you can repeat, cold reads from the cloud disk |
| Clean up | nothing | Stop or terminate the pod/VM. | `terraform destroy` (or `bench-on-gcp.sh`, which destroys the resources for you) |

## Regenerating notebooks, and the tests

A builder generates `notebooks/` and `solutions/` from the percent-format sources in `notebooks_src/`:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (CPU, offline)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above plus pytest
```

The tests (`tests/`) run offline on the numpy backend and the bundled fixtures. They pin every FLOP and byte
formula to hand-calculated values.

The lab imports the torch backend lazily. You do not need it to install, test or run T0.
`tests/test_torch_backend_fake.py` operates the backend with a stand-in `torch` module. Thus the tests pin three
things without a GPU. The first is the accounting of the GPU ops (GEMM by dtype with FP8, STREAM, transfers, P2P).
The others are the TF32 switch and the `torch._scaled_mm` call convention.

Some behaviour still needs real hardware to confirm it: the time measurement with CUDA events, stream overlap and
driver behaviour. The "Verify list" at the end of this page has these items.

## Verify list (as of 2026-09)

* The GPU spec figures in `gpubench/specs.py` (dense peaks, bandwidths, TDPs). The source is the vendor
  datasheets.
* The PyTorch APIs that the lab uses on GPUs: `torch.set_float32_matmul_precision`, `torch._scaled_mm` and
  `torch.cuda.can_device_access_peer`. `torch._scaled_mm` is private, and its signature changed between releases.
  Its use agrees with `native_functions.yaml` on main (examined 2026-09). PyPI's PyTorch is a CUDA 13 build from
  2.11 on, and it needs an R580+ driver. `cu126` builds run on R525+, but they have no Blackwell kernels.
* GCP: Deep Learning VM image family names, `install-nvidia-driver` metadata, disk types for each machine series and
  L4/A100 Spot availability for each zone. See [`deploy/gcp/README.md`](deploy/gcp/README.md).
* The Docker base image tag `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` (it exists on Docker Hub, 2026-09) and
  its driver requirement. Blackwell GPUs need the `cuda13.0` variant (R580+ driver).
* Behaviour that only a GPU can confirm:
  * The time measurement with CUDA events.
  * The overlap of the bidirectional P2P streams.
  * The overlap in the double-buffered pipeline from disk to GPU.
  * That `max_run_duration` does not act on a stopped VM.

The lab has the MIT licence.
