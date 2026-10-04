# roofline-and-fabric — read a GPU spec sheet and predict what a model will do on it

After this topic, you can look at a GPU, a model and a fabric, and you can say these things:

- Which resource is the bound for an LLM step.
- What a collective costs on each link.
- How long a replica takes to load.
- How frequently a large job fails.
- What a token costs.

You can defend these numbers in a design review.

## Start here

1. Read "The one-minute version" in [PRIMER.md](PRIMER.md). Then read §1–§2, about spec-sheet literacy and the
   roofline.
2. Run `cd roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 66 tests take
   ~30 s. Then open [`01_spec_sheets_and_the_roofline`](roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb).
   On Colab, it runs as it is. Locally, `pip install -r requirements-notebooks.txt` adds JupyterLab.
3. Measure the real hardware with [`gpu-bench-lab/notebooks/01_measure_your_roofline`](gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb).
   On a laptop, the notebook measures the roofline of your CPU. On any GPU, it measures the roofline of the GPU.
   A free Colab T4 is sufficient.

The fastest result needs nothing installed:

```python
from roofline import llm, specs    # run from roofline-core/
step = llm.decode(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), batch=1, context=1024)
print(step.bound, f"{step.time * 1e3:.2f} ms")          # memory 4.52 ms
```

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.*

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain the concepts of sections 1–10. The first five are spec sheets, the roofline, LLM inference on the roofline, the memory hierarchy and fabrics. The other five are storage and cold start, reliability, cost, the September 2026 accelerator landscape and how to get hardware. Then the primer gives a walkthrough for a design review, and drills. Every computed number comes from the core. | read it with the core | — |
| [`roofline-core/`](roofline-core/README.md) | **Predict** step times, collective costs, cold starts, failure rates and $/M tokens with the minimal implementation. It is the package `roofline`, with seven standard-library modules (`specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability`, `cost`) and four fill-in notebooks. | ~8 h with the primer | T0 |
| [`gpu-bench-lab/`](gpu-bench-lab/README.md) | **Measure** the machine that you have with the detailed lab, the package `gpubench`. It has numpy (CPU) and torch (CUDA) backends. They measure GEMM throughput, memory bandwidth, transfers between host and device and between GPUs, and the time to load the weights. The lab also has `nvidia-smi` topology and inventory parsers, and Docker and GCP Terraform deploys. | ~5 h | T0 to T3 |

The times are approximate. They come from the curriculum of the repository ([`CURRICULUM.md`](../../CURRICULUM.md),
modules 01.1–01.5).

### Work it in this order

In each step, a section of the primer goes with a core notebook (predict) and a lab notebook (measure).

| Step | Read | Predict (core, T0) | Measure (lab) |
|---|---|---|---|
| 1 | [PRIMER §1–2](PRIMER.md#1-spec-sheet-literacy) spec sheets, the roofline | `01_spec_sheets_and_the_roofline` | `01_measure_your_roofline`: T0 on your CPU, T1 on a GPU |
| 2 | [PRIMER §3–4](PRIMER.md#3-llm-inference-on-the-roofline) LLM steps per step and per kernel, KV reads, quantization, MoE, tiling and fusion | `02_llm_inference_on_the_roofline` | `02_memory_bandwidth_and_transfers`: T0 / T1 |
| 3 | [PRIMER §5](PRIMER.md#5-fabrics-quantitatively) α-β, collectives, TP cost, rails, topology | `03_fabrics_and_collective_cost` | `03_multi_gpu_topology_and_p2p`: T2 (at T0, the notebook uses sample topology output) |
| 4 | [PRIMER §6–8](PRIMER.md#6-storage-and-cold-start) cold start, reliability, $/M tokens | `04_loading_reliability_and_cost` | `04_weights_loading_and_cold_start`: T0 / T1 |
| 5 | [PRIMER §9–10](PRIMER.md#9-the-accelerator-landscape-september-2026-snapshot) the landscape, how to get hardware | — | lab `deploy/any-gpu/` (T1/T2) or `deploy/gcp/terraform/` (T3) |

Finish with the [walkthrough for a design review and the drills](PRIMER.md#in-a-design-review) of the primer.

## Run it

```bash
cd roofline-core
python3 -m pip install -r requirements.txt             # pytest only, for the tests; the library needs nothing
python3 -m pytest -q
python3 -m pip install -r requirements-notebooks.txt   # JupyterLab, to do the notebooks locally
python3 -m jupyterlab notebooks

cd ../gpu-bench-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q
python3 -m gpubench run --out results        # the measurement suite on this machine
```

On Colab, the first cell of each notebook clones the repository and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | Where | What you do here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | Do all four core notebooks. The numpy backend of the lab measures the roofline of your CPU and the throughput of your disk. | $0 |
| **T1** | Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | Measure a real GPU roofline by dtype, the HBM bandwidth, pinned against pageable copies and the load speed of the weights. | free to ~$0.7/hr |
| **T2** | Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | Measure P2P bandwidth over PCIe against NVLink. Run `nvidia-smi topo -m` on real machines. | ~$0–25 per session |
| **T3** | GCP via the lab's Terraform | Run the suite on a Spot L4 VM with auto-stop. The VM uploads the results to a bucket. | pay per use |

Prices and obtainability change monthly. See [`COMPUTE.md`](../../COMPUTE.md) and §10 of the primer. The full
learning path is in [`CURRICULUM.md`](../../CURRICULUM.md).

## How it fits

| | Read | For |
|---|---|---|
| before | [`gpu-primer`](../gpu-primer/gpu-primer.md) | why a GPU has its shape |
| before | [`gpu-deployment`](../gpu-deployment/gpu-deployment-primer.md) | scale-up against scale-out, the parallelism menu |
| before | [`gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/PRIMER.md) | sizing, TTFT/TPOT budgets |
| after | layer 02 ([`02-cuda-nccl-runtime`](../../02-cuda-nccl-runtime/README.md)) | the execution model, memory access patterns and NCCL collectives, with real measurements |
| after | layer 04 ([`04-inference-engine`](../../04-inference-engine/README.md)) | the engine techniques (batching, paged KV, quantization). This topic calculates their benefit. |
| after | layers 03 and 05 ([`03-kubernetes-gpu`](../../03-kubernetes-gpu/README.md), [`05-orchestrator`](../../05-orchestrator/README.md)) | the cold-start and failure-domain consequences at cluster scale |

## Caveats

- **Predicted against measured.** Each time value that the core prints is a roofline bound. Real kernels are slower than this
  bound. The lab measures the gap on your hardware.
- **Dated facts.** The accelerator specs, prices and obtainability are a snapshot of September 2026, with the tag
  (verify).
