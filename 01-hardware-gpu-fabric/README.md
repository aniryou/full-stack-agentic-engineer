# 01 · Hardware and fabric

Read the machine. After this layer, you can take a GPU spec sheet, a model and a network diagram, and you can say
these things:

- Which resource is the bound for an LLM step.
- What a collective costs on each link.
- How long a replica takes to load.
- How frequently a large job fails.
- What a token costs.

Then you can measure the machine that you have, to examine the prediction.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, memory, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, model routing, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the machine against which you measure every step. It has the limits that the runtime (02) operates
against and that the engine (04) meets.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`gpu-primer/`](gpu-primer/gpu-primer.md) | Explain from first principles why a GPU has its shape (a primer and [exercises](gpu-primer/gpu-primer-exercises.md)). | ~4 h with `gpu-deployment/` | read |
| [`gpu-deployment/`](gpu-deployment/gpu-deployment-primer.md) | Draw the boundary between scale-up (NVLink) and scale-out (InfiniBand/RoCE). Say what this boundary means when you serve an LLM (a primer and [exercises](gpu-deployment/gpu-deployment-exercises.md)). | (with `gpu-primer/`) | read |
| [`roofline-and-fabric/`](roofline-and-fabric/README.md) | From a spec sheet, predict if an LLM step is compute- or memory-bound. Calculate the cost of a collective on NVLink against InfiniBand. Make a budget for a cold start, a failure rate and $/M tokens. Then measure your own CPU, a GPU, or a Spot L4 on Google Cloud. | ~8 h primer + core, ~5 h lab | T0 to T3 |

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is
a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* The times are
approximate. They come from the curriculum of the repository ([`CURRICULUM.md`](../CURRICULUM.md), modules
01.0–01.5).

## Start here

1. Read [`roofline-and-fabric/PRIMER.md`](roofline-and-fabric/PRIMER.md) §1–§2. These sections are about spec-sheet
   literacy and the roofline.
2. Run `cd roofline-and-fabric/roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`.
   The 66 tests take ~30 s. Then open
   [`01_spec_sheets_and_the_roofline`](roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb).
3. If you have any GPU, measure the real hardware with
   [`gpu-bench-lab/notebooks/01_measure_your_roofline`](roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb).
   A free Colab T4 is sufficient. Without a GPU, the notebook measures the roofline of your CPU.

## What is inside `roofline-and-fabric/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](roofline-and-fabric/PRIMER.md) | It has ten sections. The first four are spec-sheet literacy, the roofline model, LLM inference on the roofline (prefill against decode, KV reads, quantization, MoE) and the memory hierarchy. Then come fabrics in numbers (α-β, the cost of a tensor-parallel all-reduce, rails and bisection, GPUDirect, `nvidia-smi topo -m`). After them come storage and cold start, reliability at scale, the cost of a token, the September 2026 accelerator landscape and how to get hardware. Then the primer gives a walkthrough for a design review, and drills. Every computed number comes from `roofline-core`. | read |
| [`roofline-core/`](roofline-and-fabric/roofline-core/README.md) | It is the minimal implementation, in the package `roofline`. It uses only the standard library. Its modules are `specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability` and `cost`. It also has four fill-in notebooks. They **predict** the performance that you can expect from the hardware. | T0 |
| [`gpu-bench-lab/`](roofline-and-fabric/gpu-bench-lab/README.md) | Its motto is "measure the machine you have". The package is `gpubench`. A numpy backend measures the roofline, memory bandwidth and disk throughput of your own CPU (T0). On one GPU, a PyTorch backend measures GEMM by dtype, HBM bandwidth, pinned against pageable copies and how fast the weights load (T1). On several GPUs, it measures the P2P bandwidth matrix (T2). The lab has `nvidia-smi` topology and inventory parsers, with bundled sample output. It has deploys for any GPU box (Docker, Colab, Kaggle, rented GPUs) and for Google Cloud. For Google Cloud, Terraform makes one Spot L4 VM that runs the suite, uploads the report to a bucket and then stops (T3). Four notebooks **measure** what the core predicts. | T0 to T3 |

Do the topic one section at a time. For each section, read, predict and then measure. Each section of the primer
goes with one core notebook and one lab notebook:

- §1–2: `01`
- §3–4: `02`
- §5: `03`
- §6–8: `04`

The topic [README](roofline-and-fabric/README.md) has the table of steps.

## Run it

```bash
cd roofline-and-fabric/roofline-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../gpu-bench-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
python3 -m gpubench info            # what is this machine?
```

Then run `python3 -m jupyterlab notebooks` in one of the two directories. Or use the Colab links in "Run in Colab"
at the end of this page.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | Read the primer. Do all four core notebooks. Run the numpy backend of the lab (the roofline and the disk throughput of your CPU). Parse the `nvidia-smi` sample output. | $0 |
| **T1** | one GPU: Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | Measure a real GPU roofline by dtype, HBM bandwidth, pinned against pageable copies and how fast the weights load. | free to ~$0.7/hr |
| **T2** | ≥ 2 GPUs: Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | Measure P2P bandwidth over PCIe against NVLink. Run `nvidia-smi topo -m` on a real machine. | ~$0–25 per session |
| **T3** | Google Cloud, via the lab's Terraform | Run the suite on a Spot L4 VM with auto-stop. The VM uploads the report to a bucket. | pay per use (~$0.1–0.3/hr on Spot) |

## How it fits

**Needed first:** [`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (the
calculation of capacity, TTFT/TPOT budgets), and the topics `gpu-primer/` and `gpu-deployment/` of this layer. The
[spiral of the curriculum](../CURRICULUM.md#31-why-this-order) goes to the concepts of layer 04 before this layer, and
it does this intentionally. The order is 00, 04, 01, 02, and then 04 again with a GPU. When you know the step loop of
the engine, each hardware number has a use. The roofline itself needs only 00.

**Leads to** three layers:

- Layer 02 ([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md)): the execution model, memory access patterns
  and NCCL collectives that explain the fabric numbers of this layer.
- Layer 03 ([`03-kubernetes-gpu`](../03-kubernetes-gpu/README.md)): cold start, failure domains and topology at
  cluster scale.
- Layer 04 ([`04-inference-engine`](../04-inference-engine/README.md)): batching, the KV cache,
  [paged attention](../04-inference-engine/paged-attention/paged-attention-primer.md),
  [FlashAttention](../04-inference-engine/flash-attention/flash-attention-primer.md) and quantization. The roofline
  calculates the benefit of these techniques.

## Caveats

- Each time that the core prints is a **roofline bound** (ideal overlap, compulsory traffic, peak clocks). Real
  kernels do not reach this bound, and the lab measures the gap. The lab puts one of these labels on its output:
  *measured*, *model*, *assumed*, *spec* or *sample output (illustrative)*.
- GPU specs, prices and availability are a snapshot from September 2026, with the tag (verify). They change every
  month.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`roofline-and-fabric/gpu-bench-lab/`** — [01_measure_your_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb) · [02_memory_bandwidth_and_transfers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/02_memory_bandwidth_and_transfers.ipynb) · [03_multi_gpu_topology_and_p2p](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/03_multi_gpu_topology_and_p2p.ipynb) · [04_weights_loading_and_cold_start](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/04_weights_loading_and_cold_start.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/01_measure_your_roofline.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/02_memory_bandwidth_and_transfers.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/03_multi_gpu_topology_and_p2p.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/04_weights_loading_and_cold_start.ipynb)
- **`roofline-and-fabric/roofline-core/`** — [01_spec_sheets_and_the_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb) · [02_llm_inference_on_the_roofline](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb) · [03_fabrics_and_collective_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb) · [04_loading_reliability_and_cost](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/04_loading_reliability_and_cost.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/01_spec_sheets_and_the_roofline.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/02_llm_inference_on_the_roofline.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/03_fabrics_and_collective_cost.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/04_loading_reliability_and_cost.ipynb)
<!-- colab-links:end -->
