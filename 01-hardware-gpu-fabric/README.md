# 01 · Hardware — GPUs, NVLink, NICs, storage, cooling

The physical layer: silicon, interconnect, and the datacenter around it. Everything above this
layer is ultimately bounded by what lives here.

**Covers:** GPU SKUs & memory (H100 / H200 / B200, HBM), NVLink / NVSwitch, scale-up vs
scale-out fabrics, RDMA NICs (InfiniBand, RoCE), network topology, storage, power and cooling —
and the arithmetic that ties them together: rooflines, collective cost, cold start, failure
rates, the cost of a token.

**Signal keywords:** NVLink, NVSwitch, InfiniBand, RoCE, HBM, memory bandwidth, scale-up/scale-out,
rail-optimized, GPUDirect, RDMA, power/cooling, TCO, fabric, roofline, arithmetic intensity,
`nvidia-smi topo`, MTBF.

## Current contents
- **`gpu-primer/`** — why a GPU is shaped the way it is, from first principles (primer + exercises).
- **`gpu-deployment/`** — GPU deployment architecture for scale-up: the scale-up (NVLink) vs
  scale-out (InfiniBand/RoCE) fabric distinction and what it means for LLM serving (primer + exercises).

### `roofline-and-fabric/` — reading the machine
Rooflines, the memory hierarchy, fabrics and the cost of a token: datasheets and network diagrams
turned into numbers you can defend in a design review — which resource bounds an LLM step, what a
collective costs on each link, how long a replica takes to load, how often a big job fails, what a
token costs. Every concept runs on a laptop (T0); a GPU and Google Cloud are optional steps up, never
prerequisites. Start at the topic's [`README.md`](roofline-and-fabric/README.md).
- **[`PRIMER.md`](roofline-and-fabric/PRIMER.md)** — ten sections: spec-sheet literacy, the roofline
  model, LLM inference on the roofline (prefill vs decode, KV reads, quantization, MoE), the memory
  hierarchy, fabrics quantitatively (α-β, tensor-parallel all-reduce cost, rails and bisection,
  GPUDirect, `nvidia-smi topo -m`), storage and cold start, reliability at scale, the cost of a token,
  the September 2026 accelerator landscape, getting hardware — then a design-review walkthrough and
  drills. Every computed number comes from `roofline-core` and is pinned by its tests.
- **`roofline-core/`** — *T0, standard library only.* The minimal implementation (package `roofline`):
  `specs`, `roofline`, `llm`, `fabric`, `storage`, `reliability`, `cost`, and four fill-in notebooks
  that **predict** what the hardware should do.
- **`gpu-bench-lab/`** — *T0 → T3.* "Measure the machine you have" (package `gpubench`): a numpy
  backend that measures your own CPU's roofline, memory bandwidth and disk throughput (T0); a PyTorch
  backend for GEMM by dtype, HBM bandwidth, pinned vs pageable copies and weight loading on one GPU
  (T1) and the P2P bandwidth matrix on several (T2); `nvidia-smi` topology and inventory parsers with
  bundled sample output; deploys for any GPU box (Docker, Colab, Kaggle, rented GPUs) and for Google
  Cloud — Terraform for one Spot L4 VM that runs the suite, uploads the report to a bucket and powers
  off (T3). Four notebooks that **measure** what the core predicts.

**Suggested order:** primer → core notebooks → lab notebooks. Each primer section pairs with one core
and one lab notebook (§1–2 → `01`, §3–4 → `02`, §5 → `03`, §6–8 → `04`), so you can also work it a
section at a time — read, predict, measure; the topic README has the step table.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the primer, all four core notebooks, the lab's numpy backend (your CPU's roofline, disk throughput), `nvidia-smi` parsing on sample output | $0 |
| **T1** | one GPU: Colab/Kaggle T4 (free), a rented 24 GB GPU, GCP L4 Spot | a real GPU roofline by dtype, HBM bandwidth, pinned vs pageable copies, weight loading | free – ~$0.7/hr |
| **T2** | ≥ 2 GPUs: Kaggle 2×T4 (free, PCIe only), 2–8× A100/H100 SXM on RunPod/Vast/Lambda, GCP `a2-highgpu-2g` | P2P bandwidth over PCIe vs NVLink, `nvidia-smi topo -m` on a real machine | ~$0–25 per session |
| **T3** | Google Cloud, via the lab's Terraform | the suite on a Spot L4 VM with auto-stop, report uploaded to a bucket | pay per use (~$0.1–0.3/hr on Spot) |

**Cross-references.** Builds on
[`00-foundations/gpu-capacity-planning`](../00-foundations/gpu-capacity-planning/PRIMER.md) (sizing,
TTFT/TPOT budgets) and this layer's `gpu-primer/` and `gpu-deployment/`. Leads to layer 02
([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md): the execution model, memory access
patterns and NCCL collectives behind the fabric numbers here) and layer 04
([`04-inference-engine`](../04-inference-engine/README.md): batching, the KV cache,
[paged attention](../04-inference-engine/paged-attention/paged-attention-primer.md),
[FlashAttention](../04-inference-engine/flash-attention/flash-attention-primer.md) and quantization —
the techniques whose payoff the roofline computes).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`roofline-and-fabric/gpu-bench-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb) `01_measure_your_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/02_memory_bandwidth_and_transfers.ipynb) `02_memory_bandwidth_and_transfers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/03_multi_gpu_topology_and_p2p.ipynb) `03_multi_gpu_topology_and_p2p.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/04_weights_loading_and_cold_start.ipynb) `04_weights_loading_and_cold_start.ipynb`

**`roofline-and-fabric/gpu-bench-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/01_measure_your_roofline.ipynb) `01_measure_your_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/02_memory_bandwidth_and_transfers.ipynb) `02_memory_bandwidth_and_transfers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/03_multi_gpu_topology_and_p2p.ipynb) `03_multi_gpu_topology_and_p2p.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/solutions/04_weights_loading_and_cold_start.ipynb) `04_weights_loading_and_cold_start.ipynb`

**`roofline-and-fabric/roofline-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb) `01_spec_sheets_and_the_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb) `02_llm_inference_on_the_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb) `03_fabrics_and_collective_cost.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/04_loading_reliability_and_cost.ipynb) `04_loading_reliability_and_cost.ipynb`

**`roofline-and-fabric/roofline-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/01_spec_sheets_and_the_roofline.ipynb) `01_spec_sheets_and_the_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/02_llm_inference_on_the_roofline.ipynb) `02_llm_inference_on_the_roofline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/03_fabrics_and_collective_cost.ipynb) `03_fabrics_and_collective_cost.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/solutions/04_loading_reliability_and_cost.ipynb) `04_loading_reliability_and_cost.ipynb`
<!-- colab-links:end -->
