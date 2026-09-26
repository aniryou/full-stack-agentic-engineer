# Curriculum — the LLM serving stack, layer by layer

A study plan for this repository: the order to work through it, what each module lets you explain, decide or
measure, where the material lives, roughly how long it takes and what hardware it needs. It covers all eight
layers, `00-foundations` to `07-application-agent-framework`. In September 2026 layers 01–05 each gained a topic
built the same way — a primer, a minimal core and a detailed lab — and the path below weaves them together with
what 00, 04, 06 and 07 already had. Where to run each tier, what it costs and how to obtain GPUs is in
[`COMPUTE.md`](COMPUTE.md).

*As of 2026-09-26. Product names, versions and prices are the ones in each primer's Verify list; re-check them there.*

---

## The one-minute version

- **Learn it as a spiral, not bottom-up.** Foundations (00) → the engine (04) → down to the hardware and runtime
  that explain the engine's behaviour (01, 02) → back to the engine with real measurements → up through
  Kubernetes (03) and the orchestrator (05) → the gateway (06) and the agents (07), whose workloads shape every
  layer below.
- **Three artifacts per topic in 01–05:** a `PRIMER.md` (concepts, worked numbers), a *core* (a minimal
  from-scratch implementation that runs on a laptop) and a *lab* (the detailed version: real GPUs, a real engine,
  a GCP deployment, each with an offline fallback).
- **Every concept is learnable at T0** — a laptop or Colab CPU, $0. Real GPUs (T1, T2) and Google Cloud (T3) turn
  predictions into measurements; they are optional steps, never prerequisites.
- **Budget about 170 hours** end to end, about 95 of them in layers 01–05 and the engine; shorter routes are in §3.3.
- **Every module ends in a design review:** the two-minute explanation and its drills. §5 indexes every drill in
  the repo and adds cross-layer ones.

---

## 1. How to use it

### 1.1 Three artifacts per topic

| Artifact | What it is | Tier | How to use it |
|---|---|---|---|
| `<topic>/PRIMER.md` | the concepts: numbered sections, each formula with a worked number and the core function that computes it, "In a design review", glossary, sources, Verify list | reading | read the sections a module lists before opening its notebooks |
| `<topic>/<core>/` | the minimal implementation: standard library + numpy, offline, readable in a sitting; 4–6 notebooks | T0 | where the concept is learned; do every exercise |
| `<topic>/<lab>/` | the detailed implementation: T0 fallbacks, GPU code paths, `deploy/` targets (any GPU box, kind or compose, GCP Terraform) | T0 → T3 | run at T0 first, then again on whatever hardware you have |

Each `<topic>/README.md` gives the order to work the topic and its tier table. The notebooks share one pattern:
exercises in `notebooks/`, worked answers in `solutions/`, both generated from `notebooks_src/` by the lab's
`tools/build_notebooks.py`. Every notebook states its tier, opens with "The one-minute version", works examples,
sets 3–6 exercises each followed by a check cell, and ends with "In a design review". To redo an exercise,
`git restore` the notebook or rebuild it. Colab links for every notebook are in each layer's `README.md`;
[`COLAB.md`](COLAB.md) has the setup.

### 1.2 Run tiers

| Tier | Where | Cost | Used for |
|---|---|---|---|
| **T0** | laptop / Colab CPU / CI | $0 | the core concept: simulators, calculators, from-scratch implementations. Every concept is learnable here. |
| **T1** | one small GPU: Colab or Kaggle T4 (free), a rented 24 GB GPU (RTX 4090 or L4, ~$0.3–0.7/hr), GCP L4 Spot | free–$0.7/hr | real kernels, real vLLM with a 0.5–2B model, real metrics |
| **T2** | a multi-GPU box, ideally with NVLink (Kaggle 2×T4 over PCIe is free; 2–8× A100/H100 on RunPod, Vast.ai or Lambda for about an hour) | ~$0–25 per session | collective bandwidth, P2P and NVLink, tensor parallelism |
| **T3** | GCP managed services (GKE, Cloud Run, Inference Gateway, DWS) via Terraform | pay per use; defaults L4 + Spot + scale-to-zero | how it looks in production on one cloud |

Notebooks at T1 and above detect the GPU, Docker, cluster or cloud they need. When it is absent they run a clearly
labelled T0 path — a simulator, a fake server, parsing of bundled tool output — and print what to run on real
hardware. Two labels keep the numbers honest: simulator output says **simulated**, and bundled tool output says
**sample output in the documented format (illustrative)**. Figures in the docs are either computed by code in the
repo or marked `(verify)`.

### 1.3 Working a module

1. Read the primer sections the module lists.
2. Do the core notebook without opening `solutions/`.
3. Run the lab notebook at T0 and write down what it predicts.
4. If you have the tier, run it again on real hardware. The gap between prediction and measurement is the lesson.
5. Close with the design review: say the two-minute version aloud, answer the drills, then check the answers.

---

## 2. What existed, what was missing, what was added

| Layer | What existed | Gap | What was added (September 2026) |
|---|---|---|---|
| **00** foundations | [transformer primer](00-foundations/transformers/docs/transformer-primer.md), three lessons and practice notebooks; [capacity-planning primer](00-foundations/gpu-capacity-planning/PRIMER.md) with `capacity.py` and practice; [open-weight model primer](00-foundations/model-landscape/open-weight-llms-primer.md) and Mistral exercises | MoE and attention variants (MLA) only in passing; no tokenization; capacity math is closed-form, with no step-level model | nothing new; the new 01 and 04 primers build on it (backlog in §6) |
| **01** hardware | [gpu-primer](01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) and [gpu-deployment primer](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) with exercise sets — qualitative, no code | nothing runnable; no quantitative roofline, fabric, storage, reliability or cost model; no measurement; no guidance on obtaining GPUs | [`roofline-and-fabric/`](01-hardware-gpu-fabric/roofline-and-fabric/): [PRIMER](01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), [`roofline-core`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/) (4 notebooks), [`gpu-bench-lab`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/) (4 notebooks; any-GPU and GCP VM deploys) |
| **02** runtime | empty | the whole layer: execution model, memory access, collectives, compatibility, containers, sharing, health | [`cuda-and-nccl/`](02-cuda-nccl-runtime/cuda-and-nccl/): [PRIMER](02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md), [`cuda-nccl-core`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/) (5), [`cuda-nccl-lab`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/) (6; Numba kernels, torch.distributed, nccl-tests, DCGM; any-GPU, GKE and Terraform deploys) |
| **03** Kubernetes | empty | the whole layer: device plugin and DRA, the scheduling cycle, gangs, topology, Kueue, capacity, startup latency | [`gpu-scheduling/`](03-kubernetes-gpu/gpu-scheduling/): [PRIMER](03-kubernetes-gpu/gpu-scheduling/PRIMER.md), [`k8s-gpu-core`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/) (5), [`k8s-gpu-lab`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/) (4; kind with fake GPUs, Kueue, JobSet, LWS; GKE Terraform) |
| **04** engine | [kv-cache](04-inference-engine/kv-cache/), [paged-attention](04-inference-engine/paged-attention/), [flash-attention](04-inference-engine/flash-attention/): primers, minimal numpy implementations, practice | the engine itself: step loop, scheduler, chunked prefill, prefix caching, sampling, speculation, quantization; no real engine; no benchmarking method | [`serving-engine/`](04-inference-engine/serving-engine/): [PRIMER](04-inference-engine/serving-engine/PRIMER.md), [`mini-engine-core`](04-inference-engine/serving-engine/mini-engine-core/) (6; a numpy nano-engine), [`vllm-serving-lab`](04-inference-engine/serving-engine/vllm-serving-lab/) (6; load generator, metrics, sizing, fake server; any-GPU, Cloud Run and GKE deploys) |
| **05** orchestrator | empty; `agentic-scaling-lab` in 06 touched capacity and autoscaling from the gateway side | routing, autoscaling signals, disaggregation, KV tiers, the Kubernetes-native stack | [`serving-orchestration/`](05-orchestrator/serving-orchestration/): [PRIMER](05-orchestrator/serving-orchestration/PRIMER.md), [`orchestrator-core`](05-orchestrator/serving-orchestration/orchestrator-core/) (5; a discrete-event fleet simulator), [`inference-gateway-lab`](05-orchestrator/serving-orchestration/inference-gateway-lab/) (5; a router, fake backends, an HPA recommender; compose, kind + llm-d, GKE Inference Gateway) |
| **06** gateway | identity and security (a core, a GCP lab, a Mistral core); scaling, admission and cost (a GCP lab, a Mistral lab) | no LLM gateway: model routing and fallback, semantic caching, token metering, OpenTelemetry GenAI conventions | nothing new; backlog in §6 |
| **07** agents | agent fundamentals (3 labs), long-running durable execution (5 labs, primers, drills), retrieval (3 labs, a primer) | agent memory, computer-use agents, evals at scale | nothing new; backlog in §6 |
| root | `README.md`, `CLAUDE.md`, `COLAB.md` | no path across layers; no guide to hardware, obtainability and cost | this file and [`COMPUTE.md`](COMPUTE.md) |

---

## 3. The learning path

### 3.1 Why this order

Bottom-up (01 → 07) is how the stack is built, not how it is best learned: a roofline means little before you know
which step of which program it bounds. Top-down starts from the most familiar layer but never explains why the
numbers are what they are. The path is a spiral around the engine:

```
00 foundations ──▶ 04 engine, concepts (T0) ──▶ 01 hardware ──▶ 02 runtime ─────┐
                                                                                  ▼
07 agents ◀── 06 gateway ◀── 05 orchestrator ◀── 03 Kubernetes ◀── 04 engine, measured (T1)
```

- **00 first**, because every later number is derived from a model's shapes (parameters, KV bytes per token) and
  from two constraints (memory, bandwidth) against two SLOs (TTFT, TPOT).
- **04 next**, because the engine's step loop is where the stack's decisions become visible: batching, the KV
  budget, prefix caching, speculation. You learn *what* the engine does before *why* it is fast or slow.
- **Down to 01 and 02 for the why.** The roofline explains why decode is memory-bound and why quantization speeds
  it up; tiling and fusion explain FlashAttention and CUDA Graphs; collectives explain why tensor parallelism stays
  inside an NVLink domain. With the engine in mind, each hardware number has a use.
- **Back to 04 with a GPU.** The engine lab's measurements are now predictions to check, and §9–12 of the engine
  primer (parallelism, LoRA, measuring, where to run) land on the hardware you just studied.
- **Up to 03 and 05.** Kubernetes makes GPUs schedulable, queueable and obtainable; the orchestrator turns many
  engines into a fleet: routing for cache hits, autoscaling on the right signal, splitting prefill from decode.
- **06 and 07 last**, as the workload that drives all of it. Turns, sessions and shared prefixes are what the engine
  caches and the router exploits, and the gateway bounds them. The spiral closes when an agent's prompt layout (07)
  shows up as a prefix-cache hit rate (04) and a routing decision (05).

If you already build agents, skim 07.1 and 06.1 first for motivation, then start the spiral.

### 3.2 The path at a glance

Hours are rough: an engineer comfortable with Python, doing every exercise before looking at the solutions.
*Min tier* is what the step needs; *best* is where its measurements are real.

| Step | Layer | Modules | Material | Hours | Min tier | Best |
|---:|---|---|---|---:|---|---|
| 1 | 00 | 00.1 | transformer primer, lessons, practice | 5 | T0 | T0 |
| 2 | 00 | 00.2, 00.3 | capacity-planning primer and practice; skim the model landscape | 3 | T0 | T0 |
| 3 | 04 | 04.0 | kv-cache, paged-attention and flash-attention primers and notebooks | 5 | T0 | T0 |
| 4 | 04 | 04.1–04.6 | serving-engine PRIMER §1–8; `mini-engine-core` 01–06 | 11 | T0 | T0 |
| 5 | 04 | 04.1 | `vllm-serving-lab` 01–02 | 3 | T0 | T1 |
| 6 | 01 | 01.0 | gpu-primer and gpu-deployment primer with their exercises | 4 | T0 | T0 |
| 7 | 01 | 01.1–01.5 | roofline-and-fabric PRIMER; `roofline-core` 01–04 | 8 | T0 | T0 |
| 8 | 01 | 01.1–01.4 | `gpu-bench-lab` 01–04 | 5 | T0 | T1; 03 at T2 |
| 9 | 02 | 02.1–02.5 | cuda-and-nccl PRIMER; `cuda-nccl-core` 01–05 | 9 | T0 | T0 |
| 10 | 02 | 02.1–02.4 | `cuda-nccl-lab` 01–05 | 7 | T0 | T1; 03–04 at T2 |
| 11 | 04 | 04.2–04.7 | serving-engine PRIMER §9–12; `vllm-serving-lab` 03–05 | 7 | T0 | T1 |
| 12 | 04 | 04.7 | `vllm-serving-lab` 06 (Cloud Run GPU) | 2 | T0 (inspect) | T3 |
| 13 | 03 | 03.1–03.6 | gpu-scheduling PRIMER; `k8s-gpu-core` 01–05 | 9 | T0 | T0 |
| 14 | 03 | 03.1–03.4, 03.6 | `k8s-gpu-lab` 01–03 | 5 | T0 | T0 + Docker |
| 15 | 03, 02 | 03.5, 02.5 | `k8s-gpu-lab` 04; `cuda-nccl-lab` 06 | 4 | T0 (inspect) | T3 |
| 16 | 05 | 05.1–05.6 | serving-orchestration PRIMER; `orchestrator-core` 01–05 | 9 | T0 | T0 |
| 17 | 05 | 05.1–05.3, 05.6 | `inference-gateway-lab` 01–04 | 6 | T0 | T0 + Docker |
| 18 | 05 | 05.6 | `inference-gateway-lab` 05 (GKE Inference Gateway) | 2 | T0 (inspect) | T3 |
| 19 | 06 | 06.1–06.5 | `agentic-scaling-lab`, then the hosted-vs-self-hosted parts of its Mistral variant | 8 | T0 | T0 |
| 20 | 06 | 06.6 | `agentic-identity-core`, then `agentic-identity-gcp-lab` | 12 | T0 | T0 (T3 optional) |
| 21 | 07 | 07.1, 07.2 | `agent-core`, then `gcp-agent-platform-lab` | 24 | T0 | T0 |
| 22 | 07 | 07.3 | a durable core, then one full long-running lab | 10 | T0 | T0 (T3 optional) |
| 23 | 07 | 07.4 | vector-databases primer, `embeddings-lab`, `rag-from-scratch`, `vector_stores` | 15 | T0 | T0 |

Total: about 170 hours. Steps 3–18 (the engine and layers 01–05) are about 95 hours, of which the three T3 steps
(12, 15, 18) are 8 and optional. "T0 + Docker" means a laptop with Docker for kind or compose; without Docker those
notebooks fall back to a bundled simulator.

### 3.3 Shorter routes

| Route | For | Modules, in order | Hours |
|---|---|---|---:|
| Serving-infrastructure core | the stack from engine to fleet, all at T0 | 00.2 → 04.0 → 04.1–04.3 (core 01–03) → 01.1–01.3 (core 01–03) → 02.3 (core 03) → 05.1–05.5 (core 01–05) → 03.3–03.5 (core 03–05), reading the matching primer sections | ~40 |
| Agent builder | what agent design does to the layers below | 07.1 → 06.1–06.3 → 04.3 (core 03, lab 04) → 05.2 and 05.5 (core 02, 05) → 06.6 (identity core) → 07.2 (notebooks 04, 08, 09) → 07.3 (a durable core) | ~30 |
| Measurement weekend | turning predictions into measurements | the one-GPU, two-GPU and NVLink sessions in [`COMPUTE.md`](COMPUTE.md) §7, after the matching core notebooks | ~10 GPU-hours |

---

## 4. Modules by layer

Module IDs are `<layer>.<n>`; primer sections are `§n`. Notebook links point at the exercise versions in
`notebooks/`; the worked answers sit beside them in `solutions/`.

### 00 · Foundations — `00-foundations/` (existing)

| Module | You can … | Where | Tier |
|---|---|---|---|
| **00.1 Transformer internals** | explain attention as a soft lookup and a block as attention + MLP with residuals; count parameters from a config; say what the KV cache stores and why decoding is sequential | [transformer primer](00-foundations/transformers/docs/transformer-primer.md) §2–8 · [lessons](00-foundations/transformers/lessons/) 01–03 · [`attention_practice`](00-foundations/transformers/practice/attention_practice.ipynb) · [`01_transformer_walkthrough`](00-foundations/transformers/notebooks/01_transformer_walkthrough.ipynb), [`02_transformer_exercises`](00-foundations/transformers/notebooks/02_transformer_exercises.ipynb) | T0 |
| **00.2 Capacity planning** | size weights and KV cache against HBM; estimate TTFT from prefill FLOPs and TPOT from bandwidth; take the GPU count as the maximum over the constraints, plus headroom and N+1 | [PRIMER](00-foundations/gpu-capacity-planning/PRIMER.md) · [`capacity.py`](00-foundations/gpu-capacity-planning/capacity.py) · [`01_capacity_practice`](00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice.ipynb) | T0 |
| **00.3 Model landscape** | say what "open weight" grants and what it does not; check a licence; place a model family by size, architecture (dense or MoE, attention variant) and deployment tier | [open-weight LLMs primer](00-foundations/model-landscape/open-weight-llms-primer.md) §2, §5, §7–8 · [Mistral exercises](00-foundations/model-landscape/mistral-primer-exercises.md) | T0 |

### 01 · Hardware — [`roofline-and-fabric`](01-hardware-gpu-fabric/roofline-and-fabric/README.md) and the existing primers

"Reading the machine: rooflines, memory hierarchy, fabrics, and the cost of a token." Primer:
[`PRIMER.md`](01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). Core:
[`roofline-core`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/) (package `roofline`). Lab:
[`gpu-bench-lab`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/) (package `gpubench`, "measure the
machine you have").

| Module | You can … | Primer | Core notebook | Lab notebook | Tier |
|---|---|---|---|---|---|
| **01.0 The GPU, qualitatively** (existing) | explain why a GPU trades latency for throughput, SIMT and divergence, the memory hierarchy, tensor cores and the precision ladder; draw the scale-up/scale-out boundary and the parallelism menu | [gpu-primer](01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) §1–6 and [exercises](01-hardware-gpu-fabric/gpu-primer/gpu-primer-exercises.md); [gpu-deployment primer](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §0–5 and [exercises](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-exercises.md) | — | — | T0 |
| **01.1 Spec sheets and the roofline** | read peak FLOP/s by precision without being misled by sparsity or boost clocks; compute arithmetic intensity and the ridge point for elementwise, reduction and GEMM kernels; measure your own device's roofline and explain its gap to the spec | §1 Spec-sheet literacy · §2 The roofline model | [`01_spec_sheets_and_the_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb) | [`01_measure_your_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb) | T0 → T1 |
| **01.2 LLM inference on the roofline** | predict prefill and decode step times from a model config and a device; find the batch where decode turns compute-bound; say what KV reads, quantization and MoE do to intensity; measure achieved memory bandwidth and pinned vs pageable host↔device transfers | §3 LLM inference on the roofline · §4 The memory hierarchy and why tiling/fusion win | [`02_llm_inference_on_the_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb) | [`02_memory_bandwidth_and_transfers`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/02_memory_bandwidth_and_transfers.ipynb) | T0 → T1 |
| **01.3 Fabrics and the cost of a collective** | compare NVLink/NVSwitch, PCIe and RDMA NICs with the α-β model; compute the TP all-reduce cost per token inside and across nodes; reason about rail-optimized fat-trees, oversubscription and bisection bandwidth; read `nvidia-smi topo -m`; measure P2P bandwidth | §5 Fabrics quantitatively | [`03_fabrics_and_collective_cost`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb) | [`03_multi_gpu_topology_and_p2p`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/03_multi_gpu_topology_and_p2p.ipynb) | T0 → T2 |
| **01.4 Loading, reliability and the cost of a token** | estimate cold start from checkpoint bytes and storage-tier bandwidth; compute cluster MTBF and a Young/Daly checkpoint interval; convert $/GPU-hr into $/M tokens at a given utilisation; argue rent vs own | §6 Storage and cold start · §7 Reliability at scale · §8 The cost of a token | [`04_loading_reliability_and_cost`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/04_loading_reliability_and_cost.ipynb) | [`04_weights_loading_and_cold_start`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/04_weights_loading_and_cold_start.ipynb) | T0 → T1 (T3) |
| **01.5 The landscape and getting hardware** | place T4 through GB300, MI300X–MI355X and TPU v5e–v7 by memory, bandwidth and interconnect; choose a capacity type (on-demand, Spot, flex-start, reservation) and a provider for an experiment | §9 The accelerator landscape · §10 Getting hardware · [`COMPUTE.md`](COMPUTE.md) | — | deploy: [`any-gpu`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/any-gpu/), [`gcp/terraform`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/gcp/terraform/) | T0 (T3) |

### 02 · Runtime — [`cuda-and-nccl`](02-cuda-nccl-runtime/cuda-and-nccl/README.md)

"The GPU software substrate: CUDA's execution model, collectives, and how a container gets a GPU." Primer:
[`PRIMER.md`](02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md). Core:
[`cuda-nccl-core`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/) (package `gpusim`). Lab:
[`cuda-nccl-lab`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/) (package `gpurt`; the same Numba kernel source
runs in the CUDA simulator at T0 and on a GPU at T1).

| Module | You can … | Primer | Core notebook | Lab notebook | Tier |
|---|---|---|---|---|---|
| **02.1 The execution model and memory access** | explain grid, block and warp, SIMT divergence, occupancy and latency hiding; count 32-byte sectors per warp access and spot uncoalesced access and shared-memory bank conflicts; write a kernel and check it in the CUDA simulator | §2 The execution model · §3 Memory access patterns | [`01_simt_warps_and_memory`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) | [`01_kernels_in_the_simulator`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb) | T0 |
| **02.2 Tiling, fusion, occupancy and launch overhead** | compute the HBM traffic of a naive vs tiled GEMM and of fused vs unfused softmax; find what limits occupancy; time memory-bound kernels on a GPU against the roofline; explain why engines capture decode steps as CUDA Graphs | §3 Memory access patterns · §4 Streams, launch overhead and CUDA Graphs | [`02_tiling_fusion_and_occupancy`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) | [`02_memory_bound_kernels_on_a_real_gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) | T0 → T1 |
| **02.3 Collectives** | state the semantics of broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all and send/recv; derive ring all-reduce as reduce-scatter + all-gather and its α-β cost; compute algbw and busbw as nccl-tests defines them; fit α and β from a message-size sweep; say which collectives TP, EP and PP use and how to debug a hang | §5 Collectives | [`03_collectives_from_scratch`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) | [`03_collectives_with_torch_distributed`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/03_collectives_with_torch_distributed.ipynb), [`04_busbw_and_the_alpha_beta_fit`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) | T0 → T2 |
| **02.4 Compatibility and containers** | apply the driver ↔ CUDA runtime ↔ compute-capability rules; explain SASS vs PTX JIT and the "no kernel image is available" error; explain how a container gets a GPU: device nodes, driver libraries injected by the NVIDIA Container Toolkit (OCI hook or CDI), the CUDA userland in the image | §1 The stack from driver to framework · §6 How a container gets a GPU | [`04_compatibility_and_containers`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) | [`05_how_a_container_sees_a_gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/05_how_a_container_sees_a_gpu.ipynb) | T0 → T1 |
| **02.5 Sharing and health** | choose MIG, time-slicing or MPS for an isolation and utilisation requirement; explain why "GPU util" misleads and which DCGM fields to read instead; triage XID errors, ECC and throttling; map all of it to GKE and to container vs VM providers | §7 Sharing a GPU · §8 Health and observability · §9 On GCP and elsewhere | [`05_sharing_and_health`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) | [`06_gpu_sharing_and_dcgm_on_gke`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) | T0 → T3 |

Deploy targets: [`any-gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/) (Docker, nccl-tests,
a Kaggle 2×T4 recipe), [`gke`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gke/) (smoke, CUDA sample,
2-GPU nccl-tests, MIG and time-sharing examples), [`gcp/terraform`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).

### 03 · Kubernetes — [`gpu-scheduling`](03-kubernetes-gpu/gpu-scheduling/README.md)

"Kubernetes for GPUs: how a GPU becomes schedulable, and how to place, share, queue and scale it." Primer:
[`PRIMER.md`](03-kubernetes-gpu/gpu-scheduling/PRIMER.md). Core:
[`k8s-gpu-core`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/) (package `gpusched`, a pure-Python scheduler,
quota and autoscaler). Lab: [`k8s-gpu-lab`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/) (package `k8sgpu`). The
scheduler sees a GPU as an integer, so nearly all of this layer is real at T0 with fake GPU capacity.

| Module | You can … | Primer | Core notebook | Lab notebook | Tier |
|---|---|---|---|---|---|
| **03.1 How Kubernetes sees a GPU** | explain capacity and allocatable, extended resources (integer, no overcommit, requests equal limits), the device plugin API (Register, ListAndWatch, Allocate), GPU labels and taints, and DRA (ResourceClaim, DeviceClass, ResourceSlice, CEL selectors); decide GPU Operator vs managed drivers; lint a GPU pod spec | §1 What Kubernetes sees · §2 GPU Operator vs managed drivers | [`01_how_kubernetes_sees_a_gpu`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) | [`01_manifests_and_the_linter`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/01_manifests_and_the_linter.ipynb) | T0 |
| **03.2 The scheduling cycle and fragmentation** | walk a pod through queue sort → filter → score → reserve/permit → bind; compare LeastAllocated and MostAllocated scoring and measure fragmentation; explain priority and preemption; diagnose a Pending pod from its events | §3 The scheduling cycle | [`02_filter_score_and_fragmentation`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/02_filter_score_and_fragmentation.ipynb) | [`03_why_is_my_pod_pending`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/03_why_is_my_pod_pending.ipynb) | T0 |
| **03.3 Gangs and topology** | explain why partial placement deadlocks and how Kueue suspend/admit, JobSet and LWS make placement all-or-nothing; place a gang in the smallest topology domain (host or NVLink domain, sub-block, block) with Kueue topology-aware scheduling | §4 Gangs · §5 Topology-aware placement | [`03_gangs_and_topology`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/03_gangs_and_topology.ipynb) | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) | T0 (+ Docker) |
| **03.4 Queues, quotas and preemption** | design ResourceFlavors, ClusterQueues, LocalQueues and cohorts with borrowing and lending limits; predict which workload is preempted and why; use fair sharing and WorkloadPriorityClass | §6 Queues, quotas and multi-tenancy with Kueue | [`04_queues_quotas_and_preemption`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/04_queues_quotas_and_preemption.ipynb) | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) | T0 (+ Docker) |
| **03.5 Getting capacity and starting fast** | choose among autoscaling from zero, Spot, reservations, DWS flex-start with queued provisioning (ProvisioningRequest) and calendar mode; write ComputeClass fallbacks; budget pod startup and cut it with image streaming, secondary boot disks, GCS FUSE or Hyperdisk ML, and model streamers | §7 Getting capacity · §8 Startup latency | [`05_autoscaling_and_obtainability`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/05_autoscaling_and_obtainability.ipynb) | [`04_gke_pools_dws_and_computeclasses`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/04_gke_pools_dws_and_computeclasses.ipynb) | T0 → T3 |
| **03.6 Cluster-level sharing; learning locally** | configure MIG, time-sharing and MPS per node pool or through DRA; say what kind with fake capacity, KWOK and the fake GPU operator reproduce faithfully and what they cannot | §9 Sharing GPUs at the cluster level · §10 Learning locally | — | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) · deploy [`kind`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/) | T0 (+ Docker) |

Deploy targets: [`kind`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/) (a laptop cluster with fake
`nvidia.com/gpu` capacity, Kueue, JobSet, LWS),
[`gcp/terraform`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) and
[`gke`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gke/) (ComputeClass fallbacks, the DWS admission check,
GCS FUSE weights).

### 04 · Inference engine — [`serving-engine`](04-inference-engine/serving-engine/README.md) and the existing kernel topics

"Inside an inference engine: the step loop, scheduling, batching, caching, speculation and quantization." Primer:
[`PRIMER.md`](04-inference-engine/serving-engine/PRIMER.md). Core:
[`mini-engine-core`](04-inference-engine/serving-engine/mini-engine-core/) (package `minengine`, a numpy
"nano-vLLM": a tiny transformer over a paged KV cache, a continuous-batching scheduler, a prefix cache, a sampler,
speculative decoding, quantization). Lab: [`vllm-serving-lab`](04-inference-engine/serving-engine/vllm-serving-lab/)
(package `servelab`; a real vLLM, and a fake OpenAI-compatible server as its T0 target).

| Module | You can … | Primer | Core notebook | Lab notebook | Tier |
|---|---|---|---|---|---|
| **04.0 KV cache, paging and attention kernels** (existing) | compute KV bytes per token and per request; explain fragmentation and how block tables, refcounts and copy-on-write fix it; explain FlashAttention's tiling and online softmax and why it is orthogonal to paging | [kv-cache primer](04-inference-engine/kv-cache/kv-cache-primer.md) · [paged-attention primer](04-inference-engine/paged-attention/paged-attention-primer.md) · [flash-attention primer](04-inference-engine/flash-attention/flash-attention-primer.md) | [`01_kv_cache_worked`](04-inference-engine/kv-cache/01_kv_cache_worked.ipynb), [`02_kv_cache_practice`](04-inference-engine/kv-cache/02_kv_cache_practice.ipynb), [`paged_attention_practice`](04-inference-engine/paged-attention/paged_attention_practice.ipynb), [`flash_attention_practice`](04-inference-engine/flash-attention/flash_attention_practice.ipynb) | — | T0 |
| **04.1 The step loop and continuous batching** | describe one engine step from the API server to the detokenizer; explain iteration-level scheduling, request states and the token budget (`max_num_batched_tokens`, `max_num_seqs`); size KV blocks and maximum concurrency from a `config.json` before serving; serve a small model and measure TTFT and ITL | §1 Anatomy of an engine · §2 Continuous batching | [`01_the_step_loop_and_continuous_batching`](04-inference-engine/serving-engine/mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) | [`01_size_before_you_serve`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb), [`02_serve_and_measure`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) | T0 → T1 |
| **04.2 Chunked prefill and the KV budget** | explain prefill/decode interference and how chunked prefill bounds ITL; choose a token budget for an SLO; explain preemption by recompute vs swap and read the preemption counter | §3 Chunked prefill and prefill/decode interference · §4 KV cache management revisited | [`02_chunked_prefill_and_the_token_budget`](04-inference-engine/serving-engine/mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | [`03_knobs_and_tradeoffs`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) | T0 → T1 |
| **04.3 Prefix caching** | explain block hashing with a parent hash, refcounts and LRU eviction of free cached blocks, and the radix-tree alternative; compute a hit rate; lay out an agent's prompt for cache hits and measure the TTFT it saves | §5 Prefix caching | [`03_prefix_caching`](04-inference-engine/serving-engine/mini-engine-core/notebooks/03_prefix_caching.ipynb) | [`04_prefix_caching_for_agents`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) | T0 → T1 |
| **04.4 Sampling and structured output** | implement temperature, top-k, top-p, min-p, penalties and seeded sampling with logprobs; explain how a grammar or FSM token mask enforces a JSON schema | §6 Sampling and structured output | [`04_sampling_and_structured_output`](04-inference-engine/serving-engine/mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) | — | T0 |
| **04.5 Speculative decoding** | show why acceptance with probability min(1, p/q) plus residual resampling preserves the target distribution; compute expected tokens per step (1−α^(k+1))/(1−α); decide when speculation pays and which drafter fits (draft model, n-gram, EAGLE, MTP) | §7 Speculative decoding | [`05_speculative_decoding`](04-inference-engine/serving-engine/mini-engine-core/notebooks/05_speculative_decoding.ipynb) | [`05_speculation_and_quantization_in_vllm`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 → T1 |
| **04.6 Quantization** | compare weight-only INT8/INT4 (GPTQ, AWQ) with FP8 W8A8 and FP8 KV; pick a scale granularity; say what each buys for prefill and for decode; check the accuracy cost | §8 Quantization | [`06_quantization`](04-inference-engine/serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) | [`05_speculation_and_quantization_in_vllm`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | T0 → T1 (FP8 needs an sm_89+ GPU) |
| **04.7 Parallelism, LoRA, measurement and deployment** | explain TP's column/row split and its two all-reduces per layer, PP, EP and DP; serve many LoRA adapters on one base model; run open- and closed-loop benchmarks with warm-up and report goodput against an SLO; choose an engine and a place to run it; deploy on Cloud Run GPU | §9 Parallelism inside the engine · §10 Multi-LoRA serving · §11 Measuring an engine · §12 Engines and where to run them | — | [`03_knobs_and_tradeoffs`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb), [`06_deploy_on_cloud_run_gpu`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) | T0 → T1 (T3) |

Deploy targets: [`any-gpu`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/) (`vllm/vllm-openai`
with a small model; a Colab/Kaggle T4 recipe), [`gcp/cloud-run`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/),
[`gcp/gke`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/). For the fleet view of a vLLM
replica (batch, latency, break-even price) see 06.5.

### 05 · Orchestrator — [`serving-orchestration`](05-orchestrator/serving-orchestration/README.md)

"Orchestrating a fleet of engines: routing, autoscaling, disaggregation and KV-cache tiers." Primer:
[`PRIMER.md`](05-orchestrator/serving-orchestration/PRIMER.md). Core:
[`orchestrator-core`](05-orchestrator/serving-orchestration/orchestrator-core/) (package `fleetsim`, a
discrete-event simulator of replicas, routers, autoscalers, a disaggregated pool and KV tiers). Lab:
[`inference-gateway-lab`](05-orchestrator/serving-orchestration/inference-gateway-lab/) (package `igwlab`, a
readable re-implementation of the endpoint picker's decision logic in front of OpenAI-compatible backends).

| Module | You can … | Primer | Core notebook | Lab notebook | Tier |
|---|---|---|---|---|---|
| **05.1 Why a layer above the engine** | explain why round-robin and least-connections fail when replicas are stateful caches; name the three decisions (which replica, how many, how to split the work); build a filter → scorer → picker router | §1 Why a layer above the engine · §2 Routing signals and algorithms | [`01_why_llm_load_balancing_is_different`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb) | [`01_router_in_process`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/01_router_in_process.ipynb) | T0 |
| **05.2 Cache-aware routing and the load trade-off** | compare least-outstanding, power-of-two choices, prefix hashing, consistent hashing with bounded loads and weighted multi-scorer routing; choose scorer weights; handle hot prefixes; explain router-side queues, priorities and shedding | §2 Routing signals and algorithms · §3 Flow control and priorities | [`02_cache_aware_routing_and_the_load_tradeoff`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) | [`02_scorer_weights_and_hot_prefixes`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) | T0 |
| **05.3 Autoscaling on the right signal** | apply the HPA rule `ceil(current × metric/target)` with its tolerance, stabilization windows and policies; scale on waiting requests, KV usage or SLO attainment rather than GPU utilisation; budget cold start and decide on scale-to-zero | §4 Autoscaling | [`03_autoscaling_on_the_right_signal`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb) | [`03_autoscaling_recommender`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) | T0 (T3) |
| **05.4 Prefill/decode disaggregation** | decide when splitting prefill from decode helps and when it does not; size the P:D ratio; compute KV transfer bytes and the bandwidth that hides them; place NIXL, llm-d, Dynamo and vLLM KV connectors | §5 Prefill/decode disaggregation | [`04_prefill_decode_disaggregation`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb) | — | T0 |
| **05.5 KV cache beyond HBM** | compute offload and onload time across HBM → DRAM → NVMe → remote tiers; size the KV working set of multi-turn agent sessions; explain cross-replica KV sharing (LMCache, Mooncake) | §6 KV cache beyond HBM | [`05_kv_cache_tiers_and_agent_sessions`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) | — | T0 |
| **05.6 Many models, and the Kubernetes-native stack** | route by base model and LoRA adapter; sketch wide-EP for large MoE models; explain how InferencePool, the llm-d Router and its endpoint picker, InferenceObjective, GKE Inference Gateway, Dynamo, Ray Serve LLM and KServe relate; run the stack locally and on GKE | §7 Multi-model, multi-LoRA and model routing · §8 Large MoE topologies (wide-EP) in brief · §9 The Kubernetes-native stack, Sep 2026 · §10 Where to run it | — | [`04_local_stack_with_llm_d`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb), [`05_gke_inference_gateway`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) | T0 (+ Docker) → T3 |

Deploy targets: [`local`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/local/) (docker compose:
three simulated backends, the router, Prometheus), [`kind`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/kind/)
(llm-d Router in standalone mode with Envoy and `llm-d-inference-sim`),
[`gcp/terraform`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/) and
[`gke`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/) (InferencePool, endpoint picker,
Gateway, InferenceObjective priorities, HPA on a Prometheus metric).

### 06 · Gateway — `06-gateway/` (existing)

| Module | You can … | Where | Tier |
|---|---|---|---|
| **06.1 Scaling by bounding tokens** | go from conversations per day to tokens per minute, in-flight turns (Little's law), cost per conversation and a provisioned-throughput break-even | [`agentic-scaling-lab`](06-gateway/scaling-admission-cost/agentic-scaling-lab/) [scaling primer](06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1–3 · [`01_scaling_math`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb) | T0 |
| **06.2 The turn as the unit of work** | bound a turn on steps, tokens, dollars and time; make a crash mid-turn produce one side effect, not two | primer §5.4 · [`02_turn_loop_and_durability`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/02_turn_loop_and_durability.ipynb) | T0 |
| **06.3 Rate limits, retries, breakers, admission** | explain a 429; smooth traffic with a token bucket; retry with full jitter inside a deadline; break circuits; degrade before shedding with `Retry-After` | primer §5.1–5.3 · [`03_rate_limits_and_admission`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/03_rate_limits_and_admission.ipynb) | T0 |
| **06.4 From a load test to settings** | show the overload feedback loop; derive concurrency, instance counts and the in-flight cap from measurements | primer §5.8, §6 · [`04_load_to_settings`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/04_load_to_settings.ipynb) | T0 |
| **06.5 Hosted API vs your own GPUs** | compute a vLLM replica's batch, latency and throughput, the fleet for peak and the break-even GPU price; decide hosted, self-hosted or spill-over — the bridge from 06 down to 04 and 05 | [Mistral variant primer](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/01-scaling-primer.md) §3.5–3.6 · [`scalelab/serving.py`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/scalelab/serving.py) · [platform mapping](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/04-platform-mapping.md) (vLLM settings) · [`01_scaling_math`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/01_scaling_math.ipynb), [`04_load_to_settings`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/04_load_to_settings.ipynb) | T0 |
| **06.6 Identity and policy for agents** | give each agent its own principal; separate its own authority from delegated authority (RFC 8693 token exchange); enforce deny-by-default policy outside the model; make the MCP server a resource server; audit every decision with both identities | [`agentic-identity-core`](06-gateway/identity-security/agentic-identity-core/) (the five moves in one file) · [identity primer](06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) · [`agentic-identity-gcp-lab` notebooks 01–09](06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/) | T0 (T3 optional) |

### 07 · Application and agents — `07-application-agent-framework/` (existing)

| Module | You can … | Where | Tier |
|---|---|---|---|
| **07.1 The loop** | build the loop with termination, tool dispatch, a step budget and an approval gate; design tool contracts with structured errors and idempotent writes | [`agent-core`](07-application-agent-framework/agent-fundamentals/agent-core/): [`01_the_agent_loop`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/01_the_agent_loop.ipynb), [`02_tools`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/02_tools.ipynb), [`03_state_and_control`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/03_state_and_control.ipynb), [`04_mini_support_agent`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/04_mini_support_agent.ipynb) | T0 |
| **07.2 The platform** | choose between a workflow and multiple agents; keep state as an event log with checkpoints; lay out context for cache hits and compaction (the agent side of 04.3); expose tools over MCP behind a policy gateway; propagate identity with OAuth; gate releases on evals; trace with `gen_ai.*` attributes; estimate cost and latency | [`gcp-agent-platform-lab`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) notebooks 00–14, e.g. [`04_context_engineering_and_caching`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/04_context_engineering_and_caching.ipynb), [`08_evals_trajectory_judge_gates`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/08_evals_trajectory_judge_gates.ipynb), [`09_tracing_and_metrics`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/09_tracing_and_metrics.ipynb), [`14_capstone_bank_agent`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/14_capstone_bank_agent.ipynb) | T0 (model API key optional) |
| **07.3 Durable, long-running agents** | state the invariants (durable state, intent before act, leases, budgets, park instead of wait); run fan-out/fan-in, human-in-the-loop, sagas and scheduled agents; map them onto a queue, a store and stateless compute | cores [`long-running-agents-core`](07-application-agent-framework/long-running-durable/long-running-agents-core/), [`lra-core`](07-application-agent-framework/long-running-durable/lra-core/lra-core/); primer [`00_primer.md`](07-application-agent-framework/long-running-durable/00_primer.md); full labs [`long-running-agents-gcp`](07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/), [`lra-gcp`](07-application-agent-framework/long-running-durable/lra/lra-gcp/); the Temporal-based variant [`long-running-agents-mistral`](07-application-agent-framework/long-running-durable/long-running-agents-mistral/) | T0 (T3 optional) |
| **07.4 Retrieval** | explain embeddings as factorizations and contrastive training; choose and tune an ANN index (IVF, PQ, HNSW); build hybrid search with fusion and reranking; evaluate retrieval with recall@k, MRR and nDCG | [vector-databases primer](07-application-agent-framework/retrieval-rag/vector-databases-primer.md) · [`embeddings-lab`](07-application-agent-framework/retrieval-rag/embeddings-lab/) · [`rag-from-scratch`](07-application-agent-framework/retrieval-rag/rag-from-scratch/) · [`vector_stores`](07-application-agent-framework/retrieval-rag/vector_stores/) | T0 |

---

## 5. Design-review drills

### 5.1 Where the drills are

| Layer | Where | What you get |
|---|---|---|
| 00 | [capacity-planning primer](00-foundations/gpu-capacity-planning/PRIMER.md), "Decisions worth defending" | the sizing decisions to argue: dense vs MoE, FP8 vs INT4, prefix caching and speculation, chunked prefill vs disaggregation |
| 00 | [Mistral exercises](00-foundations/model-landscape/mistral-primer-exercises.md) | five applied exercises and an explain-it drill, with worked answers |
| 01 | [gpu-primer exercises](01-hardware-gpu-fabric/gpu-primer/gpu-primer-exercises.md) | recall, ten numerical exercises, diagnostics and synthesis; answers in Part E |
| 01 | [gpu-deployment exercises](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-exercises.md) | recall, eight numerical workouts, design scenarios, a self-assessment; answer key in Part D |
| 01–05 | "In a design review" in each new primer: [01](01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), [02](02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md), [03](03-kubernetes-gpu/gpu-scheduling/PRIMER.md), [04](04-inference-engine/serving-engine/PRIMER.md), [05](05-orchestrator/serving-orchestration/PRIMER.md) | a two-minute walkthrough of the layer and six drill questions with answers |
| 01–05 | the closing "In a design review" of every core and lab notebook | a two-minute explanation of the notebook's idea and 2–3 questions with short answers |
| 06 | [scaling primer](06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §8 "Walking the design in a review" | a 45-minute flow for a scaling prompt and the questions that change the design |
| 06 | [Mistral scaling primer](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/01-scaling-primer.md) §8 "The deployment conversation" | the hosted-vs-self-hosted walkthrough |
| 06 | [identity primer](06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §11 "Design drills" · [`09_code_evaluation_drills`](06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/09_code_evaluation_drills.ipynb) | five system-design prompts, spot-the-bug drills, trade-offs to argue |
| 07 | [`13_code_review_exercises`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/13_code_review_exercises.ipynb), [`14_capstone_bank_agent`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/14_capstone_bank_agent.ipynb) | two buggy programs and six drills with tests; the capstone assembled and reviewed end to end |
| 07 | [long-running design drills](07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/docs/02_design_drills.md) | six system-design prompts with answer sketches, eight find-the-bug snippets, rapid-fire questions |
| 07 | [lra code-evaluation drills](07-application-agent-framework/long-running-durable/lra/lra-gcp/docs/code-evaluation-drills.md) | six find-the-bug snippets and design-round prompts |
| 07 | [long-running primer](07-application-agent-framework/long-running-durable/00_primer.md) §10 "How to explain this design" · [lra-core primer](07-application-agent-framework/long-running-durable/lra-core/lra-core/PRIMER.md) "What to be able to say in a design conversation" | the walkthrough and the talking points |

### 5.2 Cross-layer drills

Each of these crosses at least two layers; the answer sketch names the mechanism, and the modules hold the detail.
Answer aloud first, then read the sketch.

| # | Prompt | Answer sketch | Modules |
|---:|---|---|---|
| 1 | Streamed answers stutter whenever someone submits a long document. | Long prefills share engine steps with decodes and stretch the inter-token gap. Look at ITL p99, not the mean, and at tokens per step; enable chunked prefill with a smaller token budget; if ITL still misses at scale, split prefill and decode into separate pools. | 04.2, 05.4 |
| 2 | An agent's p95 TTFT doubled after a prompt change, with unchanged traffic. | The change broke the shared prefix — something volatile now sits near the top — so the prefix-cache hit rate collapsed and every turn re-prefills; prefix-affinity routing lost its signal too. Stable content first, volatile content last; watch prefix-cache hits over queries. | 07.2, 04.3, 05.2 |
| 3 | Serve a 70B model to 200 concurrent users at 20 tokens/s each: how many GPUs, which ones, connected how? | Weights are 140 GB at BF16 (70 GB at FP8). KV is about 320 KB per token for a 70B GQA model (80 layers × 8 KV heads × 128 × 2 × 2 B), so 4k tokens of context is about 1.3 GB per user and 260 GB for 200. A decode step must read weights plus KV within 50 ms. Put TP inside the NVLink domain (two all-reduces per layer), replicate across nodes, and check the batch against the ridge point. | 00.2, 01.2, 01.3, 04.7 |
| 4 | The dashboard shows 100 % GPU utilisation and throughput is poor. | "GPU util" is the fraction of time any kernel is running, not how busy the SMs are. Read SM active, tensor active and DRAM active from DCGM; compare achieved bandwidth with the roofline; look for a batch capped by KV memory (preemptions, a growing waiting queue). | 02.5, 01.2, 04.2 |
| 5 | Autoscaling on GPU utilisation adds replicas too late and never removes them. | Utilisation saturates early and says nothing about the SLO. Scale on waiting requests, KV usage or SLO attainment, with stabilization windows. The lag is cold start — node provisioning, image pull, weight load — so shorten startup and keep a buffer. | 05.3, 03.5, 01.4 |
| 6 | A 4-GPU job stays Pending while the cluster reports six free GPUs. | The free GPUs are spread across nodes or outside the topology domain the job needs, and a gang cannot start partially. Read the events first; then bin-pack (MostAllocated), place with topology-aware scheduling, or queue for atomic capacity (DWS flex-start through a ProvisioningRequest). | 03.2, 03.3, 03.5 |
| 7 | Tensor parallelism across two 8-GPU nodes was slower than on one node. | TP puts two all-reduces per layer on every token's critical path. Across nodes they cross the scale-out NIC instead of NVLink, and the α-β cost dominates. Keep TP inside the NVLink domain; use PP or replicas across nodes. | 04.7, 02.3, 01.3 |
| 8 | The self-hosted fleet costs more per conversation than the hosted API did. | $/M tokens is $/GPU-hr divided by the tokens per hour you actually produce. Idle minimum replicas, a batch held down by the latency SLO and a low cache hit rate all raise it. Compare against the break-even GPU price, and raise utilisation (cache-aware routing, autoscaling on the right signal) before shopping for cheaper GPUs. | 01.4, 04.7, 05.2, 05.3, 06.5 |
| 9 | Pods on a new node pool fail with "no kernel image is available for execution on the device". | The image carries SASS for other compute capabilities and no PTX that could be JIT-compiled for this GPU, or the driver is older than the CUDA runtime needs. Check the driver, runtime and compute-capability rules; rebuild for the right architectures or pin a matching image. | 02.4, 03.1 |
| 10 | Multi-turn agent sessions get slower every turn, and the prefix-cache hit rate drops at peak. | The sessions' KV working set exceeds HBM, so cached prefixes are evicted before the next turn arrives. Route with session affinity, add DRAM or NVMe KV tiers, and compact context at the agent. | 05.5, 04.3, 07.2 |
| 11 | A 3× burst arrives. What protects the latency SLO in the first minute: autoscaling or admission control? | Admission control: it acts in milliseconds, autoscaling in minutes. Degrade or shed at the gateway with `Retry-After`, prioritise at the router, keep engine queues short; autoscaling restores capacity afterwards. | 06.3, 05.2, 05.3 |

---

## 6. Next topics (not built yet)

Areas the repo does not cover yet, in rough priority order within each layer. Suggested homes follow `CLAUDE.md`'s
rule of one topic sub-folder per sub-domain; new material still arrives through `raw/`.

| Layer | Topic | What it would teach | Suggested home |
|---|---|---|---|
| 00 | MoE architectures | router and load-balancing losses; active vs total parameters and what each costs in memory vs bandwidth; why MoE favours large batches and expert parallelism | `00-foundations/architectures/` |
| 00 | Attention variants: MQA/GQA, MLA, sliding window, hybrid state-space layers | how each shrinks KV bytes per token or attention compute, worked on real configs | `00-foundations/architectures/` |
| 00 | Tokenization | BPE and byte-level vocabularies; tokens per word by language and domain; how the tokenizer moves cost and context limits; chat templates and special tokens | `00-foundations/tokenization/` |
| 01 | TPU deep dive | v5e, v6e and v7 Ironwood: the MXU systolic array, the ICI torus, slices and pods, the XLA/JAX compile model, vLLM on TPU; which parts of the roofline and fabric models transfer | `01-hardware-gpu-fabric/tpu/` |
| 01 | AMD accelerators | MI300X–MI355X memory capacity, ROCm and RCCL against CUDA and NCCL | `01-hardware-gpu-fabric/` (x-ref 02) |
| 01 | Power, cooling and facilities | rack power density, liquid cooling, what an NVL72 rack asks of a datacentre | `01-hardware-gpu-fabric/` |
| 02 | Kernel authoring beyond Numba | Triton and CUDA Tile kernels, autotuning, reading a profiler (Nsight) trace | `02-cuda-nccl-runtime/kernels/` |
| 03 | DRA with a real driver; multi-cluster queues | GPU and NVLink-domain claims on real hardware; MultiKueue across clusters | `03-kubernetes-gpu/` |
| 04 | SGLang and TensorRT-LLM compared with vLLM | the same benchmark (`servelab.bench`) against three engines: RadixAttention, engine builds and in-flight batching, structured-output speed | `04-inference-engine/engine-comparison/` |
| 04 | MoE and long-context serving | expert parallelism inside the engine, context parallelism, KV compression | `04-inference-engine/` |
| 05 | Multi-cluster and multi-region serving | routing across clusters and regions, capacity failover, data residency, global vs regional endpoints | `05-orchestrator/multi-cluster/` |
| 05 | KV tiers on real hardware | LMCache or Mooncake with vLLM at T1/T2, measured against `fleetsim.kvtier` | `05-orchestrator/serving-orchestration/` |
| 06 | An LLM gateway | model routing and fallback chains across models and providers; semantic caching (what is safe to cache, similarity thresholds, invalidation); token metering and per-tenant chargeback; OpenTelemetry GenAI semantic conventions for spans and metrics; streaming-aware rate limits | `06-gateway/llm-gateway/` |
| 07 | Agent memory | episodic, semantic and procedural memory; write policies, retrieval, decay and consolidation; deletion and privacy | `07-application-agent-framework/agent-memory/` |
| 07 | Computer-use agents | screenshot-to-action loops, sandboxing, latency and cost per action, verifying effects | `07-application-agent-framework/computer-use/` |
| 07 | Evals at scale | dataset versioning, judge calibration at volume, offline and online evals, regression gates in CI, the cost of evaluation | `07-application-agent-framework/evals/` |
