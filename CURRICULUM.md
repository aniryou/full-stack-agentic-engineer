# Curriculum — the LLM serving stack, layer by layer

A study plan for this repository: the order to work through it, what each module lets you explain, decide or
measure, where the material lives, roughly how long it takes and what hardware it needs. It covers all eight
layers, `00-foundations` to `07-application-agent-framework`. Layers 01–05 each have a topic in the same shape — a
primer, a minimal core and a detailed lab — and layer 04 also has two deep dives (vLLM's source and FlashAttention).
Four more topics have the same shape: mixture-of-experts and RL and thinking models in 00, quantization in 04 and
sandboxed execution in 07. The path below weaves them together with the other topics in 00, 04, 06 and 07. Where to
run each tier, what it costs and how to obtain GPUs is in [`COMPUTE.md`](COMPUTE.md).

*As of 2026-09-26. Product names, versions and prices are the ones in each primer's Verify list; re-check them there.
Every time in this file (hours per step, module or route) is an estimate for an engineer comfortable with Python who
does every exercise; treat it as a budget, not a measurement.*

---

## The one-minute version

- **Learn it as a spiral, not bottom-up.** Foundations (00) → the engine (04) → down to the hardware and runtime
  that explain the engine's behaviour (01, 02) → back to the engine with real measurements → up through
  Kubernetes (03) and the orchestrator (05) → the gateway (06) and the agents (07), whose workloads shape every
  layer below.
- **Three artifacts per topic in 01–05, and in the newer topics of 00, 04 and 07:** a `PRIMER.md` (concepts, worked
  numbers), a *core* (a minimal from-scratch implementation that runs on a laptop) and a *lab* (the detailed
  version: real GPUs, a real engine, a GCP deployment, each with an offline fallback).
- **Every concept is learnable at T0** — a laptop or Colab CPU, $0. Real GPUs (T1, T2) and Google Cloud (T3) turn
  predictions into measurements; they are optional steps, never prerequisites.
- **Budget about 267 hours** end to end, about 127 of them in layers 01–05; shorter routes are in §3.3.
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

Each `<topic>/README.md` gives the order to work the topic and its tier table. The notebooks of these topics —
layers 01–05 and the four newer topics (00.4, 00.5, 04.9, 07.5) — share one pattern: exercises in `notebooks/`,
worked answers in `solutions/`, both generated from `notebooks_src/` by the lab's `tools/build_notebooks.py`. Every
such notebook states its tier, opens with "The one-minute version", works examples, sets 3–6 exercises each
followed by a check cell that prints ✅, and ends with "In a design review". The older labs in 00, 04, 06 and 07
vary — some keep solutions beside the exercises, some checks are lighter, and the 06 scaling notebooks print
"not attempted" until an exercise is filled in — and each lab's README says what it has. To redo an exercise,
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

## 2. What each layer has, and what it does not cover yet

The right-hand column is what the built material does not teach yet, or teaches only thinly, as of 2026-09-26;
the larger items are proposed as new topics in §6.

| Layer | What is here | Not covered yet |
|---|---|---|
| **00** foundations | [transformer primer](00-foundations/transformers/docs/transformer-primer.md), three lessons and practice notebooks; [capacity-planning primer](00-foundations/gpu-capacity-planning/PRIMER.md) with `capacity.py` and practice; [open-weight model primer](00-foundations/model-landscape/open-weight-llms-primer.md) and Mistral exercises; [`mixture-of-experts/`](00-foundations/mixture-of-experts/README.md): [PRIMER](00-foundations/mixture-of-experts/PRIMER.md), [`moe-core`](00-foundations/mixture-of-experts/moe-core/) (5 notebooks, 67 tests; routers, balance, experts touched, expert-parallel all-to-alls, sizing), [`moe-lab`](00-foundations/mixture-of-experts/moe-lab/) (5 notebooks, 118 tests; a tiny MoE in torch, router hooks, vLLM with expert parallelism on two GPUs, offload and 4-bit experts; any-GPU and GKE deploys); [`rl-and-thinking-models/`](00-foundations/rl-and-thinking-models/README.md): [PRIMER](00-foundations/rl-and-thinking-models/PRIMER.md), [`rl-core`](00-foundations/rl-and-thinking-models/rl-core/) (5 notebooks, 57 tests; REINFORCE, DPO, GRPO, test-time compute, the thinking workload), [`thinking-lab`](00-foundations/rl-and-thinking-models/thinking-lab/) (5 notebooks, 91 tests; GRPO on a tiny transformer, a thinking model in vLLM with a reasoning parser, best-of-n, one GRPO step with vLLM rollouts; any-GPU deploy, the 04 lab's Cloud Run and GKE for T3) | tokenization (BPE, tokens per word by language, chat templates) beyond a paragraph of the transformer primer; attention variants — GQA/MQA, MLA, sliding window, hybrid state-space layers — beyond mentions (§6). The MoE code path is now 00.4 |
| **01** hardware | [gpu-primer](01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) and [gpu-deployment primer](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) with exercise sets — qualitative, no code; [`roofline-and-fabric/`](01-hardware-gpu-fabric/roofline-and-fabric/): [PRIMER](01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), [`roofline-core`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/) (4 notebooks), [`gpu-bench-lab`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/) (4 notebooks, 89 tests; numpy and torch backends, `nvidia-smi` parsers; any-GPU and GCP Spot VM deploys); the core has 4 notebooks and 58 tests | sustained clocks and throttling under load (the lab records `clocks.max.*` but samples neither clock nor power during a sweep); the α (latency) term of the α-β model is never measured at T1/T2; ECC and row remapping; object-store fetch at cold start (the roofline primer assumes 0.1 GB/s per stream); cross-socket host-to-device copies; TPUs, AMD and facilities (§6) |
| **02** runtime | [`cuda-and-nccl/`](02-cuda-nccl-runtime/cuda-and-nccl/): [PRIMER](02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md), [`cuda-nccl-core`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/) (5 notebooks, 128 tests; seven numpy simulators), [`cuda-nccl-lab`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/) (6 notebooks, 124 tests, 2 of them skipped without torch or numba-cuda; Numba kernels in the CUDA simulator and on a GPU, collectives over OS pipes, gloo or NCCL, nccl-tests, DCGM; any-GPU, GKE and Terraform deploys) | CUDA memory management (the caching allocator, `PYTORCH_CUDA_ALLOC_CONF`, pinned memory, UVM); persistence mode; PCIe ACS/IOMMU and diagnosing disabled P2P; GPUDirect RDMA prerequisites; reading a real `NCCL_DEBUG=INFO` log; clock and power management; driver upgrades without a reboot; operating the MPS daemon; cuBLAS/cuDNN algorithm selection; kernel authoring beyond Numba (§6) |
| **03** Kubernetes | [`gpu-scheduling/`](03-kubernetes-gpu/gpu-scheduling/): [PRIMER](03-kubernetes-gpu/gpu-scheduling/PRIMER.md), [`k8s-gpu-core`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/) (5 notebooks, 49 tests), [`k8s-gpu-lab`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/) (4 notebooks, 118 tests; manifest builders, a linter, a Pending analyser; kind with fake GPUs, Kueue, JobSet, LWS; k3s on one GPU VM; GKE Terraform) | MultiKueue; GPU Operator install and upgrade mechanics (ClusterPolicy, the drain sequence); node drain and GPU health remediation (XID/DCGM → Node Problem Detector → cordon and drain, and its interplay with PDBs); NUMA and CPU-manager alignment; Volcano beyond a table row; PodDisruptionBudgets for inference; Autopilot GPU compute classes; pulling multi-GB images; DRA with a real driver (§6) |
| **04** engine | [kv-cache](04-inference-engine/kv-cache/), [paged-attention](04-inference-engine/paged-attention/), [flash-attention](04-inference-engine/flash-attention/): primers, minimal numpy implementations, practice; [`serving-engine/`](04-inference-engine/serving-engine/): [PRIMER](04-inference-engine/serving-engine/PRIMER.md), [`mini-engine-core`](04-inference-engine/serving-engine/mini-engine-core/) (6 notebooks, 67 tests; a numpy nano-engine), [`vllm-serving-lab`](04-inference-engine/serving-engine/vllm-serving-lab/) (6 notebooks, 66 tests; load generator, metrics, sizing, fake server; any-GPU, Cloud Run and GKE deploys). Two deep dives: [`vllm-internals/`](04-inference-engine/vllm-internals/README.md) (a primer on vLLM `main` at `5840d95`, a source map, 1 notebook) and the [FlashAttention deep dive](04-inference-engine/flash-attention/flash-attention-deep-dive.md) (`fa_calculators.py` with 49 tests, a companion notebook). [`quantization/`](04-inference-engine/quantization/README.md): [PRIMER](04-inference-engine/quantization/PRIMER.md), [`quant-core`](04-inference-engine/quantization/quant-core/) (5 notebooks, 78 tests; formats, granularity, GPTQ, AWQ, SmoothQuant, KV quantization, a per-GPU cost model), [`quant-lab`](04-inference-engine/quantization/quant-lab/) (5 notebooks, 86 tests; compressed-tensors checkpoints and llm-compressor recipes, FP16 vs INT4 vs FP8 in vLLM, lm-eval, FP8 KV, NVFP4 and MXFP4; any-GPU deploy, the serving lab's Cloud Run and GKE for T3) | CUDA-graph capture as an operating concern (capture sizes, memory, when to `--enforce-eager`; 04.8 reads the modes in source); async scheduling and CPU–GPU overlap; hybrid and sliding-window KV groups; multimodal prefill and the encoder cache; cross-layer KV sharing. Thin: tokenizer/detokenizer overhead and SSE backpressure, structured-output cost, MoE inside the engine, LoRA loading, SGLang's RadixAttention. KV quantization is now 04.9.4. SGLang and TensorRT-LLM measured, long-context serving (§6) |
| **05** orchestrator | [`serving-orchestration/`](05-orchestrator/serving-orchestration/): [PRIMER](05-orchestrator/serving-orchestration/PRIMER.md), [`orchestrator-core`](05-orchestrator/serving-orchestration/orchestrator-core/) (5 notebooks, 57 tests; a discrete-event fleet simulator), [`inference-gateway-lab`](05-orchestrator/serving-orchestration/inference-gateway-lab/) (5 notebooks, 76 tests; a router with the llm-d endpoint picker's filters → scorers → picker, fake backends, an HPA recommender; compose, kind + llm-d, GKE Inference Gateway) | cost-aware routing across heterogeneous GPUs; request cancellation, migration and drain; multi-cluster and multi-region (§6); endpoint health, eviction and retry; coordinating several router replicas; fleet observability. Thin: SLO planners, canary and model rewrite, session affinity in the lab router, a precise KV-event index; KV tiers on real hardware (§6) |
| **06** gateway | identity and security (a core, a GCP lab, a Mistral core); scaling, admission and cost (a GCP lab, a Mistral lab) | the LLM gateway itself (§6): model routing and fallback chains, semantic caching, token metering and chargeback, OpenTelemetry GenAI conventions, streaming-aware rate limits, provider-key management and virtual keys, tenant isolation, vendor-neutral guardrail placement and cost. Identity: the MCP client-side authorization flow (discovery → PKCE → code → token), DPoP nonces, the SPIFFE Workload API and SVID rotation |
| **07** agents | agent fundamentals (3 labs), long-running durable execution (5 labs, primers, drills), retrieval (3 labs, a primer); [`sandboxed-execution/`](07-application-agent-framework/sandboxed-execution/README.md): [PRIMER](07-application-agent-framework/sandboxed-execution/PRIMER.md), [`sandbox-core`](07-application-agent-framework/sandboxed-execution/sandbox-core/) (5 notebooks, 81 tests; attack probes, a process sandbox, the execution contract, an egress proxy, policy rendered to Kubernetes, warm-pool sizing), [`sandbox-lab`](07-application-agent-framework/sandboxed-execution/sandbox-lab/) (5 notebooks, 112 tests; a network namespace, hardened Docker and gVisor, pod-per-execution on kind, an agent whose code tools fail closed; Docker, kind and GKE Sandbox deploys) | agent memory, episodic and semantic (a paragraph each, no lab); evals at scale; A2A across processes; structured outputs and JSON mode; reflection loops; per-turn cost budgets tied to 04.3's prefix-cache numbers; where prompt-injection defence splits between 06 and 07; streaming and online index updates (`minifaiss` has no delete or update), filtered ANN, retrieval observability; a real durable-engine comparison (Temporal, Restate, Workflows); computer-use agents (§6). Tool-execution sandboxing is now 07.5 |

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
  primer (parallelism, LoRA, measuring, where to run) land on the hardware you just studied. Then read the real
  engine: the vllm-internals primer follows a request through vLLM's source, and the FlashAttention deep dive goes
  below its attention backends.
- **Up to 03 and 05.** Kubernetes makes GPUs schedulable, queueable and obtainable; the orchestrator turns many
  engines into a fleet: routing for cache hits, autoscaling on the right signal, splitting prefill from decode.
- **06 and 07 last**, as the workload that drives all of it. Turns, sessions and shared prefixes are what the engine
  caches and the router exploits, and the gateway bounds them. The spiral closes when an agent's prompt layout (07)
  shows up as a prefix-cache hit rate (04) and a routing decision (05).
- **The four newer topics sit where their prerequisites are.** Mixture-of-experts (00.4) comes right after the
  roofline, because which experts a step streams is a roofline question; quantization (04.9) once the engine has
  been measured, as the deep dive behind its §8; RL and thinking models (00.5) after reading the real engine,
  because thinking workloads reshape the KV budget, the router (05) and the gateway's cost (06); sandboxed
  execution (07.5) right after the agent loop and platform (07.1, 07.2), whose `run_code` tool it makes safe.

If you already build agents, skim 07.1 and 06.1 first for motivation, then start the spiral.

### 3.2 The path at a glance

Hours are estimates, as above.
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
| 9 | 00 | 00.4 | mixture-of-experts PRIMER; `moe-core` 01–05; `moe-lab` 01–05 | 15.5 | T0 | T1; 04 at T2 |
| 10 | 02 | 02.1–02.5 | cuda-and-nccl PRIMER; `cuda-nccl-core` 01–05 | 9 | T0 | T0 |
| 11 | 02 | 02.1–02.4 | `cuda-nccl-lab` 01–05 (collectives over OS pipes or gloo at T0) | 7 | T0 | T1; 03–04 at T2 |
| 12 | 04 | 04.2–04.7 | serving-engine PRIMER §9–12; `vllm-serving-lab` 03–05 | 7 | T0 | T1; 03 exercise 3.6 at T2 |
| 13 | 04 | 04.9 | quantization PRIMER; `quant-core` 01–05; `quant-lab` 01–05 | 19 | T0 | T1 (FP8 on Ada or newer; lab 05 on Blackwell) |
| 14 | 04 | 04.8 | vllm-internals primer, source map and notebook; FlashAttention deep dive and its notebook | 12 | T0 | T0 (T1 to observe) |
| 15 | 00 | 00.5 | rl-and-thinking-models PRIMER; `rl-core` 01–05; `thinking-lab` 01–05 | 22 | T0 | T1 |
| 16 | 04 | 04.7 | `vllm-serving-lab` 06 (Cloud Run GPU) | 2 | T0 (inspect) | T3 |
| 17 | 03 | 03.1–03.6 | gpu-scheduling PRIMER; `k8s-gpu-core` 01–05 | 9 | T0 | T0 |
| 18 | 03 | 03.1–03.4, 03.6 | `k8s-gpu-lab` 01–03 | 5 | T0 | T0 + Docker |
| 19 | 03, 02 | 03.5, 02.5 | `k8s-gpu-lab` 04; `cuda-nccl-lab` 06 | 4 | T0 (inspect) | T3 |
| 20 | 05 | 05.1–05.6 | serving-orchestration PRIMER; `orchestrator-core` 01–05 | 9 | T0 | T0 |
| 21 | 05 | 05.1–05.3, 05.6 | `inference-gateway-lab` 01–04 | 6 | T0 | T0 + Docker |
| 22 | 05 | 05.6 | `inference-gateway-lab` 05 (GKE Inference Gateway) | 2 | T0 (inspect) | T3 |
| 23 | 06 | 06.1–06.5 | `agentic-scaling-lab`, then the hosted-vs-self-hosted parts of its Mistral variant | 8 | T0 | T0 |
| 24 | 06 | 06.6 | `agentic-identity-core`, then `agentic-identity-gcp-lab` | 12 | T0 | T0 (T3 optional) |
| 25 | 07 | 07.1, 07.2 | `agent-core`, then `gcp-agent-platform-lab` | 24 | T0 | T0 |
| 26 | 07 | 07.5 | sandboxed-execution PRIMER; `sandbox-core` 01–05; `sandbox-lab` 01–05 | 12 | T0 | T0 + Docker; 05 at T3 |
| 27 | 07 | 07.3 | a durable core, then one full long-running lab | 10 | T0 | T0 (T3 optional) |
| 28 | 07 | 07.4 | vector-databases and embeddings primers, `embeddings-lab`, `rag-from-scratch`, `vector_stores` | 28 | T0 | T0 |

Total: about 267 hours. Steps 3–22 other than 9 and 15 (layers 01–05, including the kernel topics of layer 04,
quantization and the two deep dives) are about 127 hours, of which the three T3 steps (16, 19, 22) are 8 and
optional. The four newer topics — 00.4 (step 9), 04.9 (13), 00.5 (15) and 07.5 (26) — are 68.5 of the hours.
"T0 + Docker" means a laptop with Docker for kind or compose; without Docker those notebooks fall back to a bundled
simulator.

### 3.3 Shorter routes

| Route | For | Modules, in order | Hours |
|---|---|---|---:|
| Serving-infrastructure core | the stack from engine to fleet, all at T0 | 00.2 → 04.0 → 04.1–04.3 (core 01–03) → 01.1–01.3 (core 01–03) → 04.9.1–04.9.3 (quantization core 01–03) → 02.3 (core 03) → 05.1–05.5 (core 01–05) → 03.3–03.5 (core 03–05), reading the matching primer sections | ~45 |
| Agent builder | what agent design does to the layers below | 07.1 → 06.1–06.3 → 04.3 (core 03, lab 04) → 05.2 and 05.5 (core 02, 05) → 00.5.5 (RL and thinking-models primer §7: what thinking does to serving) → 06.6 (identity core) → 07.5 (sandbox core 01–05) → 07.2 (notebooks 04, 08, 09) → 07.3 (a durable core) | ~35 |
| Measurement weekend | turning predictions into measurements | the one-GPU, two-GPU and NVLink sessions in [`COMPUTE.md`](COMPUTE.md) §7, after the matching core notebooks | ~10 GPU-hours |

### 3.4 One home per cross-layer concept

A few ideas are taught in several topics, each time for that layer's purpose. When you meet one, the module below
is where it is taught most completely; the other places apply it and can link there.

| Concept | Taught most completely in | Why there | Also applied in |
|---|---|---|---|
| Little's law | 07.5.5 (sandboxed-execution PRIMER §6, `sandbox-core` notebook 05) | the only place that states the law, separates the mean it gives from the size you need, and sizes the pool with Erlang C, each step computed by a named function (`pool.mean_occupancy`, `pool.erlang_c`) | 00.2 (concurrency = RPS × duration), 06.1 (turns and sessions in flight), 05.2 (the router's per-endpoint cap), 04.1, 01 and 02 labs, 07.2 notebook 12 |
| Prefix caching | 04.3 (serving-engine PRIMER §5, `mini-engine-core` 03, `vllm-serving-lab` 04) | block hashing with a parent hash, refcounts, LRU eviction and the radix-tree alternative, implemented, then measured on a real engine | 04.8 (vLLM's block pool in source), 05.2 and 05.5 (routing on it, KV tiers), 07.2 notebook 04 (prompt layout), 00.5.5 (dropped thinking), 00.2 (why agents are prefill-dominated), 06.1 (cost per conversation) |
| Token bucket | 06.3 (scaling primer §5.1, `agentic-scaling-lab` notebook 03) | tied to the 429 feedback loop and to admission control, with an exercise that implements it | 07.2 notebook 07 (limits on the agent's API); the 05 primer's §3 leaves per-tenant buckets to this layer |
| Circuit breaker | 07.2 (`gcp-agent-platform-lab` notebook 10) | the full state machine with half-open as an exercise, plus bulkheads, composed deadlines and a fallback chain | 06.3 (a breaker per model at the gateway), 07.3 (`long-running-agents-gcp`) |
| Prompt injection | 06.6 (identity primer §6, `agentic-identity-gcp-lab` notebook 05) | the threat model and the controls that hold whatever the model does: policy outside the model, scoped delegated tokens, input and output screening, egress | 07.2 notebook 11 (the in-agent layers: escaped data blocks, screening, redaction, an eval golden case), 07.5.4 (code tools behind an egress proxy) |
| Durable-execution invariants | 07.3 ([`00_primer.md`](07-application-agent-framework/long-running-durable/00_primer.md) §3, "The five invariants") | the fullest list — durable state, idempotent actions, exclusive progress, bounded execution, hygienic context — each with where it lives and how sagas, fan-out and approvals apply it | `lra-core` and the lra primer (three invariants), `long-running-agents-core` (five rules in code), 06.2 (the turn as the unit of work) |

---

## 4. Modules by layer

Module IDs are `<layer>.<n>`; primer sections are `§n`. Notebook links point at the exercise versions in
`notebooks/`; the worked answers sit beside them in `solutions/`. *Hours* cover the primer sections, the core
notebook and the lab notebook at T0; "+2 T3" is the optional cloud step. They add up to the steps in §3.2.

### 00 · Foundations — `00-foundations/`

| Module | You can … | Where | Hours | Tier |
|---|---|---|---:|---|
| **00.1 Transformer internals** | explain attention as a soft lookup and a block as attention + MLP with residuals; count parameters from a config; say what the KV cache stores and why decoding is sequential | [transformer primer](00-foundations/transformers/docs/transformer-primer.md) §2–8 · [lessons](00-foundations/transformers/lessons/) 01–03 · [`attention_practice`](00-foundations/transformers/practice/attention_practice.ipynb) · [`01_transformer_walkthrough`](00-foundations/transformers/notebooks/01_transformer_walkthrough.ipynb), [`02_transformer_exercises`](00-foundations/transformers/notebooks/02_transformer_exercises.ipynb) | 5 | T0; lesson 03 (`lessons/03_tiny_gpt.py`) and notebooks 01–02 need torch (T0 + CPU torch, or Colab) |
| **00.2 Capacity planning** | size weights and KV cache against HBM; estimate TTFT from prefill FLOPs and TPOT from bandwidth; take the GPU count as the maximum over the constraints, plus headroom and N+1 | [PRIMER](00-foundations/gpu-capacity-planning/PRIMER.md) · [`capacity.py`](00-foundations/gpu-capacity-planning/capacity.py) · [`01_capacity_practice`](00-foundations/gpu-capacity-planning/notebooks/01_capacity_practice.ipynb) | 2 | T0 |
| **00.3 Model landscape** | say what "open weight" grants and what it does not; check a licence; place a model family by size, architecture (dense or MoE, attention variant) and deployment tier | [open-weight LLMs primer](00-foundations/model-landscape/open-weight-llms-primer.md) §2, §5, §7–8 · [Mistral exercises](00-foundations/model-landscape/mistral-primer-exercises.md) | 1 | T0 |

#### 00.4 Mixture-of-experts — [`mixture-of-experts`](00-foundations/mixture-of-experts/README.md)

"Understand the router and the experts, and predict what sparsity does to serving." Primer:
[`PRIMER.md`](00-foundations/mixture-of-experts/PRIMER.md). Core: [`moe-core`](00-foundations/mixture-of-experts/moe-core/) (package `moecore`, numpy: the routers of six model
families, balance losses, a toy trainer, experts touched, expert-parallel all-to-alls, sizing). Lab:
[`moe-lab`](00-foundations/mixture-of-experts/moe-lab/) (package `moelab`: a tiny MoE in torch, router hooks, vLLM on one or two GPUs; every
notebook falls back to a labelled T0 path).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **00.4.1 The MoE layer** | route tokens through the real routers (Mixtral, OLMoE, Qwen3, DeepSeek-V3, gpt-oss, Llama 4) and write the sparse forward pass; count total and active parameters from a config and name the "active" convention a published number uses; argue for fine-grained and shared experts; train a tiny MoE in torch on a CPU | §1 Why sparsity · §2 The MoE layer | [`01_the_moe_layer`](00-foundations/mixture-of-experts/moe-core/notebooks/01_the_moe_layer.ipynb) | [`01_a_tiny_moe_in_torch`](00-foundations/mixture-of-experts/moe-lab/notebooks/01_a_tiny_moe_in_torch.ipynb) | 3 | T0 (torch on CPU) |
| **00.4.2 Routing and load balance** | explain router collapse; compute the Switch auxiliary loss in both normalisations, the z-loss, capacity and token dropping; balance a router with DeepSeek-V3's selection-only bias; record a real router's choices with forward hooks or `--enable-return-routed-experts` and read utilisation and hot experts | §3 Routing and load balance · §4 Training MoE in brief | [`02_routing_and_load_balance`](00-foundations/mixture-of-experts/moe-core/notebooks/02_routing_and_load_balance.ipynb) | [`02_watch_the_router`](00-foundations/mixture-of-experts/moe-lab/notebooks/02_watch_the_router.ipynb) | 3 | T0 → T1 |
| **00.4.3 Which experts a step touches** | derive E(1 − (1 − k/E)^T) and reproduce layer 01's MoE table; predict the batch where decode turns compute-bound from weights streamed ÷ weights multiplied (754 for Mixtral against 207 for a dense 8B on an H200); measure decode step time against batch for an MoE and a dense model | §5 MoE at inference: which experts a step touches | [`03_which_experts_a_batch_touches`](00-foundations/mixture-of-experts/moe-core/notebooks/03_which_experts_a_batch_touches.ipynb) | [`03_batch_vs_weight_stream`](00-foundations/mixture-of-experts/moe-lab/notebooks/03_batch_vs_weight_stream.ipynb) | 3.5 | T0 → T1 |
| **00.4.4 Expert parallelism** | price dispatch and combine on NVLink, PCIe and InfiniBand; model each EP rank as its own roofline and find the slowest; rebalance hot experts; compare TP, TP + EP and DP + EP in vLLM on two GPUs; choose a wide-EP degree | §6 Running MoE on GPUs (§6.1–6.5) | [`04_expert_parallelism_and_all_to_all`](00-foundations/mixture-of-experts/moe-core/notebooks/04_expert_parallelism_and_all_to_all.ipynb) | [`04_expert_parallelism_on_two_gpus`](00-foundations/mixture-of-experts/moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb) | 3.5 | T0 → T2 (T3 on layer 02's `l4x2` pool) |
| **00.4.5 Sizing, cost and small GPUs** | size memory by total, prefill by active and decode by bytes streamed; budget KV room on one GPU the way vLLM does; plan GPUs and EP degree against an ITL target; price $/M tokens against a dense model; choose between CPU offload and 4-bit experts on a 16–24 GB GPU | §6.6 Expert offloading for small GPUs · §6.7 Quantized experts · §7 Sizing and cost · §8 In a design review: failure modes · §9 Where to run it | [`05_sizing_and_cost`](00-foundations/mixture-of-experts/moe-core/notebooks/05_sizing_and_cost.ipynb) | [`05_moe_on_a_small_gpu`](00-foundations/mixture-of-experts/moe-lab/notebooks/05_moe_on_a_small_gpu.ipynb) | 2.5 | T0 → T1 |

Deploy targets: [`any-gpu`](00-foundations/mixture-of-experts/moe-lab/deploy/any-gpu/) (`serve_moe.sh` on one GPU with offload or on two in any
layout, `bench_layouts.sh` for TP vs EP, Colab and Kaggle recipes), [`gke`](00-foundations/mixture-of-experts/moe-lab/deploy/gke/) (a vLLM
Deployment with `--enable-expert-parallel` on layer 02's 2 × L4 `l4x2` pool; no new Terraform).

#### 00.5 RL and thinking models — [`rl-and-thinking-models`](00-foundations/rl-and-thinking-models/README.md)

"How post-training teaches a model to reason, and what thinking does to serving." Primer:
[`PRIMER.md`](00-foundations/rl-and-thinking-models/PRIMER.md). Core: [`rl-core`](00-foundations/rl-and-thinking-models/rl-core/) (package `rlcore`, standard library + numpy:
verifiable toy tasks, a table-of-softmaxes policy so every expectation can also be computed exactly, REINFORCE,
Bradley–Terry and DPO, GRPO with TRL's options and DAPO's fixes, pass@k, the serving workload model). Lab:
[`thinking-lab`](00-foundations/rl-and-thinking-models/thinking-lab/) (package `thinklab`: GRPO on a tiny transformer in torch, a thinking model in
vLLM; a fake vLLM and bundled outputs make every notebook run at T0).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **00.5.1 Policy gradients** | derive REINFORCE and say why a baseline matters; compute the KL-regularised optimum in closed form; watch RL exploit a buggy verifier and lengthen uncharged thinking; train a tiny transformer with SFT then GRPO in torch and read its reward and length curves | §1 From pretraining to post-training · §2 Policy gradients over token sequences | [`01_policy_gradients_on_a_toy_task`](00-foundations/rl-and-thinking-models/rl-core/notebooks/01_policy_gradients_on_a_toy_task.ipynb) | [`01_grpo_on_a_tiny_transformer`](00-foundations/rl-and-thinking-models/thinking-lab/notebooks/01_grpo_on_a_tiny_transformer.ipynb) | 4.5 | T0 (torch on CPU; T1 faster) |
| **00.5.2 Preferences and DPO** | fit a Bradley–Terry reward model; run RLHF and DPO on the same pairs and land both on the closed form; read TRL's DPO metrics; compute GAE; pick a KL budget against a length-biased reward model | §3 Learning from preferences | [`02_preferences_reward_models_and_dpo`](00-foundations/rl-and-thinking-models/rl-core/notebooks/02_preferences_reward_models_and_dpo.ipynb) | — | 2.5 | T0 |
| **00.5.3 GRPO with verifiable rewards** | compute group advantages, the k3 KL and the clipped surrogate by hand; measure the length bias of per-sequence averaging; weigh dynamic sampling; write DAPO's and R1's `GRPOConfig`; keep the books of one GRPO step whose rollouts an inference engine generates (the train–inference log-prob mismatch, stragglers, weight sync) | §4 RL with verifiable rewards and GRPO · §8 The RL training stack in brief | [`03_grpo_with_verifiable_rewards`](00-foundations/rl-and-thinking-models/rl-core/notebooks/03_grpo_with_verifiable_rewards.ipynb) | [`05_rl_rollouts_with_an_engine`](00-foundations/rl-and-thinking-models/thinking-lab/notebooks/05_rl_rollouts_with_an_engine.ipynb) | 5 | T0 → T1 |
| **00.5.4 Test-time compute** | estimate pass@k without bias and pass^k for reliability; predict when majority vote helps or hurts; compare a verifier with a noisy reward model; split a token budget between samples and thinking; price a correct answer | §6 Test-time compute | [`04_test_time_compute`](00-foundations/rl-and-thinking-models/rl-core/notebooks/04_test_time_compute.ipynb) | [`03_test_time_compute_for_real`](00-foundations/rl-and-thinking-models/thinking-lab/notebooks/03_test_time_compute_for_real.ipynb) | 3 | T0 → T1 |
| **00.5.5 Thinking models and serving** | serve a thinking model with `--reasoning-parser`, switch thinking on and off and budget it; avoid the `max_tokens` trap; size a heavy-tailed thinking workload with the capacity primer's formulas plus HBM and ITL caps; choose `max_model_len`; predict prefix-cache hits when templates drop old thinking; route effort by cost per correct answer | §5 Thinking models · §7 What thinking does to serving · §9 Where to run it | [`05_thinking_models_and_the_serving_workload`](00-foundations/rl-and-thinking-models/rl-core/notebooks/05_thinking_models_and_the_serving_workload.ipynb) | [`02_a_thinking_model_on_one_gpu`](00-foundations/rl-and-thinking-models/thinking-lab/notebooks/02_a_thinking_model_on_one_gpu.ipynb), [`04_serving_thinking_models`](00-foundations/rl-and-thinking-models/thinking-lab/notebooks/04_serving_thinking_models.ipynb) | 7 | T0 → T1 |

Deploy targets: [`any-gpu`](00-foundations/rl-and-thinking-models/thinking-lab/deploy/any-gpu/) (`serve.sh`: `vllm serve` with the right reasoning
parser, and `--dtype half` on a T4; `rl_step.sh`: one GRPO step with vLLM rollouts), [`gcp`](00-foundations/rl-and-thinking-models/thinking-lab/deploy/gcp/)
(the 04 serving lab's Cloud Run Terraform and GKE manifests set up for Qwen3-4B with a reasoning parser; no new
Terraform).

### 01 · Hardware — [`roofline-and-fabric`](01-hardware-gpu-fabric/roofline-and-fabric/README.md) and the GPU primers

"Reading the machine: rooflines, memory hierarchy, fabrics, and the cost of a token." Primer:
[`PRIMER.md`](01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md). Core:
[`roofline-core`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/) (package `roofline`). Lab:
[`gpu-bench-lab`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/) (package `gpubench`, "measure the
machine you have").

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **01.0 The GPU, qualitatively** | explain why a GPU trades latency for throughput, SIMT and divergence, the memory hierarchy, tensor cores and the precision ladder; draw the scale-up/scale-out boundary and the parallelism menu | [gpu-primer](01-hardware-gpu-fabric/gpu-primer/gpu-primer.md) §1–6 and [exercises](01-hardware-gpu-fabric/gpu-primer/gpu-primer-exercises.md); [gpu-deployment primer](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §0–5 and [exercises](01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-exercises.md) | — | — | 4 | T0 |
| **01.1 Spec sheets and the roofline** | read peak FLOP/s by precision without being misled by sparsity or boost clocks; compute arithmetic intensity and the ridge point for elementwise, reduction and GEMM kernels; measure your own device's roofline and explain its gap to the spec | §1 Spec-sheet literacy · §2 The roofline model | [`01_spec_sheets_and_the_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb) | [`01_measure_your_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/01_measure_your_roofline.ipynb) | 3 | T0 → T1 |
| **01.2 LLM inference on the roofline** | predict prefill and decode step times from a model config and a device; find the batch where decode turns compute-bound; say what KV reads, quantization and MoE do to intensity; measure achieved memory bandwidth and pinned vs pageable host↔device transfers | §3 LLM inference on the roofline · §4 The memory hierarchy and why tiling/fusion win | [`02_llm_inference_on_the_roofline`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb) | [`02_memory_bandwidth_and_transfers`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/02_memory_bandwidth_and_transfers.ipynb) | 3 | T0 → T1 |
| **01.3 Fabrics and the cost of a collective** | compare NVLink/NVSwitch, PCIe and RDMA NICs with the α-β model; compute the TP all-reduce cost per token inside and across nodes; reason about rail-optimized fat-trees, oversubscription and bisection bandwidth; read `nvidia-smi topo -m`; measure P2P bandwidth | §5 Fabrics quantitatively | [`03_fabrics_and_collective_cost`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb) | [`03_multi_gpu_topology_and_p2p`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/03_multi_gpu_topology_and_p2p.ipynb) | 3 | T0 → T2 |
| **01.4 Loading, reliability and the cost of a token** | estimate cold start from checkpoint bytes and storage-tier bandwidth; compute cluster MTBF and a Young/Daly checkpoint interval; convert $/GPU-hr into $/M tokens at a given utilisation; argue rent vs own | §6 Storage and cold start · §7 Reliability at scale · §8 The cost of a token | [`04_loading_reliability_and_cost`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/notebooks/04_loading_reliability_and_cost.ipynb) | [`04_weights_loading_and_cold_start`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/notebooks/04_weights_loading_and_cold_start.ipynb) | 3 | T0 → T1 (T3) |
| **01.5 The landscape and getting hardware** | place T4 through GB300, MI300X–MI355X and TPU v5e–v7 by memory, bandwidth and interconnect; choose a capacity type (on-demand, Spot, flex-start, reservation) and a provider for an experiment | §9 The accelerator landscape (September 2026 snapshot) · §10 Getting hardware · [`COMPUTE.md`](COMPUTE.md) | — | deploy: [`any-gpu`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/any-gpu/), [`gcp/terraform`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/gcp/terraform/) | 1 | T0 (T3) |

### 02 · Runtime — [`cuda-and-nccl`](02-cuda-nccl-runtime/cuda-and-nccl/README.md)

"The GPU software substrate: CUDA's execution model, collectives, and how a container gets a GPU." Primer:
[`PRIMER.md`](02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md). Core:
[`cuda-nccl-core`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/) (package `gpusim`). Lab:
[`cuda-nccl-lab`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/) (package `gpurt`; the same Numba kernel source
runs in the CUDA simulator at T0 and on a GPU at T1, and the collectives run over a `pipes` backend — a real
multi-process ring over OS pipes, no torch needed — or gloo at T0, NCCL at T2).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **02.1 The execution model and memory access** | explain grid, block and warp, SIMT divergence, occupancy and latency hiding; count 32-byte sectors per warp access and spot uncoalesced access and shared-memory bank conflicts; write a kernel and check it in the CUDA simulator | §2 The execution model · §3 Memory access patterns | [`01_simt_warps_and_memory`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/01_simt_warps_and_memory.ipynb) | [`01_kernels_in_the_simulator`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/01_kernels_in_the_simulator.ipynb) | 3 | T0 |
| **02.2 Tiling, fusion, occupancy and launch overhead** | compute the HBM traffic of a naive vs tiled GEMM and of fused vs unfused softmax; find what limits occupancy; time memory-bound kernels on a GPU against the roofline; explain why engines capture decode steps as CUDA Graphs | §3 Memory access patterns · §4 Streams, launch overhead and CUDA Graphs | [`02_tiling_fusion_and_occupancy`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/02_tiling_fusion_and_occupancy.ipynb) | [`02_memory_bound_kernels_on_a_real_gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/02_memory_bound_kernels_on_a_real_gpu.ipynb) | 3 | T0 → T1 |
| **02.3 Collectives** | state the semantics of broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all and send/recv; derive ring all-reduce as reduce-scatter + all-gather and its α-β cost; compute algbw and busbw as nccl-tests defines them; fit α and β from a message-size sweep (`python3 -m gpurt.dist.bench --backend pipes` at T0, NCCL at T2); say which collectives TP, EP and PP use and how to debug a hang | §5 Collectives | [`03_collectives_from_scratch`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/03_collectives_from_scratch.ipynb) | [`03_collectives_with_torch_distributed`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/03_collectives_with_torch_distributed.ipynb), [`04_busbw_and_the_alpha_beta_fit`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/04_busbw_and_the_alpha_beta_fit.ipynb) | 4 | T0 → T2 |
| **02.4 Compatibility and containers** | apply the driver ↔ CUDA runtime ↔ compute-capability rules; explain SASS vs PTX JIT and the "no kernel image is available" error; explain how a container gets a GPU: device nodes, driver libraries injected by the NVIDIA Container Toolkit (OCI hook or CDI), the CUDA userland in the image | §1 The stack from driver to framework · §6 How a container gets a GPU | [`04_compatibility_and_containers`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/04_compatibility_and_containers.ipynb) | [`05_how_a_container_sees_a_gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/05_how_a_container_sees_a_gpu.ipynb) | 3 | T0 → T1 |
| **02.5 Sharing and health** | choose MIG, time-slicing or MPS for an isolation and utilisation requirement; explain why "GPU util" misleads and which DCGM fields to read instead; triage XID errors, ECC and throttling; map all of it to GKE and to container vs VM providers | §7 Sharing a GPU · §8 Health and observability · §9 On GCP and elsewhere | [`05_sharing_and_health`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core/notebooks/05_sharing_and_health.ipynb) | [`06_gpu_sharing_and_dcgm_on_gke`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/notebooks/06_gpu_sharing_and_dcgm_on_gke.ipynb) | 3 (+2 T3) | T0 → T3 |

Deploy targets: [`any-gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/) (Docker, nccl-tests,
a Kaggle 2×T4 recipe), [`gke`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gke/) (smoke, CUDA sample,
2-GPU nccl-tests, MIG and time-sharing examples), [`gcp/terraform`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).

### 03 · Kubernetes — [`gpu-scheduling`](03-kubernetes-gpu/gpu-scheduling/README.md)

"Kubernetes for GPUs: how a GPU becomes schedulable, and how to place, share, queue and scale it." Primer:
[`PRIMER.md`](03-kubernetes-gpu/gpu-scheduling/PRIMER.md). Core:
[`k8s-gpu-core`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/) (package `gpusched`, a pure-Python scheduler,
quota and autoscaler). Lab: [`k8s-gpu-lab`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/) (package `k8sgpu`). The
scheduler sees a GPU as an integer, so nearly all of this layer is real at T0 with fake GPU capacity.

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **03.1 How Kubernetes sees a GPU** | explain capacity and allocatable, extended resources (integer, no overcommit, requests equal limits), the device plugin API (Register, ListAndWatch, Allocate), GPU labels and taints, and DRA (ResourceClaim, DeviceClass, ResourceSlice, CEL selectors); decide GPU Operator vs managed drivers; lint a GPU pod spec | §1 What Kubernetes sees · §2 GPU Operator vs managed drivers | [`01_how_kubernetes_sees_a_gpu`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) | [`01_manifests_and_the_linter`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/01_manifests_and_the_linter.ipynb) | 3 | T0 |
| **03.2 The scheduling cycle and fragmentation** | walk a pod through queue sort → filter → score → reserve/permit → bind; compare LeastAllocated and MostAllocated scoring and measure fragmentation; explain priority and preemption; diagnose a Pending pod from its events | §3 The scheduling cycle | [`02_filter_score_and_fragmentation`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/02_filter_score_and_fragmentation.ipynb) | [`03_why_is_my_pod_pending`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/03_why_is_my_pod_pending.ipynb) | 2 | T0 |
| **03.3 Gangs and topology** | explain why partial placement deadlocks and how Kueue suspend/admit, JobSet and LWS make placement all-or-nothing; place a gang in the smallest topology domain (host or NVLink domain, sub-block, block) with Kueue topology-aware scheduling | §4 Gangs · §5 Topology-aware placement | [`03_gangs_and_topology`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/03_gangs_and_topology.ipynb) | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) | 3 | T0 (+ Docker) |
| **03.4 Queues, quotas and preemption** | design ResourceFlavors, ClusterQueues, LocalQueues and cohorts with borrowing and lending limits; predict which workload is preempted and why; use fair sharing and WorkloadPriorityClass | §6 Queues, quotas and multi-tenancy with Kueue | [`04_queues_quotas_and_preemption`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/04_queues_quotas_and_preemption.ipynb) | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) | 2 | T0 (+ Docker) |
| **03.5 Getting capacity and starting fast** | choose among autoscaling from zero, Spot, reservations, DWS flex-start with queued provisioning (ProvisioningRequest) and calendar mode; write ComputeClass fallbacks; budget pod startup and cut it with image streaming, secondary boot disks, GCS FUSE or Hyperdisk ML, and model streamers | §7 Getting capacity · §8 Startup latency | [`05_autoscaling_and_obtainability`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/05_autoscaling_and_obtainability.ipynb) | [`04_gke_pools_dws_and_computeclasses`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/04_gke_pools_dws_and_computeclasses.ipynb) | 2 (+2 T3) | T0 → T3 |
| **03.6 Cluster-level sharing; learning locally** | explain how MIG, time-sharing and MPS are set per node pool and what DRA changes; say what kind with fake capacity, KWOK and the fake GPU operator reproduce faithfully and what they cannot | §9 Sharing GPUs at the cluster level · §10 Learning locally | [`01_how_kubernetes_sees_a_gpu`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) exercise 1.6 (time-slicing replicas) | [`02_kind_with_fake_gpus_and_kueue`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) · deploy [`kind`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/), [`gpu-vm`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gpu-vm/) (k3s + the real device plugin, time-slicing) | 2 | T0 (+ Docker); `gpu-vm` T1 |

Deploy targets: [`kind`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/) (a laptop cluster with fake
`nvidia.com/gpu` capacity, Kueue, JobSet, LWS), [`gpu-vm`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gpu-vm/)
(k3s and the real NVIDIA device plugin on one GPU VM you control),
[`gcp/terraform`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) and
[`gke`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gke/) (ComputeClass fallbacks, the DWS admission check,
GCS FUSE weights).

### 04 · Inference engine — [`serving-engine`](04-inference-engine/serving-engine/README.md) and the kernel topics

"Inside an inference engine: the step loop, scheduling, batching, caching, speculation and quantization." Primer:
[`PRIMER.md`](04-inference-engine/serving-engine/PRIMER.md). Core:
[`mini-engine-core`](04-inference-engine/serving-engine/mini-engine-core/) (package `minengine`, a numpy
"nano-vLLM": a tiny transformer over a paged KV cache, a continuous-batching scheduler, a prefix cache, a sampler,
speculative decoding, quantization). Lab: [`vllm-serving-lab`](04-inference-engine/serving-engine/vllm-serving-lab/)
(package `servelab`; a real vLLM, and a fake OpenAI-compatible server as its T0 target). Two deep dives read the
real thing: [`vllm-internals/`](04-inference-engine/vllm-internals/README.md) (vLLM `main` at `5840d95`) and the
[FlashAttention deep dive](04-inference-engine/flash-attention/flash-attention-deep-dive.md).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **04.0 KV cache, paging and attention kernels** | compute KV bytes per token and per request; explain fragmentation and how block tables, refcounts and copy-on-write fix it; explain FlashAttention's tiling and online softmax and why it is orthogonal to paging | [kv-cache primer](04-inference-engine/kv-cache/kv-cache-primer.md) · [paged-attention primer](04-inference-engine/paged-attention/paged-attention-primer.md) · [flash-attention primer](04-inference-engine/flash-attention/flash-attention-primer.md) | [`01_kv_cache_worked`](04-inference-engine/kv-cache/01_kv_cache_worked.ipynb), [`02_kv_cache_practice`](04-inference-engine/kv-cache/02_kv_cache_practice.ipynb), [`paged_attention_practice`](04-inference-engine/paged-attention/paged_attention_practice.ipynb), [`flash_attention_practice`](04-inference-engine/flash-attention/flash_attention_practice.ipynb) (five exercises) | — | 5 | T0 |
| **04.1 The step loop and continuous batching** | describe one engine step from the API server to the detokenizer; explain iteration-level scheduling, request states and the token budget (`max_num_batched_tokens`, `max_num_seqs`); size KV blocks and maximum concurrency from a `config.json` before serving; serve a small model and measure TTFT and ITL | §1 Anatomy of an engine · §2 Continuous batching | [`01_the_step_loop_and_continuous_batching`](04-inference-engine/serving-engine/mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) | [`01_size_before_you_serve`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/01_size_before_you_serve.ipynb), [`02_serve_and_measure`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/02_serve_and_measure.ipynb) | 4 | T0 → T1 |
| **04.2 Chunked prefill and the KV budget** | explain prefill/decode interference and how chunked prefill bounds ITL; choose a token budget for an SLO; explain preemption by recompute vs swap and read the preemption counter; explain why the core's round sizing inputs and the lab's model of vLLM v0.30.0 defaults give 2,164 vs 2,363 KV blocks for Llama-3.1-8B on an L4, and which number the startup log confirms | §3 Chunked prefill and prefill/decode interference · §4 KV cache management revisited | [`02_chunked_prefill_and_the_token_budget`](04-inference-engine/serving-engine/mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | [`03_knobs_and_tradeoffs`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) | 3 | T0 → T1 |
| **04.3 Prefix caching** | explain block hashing with a parent hash, refcounts and LRU eviction of free cached blocks, and the radix-tree alternative; compute a hit rate; lay out an agent's prompt for cache hits and measure the TTFT it saves | §5 Prefix caching | [`03_prefix_caching`](04-inference-engine/serving-engine/mini-engine-core/notebooks/03_prefix_caching.ipynb) | [`04_prefix_caching_for_agents`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb) | 3 | T0 → T1 |
| **04.4 Sampling and structured output** | implement temperature, top-k, top-p, min-p, penalties and seeded sampling with logprobs; explain how a grammar or FSM token mask enforces a JSON schema | §6 Sampling and structured output | [`04_sampling_and_structured_output`](04-inference-engine/serving-engine/mini-engine-core/notebooks/04_sampling_and_structured_output.ipynb) | — | 2 | T0 |
| **04.5 Speculative decoding** | show why acceptance with probability min(1, p/q) plus residual resampling preserves the target distribution; compute expected tokens per step (1−α^(k+1))/(1−α); decide when speculation pays and which drafter fits (draft model, n-gram, EAGLE, MTP) | §7 Speculative decoding | [`05_speculative_decoding`](04-inference-engine/serving-engine/mini-engine-core/notebooks/05_speculative_decoding.ipynb) | [`05_speculation_and_quantization_in_vllm`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | 3 | T0 → T1 |
| **04.6 Quantization** | compare weight-only INT8/INT4 (GPTQ, AWQ) with FP8 W8A8 and FP8 KV; pick a scale granularity; say what each buys for prefill and for decode; check the accuracy cost | §8 Quantization | [`06_quantization`](04-inference-engine/serving-engine/mini-engine-core/notebooks/06_quantization.ipynb) | [`05_speculation_and_quantization_in_vllm`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/05_speculation_and_quantization_in_vllm.ipynb) | 3 | T0 → T1 (FP8 needs an sm_89+ GPU) |
| **04.7 Parallelism, LoRA, measurement and deployment** | explain TP's column/row split and its two all-reduces per layer, PP, EP and DP; explain how one base model serves many LoRA adapters; run open- and closed-loop benchmarks with warm-up and report goodput against an SLO; choose an engine and a place to run it; deploy on Cloud Run GPU | §9 Parallelism inside the engine · §10 Multi-LoRA serving · §11 Measuring an engine · §12 Engines and where to run them | [`01`](04-inference-engine/serving-engine/mini-engine-core/notebooks/01_the_step_loop_and_continuous_batching.ipynb) exercise 1.6 (tensor parallelism); [`02`](04-inference-engine/serving-engine/mini-engine-core/notebooks/02_chunked_prefill_and_the_token_budget.ipynb) worked example 3 (open loop vs saturation, goodput) | [`03_knobs_and_tradeoffs`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/03_knobs_and_tradeoffs.ipynb) exercises 3.1–3.5 (measurement) and 3.6 (tensor parallelism on two T4s), [`06_deploy_on_cloud_run_gpu`](04-inference-engine/serving-engine/vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb) | 3 (+2 T3) | T0 → T1 (3.6 at T2; deployment T3) |
| **04.8 Reading the real engine** | follow a request through vLLM's source (API server / EngineCore split, the token-budget scheduler, block-hash prefix caching and its eviction order, KV pool sizing from `gpu_memory_utilization`, the model runner and CUDA graphs, attention-backend selection); defend an attention kernel with exact byte counts, the FA2/FA3/FA4 changes, decode kernels, paged KV and a Triton forward pass | [vllm-internals primer](04-inference-engine/vllm-internals/vllm-internals-primer.md) §1–4 first, then as needed · [source map](04-inference-engine/vllm-internals/source-map.md) · [FlashAttention deep dive](04-inference-engine/flash-attention/flash-attention-deep-dive.md) | — | [`01_block_hashes_and_eviction`](04-inference-engine/vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb), [`flash_attention_deep_dive`](04-inference-engine/flash-attention/flash_attention_deep_dive.ipynb) | 12 | T0 (observing vLLM and kernel timing T1) |

Deploy targets: [`any-gpu`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/) (`vllm/vllm-openai`
with a small model; a Colab/Kaggle T4 recipe, fp16 only on the T4), [`gcp/cloud-run`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/),
[`gcp/gke`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/). For the fleet view of a vLLM
replica (batch, latency, break-even price) see 06.5.

#### 04.9 Quantization — [`quantization`](04-inference-engine/quantization/README.md)

"Pick a number format for a model and a GPU, and know what it costs you." Primer: [`PRIMER.md`](04-inference-engine/quantization/PRIMER.md) (the
deep dive behind 04.6). Core: [`quant-core`](04-inference-engine/quantization/quant-core/) (package `quantcore`, numpy: grids from their bits,
scale granularity, GPTQ, AWQ, SmoothQuant, W8A8 epilogues, FP8 and KIVI KV caches, an eval with error bars, a
per-GPU cost model). Lab: [`quant-lab`](04-inference-engine/quantization/quant-lab/) (package `quantlab`: compressed-tensors checkpoints,
llm-compressor recipes, a scheme-aware emulator and fake vLLM, lm-eval commands and parsers; a bundled tiny model
makes every notebook run at T0).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **04.9.1 Formats and their error** | enumerate INT, FP8 (E4M3, E5M2) and FP4 grids from their bits and round onto them; compare INT4, FP4, MXFP4 and NVFP4 on Gaussian and heavy-tailed weights; predict SQNR from bits and crest factor; count bits per weight and whole-model GB; say what quantization can and cannot speed up | §1 Why quantize, and what it can and cannot speed up · §2 Number formats | [`01_number_formats_and_error`](04-inference-engine/quantization/quant-core/notebooks/01_number_formats_and_error.ipynb) | — (lab 05's exercises 5.1–5.3 on E2M1, MXFP4 and NVFP4 fit here) | 2 | T0 |
| **04.9.2 Granularity and outliers** | show why per-channel scales fix an outlier row but not an outlier input column; measure what per-token INT8 does to activations with outlier channels and what FP8 does instead; use 128×128 block scales; read a compressed-tensors checkpoint and predict its size | §3 Granularity and the bits-per-weight budget | [`02_granularity_and_outliers`](04-inference-engine/quantization/quant-core/notebooks/02_granularity_and_outliers.ipynb) | [`01_quantize_a_checkpoint`](04-inference-engine/quantization/quant-lab/notebooks/01_quantize_a_checkpoint.ipynb) | 3.5 | T0 → T1 |
| **04.9.3 Calibration and its accuracy cost** | implement GPTQ's column loop and AWQ's scale search; fold SmoothQuant and AWQ scales into a model without changing it; say which layers each method helps; measure the damage as KL, top-1 agreement and task accuracy with its standard error, and decide whether a drop is real | §4 Weight-only post-training quantization · §5 Weight-and-activation quantization · §8 Measuring the accuracy you pay | [`03_gptq_awq_and_smoothquant_from_scratch`](04-inference-engine/quantization/quant-core/notebooks/03_gptq_awq_and_smoothquant_from_scratch.ipynb) | [`03_measure_the_accuracy_cost`](04-inference-engine/quantization/quant-lab/notebooks/03_measure_the_accuracy_cost.ipynb) (and lab 01's RTN vs GPTQ vs AWQ) | 4.5 | T0 → T1 |
| **04.9.4 Activations and the KV cache** | implement a W8A8 GEMM epilogue; compare static and dynamic activation scales; keep the LM head in 16-bit; judge FP8 KV with and without calibrated scales; implement KIVI; size the cache (2,363 → 4,727 blocks for an 8B model on an L4 with FP8 KV) and say which attention backend reads it | §5 · §6 KV-cache quantization | [`04_activation_and_kv_cache_quantization`](04-inference-engine/quantization/quant-core/notebooks/04_activation_and_kv_cache_quantization.ipynb) | [`04_kv_cache_quantization_in_vllm`](04-inference-engine/quantization/quant-lab/notebooks/04_kv_cache_quantization_in_vllm.ipynb) | 3.5 | T0 → T1 (Ada or newer) |
| **04.9.5 Choosing and shipping a scheme** | say what a checkpoint runs as from a T4 to a B200; find where W4A16 stops paying (compute-bound at ~120 tokens per step on an L4, no edge over BF16 by ~460); choose schemes for a T4, an L4 and an H100 serving a 70B; price a million tokens (simulated); serve FP16, INT4 and FP8 and compare them; work the NVFP4 and MXFP4 path on Blackwell | §7 Quantization-aware training and QLoRA in brief · §9 Producing a checkpoint · §10 Choosing a scheme | [`05_choosing_a_scheme`](04-inference-engine/quantization/quant-core/notebooks/05_choosing_a_scheme.ipynb) | [`02_serve_and_compare_schemes`](04-inference-engine/quantization/quant-lab/notebooks/02_serve_and_compare_schemes.ipynb), [`05_fp4_and_the_blackwell_path`](04-inference-engine/quantization/quant-lab/notebooks/05_fp4_and_the_blackwell_path.ipynb) | 5.5 | T0 → T1 (Blackwell for lab 05) |

Deploy targets: [`any-gpu`](04-inference-engine/quantization/quant-lab/deploy/any-gpu/) (`compress.sh`: llm-compressor in its own virtualenv;
`serve.sh`: `vllm serve` per scheme after checking it against the GPU), [`gcp`](04-inference-engine/quantization/quant-lab/deploy/gcp/)
(variables and a wrapper for the serving lab's Cloud Run Terraform and GKE manifests; no new Terraform).

### 05 · Orchestrator — [`serving-orchestration`](05-orchestrator/serving-orchestration/README.md)

"Orchestrating a fleet of engines: routing, autoscaling, disaggregation and KV-cache tiers." Primer:
[`PRIMER.md`](05-orchestrator/serving-orchestration/PRIMER.md). Core:
[`orchestrator-core`](05-orchestrator/serving-orchestration/orchestrator-core/) (package `fleetsim`, a
discrete-event simulator of replicas, routers, autoscalers, a disaggregated pool and KV tiers). Lab:
[`inference-gateway-lab`](05-orchestrator/serving-orchestration/inference-gateway-lab/) (package `igwlab`, a
readable re-implementation of the endpoint picker's decision logic in front of OpenAI-compatible backends).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **05.1 Why a layer above the engine** | explain why round-robin and least-connections fail when replicas are stateful caches; name the three decisions (which replica, how many, how to split the work); build a filter → scorer → picker router | §1 Why a layer above the engine · §2 Routing signals and algorithms | [`01_why_llm_load_balancing_is_different`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/01_why_llm_load_balancing_is_different.ipynb) | [`01_router_in_process`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/01_router_in_process.ipynb) | 3 | T0 |
| **05.2 Cache-aware routing and the load trade-off** | compare least-outstanding, power-of-two choices, prefix hashing, consistent hashing with bounded loads and weighted multi-scorer routing; choose scorer weights, from the llm-d chart's 3:2:2 default to llm-d's optimized-baseline composition (a prefix-cache affinity filter with a TTFT gate plus a token-load scorer), and say why the ranking flips with engine contention and workload; handle hot prefixes; explain router-side queues, priorities and shedding | §2 Routing signals and algorithms · §3 Flow control and priorities | [`02_cache_aware_routing_and_the_load_tradeoff`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/02_cache_aware_routing_and_the_load_tradeoff.ipynb) | [`02_scorer_weights_and_hot_prefixes`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/02_scorer_weights_and_hot_prefixes.ipynb) | 3 | T0 |
| **05.3 Autoscaling on the right signal** | apply the HPA rule `ceil(current × metric/target)` with its tolerance, stabilization windows and policies; scale on demand rather than GPU utilisation — in-flight requests or prefill backlog, with KV usage as a second metric so the HPA takes the larger proposal — and say why waiting requests alone give a sawtooth and KV usage alone climbs one cold start at a time; budget cold start and decide on scale-to-zero | §4 Autoscaling | [`03_autoscaling_on_the_right_signal`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/03_autoscaling_on_the_right_signal.ipynb) | [`03_autoscaling_recommender`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/03_autoscaling_recommender.ipynb) | 3 | T0 (T3) |
| **05.4 Prefill/decode disaggregation** | decide when splitting prefill from decode helps and when it does not; size the P:D ratio; compute KV transfer bytes and the bandwidth that hides them; place NIXL, llm-d, Dynamo and vLLM KV connectors | §5 Prefill/decode disaggregation | [`04_prefill_decode_disaggregation`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/04_prefill_decode_disaggregation.ipynb) | — | 2 | T0 |
| **05.5 KV cache beyond HBM** | compute offload and onload time across HBM → DRAM → NVMe → remote tiers; size the KV working set of multi-turn agent sessions; explain cross-replica KV sharing (LMCache, Mooncake) | §6 KV cache beyond HBM | [`05_kv_cache_tiers_and_agent_sessions`](05-orchestrator/serving-orchestration/orchestrator-core/notebooks/05_kv_cache_tiers_and_agent_sessions.ipynb) | — | 2 | T0 |
| **05.6 Many models, and the Kubernetes-native stack** | route by base model and LoRA adapter; sketch wide-EP for large MoE models; explain how InferencePool, the llm-d Router and its endpoint picker, InferenceObjective, GKE Inference Gateway, Dynamo, Ray Serve LLM and KServe relate; run the stack locally and on GKE | §7 Multi-model, multi-LoRA and model routing · §8 Large MoE topologies (wide-EP) in brief · §9 The Kubernetes-native stack, September 2026 · §10 Where to run it | — | [`04_local_stack_with_llm_d`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/04_local_stack_with_llm_d.ipynb), [`05_gke_inference_gateway`](05-orchestrator/serving-orchestration/inference-gateway-lab/notebooks/05_gke_inference_gateway.ipynb) | 2 (+2 T3) | T0 (+ Docker) → T3 |

Deploy targets: [`local`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/local/) (docker compose:
three simulated backends, the router, Prometheus), [`kind`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/kind/)
(llm-d Router in standalone mode with Envoy and `llm-d-inference-sim`),
[`gcp/terraform`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/) and
[`gke`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/) (InferencePool, endpoint picker,
Gateway, InferenceObjective priorities, HPA on a Prometheus metric).

### 06 · Gateway — `06-gateway/`

| Module | You can … | Where | Hours | Tier |
|---|---|---|---:|---|
| **06.1 Scaling by bounding tokens** | go from conversations per day to tokens per minute, in-flight turns (Little's law), cost per conversation and a provisioned-throughput break-even | [`agentic-scaling-lab`](06-gateway/scaling-admission-cost/agentic-scaling-lab/) [scaling primer](06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1–3 · [`01_scaling_math`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb) | 2 | T0 |
| **06.2 The turn as the unit of work** | bound a turn on steps, tokens, dollars and time; make a crash mid-turn produce one side effect, not two | primer §5.4 · [`02_turn_loop_and_durability`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/02_turn_loop_and_durability.ipynb) | 1.5 | T0 |
| **06.3 Rate limits, retries, breakers, admission** | explain a 429; smooth traffic with a token bucket; retry with full jitter inside a deadline; break circuits; degrade before shedding with `Retry-After` | primer §5.1–5.3 · [`03_rate_limits_and_admission`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/03_rate_limits_and_admission.ipynb) | 1.5 | T0 |
| **06.4 From a load test to settings** | show the overload feedback loop; derive concurrency, instance counts and the in-flight cap from measurements | primer §5.8, §6 · [`04_load_to_settings`](06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/04_load_to_settings.ipynb) | 1 | T0 |
| **06.5 Hosted API vs your own GPUs** | compute a vLLM replica's batch, latency and throughput, the fleet for peak and the break-even GPU price; decide hosted, self-hosted or spill-over. Its serving model is simplified — a decode step is bytes ÷ (bandwidth × 0.6) + 2 ms, with no compute term — so read it as the fleet view; layer 04's `minengine/perf.py` (max of the memory and compute times, 04.1–04.2) is the reference step-time model | [Mistral variant primer](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/01-scaling-primer.md) §3.5–3.6 · [`scalelab/serving.py`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/scalelab/serving.py) · [platform mapping](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/docs/04-platform-mapping.md) (vLLM settings) · [`01_scaling_math`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/01_scaling_math.ipynb), [`04_load_to_settings`](06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/04_load_to_settings.ipynb) | 2 | T0 |
| **06.6 Identity and policy for agents** | give each agent its own principal; separate its own authority from delegated authority (RFC 8693 token exchange); enforce deny-by-default policy outside the model; make the MCP server a resource server; audit every decision with both identities | [`agentic-identity-core`](06-gateway/identity-security/agentic-identity-core/) (the five moves in one file) · [identity primer](06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) · [`agentic-identity-gcp-lab` notebooks 01–09](06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/) | 12 | T0 (T3 optional) |

### 07 · Application and agents — `07-application-agent-framework/`

| Module | You can … | Where | Hours | Tier |
|---|---|---|---:|---|
| **07.1 The loop** | build the loop with termination, tool dispatch, a step budget and an approval gate; design tool contracts with structured errors and idempotent writes | [`agent-core`](07-application-agent-framework/agent-fundamentals/agent-core/): [`01_the_agent_loop`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/01_the_agent_loop.ipynb), [`02_tools`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/02_tools.ipynb), [`03_state_and_control`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/03_state_and_control.ipynb), [`04_mini_support_agent`](07-application-agent-framework/agent-fundamentals/agent-core/notebooks/04_mini_support_agent.ipynb) | 4 | T0 |
| **07.2 The platform** | choose between a workflow and multiple agents; keep state as an event log with checkpoints; lay out context for cache hits and compaction (the agent side of 04.3); expose tools over MCP behind a policy gateway; propagate identity with OAuth; gate releases on evals; trace with `gen_ai.*` attributes; estimate cost and latency | [`gcp-agent-platform-lab`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) notebooks 00–14, e.g. [`04_context_engineering_and_caching`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/04_context_engineering_and_caching.ipynb), [`08_evals_trajectory_judge_gates`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/08_evals_trajectory_judge_gates.ipynb), [`09_tracing_and_metrics`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/09_tracing_and_metrics.ipynb), [`14_capstone_bank_agent`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/14_capstone_bank_agent.ipynb) | 20 | T0 (model API key optional) |
| **07.3 Durable, long-running agents** | state the invariants (durable state, intent before act, leases, budgets, park instead of wait); run fan-out/fan-in, human-in-the-loop, sagas and scheduled agents; map them onto a queue, a store and stateless compute | **The path:** primer [`00_primer.md`](07-application-agent-framework/long-running-durable/00_primer.md), then [`lra-core`](07-application-agent-framework/long-running-durable/lra-core/lra-core/) (standard library, nine tests, three notebooks) → [`lra-gcp`](07-application-agent-framework/long-running-durable/lra/lra-gcp/) (needs only pydantic offline: orphan re-drive, cooperative cancel, workflow versioning, chaos hooks for each crash window, [code-evaluation drills](07-application-agent-framework/long-running-durable/lra/lra-gcp/docs/code-evaluation-drills.md)). **Alternatives:** [`long-running-agents-core`](07-application-agent-framework/long-running-durable/long-running-agents-core/) (the same rules in one standard-library file) → [`long-running-agents-gcp`](07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/) (ADK 2 and Google Cloud clients: a ~420 MB install even for its offline tests); the Temporal-based [`long-running-agents-mistral`](07-application-agent-framework/long-running-durable/long-running-agents-mistral/) (its workflow half needs Python ≥ 3.12) | 10 | T0 (T3 optional) |
| **07.4 Retrieval** | explain embeddings as factorizations and contrastive training; choose and tune an ANN index (IVF, PQ, HNSW); build hybrid search with fusion and reranking; evaluate retrieval with recall@k, MRR and nDCG | [vector-databases primer](07-application-agent-framework/retrieval-rag/vector-databases-primer.md) · [embeddings primer](07-application-agent-framework/retrieval-rag/embeddings-lab/docs/primer.md) · [`embeddings-lab`](07-application-agent-framework/retrieval-rag/embeddings-lab/) (6 notebooks, 6 exercise sets) · [`rag-from-scratch`](07-application-agent-framework/retrieval-rag/rag-from-scratch/) (7 notebooks) · [`vector_stores`](07-application-agent-framework/retrieval-rag/vector_stores/) (IVF, PQ, HNSW and GraphRAG from scratch) | 28 | T0 (`rag-from-scratch`'s semantic embedder needs torch: T0 + torch, or Colab) |

#### 07.5 Sandboxed execution — [`sandboxed-execution`](07-application-agent-framework/sandboxed-execution/README.md)

"Run the model's code without handing it your keys." Primer: [`PRIMER.md`](07-application-agent-framework/sandboxed-execution/PRIMER.md). Core:
[`sandbox-core`](07-application-agent-framework/sandboxed-execution/sandbox-core/) (package `sandboxcore`, standard library: attack probes, a process sandbox, the
execution contract, an egress proxy, policy rendered to Kubernetes, warm-pool arithmetic). Lab:
[`sandbox-lab`](07-application-agent-framework/sandboxed-execution/sandbox-lab/) (package `sandboxlab`: the same controls on a network namespace, Docker with runc
or gVisor, kind and GKE Sandbox, and an agent whose code tools fail closed). No GPU anywhere.

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **07.5.1 The threat model and the isolation ladder** | name a code tool's blast radius and map each risk (secret, egress, fork bomb, disk, CPU, output) to the control that bounds it; run attack probes through every rung — process, dedicated UID, network namespace, hardened container, gVisor — and read what each stops | §1 Why a sandbox, and the threat model · §2 The isolation ladder | [`01_the_threat_model`](07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/01_the_threat_model.ipynb) | [`01_hardened_containers`](07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/01_hardened_containers.ipynb) | 2.5 | T0 (+ Docker) |
| **07.5.2 A process sandbox** | build the first real boundary: a clean environment, a private workspace, a UID per execution, rlimits, a wall-clock kill of the process group, streamed and capped output; say what it cannot stop (the network, a kernel exploit) and why `RLIMIT_NPROC` does nothing for root | §2 · §3 The execution contract | [`02_a_process_sandbox`](07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/02_a_process_sandbox.ipynb) | — | 1 | T0 |
| **07.5.3 The contract and Kubernetes** | design the request/result contract with budgets, exit reasons and an idempotency key; write policy as data and render it to Pod Security *restricted*, a default-deny NetworkPolicy, a non-retrying Job, a RuntimeClass and a ValidatingAdmissionPolicy; predict the admission chain offline and run a pod per execution on kind | §3 · §5 Sandboxes on Kubernetes | [`03_the_execution_contract_and_policies`](07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/03_the_execution_contract_and_policies.ipynb) | [`02_pod_per_execution_on_kind`](07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/02_pod_per_execution_on_kind.ipynb) | 3 | T0 (+ Docker for kind) |
| **07.5.4 Egress, secrets and the agent** | put an allowlisting egress proxy in front of a network-less sandbox that injects a credential the code never holds, refuses redirects and `CONNECT`, and audits every decision; run the 07.1 loop with `run_code`/`fetch_url` behind it and watch a poisoned document fail closed | §4 Network and secrets · §8 Observability, audit and abuse detection | [`04_egress_and_secrets`](07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/04_egress_and_secrets.ipynb) | [`03_egress_proxy_and_secret_brokering`](07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/03_egress_proxy_and_secret_brokering.ipynb), [`04_an_agent_with_a_sandbox_tool`](07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/04_an_agent_with_a_sandbox_tool.ipynb) | 3.5 | T0 |
| **07.5.5 Pools, cost and GKE Sandbox** | see why Little's law is only the floor and size a replace-after-use warm pool with Erlang C; pick an isolation level for a latency budget; price cost per action, cold start included; read the GKE Sandbox (gVisor) Terraform as a design-review checklist | §6 Latency, throughput and cost per action · §7 Browser, computer-use and GPU sandboxes in brief · §9 Where to run it | [`05_pools_latency_and_cost`](07-application-agent-framework/sandboxed-execution/sandbox-core/notebooks/05_pools_latency_and_cost.ipynb) | [`05_gke_sandbox_with_gvisor`](07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks/05_gke_sandbox_with_gvisor.ipynb) | 2 | T0 → T3 |

Deploy targets: [`docker`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/docker/) (the hardened run script, the proxy over a Unix socket,
the gVisor install), [`kind`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/kind/) (restricted Pod Security, default-deny NetworkPolicy,
the proxy, the admission policy; no gVisor), [`gcp/terraform`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gcp/terraform/) and
[`gke`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gke/) (a GKE Sandbox node pool with private nodes and no NAT).

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
| 04 | "In a design review" in the [vllm-internals primer](04-inference-engine/vllm-internals/vllm-internals-primer.md) and the [FlashAttention deep dive](04-inference-engine/flash-attention/flash-attention-deep-dive.md) | the engine as vLLM builds it and the attention-kernel choice, each with drill questions |
| 00, 04, 07 | "In a design review" in the four newer primers: [mixture-of-experts](00-foundations/mixture-of-experts/PRIMER.md#in-a-design-review), [RL and thinking models](00-foundations/rl-and-thinking-models/PRIMER.md#in-a-design-review), [quantization](04-inference-engine/quantization/PRIMER.md#in-a-design-review), [sandboxed execution](07-application-agent-framework/sandboxed-execution/PRIMER.md#in-a-design-review) | a two-minute walkthrough of the topic and six drill questions with answers (seven for sandboxed execution); the MoE primer adds its failure modes as a table ([§8](00-foundations/mixture-of-experts/PRIMER.md#8-in-a-design-review-failure-modes)) |
| 01–05, 00.4, 00.5, 04.9, 07.5 | the closing "In a design review" of every core and lab notebook | a two-minute explanation of the notebook's idea and 2–3 questions with short answers |
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
| 5 | Autoscaling on GPU utilisation adds replicas too late and never removes them. | Utilisation saturates early and says nothing about the SLO. Scale on demand — in-flight requests or prefill backlog — combined with KV usage (the HPA takes the larger proposal), with stabilization windows; waiting requests alone empty the queue, scale down and hit the next cliff, a sawtooth. The lag is cold start — node provisioning, image pull, weight load — so shorten startup and keep a buffer. | 05.3, 03.5, 01.4 |
| 6 | A 4-GPU job stays Pending while the cluster reports six free GPUs. | The free GPUs are spread across nodes or outside the topology domain the job needs, and a gang cannot start partially. Read the events first; then bin-pack (MostAllocated), place with topology-aware scheduling, or queue for atomic capacity (DWS flex-start through a ProvisioningRequest). | 03.2, 03.3, 03.5 |
| 7 | Tensor parallelism across two 8-GPU nodes was slower than on one node. | TP puts two all-reduces per layer on every token's critical path. Across nodes they cross the scale-out NIC instead of NVLink, and the α-β cost dominates. Keep TP inside the NVLink domain; use PP or replicas across nodes. | 04.7, 02.3, 01.3 |
| 8 | The self-hosted fleet costs more per conversation than the hosted API did. | $/M tokens is $/GPU-hr divided by the tokens per hour you actually produce. Idle minimum replicas, a batch held down by the latency SLO and a low cache hit rate all raise it. Compare against the break-even GPU price, and raise utilisation (cache-aware routing, autoscaling on the right signal) before shopping for cheaper GPUs. | 01.4, 04.7, 05.2, 05.3, 06.5 |
| 9 | Pods on a new node pool fail with "no kernel image is available for execution on the device". | The image carries SASS for other compute capabilities and no PTX that could be JIT-compiled for this GPU, or the driver is older than the CUDA runtime needs. Check the driver, runtime and compute-capability rules; rebuild for the right architectures or pin a matching image. | 02.4, 03.1 |
| 10 | Multi-turn agent sessions get slower every turn, and the prefix-cache hit rate drops at peak. | The sessions' KV working set exceeds HBM, so cached prefixes are evicted before the next turn arrives. Route with session affinity, add DRAM or NVMe KV tiers, and compact context at the agent. | 05.5, 04.3, 07.2 |
| 11 | A 3× burst arrives. What protects the latency SLO in the first minute: autoscaling or admission control? | Admission control: it acts in milliseconds, autoscaling in minutes. Degrade or shed at the gateway with `Retry-After`, prioritise at the router, keep engine queues short; autoscaling restores capacity afterwards. | 06.3, 05.2, 05.3 |
| 12 | The assistant switched to a thinking model at the same traffic; the fleet ran out of KV memory and ITL rose. | Output length grew by an order of magnitude with a heavy tail, and a request holds its KV for its whole generation, so concurrency and memory × time grow faster than the output: in the capacity primer's example, 10× the output needs 18× the GPUs for memory. Size on the tail, raise `max_model_len` for it, cap cost with a thinking budget rather than `max_tokens` (which truncates answers), keep the batch within the ITL SLO, expect fewer multi-turn prefix hits because templates drop old thinking, and route effort by cost per correct answer. | 00.5, 00.2, 04.2, 04.3, 06.1 |
| 13 | INT4 weights made decode about 3× faster, but long-prompt TTFT did not improve and got slightly worse. | W4A16 cuts the bytes decode streams, but its kernels dequantize to 16-bit before the matrix multiply, so prefill does the same FLOPs plus the dequantization. On an L4 its GEMM turns compute-bound at about 120 tokens per step and has no edge over BF16 by about 460, so long prefill chunks gain nothing. Prefill gets faster only with formats the tensor cores multiply natively — FP8 W8A8 on Ada or Hopper, INT8 W8A8 with SmoothQuant on older GPUs, NVFP4 on Blackwell — and check what the checkpoint actually runs as (an FP8 checkpoint on an A100 is weight-only). | 04.9, 01.2, 04.2 |
| 14 | "Qwen3-30B-A3B is 3B active, so it runs like a 3B model on our one 24 GB GPU for a low-traffic tool." | Memory follows the total: 61.1 GB in BF16 and 30.5 GB in FP8, so only 4-bit experts (18.5 GB) fit, leaving room for about five 4K-token sessions. Decode streams only the active experts at batch 1, but a batch reads the union of its tokens' experts, so the saving fades as traffic grows; an MoE pays at large batch with expert parallelism. For low traffic on one GPU, a dense model of the active size is usually the better choice. | 00.4, 00.2, 01.2, 04.9 |
| 15 | A web page told the agent to "check connectivity", and its code tool posted the service's API key to an unknown host. | The code ran with the agent's ambient authority: its environment, its files and an open network. Treat a code tool as DESTRUCTIVE tier and remove that authority: a sandbox with a clean environment, its own UID, a budget for every resource and no network by default (a network namespace, or a default-deny NetworkPolicy); allowed APIs only through an egress proxy that injects the credential outbound, refuses redirects and `CONNECT`, and audits every decision. The hosts a tool call declares are the model's claim, not a control. Give the agent its own principal with scoped, delegated tokens so a leak is bounded. | 07.5, 06.6, 07.1 |

---

## 6. Next topics (not built yet)

Areas the repo does not cover yet, in rough priority order within each layer. Suggested homes follow `CLAUDE.md`'s
rule of one topic sub-folder per sub-domain; new material still arrives through `raw/`. Built from this list so far
(2026-09-26): MoE architectures, now [`mixture-of-experts`](00-foundations/mixture-of-experts/README.md) (00.4). The
other three newer topics — RL and thinking models (00.5), quantization (04.9) and sandboxed execution (07.5) — were
not on it; nothing else below has been built.

| Layer | Topic | What it would teach | Suggested home |
|---|---|---|---|
| 00 | Attention variants: MQA/GQA, MLA, sliding window, hybrid state-space layers | how each shrinks KV bytes per token or attention compute, worked on real configs | `00-foundations/architectures/` |
| 00 | Tokenization | BPE and byte-level vocabularies; tokens per word by language and domain; how the tokenizer moves cost and context limits; chat templates and special tokens | `00-foundations/tokenization/` |
| 01 | TPU deep dive | v5e, v6e and v7 Ironwood: the MXU systolic array, the ICI torus, slices and pods, the XLA/JAX compile model, vLLM on TPU; which parts of the roofline and fabric models transfer | `01-hardware-gpu-fabric/tpu/` |
| 01 | AMD accelerators | MI300X–MI355X memory capacity, ROCm and RCCL against CUDA and NCCL | `01-hardware-gpu-fabric/` (x-ref 02) |
| 01 | Power, cooling and facilities | rack power density, liquid cooling, what an NVL72 rack asks of a datacentre | `01-hardware-gpu-fabric/` |
| 02 | Kernel authoring beyond Numba | Triton kernels beyond the FlashAttention deep dive's forward pass, CUDA Tile, autotuning, reading a profiler (Nsight) trace | `02-cuda-nccl-runtime/kernels/` |
| 03 | DRA with a real driver; multi-cluster queues | GPU and NVLink-domain claims on real hardware; MultiKueue across clusters | `03-kubernetes-gpu/` |
| 04 | SGLang and TensorRT-LLM compared with vLLM | the same benchmark (`servelab.bench`) against three engines: RadixAttention, engine builds and in-flight batching, structured-output speed | `04-inference-engine/engine-comparison/` |
| 04 | Long-context serving, and MoE at scale | context parallelism, KV compression; expert parallelism beyond 00.4's two-GPU lab (wide-EP on real multi-node hardware) | `04-inference-engine/` |
| 05 | Multi-cluster and multi-region serving | routing across clusters and regions, capacity failover, data residency, global vs regional endpoints | `05-orchestrator/multi-cluster/` |
| 05 | KV tiers on real hardware | LMCache or Mooncake with vLLM at T1/T2, measured against `fleetsim.kvtier` | `05-orchestrator/serving-orchestration/` |
| 06 | An LLM gateway | model routing and fallback chains across models and providers; semantic caching (what is safe to cache, similarity thresholds, invalidation); token metering and per-tenant chargeback; OpenTelemetry GenAI semantic conventions for spans and metrics; streaming-aware rate limits | `06-gateway/llm-gateway/` |
| 07 | Agent memory | episodic, semantic and procedural memory; write policies, retrieval, decay and consolidation; deletion and privacy | `07-application-agent-framework/agent-memory/` |
| 07 | Computer-use agents | screenshot-to-action loops, browser and desktop sandboxes (07.5 covers code execution and §7 of its primer sketches these), latency and cost per action, verifying effects | `07-application-agent-framework/computer-use/` |
| 07 | Evals at scale | dataset versioning, judge calibration at volume, offline and online evals, regression gates in CI, the cost of evaluation | `07-application-agent-framework/evals/` |
