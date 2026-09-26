# SPEC — filling layers 01–05 of full-stack-agentic-engineer

Repo: `/home/user/full-stack-agentic-engineer` (read its `CLAUDE.md`; you already have it). Scratch: `$SP` (see FACTS.md).
Read `$SP/FACTS.md` before writing anything product-specific. This file is the contract every builder and reviewer follows.

## 0. Goal and audience
The learner is a full-stack agentic engineer. The goal is **deep conceptual understanding** of each layer of the LLM serving
stack. **GCP is one deployment target, never a prerequisite**: every concept must be fully learnable on a laptop/Colab CPU;
real hardware and GCP are progressive, optional steps. Writing rule (CLAUDE.md): address an engineer explaining a design in
a design review — never a role or an employer. Vendor stacks are the same concepts worked per provider.

## 1. Run tiers (every notebook and README section declares its tier)
| Tier | Where | Cost | Used for |
|---|---|---|---|
| **T0** | laptop / Colab CPU / CI | $0 | the core concept: simulators, calculators, from-scratch implementations. **Every concept must be learnable here.** |
| **T1** | one small GPU: Colab/Kaggle T4 (free), any rented 24 GB GPU (RTX 4090/L4 ~$0.3–0.7/hr), GCP L4 Spot | free–$0.7/hr | real kernels, real vLLM with a 0.5–2B model, real metrics |
| **T2** | multi-GPU box, ideally NVLink (Kaggle 2×T4 PCIe is free; RunPod/Vast/Lambda 2–8× A100/H100 for ~1 h) | ~$0–25 per session | collectives bandwidth, P2P/NVLink, tensor parallelism |
| **T3** | GCP managed (GKE, Cloud Run, Inference Gateway, DWS) via Terraform | pay-per-use; default L4 + Spot + scale-to-zero | "how it looks in production on one cloud" |
Rules: T1+ notebooks must **degrade gracefully** — detect GPU/Docker/cluster/cloud; if absent, run a clearly labelled T0 path
(simulator, fake server, fixture parsing) and print what to run on real hardware. **Never fabricate measurements**: simulator
output is labelled "simulated"; sample tool output used as fixtures is labelled "sample output in the documented format
(illustrative)". Numbers in docs are either computed by code in the repo or explicitly cited/marked `(verify)`.

## 2. Layout per layer (paths are fixed — other agents and docs link to them)
```
<layer>/<topic>/README.md     topic index: what's here, order to work it, tier table          (builder A)
<layer>/<topic>/PRIMER.md     the concept primer shared by core and lab                        (builder A)
<layer>/<topic>/<core>/       MINIMAL implementation: T0 only, stdlib (+numpy), offline         (builder A)
<layer>/<topic>/<lab>/        DETAILED implementation: T0 fallbacks + T1/T2 + T3 GCP deploy     (builder B)
```
| Layer | Topic dir | Core dir (package) | Lab dir (package) |
|---|---|---|---|
| 01 | `01-hardware-gpu-fabric/roofline-and-fabric` | `roofline-core` (`roofline`) | `gpu-bench-lab` (`gpubench`) |
| 02 | `02-cuda-nccl-runtime/cuda-and-nccl` | `cuda-nccl-core` (`gpusim`) | `cuda-nccl-lab` (`gpurt`) |
| 03 | `03-kubernetes-gpu/gpu-scheduling` | `k8s-gpu-core` (`gpusched`) | `k8s-gpu-lab` (`k8sgpu`) |
| 04 | `04-inference-engine/serving-engine` | `mini-engine-core` (`minengine`) | `vllm-serving-lab` (`servelab`) |
| 05 | `05-orchestrator/serving-orchestration` | `orchestrator-core` (`fleetsim`) | `inference-gateway-lab` (`igwlab`) |
Root `CURRICULUM.md` and `COMPUTE.md` will be written by the integrator; you may link to them (from a topic dir:
`../../CURRICULUM.md`, `../../COMPUTE.md`; from a core/lab dir: `../../../COMPUTE.md`) — the link checker may flag only those two.

## 3. Conventions (mirror `07-application-agent-framework/agent-fundamentals/agent-core` — read its README, pyproject, Makefile, tools/, notebooks_src/01, tests)
- Package dir at lab root (`<pkg>/`, not `src/`); `pyproject.toml` (setuptools, `[tool.setuptools.packages.find] include=["<pkg>*"]`,
  `[project.optional-dependencies] dev=[pytest, nbformat, nbclient, ipykernel, ...]`, extras like `gpu=[...]` for torch/triton/vllm),
  `requirements.txt` (what the notebooks/tests need), `Makefile` (setup/test/notebooks/check/lab/clean), `README.md`, `LICENSE` (MIT), `.gitignore`.
- Cores: standard library + numpy (+ matplotlib only if charts truly help; notebooks must still run). Library code readable in a sitting:
  each module opens with a docstring stating the one idea it teaches. Target 500–1,000 lines of library code in the core.
- Notebooks are **percent-format sources** `notebooks_src/NN_name.py` → `tools/build_notebooks.py` emits blanks `notebooks/` and `solutions/`
  (copy agent-core's `tools/build_notebooks.py` and `tools/run_notebooks.py`; change only the BOOTSTRAP path/package name).
  BOOTSTRAP must: on Colab clone `https://github.com/aniryou/full-stack-agentic-engineer.git` to `/content/full-stack-agentic-engineer`,
  `cd` to THIS lab's repo-relative dir, `pip install -q -e .`; locally, walk up from cwd until `<pkg>/` exists and put it on sys.path.
  Cell kinds: `# %% [markdown]`, `# %%`, `# %% exercise` (with `### BEGIN SOLUTION` / `### END SOLUTION`), `# %% check` (asserts; prints `✅ ...`).
- Every notebook: title + `**Tier:** T0 ...` line + "The one-minute version" (what you will be able to explain) → worked examples →
  3–6 exercises each followed by a check → "In a design review" (how to explain it in two minutes + 2–3 drill questions with short answers).
  Exercises test understanding (implement the key function, predict a number, pick a setting and justify it), not boilerplate.
- Tests: `tests/test_*.py`, pytest, offline, no GPU, < 60 s, one focused test per concept; include tests that pin formulas to hand-computed values.
- Deploy assets (labs): `deploy/<target>/` each with a README (what it does, cost, cleanup). Terraform in `deploy/gcp/terraform/`:
  `versions.tf` (`required_version >= 1.9`, google `>= 8.0`), `variables.tf`, resources split by concern, `outputs.tf`, `terraform.tfvars.example`,
  cheapest defaults (L4, Spot, zonal, autoscale-from-zero / scale-to-zero, labels), `# VERIFY:` comments on product details.
  Shell scripts: `#!/usr/bin/env bash` + `set -euo pipefail`, print each step, support `DRY_RUN=1`.
- Style: match existing primers (`01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md`, `04-inference-engine/paged-attention/paged-attention-primer.md`):
  first principles, worked numbers, tables, ASCII diagrams, short paragraphs. No emojis except ✅ in checks. No marketing tone.
- Never touch: root `README.md`, `CLAUDE.md`, `COLAB.md`, `tools/`, any layer `README.md`, other layers, other builders' dirs.
  Do not `git add/commit`. Do not run `tools/inject_colab_bootstrap.py` (percent-source labs carry their own bootstrap).
- Installs: use `$SP/pipi <pkgs>` (serialized pip; several agents share the environment). Install your package editable: `$SP/pipi -e <dir>`.
  Never install torch/vllm/triton here (torch is not installable; see FACTS). Torch/vLLM code paths must import lazily and be skipped when absent.

## 4. Validation — all must pass before you report (paste the one-line result of each into your report)
```bash
cd <core-or-lab> && python3 -m pytest -q
python3 tools/build_notebooks.py && python3 tools/run_notebooks.py solutions && python3 tools/run_notebooks.py notebooks --expect-fail
$SP/tfcheck.sh deploy/gcp/terraform                      # labs with Terraform (fmt + init via local mirror + validate)
kubernetes-validate --strict -k 1.34.0 <core-k8s-yaml...> # core kinds; CRDs (Kueue/JobSet/LWS/InferencePool/...) → pytest checks apiVersion/kind vs FACTS
bash -n deploy/**/*.sh
python3 $SP/mdlinks.py <topic-dir>                         # relative links resolve (except root CURRICULUM.md/COMPUTE.md)
```
Solutions must run clean on CPU with no network; blanks must stop at the first exercise. Remove `_run_outputs/`, `__pycache__`,
`.pytest_cache`, `*.egg-info`, `.terraform*` before you finish.

## 5. Primer contract (builder A writes it; builder B links to its sections by number and title)
`PRIMER.md`, 500–900 lines: title, one-paragraph scope, **"The one-minute version"**, the numbered sections listed in §6 (keep the
numbers and titles; you may add subsections), then **"In a design review"** (a 2-minute walkthrough + 6 drill questions with
answers), **Glossary**, **Sources** (papers/official docs/repos), **Verify list** (dated product facts: versions, prices, SKUs).
Every formula gets a worked number and names the function in the core that computes it (`roofline.roofline.attainable()`).
Link, don't repeat, existing repo material: `00-foundations/gpu-capacity-planning/PRIMER.md` (sizing/TTFT/TPOT),
`00-foundations/transformers/docs/transformer-primer.md`, `01-hardware-gpu-fabric/gpu-primer/gpu-primer.md`,
`01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md` (scale-up vs scale-out, parallelism menu),
`04-inference-engine/{kv-cache,paged-attention,flash-attention}/`, `06-gateway/scaling-admission-cost/agentic-scaling-lab` (rate limits,
admission, cost), `07-application-agent-framework/` (agent workloads). Each primer has a section mapping concepts to GCP **and** to
non-GCP options (Colab/Kaggle/RunPod/Vast/Lambda/local kind) — link `COMPUTE.md` for prices/obtainability.

## 6. Per-layer plan (the curriculum). Builder A = PRIMER + README + core. Builder B = lab.

### 01 · roofline-and-fabric — "Reading the machine: rooflines, memory hierarchy, fabrics, and the cost of a token"
Primer sections: 1 Spec-sheet literacy (peak FLOPs by precision, dense vs sparse, HBM GB & TB/s, caches, links, TDP, boost clocks) ·
2 The roofline model (arithmetic intensity, ridge point, attainable FLOP/s; elementwise, reduction, GEMM `2mnk/((mk+kn+mn)·b)`) ·
3 LLM inference on the roofline (prefill vs decode intensity vs batch and context; KV reads; quantization moves bytes; MoE) ·
4 The memory hierarchy and why tiling/fusion win (register/SMEM/L2/HBM ladder; link 02 and flash-attention) ·
5 Fabrics quantitatively (NVLink/NVSwitch vs PCIe vs RDMA NICs; α-β model; TP all-reduce cost per token in and across nodes;
rail-optimized fat-tree, oversubscription, bisection bandwidth; GPUDirect; NUMA and `nvidia-smi topo -m`) ·
6 Storage and cold start (checkpoint bytes / tier bandwidth; parallel and streamed loading; caches) ·
7 Reliability at scale (per-GPU failure rates, cluster MTBF, Young/Daly checkpoint interval, replicas as failure domains) ·
8 The cost of a token ($/GPU-hr → $/M tokens; utilisation; rent vs own; link 06 PT break-even) ·
9 The accelerator landscape (Sep 2026 snapshot: NVIDIA T4/L4/A100/H100/H200/B200/GB200/GB300/RTX PRO 6000, AMD MI300X/MI325X/MI355X, TPU v5e/v6e/v7) ·
10 Getting hardware (GCP families and obtainability: on-demand/Spot/DWS/reservations; free and cheap non-GCP options).
Core `roofline`: `specs.py` (Device catalogue, dated, `(verify)`), `roofline.py`, `llm.py` (per-step FLOPs/bytes/time for prefill & decode
from a model config on a device; batch where decode turns compute-bound), `fabric.py` (links, α-β, ring all-reduce time, TP cost,
topologies, bisection/oversubscription), `storage.py` (load/cold-start), `reliability.py`, `cost.py`.
Core notebooks: `01_spec_sheets_and_the_roofline`, `02_llm_inference_on_the_roofline`, `03_fabrics_and_collective_cost`, `04_loading_reliability_and_cost`.
Lab `gpubench` ("measure the machine you have"): backends `numpy` (CPU, T0 — measuring your CPU's roofline is real and instructive) and
`torch` (CUDA, T1/T2, lazy import): GEMM throughput sweep by size/dtype (fp32/tf32/fp16/bf16; fp8 guarded), memory bandwidth (copy/scale/add/triad),
host↔device bandwidth (pinned vs pageable), GPU↔GPU P2P bandwidth (T2), safetensors/weights load throughput from disk; roofline fit (measured
ridge vs spec); `topo.py` parse `nvidia-smi topo -m`; `inventory.py` parse `nvidia-smi --query-gpu=... --format=csv`; `report.py` (JSON + markdown).
Lab notebooks: `01_measure_your_roofline`, `02_memory_bandwidth_and_transfers`, `03_multi_gpu_topology_and_p2p`, `04_weights_loading_and_cold_start`.
Deploy: `deploy/any-gpu/` (Docker `--gpus all` + plain-pip paths; Colab/Kaggle/RunPod/Vast/Lambda notes), `deploy/gcp/terraform/` (one Spot
`g2-standard-4` L4 VM, Deep Learning VM image family (verify), startup script runs the suite and uploads JSON to a GCS bucket, `max_run_duration`
auto-stop; variables to switch to e.g. `a2-highgpu-2g` for NVLink P2P).

### 02 · cuda-and-nccl — "The GPU software substrate: CUDA's execution model, collectives, and how a container gets a GPU"
Primer sections: 1 The stack from driver to framework (kernel module, libcuda, runtime, toolkit, libraries, framework wheels, Triton;
compatibility: minor-version compat, forward-compat package, compute capability, SASS vs PTX/JIT, "no kernel image" errors; sm table) ·
2 The execution model (grid/block/warp/SIMT, SMs, occupancy limits, latency hiding, divergence) ·
3 Memory access patterns (coalescing into 32 B sectors, shared memory tiling and bank conflicts, L2; tiled GEMM traffic; fusion) ·
4 Streams, launch overhead and CUDA Graphs (why engines capture decode graphs; torch.compile) ·
5 Collectives (semantics of broadcast/reduce/all-reduce/all-gather/reduce-scatter/all-to-all/send-recv; all-reduce = RS + AG; ring vs tree
vs NVLS/SHARP; α-β costs; algbw vs busbw as nccl-tests defines them; latency- vs bandwidth-bound message sizes; how TP/EP/PP use them;
NCCL topology detection, channels, protocols, env vars, debugging hangs) ·
6 How a container gets a GPU (device nodes, driver libs injected by the NVIDIA Container Toolkit via OCI hook/CDI, CUDA userland in the image;
link 03 device plugin) · 7 Sharing a GPU (MIG vs time-slicing vs MPS; isolation vs utilisation) ·
8 Health and observability (nvidia-smi, DCGM fields — why "GPU util" misleads vs SM active/occupancy/tensor/DRAM active; XID triage; ECC; throttling) ·
9 On GCP and elsewhere (GKE driver install, DLVM, GPUDirect-TCPX/TCPXO and RDMA with the gIB NCCL plugin, GKE MIG/time-sharing/MPS, DCGM in Cloud
Monitoring; containers on RunPod/Vast vs VMs on Lambda/GCP; Kaggle 2×T4 for free NCCL).
Core `gpusim` (numpy): `simt.py` (per-thread address patterns → sectors/transactions, bank conflicts, divergence cost), `occupancy.py`,
`tiling.py` (tiled GEMM byte counts; fused vs unfused softmax traffic), `compat.py` (driver ↔ CUDA ↔ compute-capability rules engine that
explains errors; version table dated `(verify)`), `collectives.py` (simulated ranks: ring/tree all-reduce, RS, AG, broadcast, all-to-all with
step traces; α-β cost; algbw/busbw), `sharing.py` (MIG profile placement packer for A100/H100 (verify profiles), time-slicing latency model),
`health.py` (DCGM/XID triage rules).
Core notebooks: `01_simt_warps_and_memory`, `02_tiling_fusion_and_occupancy`, `03_collectives_from_scratch`, `04_compatibility_and_containers`, `05_sharing_and_health`.
Lab `gpurt`: `kernels/` real CUDA kernels written with **Numba** (vector add, reduction, naive vs tiled matmul, transpose naive vs SMEM-tiled,
fused softmax) that run on CPU via `NUMBA_ENABLE_CUDASIM=1` (T0, correctness) and on a real GPU (T1, timing) — same source; optional Triton versions
(lazy, T1); `dist/` torch.distributed collectives benchmark (`gloo` on CPU = T0 semantics; `nccl` = T1/T2) computing algbw/busbw exactly like
nccl-tests + α-β fit; `nccltests.py` parse `all_reduce_perf` output; `launch.py` CUDA Graphs vs eager (T1, torch); `container.py` explain how this
container sees its GPU (/dev/nvidia*, libcuda mount from /proc/self/mountinfo, driver/runtime versions, compute capability → compat verdict);
`dcgm.py` parse DCGM-exporter Prometheus text → derived signals and alert rules. Torch-dependent code imports lazily; tests skip it.
Lab notebooks: `01_kernels_in_the_simulator`, `02_memory_bound_kernels_on_a_real_gpu`, `03_collectives_with_torch_distributed`,
`04_busbw_and_the_alpha_beta_fit`, `05_how_a_container_sees_a_gpu`, `06_gpu_sharing_and_dcgm_on_gke`.
Deploy: `deploy/any-gpu/` (docker commands; build+run nccl-tests; a Kaggle 2×T4 recipe), `deploy/gke/` manifests (nvidia-smi smoke Job,
CUDA sample Job, 2-GPU nccl-tests Job on `g2-standard-24`, MIG and time-sharing examples), `deploy/gcp/terraform/` (zonal GKE Standard, CPU system pool,
L4 Spot pool autoscaling 0→N with driver auto-install, optional time-sharing pool, optional MIG pool (A100) off by default, DCGM + managed Prometheus).

### 03 · gpu-scheduling — "Kubernetes for GPUs: how a GPU becomes schedulable, and how to place, share, queue and scale it"
Primer sections: 1 What Kubernetes sees (capacity/allocatable, extended resources — integer, no overcommit, requests==limits; device plugin API
Register/ListAndWatch/Allocate; node labels via NFD/GFD and GKE labels; GPU taints; DRA: ResourceClaim/DeviceClass/ResourceSlice, CEL selectors) ·
2 GPU Operator vs managed drivers · 3 The scheduling cycle (queue sort → filter → score (LeastAllocated vs MostAllocated bin-packing) → reserve/permit →
bind; fragmentation; priority and preemption) · 4 Gangs (all-or-nothing, deadlock of partial placement; Kueue suspend/admit; JobSet; LWS for multi-host
inference; coscheduling/Volcano alternatives) · 5 Topology-aware placement (host/NVLink domain, sub-block, block; Kueue TAS; compact placement) ·
6 Queues, quotas and multi-tenancy with Kueue (ResourceFlavor, ClusterQueue, LocalQueue, cohorts, borrowing/lending limits, preemption, fair sharing,
WorkloadPriorityClass) · 7 Getting capacity (cluster autoscaler, node auto-provisioning, scale-from-zero, Spot, reservations, DWS flex-start queued
provisioning via ProvisioningRequest, calendar mode, custom ComputeClass fallbacks, Autopilot) · 8 Startup latency (image streaming, secondary boot
disks, weights via GCS FUSE CSI/Hyperdisk ML/model streamers, startup probes) · 9 Sharing GPUs at the cluster level (MIG/time-sharing/MPS on GKE; DRA) ·
10 Learning locally (kind + fake GPU capacity or KWOK/fake-gpu-operator: what is real and what is simulated).
Core `gpusched` (pure Python): `cluster.py` (nodes with GPUs, labels, taints, topology path; pods with requests, tolerations, selectors, priority,
gang id, topology constraint), `plugins.py` (filters/scorers/preemption), `scheduler.py` (cycle + fragmentation metric), `gang.py` (all-or-nothing +
smallest-domain topology fit), `quota.py` (Kueue-like ClusterQueues with cohort borrowing/lending and reclaim preemption), `autoscaler.py` (pools
min/max, provisioning delay, Spot preemption, atomic queued provisioning, scale-down), `deviceplugin.py` (ListAndWatch/Allocate, unhealthy devices).
Core notebooks: `01_how_kubernetes_sees_a_gpu`, `02_filter_score_and_fragmentation`, `03_gangs_and_topology`, `04_queues_quotas_and_preemption`, `05_autoscaling_and_obtainability`.
Lab `k8sgpu`: `manifests.py` (typed builders → YAML for Job, JobSet, LWS, Kueue objects, DRA ResourceClaimTemplate, GKE ComputeClass), `lint.py`
(GPU pod-spec linter: gpu requests==limits, GPU taint toleration, node selector, startup probe for model load, no GPU on sidecars, CPU/mem requests),
`pending.py` ("why is my pod Pending?" from `kubectl get pod -o json` + events; fixtures), `capacity.py` (capacity-type chooser: cost vs obtainability).
Deploy: `deploy/kind/` (runnable on a laptop with Docker: kind config, script that creates the cluster, advertises fake `nvidia.com/gpu` capacity on
workers via a node-status patch and adds GKE-style topology labels; optional KWOK path for 100s of fake GPU nodes; installs Kueue v0.19.6, JobSet, LWS;
example workloads: single-GPU Job, 4-GPU gang JobSet with TAS annotations, LWS group, priority/preemption, cohort borrowing), `deploy/gcp/terraform/`
(zonal GKE Standard, system pool, L4 Spot GPU pool 0→N with driver auto-install, optional flex-start queued-provisioning pool, GCS FUSE CSI,
image streaming, managed Prometheus) + `deploy/gke/` manifests (ComputeClass fallbacks, Kueue ProvisioningRequest admission check for DWS, GCS FUSE
weights volume). Lab notebooks: `01_manifests_and_the_linter`, `02_kind_with_fake_gpus_and_kueue` (drives kubectl if a cluster is reachable; else
predicts outcomes with a bundled mini-simulator and prints the commands), `03_why_is_my_pod_pending`, `04_gke_pools_dws_and_computeclasses` (T3 walkthrough; offline = plan/inspect).

### 04 · serving-engine — "Inside an inference engine: the step loop, scheduling, batching, caching, speculation and quantization"
Primer sections: 1 Anatomy of an engine (API server → engine core: scheduler + KV cache manager + executor → detokenizer; what one step is) ·
2 Continuous batching (iteration-level scheduling; token budget `max_num_batched_tokens`, `max_num_seqs`; request states; policies) ·
3 Chunked prefill and prefill/decode interference (stall-free batching; Sarathi-Serve) · 4 KV cache management revisited (block allocation,
watermarks, preemption by recompute vs swap — link existing kv-cache/paged-attention primers) · 5 Prefix caching (block hashing with parent hash,
refcounts, LRU of free cached blocks; radix-tree alternative; hit accounting; designing agent prompts for cache hits) · 6 Sampling and structured
output (temperature/top-k/top-p/min-p, penalties, seeds, logprobs, stop; grammar/FSM token masks, JSON schema) · 7 Speculative decoding (draft/verify,
exact acceptance min(1,p/q) + residual resampling preserves the target distribution, expected tokens (1−α^(k+1))/(1−α), cost model; draft model,
n-gram/prompt lookup, EAGLE, MTP) · 8 Quantization (weight-only INT8/INT4 GPTQ/AWQ vs FP8 W8A8; KV FP8; scales per-tensor/channel/group; what each
buys for prefill vs decode; accuracy checks) · 9 Parallelism inside the engine (TP column/row-parallel with 2 all-reduces/layer, PP, EP, DP) ·
10 Multi-LoRA serving · 11 Measuring an engine (TTFT, ITL/TPOT, E2E, throughput, goodput; open vs closed loop; warm-up; length distributions; the
knobs and what each trades) · 12 Engines and where to run them (vLLM, SGLang, TensorRT-LLM, llama.cpp; Colab/any GPU box/serverless; GCP: GKE, Cloud
Run GPU, Vertex AI; TPUs).
Core `minengine` (numpy "nano-vLLM"): `model.py` (tiny deterministic decoder-only transformer, byte-level vocab, GQA, forward over paged KV via block
tables; test: paged == dense), `kv.py` (block pool, refcounts, hash-based prefix cache with LRU eviction of free blocks), `scheduler.py` (continuous
batching, token budget, chunked prefill, FCFS, preemption by recompute), `sampler.py` (greedy/temperature/top-k/top-p/min-p, seeded, logprobs; FSM
mask hook), `spec.py` (draft model + exact rejection sampling; statistical test that outputs follow the target distribution), `quant.py` (int8
per-channel, int4 group-wise, fp8-e4m3 emulation; error metrics), `engine.py` (`add_request`, `step`, `generate`), `perf.py` (roofline-style step-time
model → simulated TTFT/ITL under Poisson load; compare knobs).
Core notebooks: `01_the_step_loop_and_continuous_batching`, `02_chunked_prefill_and_the_token_budget`, `03_prefix_caching`, `04_sampling_and_structured_output`,
`05_speculative_decoding`, `06_quantization`.
Lab `servelab` (real vLLM): `bench/` (async OpenAI-compatible load generator, open-loop Poisson + closed-loop, streaming TTFT/ITL capture, synthetic
length distributions and a shared-prefix agentic workload, goodput vs SLO, reports), `metrics.py` (scrape/parse vLLM `/metrics` → KV usage, queue,
prefix hit rate, histogram quantiles), `sizing.py` (HF `config.json` + GPU memory + `gpu_memory_utilization` + `max_model_len` → KV blocks and max
concurrency; bundled sample configs), `tune.py` (sweep flags against an SLO with a pluggable backend), `fakeserver.py` (OpenAI-compatible streaming
server that emulates vLLM timing, prefix-cache hits and vLLM-named metrics — the T0 target; note `llm-d-inference-sim` as the upstream equivalent).
Deploy: `deploy/any-gpu/` (`docker run vllm/vllm-openai` with a small model, flags explained; Colab/Kaggle T4 recipe with `dtype=half`; RunPod/Vast notes),
`deploy/gcp/cloud-run/` (Terraform `google_cloud_run_v2_service` with L4, scale-to-zero, weights from GCS or HF token in Secret Manager; `gcloud run
deploy` equivalent), `deploy/gcp/gke/` (vLLM Deployment on an L4 node pool, PodMonitoring for `/metrics`; link 05 for autoscaling).
Lab notebooks: `01_size_before_you_serve`, `02_serve_and_measure`, `03_knobs_and_tradeoffs`, `04_prefix_caching_for_agents`, `05_speculation_and_quantization_in_vllm`, `06_deploy_on_cloud_run_gpu`.

### 05 · serving-orchestration — "Orchestrating a fleet of engines: routing, autoscaling, disaggregation and KV-cache tiers"
Primer sections: 1 Why a layer above the engine (replicas are stateful caches; why round-robin/least-connections fails for LLMs; the three decisions:
which replica, how many, how to split the work) · 2 Routing signals and algorithms (queue depth, running, KV utilisation, prefix affinity approximate
vs precise KV-event indexes, LoRA/session affinity; round-robin, least-outstanding, power-of-two choices, consistent hashing with bounded loads,
weighted multi-scorer filters→scorers→picker as in the llm-d EPP; the locality-vs-load tension and hot prefixes) · 3 Flow control and priorities
(router-side queues, InferenceObjective priority, saturation detection, shedding; link 06 admission control) · 4 Autoscaling (signals: waiting
requests, KV usage, TTFT/ITL SLO attainment — not GPU util; HPA algorithm `ceil(current × metric/target)`, 10% tolerance, stabilization windows,
policies; cold start anatomy; scale-to-zero; buffers; KEDA; SLA planners) · 5 Prefill/decode disaggregation (when it helps and when it doesn't; P:D
ratio sizing; KV transfer bytes/bandwidth and overlap; NIXL; xPyD; llm-d, Dynamo, vLLM KV connectors) · 6 KV cache beyond HBM (tiers
HBM→DRAM→NVMe→remote, offload/onload math, LMCache/Mooncake, cross-replica sharing, KV working set of multi-turn agent sessions) · 7 Multi-model,
multi-LoRA and model routing (pools per base model, adapter affinity, model rewrite/canary) · 8 Large MoE topologies (wide-EP) in brief ·
9 The Kubernetes-native stack, Sep 2026 (Gateway API Inference Extension InferencePool v1, llm-d Router/EPP, InferenceObjective, GKE Inference Gateway,
llm-d well-lit paths, NVIDIA Dynamo 1.x, Ray Serve LLM, KServe — how they relate) · 10 Where to run it (laptop: sims + kind + llm-d-inference-sim;
any GPU box; GCP: GKE Inference Gateway + Managed Prometheus-driven HPA).
Core `fleetsim` (pure Python discrete-event simulator): `workload.py` (Poisson/bursty arrivals; chat, RAG long-prompt, multi-turn agentic sessions with
shared system prompts and growing histories; LoRA ids), `replica.py` (engine model: queue, token-budget batching, block-hash prefix cache with LRU, KV
capacity, step-time model), `routers.py` (RoundRobin, LeastOutstanding, PowerOfTwo, PrefixHash, ConsistentHashBoundedLoad, WeightedScorer with
`prefix-cache-scorer`/`queue-scorer`/`kv-cache-utilization-scorer` + LoRA-affinity filter), `autoscale.py` (HPA algorithm with tolerance/stabilization/
policies, cold start, scale-to-zero, queue-based scaler), `disagg.py` (prefill pool + decode pool + KV transfer; P:D ratio search), `kvtier.py`,
`metrics.py` (TTFT/ITL/E2E percentiles, goodput, hit rate, load imbalance).
Core notebooks: `01_why_llm_load_balancing_is_different`, `02_cache_aware_routing_and_the_load_tradeoff`, `03_autoscaling_on_the_right_signal`,
`04_prefill_decode_disaggregation`, `05_kv_cache_tiers_and_agent_sessions`.
Lab `igwlab`: `router/` a real async HTTP router (aiohttp or starlette+httpx) in front of N OpenAI-compatible backends: scrapes each backend's `/metrics`
(vLLM names), keeps an approximate prefix index (block hashes per endpoint, LRU/TTL), filters→weighted scorers→picker configured by a YAML modelled on
`EndpointPickerConfig`, streams responses, exposes its own metrics — a readable re-implementation of the EPP decision logic (not ext-proc);
`fakebackend.py` (small OpenAI-compatible backend emulating prefix-cache TTFT benefit + vLLM metrics, for T0), `autoscale.py` (HPA recommender from
scraped metrics + manifest generator), `bench.py` (shared-prefix agentic load; compare routers). Deploy: `deploy/local/` (docker compose: 3 backends
— `llm-d-inference-sim` image or the bundled fake — + router + Prometheus), `deploy/kind/` (llm-d Router **standalone mode** via its Helm chart with
Envoy + llm-d-inference-sim replicas; optional Gateway mode; pin versions with `(verify)`), `deploy/gcp/terraform/` (zonal GKE Standard with
`gateway_api_config`, proxy-only subnet, L4 Spot pool 0→N, managed Prometheus) + `deploy/gke/` manifests (vLLM Deployment, InferencePool v1, EPP
install, Gateway `gke-l7-regional-external-managed` + HTTPRoute → InferencePool, InferenceObjective priorities, HPA on a Prometheus metric).
Lab notebooks: `01_router_in_process`, `02_scorer_weights_and_hot_prefixes`, `03_autoscaling_recommender`, `04_local_stack_with_llm_d`, `05_gke_inference_gateway`.

## 7. Report (your final message = data for the orchestrator; ≤ 250 words, no prose padding)
```
LAYER: <nn> <A|B|review>   STATUS: done|partial
FILES: <n files>, <dirs created>
TESTS: <pytest one-liner>   NOTEBOOKS: <solutions x/y passed; blanks x/y stopped>   TF: <result|n/a>   K8S: <result|n/a>   LINKS: <result>
DEVIATIONS: <anything that differs from SPEC and why>
OPEN: <VERIFY items, things you could not validate here (GPU/Docker/cloud), follow-ups>
```
