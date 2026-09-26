# SPEC — filling layers 01–05 of full-stack-agentic-engineer (and, from 2026-09-26, four more topics: §6b)

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
| 00 | `00-foundations/rl-and-thinking-models` | `rl-core` (`rlcore`) | `thinking-lab` (`thinklab`) |
| 00 | `00-foundations/mixture-of-experts` | `moe-core` (`moecore`) | `moe-lab` (`moelab`) |
| 04 | `04-inference-engine/quantization` | `quant-core` (`quantcore`) | `quant-lab` (`quantlab`) |
| 07 | `07-application-agent-framework/sandboxed-execution` | `sandbox-core` (`sandboxcore`) | `sandbox-lab` (`sandboxlab`) |
Root `CURRICULUM.md` and `COMPUTE.md` exist (the integrator updates them); you may link to them (from a topic dir:
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

## 6b. Four more topics (September 2026): the same contract, one block each

Everything in §0–§5 applies unchanged. Differences for these four builds:
- **Existing material is a hard prerequisite, not a competitor.** Each block names the sections and functions already in the repo
  that the new primer must cite and reuse; duplicating them is a *major* review finding. Cores stay standalone packages (no cross-lab
  imports), but where a formula already has a home (`roofline.llm.experts_touched`, `capacity.py`, `minengine.quant`, `servelab.sizing`)
  the new code must reproduce its numbers in a test that says so.
- **Torch is installed here (CPU, `torch 2.14`)**, so torch code paths *can* be validated on CPU at T0 in this environment. Still import it
  lazily and skip cleanly when absent — every T0 notebook must also run with torch missing (the numpy/stdlib path carries the concept).
  Never `pip install torch` (it is already here); never install vllm/triton/flash-attn (no GPU).
- **No network in tests or solutions.** Model weights are never downloaded at T0: tiny models are trained in the notebook or bundled as
  small arrays; real checkpoints (0.5–2B) are T1 paths with `(verify)` model ids.
- **Labs in `00-foundations` have no Terraform** (GCP is optional everywhere): `deploy/any-gpu/` plus a documented pointer to an existing
  lab's GCP deploy (the 04 serving lab's Cloud Run/GKE, the 02 lab's `l4x2` pool). The 07 sandbox lab *does* carry Terraform (GKE Sandbox).
- **Docker and kind are not runnable here**: notebooks that need them detect absence, print the exact commands, and use bundled sample
  output labelled "sample output in the documented format (illustrative)". Manifests are validated offline (`kubernetes-validate`).
- Facts: `$SP/FACTS.md` plus the per-topic research file `$SP/facts-<topic>.md` (written by the research agent before the builders start);
  upstream sources are cloned under `$SP/ref/` (read them instead of guessing; the docs web sites are blocked, `git clone` and
  `raw.githubusercontent.com` work). Anything not in those files is `(verify)`.

### 00 · rl-and-thinking-models — "Reinforcement learning and thinking models: how post-training teaches a model to reason, and what that does to serving"
Primer sections: 1 From pretraining to post-training (pretrain → SFT → preference/RL; what each stage changes; RL for LLMs as
"sample, score, reweight"; where the compute goes — rollouts dominate; the rollout generator is an inference engine, link 04 serving-engine) ·
2 Policy gradients over token sequences (the model as a policy; completions as trajectories; REINFORCE, variance, baselines and advantages;
the KL penalty to a reference model and why it exists; reward hacking and length bias — every claim worked on the core's toy tasks) ·
3 Learning from preferences (Bradley–Terry reward models; RLHF with PPO in brief — clipped ratio, value model, GAE; DPO: the closed form
that turns the KL-regularised objective into a classification loss, the implicit reward β·log(π/π_ref), what it loses; IPO/KTO/ORPO one line each) ·
4 RL with verifiable rewards and GRPO (verifiers for math/code/format; GRPO: G samples per prompt, group-normalised advantages, no critic,
the clipped objective, the k3 KL estimator; DAPO/Dr.GRPO fixes — clip-higher, token-level loss, dropping the std normalisation, length bias;
compute: G× generation, the reference and old-policy copies; TRL's `GRPOTrainer` loss options as the concrete reference) ·
5 Thinking models (long chain-of-thought as a behaviour learned under RLVR; the DeepSeek-R1 recipe — R1-Zero → cold start → RL → rejection-sampling
SFT → RL; length growth and "aha"; hybrid thinking modes and chat-template switches (Qwen3 `enable_thinking`), thinking budgets and
`reasoning_effort`; distillation of traces into small models; a thinking token is billed as an output token) ·
6 Test-time compute (sequential: think longer; parallel: best-of-n with a verifier or reward model, majority vote / self-consistency;
pass@k vs pass^k and the unbiased pass@k estimator 1 − C(n−c,k)/C(n,k); search with process reward models in brief; compute-optimal
allocation — when a smaller model with more samples wins; diminishing returns and over-thinking) ·
7 What thinking does to serving (output-heavy → decode-dominant; heavy-tailed thinking-length distributions; KV working set per request
grows 5–20×; ITL is the binding SLO, TTFT less so; the capacity primer's formulas with long outputs — cite `00-foundations/gpu-capacity-planning/PRIMER.md`
and `capacity.py`; thinking is dropped from history by chat templates → prefix-cache implications (link 04 §5); streaming `reasoning_content`
and vLLM's `--reasoning-parser`; budget enforcement (max_tokens vs budget forcing); cost per *correct* answer; routing by effort at the gateway,
link 06; speculative decoding on long outputs, link 04 §7) ·
8 The RL training stack in brief (rollouts with vLLM/SGLang inside the trainer, weight sync, async/off-policy RL; verl/TRL/OpenRLHF; why
serving skills transfer; agentic RL — multi-turn with tools, environments and sandboxes, link 07 sandboxed-execution; reward design and evals, link 07.2) ·
9 Where to run it (T0 toy policies and the workload model; Colab/Kaggle T4: tiny-transformer GRPO in torch and a 0.6B thinking model in vLLM;
a rented 24 GB GPU: 1.7–4B thinking models; GCP via the 04 lab's deploys with a thinking model; link `COMPUTE.md`).
Core `rlcore` (numpy): `tasks.py` — two verifiable toy environments: `SeqTask` (token-level: emit a short sequence over a tiny vocab that a verifier
checks, e.g. balanced brackets or a sorted run; deterministic verifier) and `ThinkTask` (the policy picks a thinking length L then answers; P(correct | L)
= 1 − e0·(1−q)^L so longer thinking really helps, with a per-token cost — the toy that shows RL discovering longer thinking and budgets vs accuracy);
`policy.py` (small softmax policies with explicit reference copies, seeded sampling, per-token logprobs, closed-form gradients); `pg.py` (REINFORCE,
baselines, advantages, KL penalty, entropy); `pref.py` (Bradley–Terry fit; DPO loss and gradient; implicit reward; a length-biased "annotator" to show
reward hacking); `grpo.py` (group sampling, group-normalised advantages, clipped objective, k3 KL, DAPO-style flags); `ttc.py` (best-of-n, majority
vote, unbiased pass@k, sequential thinking-length curves, compute-optimal n×L under a token budget); `workload.py` (thinking-length distributions →
tokens per request, KV working set, decode-bound step time with a roofline-style model, cost per correct answer; reproduces the capacity primer's
numbers for a no-thinking baseline in a test).
Core notebooks: `01_policy_gradients_on_a_toy_task`, `02_preferences_reward_models_and_dpo`, `03_grpo_with_verifiable_rewards`, `04_test_time_compute`,
`05_thinking_models_and_the_serving_workload`.
Lab `thinklab`: `tinyrl/` (torch, lazy: a tiny decoder-only transformer trained from scratch on a scratchpad arithmetic task — SFT warm-up on a mix of
with/without-scratchpad demonstrations, then GRPO with the verifier; reward and response-length curves; must finish in < 10 min on a laptop CPU, and
the notebook must show the curves it actually got — bundled sample curves labelled illustrative are the no-torch fallback); `thinking/` (an
OpenAI-compatible client: thinking on/off via chat-template kwargs, `reasoning_content` parsing, budget enforcement, best-of-n and majority vote,
a built-in generated eval set of verifiable arithmetic/logic problems — no download); `fakeserver.py` (a T0 OpenAI-compatible server that emits
`reasoning_content` and heavy-tailed output lengths with vLLM-named metrics, timing simulated from a roofline model — say so); `workload.py`
(measure output-length distributions from a server and derive the serving shape; drive open-loop load with long outputs; compare no-thinking vs
thinking vs budget); `parsers.py` (reasoning-block parsers for the common templates); `report.py`.
Lab notebooks: `01_grpo_on_a_tiny_transformer` (T0 with torch; T1 faster), `02_a_thinking_model_on_one_gpu` (T1: Qwen3-0.6B/1.7B in vLLM with
`--reasoning-parser`, thinking on vs off, accuracy vs budget; T0: bundled recorded outputs, illustrative), `03_test_time_compute_for_real` (T1: best-of-n
and majority vote on the real model, cost per correct answer; T0: the bundled outputs), `04_serving_thinking_models` (T0 fake server; T1 real vLLM:
ITL, KV usage, preemptions under long outputs; `max_model_len` and budget knobs; prefix caching across turns when thinking is dropped),
`05_rl_rollouts_with_an_engine` (T1: vLLM as the rollout generator for one GRPO step on a 0.5B model with TRL optional; T0: the rollout bookkeeping on the toy task).
Deploy: `deploy/any-gpu/` (docker run vLLM with a thinking model and `--reasoning-parser`; Colab/Kaggle T4 recipe with `--dtype half`; RunPod/Vast
notes) and a pointer to `04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/` with a thinking model — no new Terraform.
Must cite/reuse: capacity primer + `capacity.py` (§7), serving-engine PRIMER §5 (prefix caching), §7 (speculation), §11 (measuring), vllm-internals
§9; `06-gateway/scaling-admission-cost` (cost per conversation, routing); `07-.../gcp-agent-platform-lab` notebook 08 (evals).

### 00 · mixture-of-experts — "Mixture-of-experts models: the router, the experts, and what sparsity does to serving"
Primer sections: 1 Why sparsity (parameters vs compute per token; the scaling-law view; memory by total, FLOPs by active — cite the capacity primer's
Mistral Large 3 section; what MoE does not change: attention and the KV cache) · 2 The MoE layer (E expert MLPs; the router — linear scores, softmax
or sigmoid, top-k, renormalisation, weighted combine; shared experts; fine-grained experts (DeepSeek-V3: 256 routed + 1 shared, top-8) vs coarse
(Mixtral 8, top-2); expert size and granularity; counting total and active parameters from a config, worked for Mixtral-8x7B, DeepSeek-V3,
Qwen3-30B-A3B, gpt-oss-120b and Llama 4 Maverick — read the model code under `$SP/ref/transformers`) · 3 Routing and load balance (router collapse;
the Switch auxiliary loss α·E·Σ f_e·P_e; router z-loss; capacity factor and token dropping vs dropless (MegaBlocks); auxiliary-loss-free balancing
with per-expert bias (DeepSeek-V3) and its sequence-level complement; expert-choice routing; node/group-limited routing for EP; balance at inference:
hot experts and domain skew) · 4 Training MoE in brief (EP all-to-all in forward and backward; loss terms; upcycling from dense; MoE + MLA vs MoE + GQA
— the attention side, link the transformer primer §9) · 5 MoE at inference: which experts a step touches (**cite and reuse** roofline PRIMER §3.6 and
`roofline.llm.experts_touched` — reproduce its Mixtral/Qwen3 table in a test; skewed (Zipf) vs uniform routing; the decode crossover batch; why MoE
wants big batches; KV unchanged so the KV/weights ratio shifts; prefix caching unchanged) · 6 Running MoE on GPUs (fused MoE kernels: sort/permute
tokens by expert, grouped GEMM, vLLM's Triton `fused_moe` and its tuned configs; expert parallelism: dispatch/combine all-to-alls with
bytes = tokens × k × hidden × bytes per GPU — link 02 PRIMER §5 and DeepEP; TP vs EP for experts and the hybrid; data-parallel attention + EP (wide-EP,
link 05 PRIMER §8); expert offloading to CPU for small GPUs (vLLM `--cpu-offload-gb`, llama.cpp); quantized experts (MXFP4 in gpt-oss; link 04 quantization)) ·
7 Sizing and cost (memory by total + KV; prefill FLOPs by active; decode step time by bytes streamed at the batch; GPUs and EP degree for a target;
cost per token vs a dense model of similar quality (verify); a table of MoE families with E, k, shared, total/active (verify, dated)) ·
8 In a design review: failure modes (MoE at batch 1; EP across a slow fabric; hot experts; MoE on one 24 GB GPU; long context where KV dominates;
quantizing experts; dense vs MoE for a workload) · 9 Where to run it (T0 numpy; Colab/Kaggle T4 or a rented 24 GB GPU: a small open MoE with router
hooks — candidates `(verify)`: OLMoE-1B-7B, Qwen1.5-MoE-A2.7B INT4, Qwen3-30B-A3B INT4 on 24 GB, granite-3 MoE; Kaggle 2×T4 for EP=2; GCP: the 02 lab's
`l4x2` pool with `--enable-expert-parallel`; link `COMPUTE.md`).
Core `moecore` (numpy): `moe.py` (the layer: experts, router softmax/sigmoid, top-k, renormalisation, shared experts, combine; dense equivalence
when E=1; total/active parameter counting from a config), `routing.py` (aux loss, z-loss, capacity factor and token dropping, dropless, bias-based
balancing, expert-choice; utilisation stats), `train.py` (a small numpy trainer with manual gradients for linear/one-hidden-layer experts and a linear
router on a toy task — collapse without balancing, balance with it; keep it small and fast), `touched.py` (experts touched: closed form and Monte Carlo
with Zipf skew; weight-stream bytes per step vs batch; decode crossover with a small roofline helper), `ep.py` (EP dispatch/combine across ranks:
bytes, imbalance, the slowest rank sets the step; TP vs EP; the wide-EP layout), `sizing.py` (deployment sizing and cost; `MODELS` catalogue with
E, k, shared, hidden, total/active, verify-marked).
Core notebooks: `01_the_moe_layer`, `02_routing_and_load_balance`, `03_which_experts_a_batch_touches`, `04_expert_parallelism_and_all_to_all`, `05_sizing_and_cost`.
Lab `moelab`: `tinymoe/` (torch, lazy: a tiny MoE transformer trained on a toy task; router collapse without balancing; expert specialisation; CPU minutes
at T0 with torch, bundled curves labelled illustrative without), `hooks.py` (router-logit hooks for HF MoE models → per-token expert selection,
utilisation histograms, hot experts by domain), `stream.py` (decode step time vs batch for a MoE vs a dense model with vLLM; compared with
`moecore.touched`), `ep.py` (`--enable-expert-parallel` vs TP on 2 GPUs; parse throughput), `offload.py` (what fits on 16/24 GB with CPU offload and
INT4 experts; measured cost), `fixtures/` (bundled sample router traces and benchmark outputs, labelled illustrative), `report.py`.
Lab notebooks: `01_a_tiny_moe_in_torch` (T0 with torch), `02_watch_the_router` (T1; T0 bundled traces), `03_batch_vs_weight_stream` (T1; T0 simulated),
`04_expert_parallelism_on_two_gpus` (T2 Kaggle 2×T4; T0 simulated), `05_moe_on_a_small_gpu` (T1 offloading and INT4 experts; T0 sizing).
Deploy: `deploy/any-gpu/` (docker run vLLM with a small MoE, EP=2 on two GPUs, offload flags; Colab/Kaggle recipes), `deploy/gke/` (a vLLM Deployment
with EP=2 on the 02 lab's `l4x2` pool — manifests only, link `02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/`; no new Terraform).
Must cite/reuse: transformer primer §9 row, capacity primer, roofline PRIMER §3.6 + `roofline.llm`, 02 PRIMER §5 (all-to-all), 04 serving-engine
PRIMER §9 (EP), 05 PRIMER §8 (wide-EP), the open-weight primer §4 (families).

### 04 · quantization — "Quantization for inference: number formats, calibration, kernels and the accuracy you pay"
Primer sections: 1 Why quantize, and what it can and cannot speed up (the roofline argument — weight bytes per decode step, KV bytes per token,
tensor-core throughput by precision, cite 01 PRIMER §1–3; weight-only helps decode, W8A8/FP8 helps prefill too; the accuracy budget; where
serving-engine PRIMER §8 and `minengine.quant` stop and this primer starts — link, do not repeat) · 2 Number formats (INT8/INT4 with scale and
zero point, symmetric vs asymmetric; FP8 E4M3 vs E5M2 — range vs precision, subnormals; FP4 E2M1; block formats MXFP4 (E8M0 shared scale per 32) and
NVFP4 (E4M3 scale per 16 + an FP32 tensor scale); BF16/FP16 baselines; the uniform-quantization error model and ~6 dB per bit; outliers and why
they dominate) · 3 Granularity and the bits-per-weight budget (per-tensor / channel / group (32, 128) / token / block; static vs dynamic activation
scales; bpw accounting with scales and zero points — INT4-g128 ≈ 4.16 bpw) · 4 Weight-only post-training quantization (RTN; GPTQ — the OBS
error-compensation update column by column with a damped Hessian from calibration activations; AWQ — activation-aware per-channel scaling and the
search; SmoothQuant — migrating activation outliers into weights; what calibration data does and does not do; rotations (QuaRot/SpinQuant) in brief;
kernels: dequantize-on-the-fly (Marlin, Machete, ExLlama) and why W4A16 is fast for decode and slower for prefill) · 5 Weight-and-activation quantization
(INT8 W8A8 with SmoothQuant; FP8 W8A8 per-tensor vs per-block scales (DeepSeek-V3's 128×128) and dynamic per-token activation scales; FP4 W4A4 on
Blackwell with NVFP4; accumulators; which layers stay in high precision — embeddings, LM head, norms, attention softmax — and why) ·
6 KV-cache quantization (FP8 E4M3/E5M2 KV, per-token/per-head scales; INT4 KV with per-channel keys and per-token values (KIVI); what it buys — 2×
sessions, faster long-context decode — and its kernel conditions, cite the FlashAttention deep dive §9 and vllm-internals §6.3; prefix caching with
quantized KV) · 7 Quantization-aware training and QLoRA in brief (fake quantization with a straight-through estimator; QAT for FP8/INT4; QLoRA's NF4
base + LoRA is a training recipe, not a serving format; distillation to recover accuracy; link 00 rl-and-thinking-models §1) · 8 Measuring the accuracy
you pay (perplexity vs task accuracy vs logit KL / argmax agreement; lm-evaluation-harness; long-generation evals for thinking models; failure modes:
small models, MoE experts, long context, multilingual, tool calling; setting a budget) · 9 Producing a checkpoint (llm-compressor recipes — FP8 dynamic,
W4A16 GPTQ/AWQ, W8A8 INT8, NVFP4 — read `$SP/ref/llm-compressor`; the compressed-tensors config in `config.json` and safetensors layout, ignored modules;
GPTQModel/AutoAWQ; pre-quantized hubs; vLLM's loader and kernel selection — `--quantization`, auto-detection, Marlin/Machete/cutlass FP8, minimum
compute capability per kernel: T4 sm75, A100 sm80, L4/4090 sm89, H100 sm90, B200 sm100; `--kv-cache-dtype`; llama.cpp GGUF k-quants as the CPU/consumer path) ·
10 Choosing a scheme (a decision table by GPU generation, model size, workload (prefill- vs decode-heavy), accuracy budget and memory target; cost per
token, cite 01 PRIMER §8; consistency with the serving-engine §8 knob table).
Core `quantcore` (numpy): `formats.py` (encode/decode INT8/INT4 sym/asym, FP8 E4M3/E5M2, FP4 E2M1, MXFP4 and NVFP4 block formats; representable grids;
bits per weight), `granularity.py` (per-tensor/channel/group/token/block scales; error metrics: relative error, SQNR, argmax agreement), `gptq.py`
(Hessian from calibration activations, damping, OBS column updates; RTN baseline), `awq.py` (activation-aware scale search), `smoothquant.py`, `w8a8.py`
(fake-quant matmul with static/dynamic activation scales), `kvquant.py` (FP8 and KIVI-style KV; attention-output error), `tinymodel.py` (a small
deterministic transformer or MLP stack with synthetic outlier-heavy activations so calibration matters — no download), `cost.py` (roofline-style speed
and memory model per scheme per GPU; the decision table; reproduces `minengine.quant`/serving-engine §8 numbers in a test), `eval.py` (a built-in eval:
logit KL, argmax agreement, toy-task accuracy).
Core notebooks: `01_number_formats_and_error`, `02_granularity_and_outliers`, `03_gptq_awq_and_smoothquant_from_scratch`,
`04_activation_and_kv_cache_quantization`, `05_choosing_a_scheme`.
Lab `quantlab`: `compress.py` (real checkpoints with llm-compressor or GPTQModel when installed (lazy) — FP8 dynamic and W4A16 on a tiny HF model at T1;
at T0 the same recipes applied with the lab's own numpy/torch code to a bundled tiny model, written as a compressed-tensors-style safetensors file that the
lab's loader reads back and checks), `serve.py` (vLLM flags per scheme and GPU generation; recipes), `bench.py` (TTFT/ITL/throughput per scheme against
a fake or real server — the bundled fake server models bytes-per-weight speedups, labelled simulated), `evalharness.py` (an lm-eval CLI wrapper plus the
offline mini-eval), `kv.py` (`--kv-cache-dtype` sizing and concurrency, reproducing `servelab.sizing` numbers in a test), `fp4.py` (NVFP4/MXFP4 layout
calculators and a Blackwell throughput model, verify-marked), `report.py`.
Lab notebooks: `01_quantize_a_checkpoint` (T0 tiny model; T1 llm-compressor on a 0.5B model), `02_serve_and_compare_schemes` (T1: FP16 vs INT4 vs FP8 on a
24 GB GPU; the T4 path is INT4 only with fp16; T0: fake server, simulated), `03_measure_the_accuracy_cost` (T1 lm-eval subset; T0 offline mini-eval and
logit KL), `04_kv_cache_quantization_in_vllm` (T1 on Ada or newer; T0 sizing and simulated ITL), `05_fp4_and_the_blackwell_path` (T0 calculators; verify-marked).
Deploy: `deploy/any-gpu/` (docker run recipes per GPU generation and scheme; Colab/Kaggle T4 INT4 recipe; RunPod/Vast 4090/L4 for FP8) and a pointer to
`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/` with quantized-model variables — no new Terraform.
Must cite/reuse: 01 PRIMER §1–3, §8; serving-engine PRIMER §8 and `minengine.quant`; `servelab.sizing`; vllm-internals §6.3 and its quantization tables;
FlashAttention deep dive §9; the vllm-serving-lab notebook 05.

### 07 · sandboxed-execution — "Sandboxed execution: running model-generated code and tool calls without ambient authority"
Primer sections: 1 Why a sandbox, and the threat model (model-generated code and tool calls are untrusted input; prompt injection → code execution →
exfiltration, lateral movement with the agent's credentials, resource abuse; the OWASP agentic risks — cite the identity primer
`06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md` §6.2 rather than restating; the invariant "no ambient authority": no credentials,
no network by default, no persistent filesystem; blast radius) · 2 The isolation ladder (in-process restrictions are not a boundary; subprocess with
rlimits/seccomp/namespaces; containers — namespaces, cgroups, capabilities, seccomp, no-new-privileges, read-only rootfs, non-root, pids limit; user-space
kernels — gVisor's Sentry/Gofer, syscall interception, its cost; microVMs — Firecracker, Kata, boot times, memory, snapshot/restore; full VMs; a table of what
each isolates, which escapes it defends against, and its overhead) · 3 The execution contract (request: code or tool call + inputs + budgets — CPU time,
wall time, memory, pids, disk, output bytes — + policy; response: truncated stdout/stderr, artifacts, exit reason, usage; idempotency keys and safe
re-execution; ephemeral workspace; sessions vs one-shot; timeouts and cancellation) · 4 Network and secrets (deny-by-default egress; allowlists; an egress
proxy that injects credentials for allowed hosts so the sandbox never holds keys — link the identity primer's token exchange; DNS; exfiltration channels;
package installation inside sandboxes; secrets in prompts vs in tools) · 5 Sandboxes on Kubernetes (link 03 PRIMER §1 and §3: pod-per-execution vs
warm pools with exec; Jobs with `activeDeadlineSeconds` and `ttlSecondsAfterFinished`; `securityContext` — runAsNonRoot, readOnlyRootFilesystem, drop ALL,
seccompProfile RuntimeDefault, allowPrivilegeEscalation false; Pod Security Standards restricted; RuntimeClass — gVisor `runsc`, Kata; NetworkPolicy
default-deny with egress only to the proxy; ResourceQuota/LimitRange; emptyDir sizeLimit; ValidatingAdmissionPolicy (CEL) rejecting non-conforming
sandbox pods; dedicated tainted node pools; GKE Sandbox and Autopilot; what kind can and cannot reproduce — no gVisor in kind) · 6 Latency, throughput and
cost per action (cold start by isolation level, verify-marked: fork/exec ms, container 100–500 ms, gVisor more, Firecracker ~125 ms boot, VM 10 s+;
warm pools and snapshot/restore; the queueing model — arrival rate × execution time → pool size; cost per execution on GKE / Cloud Run jobs / managed
sandboxes; actions per turn × cost per action, link `06-gateway/scaling-admission-cost` §1) · 7 Browser, computer-use and GPU sandboxes in brief
(headless browsers: profiles, downloads, egress; screenshot-to-action loops; GPU jobs an agent launches — same policies plus device plugins, link 03) ·
8 Observability, audit and abuse detection (every execution logged with principal, policy decision, budgets used, exit reason — link the identity primer's
audit event; metrics: p50/p95 startup and run time, kill reasons; detecting fork bombs, miners, egress attempts; incident response) · 9 Where to run it
(laptop: the process sandbox and Docker hardening, gVisor if installed; kind: the pod-per-execution runner with policies; GCP: a GKE Sandbox node pool via
Terraform, Cloud Run jobs; managed sandbox services — E2B, Modal, Daytona, Vertex Agent Sandbox — in a verify-marked table; link `COMPUTE.md`).
Core `sandboxcore` (standard library only — `subprocess`, `resource`, `tempfile`, `socket`, `http.server`, `json`): `threats.py` (a scripted "LLM" that
emits attack payloads — read env secrets, read a fake `~/.ssh`, an egress attempt to a local listener, a fork bomb, a disk fill, an infinite loop, huge
output — each a probe with an expected verdict, every probe harmless by construction: stand-in secrets in a temp dir, bounded limits), `executor.py`
(the unsandboxed executor for contrast and `ProcessSandbox`: clean env, temp workspace, rlimits — CPU, AS, NPROC, FSIZE, NOFILE — wall timeout with
process-group kill, output truncation, exit reasons; states plainly what it cannot stop — the network — and the macOS caveats), `contract.py`
(ExecutionRequest/ExecutionResult dataclasses, budgets, idempotency keys, result hashing), `policy.py` (policy as data: egress allowlist, filesystem
policy, budgets; an evaluator; `render_k8s()` emitting Pod/Job/NetworkPolicy/RuntimeClass/ResourceQuota/LimitRange/ValidatingAdmissionPolicy YAML from a
policy — validated with `kubernetes-validate --strict -k 1.34.0`), `proxy.py` (a tiny allowlisting HTTP egress proxy that injects a credential header for
allowed hosts — tested against a local upstream), `pool.py` (a discrete-event model of warm pools: cold-start latency vs pool size vs arrival rate; cost per
execution), `audit.py` (structured audit events), `agent.py` (a ~60-line agent loop that dispatches a `run_code` tool through the sandbox with budgets and
audit; scripted LLM; injection scenarios).
Core notebooks: `01_the_threat_model` (run the probes through the unsandboxed executor — safely, against stand-ins — and see what leaks),
`02_a_process_sandbox` (rlimits, timeouts, truncation; what is and is not stopped), `03_the_execution_contract_and_policies` (the contract, policy as data,
rendered manifests), `04_egress_and_secrets` (the proxy: no key in the sandbox), `05_pools_latency_and_cost`.
Lab `sandboxlab`: `docker.py` (a hardened `docker run` builder — `--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`,
a seccomp profile, `--pids-limit`, `--memory`, `--cpus`, tmpfs workspace, non-root user, `--runtime runsc` when gVisor is present; runs the core's probes
and reports verdicts; without Docker prints the commands and uses bundled sample verdicts (illustrative)), `k8s/` (manifests via `render_k8s` plus a
`Runner` that creates a Job per execution with kubectl, waits, collects logs; a warm-pool variant using `kubectl exec`; the ValidatingAdmissionPolicy
rejecting pods without the sandbox securityContext/RuntimeClass; startup-latency measurement), `proxy/` (the egress proxy as a deployable container plus
the NetworkPolicy allowing egress only to it), `agent/` (a 07.1-style loop with `run_code` and `fetch_url` tools routed through the sandbox and proxy;
injection scenarios that must fail closed), `bench.py` (startup and run-time latency by isolation level; T0 measures the process sandbox; Docker/kind/GKE
when reachable), `report.py`.
Deploy: `deploy/docker/` (the hardened run script, seccomp profile, gVisor install notes), `deploy/kind/` (cluster script: a namespace with restricted PSS
labels, default-deny NetworkPolicy, the egress proxy, ResourceQuota/LimitRange, the runner Job, the ValidatingAdmissionPolicy; says plainly that gVisor
is unavailable in kind), `deploy/gcp/terraform/` (zonal GKE Standard, system pool, a **GKE Sandbox (gVisor) node pool** — `node_config.sandbox_config
{ type = "gvisor" }` (verified in the provider schema; look up the rest with `$SP/tfattrs.py`) — tainted, Spot, autoscaling 0→N; no Cloud NAT by
default (no egress); Artifact Registry for the sandbox image; managed Prometheus; cheapest defaults), `deploy/gke/` (RuntimeClass `gvisor`, the
namespace, NetworkPolicy, the runner Job with `runtimeClassName: gvisor`, the admission policy, the proxy Deployment).
Lab notebooks: `01_hardened_containers` (T0 + Docker), `02_pod_per_execution_on_kind` (T0 + Docker; predicts without), `03_egress_proxy_and_secret_brokering` (T0),
`04_an_agent_with_a_sandbox_tool` (T0), `05_gke_sandbox_with_gvisor` (T3; T0 inspects and validates manifests).
Must cite/reuse: identity primer §6.2 and its audit section, `06-gateway/scaling-admission-cost` §1 and §5 (admission, cost), 03 PRIMER §1, §3, §9,
`03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab` (`k8sgpu.manifests` style, kind deploy script style), agent-core (07.1) tool contracts.
