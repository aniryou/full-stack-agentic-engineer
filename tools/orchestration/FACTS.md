# FACTS — verified on 2026-09-26 (session research, refreshed for the §6b build the same day). Treat as the source of truth; mark anything not here `(verify)`.
Per-topic fact sheets written by the research agents are kept in the repo under [`facts/`](facts/) (see the last section).

Scratch dir: `$SP` — a per-session scratch directory outside the repo (set it before running the helpers below)
Reference files downloaded from upstream repos (read them instead of guessing): `$SP/ref/`
- `inference.networking.k8s.io_inferencepools.yaml` — InferencePool CRD (v1)
- `llmd-llm-d.ai_inferenceobjectives.yaml`, `llmd-inferenceobjective_types.go` — InferenceObjective CRD
- `llmd-architecture.md`, `llmd-plugins.md`, `llmd-router-readme.md`, `llmd-guides.md` — llm-d Router / EPP / well-lit paths
- `giex-readme.md` — Gateway API Inference Extension README
- `kueue-readme.md`, `jobset-readme.md`, `lws-readme.md`
- `vllm-loggers.py` — vLLM V1 Prometheus metric definitions (source of truth for metric names)
More upstream files: `curl -sS https://raw.githubusercontent.com/<org>/<repo>/main/<path>` works. (github.com web, docs.cloud.google.com,
docs.vllm.ai, kubernetes.io, llm-d.ai, download.pytorch.org, registry.terraform.io are BLOCKED by the egress proxy; PyPI works.)

## Kubernetes ecosystem
- DRA (Dynamic Resource Allocation) is GA in Kubernetes 1.34 and GA in GKE. API group `resource.k8s.io` (v1 in 1.34 — verify in manifests; validate with `kubernetes-validate -k 1.34.0`).
- Kueue: API `kueue.x-k8s.io/v1beta2`; latest release v0.19.6; install: `kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.19.6/manifests.yaml`.
  Topology-Aware Scheduling (TAS), ProvisioningRequest admission checks (DWS flex-start atomic provisioning) are Kueue features.
- JobSet: API `jobset.x-k8s.io/v1alpha2`. LeaderWorkerSet (LWS): API `leaderworkerset.x-k8s.io/v1`, tested on K8s 1.34–1.37.
- KWOK (Kubernetes WithOut Kubelet, kubernetes-sigs/kwok): simulate thousands of fake nodes on a laptop (`kwok`, `kwokctl`).
- Run:ai `fake-gpu-operator` (ghcr.io/run-ai/fake-gpu-operator): simulates NVIDIA GPUs (device plugin, feature discovery, MIG, Prometheus metrics) on CPU-only nodes.
- GKE: control plane >= 1.32.2-gke.1297000 auto-installs the default NVIDIA driver on all GPU nodes (incl. node auto-provisioning).
  `gpu_driver_version`: `DEFAULT` | `LATEST` (COS only) | `INSTALLATION_DISABLED`. GPU nodes are tainted `nvidia.com/gpu=present:NoSchedule` (verify).
- GKE custom ComputeClass: ordered `priorities` fallback (e.g. reservation → spot → on-demand → flex-start); `flexStart`, `spot`, `nodeRecycling` fields (CRD reference: ComputeClass, verify apiVersion `cloud.google.com/v1`).
- DWS flex-start with queued provisioning: node pool with `--flex-start --enable-queued-provisioning`; used via Kueue ProvisioningRequest (atomic, all-or-nothing, up to 7 days).
- Accelerator topology labels on GCP nodes for TAS: `cloud.google.com/gce-topology-block`, `cloud.google.com/gce-topology-subblock`, `cloud.google.com/gce-topology-host` (verify).

## Inference routing (05)
- Gateway API Inference Extension (GIE) is GA. `InferencePool` = `inference.networking.k8s.io/v1` (spec: `selector.matchLabels` (required), `targetPorts` (1–8, required),
  `endpointPickerRef` {name, port.number, kind default Service, failureMode FailClose|FailOpen}, `appProtocol` http|kubernetes.io/h2c). GKE >= 1.34.0-gke.1626000 manages the v1 CRD.
- EPP + `InferenceObjective` + `InferenceModelRewrite` moved to **llm-d Router** (repo llm-d/llm-d-router; "Inference Scheduler" renamed "llm-d Router").
  `InferenceObjective` = `llm-d.ai/v1alpha2`, spec: `poolRef`, `priority` (int32; higher served first; default 0).
- EPP config: `apiVersion: llm-d.ai/v1`, `kind: EndpointPickerConfig`, `plugins:` [{type, name, parameters}], `schedulingProfiles:` [{name, plugins: [{pluginRef, weight}]}].
  Plugin types seen upstream: `prefix-cache-scorer`, `kv-cache-utilization-scorer`, `queue-scorer`, `decode-filter`, `max-score-picker`, `single-profile-handler`;
  flow control (`featureGates: ["flowControl"]`), data producers (`approx-prefix-cache-producer`, `inflight-load-producer`, `predicted-latency-producer`).
  Metrics refresh default 50 ms. Modes: Standalone (self-managed Envoy; Helm chart; sidecar or service proxy) and Gateway mode (InferencePool + HTTPRoute).
- llm-d well-lit paths: optimized-baseline (prefix + load aware), precise-prefix-cache-routing, tiered-prefix-cache (CPU/disk offload), pd-disaggregation, wide-ep,
  flow-control, workload-autoscaling, agentic-serving, batch-serving. llm-d is a CNCF Sandbox project; GKE Gateway can front llm-d EPP.
- `llm-d-inference-sim`: vLLM simulator, OpenAI-compatible, mirrors vLLM Prometheus metrics, no GPU (github llm-d/llm-d-inference-sim; image on ghcr.io — verify tag).
- NVIDIA Dynamo 1.0 GA on 2026-03-16 (1.x since): disaggregated serving, KV-aware router, planner (SLA-based), NIXL transfers; supports vLLM/SGLang/TRT-LLM.
- GKE Inference Gateway GatewayClass names: `gke-l7-regional-external-managed`, `gke-l7-rilb` (verify); needs a proxy-only subnet (`purpose = "REGIONAL_MANAGED_PROXY"`).

## vLLM (04/05) — metric names from vllm/v1/metrics/loggers.py (main, Sep 2026)
`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` (0–1; older releases: `vllm:gpu_cache_usage_perc`),
`vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, `vllm:num_preemptions`, `vllm:prompt_tokens`, `vllm:generation_tokens`,
`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`,
`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:request_success`,
`vllm:iteration_tokens_total`, `vllm:lora_requests_info`, `vllm:cache_config_info`. (Counters are exposed with `_total` suffix by prometheus_client.)

## GCP compute (prices are approximate, us-central1, Sep 2026 — always "verify")
- GPUs are NOT usable on a Free Trial billing account; upgrade to paid (credits kept). GPU quota (`GPUS_ALL_REGIONS` + per-type regional) often starts at 0 → request it. T4/L4 approvals are usually quick; A100/H100 are hard for new/individual accounts.
- T4 (N1): ~$0.35–0.55/hr. L4 (G2, g2-standard-4): ~$0.70/hr. Spot: 60–91% off. A100 40GB (a2-highgpu-1g) ~$3.7/hr; 80GB (a2-ultragpu-1g) ~$5/hr.
- H100: on-demand only as `a3-highgpu-8g` ≈ $88/hr (~$11/GPU-hr); Spot ≈ $3.7/GPU-hr; 1g/2g/4g A3 shapes only via Spot/flex-start (verify).
- A3 Mega (H100, GPUDirect-TCPXO), A3 Ultra (H200, RDMA), A4 (B200, 8 GPUs, NVLink 1.8 TB/s/GPU), A4X (GB200 NVL72), A4X Max (GB300 NVL72): reservation / DWS / calendar-mode oriented.
  A4X Max/A4X/A4/A3 Ultra use GPUDirect RDMA via MRDMA NICs on a 4-way rail-aligned network (ConnectX-7 on A3 Ultra/A4; GB300-class A4X Max most likely ConnectX-8 — verify per machine type).
- G4 = NVIDIA RTX PRO 6000 Blackwell (96 GB), GA.
- Cloud Run GPUs (GA since June 2025): NVIDIA L4 (24 GB; min 4 vCPU/16 GiB) and RTX PRO 6000 Blackwell (96 GB; min 20 vCPU/80 GiB); per-second billing; scale to zero.
- TPU v7 "Ironwood" (TPU7x) GA 2026-04-22: 192 GB HBM/chip, ~4.6 PFLOPS FP8/chip, 9,216-chip superpod; used via GKE. v6e Trillium, v5e/v5p exist. vLLM has a TPU backend (verify current name/status).
- Terraform google provider **8.4.0** is current; validate with `$SP/tfcheck.sh <dir>` (uses a local provider mirror: google/google-beta 8.4.0, kubernetes 3.2.1, helm 3.3.0, random 3.7.2, null 3.2.4 — ONLY these providers/versions work offline; pin `>= 8.0` for google).
  Look up exact attribute names: `$SP/tfattrs.py <resource> [filter...] [--desc]`. Verified examples:
  - `google_cloud_run_v2_service`: `template.node_selector.accelerator`, `template.gpu_zonal_redundancy_disabled`, `template.containers.resources.limits`, `template.scaling.{min,max}_instance_count`.
  - `google_container_node_pool`: `node_config.guest_accelerator.{type,count,gpu_partition_size}`, `...gpu_driver_installation_config.gpu_driver_version`,
    `...gpu_sharing_config.{gpu_sharing_strategy,max_shared_clients_per_gpu}`, `node_config.{spot,flex_start,reservation_affinity,secondary_boot_disks,gcfs_config}`,
    `queued_provisioning.enabled`, `placement_policy.{type,policy_name,tpu_topology}`.
  - `google_container_cluster`: `gateway_api_config.channel`, `monitoring_config.{enable_components,managed_prometheus}`, `addons_config.gcs_fuse_csi_driver_config`,
    `addons_config.{parallelstore,lustre}_csi_driver_config`, `addons_config.ray_operator_config`.
  - `google_compute_instance`: `scheduling.{provisioning_model,instance_termination_action,max_run_duration}`.

## Non-GCP compute (approximate, Sep 2026 — verify)
- Free: Google Colab free tier T4 16 GB (~15–30 GPU-hrs/week, not guaranteed, ~12 h sessions); **Kaggle notebooks: 2×T4 or P100, 30 GPU-hrs/week** (a free 2-GPU box, PCIe only, no NVLink).
- Vast.ai (marketplace): RTX 4090 ~$0.3–0.4/hr; A100 80GB from ~$0.5–1/hr; H100 ~$1.5–1.9/hr verified hosts. RunPod (per-second): RTX 4090 from ~$0.34/hr, A100 ~$1.6/hr, H100 PCIe ~$2.9/hr; multi-GPU NVLink pods available.
  Lambda: A100 40GB ~$2/hr, H100 ~$3.3/hr, full VMs. Modal: serverless Python GPUs with monthly free credits. RunPod/Vast give containers (no kernel-module/driver control, no Kubernetes);
  Lambda/GCP give VMs.
- Environment here: no GPU, no Docker daemon, Python 3.11, PyPI reachable, torch NOT installable (download.pytorch.org blocked; PyPI torch pulls ~2.5 GB of CUDA wheels — do not install).
  numba installs from PyPI (use `NUMBA_ENABLE_CUDASIM=1` to run CUDA kernels on CPU).

## Environment for the §6b build (2026-09-26, this session)
- Python 3.11.15; installed system-wide: numpy 2.4, pytest 9.1, nbformat, nbclient, ipykernel (kernel `python3` registered), matplotlib, pyyaml, aiohttp, httpx, requests,
  kubernetes-validate 1.36 (`kubernetes-validate --strict -k 1.34.0 <yaml>`), **torch 2.14.0 (CPU build; `torch.cuda.is_available()` is False)** — import it lazily, skip cleanly when absent.
- No GPU, no Docker daemon (`docker` binary only), no kind/kubectl/helm. 4 CPUs, 15 GB RAM: keep tests < 60 s and notebook runs < 10 min each.
- Terraform 1.13.3 + offline provider mirror (google/google-beta 8.4.0, kubernetes 3.2.1, helm 3.3.0, random 3.7.2, null 3.2.4):
  `export ORCH_SCRATCH=$SP TF_CLI_CONFIG_FILE=$SP/tf/terraformrc TERRAFORM_BIN=$SP/tf/bin/terraform; $SP/tfcheck.sh <terraform-dir>`.
  Attribute lookups: `TF_SCHEMA_JSON=$SP/tf/schema.json python3 $SP/tfattrs.py google_container_node_pool sandbox` → `node_config.sandbox_config.type` (required) — GKE Sandbox is `sandbox_config { type = "gvisor" }`.
- Installs: `$SP/pipi <pkgs>` only (serialized pip). Never install torch (present), vllm, triton, flash-attn, transformers-with-models that download weights.
- Network: `git clone https://github.com/<org>/<repo>` and `https://raw.githubusercontent.com/...` work; github.com web pages, api.github.com, codeload, docs sites (docs.vllm.ai, kubernetes.io, gvisor.dev, cloud docs), arxiv.org, huggingface.co and download.pytorch.org are blocked. PyPI works.
- Upstream sources already cloned (shallow, some sparse) under `$SP/ref/`: vllm (docs/, vllm/reasoning, vllm/model_executor/layers/quantization, .../fused_moe, vllm/engine, vllm/entrypoints/openai, vllm/config, vllm/v1/core, vllm/distributed, examples/), llm-compressor (full), compressed-tensors, trl (docs/source, trl/trainer, examples/scripts), transformers (models: qwen3_moe, qwen2_moe, mixtral, deepseek_v3, llama4, gpt_oss, olmoe, granitemoe; docs/source/en/quantization; integrations), gvisor (g3doc), k8s-website (concepts/security, containers, reference/access-authn-authz, workloads/controllers, services-networking, tasks/configure-pod-container, policy, scheduling-eviction, tasks/administer-cluster), firecracker (docs), kata (docs), kind (site docs), e2b, deepseek-v3 (README + inference/ incl. configs and model.py), deepseek-r1, deepep, dapo, verl (docs), gpt-oss, llama-models (models/llama4), kimi-k2, olmoe, qwen3, megablocks, gptq, marlin, llm-awq, smoothquant, kivi, gptqmodel, lm-eval (docs, gsm8k/mmlu tasks), modelopt (docs, examples/llm_ptq), tf-google (website/docs/r), sglang (docs).
  Use `grep -rn` over these instead of guessing; cite the file path in the fact sheet. Missing repo? `flock $SP/ref/.lock git clone --depth 1 https://github.com/<org>/<repo> $SP/ref/<name>`.
- Repo facts that the new topics must stay consistent with: vLLM pinned at 0.30.0 / main@5840d95 (verify); Kueue v0.19.6; K8s 1.34; the roofline core's model catalogue (`01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/roofline/llm.py`: Mixtral-8x7B 46.7 B total / 12.9 B active, Qwen3-30B-A3B 30.5 B / 3.35 B); the capacity primer's Mistral Large 3 (675 B MoE, 41 B active); `minengine.quant` formats (int8 per-channel, int4 group-wise, fp8-e4m3 emulation); `servelab.sizing` (KV blocks from config.json + gpu_memory_utilization).

## Per-topic fact sheets for the four §6b topics (2026-09-26)
Written by the research agents from the cloned upstream sources before the builders started; each fact names the file it was read
from, and the unverified items are listed at the end of each sheet. They are the verification record behind those four primers.
- [`facts/mixture-of-experts.md`](facts/mixture-of-experts.md) — model architectures computed from configs, router code per family, load-balance and capacity code, experts touched, EP bytes, vLLM v0.30.0 MoE flags, small-GPU fits.
- [`facts/rl-and-thinking-models.md`](facts/rl-and-thinking-models.md) — vLLM reasoning parsers and fields, Qwen3 and DeepSeek-R1 templates and settings, TRL GRPO/DPO defaults, DAPO, verl, pass@k, T4 sizing predictions.
- [`facts/quantization.md`](facts/quantization.md) — vLLM quantization schemes and kernel minimum capabilities, the compressed-tensors format, llm-compressor recipes, GPTQ/AWQ/SmoothQuant/KIVI as code, FP8/FP4 formats, lm-eval.
- [`facts/sandboxed-execution.md`](facts/sandboxed-execution.md) — gVisor, Pod Security Standards, RuntimeClass, ValidatingAdmissionPolicy, NetworkPolicy, Jobs, Firecracker and Kata, kind's NetworkPolicy enforcement, GKE Sandbox in Terraform (`sandbox_config.type = "GVISOR"`), rlimit and subprocess pitfalls.
