# Compute — where to run each tier, what it costs, and how not to overspend

Where every notebook and deploy target in this repo can run, from a laptop to Google Cloud: the four run tiers,
free and cheap GPU options, what obtaining GPUs on GCP involves, which lab needs which tier, and the habits that
keep a paid session from becoming a bill. The study plan that uses these tiers is [`CURRICULUM.md`](CURRICULUM.md).

*Prices are approximate, as of 2026-09-26 (GCP prices for us-central1). They move; everything marked `(verify)`
and everything in the Verify list (§9) should be re-checked before you rely on it.*

---

## The one-minute version

- **T0 is $0 and teaches every concept:** a laptop, Colab CPU or CI. A laptop with Docker adds the local Kubernetes
  (kind) and compose stacks for layers 03 and 05, and the container sandboxes of layer 07. No GPU needed.
- **T1 (one GPU) is free** on Colab or Kaggle (T4), or about $0.3–0.7/hr for a 24 GB GPU (RTX 4090 on RunPod or
  Vast.ai, L4 on GCP).
- **T2 (several GPUs) is free over PCIe** on Kaggle's 2×T4. NVLink needs a rented 2–8× A100/H100 box for about an
  hour: roughly $1–6 for two GPUs, up to ~$25 for eight.
- **T3 (GCP managed) is optional.** GPUs need a paid billing account (Free Trial credits cannot run them) and quota.
  T4 and L4 are easy to obtain; A100 and H100 are not, and an on-demand H100 comes only as an 8-GPU VM at about
  $88/hr. Rent A100/H100 elsewhere; use GCP for the managed-service view, on L4 Spot with scale-to-zero.
- **Every paid session:** run the T0 path first, set an auto-stop, tear down (`terraform destroy`, terminate the
  pod), then check that nothing is left running.

---

## 1. Run tiers

| Tier | Where | Cost | Used for | What it cannot show |
|---|---|---|---|---|
| **T0** | laptop / Colab CPU / CI | $0 | the core concept: simulators, calculators, from-scratch implementations. Every concept is learnable here. | real kernel timings, real bandwidths, real engine latencies — T0 output of those is labelled "simulated" |
| **T1** | one small GPU: Colab or Kaggle T4 (free), a rented 24 GB GPU (RTX 4090 or L4, ~$0.3–0.7/hr), GCP L4 Spot | free–$0.7/hr | real kernels, real vLLM with a 0.5–2B model, real metrics | anything that happens between GPUs |
| **T2** | a multi-GPU box, ideally with NVLink (Kaggle 2×T4 over PCIe is free; RunPod, Vast.ai or Lambda 2–8× A100/H100 for ~1 h) | ~$0–25 per session | collective bandwidth, P2P and NVLink, tensor parallelism | managed-cloud integration |
| **T3** | GCP managed services (GKE, Cloud Run, Inference Gateway, DWS) via Terraform | pay per use; defaults L4 + Spot + scale-to-zero | how it looks in production on one cloud | nothing conceptual that T0 lacks; it adds the product surface |

Notebooks at T1 and above detect the GPU, Docker, cluster or cloud they need. If it is missing they run a labelled
T0 path and print the commands to run on real hardware, so no notebook requires hardware you do not have.

### 1.1 Where to run what

| You want to | Cheapest faithful option |
|---|---|
| learn any concept in the repo | T0: a laptop or Colab CPU |
| time a kernel, measure memory bandwidth, load weights | Colab or Kaggle T4 |
| serve a real model with vLLM | Colab or Kaggle T4 with a 0.5–2B model and `--dtype half` (vLLM picks its Triton attention backend there); a 24 GB GPU for 7–8B models at 16-bit |
| try FP8 | a GPU with compute capability 8.9 or higher: RTX 4090 (RunPod, Vast.ai) or L4 (GCP Spot) |
| try FP4 (NVFP4 W4A4) | one rented Blackwell GPU: B200 or RTX PRO 6000 (GCP G4, Cloud Run) `(verify)` |
| serve a thinking model | Colab or Kaggle T4 with `Qwen/Qwen3-0.6B`, `--reasoning-parser qwen3` and `--dtype half`; a 24 GB GPU for a 4B model |
| measure NCCL collectives over PCIe | Kaggle 2×T4 |
| compare tensor and expert parallelism for an MoE | Kaggle 2×T4 with a small MoE in vLLM |
| measure NVLink P2P and busbw | 2× A100 or H100 SXM on RunPod, Vast.ai or Lambda for an hour |
| practise Kubernetes GPU scheduling, Kueue, gangs | kind with fake GPU capacity on a laptop — no GPU needed |
| run llm-d and an inference router locally | Docker compose or kind with `llm-d-inference-sim` — no GPU needed |
| run model-generated code behind a real kernel boundary | Docker with gVisor (`runsc`) on Linux, or a GKE Sandbox node pool; kind cannot run gVisor — no GPU needed |
| see the managed-cloud version | GCP: L4 Spot node pools that scale from zero, Cloud Run GPU that scales to zero |

---

## 2. Choosing a GPU for an experiment

What the GPUs you are likely to rent can and cannot show. All values `(verify)`; the dated catalogue the notebooks
compute with is `roofline.specs` in [`roofline-core`](01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/).

| GPU | Memory | Bandwidth | Compute capability | BF16 | FP8 | NVLink | MIG | Cheapest places |
|---|---|---|---|---|---|---|---|---|
| T4 | 16 GB GDDR6 | ~320 GB/s | 7.5 | no (use FP16) | no | no | no | Colab free, Kaggle (2×), GCP N1 |
| L4 | 24 GB GDDR6 | ~300 GB/s | 8.9 | yes | yes | no | no | GCP G2 (Spot), Cloud Run |
| RTX 4090 | 24 GB GDDR6X | ~1.0 TB/s | 8.9 | yes | yes | no | no | Vast.ai, RunPod |
| A100 40/80 GB | HBM2 / HBM2e | ~1.6–2.0 TB/s | 8.0 | yes | no | SXM: ~600 GB/s | yes | Vast.ai, RunPod, Lambda, GCP A2 |
| H100 80 GB | HBM3 (SXM) / HBM2e (PCIe) | ~3.35 TB/s SXM, ~2.0 TB/s PCIe | 9.0 | yes | yes | SXM: ~900 GB/s | yes | Vast.ai, RunPod, Lambda, GCP A3 |
| RTX PRO 6000 Blackwell | 96 GB GDDR7 | ~1.6 TB/s | 12.0 | yes | yes (and FP4) | no | yes `(verify)` | GCP G4, Cloud Run |

Consequences for the labs:

- **Decode-bound experiments care about bandwidth per dollar.** An RTX 4090 has about 3× an L4's memory bandwidth
  for a similar hourly price, so ITL and bandwidth measurements look very different on the two.
- **A T4 has no BF16 and no TF32.** Run vLLM with `--dtype half`, and read only the fp32 and fp16 rows of a GEMM
  sweep. vLLM v0.30.0, the version layer 04 pins, supports compute capability 7.5 and up, so a T4 works and a V100
  or a P100 does not; FlashAttention is not available on a T4, so vLLM falls back to its Triton attention backend
  (`TRITON_ATTN`) `(verify)`. The details are in layer 04's
  [`deploy/any-gpu`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md).
- **FP8 needs compute capability 8.9 or higher** (L4, RTX 4090, H100). On a T4 (or an A100) an FP8 checkpoint loads
  but runs weight-only, and a T4 has no FP8 KV-cache backend; the T4's quantized paths are INT4 weight-only and INT8
  W8A8. `python3 -m quantlab plan --gpu T4` in layer 04's
  [`quant-lab`](04-inference-engine/quantization/quant-lab/) prints the table for the pinned vLLM `(verify)`.
- **FP4 (NVFP4 W4A4) needs Blackwell** (B200 or RTX PRO 6000) and CUDA 12.8 or later; only the quantization lab's
  notebook 05 asks for one, and it has a T0 path `(verify)`.
- **NVLink needs SXM A100/H100** (or PCIe cards joined by a bridge). Kaggle's 2×T4, GCP's 2×L4 `g2-standard-24` and
  multi-4090 boxes are PCIe-only: useful as the contrast, not as the thing being measured.
- **MIG needs an A100/H100-class GPU;** time-slicing works on any GPU.

---

## 3. Free options

### 3.1 Laptop or CI (T0)

- Python 3.10+ with numpy runs every core; the labs' `requirements.txt` add what their notebooks need. One
  exception: the Mistral durable lab (`long-running-agents-mistral`) needs Python 3.12 or later for
  `mistralai-workflows` (`>=3.12,<3.15` as of 3.15.0, 2026-09-26 `(verify)`); its standard-library core runs on 3.10.
- Docker is needed only for the local cluster and serving stacks: layer 03's
  [`deploy/kind/`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/) (kind with fake `nvidia.com/gpu`
  capacity, Kueue, JobSet, LWS; optional KWOK for hundreds of fake nodes) and layer 05's
  [`deploy/local/`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/local/) (compose) and
  [`deploy/kind/`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/kind/) (llm-d Router with
  `llm-d-inference-sim`). Without Docker those notebooks fall back to bundled simulators. On Apple silicon, check
  that the images publish an arm64 variant `(verify)`.
- GCP Free Trial credits cannot pay for GPUs, but they can pay for CPU-only practice: a small VM to host kind,
  `terraform plan`, a GKE cluster without its GPU pool `(verify)`.

### 3.2 Google Colab (T1)

- Free tier: a T4 16 GB when available, about 15–30 GPU-hours a week, not guaranteed; sessions up to ~12 h and
  shorter when idle `(verify)`. Paid tiers can offer L4 or A100 through compute units `(verify)`.
- Choose *Runtime → Change runtime type → T4 GPU*. Every notebook's first cell clones this repo and installs its lab
  ([`COLAB.md`](COLAB.md)).
- Good for: layer 01 lab notebooks 01, 02 and 04; layer 02 lab notebooks 02 and 05; layer 04 lab notebooks 02–05
  (except FP8).
- PyTorch with CUDA is preinstalled (on Kaggle too), so the labs' GPU paths need no extra install; `pip install vllm`
  brings the PyTorch version vLLM pins and takes several minutes `(verify)`.
- Not for: more than one GPU, Docker, kind or compose.

### 3.3 Kaggle (T1, and T2 without NVLink)

- Free: 2×T4 (or one P100), 30 GPU-hours a week `(verify)`, PCIe only. Pick *GPU T4 x2*: the P100 (compute
  capability 6.0) is below vLLM's minimum and has no tensor cores worth measuring. It is the only free two-GPU box: real NCCL
  collectives, `nvidia-smi topo -m`, P2P over PCIe (check whether peer access is enabled; NCCL falls back to host
  memory if not `(verify)`), and nccl-tests via the recipe in layer 02's
  [`deploy/any-gpu/`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/).
- Switch *Internet* on in the notebook settings to clone and pip-install; it requires a phone-verified account
  `(verify)`.
- The Colab bootstrap cell does nothing on Kaggle. Upload or import the lab notebook, add this as its first cell,
  then run the rest; the bootstrap finds the package by walking up from the working directory:

```
!git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer /kaggle/working/fsae
%cd /kaggle/working/fsae/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
!pip install -q -e .
```

---

## 4. Cheap GPU providers

| Provider | What you get | Billing | Multi-GPU and NVLink | Approx. price (Sep 2026, verify) | Stopping vs deleting |
|---|---|---|---|---|---|
| **RunPod** | a container ("pod") from an image you choose; no driver or kernel-module control, no Kubernetes | per second | multi-GPU pods; NVLink on SXM types | RTX 4090 from ~$0.34/hr; A100 ~$1.6/hr; H100 PCIe ~$2.9/hr | a stopped pod still bills its volume disk; terminate it to end all charges `(verify)` |
| **Vast.ai** | a container on a marketplace host; same limits; reliability varies by host | per second `(verify)` | depends on the host: filter on GPU count and SXM/NVLink | RTX 4090 ~$0.3–0.4/hr; A100 80 GB from ~$0.5–1/hr; H100 ~$1.5–1.9/hr on verified hosts | a stopped instance still bills storage; destroy it `(verify)` |
| **Lambda** | a full VM with root; driver and CUDA preinstalled `(verify)` | per minute `(verify)` | 1–8 GPU instances; 8× SXM with NVLink `(verify)` | A100 40 GB ~$2/hr; H100 ~$3.3/hr | instances are terminated, not stopped `(verify)` |
| **Modal** | serverless containers defined in Python; scale to zero | per second `(verify)` | several GPUs per container `(verify)` | monthly free credits `(verify amount)` | nothing runs between calls unless you keep containers warm |

**Container or VM.** In a container (RunPod, Vast.ai, Modal) the host already runs the NVIDIA driver and injects
`/dev/nvidia*` and the driver libraries; your image brings the CUDA userland. That is exactly the view layer 02's
`05_how_a_container_sees_a_gpu` explains, and it is all that layers 01 and 04 and layer 02's kernels and NCCL
collectives need. You cannot change the driver, enable MIG, load kernel modules or, usually, run Docker or
Kubernetes inside `(verify)`, so start the pod from the image a `deploy/any-gpu/` recipe names (for example
`vllm/vllm-openai`) instead of calling `docker run`. On a VM (Lambda, GCP) you own the host — the driver, the NVIDIA
Container Toolkit, `docker run --gpus all`, a single-node Kubernetes if you want a real device plugin — at the price
of doing that setup yourself.

**Which to pick.** RunPod or Vast.ai for T1 on a 4090 and for short T2 NVLink sessions; Lambda when you need a VM
(Docker with `--gpus all`, building nccl-tests, 8× H100 SXM); Modal for scripted, repeatable benchmark runs that
scale to zero.

---

## 5. Google Cloud

### 5.1 Before the first GPU

1. **Billing.** GPUs cannot be used on a Free Trial billing account. Upgrade it to a paid account; remaining trial
   credits are kept.
2. **Quota.** GPUs have a global `GPUS_ALL_REGIONS` quota and a per-region quota per GPU model, and both often
   start at zero. Request one or two of what the lab needs (*IAM & Admin → Quotas & System Limits* `(verify)`).
   Spot capacity counts against separate preemptible quota metrics, and Cloud Run GPUs have their own quota
   `(verify)`.
3. **Approval.** T4 and L4 requests are usually approved quickly. A100 and H100 are hard to obtain for new or
   individual accounts: plan GCP work around L4 and rent A100/H100 elsewhere (§4).
4. **One project per lab,** with a budget and alerts (§8), so a whole lab can be torn down or shut down at once.

### 5.2 GPUs on GCP: shapes, prices, obtainability

| GPU | Machine types | On-demand (~, us-central1) | Spot | Obtainability for an individual account |
|---|---|---|---|---|
| T4 16 GB | N1 with T4s attached | ~$0.35–0.55/hr | 60–91 % off | quota usually approved quickly |
| L4 24 GB | G2: `g2-standard-4` (1 GPU) … `g2-standard-24` (2 GPUs) | ~$0.70/hr (`g2-standard-4`) | 60–91 % off, i.e. roughly $0.06–0.28/hr `(verify the current Spot price)` | easy; also on Cloud Run and GKE |
| RTX PRO 6000 96 GB | G4 | `(verify)` | `(verify)` | generally available; also on Cloud Run |
| A100 40 GB | A2: `a2-highgpu-1g`; `a2-highgpu-2g` and up for NVLink between GPUs | ~$3.7/hr (1 GPU) | 60–91 % off | hard |
| A100 80 GB | `a2-ultragpu-1g` and up | ~$5/hr (1 GPU) | 60–91 % off | hard |
| H100 80 GB | A3: on demand only as `a3-highgpu-8g`; 1-, 2- and 4-GPU shapes only via Spot or flex-start `(verify)` | ≈ $88/hr for 8 GPUs (~$11 per GPU-hour) | ≈ $3.7 per GPU-hour | hard; one on-demand NVLink hour costs ~$88 |
| H100 Mega, H200, B200, GB200 NVL72, GB300 NVL72 | A3 Mega (GPUDirect-TCPXO), A3 Ultra, A4, A4X, A4X Max (GPUDirect RDMA over ConnectX-7 NICs, 4-way rail-aligned) | reservation, DWS or calendar mode | — | not a learning-budget option |

### 5.3 Ways to get capacity

| Capacity type | What you get | Price | Catch | Where the labs use it |
|---|---|---|---|---|
| On-demand (`STANDARD`) | a VM or node now, if the zone has capacity and you have quota | list price | A100/H100 stock-outs; H100 only as an 8-GPU VM | fallback when Spot is unavailable |
| Spot (`SPOT`) | the same machine at 60–91 % off | the cheapest | can be preempted at any time with ~30 s notice `(verify)`; no capacity guarantee; separate quota | the default for every GPU VM and node pool in the labs' Terraform |
| DWS flex-start (`FLEX_START`) | ask for N GPUs; they are provisioned when capacity frees up and then run for a bounded time (up to 7 days); on GKE with queued provisioning the request is all-or-nothing, through a Kueue ProvisioningRequest | discounted against on-demand `(verify)` | an unknown wait; a bounded run | layer 03's optional flex-start pool; the practical way to try A100/H100 without a reservation |
| DWS calendar mode | a block of capacity booked for fixed future dates | `(verify)` | you pay for the whole window `(verify durations and lead time)` | not needed for the labs |
| Reservations and committed use | capacity held in a zone; 1- or 3-year commitments for discounts | billed whether used or not | idle cost | production only; a ComputeClass fallback list usually starts from one ([`CURRICULUM.md`](CURRICULUM.md) module 03.5) |

### 5.4 What the labs create, and what bills besides the GPU

| Lab deploy target | Creates (defaults) | Keeps billing while idle |
|---|---|---|
| 01 [`gpu-bench-lab/deploy/gcp/terraform`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/gcp/terraform/) | one Spot `g2-standard-4` (L4) VM from a Deep Learning VM image; a startup script runs the suite and uploads JSON to a GCS bucket; `max_run_duration` ends it; a variable switches to `a2-highgpu-2g` for NVLink | the boot disk until deleted; the bucket |
| 02 [`cuda-nccl-lab/deploy/gcp/terraform`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/) | zonal GKE Standard: a CPU system pool, an L4 Spot pool scaling 0→N, an optional time-sharing pool, an optional A100 MIG pool (off by default), DCGM and Managed Prometheus | the cluster fee, the system pool, metric ingestion |
| 03 [`k8s-gpu-lab/deploy/gcp/terraform`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) | zonal GKE Standard: a system pool, an L4 Spot pool 0→N, an optional flex-start queued-provisioning pool, GCS FUSE CSI, image streaming, Managed Prometheus | the cluster fee, the system pool, metric ingestion |
| 04 [`vllm-serving-lab/deploy/gcp/cloud-run`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/) | a Cloud Run service with one L4, scaling to zero; weights from GCS or a Hugging Face token in Secret Manager | an idle instance until it is scaled in `(verify how long)`; the weights bucket |
| 04 [`vllm-serving-lab/deploy/gcp/gke`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/) | a vLLM Deployment on an L4 node pool, scraped by PodMonitoring | the GPU node for as long as the Deployment has replicas |
| 05 [`inference-gateway-lab/deploy/gcp/terraform`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/) | zonal GKE Standard with the Gateway API, a proxy-only subnet, an L4 Spot pool 0→N, Managed Prometheus; the Gateway creates a regional external managed load balancer | the load balancer `(verify pricing)`, the cluster fee, the system pool |
| 07 [`sandbox-lab/deploy/gcp/terraform`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gcp/terraform/) | zonal GKE Standard with private nodes and Dataplane V2, a one-node `e2-standard-2` system pool, a GKE Sandbox (gVisor) Spot pool 0→3, an Artifact Registry repository, Managed Prometheus; Cloud NAT only with `enable_nat = true`; no GPU | the cluster fee, the system pool (roughly $0.07/h on demand in us-central1 `(verify)`), image storage, Cloud NAT if enabled |

The 00 MoE and thinking labs and the 04 quantization lab add no Terraform: MoE's expert-parallel Deployment runs on
layer 02's cluster (its 2 × L4 `l4x2` pool), and the thinking and quantized models run on the serving lab's Cloud Run
and GKE targets above with different model ids and engine flags.

Notes that apply across labs:

- **GKE.** A Standard cluster pays a management fee per hour; the GKE free tier offsets one zonal (or Autopilot)
  cluster per billing account `(verify)`. With the GPU pool at zero, the CPU system pool is the idle cost. GKE
  installs the NVIDIA driver on GPU nodes itself (control plane 1.32.2-gke.1297000 or later), so there is nothing to
  install. The cluster autoscaler removes an empty GPU node only after it has been unneeded for a while (about ten
  minutes by default `(verify)`).
- **Cloud Run GPU.** L4 (24 GB; at least 4 vCPU and 16 GiB) and RTX PRO 6000 (96 GB; at least 20 vCPU and 80 GiB),
  billed per second, scale to zero. GPU services use instance-based billing, so an idle instance costs money until it
  is scaled in `(verify)`; turning GPU zonal redundancy off is cheaper `(verify)`.
- **Everything else:** boot and persistent disks (a PVC's disk can outlive its cluster), GCS buckets for weights and
  results, Artifact Registry images, static IPs, Cloud NAT if you use private nodes. Downloading a model is ingress
  and free; results leaving the region are egress.

### 5.5 TPUs

No module needs a TPU. TPU v5e, v6e (Trillium) and v7 (Ironwood, generally available since 2026-04-22: 192 GB of HBM
and ~4.6 PFLOPS FP8 per chip) are used through GKE, and vLLM has a TPU backend `(verify)`. A TPU module is in the
backlog ([`CURRICULUM.md`](CURRICULUM.md) §6).

---

## 6. Which lab needs which tier

Every core runs entirely at T0, and every lab notebook has a T0 path (the second column). "Real run" is the tier at
which a lab notebook's numbers become measurements.

### 00 · [`moe-lab`](00-foundations/mixture-of-experts/moe-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_a_tiny_moe_in_torch` | torch on a CPU trains the tiny MoE (a real run, seconds); without torch, curves this code recorded | — | — | — |
| `02_watch_the_router` | bundled router traces in vLLM's format (illustrative), or the tiny MoE's own traces with torch | T1: forward hooks on OLMoE-1B-7B through transformers, or `vllm serve --enable-return-routed-experts` | Colab or Kaggle T4 (OLMoE-1B-7B in fp16) | — |
| `03_batch_vs_weight_stream` | the roofline with stated efficiencies (simulated) | T1: decode step time against batch for an MoE and a dense model in vLLM | Colab or Kaggle T4; a 24 GB L4 or RTX 4090 | a single-GPU MoE is layer 04's Cloud Run or GKE deploy with the model id changed |
| `04_expert_parallelism_on_two_gpus` | a per-GPU roofline and an α-β link (simulated); bundled `vllm bench serve` results (illustrative) | T2: TP, TP + EP and DP + EP in vLLM | Kaggle 2×T4 (PCIe, $0) | layer 02's `l4x2` pool (`g2-standard-24`, 2 × L4) via `deploy/gke` |
| `05_moe_on_a_small_gpu` | the sizing arithmetic (simulated); two start-up logs (illustrative) | T1: `--cpu-offload-gb` or a 4-bit checkpoint | Colab or Kaggle T4 (16 GB); a 24 GB L4 or RTX 4090 for Qwen3-30B-A3B with 4-bit experts | — |
| [`deploy/any-gpu`](00-foundations/mixture-of-experts/moe-lab/deploy/any-gpu/) | `DRY_RUN=1` | T1/T2 | `serve_moe.sh` (Docker or pip) on one GPU with offload or two in any layout, `bench_layouts.sh`; Colab, Kaggle and rented-GPU recipes | — |
| [`deploy/gke`](00-foundations/mixture-of-experts/moe-lab/deploy/gke/) | schema checks offline | T3 | — | Qwen1.5-MoE-A2.7B with `--enable-expert-parallel` on layer 02's `l4x2` pool, benchmarked from a CPU Job; no Terraform here: the cluster is layer 02's [`deploy/gcp/terraform`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/) |

### 00 · [`thinking-lab`](00-foundations/rl-and-thinking-models/thinking-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_grpo_on_a_tiny_transformer` | torch on a CPU trains a tiny transformer with SFT then GRPO (a real run, about a minute); without torch, a recorded run (illustrative) | T1 only makes it faster | — | — |
| `02_a_thinking_model_on_one_gpu` | bundled responses in vLLM's format (illustrative) and an in-process fake server (simulated) | T1: `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3` | Colab or Kaggle T4 (`--dtype half`) | the 04 serving lab's Cloud Run (one L4, scale to zero) or GKE deploy with Qwen3-4B and a reasoning parser, via `deploy/gcp` |
| `03_test_time_compute_for_real` | a bundled table of the simulated model's outcomes (illustrative) | T1: n samples per problem from a real thinking model — a few thousand requests, tens of minutes on a T4 `(verify)` | Colab or Kaggle T4 | — |
| `04_serving_thinking_models` | the fake vLLM and the engine emulator (simulated) | T1: ITL, KV usage, preemptions and cached tokens on a real engine | Colab or Kaggle T4; a 24 GB GPU for Qwen3-4B | as for 02 |
| `05_rl_rollouts_with_an_engine` | the same bookkeeping on the tiny transformer's rollouts | T1: vLLM generates the rollouts for `Qwen/Qwen2.5-0.5B-Instruct`, transformers takes one GRPO step | Colab or Kaggle T4 (the fit in 15 GB is `(verify)`); a 24 GB GPU with room | — |
| [`deploy/any-gpu`](00-foundations/rl-and-thinking-models/thinking-lab/deploy/any-gpu/) | `DRY_RUN=1` | T1 | `serve.sh` (Docker or pip), `rl_step.sh`; Colab/Kaggle T4 and 24 GB recipes, RunPod/Vast notes | — |
| [`deploy/gcp`](00-foundations/rl-and-thinking-models/thinking-lab/deploy/gcp/) | the tfvars example and the manifests checked offline by the lab's tests | T3 | — | the 04 serving lab's Cloud Run Terraform or GKE manifests set up for a thinking model (longer request timeout, lower concurrency); no Terraform here |

### 01 · [`gpu-bench-lab`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_measure_your_roofline` | the numpy backend measures your CPU's roofline (a real measurement) | T1: GEMM sweep by size and dtype | Colab T4 (fp32 and fp16 only); a 4090 for bf16 and FP8 | the L4 Spot VM |
| `02_memory_bandwidth_and_transfers` | CPU copy/scale/add/triad | T1: device bandwidth, pinned vs pageable host↔device | Colab T4 | the L4 Spot VM |
| `03_multi_gpu_topology_and_p2p` | parses bundled `nvidia-smi topo -m` sample output | T2: P2P bandwidth | Kaggle 2×T4 (PCIe, $0); 2× A100/H100 SXM for NVLink (~$1–6/hr) | `a2-highgpu-2g` (A100, NVLink; quota is hard) |
| `04_weights_loading_and_cold_start` | disk and safetensors load throughput on your machine | T1: load into GPU memory | Colab T4 | the L4 Spot VM, with `max_run_duration` |
| [`deploy/any-gpu`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/any-gpu/) | — | T1/T2 | Docker `--gpus all` or plain pip; notes for Colab, Kaggle, RunPod, Vast.ai, Lambda | — |
| [`deploy/gcp/terraform`](01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab/deploy/gcp/terraform/) | `terraform validate` and `plan` offline | T3 | — | the whole suite on a Spot L4 VM with `max_run_duration`, results uploaded to a bucket; `a2-highgpu-2g` for NVLink |

### 02 · [`cuda-nccl-lab`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_kernels_in_the_simulator` | Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`): correctness, not speed | — | — | — |
| `02_memory_bound_kernels_on_a_real_gpu` | the same kernels in the simulator | T1: timings against the roofline | Colab T4 | an L4 node |
| `03_collectives_with_torch_distributed` | the `pipes` backend (a real ring all-reduce across OS processes, no torch) or `gloo`: real semantics and real CPU timings | T2: the `nccl` backend | Kaggle 2×T4 | 2-GPU Job on `g2-standard-24` (2× L4, PCIe) |
| `04_busbw_and_the_alpha_beta_fit` | fits bundled `all_reduce_perf` sample output (illustrative), or your own `pipes`/gloo sweep | T2: nccl-tests | Kaggle 2×T4; an NVLink box for the contrast | nccl-tests Job on `g2-standard-24` |
| `05_how_a_container_sees_a_gpu` | runs in any Linux container and explains what is missing | T1 | any GPU notebook or container: Colab, Kaggle, RunPod, Vast.ai | `nvidia-smi` smoke Job on GKE |
| `06_gpu_sharing_and_dcgm_on_gke` | MIG placement packer and DCGM-exporter sample metrics | T1/T2 on a GPU VM (dcgm-exporter in Docker, MIG and MPS by hand); T3 | a Lambda or GCP GPU VM for MIG (A100/H100) | L4 Spot pool, optional time-sharing pool; the A100 MIG pool is off by default |
| [`deploy/any-gpu`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/), [`deploy/gke`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gke/) | — | T1/T2, T3 | Docker commands, nccl-tests build and run, the Kaggle 2×T4 recipe | smoke, CUDA sample, 2-GPU nccl-tests, MIG and time-sharing manifests |
| [`deploy/gcp/terraform`](02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/) | `terraform validate` and `plan` offline | T3 | — | the GKE cluster the `deploy/gke` Jobs run on (§5.4) |

### 03 · [`k8s-gpu-lab`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/)

No GPU is needed for any layer-03 notebook: the scheduler sees a GPU as an integer resource. Fake capacity cannot
show the device plugin injecting a real device, driver installation or DCGM — `deploy/gpu-vm` (T1) shows the first,
layer 02 the rest.

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_manifests_and_the_linter` | typed manifest builders and the GPU pod-spec linter | — | — | — |
| `02_kind_with_fake_gpus_and_kueue` | without a cluster, a bundled mini-simulator predicts outcomes and prints the commands | T0 + Docker: kind with fake GPUs, Kueue v0.19.6, JobSet, LWS | a laptop with Docker | — |
| `03_why_is_my_pod_pending` | bundled `kubectl get pod -o json` and event fixtures | any cluster | kind | GKE |
| `04_gke_pools_dws_and_computeclasses` | plan and inspect offline | T3 | — | zonal GKE, L4 Spot pool 0→N, optional flex-start queued-provisioning pool, ComputeClass fallbacks, the Kueue ProvisioningRequest admission check, GCS FUSE weights |
| [`deploy/kind`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/kind/) | — | T0 + Docker | a laptop with Docker: the real kube-scheduler, Kueue, JobSet and LWS against fake `nvidia.com/gpu` capacity, optionally hundreds of KWOK nodes | — |
| [`deploy/gpu-vm`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gpu-vm/) | — | T1 | any GPU VM you control (Lambda, a GCP L4 Spot VM): k3s plus the real NVIDIA device plugin, optional time-slicing; not a container pod (RunPod, Vast.ai) | — |
| [`deploy/gcp/terraform`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/), [`deploy/gke`](03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gke/) | `terraform validate` and schema checks offline | T3 | — | the cluster and manifests above (§5.4) |

### 04 · [`vllm-serving-lab`](04-inference-engine/serving-engine/vllm-serving-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_size_before_you_serve` | sizing from bundled `config.json` samples | — | — | — |
| `02_serve_and_measure` | `fakeserver.py`: an OpenAI-compatible server emulating vLLM timing and metrics (simulated) | T1: `vllm serve` with a 0.5–2B model | Colab or Kaggle T4: fp16 only (`--dtype half`), Triton attention backend | vLLM on a GKE L4 pool (`deploy/gcp/gke`) |
| `03_knobs_and_tradeoffs` | the fake server | T1; exercise 3.6 (tensor parallelism) T2 | T4; a 4090 or L4 for larger batches; Kaggle 2×T4 for 3.6 | — |
| `04_prefix_caching_for_agents` | the fake server's prefix-cache emulation | T1 | T4 | — |
| `05_speculation_and_quantization_in_vllm` | simulated | T1; FP8 needs compute capability 8.9+ | a 4090 (RunPod, Vast.ai) or a GCP L4 Spot VM; a T4 for n-gram speculation and weight-only INT4/INT8 | — |
| `06_deploy_on_cloud_run_gpu` | render and inspect the Terraform and the `gcloud run deploy` equivalent offline | T3 | — | Cloud Run with an L4, per-second billing, scale to zero |
| [`deploy/any-gpu`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/) | — | T1 | `vllm/vllm-openai` as the pod image on RunPod or Vast.ai; the Colab/Kaggle T4 recipe (fp16, Triton attention, compute capability 7.5 minimum) | — |
| [`deploy/gcp/cloud-run`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/), [`deploy/gcp/gke`](04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/) | `DRY_RUN=1`, `terraform validate` and schema checks offline | T3 | — | Cloud Run with an L4 scaling to zero; a vLLM Deployment on a GKE L4 pool (§5.4) |
| [`vllm-internals`](04-inference-engine/vllm-internals/README.md) notebook, [FlashAttention deep dive](04-inference-engine/flash-attention/flash-attention-deep-dive.md) notebook | standard library or numpy: the whole notebook | T1 to watch vLLM's metrics and logs, or to time attention kernels | Colab or Kaggle T4 | — |

### 04 · [`quant-lab`](04-inference-engine/quantization/quant-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_quantize_a_checkpoint` | the bundled tiny Llama quantized with the lab's numpy recipes, written as a compressed-tensors checkpoint and read back | T1: llm-compressor on `Qwen/Qwen2.5-0.5B-Instruct` | Colab or Kaggle T4 | — |
| `02_serve_and_compare_schemes` | a roofline emulator and a fake OpenAI-compatible server (simulated) | T1: `vllm serve` with FP16, INT4 and FP8 checkpoints | a T4 for INT4 (fp16) and INT8 W8A8; an RTX 4090 or L4 for FP8 | a quantized model on the serving lab's Cloud Run (one L4, scale to zero) or GKE, via `deploy/gcp` |
| `03_measure_the_accuracy_cost` | an offline mini-eval on the bundled model; lm-eval sample output (illustrative) | T1: `lm_eval` against a checkpoint or a running server | Colab or Kaggle T4 | — |
| `04_kv_cache_quantization_in_vllm` | the serving lab's memory model, reproduced; per-step times simulated; a bundled startup log (illustrative) | T1 on Ada or newer: `--kv-cache-dtype fp8` | an RTX 4090 or L4 (a T4 has no FP8 KV backend) | — |
| `05_fp4_and_the_blackwell_path` | calculators and emulation (simulated; datasheet peaks `(verify)`) | T1 on one Blackwell GPU with CUDA 12.8+ | a rented B200 or RTX PRO 6000 `(verify)` | Cloud Run offers the RTX PRO 6000 `(verify)` |
| [`deploy/any-gpu`](04-inference-engine/quantization/quant-lab/deploy/any-gpu/) | `DRY_RUN=1` | T1 | `compress.sh` (llm-compressor in its own virtualenv), `serve.sh` per scheme; the Colab/Kaggle T4 INT4 recipe; RunPod or Vast.ai 4090 or L4 for FP8 | — |
| [`deploy/gcp`](04-inference-engine/quantization/quant-lab/deploy/gcp/) | `DRY_RUN=1`; the tfvars validated against the serving lab's Terraform | T3 | — | the serving lab's Cloud Run service or GKE Deployment with a quantized model; no Terraform here |

### 05 · [`inference-gateway-lab`](05-orchestrator/serving-orchestration/inference-gateway-lab/)

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_router_in_process` | the router in front of in-process fake backends | — | — | — |
| `02_scorer_weights_and_hot_prefixes` | the same, with a shared-prefix agentic load | — | — | — |
| `03_autoscaling_recommender` | HPA recommendations from scraped fake metrics, and the generated manifests | T3 to apply them | — | an HPA on a Managed Prometheus metric |
| `04_local_stack_with_llm_d` | without Docker, a labelled fallback and the commands to run | T0 + Docker: compose (three simulated backends, the router, Prometheus) or kind (llm-d Router standalone with Envoy and `llm-d-inference-sim`); T1/T2: the lab router in front of real vLLM replicas | a laptop with Docker; for real backends, Colab or Kaggle T4 via [`deploy/any-gpu`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/any-gpu/) | — |
| `05_gke_inference_gateway` | plan and inspect offline | T3 | — | GKE with the Gateway API, a proxy-only subnet, an L4 Spot pool 0→N, InferencePool v1 with the endpoint picker, a `gke-l7-regional-external-managed` Gateway, InferenceObjective priorities |
| [`deploy/local`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/local/), [`deploy/kind`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/kind/) | — | T0 + Docker | a laptop with Docker | — |
| [`deploy/gcp/terraform`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/), [`deploy/gke`](05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/) | Terraform `validate` and CRD schema checks offline | T3 | — | ~$0.16/h with the GPU pool at 0, ~$0.44/h with one L4 Spot node `(verify)`; `deploy/gke` keeps one L4 node while installed |

### 07 · [`sandbox-lab`](07-application-agent-framework/sandboxed-execution/sandbox-lab/)

No GPU is needed anywhere in this lab: its tiers are T0, T0 + Docker and T3.

| Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3) |
|---|---|---|---|---|
| `01_hardened_containers` | the process sandbox and its attack probes, measured on your machine; the Docker and gVisor rungs print their commands and read bundled verdicts (illustrative) | T0 + Docker: the same probes against a hardened container; gVisor (`runsc`) on Linux | a laptop with Docker; gVisor needs Linux and sudo | — |
| `02_pod_per_execution_on_kind` | manifests validated against Kubernetes 1.34, an offline admission predictor, a simulated cluster that runs the code in the process sandbox | T0 + Docker: kind with restricted Pod Security, default-deny NetworkPolicy and the admission policy (no gVisor) | a laptop with Docker | — |
| `03_egress_proxy_and_secret_brokering` | the proxy, a stand-in upstream and a sandbox in-process over a Unix socket — real, not simulated | — | — | — |
| `04_an_agent_with_a_sandbox_tool` | a scripted model, the process sandbox and the proxy, in-process | — | — | — |
| `05_gke_sandbox_with_gvisor` | the Terraform read as text, the manifests validated, pool size and cost per execution computed | T3 | — | zonal GKE with private nodes and no NAT, a GKE Sandbox (gVisor) Spot pool from zero (§5.4) |
| [`deploy/docker`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/docker/), [`deploy/kind`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/kind/) | `DRY_RUN=1` | T0 + Docker | a laptop with Docker | — |
| [`deploy/gcp/terraform`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gcp/terraform/), [`deploy/gke`](07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gke/) | Terraform `validate` and `kubernetes-validate` offline; `python3 -m sandboxlab gke-review` | T3 | — | the cluster in §5.4 |

### The other topics in 00, 01, 04, 06 and 07

| Labs | Tier | Notes |
|---|---|---|
| 00 transformers, capacity planning, model landscape; 01 gpu-primer and gpu-deployment exercises; 04 kv-cache, paged-attention, flash-attention (practice and deep-dive notebooks) | T0 | numpy and matplotlib; the transformer walkthrough notebooks use CPU PyTorch (preinstalled on Colab) |
| 06 identity labs, scaling labs | T0 | `agentic-identity-gcp-lab`'s Terraform is an optional T3 step; the Mistral variants run fully offline on scripted or fake clients; a `MISTRAL_API_KEY` only switches on live calls (per-token cost, no GPU) and un-skips one test |
| 07 agent labs, long-running labs, retrieval labs | T0 | Gemini, Mistral, Anthropic or OpenAI keys are optional; `rag-from-scratch` embeds with a hashing embedder at T0 (labelled; not semantic, so its retrieval numbers are illustrative) and, with torch installed (T0 + torch, or Colab), with the ~90 MB `all-MiniLM-L6-v2` model on CPU — `sentence-transformers` pulls in torch, a multi-GB install; the long-running labs' GCP deploys are optional T3 steps |

---

## 7. Planning paid sessions

GPU time is cheapest when the T0 work is already done and several labs share one session.

| Session | Hardware | Cost | Run |
|---|---|---|---|
| A · one GPU | Colab or Kaggle T4; a rented 4090 or a GCP L4 Spot VM when you need FP8 | $0, or ~$0.3–0.7/hr | 01 lab 01, 02, 04 · 02 lab 02, 05 · 04 lab 02–05 and the vllm-internals observations · 05 lab 04 with one real vLLM backend · 00 MoE lab 02, 03, 05 · 00 thinking lab 02–05 · 04 quantization lab 01–04 (04 needs FP8) |
| B · two GPUs over PCIe | Kaggle 2×T4 | $0 | 01 lab 03 · 02 lab 03, 04 (and the nccl-tests recipe) · 04 lab exercise 3.6 (tensor parallelism) · 05 lab 04 with two vLLM replicas · 00 MoE lab 04 (tensor vs expert parallelism) |
| C · NVLink | 2× A100 or H100 SXM on RunPod, Vast.ai or Lambda, about an hour | ~$1–6 | session B again, to compare P2P and busbw with PCIe; optionally a vLLM run with `--tensor-parallel-size 2` to see the all-reduce cost in ITL |
| D · GCP, one lab at a time | each lab's Terraform | GPU at the Spot price plus the cluster or load-balancer overhead | 01 VM suite · 04 Cloud Run (also with a quantized or thinking model) · 02, 03, 05 on GKE (the MoE lab's EP Deployment rides on 02's) · 07 GKE Sandbox (no GPU); destroy before starting the next |
| E · Blackwell | one rented B200 or RTX PRO 6000 `(verify)` | per hour at the provider's price | 04 quantization lab 05 (NVFP4 W4A4) |

---

## 8. Cost-safety habits

**Before a session**

- Finish the T0 path first, and write down the commands and the numbers you expect; the paid session only measures.
- GCP: create a budget with alerts at 50, 90 and 100 % on the billing account. Budgets alert; they do not stop
  spending.

```bash
gcloud billing budgets create --billing-account=XXXXXX-XXXXXX-XXXXXX \
  --display-name="learning-labs" --budget-amount=20USD \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

- Keep secrets out of git — the repo is public. Hugging Face tokens go in environment variables or Secret Manager,
  never in a notebook. Before your first commit in a deploy directory, confirm that `terraform.tfvars` and state are
  ignored: `git check-ignore -v deploy/gcp/terraform/terraform.tfvars`.

**During a session**

- Give every VM an end: `scheduling.max_run_duration` with `instance_termination_action = "DELETE"` in Terraform, or
  `--max-run-duration=2h --instance-termination-action=DELETE` on `gcloud compute instances create`. GKE node pools
  have `node_config.max_run_duration` too.
- Prefer scale-to-zero: Cloud Run `min_instance_count = 0`; GKE GPU pools with a minimum of 0 nodes. A Deployment
  with one replica keeps a GPU node alive: `kubectl scale deployment/<name> --replicas=0` when you are done.
- Save results as you go (the layer 01 VM uploads its JSON to GCS), since Spot can end the session at any moment.
- On RunPod and Vast.ai, end the script by stopping or destroying the instance through the provider's CLI
  (`runpodctl`, `vastai`) `(verify the commands)`, so a finished run does not sit idle.

**After a session**

- `terraform destroy` in the directory that holds the state. Do not delete the state before destroying.
- Sweep for leftovers:

```bash
gcloud compute instances list
gcloud compute disks list --filter="-users:*"     # unattached disks, e.g. from PVCs
gcloud compute addresses list
gcloud compute forwarding-rules list
gcloud container clusters list
gcloud run services list
gcloud storage buckets list
```

- RunPod: *terminate* the pod, not just stop it, and delete network volumes you no longer need. Vast.ai: *destroy*
  the instance. Lambda: terminate the instance and delete unused persistent storage. Modal: check that no deployed
  app keeps containers warm (`modal app stop <app>` `(verify)`). Colab and Kaggle: end the runtime or session to save
  quota.
- Last resort on GCP: shut the lab's project down (`gcloud projects delete <project>`); billing for it stops, and
  the project can be restored for a limited period `(verify)`.

---

## 9. Verify list — 2026-09-26

| Item | Value used here | Where it matters |
|---|---|---|
| Colab free tier | T4 16 GB; ~15–30 GPU-h/week, not guaranteed; ~12 h sessions; paid tiers' GPUs | §3.2, T1 |
| Kaggle | 2×T4 or P100; 30 GPU-h/week; Internet setting needs a verified account; P2P between the two T4s | §3.3, T1/T2 |
| RunPod | RTX 4090 from ~$0.34/hr, A100 ~$1.6/hr, H100 PCIe ~$2.9/hr; per-second billing; NVLink pods; stopped pods bill volume storage; `runpodctl` | §4, §8 |
| Vast.ai | RTX 4090 ~$0.3–0.4/hr, A100 80 GB ~$0.5–1/hr, H100 ~$1.5–1.9/hr on verified hosts; billing granularity; storage billed when stopped; `vastai` | §4, §8 |
| Lambda | A100 40 GB ~$2/hr, H100 ~$3.3/hr; billing granularity; 8× SXM NVLink instances; terminate-only; preinstalled stack | §4, §8 |
| Modal | monthly free credit amount; per-second billing; GPUs per container; `modal app stop` | §4, §8 |
| GCP billing | Free Trial cannot use GPUs; upgrading keeps credits; trial usable for CPU-only work | §5.1, §3.1 |
| GCP GPU quota | `GPUS_ALL_REGIONS` plus per-region, per-model metrics; preemptible metrics for Spot; Cloud Run GPU quota; console path; approval times | §5.1 |
| GCP GPU prices (us-central1) | T4 ~$0.35–0.55/hr; `g2-standard-4` ~$0.70/hr; Spot 60–91 % off; `a2-highgpu-1g` ~$3.7/hr; `a2-ultragpu-1g` ~$5/hr; `a3-highgpu-8g` ≈ $88/hr; H100 Spot ≈ $3.7/GPU-hr; G4 prices | §5.2 |
| A3 shapes | on-demand H100 only as `a3-highgpu-8g`; 1g/2g/4g only via Spot or flex-start | §5.2 |
| Spot | preemption notice (~30 s); separate quota | §5.3 |
| DWS | flex-start run limit (7 days), discount and quota; queued provisioning via ProvisioningRequest; calendar-mode durations and lead time | §5.3 |
| GKE | management fee and the free tier's zonal-cluster credit; automatic driver install from 1.32.2-gke.1297000; GPU node taint; autoscaler scale-down delay; Inference Gateway GatewayClass names; load-balancer pricing | §5.4 |
| Cloud Run GPU | L4 (min 4 vCPU/16 GiB) and RTX PRO 6000 (min 20 vCPU/80 GiB); per-second, instance-based billing; idle time before scale-in; zonal-redundancy pricing; regions | §5.4 |
| GPU capabilities | the §2 table (memory, bandwidth, compute capability, BF16/FP8, NVLink, MIG incl. RTX PRO 6000); vLLM v0.30.0 minimum compute capability 7.5 and its Triton attention fallback on a T4; FP8 and INT4/INT8 kernels in the pinned vLLM | §2, §6 |
| TPUs | v7 Ironwood GA 2026-04-22, 192 GB HBM and ~4.6 PFLOPS FP8 per chip; vLLM TPU backend name and status | §5.5 |
| Tooling | Terraform google provider 8.4.0 (labs pin `>= 8.0`); Kueue v0.19.6; `gcloud billing budgets create` flags; project-deletion recovery window | §6, §8 |
| MoE lab | model ids (`allenai/OLMoE-1B-7B-0924(-Instruct)`, `Qwen/Qwen1.5-MoE-A2.7B` and its GPTQ-Int4, `Qwen/Qwen3-30B-A3B-GPTQ-Int4`, the granite-3.0 MoE models); vLLM v0.30.0 `--enable-expert-parallel`, `--cpu-offload-gb`, `--enable-return-routed-experts`; MXFP4 (gpt-oss) needs compute capability 8.0, so not a T4; PCIe peer bandwidths on Kaggle's 2×T4 | §6 (moe-lab) |
| Thinking lab | model ids (`Qwen/Qwen3-0.6B`, `-1.7B`, `-4B`, `-4B-Thinking-2507`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen2.5-0.5B-Instruct`); vLLM v0.30.0 `--reasoning-parser` and the `reasoning` response field; TRL 1.14.0 `GRPOConfig` defaults; Qwen3 in fp16 on a T4; the GRPO step's fit in 15 GB; Cloud Run's maximum request timeout | §6 (thinking-lab) |
| Quantization lab | llm-compressor 0.14.0, compressed-tensors 0.19.0, lm-eval 0.4.13, GPTQModel 7.5.0; `--quantization fp8` works at vLLM 0.30.0 and raises at `main` (use `fp8_per_tensor`); bitsandbytes and GGUF as out-of-tree plugins; kernel floors (FP8 W8A8 from compute capability 8.9, NVFP4 W4A4 on 10.x–12.x with CUDA 12.8+, INT8 W8A8 not on 10.0+, no FP8 KV backend on 7.5); pre-quantized model ids; Blackwell rental prices | §2, §6 (quant-lab) |
| Sandboxes | gVisor's install layout and `runsc` release channel; GKE Sandbox `sandbox_config.type = "GVISOR"` (case-sensitive in the provider), the `gvisor` RuntimeClass GKE creates and its node taint; kind v0.33.0 with node image K8s 1.34.11 and kindnetd's NetworkPolicy enforcement (it fails open); `e2-standard-2` ~$0.07/h on demand; GKE Sandbox carries no surcharge | §5.4, §6 (sandbox-lab) |
