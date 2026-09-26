# gpu-scheduling — Kubernetes for GPUs

**How a GPU becomes schedulable, and how to place, share, queue and scale it.** The layer between the
GPU software substrate (layer 02: driver, CUDA, container runtime) and the inference engine (layer 04):
the device plugin and DRA, the scheduler's filter/score/preempt cycle and GPU fragmentation, gangs and
topology-aware placement, Kueue quotas with borrowing and reclaim, and getting GPU capacity — autoscaling
from zero, Spot, DWS flex-start, startup latency, sharing.

Every concept here is learnable on a laptop (T0). A kind cluster with fake GPU capacity adds the real
scheduler and Kueue (T0 + Docker). GKE is the optional production step (T3).

## What's here

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](PRIMER.md) | the concept primer shared by core and lab: ten numbered sections, a design-review walkthrough with drill questions, glossary, sources and a dated verify list | – |
| [`k8s-gpu-core/`](k8s-gpu-core/) | **minimal**: `gpusched`, a pure-Python simulator of the device plugin, scheduler, gangs + Kueue TAS, Kueue quotas and a GPU cluster autoscaler; five fill-in notebooks | T0 |
| [`k8s-gpu-lab/`](k8s-gpu-lab/) | **detailed**: `k8sgpu` — typed manifest builders (Job, JobSet, LWS, Kueue, DRA, ComputeClass), a GPU pod-spec linter, a "why is my pod Pending?" analyser, a capacity-type chooser; `deploy/kind` (fake GPUs, Kueue, JobSet, LWS), `deploy/gcp/terraform` and `deploy/gke` | T0 → T3 |

## Order to work it

Read the primer section, then do the core notebook, then (optionally) the lab notebook.

| Step | Primer | Core notebook (T0) | Lab notebook |
|---|---|---|---|
| 1 | §1 What Kubernetes sees · §2 GPU Operator vs managed drivers | `01_how_kubernetes_sees_a_gpu` | `01_manifests_and_the_linter` (T0) |
| 2 | §3 The scheduling cycle | `02_filter_score_and_fragmentation` | `03_why_is_my_pod_pending` (T0) |
| 3 | §4 Gangs · §5 Topology-aware placement | `03_gangs_and_topology` | `02_kind_with_fake_gpus_and_kueue` (T0 + Docker; simulator fallback) |
| 4 | §6 Queues, quotas and multi-tenancy with Kueue | `04_queues_quotas_and_preemption` | `02_kind_with_fake_gpus_and_kueue` |
| 5 | §7 Getting capacity · §8 Startup latency | `05_autoscaling_and_obtainability` | `04_gke_pools_dws_and_computeclasses` (T3; offline it plans and inspects) |
| 6 | §9 Sharing GPUs at the cluster level · §10 Learning locally | (01 covers time-slicing replicas) | `deploy/kind` in the lab |

About 9 hours for the primer and the core, 5 more for the lab's T0 path.

## Tiers

| Tier | Where | What you get here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the whole core; the lab's manifests, linter and Pending analyser; the lab's kind notebook falls back to a simulator without Docker | $0 |
| **T0 + Docker** | laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue v0.19.6, JobSet and LWS against fake `nvidia.com/gpu` capacity (optionally hundreds of KWOK nodes) | $0 |
| T1 / T2 | any GPU VM you control (Lambda, a GCP VM) with k3s or kubeadm and the GPU Operator | the real device plugin, GPU Feature Discovery labels, MIG or time-slicing on one box | per [`COMPUTE.md`](../../COMPUTE.md) |
| **T3** | GKE via the lab's Terraform | a zonal cluster, a Spot L4 pool from zero with driver auto-install, flex-start queued provisioning, image streaming, GCS FUSE; ComputeClass and Kueue ProvisioningRequest manifests | pay per use; L4 Spot, scale to zero, destroy after |

RunPod and Vast give you containers, not nodes, so they cannot teach this layer; use them for layers 01, 02
and 04. Prices and GPU obtainability: [`COMPUTE.md`](../../COMPUTE.md). Where this sits in the whole
course: [`CURRICULUM.md`](../../CURRICULUM.md).

## Builds on, and leads to

* Layer 01 — [`roofline-and-fabric/PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §5
  (fabric bandwidth: why topology matters), §6 (cold-start arithmetic), §7 (checkpoint intervals), §10
  (GPU families and obtainability); [`gpu-deployment-primer.md`](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md)
  §4 and §7 (the parallelism menu; Kubernetes specifics in brief).
* Layer 02 — `02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md` §6 (how a container gets a GPU), §7 (MIG,
  time-slicing, MPS mechanics), §8 (health and DCGM).
* Layer 05 — the orchestrator that scales LeaderWorkerSet groups and replicas on queue and SLO signals.
* Layer 06 — [`agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/): admission
  control and cost at the gateway, the same "shape demand to capacity" idea one layer up.
