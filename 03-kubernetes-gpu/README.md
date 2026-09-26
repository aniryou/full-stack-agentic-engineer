# 03 · Kubernetes + GPU Operator + scheduler

Cluster-level resource management: how GPU nodes are exposed, scheduled, and shared
across workloads.

**Covers:** the NVIDIA GPU Operator, device plugins, node feature discovery,
scheduling for GPUs (bin-packing, gang scheduling, topology-aware placement,
Volcano/Kueue), taints/tolerations, NUMA alignment, multi-tenancy & quotas at the
cluster level, node autoscaling of GPU pools.

**Signal keywords:** Kubernetes, GPU Operator, device plugin, gang scheduling, Volcano,
Kueue, topology-aware, bin-packing, node pool, taint, NUMA, cluster autoscaler.

## Current contents

### `gpu-scheduling/` — Kubernetes for GPUs
How a GPU becomes schedulable, and how to place, share, queue and scale it: the device plugin and DRA,
the scheduler's filter/score/preempt cycle and GPU fragmentation, gangs and topology-aware placement,
Kueue quotas with cohort borrowing and reclaim, and getting GPU capacity — autoscaling from zero, Spot,
DWS flex-start, startup latency, sharing. Every concept runs on a laptop (T0); a kind cluster with fake
GPUs adds the real control plane, and GKE is one optional production step, never a prerequisite. Start
at the topic's [`README.md`](gpu-scheduling/README.md).
- **[`PRIMER.md`](gpu-scheduling/PRIMER.md)** — ten sections: what Kubernetes sees (extended resources,
  the device plugin API, labels, taints, DRA), GPU Operator vs managed drivers, the scheduling cycle
  (filter, score, fragmentation and stranded GPUs, preemption), gangs, topology-aware placement, Kueue
  queues and quotas, getting capacity, startup latency, sharing GPUs, learning locally — then a
  design-review walkthrough, drills, glossary, sources and a dated verify list. Worked numbers come from
  `k8s-gpu-core`, with the function named next to each.
- **[`k8s-gpu-core/`](gpu-scheduling/k8s-gpu-core/)** — *T0, standard library only.* The minimal
  implementation (package `gpusched`): a simulator of the device plugin and kubelet, kube-scheduler's
  filters, scores and preemption, gangs and Kueue TAS, Kueue quotas with borrowing, lending and reclaim,
  and a GPU cluster autoscaler (scale from zero, Spot restarts, queued provisioning, startup latency);
  five fill-in notebooks.
- **[`k8s-gpu-lab/`](gpu-scheduling/k8s-gpu-lab/)** — *T0 → T3.* The detailed lab (package `k8sgpu`):
  typed manifest generators (Job, JobSet, LeaderWorkerSet, Kueue, DRA, ComputeClass), a GPU pod-spec
  linter, a "why is my pod Pending?" explainer with fixtures, a capacity-type chooser and an offline
  predictor for every kind scenario (T0) → a kind cluster with fake `nvidia.com/gpu` capacity and the
  real kube-scheduler, Kueue v0.19.6, JobSet and LeaderWorkerSet, on a laptop with Docker (T0 + Docker)
  → k3s with the real NVIDIA device plugin on one GPU VM (T1/T2) → GKE via Terraform (T3): a Spot L4
  pool from zero, DWS flex-start queued provisioning through Kueue's ProvisioningRequest check,
  ComputeClass fallbacks, GCS FUSE weights, time-sharing. Four notebooks.

**Suggested order:** a primer section, then its core notebook, then (optionally) the lab notebook —
§1–2 → core `01`, lab `01`; §3 → core `02`, lab `03`; §4–6 → core `03`–`04`, lab `02`; §7–8 → core
`05`, lab `04`; §9–10 → core `01` exercise 1.6, the lab's `deploy/gpu-vm`, `deploy/gke` time-sharing
and `deploy/kind`. The topic README has the step table.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the primer; all five core notebooks; the lab's generators, linter, Pending explainer, capacity model and kind predictor | $0 |
| **T0 + Docker** | a laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue, JobSet and LWS against fake GPUs, optionally hundreds of KWOK nodes | $0 |
| **T1 / T2** | any GPU VM you control (Lambda, a GCP VM) | the lab's `deploy/gpu-vm`: k3s + the real NVIDIA device plugin, GPU Feature Discovery labels, time-slicing | the VM's hourly price |
| **T3** | GKE, via the lab's Terraform | zonal GKE Standard, Spot L4 pool 0→N, DWS flex-start through Kueue, ComputeClass, GCS FUSE, time-sharing | pay per use (~$0.23/h idle + ~$0.25/h per busy Spot L4 node) |

**Cross-references.** Builds on layer 01 — the
[GPU deployment primer](../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §7
(Kubernetes specifics in brief; §4 the parallelism menu) and
[`roofline-and-fabric/`](../01-hardware-gpu-fabric/roofline-and-fabric/README.md) (fabric bandwidth and
why topology matters, cold-start arithmetic, checkpoint intervals, GPU families and obtainability) — and
layer 02 ([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md): how a container gets a GPU; MIG,
time-slicing and MPS mechanics; DCGM health). Leads to layer 05
([`05-orchestrator`](../05-orchestrator/README.md): autoscaling replicas and LeaderWorkerSet groups on
queue and SLO signals, prefill/decode disaggregation) and layer 06
([`agentic-scaling-lab`](../06-gateway/scaling-admission-cost/agentic-scaling-lab/): admission control
and cost at the gateway — "shape demand to capacity" one layer up from Kueue's quotas).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`gpu-scheduling/k8s-gpu-core/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) `01_how_kubernetes_sees_a_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/02_filter_score_and_fragmentation.ipynb) `02_filter_score_and_fragmentation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/03_gangs_and_topology.ipynb) `03_gangs_and_topology.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/04_queues_quotas_and_preemption.ipynb) `04_queues_quotas_and_preemption.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/05_autoscaling_and_obtainability.ipynb) `05_autoscaling_and_obtainability.ipynb`

**`gpu-scheduling/k8s-gpu-core/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/01_how_kubernetes_sees_a_gpu.ipynb) `01_how_kubernetes_sees_a_gpu.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/02_filter_score_and_fragmentation.ipynb) `02_filter_score_and_fragmentation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/03_gangs_and_topology.ipynb) `03_gangs_and_topology.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/04_queues_quotas_and_preemption.ipynb) `04_queues_quotas_and_preemption.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/05_autoscaling_and_obtainability.ipynb) `05_autoscaling_and_obtainability.ipynb`

**`gpu-scheduling/k8s-gpu-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/01_manifests_and_the_linter.ipynb) `01_manifests_and_the_linter.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) `02_kind_with_fake_gpus_and_kueue.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/03_why_is_my_pod_pending.ipynb) `03_why_is_my_pod_pending.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/04_gke_pools_dws_and_computeclasses.ipynb) `04_gke_pools_dws_and_computeclasses.ipynb`

**`gpu-scheduling/k8s-gpu-lab/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/01_manifests_and_the_linter.ipynb) `01_manifests_and_the_linter.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/02_kind_with_fake_gpus_and_kueue.ipynb) `02_kind_with_fake_gpus_and_kueue.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/03_why_is_my_pod_pending.ipynb) `03_why_is_my_pod_pending.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/04_gke_pools_dws_and_computeclasses.ipynb) `04_gke_pools_dws_and_computeclasses.ipynb`
<!-- colab-links:end -->
