# 03 · Kubernetes and GPU scheduling

Make GPUs schedulable: after this layer you can explain how a GPU becomes an integer Kubernetes can schedule, why a
GPU pod is Pending and how to fix it, how to place gangs without deadlock, how to share a cluster between teams with
Kueue quotas, and how to get GPU capacity — from zero, on Spot or through queued provisioning — and start it fast.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the cluster: it takes the GPUs the runtime (02) exposes and hands them to the engines (04) and
fleets (05) above.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`gpu-scheduling/`](gpu-scheduling/README.md) | read a `FailedScheduling` message and fix the pod; measure GPU fragmentation; place a gang in the smallest topology domain; write Kueue ClusterQueues with borrowing and predict who is preempted; choose between Spot, DWS flex-start and reservations and budget startup latency — with a primer, a pure-Python simulator, a kind cluster with fake GPUs, one real GPU VM and GKE | ~9 h primer + core; ~5 h lab (+2 h on GKE) | T0 → T3 |

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* "T0 + Docker" is a laptop with Docker, still free.
Times are rough and come from the repo's curriculum ([`CURRICULUM.md`](../CURRICULUM.md), modules 03.1–03.6).

## Start here

1. Read [`gpu-scheduling/PRIMER.md`](gpu-scheduling/PRIMER.md) §1: what Kubernetes sees.
2. `cd gpu-scheduling/k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 59 tests,
   ~25 s; then open
   [`01_how_kubernetes_sees_a_gpu`](gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb).
3. Follow the step table in the topic [`README.md`](gpu-scheduling/README.md): each primer section pairs with a core
   notebook and, optionally, a lab notebook.

## What is inside `gpu-scheduling/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](gpu-scheduling/PRIMER.md) | ten sections: what Kubernetes sees (extended resources, the device plugin API, labels, taints, DRA), GPU Operator vs managed drivers, the scheduling cycle (filter, score, fragmentation and stranded GPUs, preemption), gangs, topology-aware placement, Kueue queues and quotas, getting capacity, startup latency, sharing GPUs, learning locally — then a design-review walkthrough, drills, glossary, sources and a dated verify list. Worked numbers come from `k8s-gpu-core`, with the function named next to each | read |
| [`k8s-gpu-core/`](gpu-scheduling/k8s-gpu-core/) | the minimal implementation, package `gpusched`, standard library only: a simulator of the device plugin and kubelet, kube-scheduler's filters, scores and preemption, gangs and Kueue TAS, Kueue quotas with borrowing, lending and reclaim, and a GPU cluster autoscaler (scale from zero, Spot restarts, queued provisioning, startup latency); five fill-in notebooks | T0 |
| [`k8s-gpu-lab/`](gpu-scheduling/k8s-gpu-lab/) | the detailed lab, package `k8sgpu`: typed manifest generators (Job, JobSet, LeaderWorkerSet, Kueue, DRA, ComputeClass), a GPU pod-spec linter, a "why is my pod Pending?" explainer with fixtures, a capacity-type chooser and an offline predictor for every kind scenario (T0) → a kind cluster with fake `nvidia.com/gpu` capacity and the real kube-scheduler, Kueue v0.19.6, JobSet and LeaderWorkerSet (T0 + Docker) → k3s with the real NVIDIA device plugin on one GPU VM (T1/T2) → GKE via Terraform (T3): a Spot L4 pool from zero, DWS flex-start queued provisioning through Kueue's ProvisioningRequest check, ComputeClass fallbacks, GCS FUSE weights, time-sharing. Four notebooks | T0 → T3 |

Suggested order — a primer section, then its core notebook, then (optionally) the lab notebook: §1–2 → core `01`,
lab `01`; §3 → core `02`, lab `03`; §4–6 → core `03`–`04`, lab `02`; §7–8 → core `05`, lab `04`; §9–10 → core `01`
exercise 1.6, the lab's `deploy/gpu-vm`, `deploy/gke` time-sharing and `deploy/kind`.

## Run it

```bash
cd gpu-scheduling/k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../k8s-gpu-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
python3 -m k8sgpu kind predict s2   # the predictor's step-by-step outcome for a gang scenario (simulated)
```

Then `python3 -m jupyterlab notebooks` in either directory, or the Colab links below. With Docker, the lab's
[`deploy/kind`](gpu-scheduling/k8s-gpu-lab/deploy/kind/README.md) runs the same scenarios on a real control plane.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the primer; all five core notebooks; the lab's generators, linter, Pending explainer, capacity model and kind predictor | $0 |
| **T0 + Docker** | a laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue, JobSet and LWS against fake GPUs, optionally hundreds of KWOK nodes | $0 |
| **T1 / T2** | any GPU VM you control (Lambda, a GCP VM) | the lab's `deploy/gpu-vm`: k3s + the real NVIDIA device plugin, GPU Feature Discovery labels, time-slicing | the VM's hourly price |
| **T3** | GKE, via the lab's Terraform | zonal GKE Standard, Spot L4 pool 0→N, DWS flex-start through Kueue, ComputeClass, GCS FUSE, time-sharing | pay per use (~$0.23/h idle + ~$0.25/h per busy Spot L4 node) |

## How it fits

**Needed first:** layer 01 — the [GPU deployment primer](../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md)
§7 (Kubernetes specifics in brief; §4 the parallelism menu) and
[`roofline-and-fabric/`](../01-hardware-gpu-fabric/roofline-and-fabric/README.md) (fabric bandwidth and why topology
matters, cold-start arithmetic, checkpoint intervals, GPU families and obtainability) — and layer 02
([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md): how a container gets a GPU; MIG, time-slicing and MPS
mechanics; DCGM health). The [curriculum's spiral](../CURRICULUM.md#31-why-this-order) brings you here after
layer 04 has been measured on a GPU, so the pods being scheduled are engines you have already run; the engine itself
is not a prerequisite. **Leads to** layer 05 ([`05-orchestrator`](../05-orchestrator/README.md): autoscaling replicas
and LeaderWorkerSet groups on queue and SLO signals, prefill/decode disaggregation) and layer 06
([`agentic-scaling-lab`](../06-gateway/scaling-admission-cost/agentic-scaling-lab/): admission control and cost at the
gateway — "shape demand to capacity" one layer up from Kueue's quotas).

## Caveats

- The core's outputs are **simulated**; durations, prices, stockout and preemption rates are illustrative inputs.
  On kind the scheduler, Kueue and every event are real, but the GPUs are an extended resource patched into node
  status: no device plugin, no `/dev/nvidia*`, no CUDA.
- RunPod and Vast.ai rent containers, not nodes, so they cannot teach this layer; the T1/T2 path needs a VM.
- Product versions, GKE details and prices are as of September 2026 and marked (verify).
- Not covered yet: NUMA and CPU-manager alignment, Volcano beyond a table row, MultiKueue, GPU Operator install and
  upgrade mechanics, node health remediation — see
  [`CURRICULUM.md` §2](../CURRICULUM.md#2-what-each-layer-has-and-what-it-does-not-cover-yet).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`gpu-scheduling/k8s-gpu-core/`** — [01_how_kubernetes_sees_a_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) · [02_filter_score_and_fragmentation](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/02_filter_score_and_fragmentation.ipynb) · [03_gangs_and_topology](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/03_gangs_and_topology.ipynb) · [04_queues_quotas_and_preemption](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/04_queues_quotas_and_preemption.ipynb) · [05_autoscaling_and_obtainability](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/05_autoscaling_and_obtainability.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/01_how_kubernetes_sees_a_gpu.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/02_filter_score_and_fragmentation.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/03_gangs_and_topology.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/04_queues_quotas_and_preemption.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/05_autoscaling_and_obtainability.ipynb)
- **`gpu-scheduling/k8s-gpu-lab/`** — [01_manifests_and_the_linter](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/01_manifests_and_the_linter.ipynb) · [02_kind_with_fake_gpus_and_kueue](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) · [03_why_is_my_pod_pending](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/03_why_is_my_pod_pending.ipynb) · [04_gke_pools_dws_and_computeclasses](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/04_gke_pools_dws_and_computeclasses.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/01_manifests_and_the_linter.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/02_kind_with_fake_gpus_and_kueue.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/03_why_is_my_pod_pending.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/04_gke_pools_dws_and_computeclasses.ipynb)
<!-- colab-links:end -->
