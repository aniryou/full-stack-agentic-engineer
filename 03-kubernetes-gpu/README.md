# 03 · Kubernetes and GPU scheduling

Make GPUs schedulable. After this layer, you can explain these things:

- How a GPU becomes an integer that Kubernetes can schedule.
- Why a GPU pod is Pending, and how to repair it.
- How to place gangs without deadlock.
- How to share a cluster between teams with Kueue quotas.
- How to get GPU capacity (from zero, on Spot or through queued provisioning) and start it fast.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, memory, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, model routing, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the cluster. It takes the GPUs that the runtime (02) makes available. It gives them to the engines
(04) and the fleets (05) above it.

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`gpu-scheduling/`](gpu-scheduling/README.md) | read a `FailedScheduling` message and repair the pod. You can also measure GPU fragmentation and place a gang in the smallest topology domain. You can write Kueue ClusterQueues with borrowing, and predict which workloads Kueue preempts. You can select between Spot, DWS flex-start and reservations, and make a budget for the startup latency. The topic has a primer, a pure-Python simulator, a kind cluster with fake GPUs, one real GPU VM and GKE. | ~9 h primer + core, ~5 h lab (+2 h on GKE) | T0 to T3 |

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab or Kaggle T4, or a rented card). T2
is a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.*

"T0 + Docker" is a laptop with Docker, also at no cost. The times are approximate. They come from the repo's curriculum
([`CURRICULUM.md`](../CURRICULUM.md), modules 03.1–03.6).

## Start here

1. Read [`gpu-scheduling/PRIMER.md`](gpu-scheduling/PRIMER.md) §1: what Kubernetes sees.
2. Run `cd gpu-scheduling/k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. It
   runs 59 tests in ~25 s. Then open
   [`01_how_kubernetes_sees_a_gpu`](gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb).
3. Use the step table in the topic [`README.md`](gpu-scheduling/README.md). Each primer section goes with a core
   notebook and, as an option, a lab notebook.

## What is inside `gpu-scheduling/`

| Path | What it is | Tier |
|---|---|---|
| [`PRIMER.md`](gpu-scheduling/PRIMER.md) | ten sections. The first is what Kubernetes sees: extended resources, the device plugin API, labels, taints and DRA. The second compares GPU Operator and managed drivers. The third is the scheduling cycle: filter, score, fragmentation and stranded GPUs, and preemption. The next three are gangs, topology-aware placement, and Kueue queues and quotas. The last four are how to get capacity, startup latency, how to share GPUs and how to learn locally. After them come a walkthrough for a design review, drills, a glossary, sources and a dated verify list. The worked numbers come from `k8s-gpu-core`, and the primer names the function next to each number. | read |
| [`k8s-gpu-core/`](gpu-scheduling/k8s-gpu-core/) | the minimal implementation, package `gpusched`, with the standard library only. It has a simulator of the device plugin and kubelet, of kube-scheduler's filters, scores and preemption, and of gangs and Kueue TAS. The simulator also has Kueue quotas with borrowing, lending and reclaim, and a GPU cluster autoscaler. The autoscaler has scale from zero, Spot restarts, queued provisioning and startup latency. The core has five fill-in notebooks. | T0 |
| [`k8s-gpu-lab/`](gpu-scheduling/k8s-gpu-lab/) | the detailed lab, package `k8sgpu`. At T0, it has typed manifest generators (Job, JobSet, LeaderWorkerSet, Kueue, DRA, ComputeClass) and a GPU pod-spec linter. At T0, it also has a "why is my pod Pending?" explainer with fixtures, a capacity-type chooser and an offline predictor for every kind scenario. At T0 + Docker, it has a kind cluster with fake `nvidia.com/gpu` capacity, the real kube-scheduler, Kueue v0.19.6, JobSet and LeaderWorkerSet. At T1/T2, it has k3s with the real NVIDIA device plugin on one GPU VM. At T3, it has GKE through Terraform: a Spot L4 pool from zero, and DWS flex-start queued provisioning through Kueue's ProvisioningRequest check. The T3 path also has ComputeClass fallbacks, GCS FUSE weights and time-sharing. The lab has four notebooks. | T0 to T3 |

The suggested order is a primer section, then its core notebook, then the lab notebook, which is optional:

- §1–2: core `01`, lab `01`.
- §3: core `02`, lab `03`.
- §4–6: core `03` and `04`, lab `02`.
- §7–8: core `05`, lab `04`.
- §9–10: core `01` exercise 1.6, the lab's `deploy/gpu-vm`, `deploy/gke` time-sharing and `deploy/kind`.

## Run it

```bash
cd gpu-scheduling/k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q
cd ../k8s-gpu-lab && python3 -m pip install -r requirements.txt && python3 -m pip install -e . && python3 -m pytest -q
python3 -m k8sgpu kind predict s2   # the predictor's step-by-step outcome for a gang scenario (simulated)
```

Then run `python3 -m jupyterlab notebooks` in one of the two directories, or use the Colab links in "Run in Colab".
With Docker, the lab's [`deploy/kind`](gpu-scheduling/k8s-gpu-lab/deploy/kind/README.md) runs the same scenarios on a
real control plane.

| Tier | Where | What you do in this topic | Cost (Sep 2026, verify) |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the primer, all five core notebooks, and the lab's generators, linter, Pending explainer, capacity model and kind predictor | $0 |
| **T0 + Docker** | a laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue, JobSet and LWS against fake GPUs. As an option, you can add hundreds of KWOK nodes. | $0 |
| **T1 / T2** | any GPU VM you control (Lambda, a GCP VM) | the lab's `deploy/gpu-vm`: k3s with the real NVIDIA device plugin, GPU Feature Discovery labels, time-slicing | the VM's hourly price |
| **T3** | GKE, through the lab's Terraform | zonal GKE Standard, a Spot L4 pool that scales from 0 to N, DWS flex-start through Kueue, ComputeClass, GCS FUSE, time-sharing | pay per use (~$0.23/h idle + ~$0.25/h per busy Spot L4 node) |

## How it fits

**Needed first:** layer 01 and layer 02. From layer 01, you need the
[GPU deployment primer](../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §7 (Kubernetes specifics
in brief, and §4 for the parallelism menu). You also need
[`roofline-and-fabric/`](../01-hardware-gpu-fabric/roofline-and-fabric/README.md): fabric bandwidth and why topology
is important, cold-start arithmetic, checkpoint intervals, GPU families and obtainability. From layer 02
([`02-cuda-nccl-runtime`](../02-cuda-nccl-runtime/README.md)), you need these subjects: how a container gets a GPU,
the mechanics of MIG, time-slicing and MPS, and DCGM health.

The [curriculum's spiral](../CURRICULUM.md#31-why-this-order) brings you here after you measure layer 04 on a GPU.
Thus the pods that you schedule here are engines that you ran before. You do not need to know the engine itself first.

**Leads to** layer 05 and layer 06. Layer 05 ([`05-orchestrator`](../05-orchestrator/README.md)) autoscales replicas
and LeaderWorkerSet groups on queue and SLO signals, and it has prefill/decode disaggregation. Layer 06
([`agentic-scaling-lab`](../06-gateway/scaling-admission-cost/agentic-scaling-lab/)) has admission control and cost at
the gateway. Admission control and cost at the gateway are "shape demand to capacity" one layer above Kueue's
quotas.

## Caveats

- The core's outputs are **simulated**. The durations, prices, stockout rates and preemption rates are illustrative
  inputs. On kind, the scheduler, Kueue and every event are real. But the GPUs are an extended resource that the lab
  patches into the node status: there is no device plugin, no `/dev/nvidia*` and no CUDA.
- RunPod and Vast.ai rent containers, not nodes. Thus they cannot teach this layer. For the T1/T2 path, a VM is
  necessary.
- The product versions, GKE details and prices are as of September 2026. They have the (verify) tag.
- This layer does not cover these topics yet: NUMA and CPU-manager alignment, Volcano beyond a table row, and
  MultiKueue. This layer also does not cover these topics yet: how to install and upgrade GPU Operator, and how to
  repair nodes that are not healthy. See
  [`CURRICULUM.md` §2](../CURRICULUM.md#2-what-each-layer-has-and-what-it-does-not-cover-yet).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`gpu-scheduling/k8s-gpu-core/`** — [01_how_kubernetes_sees_a_gpu](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb) · [02_filter_score_and_fragmentation](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/02_filter_score_and_fragmentation.ipynb) · [03_gangs_and_topology](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/03_gangs_and_topology.ipynb) · [04_queues_quotas_and_preemption](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/04_queues_quotas_and_preemption.ipynb) · [05_autoscaling_and_obtainability](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/notebooks/05_autoscaling_and_obtainability.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/01_how_kubernetes_sees_a_gpu.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/02_filter_score_and_fragmentation.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/03_gangs_and_topology.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/04_queues_quotas_and_preemption.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core/solutions/05_autoscaling_and_obtainability.ipynb)
- **`gpu-scheduling/k8s-gpu-lab/`** — [01_manifests_and_the_linter](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/01_manifests_and_the_linter.ipynb) · [02_kind_with_fake_gpus_and_kueue](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb) · [03_why_is_my_pod_pending](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/03_why_is_my_pod_pending.ipynb) · [04_gke_pools_dws_and_computeclasses](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/notebooks/04_gke_pools_dws_and_computeclasses.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/01_manifests_and_the_linter.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/02_kind_with_fake_gpus_and_kueue.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/03_why_is_my_pod_pending.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/solutions/04_gke_pools_dws_and_computeclasses.ipynb)
<!-- colab-links:end -->
