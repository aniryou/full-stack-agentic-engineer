# gpu-scheduling — Kubernetes for GPUs

After this topic, you can explain how a GPU becomes schedulable. You can also place GPU work on Kubernetes, share
it, put it in queues and scale it. The topic covers these things:

- The device plugin and DRA.
- The scheduler's filter/score/preempt cycle and GPU fragmentation.
- Gangs and topology-aware placement.
- Kueue quotas with borrowing and reclaim.
- How to get GPU capacity: autoscaling from zero, Spot, DWS flex-start, startup latency and how to share GPUs.

## Start here

1. Read [PRIMER.md](PRIMER.md) "The one-minute version". Then read §1, what Kubernetes sees.
2. Run `cd k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. It runs 59 tests in
   about 25 seconds. Then open [`01_how_kubernetes_sees_a_gpu`](k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb).
3. If you have Docker on your laptop, run the real scheduler and Kueue against fake GPUs. Use
   [`k8s-gpu-lab/deploy/kind`](k8s-gpu-lab/deploy/kind/README.md) and the lab's
   [`02_kind_with_fake_gpus_and_kueue`](k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb).

The fastest win needs nothing installed. Sixteen 1-GPU pods, spread over four 8-GPU nodes, leave no room for one
8-GPU pod:

```python
from gpusched import Scheduler, gpu_pod, make_cluster, stranded_gpus    # run from k8s-gpu-core/
cluster = make_cluster(hosts=4)
sched = Scheduler(cluster)
sched.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(16)])
sched.run()
print(sched.schedule_one(gpu_pod("train", 8)).message)   # 0/4 nodes are available: 4 Insufficient nvidia.com/gpu. ...
print(stranded_gpus(cluster, 8))                          # 16
```

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab or Kaggle T4, or a rented card). T2
is a multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* "T0 + Docker"
is a laptop with Docker, also at no cost.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the concepts that the core and the lab share. The primer has ten numbered sections, a walkthrough for a design review with drill questions, a glossary, sources and a dated verify list. | read it with the core | — |
| [`k8s-gpu-core/`](k8s-gpu-core/) | predict placement, fragmentation, preemption and autoscaling with the **minimal** implementation. This is `gpusched`, a pure-Python simulator of these parts: the device plugin, the scheduler, gangs with Kueue TAS, Kueue quotas and a GPU cluster autoscaler. It has five fill-in notebooks. | ~9 h with the primer | T0 |
| [`k8s-gpu-lab/`](k8s-gpu-lab/) | write, lint and debug real GPU manifests with the **detailed** lab, `k8sgpu`. It has typed manifest builders (Job, JobSet, LWS, Kueue, DRA, ComputeClass) and a GPU pod-spec linter. It also has a "why is my pod Pending?" analyser and a capacity-type chooser. It has four deploy folders. They are [`deploy/kind`](k8s-gpu-lab/deploy/kind/) (fake GPUs, Kueue, JobSet, LWS), [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/) (k3s and the real device plugin on one GPU VM), [`deploy/gcp`](k8s-gpu-lab/deploy/gcp/) (Terraform) and [`deploy/gke`](k8s-gpu-lab/deploy/gke/) (manifests). | ~5 h at T0 (+2 h on GKE) | T0 to T3 |

The times are approximate. The primer and the core take about 9 hours. The T0 path of the lab takes 5 more hours.
The repo's curriculum ([`CURRICULUM.md`](../../CURRICULUM.md)) has this topic as modules 03.1–03.6.

### Work it in this order

Read the primer section. Then do the core notebook. Then, as an option, do the lab notebook.

| Step | Primer | Core notebook (T0) | Lab notebook |
|---|---|---|---|
| 1 | §1 What Kubernetes sees · §2 GPU Operator vs managed drivers | `01_how_kubernetes_sees_a_gpu` | `01_manifests_and_the_linter` (T0) |
| 2 | §3 The scheduling cycle | `02_filter_score_and_fragmentation` | `03_why_is_my_pod_pending` (T0) |
| 3 | §4 Gangs · §5 Topology-aware placement | `03_gangs_and_topology` | `02_kind_with_fake_gpus_and_kueue` (T0 + Docker, with a simulator fallback) |
| 4 | §6 Queues, quotas and multi-tenancy with Kueue | `04_queues_quotas_and_preemption` | `02_kind_with_fake_gpus_and_kueue` |
| 5 | §7 Getting capacity · §8 Startup latency | `05_autoscaling_and_obtainability` | `04_gke_pools_dws_and_computeclasses` (T3, and offline it plans and inspects) |
| 6 | §9 Sharing GPUs at the cluster level · §10 Learning locally | `01_how_kubernetes_sees_a_gpu`, exercise 1.6 (time-slicing replicas) | §9: [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/) (time-slicing with the real device plugin, T1) and [`deploy/gke/50-time-sharing-l4.yaml`](k8s-gpu-lab/deploy/gke/50-time-sharing-l4.yaml) (GKE time-sharing, T3). §10: [`deploy/kind`](k8s-gpu-lab/deploy/kind/) (T0 + Docker) |

Finish with the primer's "In a design review" section. It has a walkthrough and drill questions.

## Run it

```bash
cd k8s-gpu-core
python3 -m pip install -r requirements.txt     # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                           # 59 tests, ~25 s
python3 -m jupyterlab notebooks                # do the exercises; finished versions are in solutions/

cd ../k8s-gpu-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                           # 128 tests, ~30 s, offline
python3 -m jupyterlab notebooks
```

On Colab, the first cell of each notebook clones the repo and installs its lab. The links are in the
[layer README](../README.md#run-in-colab).

| Tier | Where | What you get here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the whole core, and the lab's manifests, linter and Pending analyser. Without Docker, the lab's kind notebook uses a simulator instead. | $0 |
| **T0 + Docker** | laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue v0.19.6, JobSet and LWS against fake `nvidia.com/gpu` capacity (as an option, hundreds of KWOK nodes) | $0 |
| T1 / T2 | any GPU VM that you control (Lambda, a GCP VM), with k3s or kubeadm and the GPU Operator | the real device plugin, GPU Feature Discovery labels, and MIG or time-slicing on one box. The lab's [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/) has k3s and the device plugin, with optional time-slicing. | the VM's hourly price ([`COMPUTE.md`](../../COMPUTE.md)) |
| **T3** | GKE, through the lab's Terraform | a zonal cluster, a Spot L4 pool from zero with driver auto-install, flex-start queued provisioning, image streaming and GCS FUSE. The T3 path also has ComputeClass and Kueue ProvisioningRequest manifests. | pay per use. Use L4 Spot and scale to zero. Destroy the cluster after use. |

RunPod and Vast give you containers, not nodes. Thus they cannot teach this layer. Use them for layers 01, 02 and 04.
For prices and GPU obtainability, see [`COMPUTE.md`](../../COMPUTE.md). For the position of this topic in the whole
course, see [`CURRICULUM.md`](../../CURRICULUM.md).

## How it fits

This topic is between the GPU software substrate (layer 02: driver, CUDA, container runtime) and the inference
engine (layer 04).

| | Read | For |
|---|---|---|
| before | layer 01: [`roofline-and-fabric/PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §5, §6, §7, §10 | fabric bandwidth (why topology is important), cold-start arithmetic, checkpoint intervals, GPU families and obtainability |
| before | layer 01: [`gpu-deployment-primer.md`](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §4 and §7 | the parallelism menu, and Kubernetes specifics in brief |
| beside | layer 02: [`02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md`](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §6, §7, §8 | how a container gets a GPU, the mechanics of MIG, time-slicing and MPS, and health and DCGM |
| after | layer 05: [`serving-orchestration/PRIMER.md`](../../05-orchestrator/serving-orchestration/PRIMER.md) §4, §5 | the autoscaling of replicas and LeaderWorkerSet groups on queue and SLO signals, the anatomy of a cold start, and prefill/decode disaggregation |
| after | layer 06: [`agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | admission control and cost at the gateway. This is the same "shape demand to capacity" idea, one layer above. |

## Caveats

- **Simulated against real.** The core simulates its outputs from illustrative inputs. On kind, the scheduler, Kueue,
  taints, labels and events are real. But the GPUs are a patched extended resource: there is no device plugin, no
  `/dev/nvidia*` and no CUDA. `deploy/gpu-vm` adds the real device path.
- **Dated facts.** The upstream versions, the GKE behaviour and the prices are as of September 2026. They have the
  (verify) tag. The primer's Verify list collects them.
