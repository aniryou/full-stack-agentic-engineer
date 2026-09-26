# gpu-scheduling — Kubernetes for GPUs

After this topic you can explain how a GPU becomes schedulable, and place, share, queue and scale GPU work on
Kubernetes: the device plugin and DRA, the scheduler's filter/score/preempt cycle and GPU fragmentation, gangs and
topology-aware placement, Kueue quotas with borrowing and reclaim, and getting GPU capacity — autoscaling from zero,
Spot, DWS flex-start, startup latency, sharing.

## Start here

1. Read [PRIMER.md](PRIMER.md) "The one-minute version", then §1 — what Kubernetes sees.
2. `cd k8s-gpu-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 49 tests, well under a
   second; then open [`01_how_kubernetes_sees_a_gpu`](k8s-gpu-core/notebooks/01_how_kubernetes_sees_a_gpu.ipynb).
3. With Docker on your laptop, run the real scheduler and Kueue against fake GPUs:
   [`k8s-gpu-lab/deploy/kind`](k8s-gpu-lab/deploy/kind/README.md) and the lab's
   [`02_kind_with_fake_gpus_and_kueue`](k8s-gpu-lab/notebooks/02_kind_with_fake_gpus_and_kueue.ipynb).

The fastest win, with nothing installed — sixteen 1-GPU pods spread over four 8-GPU nodes leave no room for one
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

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* "T0 + Docker" is a laptop with Docker, still free.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the concepts shared by core and lab: ten numbered sections, a design-review walkthrough with drill questions, glossary, sources and a dated verify list | read alongside the core | — |
| [`k8s-gpu-core/`](k8s-gpu-core/) | predict placement, fragmentation, preemption and autoscaling with the **minimal** implementation: `gpusched`, a pure-Python simulator of the device plugin, scheduler, gangs + Kueue TAS, Kueue quotas and a GPU cluster autoscaler; five fill-in notebooks | ~9 h with the primer | T0 |
| [`k8s-gpu-lab/`](k8s-gpu-lab/) | write, lint and debug real GPU manifests with the **detailed** lab: `k8sgpu` — typed manifest builders (Job, JobSet, LWS, Kueue, DRA, ComputeClass), a GPU pod-spec linter, a "why is my pod Pending?" analyser, a capacity-type chooser; [`deploy/kind`](k8s-gpu-lab/deploy/kind/) (fake GPUs, Kueue, JobSet, LWS), [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/) (k3s + the real device plugin on one GPU VM), [`deploy/gcp`](k8s-gpu-lab/deploy/gcp/) (Terraform) and [`deploy/gke`](k8s-gpu-lab/deploy/gke/) (manifests) | ~5 h at T0 (+2 h on GKE) | T0 → T3 |

Times are rough: about 9 hours for the primer and the core, 5 more for the lab's T0 path (the repo's curriculum,
`CURRICULUM.md` at the repo root, modules 03.1–03.6).

### Work it in this order

Read the primer section, then do the core notebook, then (optionally) the lab notebook.

| Step | Primer | Core notebook (T0) | Lab notebook |
|---|---|---|---|
| 1 | §1 What Kubernetes sees · §2 GPU Operator vs managed drivers | `01_how_kubernetes_sees_a_gpu` | `01_manifests_and_the_linter` (T0) |
| 2 | §3 The scheduling cycle | `02_filter_score_and_fragmentation` | `03_why_is_my_pod_pending` (T0) |
| 3 | §4 Gangs · §5 Topology-aware placement | `03_gangs_and_topology` | `02_kind_with_fake_gpus_and_kueue` (T0 + Docker; simulator fallback) |
| 4 | §6 Queues, quotas and multi-tenancy with Kueue | `04_queues_quotas_and_preemption` | `02_kind_with_fake_gpus_and_kueue` |
| 5 | §7 Getting capacity · §8 Startup latency | `05_autoscaling_and_obtainability` | `04_gke_pools_dws_and_computeclasses` (T3; offline it plans and inspects) |
| 6 | §9 Sharing GPUs at the cluster level · §10 Learning locally | `01_how_kubernetes_sees_a_gpu`, exercise 1.6 (time-slicing replicas) | §9: [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/) (time-slicing with the real device plugin, T1) and [`deploy/gke/50-time-sharing-l4.yaml`](k8s-gpu-lab/deploy/gke/50-time-sharing-l4.yaml) (GKE time-sharing, T3); §10: [`deploy/kind`](k8s-gpu-lab/deploy/kind/) (T0 + Docker) |

Finish with the primer's "In a design review": a walkthrough and drill questions.

## Run it

```bash
cd k8s-gpu-core
python3 -m pip install -r requirements.txt     # only to run the notebooks and tests; the library is stdlib-only
python3 -m pytest -q                           # 49 tests, well under a second
python3 -m jupyterlab notebooks                # do the exercises; finished versions are in solutions/

cd ../k8s-gpu-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                           # 118 tests, ~5 s, offline
python3 -m jupyterlab notebooks
```

On Colab, every notebook's first cell clones the repo and installs its lab; the links are in the
[layer README](../README.md#run-in-colab).

| Tier | Where | What you get here | Cost |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the whole core; the lab's manifests, linter and Pending analyser; the lab's kind notebook falls back to a simulator without Docker | $0 |
| **T0 + Docker** | laptop with Docker | the lab's `deploy/kind`: the real kube-scheduler, Kueue v0.19.6, JobSet and LWS against fake `nvidia.com/gpu` capacity (optionally hundreds of KWOK nodes) | $0 |
| T1 / T2 | any GPU VM you control (Lambda, a GCP VM) with k3s or kubeadm and the GPU Operator | the real device plugin, GPU Feature Discovery labels, MIG or time-slicing on one box (the lab's [`deploy/gpu-vm`](k8s-gpu-lab/deploy/gpu-vm/): k3s + the device plugin, optional time-slicing) | the VM's hourly price (`COMPUTE.md` at the repo root) |
| **T3** | GKE via the lab's Terraform | a zonal cluster, a Spot L4 pool from zero with driver auto-install, flex-start queued provisioning, image streaming, GCS FUSE; ComputeClass and Kueue ProvisioningRequest manifests | pay per use; L4 Spot, scale to zero, destroy after |

RunPod and Vast give you containers, not nodes, so they cannot teach this layer; use them for layers 01, 02 and 04.
Prices and GPU obtainability: `COMPUTE.md` at the repo root. Where this sits in the whole course: `CURRICULUM.md` at
the repo root.

## How it fits

This topic sits between the GPU software substrate (layer 02: driver, CUDA, container runtime) and the inference
engine (layer 04).

| | Read | For |
|---|---|---|
| before | layer 01 — [`roofline-and-fabric/PRIMER.md`](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §5, §6, §7, §10 | fabric bandwidth (why topology matters), cold-start arithmetic, checkpoint intervals, GPU families and obtainability |
| before | layer 01 — [`gpu-deployment-primer.md`](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md) §4 and §7 | the parallelism menu; Kubernetes specifics in brief |
| beside | layer 02 — `02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md` §6, §7, §8 | how a container gets a GPU; MIG, time-slicing, MPS mechanics; health and DCGM |
| after | layer 05 — [`serving-orchestration/PRIMER.md`](../../05-orchestrator/serving-orchestration/PRIMER.md) §4, §5 | autoscaling replicas and LeaderWorkerSet groups on queue and SLO signals; cold-start anatomy; prefill/decode disaggregation |
| after | layer 06 — [`agentic-scaling-lab`](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/) | admission control and cost at the gateway, the same "shape demand to capacity" idea one layer up |

## Caveats

- **Simulated vs real.** The core's outputs are simulated from illustrative inputs. On kind the scheduler, Kueue,
  taints, labels and events are real, but the GPUs are a patched extended resource — no device plugin, no
  `/dev/nvidia*`, no CUDA; `deploy/gpu-vm` adds the real device path.
- **Dated facts.** Upstream versions, GKE behaviour and prices are as of September 2026 and marked (verify); the
  primer's Verify list collects them.
