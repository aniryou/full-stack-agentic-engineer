# deploy/kind — a GPU scheduling lab on a laptop, with fake GPUs

**What it does.** Creates a 6-node [kind](https://kind.sigs.k8s.io/) cluster (Kubernetes 1.34),
turns four workers into fake 4-GPU L4 nodes with GKE-style labels and taints, and installs the
real Kueue, JobSet and LeaderWorkerSet controllers. The scenarios under `workloads/` then show
quota, gangs, topology-aware placement, priority preemption and cohort reclaim — decided by
the real kube-scheduler and Kueue — while the notebooks and `python -m k8sgpu kind run` compare
each step with the bundled predictor.

**Cost.** $0: everything runs in Docker on your machine. About 3-4 GB of Docker memory.

**Needs.** Docker, kind v0.33.0, kubectl >= 1.27 (for `kubectl patch --subresource=status`),
network access to github.com (release manifests) and Docker Hub (busybox). Python 3.10+ with
`pip install -e ..` (from the lab root) for the scenario runner.

## Run it

```bash
# from the lab root (03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab)
DRY_RUN=1 deploy/kind/up.sh        # read the whole procedure first; nothing is executed
deploy/kind/up.sh                  # cluster + fake GPUs + Kueue/JobSet/LWS + the lab queues
python3 -m k8sgpu kind status      # ready: kind-gpu-lab: 6 nodes, 16 fake GPUs, Kueue installed
python3 -m k8sgpu kind predict s2  # the prediction for scenario s2
python3 -m k8sgpu kind run s2      # apply step by step, observe, compare; exits 1 on a mismatch
python3 -m k8sgpu kind observe s2  # re-read the cluster later and compare again
deploy/kind/down.sh                # delete everything
```

| Script | Does |
|---|---|
| `up.sh` | `kind create cluster` (image pinned in `../versions.env`), preload busybox, then the three below, then `manifests/0*-3*.yaml` |
| `fake-gpus.sh` | from `topology.txt`: node-pool / accelerator / `gce-topology-{block,subblock,host}` labels, the `nvidia.com/gpu=present:NoSchedule` taint, and `nvidia.com/gpu` capacity via a status patch. **Re-run after a Docker restart** (a re-registering kubelet zeroes extended resources it does not manage) |
| `install-addons.sh` | JobSet, LeaderWorkerSet, Kueue release manifests (`kubectl apply --server-side`), waits for the controllers |
| `kwok.sh` | optional: KWOK controller + 32 fake 8-GPU H100 nodes (2 blocks x 4 subblocks x 4 hosts) + the `fleet` queue; `--delete` removes them |
| `down.sh` | `kind delete cluster` |

All scripts print each command (`+ kubectl ...`) and honour `DRY_RUN=1`; `CLUSTER_NAME`
(default `gpu-lab`) and `KUBE_CONTEXT` override the target.

## How a GPU is faked (and what that does not fake)

```text
kubectl patch node gpu-lab-worker2 --subresource=status --type=json \
  -p '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"4"},
       {"op":"add","path":"/status/allocatable/nvidia.com~1gpu","value":"4"}]'
```

`nvidia.com/gpu` is an *extended resource*: to the scheduler it is just an integer per node.
The kubelet keeps capacity it did not set and recomputes allocatable from it; at admission it
asks the device manager only about resources a device plugin registered, so pods requesting
the fake GPUs start normally — with no `/dev/nvidia*` (each lab pod prints that). Real on this
cluster: filtering and scoring, taints and tolerations, Kueue admission, quota, cohorts,
preemption, TAS placement, JobSet/LWS gang lifecycles, events. Not real: devices, drivers,
NVLink, anything CUDA. For device-level behaviour see layer 02; for DRA with simulated devices
see the upstream [dra-example-driver](https://github.com/kubernetes-sigs/dra-example-driver)
(its kind demo uses the same `resource.k8s.io/v1` API the lab's `manifests.resource_claim_template()` emits).

## The cluster and the queues

The block/subblock/host labels below are faked for teaching: real L4 (G2) nodes on GKE are not
known to carry GCE topology labels (Google's TAS examples use them on A3/A4/A4X; verify), so
the lab's GKE flavors are plain quota. The mechanism you practise here is the one those
families use.

```text
gpu-lab-control-plane                     (control-plane taint)
gpu-lab-worker   pool=system              no GPUs, no taint: Kueue/JobSet/LWS controllers run here
block-a
 ├─ subblock-a1: gpu-lab-worker2 (host-a1-1, 4 GPUs)   gpu-lab-worker3 (host-a1-2, 4 GPUs)
 └─ subblock-a2: gpu-lab-worker4 (host-a2-1, 4 GPUs)   gpu-lab-worker5 (host-a2-2, 4 GPUs)
```

`manifests/` (generated from `k8sgpu/scenarios.py`): namespaces `team-a`, `team-b`, `zoo`;
Topology `gke-default` (block > subblock > host > hostname); ResourceFlavor `gpu-l4` (TAS,
injects the GPU toleration); ClusterQueues `team-a-cq` / `team-b-cq` in cohort `gpu-lab`
(8 GPUs nominal each, borrowingLimit 4, `withinClusterQueue: LowerPriority`,
`reclaimWithinCohort: Any`); LocalQueue `gpu-queue` in each team namespace;
WorkloadPriorityClasses `low` (100) and `high` (1000).

## Scenarios by hand

The runner applies one file at a time and waits for Kueue to settle, because admission order
decides who is preempted. By hand, do the same:

```bash
for f in deploy/kind/workloads/s4-priority-preemption/*.yaml; do
  kubectl apply -f "$f"; sleep 15; kubectl get workloads -A; done
kubectl get pods -A -o wide
kubectl get workloads -n team-a -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Preempted")].reason}{"\n"}{end}'
```

Clean up between scenarios with `python3 -m k8sgpu kind reset` (deletes Jobs, JobSets and LWS
in the lab namespaces; the queues stay).

## Troubleshooting

* **Pods Pending with `Insufficient nvidia.com/gpu` everywhere after a restart**: re-run
  `deploy/kind/fake-gpus.sh`.
* **`failed calling webhook ... kueue`** right after install: the webhook lags its Deployment;
  `up.sh` retries, or wait 30 s and re-apply.
* **Image pulls fail / rate-limited**: `docker pull busybox:1.38.0 && kind load docker-image busybox:1.38.0 --name gpu-lab`.
* **A scenario differs from the prediction**: `python3 -m k8sgpu kind observe <s>` after a
  minute (TAS requeues freed capacity in ~10 s batches); if it still differs, the printed
  Workload condition says why — and you have found something the predictor does not model.
