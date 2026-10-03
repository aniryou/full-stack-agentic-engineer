# deploy/kind — a GPU scheduling lab on a laptop, with fake GPUs

**What it does.** The scripts create a 6-node [kind](https://kind.sigs.k8s.io/) cluster (Kubernetes 1.34). They
change four workers into fake 4-GPU L4 nodes with GKE-style labels and taints. They also install the real Kueue,
JobSet and LeaderWorkerSet controllers.

Then the scenarios in `workloads/` show quota, gangs, topology-aware placement, priority preemption and cohort
reclaim. The real kube-scheduler and Kueue make these decisions. At the same time, the notebooks and `python -m k8sgpu kind run` compare each step with the bundled predictor.

**Cost.** $0. Everything runs in Docker on your machine. The cluster uses about 3-4 GB of Docker memory.

**Clean up.** `deploy/kind/down.sh` deletes the cluster. `python3 -m k8sgpu kind reset` removes the workloads
between scenarios and keeps the queues.

**Needs.** Docker, kind v0.33.0, kubectl >= 1.27 (for `kubectl patch --subresource=status`), and network access
to github.com (release manifests) and Docker Hub (busybox). The scenario runner needs Python 3.10+ with
`pip install -e ..` (from the lab root).

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
| `up.sh` | It runs `kind create cluster` (image pinned in `../versions.env`) and preloads busybox. Then it runs the next three scripts in this table. Then it applies `manifests/0*-3*.yaml` |
| `fake-gpus.sh` | It reads `topology.txt` and sets the node-pool, accelerator and `gce-topology-{block,subblock,host}` labels, the `nvidia.com/gpu=present:NoSchedule` taint, and the `nvidia.com/gpu` capacity through a status patch. **Re-run after a Docker restart**. When the kubelet registers again, it sets to zero the extended resources that it does not manage |
| `install-addons.sh` | It applies the JobSet, LeaderWorkerSet and Kueue release manifests (`kubectl apply --server-side`). Then it waits for the controllers |
| `kwok.sh` | Optional. It installs the KWOK controller, 32 fake 8-GPU H100 nodes (2 blocks x 4 subblocks x 4 hosts) and the `fleet` queue. `--delete` removes them |
| `down.sh` | `kind delete cluster` |

All scripts print each command (`+ kubectl ...`) and obey `DRY_RUN=1`. `CLUSTER_NAME` (default `gpu-lab`) and
`KUBE_CONTEXT` set a different target.

## How a GPU is faked (and what that does not fake)

```text
kubectl patch node gpu-lab-worker2 --subresource=status --type=json \
  -p '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"4"},
       {"op":"add","path":"/status/allocatable/nvidia.com~1gpu","value":"4"}]'
```

`nvidia.com/gpu` is an *extended resource*. For the scheduler, it is only an integer per node. The kubelet keeps
capacity that it did not set, and it calculates allocatable again from that capacity. At admission, the kubelet
asks the device manager only about resources that a device plugin registered. Thus pods that request the fake
GPUs start normally, but with no `/dev/nvidia*` (each lab pod prints that).

These parts are real on this cluster: filtering and scoring, taints and tolerations, Kueue admission, quota,
cohorts, preemption, TAS placement, JobSet/LWS gang lifecycles, events. These parts are not real: devices,
drivers, NVLink, anything CUDA. For device-level behaviour, see layer 02. For DRA with simulated devices, see the
upstream [dra-example-driver](https://github.com/kubernetes-sigs/dra-example-driver). Its kind demo uses the same
`resource.k8s.io/v1` API that the lab's `manifests.resource_claim_template()` emits.

## The cluster and the queues

The lab fakes the block/subblock/host labels in the diagram that follows, to teach. No known source shows
that real L4 (G2) nodes on GKE have GCE topology labels. Google's TAS examples use them on A3/A4/A4X (verify).
Thus the lab's GKE flavors are plain quota. The mechanism that you use here is the one that those families use.

```text
gpu-lab-control-plane                     (control-plane taint)
gpu-lab-worker   pool=system              no GPUs, no taint: Kueue/JobSet/LWS controllers run here
block-a
 ├─ subblock-a1: gpu-lab-worker2 (host-a1-1, 4 GPUs)   gpu-lab-worker3 (host-a1-2, 4 GPUs)
 └─ subblock-a2: gpu-lab-worker4 (host-a2-1, 4 GPUs)   gpu-lab-worker5 (host-a2-2, 4 GPUs)
```

`manifests/` (generated from `k8sgpu/scenarios.py`) contains these objects:

- Namespaces `team-a`, `team-b`, `zoo`.
- Topology `gke-default` (block > subblock > host > hostname).
- ResourceFlavor `gpu-l4` (TAS). It injects the GPU toleration.
- ClusterQueues `team-a-cq` / `team-b-cq` in cohort `gpu-lab` (8 GPUs nominal each, borrowingLimit 4,
  `withinClusterQueue: LowerPriority`, `reclaimWithinCohort: Any`).
- LocalQueue `gpu-queue` in each team namespace.
- WorkloadPriorityClasses `low` (100) and `high` (1000).

## Scenarios by hand

The runner applies one file at a time. After each file, it waits until Kueue is stable, because the admission
order decides which workloads Kueue preempts. When you work by hand, do the same steps:

```bash
for f in deploy/kind/workloads/s4-priority-preemption/*.yaml; do
  kubectl apply -f "$f"; sleep 15; kubectl get workloads -A; done
kubectl get pods -A -o wide
kubectl get workloads -n team-a -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Preempted")].reason}{"\n"}{end}'
```

Between scenarios, clean up with `python3 -m k8sgpu kind reset`. It deletes the Jobs, JobSets and LWS in the lab
namespaces. The queues stay.

## Troubleshooting

* **Pods Pending with `Insufficient nvidia.com/gpu` everywhere after a restart**: run
  `deploy/kind/fake-gpus.sh` again.
* **`failed calling webhook ... kueue`** immediately after the install: the webhook becomes ready later than its
  Deployment. `up.sh` retries. You can also wait 30 s and apply again.
* **Image pulls fail / rate-limited**: run `docker pull busybox:1.38.0 && kind load docker-image busybox:1.38.0 --name gpu-lab`.
* **A scenario differs from the prediction**: wait a minute, then run `python3 -m k8sgpu kind observe <s>`.
  TAS requeues freed capacity in ~10 s batches. If the result still differs, the printed Workload condition
  tells why. Then you found something that the predictor does not model.
