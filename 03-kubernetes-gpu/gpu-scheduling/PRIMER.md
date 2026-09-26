# Kubernetes for GPUs: how a GPU becomes schedulable, and how to place, share, queue and scale it

*Layer 03 of the stack. Facts checked 26 September 2026 against upstream sources — Kubernetes
(kube-scheduler, kubelet, API validation), Kueue v0.19 (`kueue.x-k8s.io/v1beta2`), JobSet, LeaderWorkerSet,
the NVIDIA device plugin and DRA driver, cluster-autoscaler. GKE specifics that could not be checked
against Google's documentation are marked (verify). Every worked number is computed by the core simulator,
[`k8s-gpu-core`](k8s-gpu-core) (package `gpusched`); the function is named next to the number. Durations,
prices and failure rates are inputs, and their outputs are labelled simulated.*

This primer covers the layer between the GPU software substrate (driver, CUDA, container runtime —
layer 02) and the inference engine (layer 04): how a node's GPUs become a number Kubernetes can schedule,
how the scheduler places pods on those numbers and why that fragments GPUs, why multi-pod jobs must be
placed all-or-nothing and close together, how Kueue shares a fleet between teams with quotas that borrow
and reclaim, and how GPU capacity is obtained, started and shared — for an engineer who knows Kubernetes
basics (pods, nodes, labels) and wants to explain a GPU platform's design in a review. The short version
is §7 of the [GPU deployment primer](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md).
The detailed lab, [`k8s-gpu-lab`](k8s-gpu-lab), takes the same ideas to manifests, kind with fake GPUs, and GKE.

---

## The one-minute version

A GPU is an **integer** to Kubernetes. A device plugin tells the kubelet how many devices a node has and
which are healthy; the node advertises `nvidia.com/gpu: 8`; pods request whole GPUs with requests equal to
limits; nothing is ever overcommitted. Labels say *which* GPU; taints keep other pods off GPU nodes.

The scheduler places **one pod at a time**: filter, score, bind. Its default score spreads pods using CPU
and memory and ignores GPUs, so small GPU pods **fragment** nodes until a large pod fits nowhere although
half the GPUs are free. Bin-pack GPU pools instead.

Distributed jobs are **gangs**: useless unless every pod runs, and slow unless the pods are **close**
(same NVLink domain, sub-block, block). Placing their pods one by one can deadlock the cluster. Admit
gangs whole (Kueue), place them topology-aware (Kueue TAS), run them as JobSets or LeaderWorkerSets.

**Kueue** decides *whether* a job may start: ClusterQueues own GPU quota per flavor, cohorts lend idle
quota, and owners **reclaim** it by preemption. **Capacity** is the hard part: GPU nodes take minutes to
become useful, may not be available at all, may be reclaimed (Spot), and a gang needs all its nodes at
once — which is what queued, all-or-nothing provisioning (DWS flex-start) is for.

---

## 1. What Kubernetes sees

### 1.1 A GPU is an extended resource

A Node's status carries two resource lists. **Capacity** is what the node has; **allocatable** is what
the scheduler may hand out. For `cpu` and `memory`, allocatable is capacity minus system reservations.
For a GPU it is the number of **healthy** devices:

```
$ kubectl describe node gpu-node-0          # the parts that mention GPUs
Labels:       cloud.google.com/gke-accelerator=nvidia-h100-80gb
Taints:       nvidia.com/gpu=present:NoSchedule
Capacity:     nvidia.com/gpu: 8
Allocatable:  nvidia.com/gpu: 7             # one device reported Unhealthy
```

`nvidia.com/gpu` is an **extended resource**: a name with a domain prefix outside `kubernetes.io`. The
API server enforces three rules on a container's extended resources, and the scheduler never breaks the
fourth (`gpusched.cluster.effective_requests()` reproduces them with the upstream messages):

| Rule | Rejected spec | Message |
|---|---|---|
| Whole numbers only | `limits: {nvidia.com/gpu: 0.5}` | `must be an integer` |
| Request must equal limit | `requests: 1`, `limits: 2` | `must be equal to nvidia.com/gpu limit of 2` |
| A limit is required | `requests: 1`, no limit | `Limit must be set for non overcommitable resources` |
| No overcommit | the sum of requests on a node never exceeds allocatable | (pod stays Pending) |

A limit without a request sets the request to the limit — so writing only `limits: {nvidia.com/gpu: 1}` is
the normal form. There are no fractional GPUs at this layer: sharing (section 9) works by advertising
*more units*, not smaller ones.

### 1.2 The device plugin API

The kubelet knows nothing about GPUs either. A **device plugin** — for NVIDIA GPUs, the NVIDIA device
plugin (installed by the GPU Operator), or on GKE Google's own GPU device plugin — runs on every GPU node
as a DaemonSet and talks gRPC (`deviceplugin/v1beta1`) over Unix sockets:

```
device plugin                                   kubelet (device manager)
     │  Register(version=v1beta1, endpoint,              │
     │           resource_name=nvidia.com/gpu) ─────────►│  /var/lib/kubelet/device-plugins/kubelet.sock
     │◄──────────────────────────── ListAndWatch() ───── │
     │  stream: [{ID, health: Healthy|Unhealthy}] ─────► │  capacity = all IDs, allocatable = healthy IDs
     │                                                   │  → Node.status  → scheduler sees an integer
     │   (pod bound here, container starting)            │
     │◄──── GetPreferredAllocation(available, size) ──── │  topology hint: keep GPUs on one NVLink island
     │◄──── Allocate([device IDs]) ───────────────────── │
     │  env / mounts / device nodes / CDI devices ─────► │  container created with those devices
```

`gpusched.deviceplugin` models the contract: `DevicePlugin.list_and_watch()`, `Kubelet.node_status()`
(capacity counts every device, allocatable only healthy ones — as the kubelet's `GetCapacity` does), and
`Kubelet.admit()`. With the NVIDIA plugin's default `envvar` strategy, `Allocate` answers with
`NVIDIA_VISIBLE_DEVICES=<device UUIDs>`, and the NVIDIA Container Toolkit injects device nodes and driver
libraries when the container is created — the container half is layer 02
([cuda-and-nccl primer](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §6 *How a container gets a GPU*).

Two consequences are worth saying in a review:

* **A failing GPU does not evict anything.** When a device turns Unhealthy (an XID error, a GPU fallen off
  the bus), allocatable drops from 8 to 7 (`Kubelet.node_status()`), running pods keep their devices, and
  only new placements see the change. Draining the node is an operational decision.
* **The scheduler and the kubelet keep separate books.** The scheduler binds a pod using the last node
  status it saw; the kubelet admits it against the devices it actually has. If a device failed in between,
  the pod fails with `UnexpectedAdmissionError` — `Allocate failed due to requested number of devices
  unavailable for nvidia.com/gpu. Requested: 2, Available: 1, which is unexpected` (`Kubelet.admit()`; the
  wording is the kubelet's) — and its controller must recreate it.

### 1.3 Labels: which GPU, and where

The resource name says nothing about the model. Placement by GPU type uses **node labels**:

| Source | Example labels |
|---|---|
| GKE | `cloud.google.com/gke-accelerator=nvidia-l4` (also what cluster-autoscaler keys GPU nodes on), `cloud.google.com/gke-nodepool`, Spot and MIG labels (verify names in GKE docs) |
| NVIDIA GPU Feature Discovery (in the GPU Operator) | `nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3`, `nvidia.com/gpu.memory` (MiB), `nvidia.com/gpu.count`, `nvidia.com/gpu.family`, `nvidia.com/cuda.driver-version.full`, `nvidia.com/mig.strategy`, `nvidia.com/gpu.replicas`, `nvidia.com/gpu.clique` (multi-node NVLink domain) |
| Node Feature Discovery | PCI vendor labels such as `feature.node.kubernetes.io/pci-10de.present=true` (10de is NVIDIA's PCI vendor ID; exact form depends on NFD config — verify) |
| GCE topology | `cloud.google.com/gce-topology-block`, `-subblock`, `-host` (verify) — section 5 |

A pod selects a type with a `nodeSelector` (or node affinity); section 3 shows the scheduler's view.

### 1.4 Taints: keeping the wrong pods off

GPU nodes are **tainted** — GKE uses `nvidia.com/gpu=present:NoSchedule` (verify) — so ordinary pods do
not land on the most expensive nodes in the cluster. The matching toleration comes from the
**ExtendedResourceToleration** admission plugin: it adds `{key: nvidia.com/gpu, operator: Exists, effect:
NoSchedule}` to every pod that *requests* `nvidia.com/gpu` (`gpusched.cluster.extended_resource_toleration()`,
which `gpu_pod()` applies). The plugin is **off by default** in kube-apiserver. GKE enables it (verify);
kubeadm, kind and most self-managed clusters need `--enable-admission-plugins=...,ExtendedResourceToleration`,
or every GPU pod must carry the toleration itself, else it stays Pending on `untolerated taint(s)`.
The upstream matching rule is short enough to learn exactly (`Toleration.tolerates()`): if the toleration
names an effect it must match; if it names a key it must match (an empty key with `Exists` tolerates
everything); then `Exists` matches any value and `Equal` needs the same value. `NoSchedule` and `NoExecute`
taints filter nodes; `PreferNoSchedule` only lowers the score.

### 1.5 Dynamic Resource Allocation (DRA)

A counter cannot say "an H100 with at least 40 GiB free, on the same PCIe root as this NIC". **DRA**,
GA in Kubernetes 1.34 (`resource.k8s.io/v1`), replaces it with structured objects:

| Object | Written by | Holds |
|---|---|---|
| `ResourceSlice` | the DRA driver, per node or pool | devices with **attributes** (model, index, UUID, MIG profile…) and **capacity** (memory…) |
| `DeviceClass` | the cluster admin | a named selection, e.g. `gpu.nvidia.com` = `device.driver == 'gpu.nvidia.com' && device.attributes['gpu.nvidia.com'].type == 'gpu'` |
| `ResourceClaimTemplate` / `ResourceClaim` | the workload | `devices.requests[]`: `exactly` {`deviceClassName`, CEL `selectors`, `allocationMode`, `count`}, or `firstAvailable` alternatives; `constraints[].matchAttribute` (e.g. all devices on one PCIe root) |

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata: {name: one-h100}
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly:
          deviceClassName: gpu.nvidia.com
          selectors:
          - cel: {expression: "device.attributes['gpu.nvidia.com'].productName.lowerAscii().matches('^.*h100.*$')"}
```

The pod lists the claim in `spec.resourceClaims` and each container references it in
`resources.claims`. The scheduler's `DynamicResources` plugin allocates concrete devices while it
schedules, so it can reason about attributes and sharing rather than counts. The NVIDIA DRA driver
(`kubernetes-sigs/dra-driver-nvidia-gpu`) supports **ComputeDomains** — multi-node NVLink (IMEX) domains
for GB200/GB300-class racks — officially, while its GPU-allocation plugin was still marked not officially
supported and off by default in September 2026 (verify before relying on it). Device plugins and DRA
coexist; most GPU fleets still schedule `nvidia.com/gpu` counts today; Kueue's DRA support nears beta.

---

## 2. GPU Operator vs managed drivers

Before any of section 1 works, a GPU node needs a kernel driver, the NVIDIA Container Toolkit wired into
the container runtime, the device plugin, and ideally feature discovery and metrics. Two ways to get them:

| | NVIDIA GPU Operator | Managed by the platform (GKE) |
|---|---|---|
| What installs the driver | a driver container per node (or a pre-installed host driver) | GKE's driver installer; automatic on control planes ≥ 1.32.2-gke.1297000, version `DEFAULT`, `LATEST` (COS only) or `INSTALLATION_DISABLED` per node pool |
| Device plugin | NVIDIA device plugin | Google's GPU device plugin (open source: `GoogleCloudPlatform/container-engine-accelerators`) |
| Also brings | NFD, GPU Feature Discovery labels, DCGM + dcgm-exporter, MIG manager, node status exporter, validator; configured by the `ClusterPolicy` CRD (plus the `NVIDIADriver` CRD for per-pool drivers) | time-sharing, MPS and MIG as node-pool settings; GPU metrics in Cloud Monitoring (layer 02 §8–9) |
| You own | versions and upgrades of every component; compatibility with the node OS and kernel | picking the driver channel; less choice, fewer moving parts |
| Typical home | on-prem, self-managed clusters, other clouds, GKE with installation disabled | GKE Standard and Autopilot |

The choice matters beyond convenience. The driver version bounds which CUDA userlands in your images
work (layer 02 §1). A driver upgrade is a node drain. And a GPU node is `Ready` before its GPUs are
allocatable — the driver and plugin start after the kubelet — which is why the cluster autoscaler treats
GPU nodes without allocatable GPUs as `resourceUnready` rather than usable (section 7), and why a node
that stays in that state usually means a driver install failure.

---

## 3. The scheduling cycle

### 3.1 One pod at a time

```
 queue ─► PreEnqueue ─► QueueSort ─► PreFilter ─► Filter ─► PostFilter ─► PreScore ─► Score ─► Reserve ─► Permit ─► PreBind ─► Bind
          (scheduling    (priority,               (per node:  (no node?    (0-100 per plugin,              (can hold a pod:
           gates)         then age)                pass/fail)  preemption)  weighted sum)                   gangs, §4)
```

kube-scheduler takes the highest-priority pending pod (then the oldest), runs the **filter** plugins
against every node, **scores** the survivors, and binds the pod to the best one. Then the next pod. It
never considers two pods together, never looks ahead, and never moves a running pod — three facts that
explain fragmentation (3.4), preemption's limits (3.5) and gang deadlock (4.1). The default plugin set
and score weights (upstream `getDefaultPlugins`):

| Plugin | Filter | Score weight | Relevance to GPUs |
|---|---|---|---|
| NodeUnschedulable | cordoned nodes | – | drains |
| TaintToleration | untolerated NoSchedule/NoExecute taints | 3 | keeps CPU pods off GPU nodes |
| NodeAffinity | nodeSelector / required affinity | 2 | GPU model selection |
| NodeResourcesFit | requests ≤ allocatable − requested | 1 | **the only place GPUs are counted** |
| PodTopologySpread, InterPodAffinity | spread / affinity rules | 2, 2 | replica spreading |
| DynamicResources | DRA claims allocatable | 2 | section 1.5 |
| NodeResourcesBalancedAllocation, ImageLocality | – | 1, 1 | pull toward spreading and cached images |
| DefaultPreemption | PostFilter | – | section 3.5 |

### 3.2 Filter

Each node passes, or fails with the reason of the **first** plugin that rejects it — in the order
unschedulable, taints, node affinity, resources (`gpusched.plugins.run_filters()`). When every node fails,
the pod's `FailedScheduling` event is a histogram of those reasons, sorted as strings
(`gpusched.scheduler.fit_error()`, same format as upstream `FitError`). For an 8-GPU pod that selects
H100s, in a cluster of one H100 node with 4 GPUs busy, one cordoned H100 node, one L4 node and one CPU
node (notebook 02):

```
0/4 nodes are available: 1 Insufficient nvidia.com/gpu, 1 node(s) were unschedulable,
2 node(s) didn't match Pod's node affinity/selector.
```

Read it per reason: *Insufficient* is capacity (wait, preempt, scale up, defragment); *affinity/selector*
and *untolerated taint* mean the pod asked for a node shape that is not here. The CPU node reports the
selector, not the missing GPU, because the selector filter runs first. Lab notebook
`03_why_is_my_pod_pending` turns this into a diagnosis tool.

### 3.3 Score: spread or pack

`NodeResourcesFit` scores with one of two strategies over a configured list of `(resource, weight)`,
counting the incoming pod as already placed and using integer arithmetic (MaxNodeScore = 100):

```
LeastAllocated  = Σ w_r · ((alloc_r − requested_r) · 100 // alloc_r)   //  Σ w_r      (spread; the default)
MostAllocated   = Σ w_r · (min(requested_r, alloc_r) · 100 // alloc_r)  //  Σ w_r      (bin-pack)
```

On an 8-GPU node with 6 GPUs requested, a 1-GPU pod scored on the GPU alone gets `MostAllocated` =
7 · 100 // 8 = **87** and `LeastAllocated` = 1 · 100 // 8 = **12** (`plugins.most_allocated()`,
`plugins.least_allocated()`). Two details decide GPU behaviour:

* **The default scoring resources are `cpu` and `memory` (weight 1 each).** GPUs are not scored at all:
  two nodes with the same CPU and memory use score the same whether they have 1 or 7 GPUs busy. GPU
  spreading happens as a side effect of CPU/memory spreading.
* **An extended resource the pod does not request is skipped**, not scored as zero. Adding
  `nvidia.com/gpu` to the list therefore does not repel CPU pods from GPU nodes — taints do that.

To pack GPU pools, configure a scheduler profile with `MostAllocated` and a GPU weight:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: gpu-binpack
  pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: MostAllocated
        resources: [{name: cpu, weight: 1}, {name: memory, weight: 1}, {name: nvidia.com/gpu, weight: 5}]
```

On a managed control plane you cannot edit the default scheduler; GKE's `optimize-utilization`
autoscaling profile switches scheduling toward bin-packing (verify), or you run a second scheduler with
this profile and select it with `schedulerName`. Packing has costs you should name: replicas of one
service concentrate on few nodes (keep them apart with `topologySpreadConstraints`), and other default
scorers (BalancedAllocation, default PodTopologySpread for Deployments) still pull toward spreading.

### 3.4 Fragmentation, measured

Sixteen 1-GPU inference pods (8 cores, 64 GiB each) arrive at four empty 8-GPU nodes, then one 8-GPU
training pod (`scheduler.Scheduler`, notebook 02):

```
default (LeastAllocated, cpu+memory)          MostAllocated, GPU weight 5
h0  ####....  4/8                             h0  ########  8/8
h1  ####....  4/8                             h1  ########  8/8
h2  ####....  4/8                             h2  ........  0/8
h3  ####....  4/8                             h3  ........  0/8
8-GPU pod: Pending                            8-GPU pod: bound to h2
```

Sixteen GPUs are free on the left and none is usable by an 8-GPU pod. A useful capacity metric is
**stranded GPUs for a pod shape of k GPUs** — the free GPUs that cannot host one more such pod,
`Σ_nodes (free_n mod k)` — and **fragmentation** = stranded / free (`scheduler.stranded_gpus()`,
`scheduler.fragmentation()`): 16 and 100% on the left, 0 and 0% on the right. Track it per pod shape
you care about, next to utilisation: a cluster can be 50% utilised and 100% fragmented for its largest
jobs. The shapes themselves come from model sizing — how many GPUs one replica needs for its weights and
KV cache ([capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md)). Other levers:
separate node pools per pod shape, pod shapes that divide the node (a 5-GPU pod strands 3 GPUs on every
8-GPU node), and defragmentation by rescheduling (the descheduler, or Kueue
preemption) — the scheduler itself never moves a running pod.

### 3.5 Priority and preemption

A pod's priority comes from a `PriorityClass` (`scheduling.k8s.io/v1`; `preemptionPolicy:
PreemptLowerPriority` or `Never`). When no node passes the filters, the `DefaultPreemption` PostFilter
looks for nodes where evicting **lower-priority** pods would make room (`plugins.select_victims()`): it
removes all of them, checks the preemptor fits, then adds them back **most important first** (higher
priority, then earlier start) and evicts only those that do not fit back. Among candidate nodes it picks
(`plugins.pick_preemption_node()`), in order: fewest PodDisruptionBudget violations, lowest
highest-victim priority, lowest sum of victim priorities — each offset by 2³¹, so fewer victims wins
first — fewest victims, and the latest start time of the highest-priority victims. The preemptor gets
`nominatedNodeName` and binds after the victims' graceful termination.

Worked example (notebook 02): node *a* runs two priority-0 batch pods of 4 GPUs, node *b* one priority-500
evaluation pod of 8 GPUs; a priority-1000 serving pod needs 4. Node *a* costs one priority-0 victim, node
*b* one priority-500 victim, so the newer batch pod on *a* is evicted. What preemption does **not** do:
evict equal or higher priority, consolidate free GPUs, or understand jobs — preempting room for the first
pod of an 8-pod job achieves nothing if the other seven cannot follow.

---

## 4. Gangs

### 4.1 Why partial placement deadlocks

A distributed training job or a multi-host inference replica is a **gang**: its workers block at the
first collective (`init_process_group`, the first all-reduce) until every rank has joined. Partial
placement is therefore pure waste, and with two gangs it is a deadlock. Three nodes with 4 GPUs each;
jobs A and B each need 4 workers × 2 GPUs; their controllers create pods at the same time
(`gang.interleave()`, notebook 03):

```
a0 b0 a1 b1 a2 b2 a3 b3  (arrival)       h0: a0 b1   h1: b0 a2   h2: a1 b2      a3, b3: Pending
                                         12/12 GPUs allocated, 0 jobs running, forever
```

Neither job can finish its set, neither will release what it holds, and equal priorities mean preemption
does not break the tie. Admitting each job **whole** runs A on 8 GPUs while B waits holding nothing
(`gang.admit_gangs()`): 4 GPUs idle instead of 12, and B starts when A finishes.

### 4.2 Ways to get all-or-nothing

| Mechanism | How | Notes |
|---|---|---|
| **Kueue** (job level) | Kueue's webhook suspends queued Jobs on creation (`spec.suspend: true`; plain pods get the scheduling gate `kueue.x-k8s.io/admission`); Kueue admits the whole Workload against quota, then unsuspends it and injects the flavor's node selectors | quota is *logical*: an admitted job's pods may still not all fit on real nodes (fragmentation, pods Kueue does not manage). `waitForPodsReady` (on by default since the v1beta2 Configuration: timeout 30 min, `recoveryTimeout` the same, `blockAdmission: false`; only the alpha `DisableWaitForPodsReady` feature gate turns it off) evicts and requeues a job whose pods are not all Ready, with backoff; `blockAdmission: true` admits one at a time; TAS (section 5) and ProvisioningRequest (section 7) check physical capacity |
| **Coscheduling** plugin (kubernetes-sigs/scheduler-plugins) | a `PodGroup` with `minMember`; the **Permit** stage holds reserved pods until `minMember` are reserved, else rejects after a timeout | runs inside a second scheduler profile; PodGroup API `scheduling.x-k8s.io/v1alpha1` (verify) |
| **Volcano** | its own batch scheduler with `PodGroup.minAvailable`, queues and fair share | a replacement scheduler; common in HPC-style clusters |
| **Kubernetes native** (KEP-4671) | `Workload` and `PodGroup` APIs in `scheduling.k8s.io`; the scheduler places a pod group together | alpha in 1.35 behind the `GenericWorkload` feature gate, beta in 1.37 and stable targeted for 1.38 per the KEP metadata (verify the release you run); Kueue plans to integrate |

The pattern that works today on GKE and elsewhere is Kueue in front of the default scheduler: jobs queue
as whole units, TAS or a ProvisioningRequest makes sure the nodes exist and fit, and `waitForPodsReady`
catches what slips through.

### 4.3 JobSet and LeaderWorkerSet

A gang also needs a workload API that treats its pods as one thing:

* **JobSet** (`jobset.x-k8s.io/v1alpha2`) — a group of Jobs for training and HPC: `replicatedJobs`
  (e.g. one driver, N workers), a headless Service for stable hostnames, `failurePolicy.maxRestarts`
  (a failed Job recreates the whole set — resume from checkpoint), `successPolicy`,
  `startupPolicy.startupPolicyOrder: InOrder`, and `alpha.jobset.sigs.k8s.io/exclusive-topology` to give
  each child Job a whole topology domain. Kueue admits a JobSet as one Workload.
* **LeaderWorkerSet** (`leaderworkerset.x-k8s.io/v1`) — replicas that are **groups** of pods, for
  multi-host inference: `leaderWorkerTemplate.size` pods per group (one leader, size−1 workers, optional
  separate `leaderTemplate`), `restartPolicy: RecreateGroupOnPodRestart` so a failed shard restarts its
  group, `startupPolicy: LeaderCreated | LeaderReady`, environment `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE`,
  `LWS_WORKER_INDEX`, `leaderworkerset.sigs.k8s.io/exclusive-topology` to keep a group in one domain,
  and a scale subresource so an HPA scales *groups* ([serving-orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md)
  §4 *Autoscaling*). Its sibling **DisaggregatedSet** coordinates prefill and decode LWSs (same primer, §5).

---

## 5. Topology-aware placement

### 5.1 The tree

"All pods running" is not enough: a gang runs at the speed of the slowest link its collectives cross.

```
block ──────────── spine links (oversubscribed)
 ├─ sub-block ──── hosts on the same leaf switches: full bandwidth inside
 │   ├─ host ───── NVLink / NVSwitch between its 8 GPUs (hundreds of GB/s per GPU)
 │   │   └─ GPU
 │   └─ host
 └─ sub-block
```

The bandwidth of each level, and what a tensor-parallel all-reduce costs across it, is layer 01
([roofline-and-fabric primer](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), §5.3 and §5.4). The design rule that follows
here: keep tensor-parallel and expert-parallel groups inside the NVLink domain, and keep every gang in the
**smallest** domain that holds it. Inside one node the same rule applies to devices — the device plugin's
`GetPreferredAllocation` keeps a container on one NVLink island (`DevicePlugin.get_preferred_allocation()`),
and the kubelet's Topology Manager can align CPUs and devices by NUMA node.

### 5.2 Kueue Topology-Aware Scheduling

Kueue TAS (beta, on by default since v0.14) models the tree with node labels. An admin creates a
`Topology` (`kueue.x-k8s.io/v1beta2`) listing the levels, coarsest first — on GCE the
`cloud.google.com/gce-topology-block`, `-subblock` and `-host` labels (verify), then
`kubernetes.io/hostname` — and points a ResourceFlavor at it with `spec.topologyName`. Workloads ask with
pod-template annotations:

| Annotation | Meaning |
|---|---|
| `kueue.x-k8s.io/podset-required-topology: <level label>` | all pods in one domain of that level, or wait |
| `kueue.x-k8s.io/podset-preferred-topology: <level label>` | try that level, then each level up, then spread |
| `kueue.x-k8s.io/podset-unconstrained-topology: "true"` | anywhere, but with TAS's accurate capacity accounting |
| `kueue.x-k8s.io/podset-slice-required-topology` + `podset-slice-size` | every slice of N pods inside one domain (e.g. each 4-host slice in one sub-block) |

TAS computes free capacity per domain from Ready, schedulable nodes' allocatable, minus TAS workloads and
all other pods, then assigns pods to domains before the job starts — which also makes it a physical
capacity check that plain quota is not. Two algorithms (defaults since v0.15), both in `gpusched.gang`:

* **BestFit** (required and preferred): among domains that can hold the whole pod set, take the
  **tightest**; when it must split across child domains, take the ones with the most room first and pick
  the last as the tightest that holds the remainder (`gang.best_fit()`). The design's own example — a
  rack whose nodes have room for 3, 3, 2 and 1 pods, 7 pods to place — gives 3 + 3 + **1**, keeping the
  2-slot node whole. Below the chosen domain Kueue repeats the choice level by level over the children
  of *all* chosen domains pooled, so the host split need not follow the sub-block split.
* **LeastFreeCapacity** (unconstrained): one flat list of hosts, tightest first. It takes the tightest single
  host that holds the whole pod set, and only if none does fills the smallest gaps first — 1 + 2 + 3 + 1 for
  the same example (`gang.least_free_capacity()`) — keeping large domains intact for constrained jobs.

Worked placement (`gang.place_gang()`, notebook 03), in 8-GPU pods per sub-block free: b0-s0 = 1,
b0-s1 = 3, b1-s0 = 2 (one of its hosts has a single GPU busy, so it counts zero), b1-s1 = 4.

| Gang of 8-GPU pods | Constraint | Placement |
|---|---|---|
| 3 | required sub-block | b0-s1 — exactly 3; b1-s1 (4) is kept for something bigger |
| 2 | required sub-block | b1-s0 |
| 5 | required sub-block | none: waits |
| 5 | preferred sub-block | block b1 (2 + 4 = 6 ≥ 5); the host pass over both sub-blocks gives 2 in b1-s0 + 3 in b1-s1 |
| 3 | unconstrained | LeastFreeCapacity: no host holds 3, so 1 in b0-s0 + 2 in b0-s1 — straddles sub-blocks |

Choose the constraint per workload: **required** for a multi-host inference group whose shards exchange
activations every token (a slow replica is slow for its whole life); **preferred** for long training
jobs (waiting forever for a perfect block is worse than a slightly slower one); **unconstrained** for
embarrassingly parallel batch. TAS also replaces a failed node inside the assigned domain when it can
(hot swap), and evicts the workload when it cannot.

### 5.3 Compact placement and exclusive topology

Two related tools work at other layers. **Compact placement** asks the cloud to create a node pool's VMs
physically close (on GKE `placement_policy.type = "COMPACT"` on the node pool) — topology at creation
time rather than at scheduling time. **Exclusive topology** (JobSet and LWS annotations) gives each job
or group a whole domain so neighbours cannot share its links.

---

## 6. Queues, quotas and multi-tenancy with Kueue

### 6.1 The objects

The scheduler decides where a pod runs. **Kueue** (`kueue.x-k8s.io/v1beta2`, release v0.19.6 in September
2026) decides *whether and when* a job may start, and which flavor of capacity it gets:

| Object | Scope | Purpose |
|---|---|---|
| `ResourceFlavor` | cluster | a kind of capacity: node labels, taints/tolerations, optional `topologyName` (e.g. `h100-spot`, `h100-reserved`, `l4`) |
| `ClusterQueue` | cluster | quota per flavor and resource (`resourceGroups[].flavors[].resources[]`: `nominalQuota`, `borrowingLimit`, `lendingLimit`), `cohortName`, `preemption`, `queueingStrategy`, `admissionChecksStrategy`, `fairSharing` |
| `LocalQueue` | namespace | a team's entry point; points at one ClusterQueue |
| `Workload` | namespace | Kueue's view of one job: pod sets × requests, priority, admission status |
| `WorkloadPriorityClass` | cluster | queueing/preemption priority (label `kueue.x-k8s.io/priority-class`), independent of pod priority |
| `AdmissionCheck`, `ProvisioningRequestConfig` | cluster | extra gates before admission, e.g. a ProvisioningRequest for capacity (section 7) |
| `Topology`, `Cohort` | cluster | TAS levels (section 5); hierarchical cohorts |

```
Job (label kueue.x-k8s.io/queue-name: gpus)
  │ created suspended
  ▼
Workload ─► LocalQueue team-a/gpus ─► ClusterQueue team-a-cq ─► quota reserved ─► admission checks ─► admitted
                                        (flavor chosen,                             (ProvisioningRequest,   │
                                         maybe borrowing,                            TAS assignment)       ▼
                                         maybe preempting)                                        Job unsuspended; pods
                                                                                                  get flavor nodeSelector
```

### 6.2 Quota arithmetic: nominal, borrowing, lending

ClusterQueues with the same `cohortName` lend each other **unused** nominal quota. Kueue's rule for how
much of a (flavor, resource) a ClusterQueue can use right now (`pkg/cache/scheduler/resource_node.go`,
flat cohort; `quota.Kueue.available()`):

```
guaranteed      = nominal − lendingLimit          (0 if no lendingLimit; never lent out)
cohort pool     = Σ_members (nominal − guaranteed)
pool in use     = Σ_members max(0, usage − guaranteed)
from cohort     = pool − pool in use,  capped at (nominal − guaranteed) − max(0, usage − guaranteed) + borrowingLimit
available       = max(0, guaranteed − usage) + from cohort
```

Kueue's documentation examples come out exactly: team A (9 CPUs) and team B (12) in one cohort, both idle
— A can use **21**; with A's `borrowingLimit: 1`, **10**; with B's `lendingLimit: 1` instead, A can use
**10** (the documented 9 + 1 in both cases). With GPUs (notebook 04): two ClusterQueues of 16 GPUs each;
team B runs three 8-GPU jobs, the third **borrowing** 8 of team A's idle GPUs. Team A can now use 8, not 16.

### 6.3 Preemption: nominal quota is only a guarantee if you reclaim

Team A submits a 16-GPU job. With the default `reclaimWithinCohort: Never`, nothing happens: the job
waits (`couldn't assign flavors to pod set main: insufficient unused quota for nvidia.com/gpu in flavor
h100, 8 more needed` — `Kueue.explain()`) until B's borrowing job ends on its own. **Nominal quota is not a
guarantee unless reclaim is on** (or a `lendingLimit` keeps part of it home). The policies:

| Field | Values | Effect |
|---|---|---|
| `reclaimWithinCohort` | Never, LowerPriority, Any | preempt workloads of **borrowing** ClusterQueues in the cohort to get lent quota back |
| `withinClusterQueue` | Never, LowerPriority, LowerOrNewerEqualPriority | preempt lower-priority (or equal-priority newer) workloads in the same queue |
| `borrowWithinCohort` | policy Never / LowerPriority, `maxPriorityThreshold` | preempt while the preemptor itself borrows (classic preemption only) |

Classic preemption (`quota.Kueue.preemption_targets()`) is eligible when the preemptor fits within its
nominal quota or `borrowWithinCohort` is on. Candidates are ordered **other ClusterQueues first, then
lowest priority, then most recently admitted**; they are removed greedily until the preemptor fits — a
candidate from another queue only while that queue is still borrowing — and the set is then minimised in
reverse. With `reclaimWithinCohort: Any`, team A's 16-GPU job preempts exactly B's third (newest,
borrowing) job and is admitted; B is back at its nominal 16. With `withinClusterQueue: LowerPriority`, a
priority-100 job in a full queue evicts the newest priority-0 job.

**Fair sharing** is the alternative algorithm: each ClusterQueue gets a weighted share value of the
borrowable resources, admission favours the lowest share and preemption takes from the highest, under
`preemptionStrategies` such as `[LessThanOrEqualToFinalShare, LessThanInitialShare]`. Admission Fair
Sharing extends the idea to LocalQueues within a ClusterQueue.

### 6.4 Queueing strategy and flavors

`StrictFIFO` admits in order (priority, then creation) and a head that does not fit blocks the queue;
`BestEffortFIFO` (the default) lets later workloads past a blocked one. On a 16-GPU queue receiving 8, 16
and 8 GPU jobs, StrictFIFO admits only the first; BestEffortFIFO admits both 8-GPU jobs — better
utilisation, but big jobs can starve behind a stream of small ones. Flavors are tried in the listed order:
by default a flavor that fits (even by borrowing) is taken (`whenCanBorrow: MayStopSearch`), and a flavor
that needs preemption is only chosen if no later flavor fits (`whenCanPreempt: TryNextFlavor`) — so
listing `h100-reserved` before `h100-spot` expresses a cost preference.

A serving/batch split in one cohort (notebook 04, exercise 4.4):

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata: {name: serving-cq}
spec:
  cohortName: shared
  namespaceSelector: {}
  preemption: {reclaimWithinCohort: Any, withinClusterQueue: Never}
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: h100
      resources: [{name: "nvidia.com/gpu", nominalQuota: 16, lendingLimit: 8}]
```

Serving keeps 8 GPUs at home, lends the other 8, and recalls them at once when it scales up; batch
(`borrowingLimit: 8`, `withinClusterQueue: LowerPriority`) fills idle quota and sorts itself out by
priority. The same idea one layer up — admit whole units of work against a budget, shed or queue the
rest — is the gateway's admission control
([agentic scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.3).

---

## 7. Getting capacity

### 7.1 The cluster autoscaler for GPU pools

A Pending GPU pod is a request for a machine. The cluster autoscaler, every scan interval (10 s):

```
pending pods ─► simulate them on each node pool's template node ─► bin-pack: nodes needed per pool
            ─► expander picks a pool ─► cloud creates VMs ─► driver + device plugin ─► GPUs allocatable ─► scheduler binds
```

* **How many nodes** — a bin-packing estimate of the pending pods onto empty template nodes
  (`autoscaler.nodes_needed()`, first-fit decreasing): pods of 4, 4, 2, 2, 1, 1, 1, 1 GPUs need **2**
  8-GPU nodes; three 5-GPU pods need **3** and strand 3 GPUs on each.
* **Which pool** — the expander: `least-waste` (the default: least idle CPU, then memory), `random`,
  `most-pods`, `least-nodes`, `price`, `priority`, `grpc`, chainable. Least-waste compares idle
  resources, not dollars. The simulator's version (`autoscaler.least_waste()`) ranks idle GPUs, and asked
  for 4 + 2 + 1 + 1 GPUs it finds one 8-GPU H100 node and two 4-GPU L4 nodes equally wasteful (zero idle
  GPUs) and takes the H100 on its tie-break. So pods pin a GPU type with a node selector, and a `price` or
  `priority` expander or a ComputeClass (7.3) encodes cost.
* **Down again** — a node goes after it has been unneeded for `--scale-down-unneeded-time` (10 min) and
  no scale-up happened for `--scale-down-delay-after-add` (10 min); for GPU nodes only GPU utilisation
  counts against `--scale-down-gpu-utilization-threshold` (0.5). Nodes that do not register within
  `--max-node-provision-time` (15 min) are given up on.

Scale from zero, simulated (`autoscaler.simulate()`, notebook 05, which models both 10-minute delays):
a 1-hour job on an empty 1-GPU pool whose nodes take 300 s to show allocatable GPUs starts at 300 s,
finishes at 3,900 s, and its node is removed at 4,500 s (the only scale-up was at t = 0, so the unneeded
time binds) — **1.25 node-hours billed for 1 hour of work**, before any image or weights.

### 7.2 Obtainability

GPUs are the resource a cloud may not have. The capacity types, and how each fails:

| Type | You get | How it fails | Fits |
|---|---|---|---|
| On-demand | a VM now, if the zone has it | stockout: the scale-up errors and retries; large parts are scarce | stateless serving, dev |
| Spot | 60–91% cheaper (GCP, verify) | reclaimed at any time, with short notice | interruptible, retryable batch; extra serving replicas |
| Reservation | capacity held for you, billed whether used or not | you pay for idle | steady baseline load |
| DWS flex-start (GCP) | a queued request provisioned **all at once**, for up to 7 days | waits until capacity exists | gangs with a flexible start: training, fine-tuning, big evals |
| DWS calendar mode (GCP) | a future fixed-duration block (verify) | plan ahead | planned runs |

Two account facts gate all of it on GCP: GPUs are not usable on a Free Trial billing account, and GPU quota
often starts at zero. Machine families and prices are layer 01 §10.1 and [`COMPUTE.md`](../../COMPUTE.md).

**Spot and gangs.** With independent reclaims at rate λ per node-hour, a gang of N nodes survives T hours
with probability `e^(−N·λ·T)` (`autoscaler.gang_survival()`); if every reclaim restarts it from scratch,
T hours of work with restart overhead R take `(e^(N·λ·T) − 1)·(1/(N·λ) + R)` hours on average
(`autoscaler.expected_runtime_h()`, checked by Monte Carlo in the tests, with and without R).
At an illustrative λ = 0.005 per node-hour, a 24-hour job with R = 15 min:

| Nodes | Survives 24 h | Expected wall-clock |
|---:|---:|---:|
| 1 | 88.7% | 25.5 h |
| 4 | 61.9% | 31.0 h |
| 16 | 14.7% | 74.2 h |

Gang size multiplies the hazard. Checkpoint at the interval layer 01 §7.2 derives (Young/Daly), or put
large gangs on reserved or flex-start capacity.

### 7.3 Queued provisioning, ComputeClass, Autopilot

A gang has a problem an ordinary scale-up cannot solve: nodes arrive one at a time, and every node that
arrives before the last one is billed while it waits. Worse, a pool whose `max_nodes` is below the gang
size scales up *part* of the gang, and those nodes idle forever (simulated: 3 nodes, 6 node-hours in 2
hours, job never starts). **Queued, all-or-nothing provisioning** holds the request until the whole gang
can be created:

* **ProvisioningRequest** (`autoscaling.x-k8s.io/v1`, cluster-autoscaler): classes
  `check-capacity.autoscaling.x-k8s.io` and `best-effort-atomic-scale-up.autoscaling.x-k8s.io`, plus
  provider classes — GKE's queued provisioning, `queued-provisioning.gke.io` (verify). Kueue drives it as an
  **admission check** (`ProvisioningRequestConfig`): quota is reserved, the request is created, and the job
  is admitted only when the capacity is `Provisioned`.
* **GKE DWS flex-start** node pools with queued provisioning (`--flex-start --enable-queued-provisioning`;
  in Terraform `node_config.flex_start` and `queued_provisioning.enabled`) serve those requests atomically.

Simulated (`autoscaler.provision()`, notebook 05): 16 nodes of 8 GPUs, each missing node granted with 3%
probability per minute, 300 s boot. In one run (seed 0) the gang can start after 2.47 h either way; the
ordinary pool paid **31.3 node-hours (251 GPU-hours)** for nodes waiting on their peers, the queued pool
**1.3** (only the boot, in every run); over 200 seeds the ordinary pool averages **22.4 node-hours** and a
1.95 h start. Queued provisioning does not create capacity; it stops you paying for the partial set.

A **custom ComputeClass** (`cloud.google.com/v1`, verify) gives one class of workload an ordered fallback
list — for example reservation → Spot → on-demand → flex-start — and GKE's node auto-provisioning creates
matching node pools on demand; pods select the class with a node selector (verify the field names against
the CRD; the lab's `deploy/gke/` has an example). **Autopilot** takes node pools away entirely: a pod
requests `nvidia.com/gpu` and selects an accelerator type, and GKE provisions and bills per pod (verify
selectors and limits). Outside GCP the same ideas are Karpenter node pools with capacity types (AWS,
Azure), capacity reservations and blocks for ML, or — on-prem — a fixed fleet where quota (section 6) is the
only elasticity (verify provider specifics).

---

## 8. Startup latency

Capacity is not useful until a pod on it serves. The chain from Pending to Ready, and the bandwidth
arithmetic of each hop, is layer 01 §6.2; this is the Kubernetes side:

```
Pending ─► node (create VM, boot) ─► driver + device plugin ─► image pull ─► weights ─► engine warm-up ─► Ready
            warm pool, min nodes,       GKE auto-install;          image          GCS FUSE,    startup probe,
            balloon pods                resourceUnready until      streaming,     Hyperdisk ML, readiness gate
                                        GPUs are allocatable       secondary      model
                                                                   boot disks     streamers
```

Illustrative inputs through `autoscaler.startup_latency()` (notebook 05): a new node (150 s) with driver
install (90 s), a 12 GB image pulled at 0.25 GB/s, 16 GB of weights at 0.5 GB/s and 60 s of warm-up take
**380 s**; the same replica on a warm node, with the image streamed (2 GB/s effective) and weights from a
fast cache (4 GB/s), takes **70 s**, most of it warm-up. The levers, stage by stage:

* **Node** — keep `min_nodes` above zero for latency-critical pools, or run **balloon pods**: placeholder
  pods requesting the GPU shape to keep warm, in a `PriorityClass` that is negative but not below the
  autoscaler's `--expendable-pods-priority-cutoff` (default −10: lower pods neither trigger scale-up nor
  block scale-down; the FAQ uses −10), with `terminationGracePeriodSeconds: 0`. Real pods preempt them
  at once (section 3.5), and the Pending balloon makes the autoscaler add a node for it.
* **Image** — GKE **image streaming** (`gcfs_config` in Terraform) starts containers before the image is
  fully pulled; **secondary boot disks** (`secondary_boot_disks`) ship a disk with images or data
  preloaded (verify supported sources and registries).
* **Weights** — mount a bucket with the **Cloud Storage FUSE CSI driver**
  (`addons_config.gcs_fuse_csi_driver_config`) and its caching options, a read-only-many **Hyperdisk ML**
  volume, or stream weights into GPU memory with a model streamer (verify current options per engine);
  layer 01 §6.1 has the parallel-read arithmetic.
* **Warm-up** — a **startup probe** long enough for load + CUDA-graph capture so the kubelet does not kill
  a slow-loading pod, and a readiness probe so traffic arrives only when the engine is serving (what the
  engine does at start-up: [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) §1).

A cold start of minutes is why autoscaling LLM replicas needs headroom and scale-ahead signals
([serving-orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md) §4.3 *Cold start
anatomy*), and why "scale to zero" is a cost decision with a latency price.

---

## 9. Sharing GPUs at the cluster level

Because a GPU is an integer, sharing one means the node advertises *more integers*. The mechanics of each
method (MIG profiles and their placement, MPS, time-slicing latency) are layer 02
([cuda-and-nccl primer](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §7 *Sharing a GPU*); the cluster
view:

| Method | What the node advertises | Isolation | Good for |
|---|---|---|---|
| Whole GPU | `nvidia.com/gpu: 8` | full | training, large-model serving |
| **MIG** (A100, H100 and newer) | each partition is a device: `nvidia.com/gpu` (single strategy) or `nvidia.com/mig-1g.10gb` etc. (mixed); GKE `gpu_partition_size` per node pool | hardware: memory, cache and SMs split; faults contained | many small models with predictable size |
| **Time-slicing** | `replicas` × GPUs, e.g. 8 × 10 = **80** `nvidia.com/gpu` (or `nvidia.com/gpu.shared`) | none: shared memory, one fault can take all | dev, notebooks, bursty light inference |
| **MPS** | replicas with per-client memory and compute limits | partial | throughput-oriented small kernels |
| **DRA** | claims shared by several containers or pods; MIG devices as `mig.nvidia.com` (profile attribute) | per driver | the direction of travel (section 1.5) |

The trap with time-slicing: a container that requests 2 "GPUs" may get two slices of the **same** physical
GPU — `DevicePlugin(replicas=10)` hands out `GPU-fake-0000` twice in the simulator, exactly as the NVIDIA
plugin can. NVIDIA's `failRequestsGreaterThanOne` option rejects such requests, and on GKE time-sharing
nodes a container may request at most one `nvidia.com/gpu` (the GKE device plugin enforces it). MIG
partition counts are fixed per profile — an A100 40 GB offers seven `1g.5gb`, three `2g.10gb` or two
`3g.20gb` slices; H100 80 GB seven `1g.10gb` (per GKE's device plugin; verify for your GPU and driver).
Node-level sharing settings are per node pool on GKE (`gpu_sharing_config.gpu_sharing_strategy`,
`max_shared_clients_per_gpu`), so sharing is a *pool* decision: put shared and exclusive GPUs in different
pools and let labels route the pods.

---

## 10. Learning locally

Every mechanism in this primer except real devices and real provisioning can run on a laptop.

### 10.1 kind with fake GPU capacity

A kind cluster (Kubernetes in Docker) runs the real scheduler, real Kueue, JobSet and LWS. Extended
resources can be **advertised by hand** — a node-status patch sets capacity, and the kubelet derives
allocatable from capacity on its next status update:

```bash
kubectl patch node kind-worker --subresource=status --type=json \
  -p '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"8"}]'
kubectl label node kind-worker cloud.google.com/gke-accelerator=nvidia-h100-80gb
kubectl taint node kind-worker nvidia.com/gpu=present:NoSchedule
```

(`~1` escapes `/` in a JSON-patch path.) kind leaves ExtendedResourceToleration off (section 1.4): give GPU
pods the `nvidia.com/gpu` Exists/NoSchedule toleration yourself, or enable the plugin through a kind
`kubeadmConfigPatches` entry for the API server (keep `NodeRestriction`). The scheduler then places GPU pods,
Kueue admits and preempts against real quota, TAS reads your topology labels — and containers get no device,
because no device plugin answers `Allocate`. Lab notebook `02_kind_with_fake_gpus_and_kueue` scripts this (and falls back to a
bundled simulator when Docker is absent).

### 10.2 KWOK and fake-gpu-operator

**KWOK** (Kubernetes WithOut Kubelet) creates hundreds of fake Nodes with any capacity and labels, and fakes
pod lifecycles — enough to test scheduling, Kueue and TAS behaviour at fleet scale on a laptop. Run:ai's
**fake-gpu-operator** goes one step further on CPU-only nodes: a simulated device plugin, GPU Feature
Discovery labels, MIG and Prometheus metrics, so dashboards and label-based placement behave as on GPUs.

| Real on a laptop | Simulated on a laptop |
|---|---|
| scheduler filters, scores, preemption, events | GPU devices, driver, `Allocate`, CUDA |
| Kueue admission, borrowing, reclaim, TAS assignment | NVLink / network bandwidth and topology effects on speed |
| JobSet / LWS lifecycles, restarts, gang admission | node provisioning, stockouts, Spot reclaims, DWS (this core's `autoscaler.py`) |
| manifests, linting, "why Pending" diagnosis | DCGM metrics (fake-gpu-operator), real utilisation |

### 10.3 Where each concept runs

| Concept | T0: laptop / Colab CPU | T1–T2: rented GPU box | T3: GKE | Other clouds |
|---|---|---|---|---|
| device plugin, capacity/allocatable | `k8s-gpu-core` notebook 01 | a Lambda VM or any VM with GPUs + k3s/kubeadm + GPU Operator (RunPod/Vast are containers: no kubelet of your own) | GPU node pool, driver auto-install | EKS/AKS + GPU Operator or the provider's device plugin (verify) |
| filter/score/preemption | notebook 02; kind | same | the managed scheduler (+ your own profile) | same |
| gangs, JobSet/LWS, Kueue, TAS | notebooks 03–04; kind + Kueue v0.19.6 | same | Kueue on GKE; GCE topology labels | Kueue anywhere; provider topology labels differ |
| autoscaling, Spot, queued provisioning | notebook 05 (simulated) | – | node pools, Spot, DWS flex-start, ComputeClass (lab notebook 04) | Karpenter, capacity reservations (verify) |
| sharing (MIG, time-slicing) | notebook 01, exercise 1.6 (time-slicing replicas) | an A100/H100 VM for MIG | node-pool sharing settings | GPU Operator configs |

Colab and Kaggle give you notebooks, not clusters: they run the core (T0) but not kind, which needs a Docker
daemon. Prices, free tiers and how obtainable each GPU is: [`COMPUTE.md`](../../COMPUTE.md). The order to
work the whole curriculum: [`CURRICULUM.md`](../../CURRICULUM.md).

---

## In a design review

**The two-minute walkthrough.** "GPUs reach Kubernetes as integers: the device plugin — managed by GKE here,
the GPU Operator elsewhere — advertises healthy devices, requests are whole GPUs with requests equal to
limits, GPU nodes are tainted and pods pick a model with labels. We run separate node pools per GPU shape.
The default scheduler spreads pods by CPU and memory and ignores GPUs, which strands GPUs, so GPU pools use
a MostAllocated profile with the GPU weighted, and we track stranded GPUs for our largest pod shape.
Anything with more than one pod — training JobSets, multi-host LeaderWorkerSet replicas — goes through
Kueue: jobs are admitted whole against ClusterQueue quota, placed topology-aware with TAS (required for
inference groups, preferred for training), and evicted and requeued if their pods do not all come up.
Teams share a cohort: idle quota is lent, lending limits keep a floor for serving, and reclaim makes nominal
quota a real guarantee. Capacity: serving on on-demand with Spot behind it; interruptible batch on Spot;
large gangs through a ProvisioningRequest on DWS flex-start so they get all their nodes at once; the steady
base on reservations. Cold start is minutes, so latency-critical pools keep warm nodes, images stream and
weights come from a cache."

**Drill questions.**

1. *An 8-GPU pod is Pending with `Insufficient nvidia.com/gpu` although the cluster is half idle. Why, and
   what do you change?* — Fragmentation: the free GPUs sit in pieces smaller than 8 on each node. Pack GPU
   pools (MostAllocated with a GPU weight), separate pools by pod shape, or preempt/defragment; measure
   stranded GPUs for the shape.
2. *Two training jobs have been "running" for an hour and neither has logged a step. What happened?* —
   Partial gang placement: each holds some GPUs and waits for the rest. Admit jobs whole (Kueue with
   `waitForPodsReady`, or a gang scheduler) so one runs and the other waits holding nothing.
3. *Team A has 16 GPUs of nominal quota, runs nothing, and its 16-GPU job is Pending. Why?* — Its unused
   quota was lent to the cohort and `reclaimWithinCohort` is `Never`, so borrowers keep it until they finish.
   Enable reclaim, or set a `lendingLimit` to keep part of the quota home.
4. *Required or preferred topology for a 2-node tensor-parallel serving group, and for a 32-node training
   job?* — Required at the smallest multi-node domain for serving (a split replica is slow for its whole
   life); preferred at block level for training (locality, but not at the price of waiting indefinitely).
5. *Why not run a 16-node training job on an autoscaled Spot pool?* — Any reclaim restarts the gang:
   survival falls as `e^(−N·λ·T)` (14.7% for 16 nodes over 24 h at λ = 0.005/node-h, simulated), and
   node-by-node scale-up bills idle partial gangs. Use flex-start or reservations and checkpoint.
6. *A GPU node shows capacity 8, allocatable 7. What does that mean for running and new pods?* — One device
   is Unhealthy. Running pods keep their GPUs; new pods see 7; a pod the scheduler placed on stale status
   can fail with `UnexpectedAdmissionError`. Drain and repair the node deliberately.

---

## Glossary

| Term | Meaning |
|---|---|
| Extended resource | A resource named with a domain prefix (`nvidia.com/gpu`); integer, requests = limits, never overcommitted |
| Capacity / allocatable | What a node has / what the scheduler may assign; for GPUs, all devices / healthy devices |
| Device plugin | Node agent that registers a resource with the kubelet, streams device health and answers `Allocate` |
| DRA | Dynamic Resource Allocation: devices described by ResourceSlices and requested by ResourceClaims with CEL selectors |
| GPU Operator | NVIDIA's operator installing driver, container toolkit, device plugin, GFD, DCGM and MIG manager |
| GFD / NFD | GPU Feature Discovery / Node Feature Discovery: write node labels describing hardware |
| Taint / toleration | Node mark that repels pods / pod permission to ignore it |
| Filter / score | Scheduler phases: feasible nodes, then ranking of feasible nodes |
| LeastAllocated / MostAllocated | NodeResourcesFit scoring strategies: spread / bin-pack |
| Stranded GPUs | Free GPUs that cannot host one more pod of a given size |
| Preemption | Evicting lower-priority pods (scheduler) or workloads (Kueue) to make room |
| Gang | A group of pods that is useful only when all run; placed all-or-nothing |
| JobSet | API for a group of Jobs run as one training/HPC workload |
| LeaderWorkerSet (LWS) | API whose replicas are groups of pods, for multi-host inference |
| Kueue | Job queueing: quotas, cohorts, admission, preemption, topology-aware scheduling |
| ResourceFlavor | Kueue's name for a kind of capacity (GPU model, Spot vs on-demand) |
| ClusterQueue / LocalQueue | Kueue quota holder / a namespace's entry point to it |
| Cohort | ClusterQueues that lend each other unused quota |
| Nominal quota, borrowingLimit, lendingLimit | Owned quota; cap on borrowing; cap on lending |
| TAS | Topology-Aware Scheduling in Kueue: places pod sets inside topology domains |
| BestFit / LeastFreeCapacity | TAS algorithms: tightest domain that fits / tightest single host, else smallest gaps first |
| Cluster autoscaler | Adds nodes for pending pods and removes idle ones |
| Expander | The autoscaler's rule for choosing which node pool to grow |
| ProvisioningRequest | API asking the autoscaler for capacity for a set of pods, possibly atomically |
| DWS flex-start | GCP Dynamic Workload Scheduler mode provisioning a queued request all at once, up to 7 days |
| ComputeClass | GKE CRD giving workloads an ordered list of capacity options |
| MIG / time-slicing / MPS | Hardware partitioning / temporal sharing / spatial sharing of one GPU |
| KWOK | Kubernetes WithOut Kubelet: fake nodes and pods for testing control-plane behaviour |

---

## Sources

* Kubernetes source (master, September 2026): `pkg/scheduler/apis/config/v1/default_plugins.go` and
  `defaults.go` (default plugins, weights, NodeResourcesFit defaults);
  `pkg/scheduler/framework/plugins/noderesources/{least_allocated,most_allocated,resource_allocation,fit}.go`;
  `pkg/scheduler/framework/preemption/preemption.go`, `plugins/defaultpreemption/default_preemption.go`;
  `pkg/scheduler/framework/types.go` (FitError message); `pkg/apis/core/validation/validation.go`
  (extended-resource rules); `pkg/kubelet/cm/devicemanager/manager.go` (capacity vs allocatable);
  `staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1/api.proto`;
  `plugin/pkg/admission/extendedresourcetoleration` and `pkg/kubeapiserver/options/plugins.go` (default-off);
  `staging/src/k8s.io/api/resource/v1/types.go` (DRA); kind `pkg/cluster/internal/kubeadm/config.go`.
* KEP-4671 Gang Scheduling (`kubernetes/enhancements`, `keps/sig-scheduling/4671-gang-scheduling`).
* Kueue (`kubernetes-sigs/kueue`): README (v0.19.6), `apis/kueue/v1beta2/*_types.go`, concepts docs
  (cluster_queue, preemption, topology_aware_scheduling, admission_check/provisioning_request,
  workload_priority_class), `apis/config/v1beta2/defaults.go` (waitForPodsReady), KEP-2724 and
  `pkg/cache/scheduler/tas_flavor_snapshot.go` (TAS algorithms), `pkg/cache/scheduler/resource_node.go`
  (quota arithmetic), `pkg/scheduler/preemption/preemption.go` (classic preemption).
* JobSet (`kubernetes-sigs/jobset`, `api/jobset/v1alpha2`), LeaderWorkerSet (`kubernetes-sigs/lws`,
  `api/leaderworkerset/v1`).
* NVIDIA: `k8s-device-plugin` README and GPU Feature Discovery label table; `gpu-operator` README and
  ClusterPolicy types; `kubernetes-sigs/dra-driver-nvidia-gpu` README and quickstart specs.
* Google: `GoogleCloudPlatform/container-engine-accelerators` (GKE GPU device plugin: sharing rules, MIG
  partition sizes); Terraform google provider 8.x schemas for node-pool and cluster attributes.
* cluster-autoscaler (`kubernetes/autoscaler`): FAQ (expanders, scale-down flags, expendable pods and
  overprovisioning), ProvisioningRequest v1 types, GCE cloud provider (GPU label). Kubernetes docs:
  *Advertise Extended Resources for a Node*; KWOK (`kubernetes-sigs/kwok`); Run:ai `fake-gpu-operator`.

---

## Verify list

Dated 26 September 2026. Re-check before relying on any of these.

| Fact | Status here |
|---|---|
| Kueue latest release v0.19.6, API `kueue.x-k8s.io/v1beta2`, TAS beta and on by default | from upstream README / docs |
| JobSet `jobset.x-k8s.io/v1alpha2` (release v0.12.0 in its README); LWS `leaderworkerset.x-k8s.io/v1`, tested on K8s 1.34–1.37 | from upstream READMEs |
| DRA `resource.k8s.io/v1` GA in 1.34; NVIDIA DRA GPU plugin "not yet officially supported", ComputeDomains supported | upstream; changes fast |
| Native gang scheduling (KEP-4671): alpha 1.35, beta 1.37, stable targeted 1.38 | KEP metadata — verify the release you run |
| Kueue `waitForPodsReady` on by default (v1beta2 Configuration, timeout 30 min); `DisableWaitForPodsReady` alpha gate | Kueue v0.19.6 `apis/config/v1beta2/defaults.go` |
| GKE GPU taint `nvidia.com/gpu=present:NoSchedule` and ExtendedResourceToleration enabled; `cloud.google.com/gce-topology-{block,subblock,host}` labels | verify (the plugin is default-off upstream) |
| GKE driver auto-install from 1.32.2-gke.1297000; `gpu_driver_version` DEFAULT / LATEST / INSTALLATION_DISABLED | from session research |
| GKE `optimize-utilization` profile makes the scheduler bin-pack | verify |
| ProvisioningRequest class for GKE queued provisioning `queued-provisioning.gke.io`; DWS flex-start up to 7 days; calendar mode | verify |
| ComputeClass `cloud.google.com/v1` field names (`priorities`, `spot`, `flexStart`, reservations, `nodePoolAutoCreation`) | verify against the CRD |
| Autopilot GPU selectors and limits; image streaming and secondary boot disk sources; GCS FUSE CSI caching options; Hyperdisk ML; model streamers per engine | verify |
| Spot discount 60–91% (GCP); Spot reclaim rates | discount from session research; the λ used above is illustrative, not a measured rate |
| cluster-autoscaler defaults: least-waste expander, 10 min unneeded and delay-after-add, 0.5 thresholds, 15 min provision time | from the FAQ on master |
| MIG profile counts per GPU (A100 40 GB 1g.5gb ×7, 2g.10gb ×3, 3g.20gb ×2; H100 80 GB 1g.10gb ×7) | from GKE's device plugin source; verify for your driver |
| NFD PCI label form `feature.node.kubernetes.io/pci-10de.present` | depends on NFD configuration |
