# Kubernetes for GPUs: how a GPU becomes schedulable, and how to place, share, queue and scale it

*This is layer 03 of the stack. A check on 26 September 2026 compared the facts with upstream sources. The sources
are Kubernetes (kube-scheduler, kubelet, API validation), Kueue v0.19 (`kueue.x-k8s.io/v1beta2`), JobSet,
LeaderWorkerSet, the NVIDIA device plugin and DRA driver, and cluster-autoscaler. For some GKE specifics, a check
against Google's documentation was not possible. These specifics have the (verify) tag.*

*The core simulator, [`k8s-gpu-core`](k8s-gpu-core) (package `gpusched`), calculates every worked number. The name
of the function is next to the number. Durations, prices and failure rates are inputs, and their outputs have the
label simulated.*

This primer covers the layer between the GPU software substrate (driver, CUDA, container runtime, layer 02) and the
inference engine (layer 04). It explains these topics:

- how the GPUs of a node become a number that Kubernetes can schedule,
- how the scheduler places pods on those numbers, and why that causes GPU fragmentation,
- why you must place multi-pod jobs all-or-nothing and near each other,
- how Kueue shares a fleet between teams, with quotas that borrow and reclaim,
- how you get GPU capacity, start it and share it.

The primer is for an engineer who knows the basics of Kubernetes (pods, nodes, labels). That engineer wants to
explain the design of a GPU platform in a review.

The short version of this primer is §7 of the [GPU deployment primer](../../01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md).
The detailed lab, [`k8s-gpu-lab`](k8s-gpu-lab), uses the same ideas in manifests, in kind with fake GPUs, and in
GKE.

---

## The one-minute version

A GPU is an **integer** to Kubernetes.

- A device plugin tells the kubelet how many devices a node has and which devices are healthy. The node
  advertises `nvidia.com/gpu: 8`. Pods request whole GPUs, with requests equal to limits. Kubernetes never
  overcommits the GPUs. Labels tell *which* GPU. Taints keep other pods off GPU nodes.
- The scheduler places **one pod at a time**: filter, score, bind. Its default score spreads pods by CPU and
  memory, and ignores GPUs. Thus small GPU pods cause **fragmentation** on nodes, until a large pod fits on no
  node although half the GPUs are free. Bin-pack GPU pools instead.
- Distributed jobs are **gangs**. A gang has no use unless every pod runs, and it is slow unless its pods are
  **near** each other (same NVLink domain, sub-block, block). If the scheduler places their pods one at a time,
  the cluster can go into a deadlock. Admit gangs whole (Kueue). Use topology-aware placement for them (Kueue
  TAS). Run them as JobSets or LeaderWorkerSets.
- **Kueue** decides *if* a job can start. ClusterQueues own GPU quota per flavor. Cohorts lend idle quota, and
  the owners **reclaim** it by preemption.
- **Capacity** is the hard part. GPU nodes take minutes to become useful. It is possible that they are not
  available at all. The cloud can reclaim them (Spot). A gang needs all its nodes at the same time. Queued,
  all-or-nothing provisioning (DWS flex-start) is for this problem.

---

## 1. What Kubernetes sees

### 1.1 A GPU is an extended resource

The status of a Node has two resource lists. **Capacity** is what the node has. **Allocatable** is what the
scheduler can give to pods. For `cpu` and `memory`, allocatable is capacity minus the system reservations. For a
GPU, it is the number of **healthy** devices:

```
$ kubectl describe node gpu-node-0          # the parts that mention GPUs
Labels:       cloud.google.com/gke-accelerator=nvidia-h100-80gb
Taints:       nvidia.com/gpu=present:NoSchedule
Capacity:     nvidia.com/gpu: 8
Allocatable:  nvidia.com/gpu: 7             # one device reported Unhealthy
```

`nvidia.com/gpu` is an **extended resource**: a name with a domain prefix outside `kubernetes.io`. The API server
applies three rules to the extended resources of a container. The scheduler never breaks the fourth rule.
`gpusched.cluster.effective_requests()` reproduces the four rules with the upstream messages:

| Rule | Rejected spec | Message |
|---|---|---|
| Whole numbers only | `limits: {nvidia.com/gpu: 0.5}` | `must be an integer` |
| Request must equal limit | `requests: 1`, `limits: 2` | `must be equal to nvidia.com/gpu limit of 2` |
| A limit is necessary | `requests: 1`, no limit | `Limit must be set for non overcommitable resources` |
| No overcommit | the sum of requests on a node is never more than allocatable | (the pod stays Pending) |

A limit without a request sets the request to the limit. Thus the normal form is to write only
`limits: {nvidia.com/gpu: 1}`. There are no fractional GPUs at this layer. To share a GPU (section 9), the node
advertises *more units*, not smaller units.

### 1.2 The device plugin API

Also, the kubelet knows nothing about GPUs. A **device plugin** runs on every GPU node as a DaemonSet. For NVIDIA
GPUs, this is the NVIDIA device plugin, which the GPU Operator installs. On GKE, it is Google's own GPU device
plugin. The device plugin uses gRPC (`deviceplugin/v1beta1`) over Unix sockets:

```
device plugin                                   kubelet (device manager)
     │  Register(version=v1beta1, endpoint,              │
     │           resource_name=nvidia.com/gpu,           │
     │           options) ──────────────────────────────►│  /var/lib/kubelet/device-plugins/kubelet.sock
     │  GetDevicePluginOptions() answers the same:       │  NVIDIA: GetPreferredAllocationAvailable=true,
     │  which optional calls the kubelet should make     │          PreStartRequired=false
     │◄──────────────────────────── ListAndWatch() ───── │
     │  stream: [{ID, health: Healthy|Unhealthy}] ─────► │  capacity = all IDs, allocatable = healthy IDs
     │                                                   │  → Node.status  → scheduler sees an integer
     │   (pod bound here, container starting)            │
     │◄──── GetPreferredAllocation(available, size) ──── │  only if GetPreferredAllocationAvailable;
     │                                                   │  topology hint: keep GPUs on one NVLink island
     │◄──── Allocate([device IDs]) ───────────────────── │
     │  env / mounts / device nodes / CDI devices ─────► │
     │◄──── PreStartContainer([device IDs]) ──────────── │  only if PreStartRequired (reset or initialise
     │                                                   │  a device); NVIDIA's plugin does not ask for it
     │                                                   │  container created with those devices
```

The `DevicePlugin` service has five calls: `GetDevicePluginOptions`, `ListAndWatch`, `GetPreferredAllocation`,
`Allocate`, `PreStartContainer`. The kubelet makes the two optional calls only when the options that the plugin
sends with `Register` ask for them. `gpusched.deviceplugin` models the contract: `DevicePlugin.list_and_watch()`,
`Kubelet.node_status()` and `Kubelet.admit()`. In `Kubelet.node_status()`, capacity counts every device and
allocatable counts only the healthy devices, as the `GetCapacity` function of the kubelet does.

With the default `envvar` strategy of the NVIDIA plugin, `Allocate` answers with
`NVIDIA_VISIBLE_DEVICES=<device UUIDs>`. Then, at the creation of the container, the NVIDIA Container Toolkit
injects device nodes and driver libraries. The container half is layer 02
([`02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md`](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §6 *How a container gets a GPU*).

Two consequences are important to say in a review:

* **A failing GPU does not evict anything.** A device can become Unhealthy (an XID error, a GPU that is no
  longer on the bus). Then allocatable decreases from 8 to 7 (`Kubelet.node_status()`). Pods that run keep their
  devices, and only new placements see the change. A node drain is an operational decision.
* **The scheduler and the kubelet keep separate books.** The scheduler binds a pod with the last node status
  that it saw. The kubelet admits the pod against the devices that it actually has. If a device failed between
  these two events, the pod fails with `UnexpectedAdmissionError`. The message is `Allocate failed due to requested number of devices unavailable for nvidia.com/gpu. Requested: 2, Available: 1, which is unexpected`
  (`Kubelet.admit()`, with the words of the kubelet). Then the controller of the pod must create it again.

### 1.3 Labels: which GPU, and where

The resource name tells nothing about the model of the GPU. Placement by GPU type uses **node labels**:

| Source | Example labels |
|---|---|
| GKE | `cloud.google.com/gke-accelerator=nvidia-l4` (cluster-autoscaler also uses this label as the key for GPU nodes), `cloud.google.com/gke-nodepool`, Spot and MIG labels (verify names in GKE docs) |
| NVIDIA GPU Feature Discovery (in the GPU Operator) | `nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3`, `nvidia.com/gpu.memory` (MiB), `nvidia.com/gpu.count`, `nvidia.com/gpu.family`, `nvidia.com/cuda.driver-version.full`, `nvidia.com/mig.strategy`, `nvidia.com/gpu.replicas`, `nvidia.com/gpu.clique` (multi-node NVLink domain) |
| Node Feature Discovery | PCI vendor labels, for example `feature.node.kubernetes.io/pci-10de.present=true`. 10de is the PCI vendor ID of NVIDIA. The exact form depends on the NFD config (verify). |
| GCE topology | `cloud.google.com/gce-topology-block`, `-subblock`, `-host` (verify). See section 5. |

A pod selects a type with a `nodeSelector` (or node affinity). Section 3 shows the view of the scheduler.

### 1.4 Taints: keeping the wrong pods off

GPU nodes have a **taint**. GKE uses `nvidia.com/gpu=present:NoSchedule` (verify). Thus ordinary pods do not go
onto the nodes with the highest cost in the cluster. The toleration for this taint comes from the
**ExtendedResourceToleration** admission plugin. The plugin adds
`{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}` to every pod that *requests* `nvidia.com/gpu`
(`gpusched.cluster.extended_resource_toleration()`, which `gpu_pod()` applies).

The plugin is **off by default** in kube-apiserver. GKE turns it on (verify). kubeadm, kind and most
self-managed clusters need `--enable-admission-plugins=...,ExtendedResourceToleration`. Without the plugin, every
GPU pod must carry the toleration itself. If a pod does not carry it, the pod stays Pending on
`untolerated taint(s)`.

The upstream rule for a match is short, and you can learn it exactly (`Toleration.tolerates()`):

- If the toleration names an effect, the effect must match.
- If the toleration names a key, the key must match. An empty key with `Exists` tolerates everything.
- Then `Exists` matches any value, and `Equal` needs the same value.

`NoSchedule` and `NoExecute` taints filter nodes. `PreferNoSchedule` only decreases the score.

### 1.5 Dynamic Resource Allocation (DRA)

A counter cannot say "an H100 with at least 40 GiB free, on the same PCIe root as this NIC". **DRA** is GA in
Kubernetes 1.34 (`resource.k8s.io/v1`). It replaces the counter with structured objects:

| Object | Written by | Holds |
|---|---|---|
| `ResourceSlice` | the DRA driver, per node or pool | devices with **attributes** (model, index, UUID, MIG profile…) and **capacity** (memory…) |
| `DeviceClass` | the cluster admin | a named selection, for example `gpu.nvidia.com` = `device.driver == 'gpu.nvidia.com' && device.attributes['gpu.nvidia.com'].type == 'gpu'` |
| `ResourceClaimTemplate` / `ResourceClaim` | the workload | `devices.requests[]`: `exactly` {`deviceClassName`, CEL `selectors`, `allocationMode`, `count`}, or `firstAvailable` alternatives. Also `constraints[].matchAttribute` (for example, all devices on one PCIe root). |

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

The pod lists the claim in `spec.resourceClaims`. Each container refers to the claim in `resources.claims`. The
`DynamicResources` plugin of the scheduler allocates specific devices during the scheduling cycle. Thus the plugin
can make decisions from attributes and from how pods share devices, not from counts.

The NVIDIA DRA driver (`kubernetes-sigs/dra-driver-nvidia-gpu`) officially supports **ComputeDomains**. A
ComputeDomain is a multi-node NVLink (IMEX) domain for GB200/GB300-class racks. But in September 2026, the
maintainers still marked the GPU-allocation plugin of the driver as not officially supported. That plugin was also off by
default (verify before you rely on it).

Device plugins and DRA exist together. Most GPU fleets still schedule `nvidia.com/gpu` counts today. The DRA
support of Kueue is near beta.

---

## 2. GPU Operator vs managed drivers

Before any part of section 1 works, a GPU node needs these components:

- a kernel driver,
- the NVIDIA Container Toolkit, connected to the container runtime,
- the device plugin,
- ideally, also feature discovery and metrics.

There are two ways to get them:

| | NVIDIA GPU Operator | Managed by the platform (GKE) |
|---|---|---|
| What installs the driver | a driver container per node (or a pre-installed host driver) | GKE's driver installer. It is automatic on control planes ≥ 1.32.2-gke.1297000. The version is `DEFAULT`, `LATEST` (COS only) or `INSTALLATION_DISABLED` per node pool. |
| Device plugin | NVIDIA device plugin | Google's GPU device plugin (open source: `GoogleCloudPlatform/container-engine-accelerators`) |
| Also brings | NFD, GPU Feature Discovery labels, DCGM + dcgm-exporter, MIG manager, node status exporter, validator. The `ClusterPolicy` CRD configures them, plus the `NVIDIADriver` CRD for per-pool drivers. | time-sharing, MPS and MIG as node-pool settings, and GPU metrics in Cloud Monitoring (layer 02 §8–9) |
| You own | versions and upgrades of every component, and compatibility with the node OS and kernel | the selection of the driver channel. You get less choice and fewer components. |
| Typical home | on-prem, self-managed clusters, other clouds, GKE with installation disabled | GKE Standard and Autopilot |

The choice is important for more than convenience. The driver version sets a limit on which CUDA userlands in
your images work (layer 02 §1). A driver upgrade is a node drain. Also, a GPU node is `Ready` before its GPUs are
allocatable, because the driver and plugin start after the kubelet. Thus the cluster autoscaler treats GPU nodes
without allocatable GPUs as `resourceUnready`, not as usable (section 7). Also, a node that stays in that state
usually means a failure of the driver install.

---

## 3. The scheduling cycle

### 3.1 One pod at a time

```
 scheduling cycle, one pod at a time
 queue ─► PreEnqueue  ─► QueueSort  ─► PreFilter ─► Filter     ─► PostFilter  ─► PreScore ─► Score        ─► NormalizeScore  ─► Reserve    ─► Permit
          (scheduling    (priority,                 (per node:    (no node?                  (per plugin,    (to 0-100; then    (claim        (can hold a pod:
          gates)         then age)                  pass/fail)    preemption)                per node)       × weight, sum)     resources)    gangs, §4)

 binding cycle, off the main loop; a failure from Reserve on runs Unreserve and requeues the pod
   ─► PreBind   ─► Bind      ─► PostBind
      (volumes,    (sets        (informational,
      DRA)         nodeName)    cleanup)
```

kube-scheduler takes the pending pod with the highest priority (then the oldest). It runs the **filter** plugins
against every node. It **scores** the nodes that pass, and it binds the pod to the best node. Then it does the
same for the next pod. It never examines two pods together, never looks ahead, and never moves a pod that runs.
These three facts explain fragmentation (3.4), the limits of preemption (3.5) and gang deadlock (4.1).

The table gives the default plugin set and score weights (upstream `getDefaultPlugins`, release-1.34). The
scheduler adds DynamicResources when the DynamicResourceAllocation feature is on. That is the default since DRA
became GA in 1.34. DynamicResources has no weight, because it implements no Score:

| Plugin | Filter | Score weight | Relevance to GPUs |
|---|---|---|---|
| NodeUnschedulable | cordoned nodes | – | drains |
| TaintToleration | untolerated NoSchedule/NoExecute taints | 3 | keeps CPU pods off GPU nodes |
| NodeAffinity | nodeSelector / required affinity | 2 | GPU model selection |
| NodeResourcesFit | requests ≤ allocatable − requested | 1 | **the only place that counts GPUs** |
| PodTopologySpread, InterPodAffinity | spread / affinity rules | 2, 2 | spread of replicas |
| DynamicResources | DRA claims allocatable (also PostFilter, Reserve, PreBind) | – (no Score) | section 1.5 |
| NodeResourcesBalancedAllocation, ImageLocality | – | 1, 1 | pull toward spread pods and toward cached images |
| DefaultPreemption | PostFilter | – | section 3.5 |

### 3.2 Filter

Each node passes, or fails with the reason of the **first** plugin that rejects it. The order is unschedulable,
taints, node affinity, resources (`gpusched.plugins.run_filters()`). When every node fails, the
`FailedScheduling` event of the pod is a histogram of those reasons, sorted as strings.
`gpusched.scheduler.fit_error()` gives the same format as upstream `FitError`. Here is the event for an 8-GPU
pod that selects H100s (notebook 02). The cluster has one H100 node with 4 GPUs busy, one cordoned H100 node, one
L4 node and one CPU node:

```
0/4 nodes are available: 1 Insufficient nvidia.com/gpu, 1 node(s) were unschedulable,
2 node(s) didn't match Pod's node affinity/selector.
```

Read the event one reason at a time. *Insufficient* is capacity (wait, preempt, scale up, defragment).
*affinity/selector* and *untolerated taint* mean that the pod asked for a node shape that is not here. The CPU node
reports the selector, not the GPU that it does not have, because the selector filter runs first. Lab notebook
`03_why_is_my_pod_pending` makes this into a diagnosis tool.

### 3.3 Score: spread or pack

`NodeResourcesFit` scores with one of two strategies over a configured list of `(resource, weight)`. It counts the
new pod as already placed, and it uses integer arithmetic (MaxNodeScore = 100):

$$
\begin{aligned}
\text{LeastAllocated} &= \left\lfloor \frac{\sum_r w_r \cdot \lfloor (\mathit{alloc}_r - \mathit{requested}_r) \cdot 100 / \mathit{alloc}_r \rfloor}{\sum_r w_r} \right\rfloor && \text{(spread; the default)} \\
\text{MostAllocated} &= \left\lfloor \frac{\sum_r w_r \cdot \lfloor \min(\mathit{requested}_r, \mathit{alloc}_r) \cdot 100 / \mathit{alloc}_r \rfloor}{\sum_r w_r} \right\rfloor && \text{(bin-pack)}
\end{aligned}
$$

The example is an 8-GPU node with 6 GPUs requested. A 1-GPU pod, with the score on the GPU alone, gets
`MostAllocated` = ⌊7 · 100 / 8⌋ = **87**. It gets `LeastAllocated` = ⌊1 · 100 / 8⌋ = **12**
(`plugins.most_allocated()`, `plugins.least_allocated()`). Two details decide the GPU behaviour:

* **The default scoring resources are `cpu` and `memory` (weight 1 each).** The default score does not include
  GPUs at all. Two nodes with the same CPU and memory use get the same score, with 1 or with 7 GPUs busy. GPU
  pods spread as a side effect of the CPU/memory spread.
* **An extended resource the pod does not request is skipped**. It does not get a score of zero. Thus
  `nvidia.com/gpu` in the list does not keep CPU pods away from GPU nodes. Taints do that.

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

On a managed control plane, you cannot edit the default scheduler. GKE's `optimize-utilization` autoscaling
profile moves the scheduler toward a bin-pack strategy (verify). Or you run a second scheduler with this profile
and select it with `schedulerName`.

A packed layout has costs, and it is good to name them. Replicas of one service collect on few nodes (keep them
apart with `topologySpreadConstraints`). Other default scorers (BalancedAllocation, default PodTopologySpread for
Deployments) still pull toward a spread layout.

### 3.4 Fragmentation, measured

Sixteen 1-GPU inference pods (8 cores, 64 GiB each) arrive at four empty 8-GPU nodes. Then one 8-GPU training pod
arrives (`scheduler.Scheduler`, notebook 02):

```
default (LeastAllocated, cpu+memory)          MostAllocated, GPU weight 5
h0  ####....  4/8                             h0  ########  8/8
h1  ####....  4/8                             h1  ########  8/8
h2  ####....  4/8                             h2  ........  0/8
h3  ####....  4/8                             h3  ........  0/8
8-GPU pod: Pending                            8-GPU pod: bound to h2
```

On the left, sixteen GPUs are free, and an 8-GPU pod can use none of them. A useful capacity metric is **stranded
GPUs for a pod shape**. These are the free GPUs that cannot hold one more pod of that shape. Also,
**fragmentation** = stranded / free.

A pod shape is a bundle: $k$ GPUs *plus* CPU and memory. Thus, in general, node $n$ can take
$\mathit{fits}_n = \min_{\text{resources } r} \lfloor \mathit{free}_{n,r} / \mathit{request}_r \rfloor$ more such
pods. Its usable GPUs are $k \cdot \mathit{fits}_n$, and $\text{stranded} = \sum_{\text{nodes}} (\text{free GPUs}_n - k \cdot \mathit{fits}_n)$.
GPUs become stranded in two ways:

* **GPU-count fragmentation**: only the GPUs set the limit, and stranded is
  $\sum_{\text{nodes}} (\mathit{free}_n \bmod k)$. `scheduler.stranded_gpus()` and `scheduler.fragmentation()`
  calculate this term. The values are 16 and 100% on the left, and 0 and 0% on the right.
* **CPU/memory-bundle stranding**: CPU or memory runs out first. These nodes have 96 cores and 768 GiB. 1-GPU
  pods that request 16 cores and 64 GiB fit min(8 GPUs / 1, 96 cores / 16, 768 GiB / 64 GiB) = **6** per node. The
  scheduler binds six, and the seventh is Pending on `Insufficient cpu`. Thus 2 GPUs are stranded, although
  $8 \bmod 1 = 0$ (`scheduler.Scheduler`).

  Keep a 1-GPU pod within ${1/N}$ of the allocatable CPU and memory of an $N$-GPU node, minus DaemonSets and
  sidecars. The lab's `k8sgpu.machines.stranded_gpus()` does this for real machine shapes, after GKE's
  reservations.

Monitor the stranded GPUs for each pod shape that is important to you, next to utilisation. A cluster can be 50%
utilised and 100% fragmented for its largest jobs. The shapes come from the size of the model: the number of GPUs
that one replica needs for weights and KV cache
([capacity-planning primer](../../00-foundations/gpu-capacity-planning/PRIMER.md)). These are the other levers:

- separate node pools per pod shape,
- pod shapes that divide the node (a 5-GPU pod leaves 3 GPUs stranded on every 8-GPU node),
- CPU and memory requests sized to the per-GPU share of the node,
- defragmentation, which schedules pods again (the descheduler, or Kueue preemption).

The scheduler itself never moves a pod that runs.

### 3.5 Priority and preemption

The priority of a pod comes from a `PriorityClass` (`scheduling.k8s.io/v1`, with
`preemptionPolicy: PreemptLowerPriority` or `Never`). When no node passes the filters, the `DefaultPreemption`
PostFilter looks for nodes where the eviction of **lower-priority** pods can make room
(`plugins.select_victims()`). On each node, it does these steps:

1. It removes all of those pods.
2. It does a check that the preemptor fits.
3. It adds the pods back, **most important first** (higher priority, then earlier start).
4. It evicts only the pods that do not fit back.

To select from the candidate nodes (`plugins.pick_preemption_node()`), it compares these values, in this order:

1. the fewest PodDisruptionBudget violations,
2. the lowest priority of the highest-priority victim,
3. the lowest sum of victim priorities (each priority has an offset of 2³¹, so fewer victims wins first),
4. the fewest victims,
5. the latest start time of the highest-priority victims.

The preemptor gets `nominatedNodeName`, and it binds after the graceful termination of the victims.

Worked example (notebook 02): node *a* runs two priority-0 batch pods of 4 GPUs. Node *b* runs one priority-500
evaluation pod of 8 GPUs. A priority-1000 serving pod needs 4 GPUs. Node *a* costs one priority-0 victim, and node
*b* costs one priority-500 victim. Thus the scheduler evicts the newer batch pod on *a*.

Preemption does **not** do these things: evict pods of equal or higher priority, consolidate free GPUs, or
understand jobs. Preemption can make room for the first pod of an 8-pod job. But this does nothing useful if the
other seven pods cannot follow.

---

## 4. Gangs

### 4.1 Why partial placement deadlocks

A distributed training job or a multi-host inference replica is a **gang**. Its workers block at the first
collective (`init_process_group`, the first all-reduce) until every rank has joined. Thus partial placement is
only waste, and with two gangs it is a deadlock. In the example, there are three nodes with 4 GPUs each. Jobs A
and B each need 4 workers × 2 GPUs. Their controllers create pods at the same time (`gang.interleave()`,
notebook 03):

```
a0 b0 a1 b1 a2 b2 a3 b3  (arrival)       h0: a0 b1   h1: b0 a2   h2: a1 b2      a3, b3: Pending
                                         12/12 GPUs allocated, 0 jobs running, forever
```

Neither job can complete its set, and neither job will release what it holds. The priorities are equal, so
preemption does not break the tie. If you admit each job **whole**, A runs on 8 GPUs while B waits and holds
nothing (`gang.admit_gangs()`). Then 4 GPUs are idle instead of 12, and B starts when A is complete.

### 4.2 Ways to get all-or-nothing

| Mechanism | How | Notes |
|---|---|---|
| **Kueue** (job level) | The webhook of Kueue suspends queued Jobs at creation (`spec.suspend: true`). Plain pods get the scheduling gate `kueue.x-k8s.io/admission`. Kueue admits the whole Workload against quota. Then it unsuspends the Workload and injects the node selectors of the flavor. | Quota is *logical*. It is possible that the pods of an admitted job do not all fit on real nodes (fragmentation, pods that Kueue does not manage). `waitForPodsReady` evicts and requeues a job whose pods are not all Ready, with backoff. It is on by default since the v1beta2 Configuration: timeout 30 min, `recoveryTimeout` the same, `blockAdmission: false`. Only the alpha `DisableWaitForPodsReady` feature gate turns it off. `blockAdmission: true` admits one workload at a time. TAS (section 5) and ProvisioningRequest (section 7) do a check of physical capacity. |
| **Coscheduling** plugin (kubernetes-sigs/scheduler-plugins) | A `PodGroup` with `minMember`. The **Permit** stage holds reserved pods until the number of reserved pods is `minMember`. If that does not occur, it rejects them after a timeout. | It runs inside a second scheduler profile. The PodGroup API is `scheduling.x-k8s.io/v1alpha1` (verify). |
| **Volcano** | its own batch scheduler with `PodGroup.minAvailable`, queues and fair share | a replacement scheduler, common in HPC-style clusters |
| **Kubernetes native** (KEP-4671) | `Workload` and `PodGroup` APIs in `scheduling.k8s.io`. The scheduler places a pod group together. | It is alpha in 1.35 behind the `GenericWorkload` feature gate, and beta in 1.37. The KEP metadata gives 1.38 as the target for stable (verify the release you run). Kueue plans to integrate it. |

The pattern that works today, on GKE and elsewhere, is Kueue in front of the default scheduler. Jobs wait in the
queue as whole units. TAS or a ProvisioningRequest makes sure that the nodes exist and fit. `waitForPodsReady`
catches the rest.

### 4.3 JobSet and LeaderWorkerSet

A gang also needs a workload API that treats its pods as one thing:

* **JobSet** (`jobset.x-k8s.io/v1alpha2`) is a group of Jobs for training and HPC. It has `replicatedJobs` (for
  example, one driver and $N$ workers) and a headless Service for stable hostnames. It has
  `failurePolicy.maxRestarts` (a failed Job creates the whole set again, so resume from a checkpoint). It also has
  `successPolicy`, `startupPolicy.startupPolicyOrder: InOrder`, and
  `alpha.jobset.sigs.k8s.io/exclusive-topology`, which gives each child Job a whole topology domain. Kueue admits a
  JobSet as one Workload.
* **LeaderWorkerSet** (`leaderworkerset.x-k8s.io/v1`) has replicas that are **groups** of pods, for multi-host
  inference. Each group has `leaderWorkerTemplate.size` pods: one leader, size−1 workers, and an optional separate
  `leaderTemplate`. With `restartPolicy: RecreateGroupOnPodRestart`, a failed shard restarts its group. It also
  has `startupPolicy: LeaderCreated | LeaderReady`, the environment variables `LWS_LEADER_ADDRESS`,
  `LWS_GROUP_SIZE` and `LWS_WORKER_INDEX`, and `leaderworkerset.sigs.k8s.io/exclusive-topology` to keep a group
  in one domain. Its scale subresource lets an HPA scale *groups*
  ([`05-orchestrator/serving-orchestration/PRIMER.md`](../../05-orchestrator/serving-orchestration/PRIMER.md)
  §4 *Autoscaling*). A related API, **DisaggregatedSet**, coordinates prefill and decode LWSs (same primer, §5).

---

## 5. Topology-aware placement

### 5.1 The tree

"All pods running" is not sufficient. A gang runs at the speed of the slowest link that its collectives cross.

```
block ──────────── spine links (oversubscribed)
 ├─ sub-block ──── hosts on the same leaf switches: full bandwidth inside
 │   ├─ host ───── NVLink / NVSwitch between its 8 GPUs (hundreds of GB/s per GPU)
 │   │   └─ GPU
 │   └─ host
 └─ sub-block
```

Layer 01 gives the bandwidth of each level, and the cost of a tensor-parallel all-reduce across it
([roofline-and-fabric primer](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md), §5.3 and §5.4). Here,
the design rule has two parts. Keep tensor-parallel and expert-parallel groups inside the NVLink domain. Keep every
gang in the **smallest** domain that holds it.

Inside one node, the same rule applies to devices. The `GetPreferredAllocation` call of the device plugin keeps a
container on one NVLink island (`DevicePlugin.get_preferred_allocation()`). The Topology Manager of the kubelet can
align CPUs and devices by NUMA node.

### 5.2 Kueue Topology-Aware Scheduling

Kueue TAS (beta, on by default since v0.14) models the tree with node labels. An admin creates a `Topology`
(`kueue.x-k8s.io/v1beta2`) that lists the levels, the largest first. On GCE, the levels are the
`cloud.google.com/gce-topology-block`, `-subblock` and `-host` labels, then `kubernetes.io/hostname`. A3/A4/A4X
nodes carry these GCE labels, and G2/L4 nodes do not (verify, §10.3). The admin then points a ResourceFlavor at the
Topology with `spec.topologyName`. Workloads ask for a placement with pod-template annotations:

| Annotation | Meaning |
|---|---|
| `kueue.x-k8s.io/podset-required-topology: <level label>` | all pods in one domain of that level, or wait |
| `kueue.x-k8s.io/podset-preferred-topology: <level label>` | try that level, then each level up, then spread |
| `kueue.x-k8s.io/podset-unconstrained-topology: "true"` | anywhere, but TAS still counts the capacity accurately |
| `kueue.x-k8s.io/podset-slice-required-topology` + `podset-slice-size` | every slice of $N$ pods inside one domain (for example, each 4-host slice in one sub-block) |

TAS calculates the free capacity per domain. It starts from the allocatable of Ready, schedulable nodes, and
subtracts TAS workloads and all other pods. Then it assigns pods to domains before the job starts. Thus TAS is
also a check of physical capacity, and plain quota is not. There are two algorithms (defaults since v0.15), both
in `gpusched.gang`:

* **BestFit** (required and preferred): from the domains that can hold the whole pod set, it takes the
  **tightest**. When it must divide the pod set across child domains, it takes the child domains with the most
  room first. For the last child domain, it selects the tightest domain that holds the remainder
  (`gang.best_fit()`). The design's own example is a rack whose nodes have room for 3, 3, 2 and 1 pods, with 7
  pods to place. BestFit gives 3 + 3 + **1**, and keeps the 2-slot node whole.

  Below the selected domain, Kueue repeats the selection level by level, over the children of *all* selected
  domains together. Thus the host split does not have to follow the sub-block split.
* **LeastFreeCapacity** (unconstrained): one flat list of hosts, tightest first. It takes the tightest single host
  that holds the whole pod set. Only if no host does, it fills the smallest gaps first: 1 + 2 + 3 + 1 for the same
  example (`gang.least_free_capacity()`). This keeps large domains whole for constrained jobs.

Worked placement (`gang.place_gang()`, notebook 03). The free space per sub-block, in 8-GPU pods, is b0-s0 = 1,
b0-s1 = 3, b1-s0 = 2 and b1-s1 = 4. In b1-s0, one of the hosts has a single GPU busy, so that host counts zero.

| Gang of 8-GPU pods | Constraint | Placement |
|---|---|---|
| 3 | required sub-block | b0-s1, exactly 3. TAS keeps b1-s1 (4) for something larger. |
| 2 | required sub-block | b1-s0 |
| 5 | required sub-block | none: waits |
| 5 | preferred sub-block | block b1 (2 + 4 = 6 ≥ 5). The host pass over both sub-blocks gives 2 in b1-s0 + 3 in b1-s1 |
| 3 | unconstrained | LeastFreeCapacity: no host holds 3, so 1 in b0-s0 + 2 in b0-s1 — the gang crosses sub-blocks |

Select the constraint for each workload:

- **required** for a multi-host inference group whose shards exchange activations every token (a slow replica is
  slow for its whole life),
- **preferred** for long training jobs (a wait forever for a perfect block is worse than a block that is slightly
  slower),
- **unconstrained** for embarrassingly parallel batch.

When it can, TAS also replaces a failed node inside the assigned domain (hot swap). When it cannot, TAS evicts the
workload.

### 5.3 Compact placement and exclusive topology

Two related tools work at other layers. **Compact placement** asks the cloud to create the VMs of a node pool
physically near each other (on GKE, `placement_policy.type = "COMPACT"` on the node pool). This is topology at
creation time, not at scheduling time. **Exclusive topology** (JobSet and LWS annotations) gives each job or group
a whole domain, so neighbours cannot share its links.

---

## 6. Queues, quotas and multi-tenancy with Kueue

### 6.1 The objects

The scheduler decides where a pod runs. **Kueue** (`kueue.x-k8s.io/v1beta2`, release v0.19.6 in September 2026)
decides *if and when* a job can start, and which flavor of capacity it gets:

| Object | Scope | Purpose |
|---|---|---|
| `ResourceFlavor` | cluster | a kind of capacity: node labels, taints/tolerations, optional `topologyName` (for example `h100-spot`, `h100-reserved`, `l4`) |
| `ClusterQueue` | cluster | quota per flavor and resource (`resourceGroups[].flavors[].resources[]`: `nominalQuota`, `borrowingLimit`, `lendingLimit`), `cohortName`, `preemption`, `queueingStrategy`, `admissionChecksStrategy`, `fairSharing` |
| `LocalQueue` | namespace | the entry point of a team. It points at one ClusterQueue. |
| `Workload` | namespace | Kueue's view of one job: pod sets × requests, priority, admission status |
| `WorkloadPriorityClass` | cluster | priority for the queue and for preemption (label `kueue.x-k8s.io/priority-class`), independent of pod priority |
| `AdmissionCheck`, `ProvisioningRequestConfig` | cluster | more gates before admission, for example a ProvisioningRequest for capacity (section 7) |
| `Topology`, `Cohort` | cluster | TAS levels (section 5), hierarchical cohorts |

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

ClusterQueues with the same `cohortName` lend each other **unused** nominal quota. Here is the rule of Kueue for
how much of a (flavor, resource) a ClusterQueue can use now (`pkg/cache/scheduler/resource_node.go`, flat cohort,
`quota.Kueue.available()`):

$$
\begin{aligned}
\text{guaranteed} &= \text{nominal} - \texttt{lendingLimit} \qquad \text{(0 if no }\texttt{lendingLimit}\text{; never lent out)} \\
\text{cohort pool} &= \textstyle\sum_{\text{members}} (\text{nominal} - \text{guaranteed}) \\
\text{pool in use} &= \textstyle\sum_{\text{members}} \max(0, \text{usage} - \text{guaranteed}) \\
\text{from cohort} &= \text{pool} - \text{pool in use}, \text{ capped at} \\
&\qquad (\text{nominal} - \text{guaranteed}) - \max(0, \text{usage} - \text{guaranteed}) + \texttt{borrowingLimit} \\
\text{available} &= \max(0, \text{guaranteed} - \text{usage}) + \text{from cohort}
\end{aligned}
$$

The examples in the Kueue documentation come out exactly. Team A (9 CPUs) and team B (12) are in one cohort, and
both are idle. Then A can use **21**. With A's `borrowingLimit: 1`, A can use **10**. With B's `lendingLimit: 1`
instead, A can use **10** (the documented 9 + 1 in both cases).

With GPUs (notebook 04), there are two ClusterQueues of 16 GPUs each. Team B runs three 8-GPU jobs, and the third
job **borrows** 8 of the idle GPUs of team A. Team A can now use 8, not 16.

### 6.3 Preemption: nominal quota is only a guarantee if you reclaim

Team A submits a 16-GPU job. With the default `reclaimWithinCohort: Never`, nothing occurs. The job waits
(`couldn't assign flavors to pod set main: insufficient unused quota for nvidia.com/gpu in flavor h100, 8 more needed`,
from `Kueue.explain()`) until the job of B that borrows quota ends on its own. **Nominal quota is not a guarantee
unless reclaim is on** (or a `lendingLimit` keeps part of it home). These are the policies:

| Field | Values | Effect |
|---|---|---|
| `reclaimWithinCohort` | Never, LowerPriority, Any | preempt workloads of ClusterQueues in the cohort that **borrow**, to get lent quota back |
| `withinClusterQueue` | Never, LowerPriority, LowerOrNewerEqualPriority | preempt lower-priority (or equal-priority newer) workloads in the same queue |
| `borrowWithinCohort` | policy Never / LowerPriority, `maxPriorityThreshold` | preempt while the preemptor itself borrows (classic preemption only) |

Classic preemption (`quota.Kueue.preemption_targets()`) can occur when the preemptor fits within its nominal
quota, or when `borrowWithinCohort` is on. Kueue puts the candidates in this order: **other ClusterQueues first,
then lowest priority, then most recently admitted**. It removes them greedily until the preemptor fits. It removes
a candidate from another queue only while that queue still borrows. Then it decreases the set to the minimum, in
reverse order.

With `reclaimWithinCohort: Any`, the 16-GPU job of team A preempts exactly the third job of B (the newest, which
borrows), and Kueue admits it. B is back at its nominal 16. With `withinClusterQueue: LowerPriority`, a
priority-100 job in a full queue evicts the newest priority-0 job.

**Fair sharing** is the alternative algorithm. Each ClusterQueue gets a weighted share value of the borrowable
resources. Admission selects the lowest share first, and preemption takes from the highest share first. This
occurs under `preemptionStrategies` such as `[LessThanOrEqualToFinalShare, LessThanInitialShare]`. Admission Fair
Sharing applies the idea to LocalQueues within a ClusterQueue.

### 6.4 Queueing strategy and flavors

`StrictFIFO` admits in order (priority, then creation), and a head that does not fit blocks the queue.
`BestEffortFIFO` (the default) lets later workloads go past a blocked workload. Take a 16-GPU queue that receives
8, 16 and 8 GPU jobs. StrictFIFO admits only the first job. BestEffortFIFO admits both 8-GPU jobs. This gives
better utilisation, but large jobs can wait indefinitely behind a stream of small jobs.

Kueue tries the flavors in the listed order. By default, it takes a flavor that fits, even if it must borrow
(`whenCanBorrow: MayStopSearch`). It selects a flavor that needs preemption only if no later flavor fits
(`whenCanPreempt: TryNextFlavor`). Thus, if you list `h100-reserved` before `h100-spot`, you express a cost
preference.

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

Serving keeps 8 GPUs at home and lends the other 8. When serving scales up, it gets them back at once. Batch
(`borrowingLimit: 8`, `withinClusterQueue: LowerPriority`) fills idle quota, and puts its own jobs in order by
priority. The same idea is the admission control of the gateway, one layer up
([agentic scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.3).
It admits whole units of work against a budget, and it sheds the rest or puts the rest in a queue.

---

## 7. Getting capacity

### 7.1 The cluster autoscaler for GPU pools

A Pending GPU pod is a request for a machine. The cluster autoscaler does these steps at every scan interval
(10 s):

```
pending pods ─► simulate them on each node pool's template node ─► bin-pack: nodes needed per pool
            ─► expander picks a pool ─► cloud creates VMs ─► driver + device plugin ─► GPUs allocatable ─► scheduler binds
```

* **How many nodes**: a bin-pack estimate of the pending pods onto empty template nodes
  (`autoscaler.nodes_needed()`, first-fit decreasing). Pods of 4, 4, 2, 2, 1, 1, 1, 1 GPUs need **2** 8-GPU
  nodes. Three 5-GPU pods need **3** nodes, and leave 3 GPUs stranded on each.
* **Which pool**: the expander. The expanders are `least-waste` (the default: least idle CPU, then memory),
  `random`, `most-pods`, `least-nodes`, `price`, `priority` and `grpc`, and you can chain them. Least-waste
  compares idle resources, not dollars. The version in the simulator (`autoscaler.least_waste()`) ranks idle GPUs.
  The request in the example is 4 + 2 + 1 + 1 GPUs. The simulator finds that one 8-GPU H100 node and two 4-GPU
  L4 nodes have the same waste (zero idle GPUs).

  It takes the H100 on its tie-break. Thus pods select one GPU type with a node selector. A `price` or
  `priority` expander, or a ComputeClass (7.3), expresses the cost.
* **Down again**: a node goes away after it stays unneeded for `--scale-down-unneeded-time` (10 min), and no
  scale-up occurred for `--scale-down-delay-after-add` (10 min). For GPU nodes, only GPU utilisation counts
  against `--scale-down-gpu-utilization-threshold` (0.5). The autoscaler stops its wait for nodes that do not
  register within `--max-node-provision-time` (15 min).

Scale from zero, simulated (`autoscaler.simulate()`, notebook 05, which models both 10-minute delays): a 1-hour job
goes onto an empty 1-GPU pool. The nodes of the pool take 300 s to show allocatable GPUs. The job starts at 300 s
and ends at 3,900 s, and the autoscaler removes its node at 4,500 s. The only scale-up was at $t$ = 0, so the
unneeded time sets the limit. The result is **1.25 node-hours billed for 1 hour of work**, before any image or
weights.

### 7.2 Obtainability

GPUs are the resource that a cloud sometimes does not have. The table gives the capacity types, and how each type
fails:

| Type | You get | How it fails | Fits |
|---|---|---|---|
| On-demand | a VM now, if the zone has it | stockout: the scale-up gives an error and retries. Large parts are hard to get. | stateless serving, dev |
| Spot | a cost that is 60–91% lower (GCP, verify) | the cloud can reclaim it at any time, with short notice | interruptible, retryable batch, and extra serving replicas |
| Reservation | capacity that the cloud holds for you, billed if you use it or not | you pay for idle capacity | steady baseline load |
| DWS flex-start (GCP) | a queued request that the cloud provisions **all at once**, for up to 7 days | waits until capacity exists | gangs with a flexible start: training, fine-tuning, large evals |
| DWS calendar mode (GCP) | a future block with a specified duration (verify) | you must plan ahead | planned runs |

On GCP, two account facts are gates for all of this. GPUs are not usable on a Free Trial billing account. GPU
quota often starts at zero. Layer 01 §10.1 and [`COMPUTE.md`](../../COMPUTE.md) give the machine families and
prices.

**Spot and gangs.** If reclaims are independent, at rate $\lambda$ per node-hour, a gang of $N$ nodes survives
$T$ hours with probability $e^{-N \cdot \lambda \cdot T}$ (`autoscaler.gang_survival()`). If every reclaim restarts
the gang from the start, $T$ hours of work with restart overhead $R$ take
$(e^{N \cdot \lambda \cdot T} - 1) \cdot (1/(N \cdot \lambda) + R)$ hours on average
(`autoscaler.expected_runtime_h()`). The tests check this formula by Monte Carlo, with and without $R$. The table
uses an illustrative $\lambda$ = 0.005 per node-hour, and a 24-hour job with $R$ = 15 min:

| Nodes | Survives 24 h | Expected wall-clock |
|---:|---:|---:|
| 1 | 88.7% | 25.5 h |
| 4 | 61.9% | 31.0 h |
| 16 | 14.7% | 74.2 h |

The size of the gang multiplies the hazard. Write checkpoints at the interval that layer 01 §7.2 calculates
(Young/Daly). Or put large gangs on reserved or flex-start capacity.

### 7.3 Queued provisioning, ComputeClass, Autopilot

A gang has a problem that an ordinary scale-up cannot solve. Nodes arrive one at a time, and the cloud bills every
node that arrives before the last node while that node waits. It is worse when the `max_nodes` of a pool is below
the size of the gang. Then the pool scales up *part* of the gang, and those nodes stay idle forever. In the
simulation, 3 nodes use 6 node-hours in 2 hours, and the job never starts. **Queued, all-or-nothing provisioning**
holds the request until the cloud can create the whole gang:

* **ProvisioningRequest** (`autoscaling.x-k8s.io/v1`, cluster-autoscaler) has the classes
  `check-capacity.autoscaling.x-k8s.io` and `best-effort-atomic-scale-up.autoscaling.x-k8s.io`. It also has
  provider classes, for example GKE's queued provisioning, `queued-provisioning.gke.io` (verify). Kueue uses it as
  an **admission check** (`ProvisioningRequestConfig`). Kueue reserves the quota and creates the request. Kueue
  admits the job only when the capacity is `Provisioned`.
* **GKE DWS flex-start** node pools with queued provisioning serve those requests atomically. The flags are
  `--flex-start --enable-queued-provisioning`. In Terraform, the fields are `node_config.flex_start` and
  `queued_provisioning.enabled`.

Simulated (`autoscaler.provision()`, notebook 05): 16 nodes of 8 GPUs and a 300 s boot. Each absent node has a 3%
probability per minute that the cloud grants it. In one run (seed 0), the gang can start after 2.47 h with either
pool.

The ordinary pool paid **31.3 node-hours (251 GPU-hours)** for nodes that waited for their peers. The queued pool
paid **1.3** (only the boot, in every run). Also, over 200 seeds the ordinary pool averages **22.4 node-hours** and a
1.95 h start. Queued provisioning does not create capacity. With it, you do not pay for the partial set.

A **custom ComputeClass** (`cloud.google.com/v1`, verify) gives one class of workload an ordered fallback list.
An example is reservation, then Spot, then on-demand, then flex-start. GKE's node auto-provisioning creates node
pools that match, on demand. Pods select the class with a node selector (verify the field names against the CRD).
The lab's `deploy/gke/` has an example.

**Autopilot** removes node pools entirely. A pod requests `nvidia.com/gpu` and selects an accelerator type. Then
GKE provisions and bills per pod (verify selectors and limits). Outside GCP, the same ideas are Karpenter node pools
with capacity types (AWS, Azure), and capacity reservations and blocks for ML. On-prem, the same idea is a fleet of
constant size where quota (section 6) is the only elasticity (verify provider specifics).

---

## 8. Startup latency

Capacity is not useful until a pod on it serves. Layer 01 §6.2 gives the chain from Pending to Ready, and the
bandwidth arithmetic of each hop. This is the Kubernetes side:

```
Pending ─► node (create VM, boot) ─► driver + device plugin ─► image pull ─► weights ─► engine warm-up ─► Ready
            warm pool, min nodes,       GKE auto-install;          image          GCS FUSE,    startup probe,
            balloon pods                resourceUnready until      streaming,     Hyperdisk ML, readiness gate
                                        GPUs are allocatable       secondary      model
                                                                   boot disks     streamers
```

The example puts illustrative inputs through `autoscaler.startup_latency()` (notebook 05). The first case has
these stages:

- a new node (150 s) with driver install (90 s),
- a 12 GB image pulled at 0.25 GB/s,
- 16 GB of weights at 0.5 GB/s,
- 60 s of warm-up.

Together, these stages take **380 s**. The same replica on a warm node has image streaming (2 GB/s effective) and
weights from a fast cache (4 GB/s). It takes **70 s**, most of it warm-up. These are the levers, stage by stage:

* **Node**: keep `min_nodes` above zero for latency-critical pools. Or run **balloon pods**. These are placeholder
  pods that request the GPU shape that you want to keep warm. Put them in a `PriorityClass` that is negative but
  not below the `--expendable-pods-priority-cutoff` of the autoscaler. Its default is −10, and the FAQ uses −10.
  Pods with a priority below the cutoff do not trigger a scale-up and do not block a scale-down.

  Give the balloon pods `terminationGracePeriodSeconds: 0`. Real pods preempt them at once (section 3.5). Then the Pending balloon makes the autoscaler add a node for it.
* **Image**: GKE **image streaming** (`gcfs_config` in Terraform) starts containers before the full image pull is
  complete. **Secondary boot disks** (`secondary_boot_disks`) supply a disk with images or data preloaded (verify
  supported sources and registries).
* **Weights**: mount a bucket with the **Cloud Storage FUSE CSI driver**
  (`addons_config.gcs_fuse_csi_driver_config`) and its caching options. Or mount a read-only-many **Hyperdisk ML**
  volume. Or load the weights into GPU memory with a model streamer (verify current options per engine). Layer 01
  §6.1 has the parallel-read arithmetic.
* **Warm-up**: use a **startup probe** with a sufficient duration for load + CUDA-graph capture. Then the kubelet
  does not stop a pod that loads slowly. Use a readiness probe, so traffic arrives only when the engine serves.
  [`04-inference-engine/serving-engine/PRIMER.md`](../../04-inference-engine/serving-engine/PRIMER.md) §1 tells
  what the engine does at start-up.

A cold start takes minutes. Thus the autoscaling of LLM replicas needs headroom and scale-ahead signals
([`05-orchestrator/serving-orchestration/PRIMER.md`](../../05-orchestrator/serving-orchestration/PRIMER.md) §4.3 *Cold start
anatomy*). For the same reason, "scale to zero" is a cost decision with a latency price.

---

## 9. Sharing GPUs at the cluster level

A GPU is an integer. Thus, to share one GPU, the node advertises *more integers*. Layer 02 gives the mechanics of
each method: MIG profiles and their placement, MPS, and time-slicing latency
([`02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md`](../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md) §7 *Sharing a GPU*).
This is the cluster view:

| Method | What the node advertises | Isolation | Good for |
|---|---|---|---|
| Whole GPU | `nvidia.com/gpu: 8` | full | training, large-model serving |
| **MIG** (A100, H100 and newer) | each partition is a device: `nvidia.com/gpu` (single strategy), or `nvidia.com/mig-1g.10gb` and the other profiles (mixed). GKE uses `gpu_partition_size` per node pool. | hardware. The GPU divides memory, cache and SMs, and a fault stays in its partition. | many small models with predictable size |
| **Time-slicing** | `replicas` × GPUs, for example 8 × 10 = **80** `nvidia.com/gpu` (or `nvidia.com/gpu.shared`) | none: shared memory, and one fault can stop all | dev, notebooks, bursty light inference |
| **MPS** | replicas with per-client memory and compute limits | partial | throughput-oriented small kernels |
| **DRA** | claims that several containers or pods share, and MIG devices as `mig.nvidia.com` (profile attribute) | per driver | the future direction (section 1.5) |

The trap with time-slicing: a container that requests 2 "GPUs" can get two slices of the **same** physical GPU.
The NVIDIA plugin takes replicas from the least-loaded GPUs. Thus, on a busy node, one lightly used GPU supplies
both slices (`DevicePlugin(replicas=10)`, notebook 01, exercise 1.6). That is the default `distributed`
allocation policy of the plugin.

Its `--shared-devices-allocation-policy` flag (v0.20.0 and later, verify) also offers `packed`, which fills the
busiest GPU first. With `packed`, the plugin touches fewer GPUs, and a multi-replica request goes onto one GPU
whenever that GPU has room. The main branch adds `spread`, which gives a request different physical GPUs while
there are sufficient GPUs (`DevicePlugin(allocation_policy=...)`). `spread` is not in a release as of 2026-09-26
(verify).

None of the three gives isolation. Instead, `failRequestsGreaterThanOne` fails such a container at admission
(`DevicePlugin(fail_requests_greater_than_one=True)`). On GKE time-sharing nodes, a container can request at most
one `nvidia.com/gpu`, and the GKE device plugin applies this limit.

The partition count of each MIG profile does not change. An A100 40 GB offers seven `1g.5gb`, three `2g.10gb` or
two `3g.20gb` slices. An H100 80 GB offers seven `1g.10gb` slices. These counts come from GKE's device plugin (verify
for your GPU and driver).

On GKE, the node-level settings to share GPUs are per node pool (`gpu_sharing_config.gpu_sharing_strategy`,
`max_shared_clients_per_gpu`). Thus the decision to share is a *pool* decision. Put shared and exclusive GPUs in
different pools, and let labels route the pods.

---

## 10. Learning locally

Every mechanism in this primer can run on a laptop, except real devices and real provisioning.

### 10.1 kind with fake GPU capacity

A kind cluster (Kubernetes in Docker) runs the real scheduler, real Kueue, JobSet and LWS. You can **advertise
extended resources by hand**. A node-status patch sets capacity, and the kubelet calculates allocatable from
capacity at its next status update:

```bash
kubectl patch node kind-worker --subresource=status --type=json \
  -p '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"8"}]'
kubectl label node kind-worker cloud.google.com/gke-accelerator=nvidia-h100-80gb
kubectl taint node kind-worker nvidia.com/gpu=present:NoSchedule
```

(`~1` escapes `/` in a JSON-patch path.) kind leaves ExtendedResourceToleration off (section 1.4). Thus give GPU
pods the `nvidia.com/gpu` Exists/NoSchedule toleration yourself. Or turn on the plugin through a kind
`kubeadmConfigPatches` entry for the API server (keep `NodeRestriction`).

Then the scheduler places GPU pods. Kueue admits and preempts against real quota, and TAS reads your topology
labels. But containers get no device, because no device plugin answers `Allocate`. Lab notebook
`02_kind_with_fake_gpus_and_kueue` puts these steps in a script. When Docker is absent, the notebook uses a bundled
simulator instead.

The same kind setup teaches sandbox pods for model-generated code in
[07-application-agent-framework/sandboxed-execution](../../07-application-agent-framework/sandboxed-execution/README.md).
These pods have a RuntimeClass, Pod Security *restricted*, a default-deny NetworkPolicy and an admission policy.
The kind cluster cannot run gVisor, but the GKE Sandbox node pool of that topic can.

### 10.2 KWOK and fake-gpu-operator

**KWOK** (Kubernetes WithOut Kubelet) creates hundreds of fake Nodes with any capacity and labels, and it simulates
pod lifecycles. This is sufficient for tests of scheduling, Kueue and TAS behaviour at fleet scale on a laptop.
Run:ai's **fake-gpu-operator** goes one step further on CPU-only nodes. It gives a simulated device plugin, GPU
Feature Discovery labels, MIG and Prometheus metrics. Thus dashboards and label-based placement behave as they do on
GPUs.

| Real on a laptop | Simulated on a laptop |
|---|---|
| scheduler filters, scores, preemption, events | GPU devices, driver, `Allocate`, CUDA |
| Kueue admission, borrowed quota, reclaim, TAS assignment | NVLink / network bandwidth and topology effects on speed |
| JobSet / LWS lifecycles, restarts, gang admission | node provisioning, stockouts, Spot reclaims, DWS (this core's `autoscaler.py`) |
| manifests, lint checks, "why Pending" diagnosis | DCGM metrics (fake-gpu-operator), real utilisation |

### 10.3 Where each concept runs

| Concept | T0: laptop / Colab CPU | T1–T2: rented GPU box | T3: GKE | Other clouds |
|---|---|---|---|---|
| device plugin, capacity/allocatable | `k8s-gpu-core` notebook 01 | a Lambda VM or any VM with GPUs + k3s/kubeadm + GPU Operator (RunPod/Vast are containers: no kubelet of your own) | GPU node pool, driver auto-install | EKS/AKS + GPU Operator or the provider's device plugin (verify) |
| filter/score/preemption | notebook 02, kind | same | the managed scheduler (+ your own profile) | same |
| gangs, JobSet/LWS, Kueue, TAS | notebooks 03–04, kind + Kueue v0.19.6 | same | Kueue on GKE. TAS reads the GCE topology labels. A3/A4/A4X pools (GPUDirect/RDMA network, compact placement) carry them, and the L4 pools that the lab uses do not. Thus the lab's GKE Kueue flavors are non-TAS (verify). | Kueue anywhere. Provider topology labels are different. |
| autoscaling, Spot, queued provisioning | notebook 05 (simulated) | – | node pools, Spot, DWS flex-start, ComputeClass (lab notebook 04) | Karpenter, capacity reservations (verify) |
| shared GPUs (MIG, time-slicing) | notebook 01, exercise 1.6 (time-slicing replicas) | the lab's `deploy/gpu-vm` (time-slicing with the real device plugin), and an A100/H100 VM for MIG | node-pool settings to share GPUs (the lab's `deploy/gke/50-time-sharing-l4.yaml`) | GPU Operator configs |

Colab and Kaggle give you notebooks, not clusters. They run the core (T0) but not kind, because kind needs a Docker
daemon. [`COMPUTE.md`](../../COMPUTE.md) gives the prices, the free tiers and how easy it is to get each GPU.
[`CURRICULUM.md`](../../CURRICULUM.md) gives the order in which to work through the whole curriculum.

---

## In a design review

**The two-minute walkthrough.** "GPUs come to Kubernetes as integers. The device plugin advertises healthy devices.
GKE manages it here, and the GPU Operator manages it elsewhere. Requests are whole GPUs, with requests equal to
limits. GPU nodes have taints, and pods select a model with labels.

"We run separate node pools per GPU shape. The default scheduler spreads pods by CPU and memory, and ignores GPUs.
This leaves GPUs stranded. Thus GPU pools use a MostAllocated profile with a weight on the GPU. We monitor stranded
GPUs for our largest pod shape.

"Every workload with more than one pod goes through Kueue, for example training JobSets and multi-host
LeaderWorkerSet replicas. Kueue admits the jobs whole against ClusterQueue quota. TAS gives them topology-aware
placement (required for inference groups, preferred for training). If their pods do not all come up, Kueue evicts
and requeues them. Teams share a cohort. They lend idle quota, a `lendingLimit` keeps a floor for serving, and
reclaim makes nominal quota a real guarantee.

"For capacity, serving runs on on-demand, with Spot behind it. Interruptible batch runs on Spot. Large gangs go
through a ProvisioningRequest on DWS flex-start, so they get all their nodes at the same time. The steady base runs
on reservations. A cold start takes minutes. Thus latency-critical pools keep warm nodes and use image streaming,
and weights come from a cache."

**Drill questions.**

1. *An 8-GPU pod is Pending with `Insufficient nvidia.com/gpu` although the cluster is half idle. Why, and
   what do you change?* Fragmentation: on each node, the free GPUs are in pieces smaller than 8. Pack GPU pools
   (MostAllocated with a GPU weight), separate pools by pod shape, or preempt/defragment. Measure stranded GPUs
   for the shape.
2. *Two training jobs have been "running" for an hour and neither has logged a step. What occurred?* Partial
   gang placement: each job holds some GPUs and waits for the rest. Admit jobs whole (Kueue with
   `waitForPodsReady`, or a gang scheduler). Then one job runs, and the other waits and holds nothing.
3. *Team A has 16 GPUs of nominal quota, runs nothing, and its 16-GPU job is Pending. Why?* The ClusterQueue of
   team A lent its unused quota to the cohort, and `reclaimWithinCohort` is `Never`. Thus the borrowers keep the
   quota until they complete. Turn on reclaim, or set a `lendingLimit` to keep part of the quota home.
4. *Required or preferred topology for a 2-node tensor-parallel serving group, and for a 32-node training
   job?* For serving, use the required constraint at the smallest multi-node domain (a split replica is slow
   for its whole life). For training, use the preferred constraint at block level (locality, but not at the
   price of an indefinite wait).
5. *Why not run a 16-node training job on an autoscaled Spot pool?* Any reclaim restarts the gang. Survival
   decreases as $e^{-N \cdot \lambda \cdot T}$ (14.7% for 16 nodes over 24 h at $\lambda$ = 0.005/node-h,
   simulated). Also, a node-by-node scale-up bills idle partial gangs. Use flex-start or reservations, and write
   checkpoints.
6. *A GPU node shows capacity 8, allocatable 7. What does that mean for the pods that run now and for new
   pods?* One device is Unhealthy. Pods that run keep their GPUs, and new pods see 7. A pod that the scheduler
   placed on stale status can fail with `UnexpectedAdmissionError`. Drain and repair the node as an
   operational decision.

---

## Glossary

| Term | Meaning |
|---|---|
| Extended resource | A resource named with a domain prefix (`nvidia.com/gpu`). It is an integer, with requests = limits, and Kubernetes never overcommits it. |
| Capacity / allocatable | What a node has / what the scheduler can assign. For GPUs, all devices / healthy devices. |
| Device plugin | Node agent that registers a resource with the kubelet, sends a stream of device health and answers `Allocate` |
| DRA | Dynamic Resource Allocation. ResourceSlices describe the devices, and ResourceClaims with CEL selectors request them. |
| GPU Operator | NVIDIA's operator that installs the driver, container toolkit, device plugin, GFD, DCGM and MIG manager |
| GFD / NFD | GPU Feature Discovery / Node Feature Discovery. They write node labels that describe the hardware. |
| Taint / toleration | Node mark that keeps pods away / pod permission to ignore it |
| Filter / score | Scheduler phases: feasible nodes, then ranking of feasible nodes |
| LeastAllocated / MostAllocated | NodeResourcesFit score strategies: spread / bin-pack |
| Stranded GPUs | Free GPUs that cannot hold one more pod of a given shape. Per node: free GPUs − $k$ × (min over resources of how many such pods fit). This is $\mathit{free} \bmod k$ when only GPUs set the limit (fragmentation). It is more when the CPU or memory of the pod runs out first (bundle stranding). Fragmentation = stranded / free. |
| Preemption | The eviction of lower-priority pods (scheduler) or workloads (Kueue) to make room |
| Gang | A group of pods that is useful only when all pods run. Its placement is all-or-nothing. |
| JobSet | API for a group of Jobs that run as one training/HPC workload |
| LeaderWorkerSet (LWS) | API whose replicas are groups of pods, for multi-host inference |
| Kueue | A job queue system: quotas, cohorts, admission, preemption, topology-aware scheduling |
| ResourceFlavor | Kueue's name for a kind of capacity (GPU model, Spot or on-demand) |
| ClusterQueue / LocalQueue | Kueue quota holder / a namespace's entry point to it |
| Cohort | ClusterQueues that lend each other unused quota |
| Nominal quota, borrowingLimit, lendingLimit | Owned quota, the limit on what a queue borrows, the limit on what a queue lends |
| TAS | Topology-Aware Scheduling in Kueue: places pod sets inside topology domains |
| BestFit / LeastFreeCapacity | TAS algorithms: tightest domain that fits / tightest single host, else smallest gaps first |
| Cluster autoscaler | Adds nodes for pending pods and removes idle nodes |
| Expander | The autoscaler's rule to select which node pool to scale up |
| ProvisioningRequest | API that asks the autoscaler for capacity for a set of pods, possibly atomically |
| DWS flex-start | GCP Dynamic Workload Scheduler mode that provisions a queued request all at once, for up to 7 days |
| ComputeClass | GKE CRD that gives workloads an ordered list of capacity options |
| MIG / time-slicing / MPS | One GPU divided in hardware / shared in time / shared in space |
| KWOK | Kubernetes WithOut Kubelet: fake nodes and pods for tests of control-plane behaviour |

---

## Sources

* Kubernetes source (master, September 2026):
  * `pkg/scheduler/apis/config/v1/default_plugins.go` and `defaults.go` (default plugins, weights, NodeResourcesFit
    defaults),
  * `pkg/scheduler/framework/plugins/noderesources/{least_allocated,most_allocated,resource_allocation,fit}.go`,
  * `pkg/scheduler/framework/preemption/preemption.go`, `plugins/defaultpreemption/default_preemption.go`,
  * `pkg/scheduler/framework/types.go` (FitError message),
  * `pkg/apis/core/validation/validation.go` (extended-resource rules),
  * `pkg/kubelet/cm/devicemanager/manager.go` (capacity against allocatable),
  * `staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1/api.proto`,
  * `plugin/pkg/admission/extendedresourcetoleration` and `pkg/kubeapiserver/options/plugins.go` (default-off),
  * `staging/src/k8s.io/api/resource/v1/types.go` (DRA),
  * kind `pkg/cluster/internal/kubeadm/config.go`.
* KEP-4671 Gang Scheduling (`kubernetes/enhancements`, `keps/sig-scheduling/4671-gang-scheduling`).
* Kueue (`kubernetes-sigs/kueue`): README (v0.19.6), `apis/kueue/v1beta2/*_types.go`, concepts docs
  (cluster_queue, preemption, topology_aware_scheduling, admission_check/provisioning_request,
  workload_priority_class), `apis/config/v1beta2/defaults.go` (waitForPodsReady), KEP-2724 and
  `pkg/cache/scheduler/tas_flavor_snapshot.go` (TAS algorithms), `pkg/cache/scheduler/resource_node.go`
  (quota arithmetic), `pkg/scheduler/preemption/preemption.go` (classic preemption).
* JobSet (`kubernetes-sigs/jobset`, `api/jobset/v1alpha2`), LeaderWorkerSet (`kubernetes-sigs/lws`,
  `api/leaderworkerset/v1`).
* NVIDIA:
  * `k8s-device-plugin` README, GPU Feature Discovery labels, `internal/rm/allocate.go` (replicas),
  * `gpu-operator` README and ClusterPolicy types,
  * `kubernetes-sigs/dra-driver-nvidia-gpu` README, quickstart specs.
* Google:
  * `GoogleCloudPlatform/container-engine-accelerators` (GKE GPU device plugin: the rules to share GPUs, MIG
    partition sizes),
  * Terraform google provider 8.x schemas for node-pool and cluster attributes.
* cluster-autoscaler (`kubernetes/autoscaler`): FAQ (expanders, scale-down flags, expendable pods and
  overprovisioning), ProvisioningRequest v1 types, GCE cloud provider (GPU label). The Kubernetes docs:
  *Advertise Extended Resources for a Node*. Also KWOK (`kubernetes-sigs/kwok`) and Run:ai `fake-gpu-operator`.

---

## Verify list

Dated 26 September 2026. Examine these facts again before you rely on any of them.

| Fact | Status here |
|---|---|
| Kueue latest release v0.19.6, API `kueue.x-k8s.io/v1beta2`, TAS beta and on by default | from upstream README / docs |
| JobSet `jobset.x-k8s.io/v1alpha2` (release v0.12.0 in its README). LWS `leaderworkerset.x-k8s.io/v1`, tested on K8s 1.34–1.37. | from upstream READMEs |
| DRA `resource.k8s.io/v1` GA in 1.34. NVIDIA DRA GPU plugin "not yet officially supported", ComputeDomains supported. | upstream. It changes fast. |
| Native gang scheduling (KEP-4671): alpha 1.35, beta 1.37, stable targeted 1.38 | KEP metadata (verify the release you run) |
| Kueue `waitForPodsReady` on by default (v1beta2 Configuration, timeout 30 min), and the `DisableWaitForPodsReady` alpha gate | Kueue v0.19.6 `apis/config/v1beta2/defaults.go` |
| GKE GPU taint `nvidia.com/gpu=present:NoSchedule` and ExtendedResourceToleration on. `cloud.google.com/gce-topology-{block,subblock,host}` labels. | verify (the plugin is default-off upstream) |
| GKE driver auto-install from 1.32.2-gke.1297000. `gpu_driver_version` DEFAULT / LATEST / INSTALLATION_DISABLED. | from session research |
| GKE `optimize-utilization` profile makes the scheduler bin-pack | verify |
| ProvisioningRequest class for GKE queued provisioning `queued-provisioning.gke.io`, DWS flex-start up to 7 days, calendar mode | verify |
| ComputeClass `cloud.google.com/v1` field names (`priorities`, `spot`, `flexStart`, reservations, `nodePoolAutoCreation`) | verify against the CRD |
| Autopilot GPU selectors and limits, image streaming and secondary boot disk sources, GCS FUSE CSI caching options, Hyperdisk ML, model streamers per engine | verify |
| Spot discount 60–91% (GCP), Spot reclaim rates | The discount is from session research. The $\lambda$ that §7.2 uses is illustrative, not a measured rate. |
| cluster-autoscaler defaults: least-waste expander, 10 min unneeded and delay-after-add, 0.5 thresholds, 15 min provision time | from the FAQ on master |
| MIG profile counts per GPU (A100 40 GB 1g.5gb ×7, 2g.10gb ×3, 3g.20gb ×2, and H100 80 GB 1g.10gb ×7) | from GKE's device plugin source (verify for your driver) |
| NFD PCI label form `feature.node.kubernetes.io/pci-10de.present` | depends on NFD configuration |
