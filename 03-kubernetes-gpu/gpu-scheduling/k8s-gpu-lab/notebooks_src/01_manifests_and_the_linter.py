# %% [markdown]
# # 01 · Manifests and the linter: what makes a pod a GPU pod
#
# **Tier:** T0. It runs on a laptop or Colab CPU, with no cluster, at $0. Everything here is builders, a linter
# and arithmetic. Notebook 02 applies the same objects to a kind cluster.
#
# ## The one-minute version
# A GPU pod is ordinary Kubernetes plus a few fields. The design depends on each of these fields:
#
# * an **integer `nvidia.com/gpu` limit**. It is an *extended resource*. The scheduler counts it and never
#   overcommits it, thus request == limit (primer §1 *What Kubernetes sees*).
# * a **toleration** for the `nvidia.com/gpu` taint and a **selector** for the accelerator model.
# * **CPU/memory requests sized to the per-GPU share** of the node. If not, the other GPUs of the node become
#   stranded (primer §3.4 *Fragmentation, measured*, which the section *Stranded GPUs* in this notebook makes
#   more general).
# * a **startup probe** that lasts longer than the weight load (primer §8 *Startup latency*).
# * for gangs, a **queue label** and a **topology annotation**. Kueue reads them (primer §4 *Gangs*,
#   §5 *Topology-aware placement*).
#
# At the end, you can build these objects with `k8sgpu.manifests` and explain every finding of
# `k8sgpu.lint`. You can also calculate the two numbers that people frequently set incorrectly: the
# startup-probe budget and the CPU request of a 1-GPU pod. Primer: [`../../PRIMER.md`](../../PRIMER.md).

# %%
import copy

from k8sgpu import lint, machines
from k8sgpu import manifests as m

# A single-GPU Job, built the way every lab object is built.
c = m.GPUContainer(name="train", image="busybox:1.38.0", gpus=1, cpu="4", memory="16Gi",
                   command=["sh", "-c", "echo hello from a GPU pod; sleep 30"])
job = m.job("hello-gpu", "team-a", m.pod_template(m.pod_spec([c])))
print(m.to_yaml(job))

# %% [markdown]
# Read the `resources` block. The GPU is in **limits** and in **requests**, with the same whole number.
# CPU has a request but no limit, because a CFS limit throttles the process that feeds the GPU. Memory has
# both. The pod spec has the toleration and the accelerator selector. The builder adds them for two reasons:
#
# * Without the toleration, a GPU pod stays Pending on some clusters. These clusters put a taint on GPU nodes
#   and do not run the ExtendedResourceToleration admission plugin. Examples are kind and the kubeadm
#   defaults. GKE turns on the plugin (verify).
# * Without the selector, the scheduler can put the pod on any GPU model.
#
# ## What the API server rejects — and what it does not
# `nvidia.com/gpu` is an extended resource. It has whole units and no overcommit. The API server applies
# these rules to Pods, Jobs and Deployments. The linter gives the same three messages as the API server, word
# for word:

# %%
for res in ({"requests": {m.GPU: 1}},                        # request without a limit
            {"requests": {m.GPU: 1}, "limits": {m.GPU: 2}},  # request != limit
            {"limits": {m.GPU: "500m"}}):                     # half a GPU
    print(res, "->", lint.check_gpu_resources(res))

# %% [markdown]
# The API server validates a **CRD** that contains a pod template (JobSet, LeaderWorkerSet) against its own
# schema only. Thus `kubectl apply` succeeds. The error comes minutes later, when the controller tries to
# create pods. It is an event on an object that you no longer look at. The classic case is a
# LeaderWorkerSet group. StatefulSets run this group, and StatefulSets accept only `restartPolicy: Always`.

# %%
tmpl = m.pod_template(m.pod_spec([m.GPUContainer(gpus=4)], shm_size="1Gi"))   # restartPolicy: Never (a Job habit)
lws = m.leader_worker_set("llm", "team-b", size=2, worker_template=tmpl, queue="gpu-queue")
print(lint.format_findings(lint.lint(lws)))

# %% [markdown]
# ## Exercise 1.1 — the API server's GPU rules
#
# Write `gpu_resources_ok(resources)` for the `resources` dict of one container. It returns a list of
# problems (an empty list means accepted). Apply the three rules for `nvidia.com/gpu`:
# 1. a quantity must be a whole number (`"500m"` and `0.5` are not).
# 2. a request must have a limit.
# 3. if both are present, they must be equal.
#
# Use `lint.parse_quantity(q)`. It returns an exact `Fraction`. Return any non-empty strings that you
# want. The check compares *which* rules find a problem, not the text of the messages.

# %% exercise
def gpu_resources_ok(resources: dict) -> list[str]:
    req = (resources.get("requests") or {}).get(m.GPU)
    lim = (resources.get("limits") or {}).get(m.GPU)
    problems = []
    ### BEGIN SOLUTION
    for q in (req, lim):
        if q is not None and lint.parse_quantity(q).denominator != 1:
            problems.append(f"{q} is not an integer")
    if req is not None and lim is None:
        problems.append("limit required")
    elif req is not None and lint.parse_quantity(req) != lint.parse_quantity(lim):
        problems.append("request must equal limit")
    ### END SOLUTION
    return problems

# %% check
cases = [{"limits": {m.GPU: 2}}, {"requests": {m.GPU: 1}}, {"requests": {m.GPU: 1}, "limits": {m.GPU: 2}},
         {"limits": {m.GPU: "500m"}}, {"requests": {m.GPU: "1"}, "limits": {m.GPU: 1}}, {}]
for res in cases:
    assert bool(gpu_resources_ok(res)) == bool(lint.check_gpu_resources(res)), res
assert len(gpu_resources_ok({"requests": {m.GPU: 1}})) == 1
print("✅ gpu_resources_ok agrees with the API server on", len(cases), "cases")

# %% [markdown]
# ## Exercise 1.2 — size the startup probe
#
# A serving pod loads weights before it answers `/health`. Until the **startup** probe succeeds, the
# kubelet does not run the liveness probe. If the startup probe fails `failureThreshold` times, the kubelet
# stops the container and the load starts again. This loop continues forever.
#
# The budget is `failureThreshold × periodSeconds`. Write `failure_threshold(load_s, period_s,
# margin=1.5)`. It returns the smallest integer threshold with a budget that covers `load_s × margin`.
# Then set two variables:
#
# * `load_s`: the load time of an 8B model in bf16 (16 GB), read at 400 MB/s, plus 60 s of engine init
#   (CUDA graphs, warm-up).
# * `threshold`: the failure threshold for this load with a 10 s period.

# %% exercise
import math

def failure_threshold(load_s: float, period_s: float, margin: float = 1.5) -> int:
    ### BEGIN SOLUTION
    return max(1, math.ceil(load_s * margin / period_s))
    ### END SOLUTION

### BEGIN SOLUTION
load_s = 16e9 / 400e6 + 60          # 40 s of reading + 60 s of init
threshold = failure_threshold(load_s, 10)
### END SOLUTION

# %% check
assert load_s == 100 and threshold == 15
assert failure_threshold(300, 10) == m.startup_probe_for(300, period_s=10)["failureThreshold"] == 45
print(f"✅ load {load_s:.0f} s -> failureThreshold {threshold} x 10 s = {threshold * 10} s of budget")

# %% [markdown]
# ## Stranded GPUs: the CPU request is a GPU decision
# GPUs become stranded in two ways, and one formula covers the two ways. Take a pod shape of $k$ GPUs. The
# number of pods that still fit on a node is the minimum, over all resources, of $\lfloor \mathit{free} / \mathit{request} \rfloor$. The
# stranded GPUs are the free GPUs minus $k$ × that number:
#
# * **GPU-count fragmentation**: only the GPUs run out. Stranded = $\mathit{free} \bmod k$ per node. This
#   is the count that primer §3.4 *Fragmentation, measured* records (a 3-GPU pod shape leaves 2 of 8 GPUs
#   idle).
# * **Resource-bundle stranding**: CPU or memory runs out first. A node is a bundle. `g2-standard-48` has
#   4 L4s and 48 vCPUs. After GKE's reservations, approximately 47.8 vCPUs and 181 GiB are allocatable
#   (`k8sgpu.machines.allocatable`, formula marked *verify*). If each 1-GPU pod asks for 16 vCPUs, only two
#   pods fit and two GPUs are idle. You pay for these two GPUs, but you cannot use them.
#
# The per-GPU share is the budget before anything else runs on the node. DaemonSets (logging, monitoring, the
# device plugin) and injected sidecars (GKE's GCS FUSE sidecar, a service-mesh proxy) take from the same
# bundle. Thus, set the size of the pods against the resources that stay free. The numbers in the next cell
# use an illustrative 0.5 vCPU / 1 GiB DaemonSet budget. Read your budget from `kubectl describe node`.

# %%
for name in ("g2-standard-4", "g2-standard-48", "a3-highgpu-8g"):
    a, share = machines.allocatable(name), machines.per_gpu_share(name)
    ds = machines.per_gpu_share(name, daemonset_cpu=0.5, daemonset_mem_gib=1.0)
    print(f"{name:15} allocatable cpu {a.cpu:7.2f}  mem {a.memory_gib:8.1f} GiB  gpus {a.gpus}"
          f"   -> per-GPU share: cpu {share.cpu:6.2f}, mem {share.memory_gib:7.1f} GiB"
          f"   (after DaemonSets: cpu {ds.cpu:6.2f}, mem {ds.memory_gib:7.1f})")
# the same function, when only GPUs bind, is primer §3.4's free mod k:
three_gpu = machines.stranded_gpus(machines.allocatable("a3-highgpu-8g"), pod_cpu=0, pod_mem_gib=0, pod_gpus=3)
print("a3-highgpu-8g packed with 3-GPU pods strands", three_gpu, "GPUs =", 8 % 3, "= 8 mod 3")

# %% [markdown]
# ## Exercise 1.3 — how many GPUs does a request strand?
#
# Write `stranded(node_cpu, node_mem, node_gpus, pod_cpu, pod_mem, pod_gpus)`. The function packs identical
# pods onto one empty node until *any* resource runs out. Then it returns the number of idle GPUs.
#
# Then answer this question. On a `g2-standard-48`, a 1-GPU pod has memory 40 GiB. What is the largest
# **whole** number of vCPUs that it can request with no stranded GPU? Put the answer in `max_cpu`.

# %% exercise
def stranded(node_cpu, node_mem, node_gpus, pod_cpu, pod_mem, pod_gpus) -> int:
    ### BEGIN SOLUTION
    fits = min(math.floor(node_cpu / pod_cpu), math.floor(node_mem / pod_mem), node_gpus // pod_gpus)
    return node_gpus - fits * pod_gpus
    ### END SOLUTION

a48 = machines.allocatable("g2-standard-48")
### BEGIN SOLUTION
max_cpu = max(c for c in range(1, 49) if stranded(a48.cpu, a48.memory_gib, 4, c, 40, 1) == 0)
### END SOLUTION

# %% check
assert stranded(a48.cpu, a48.memory_gib, 4, 16, 40, 1) == 2
assert stranded(a48.cpu, a48.memory_gib, 4, 11, 40, 1) == 0
assert stranded(a48.cpu, a48.memory_gib, 4, 4, 100, 1) == 3          # memory can strand too
assert stranded(8, 1000, 8, 0.001, 0.001, 3) == 2                     # only GPUs bind: 8 mod 3
assert max_cpu == 11
print(f"✅ a 1-GPU pod on g2-standard-48 can ask for at most {max_cpu} vCPUs; 16 would strand 2 GPUs")

# %% [markdown]
# ## Gangs: a queue label and a topology annotation
# A gang is a distributed job that is of no use until *all* of its pods run. Kueue admits a JobSet as one
# Workload (all pods or none). With Topology-Aware Scheduling, Kueue also selects **where** the pods go.
# The pod-template annotation `kueue.x-k8s.io/podset-required-topology: <node label>` says that every pod
# must go inside one domain of that level. One host is one NVLink domain, and one subblock is a few hosts on
# the same leaf switch. The labels are GCE's placement labels (verify).

# %%
print("levels, coarse to fine:", m.GKE_TOPOLOGY_LEVELS)

# %% [markdown]
# ## Exercise 1.4 — build a 2-host tensor-parallel gang
#
# Build `gang`. It is a JobSet with the name `tp16` in namespace `team-a`, in the LocalQueue `gpu-queue`.
# It has one replicated job `workers` of **2 pods × 8 GPUs** (think of one model sharded over 16 GPUs on
# two 8-GPU hosts). The pods of this job must share a **subblock**. Each pod has 8 GPUs, 32 vCPUs, 256Gi
# memory, a memory-backed `/dev/shm` of 16Gi and the accelerator `nvidia-h100-80gb`.
# Use `m.GPUContainer`, `m.pod_spec(..., accelerator=..., shm_size=...)`,
# `m.topology_annotations`, `m.pod_template`, `m.replicated_job` and `m.jobset`.

# %% exercise
### BEGIN SOLUTION
worker = m.GPUContainer(name="worker", image="busybox:1.38.0", gpus=8, cpu="32", memory="256Gi")
tmpl = m.pod_template(m.pod_spec([worker], accelerator="nvidia-h100-80gb", shm_size="16Gi"),
                      annotations=m.topology_annotations(required=m.TOPOLOGY_SUBBLOCK))
gang = m.jobset("tp16", "team-a", [m.replicated_job("workers", tmpl, parallelism=2)], queue="gpu-queue")
### END SOLUTION

# %% check
rj = gang["spec"]["replicatedJobs"][0]
pod = rj["template"]["spec"]["template"]
assert gang["kind"] == "JobSet" and gang["metadata"]["labels"][m.QUEUE_LABEL] == "gpu-queue"
assert rj["template"]["spec"]["parallelism"] == 2
assert pod["metadata"]["annotations"][m.TAS_REQUIRED] == m.TOPOLOGY_SUBBLOCK
assert m.gpu_count(pod["spec"]["containers"][0]) == 8
assert [f for f in lint.lint(gang, machine="a3-highgpu-8g") if f.severity != "info"] == []
print("✅ a lint-clean 16-GPU gang that Kueue will place inside one subblock")
print(m.to_yaml(gang)[:600], "...")

# %% [markdown]
# ## Exercise 1.5 — make a serving Deployment lint-clean
#
# The Deployment in the exercise cell serves a model on a `g2-standard-8` (1 L4, 8 vCPUs, 32 GB). Its
# weights take approximately 240 s to load. Edit `broken` **in place** (it is a plain dict). Continue until
# `lint.lint(broken, machine="g2-standard-8", expected_load_s=240)` returns no errors and no
# warnings. Every finding tells you its repair.
#
# One of the findings is a trade-off, not a bug. By default, a rolling update starts one more GPU pod (a
# surge). With `maxSurge: 0, maxUnavailable: 1`, you do not need a spare GPU. But with one replica, every
# rollout is an outage for the full time of a cold start. That is acceptable for a lab. A production
# deployment runs at least two replicas (on on-demand capacity first) or keeps headroom for a surge.

# %%
print(lint.format_findings(lint.lint(
    {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "preview"},
     "spec": {"template": {"spec": {"containers": [{"name": "vllm", "image": "vllm/vllm-openai:v0.30.0",
              "ports": [{"containerPort": 8000}], "resources": {"requests": {m.GPU: 1}},
              "livenessProbe": {"httpGet": {"path": "/health", "port": 8000}}}]}}}},
    machine="g2-standard-8", expected_load_s=240)))

# %% exercise
broken = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "llm", "namespace": "serving"},
          "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "llm"}},
                   "template": {"metadata": {"labels": {"app": "llm"}},
                                "spec": {"containers": [{
                                    "name": "vllm", "image": "vllm/vllm-openai:v0.30.0",
                                    "ports": [{"containerPort": 8000}],
                                    "resources": {"requests": {m.GPU: 1}},
                                    "livenessProbe": {"httpGet": {"path": "/health", "port": 8000}}}]}}}}
original = copy.deepcopy(broken)  # the check compares your edit with this
### BEGIN SOLUTION
spec = broken["spec"]["template"]["spec"]
c0 = spec["containers"][0]
c0["resources"] = {"requests": {"cpu": "6", "memory": "24Gi", m.GPU: 1}, "limits": {"memory": "24Gi", m.GPU: 1}}
c0["startupProbe"] = m.startup_probe_for(240, "/health", 8000)
spec["tolerations"] = [dict(m.GPU_TOLERATION)]
spec["nodeSelector"] = {m.GKE_ACCELERATOR: "nvidia-l4"}
broken["spec"]["strategy"] = {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}}
### END SOLUTION

# %% check
def serious(obj):
    return [f for f in lint.lint(obj, machine="g2-standard-8", expected_load_s=240) if f.severity in ("error", "warning")]


def gpus(container):  # the GPU count a container asks for: the limit, or the request if no limit is set
    res = container.get("resources") or {}
    q = (res.get("limits") or {}).get(m.GPU, (res.get("requests") or {}).get(m.GPU, 0))
    return int(lint.parse_quantity(q))


was = serious(original)
assert len(was) >= 5, "`original` must be the Deployment as given: the linter flags it with 1 error and 4 warnings"
assert not serious(broken), lint.format_findings(serious(broken))
assert broken.get("kind") == "Deployment" and broken["metadata"] == original["metadata"], \
    "fix the same Deployment in place: same kind, name and namespace"
got = broken["spec"]["template"]["spec"].get("containers") or []
want = original["spec"]["template"]["spec"]["containers"]
assert [(x.get("name"), x.get("image")) for x in got] == [(x["name"], x["image"]) for x in want], \
    "keep the vllm container: a pod without it serves nothing"
assert gpus(got[0]) == gpus(want[0]) and (got[0]["resources"].get("limits") or {}).get(m.GPU) is not None, \
    "the container must still get its GPU, as a limit"
print("✅ lint-clean — cleared:", ", ".join(sorted({f.rule for f in was})))
print(lint.format_findings(lint.lint(broken, machine="g2-standard-8", expected_load_s=240)))

# %% [markdown]
# ## Two more ways to ask for a GPU
# **DRA** (Dynamic Resource Allocation, `resource.k8s.io/v1`, GA in Kubernetes 1.34) replaces
# "give me 1 of this counter" with "give me a device matching this CEL expression". In DRA, a driver
# publishes devices and their attributes in `ResourceSlice`s. An admin defines `DeviceClass`es, and a pod
# refers to a `ResourceClaimTemplate`. **GKE ComputeClasses** move the other half of the decision, *which
# node to create*, into an ordered fallback list (notebook 04). Primer §1 and §9.

# %%
rct = m.resource_claim_template("one-l4", "default",
                                cel=['device.attributes["gpu.nvidia.com"].productName == "NVIDIA L4"'])
print(m.to_yaml(rct))

# %% [markdown]
# ## In a design review
# *"How do you run a GPU workload on Kubernetes without wasting GPUs?"* The answer, in two minutes:
#
# * The GPU is an integer extended resource that the device plugin advertises. Request it as a limit.
# * GPU nodes have a taint. Thus the pod has a toleration for the taint and selects its accelerator.
# * Set CPU and memory to the per-GPU share of the node. If not, you strand GPUs.
# * A serving pod must have a startup probe sized to the weight load.
# * Multi-pod jobs are gangs. Put them in a queue (Kueue) and pin them to a topology domain.
# * Run the linter before you apply. CRDs accept broken pod templates.
#
# **Drill 1.** *You applied a JobSet with no error, but no pods ever appeared. Where do you look first?*
# Look at the events on the child Jobs or on the JobSet. Its controller failed the pod validation that the
# CRD did not do (for example, an incorrect `restartPolicy` or a GPU request without a limit).
#
# **Drill 2.** *Why does the API server reject `requests: {nvidia.com/gpu: 1}` without a limit, when it
# accepts the same for CPU?* You cannot overcommit an extended resource, thus the limit is the allocation.
# The API server rejects the request without the limit (and fills the request from the limit).
#
# **Drill 3.** *A 4-GPU node runs only two 1-GPU pods, and the other GPUs are idle. Why?*
# The CPU or memory requests of the pods are more than a quarter of the node's allocatable. The CPU ran out
# before the GPUs. Set the requests to the per-GPU share.
