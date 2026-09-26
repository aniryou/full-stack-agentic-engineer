# %% [markdown]
# # 01 · Manifests and the linter: what makes a pod a GPU pod
#
# **Tier:** T0 — laptop or Colab CPU, no cluster, $0. Everything here is builders, a linter and
# arithmetic; notebook 02 applies the same objects to a kind cluster.
#
# ## The one-minute version
# A GPU pod is ordinary Kubernetes plus a handful of load-bearing fields:
#
# * an **integer `nvidia.com/gpu` limit** — an *extended resource*, counted by the scheduler,
#   never overcommitted, so request == limit (primer §1 *What Kubernetes sees*);
# * a **toleration** for the `nvidia.com/gpu` taint and a **selector** for the accelerator model;
# * **CPU/memory requests sized to the per-GPU share** of the node, or the node's other GPUs are
#   stranded (primer §3.4 *Fragmentation, measured*, generalised below);
# * a **startup probe** that outlasts the weight load (primer §8 *Startup latency*);
# * for gangs, a **queue label** and a **topology annotation** that Kueue reads (primer §4 *Gangs*,
#   §5 *Topology-aware placement*).
#
# By the end you can build these objects with `k8sgpu.manifests`, explain every finding of
# `k8sgpu.lint`, and size the two numbers people get wrong: the startup-probe budget and the
# CPU request of a 1-GPU pod. Primer: [`../../PRIMER.md`](../../PRIMER.md).

# %%
from k8sgpu import lint, machines
from k8sgpu import manifests as m

# A single-GPU Job, built the way every lab object is built.
c = m.GPUContainer(name="train", image="busybox:1.38.0", gpus=1, cpu="4", memory="16Gi",
                   command=["sh", "-c", "echo hello from a GPU pod; sleep 30"])
job = m.job("hello-gpu", "team-a", m.pod_template(m.pod_spec([c])))
print(m.to_yaml(job))

# %% [markdown]
# Read the `resources` block: the GPU appears in **limits** and **requests** with the same whole
# number; CPU has a request but no limit (a CFS limit throttles the process that feeds the GPU);
# memory has both. The pod spec carries the toleration and the accelerator selector — the builder
# adds them because without the toleration a GPU pod stays Pending on clusters that taint GPU
# nodes and do not run the ExtendedResourceToleration admission plugin (kind, kubeadm defaults;
# GKE enables the plugin, verify), and without the selector it can land on any GPU model.
#
# ## What the API server rejects — and what it does not
# `nvidia.com/gpu` is an extended resource: whole units, no overcommit. The API server enforces
# that for Pods, Jobs and Deployments. The linter reproduces its three messages verbatim:

# %%
for res in ({"requests": {m.GPU: 1}},                        # request without a limit
            {"requests": {m.GPU: 1}, "limits": {m.GPU: 2}},  # request != limit
            {"limits": {m.GPU: "500m"}}):                     # half a GPU
    print(res, "->", lint.check_gpu_resources(res))

# %% [markdown]
# A **CRD** that embeds a pod template (JobSet, LeaderWorkerSet) is validated against its own
# schema only: `kubectl apply` succeeds, and the error appears minutes later, when the controller
# tries to create pods — as an event on an object you are no longer looking at. The classic case:
# a LeaderWorkerSet group is run by StatefulSets, which accept only `restartPolicy: Always`.

# %%
tmpl = m.pod_template(m.pod_spec([m.GPUContainer(gpus=4)], shm_size="1Gi"))   # restartPolicy: Never (a Job habit)
lws = m.leader_worker_set("llm", "team-b", size=2, worker_template=tmpl, queue="gpu-queue")
print(lint.format_findings(lint.lint(lws)))

# %% [markdown]
# ## Exercise 1.1 — the API server's GPU rules
#
# Write `gpu_resources_ok(resources)` returning a list of problems (empty = accepted) for one
# container's `resources` dict, applying the three rules for `nvidia.com/gpu`:
# 1. a quantity must be a whole number (`"500m"` and `0.5` are not);
# 2. a request needs a limit;
# 3. if both are given they must be equal.
#
# Use `lint.parse_quantity(q)` (returns an exact `Fraction`). Return any non-empty strings you
# like; the check compares *which* rules fire, not the wording.

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
# A serving pod loads weights before it answers `/health`. Until the **startup** probe succeeds
# the kubelet does not run the liveness probe; if the startup probe fails `failureThreshold`
# times, the container is killed and the load starts again — forever.
#
# Budget = `failureThreshold × periodSeconds`. Write `failure_threshold(load_s, period_s,
# margin=1.5)`: the smallest integer threshold whose budget covers `load_s × margin`. Then set
# two variables: `load_s`, the load time of an 8B model in bf16 (16 GB) read at 400 MB/s plus
# 60 s of engine init (CUDA graphs, warm-up), and `threshold`, the failure threshold for it with
# a 10 s period.

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
# GPUs strand in two ways, and one formula covers both. For a pod shape of *k* GPUs, the pods
# that still fit a node are the minimum over resources of ⌊free / request⌋, and the stranded
# GPUs are the free GPUs minus *k* × that:
#
# * **GPU-count fragmentation** — only GPUs bind: stranded = `free mod k` per node, the count
#   primer §3.4 *Fragmentation, measured* tracks (a 3-GPU pod shape leaves 2 of 8 GPUs idle).
# * **Resource-bundle stranding** — CPU or memory binds first. A node is a bundle:
#   `g2-standard-48` has 4 L4s and 48 vCPUs; after GKE's reservations about 47.8 vCPUs and
#   181 GiB are allocatable (`k8sgpu.machines.allocatable`, formula marked *verify*). If each
#   1-GPU pod asks for 16 vCPUs, only two pods fit and two GPUs idle — paid for, unusable.
#
# The per-GPU share is the budget before anything else runs on the node. DaemonSets (logging,
# monitoring, the device plugin) and injected sidecars (GKE's GCS FUSE sidecar, a service-mesh
# proxy) take from the same bundle, so size pods against what is left; the numbers below use an
# illustrative 0.5 vCPU / 1 GiB DaemonSet budget (read yours from `kubectl describe node`).

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
# Write `stranded(node_cpu, node_mem, node_gpus, pod_cpu, pod_mem, pod_gpus)`: pack identical
# pods onto one empty node until *any* resource runs out, and return the GPUs left idle.
# Then answer: on a `g2-standard-48`, what is the largest **whole** number of vCPUs a 1-GPU pod
# can request (memory 40 GiB) without stranding a GPU? Put it in `max_cpu`.

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
# A distributed job is useless until *all* its pods run: a gang. Kueue admits a JobSet as one
# Workload (all pods or none), and with Topology-Aware Scheduling it also chooses **where**: the
# pod-template annotation `kueue.x-k8s.io/podset-required-topology: <node label>` demands that
# every pod lands inside one domain of that level (one host = one NVLink domain; one subblock =
# a few hosts on the same leaf switch). The labels are GCE's placement labels (verify).

# %%
print("levels, coarse to fine:", m.GKE_TOPOLOGY_LEVELS)

# %% [markdown]
# ## Exercise 1.4 — build a 2-host tensor-parallel gang
#
# Build `gang`: a JobSet named `tp16` in namespace `team-a`, queued on LocalQueue `gpu-queue`,
# with one replicated job `workers` of **2 pods × 8 GPUs** (think: one model sharded over 16
# GPUs on two 8-GPU hosts) whose pods must share a **subblock**. Each pod: 8 GPUs, 32 vCPUs,
# 256Gi memory, a memory-backed `/dev/shm` of 16Gi, accelerator `nvidia-h100-80gb`.
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
# The Deployment below serves a model on a `g2-standard-8` (1 L4, 8 vCPUs, 32 GB). Its weights
# take about 240 s to load. Edit `broken` **in place** (it is a plain dict) until
# `lint.lint(broken, machine="g2-standard-8", expected_load_s=240)` returns no errors and no
# warnings. Every finding names its fix. One of them is a trade-off, not a bug: a rolling update
# surges an extra GPU pod by default. `maxSurge: 0, maxUnavailable: 1` avoids needing a spare GPU,
# but with one replica every rollout is an outage lasting a whole cold start. That is fine for a
# lab; production runs at least two replicas (on on-demand capacity first) or keeps surge headroom.

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
left = lint.lint(broken, machine="g2-standard-8", expected_load_s=240)
assert not [f for f in left if f.severity in ("error", "warning")], lint.format_findings(left)
print("✅ lint-clean:", lint.format_findings(left))

# %% [markdown]
# ## Two more ways to ask for a GPU
# **DRA** (Dynamic Resource Allocation, `resource.k8s.io/v1`, GA in Kubernetes 1.34) replaces
# "give me 1 of this counter" with "give me a device matching this CEL expression": a driver
# publishes devices and their attributes in `ResourceSlice`s, an admin defines `DeviceClass`es,
# and a pod references a `ResourceClaimTemplate`. **GKE ComputeClasses** move the other half —
# *which node to create* — into an ordered fallback list (notebook 04). Primer §1 and §9.

# %%
rct = m.resource_claim_template("one-l4", "default",
                                cel=['device.attributes["gpu.nvidia.com"].productName == "NVIDIA L4"'])
print(m.to_yaml(rct))

# %% [markdown]
# ## In a design review
# *"How do you run a GPU workload on Kubernetes without wasting GPUs?"* — in two minutes:
# the GPU is an integer extended resource the device plugin advertises; request it as a limit.
# GPU nodes are tainted, so the pod tolerates the taint and selects its accelerator. Size CPU
# and memory to the node's per-GPU share or you strand GPUs. Serving pods need a startup probe
# sized to the weight load. Multi-pod jobs are gangs: queue them (Kueue) and pin them to a
# topology domain. And lint before you apply: CRDs accept broken pod templates.
#
# **Drill 1.** *A JobSet was applied fine but no pods ever appeared. First place to look?*
# Events on the child Jobs / the JobSet: its controller hit pod validation the CRD did not do
# (e.g. a bad `restartPolicy` or a GPU request without a limit).
#
# **Drill 2.** *Why is `requests: {nvidia.com/gpu: 1}` without a limit rejected when CPU is fine?*
# Extended resources cannot be overcommitted, so the limit is the allocation; the API server
# requires it (and fills the request from it).
#
# **Drill 3.** *A 4-GPU node runs only two 1-GPU pods and the rest sit idle. Why?*
# The pods' CPU or memory requests exceed a quarter of the node's allocatable: CPU ran out
# before GPUs. Size requests to the per-GPU share.
