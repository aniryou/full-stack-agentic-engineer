# %% [markdown]
# # 01 · How Kubernetes sees a GPU
#
# **Tier:** T0 — pure Python on a laptop, Colab CPU or CI. No cluster, no GPU. The same objects on a
# real cluster are in the lab (`k8s-gpu-lab`): notebook `02_kind_with_fake_gpus_and_kueue` (kind, T0/T1)
# and `04_gke_pools_dws_and_computeclasses` (GKE, T3).
#
# ## The one-minute version
#
# Kubernetes has no idea what a GPU is. A **device plugin** on each node tells the kubelet "I manage
# `nvidia.com/gpu`; here are 8 device IDs and their health". The kubelet publishes **capacity** (all
# devices) and **allocatable** (healthy devices) in the Node status, and the scheduler treats that
# number as an opaque integer. So a GPU request must be a whole number, must equal its limit, and is
# never overcommitted. *Which* GPU (L4 or H100) and *whether* a pod may land on a GPU node at all are
# ordinary **labels** and **taints**. After this notebook you can explain every GPU line of
# `kubectl describe node`, and why a pod the scheduler placed can still be rejected by the kubelet.
#
# Primer: §1 *What Kubernetes sees* and §2 *GPU Operator vs managed drivers* in `../../PRIMER.md`.

# %%
from gpusched import (GPU, AdmissionError, Cluster, DevicePlugin, Kubelet, Node, Pod, Taint, Toleration,
                      effective_requests, gpu_node, gpu_pod, make_gpus, run_filters)
from gpusched.deviceplugin import KUBELET_SOCKET, UNHEALTHY

# A node with 8 (fake) GPUs in two NVLink islands of 4, and the kubelet's device manager.
plugin = DevicePlugin(make_gpus(8, island_size=4))
kubelet = Kubelet()
print("1. Register() on", KUBELET_SOCKET, "->", plugin.register_request())
kubelet.register(plugin)
print("2. ListAndWatch() first message:", plugin.list_and_watch()[:2], "... (8 devices)")
print("3. node status the kubelet publishes:", kubelet.node_status())

# %% [markdown]
# That last line is all the scheduler will ever know: `nvidia.com/gpu: 8`. When a pod that asks for
# GPUs lands on the node, the kubelet picks free healthy device IDs (asking the plugin for a
# *preferred* set first) and calls `Allocate`. With the NVIDIA plugin's default `envvar` strategy
# the answer is one environment variable; the NVIDIA Container Toolkit reads it when the container
# is created and injects the device nodes and driver libraries (layer 02 explains that half).

# %%
print(kubelet.admit("trainer-0", 4))
print(kubelet.admit("server-0", 2))
print("devices pinned per container:", kubelet.assigned)

# %% [markdown]
# ## Exercise 1.1 — taints and tolerations
#
# GPU nodes are tainted (GKE uses `nvidia.com/gpu=present:NoSchedule` — verify) so that ordinary pods stay
# off expensive nodes. A pod may land on a tainted node only if one of its tolerations *tolerates*
# the taint. Implement the upstream rule `tolerates(tol, taint)`:
#
# 1. if the toleration names an effect, it must equal the taint's effect;
# 2. if the toleration names a key, it must equal the taint's key (an empty key matches every key);
# 3. operator `Exists` then matches any value; operator `Equal` (or empty) needs equal values.

# %% exercise
def tolerates(tol: Toleration, taint: Taint) -> bool:
    ### BEGIN SOLUTION
    if tol.effect and tol.effect != taint.effect:
        return False
    if tol.key and tol.key != taint.key:
        return False
    if tol.operator == "Exists":
        return True
    return tol.operator in ("", "Equal") and tol.value == taint.value
    ### END SOLUTION

# %% check
gpu_taint = Taint(GPU, "present", "NoSchedule")
cases = [Toleration(GPU, "Exists"), Toleration(GPU, "Equal", "present", "NoSchedule"),
         Toleration(GPU, "Equal", "absent"), Toleration(GPU, "Exists", effect="NoExecute"),
         Toleration(operator="Exists"), Toleration("dedicated", "Exists")]
assert [tolerates(t, gpu_taint) for t in cases] == [True, True, False, False, True, False]
assert all(tolerates(t, gpu_taint) == t.tolerates(gpu_taint) for t in cases)
print("✅ tolerates() matches the core/v1 rule")

# %% [markdown]
# Nobody writes that toleration by hand on GKE: the **ExtendedResourceToleration** admission plugin
# adds `{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}` to every pod that *requests*
# `nvidia.com/gpu`. `gpu_pod()` in this package does the same. So the taint does exactly one job:
# it keeps pods that do **not** ask for GPUs off GPU nodes.

# %%
cluster = Cluster([gpu_node("gpu-0", 8), Node("cpu-0", {"cpu": 32000, "memory": 131072})])
web = Pod("web", {"cpu": 500, "memory": 512})              # no GPU request, no toleration
trainer = gpu_pod("trainer", 1)                           # toleration added for it
print("trainer tolerations:", trainer.tolerations)
for pod in (web, trainer):
    print(f"{pod.name:<8}", {n.name: run_filters(pod, n) or "fits" for n in cluster.nodes.values()})

# %% [markdown]
# ## Exercise 1.2 — the API server's rules for a GPU request
#
# `nvidia.com/gpu` is an *extended resource*: a name with a domain prefix outside `kubernetes.io`.
# For such resources the API server enforces three rules and one default:
#
# * the quantity must be an integer — message `must be an integer`;
# * if both request and limit are set they must be equal — `must be equal to nvidia.com/gpu limit of N`;
# * a request without a limit is rejected — `Limit must be set for non overcommitable resources`;
# * a limit without a request defaults the request to the limit.
#
# Write `gpu_request(requests, limits)` that returns the effective integer GPU request (0 if the
# container asks for none) or raises `ValueError` whose message contains the upstream text.

# %% exercise
def gpu_request(requests: dict, limits: dict) -> int:
    ### BEGIN SOLUTION
    req, lim = requests.get(GPU), limits.get(GPU)
    if req is None and lim is None:
        return 0
    for q in (req, lim):
        if q is not None and q != int(q):
            raise ValueError(f"{GPU}: must be an integer")
    if lim is None:
        raise ValueError("Limit must be set for non overcommitable resources")
    if req is not None and req != lim:
        raise ValueError(f"must be equal to {GPU} limit of {lim}")
    return int(lim)
    ### END SOLUTION

# %% check
assert gpu_request({"cpu": 4000}, {}) == 0
assert gpu_request({}, {GPU: 2}) == 2 == effective_requests({}, {GPU: 2})[GPU]
assert gpu_request({GPU: 4}, {GPU: 4}) == 4
for bad, text in [(({}, {GPU: 0.5}), "must be an integer"),
                  (({GPU: 1}, {GPU: 2}), "must be equal to nvidia.com/gpu limit of 2"),
                  (({GPU: 1}, {}), "Limit must be set")]:
    try:
        gpu_request(*bad)
        raise AssertionError(f"{bad} should be rejected")
    except ValueError as e:
        assert text in str(e), (bad, e)
print("✅ integers only, requests == limits: no fractional GPUs and no overcommit at this layer")

# %% [markdown]
# **Why no fractions?** The scheduler and kubelet only count. A GPU shared by several pods must
# therefore be *advertised as several units* by the device plugin (time-slicing, MPS) or *split
# into real partitions* (MIG, each partition its own device) — primer §9, with the mechanics in layer
# 02. Dynamic Resource Allocation (DRA, GA in Kubernetes 1.34) replaces the counter with a structured
# claim; here is the claim you would write instead of `limits: {nvidia.com/gpu: 1}`:

# %%
claim_template = {
    "apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaimTemplate",
    "metadata": {"name": "one-gpu"},
    "spec": {"spec": {"devices": {"requests": [{"name": "gpu", "exactly": {
        "deviceClassName": "gpu.nvidia.com",
        "selectors": [{"cel": {"expression":
            "device.attributes['gpu.nvidia.com'].productName.lowerAscii().matches('^.*h100.*$')"}}]}}]}}},
}
print(claim_template)

# %% [markdown]
# ## Labels: which GPU is this?
#
# The scheduler matches pods to GPU *types* with node labels. GKE sets
# `cloud.google.com/gke-accelerator` itself; on other clusters NVIDIA's GPU Feature Discovery (part
# of the GPU Operator) writes labels such as `nvidia.com/gpu.product` and `nvidia.com/gpu.memory`.
# Topology labels (block / sub-block / host) feed notebook 03.

# %%
fleet = Cluster([gpu_node("l4-0", 1, accelerator="nvidia-l4"), gpu_node("l4-1", 1, accelerator="nvidia-l4"),
                 gpu_node("h100-0", 8, topology=("b0", "b0-s0", "h100-0")),
                 Node("cpu-0", {"cpu": 32000, "memory": 131072})])
print(fleet.nodes["h100-0"].labels)
fleet.bind(gpu_pod("busy", 1), "l4-0")

# %% [markdown]
# ## Exercise 1.3 — where can this pod run?
#
# Write `where_can_it_run(pod, cluster)`: the names of nodes where the pod passes all three checks
# the scheduler's filters make — every `node_selector` label matches, every `NoSchedule`/`NoExecute`
# taint is tolerated, and every requested resource fits in `node.free(resource)`. Use your
# `tolerates()`.

# %% exercise
def where_can_it_run(pod: Pod, cluster: Cluster) -> list:
    names = []
    for node in cluster.nodes.values():
        ### BEGIN SOLUTION
        if any(node.labels.get(k) != v for k, v in pod.node_selector.items()):
            continue
        if any(t.effect in ("NoSchedule", "NoExecute") and not any(tolerates(tol, t) for tol in pod.tolerations)
               for t in node.taints):
            continue
        if any(q > node.free(r) for r, q in pod.requests.items() if q > 0):
            continue
        names.append(node.name)
        ### END SOLUTION
    return names

# %% check
l4_server = gpu_pod("l4-server", 1, node_selector={"cloud.google.com/gke-accelerator": "nvidia-l4"})
assert where_can_it_run(l4_server, fleet) == ["l4-1"]            # l4-0 is busy
assert where_can_it_run(gpu_pod("any-gpu", 1), fleet) == ["l4-1", "h100-0"]
assert where_can_it_run(gpu_pod("eight", 8), fleet) == ["h100-0"]
assert where_can_it_run(Pod("web", {"cpu": 500, "memory": 512}), fleet) == ["cpu-0"]
for p in (l4_server, gpu_pod("eight", 8)):
    assert where_can_it_run(p, fleet) == [n.name for n in fleet.nodes.values() if not run_filters(p, n)]
print("✅ selector, taints and integer resources decide feasibility; nothing else knows it is a GPU")

# %% [markdown]
# ## Exercise 1.4 — predict the node after a GPU fails
#
# Back to the first node: `trainer-0` holds 4 GPUs and `server-0` holds 2 (6 requested). Now the
# plugin's health check marks device `GPU-fake-0006` (one of the two free ones) **Unhealthy** — say
# an XID 79, "GPU has fallen off the bus". Predict, *before* running anything:
#
# * `capacity` and `allocatable` in the node status;
# * how many GPUs the scheduler now considers free on this node (allocatable − requested);
# * whether the kubelet can admit a new 2-GPU pod.

# %% exercise
# predicted_capacity = ...              an int
# predicted_allocatable = ...           an int
# predicted_free_for_scheduler = ...    an int
# predicted_two_gpu_pod_admitted = ...  True or False
### BEGIN SOLUTION
predicted_capacity = 8        # capacity counts every device the plugin reports
predicted_allocatable = 7     # allocatable counts healthy ones only
predicted_free_for_scheduler = 7 - 6
predicted_two_gpu_pod_admitted = False
### END SOLUTION

# %% check
plugin.set_health("GPU-fake-0006", UNHEALTHY)
status = kubelet.node_status()
assert (predicted_capacity, predicted_allocatable) == (status["capacity"][GPU], status["allocatable"][GPU])
assert predicted_free_for_scheduler == status["allocatable"][GPU] - 6
try:
    kubelet.admit("server-1", 2)
    admitted = True
except AdmissionError as e:
    admitted = False
    print("kubelet:", e)
assert predicted_two_gpu_pod_admitted == admitted
print("✅", status, "- running pods keep their GPUs; new work sees one fewer")

# %% [markdown]
# The scheduler would not *send* a 2-GPU pod here once the status updates. The rejection above
# happens in the window between the scheduler's decision and the kubelet's admission — the two are
# separate accountants, and the kubelet wins. The pod fails with reason `UnexpectedAdmissionError`
# and its controller must recreate it.
#
# ## Exercise 1.5 — keep a container's GPUs on one NVLink island
#
# Within a node, not all GPU pairs are equal: GPUs on the same NVLink island (or at least the same
# NUMA node) talk much faster. That is why the device-plugin API has `GetPreferredAllocation`.
# Implement the plugin side: given the free device IDs, a map `island_of[id]`, and a size, return
# `size` IDs from the **tightest** island that can hold them all (fewest free devices, ties by
# island number); if no single island can, return the first `size` free IDs.

# %% exercise
def preferred(free: list, island_of: dict, size: int) -> list:
    ### BEGIN SOLUTION
    groups = {}
    for d in free:
        groups.setdefault(island_of[d], []).append(d)
    fitting = [(len(ids), isl) for isl, ids in groups.items() if len(ids) >= size]
    if fitting:
        return groups[min(fitting)[1]][:size]
    return free[:size]
    ### END SOLUTION

# %% check
devs = make_gpus(8, island_size=2)                       # islands {0,1} {2,3} {4,5} {6,7}
island_of = {d.id: d.island for d in devs}
free = [d.id for d in devs if d.id not in ("GPU-fake-0000", "GPU-fake-0005")]
assert preferred(free, island_of, 2) == ["GPU-fake-0002", "GPU-fake-0003"]     # not 1 + 2 across islands
assert preferred(free, island_of, 1) == ["GPU-fake-0001"]                      # fill the broken island first
assert preferred(free, island_of, 3) == free[:3]
assert preferred(free, island_of, 2) == DevicePlugin(devs).get_preferred_allocation(free, [], 2)
print("✅ topology-aware allocation inside one node - notebook 03 does the same across nodes")

# %% [markdown]
# ## In a design review
#
# **Two-minute version.** "A GPU becomes schedulable through three hops. The NVIDIA device plugin —
# installed by the GPU Operator, or by GKE itself — registers `nvidia.com/gpu` with the kubelet and
# streams device health; the kubelet turns that into capacity and allocatable on the Node; the
# scheduler sees an integer and never overcommits it. Requests are whole GPUs with requests equal
# to limits. We taint GPU nodes so only GPU pods land there — the ExtendedResourceToleration
# plugin adds the toleration automatically — and select GPU types with node labels. If we need
# fractions we choose MIG for isolation or time-slicing for density, knowing time-slicing gives
# no memory or fault isolation. A GPU that goes unhealthy drops allocatable but does not evict
# running pods, so health alerts and node drains are part of the design."
#
# **Drill questions.**
#
# 1. *A pod asks for `nvidia.com/gpu: 0.5`. What happens?* — The API server rejects it: extended
#    resources must be integers. Share a GPU by advertising replicas (time-slicing/MPS) or MIG devices.
# 2. *`kubectl describe node` shows capacity 8, allocatable 7. What does that mean?* — The plugin
#    reports one device Unhealthy (e.g. an XID). Pods using it keep running; new pods see 7.
# 3. *Why do CPU-only pods never land on our GPU nodes, although nobody added tolerations anywhere?*
#    — GPU nodes carry a `nvidia.com/gpu` NoSchedule taint and only pods that *request* the
#    resource get the matching toleration, from the ExtendedResourceToleration admission plugin.
