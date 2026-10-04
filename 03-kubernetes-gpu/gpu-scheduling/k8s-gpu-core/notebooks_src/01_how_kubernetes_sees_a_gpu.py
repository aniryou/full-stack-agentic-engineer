# %% [markdown]
# # 01 · How Kubernetes sees a GPU
#
# **Tier:** T0. It is pure Python on a laptop, Colab CPU or CI, with no cluster and no GPU. The lab
# (`k8s-gpu-lab`) has the same objects on a real cluster: notebook `02_kind_with_fake_gpus_and_kueue`
# (kind with fake GPUs, T0 + Docker) and `04_gke_pools_dws_and_computeclasses` (GKE, T3).
#
# ## The one-minute version
#
# Kubernetes does not know what a GPU is. A **device plugin** on each node tells the kubelet "I manage
# `nvidia.com/gpu`; here are 8 device IDs and their health". The kubelet publishes **capacity** (all
# devices) and **allocatable** (healthy devices) in the Node status. The scheduler treats that number as
# an opaque integer. Thus a GPU request must be a whole number. It must equal its limit, and the
# scheduler never overcommits it.
#
# Ordinary **labels** and **taints** answer two other questions. They tell *which* GPU a node has (L4
# or H100), and *if* a pod can land on a GPU node at all. After this notebook, you can explain every
# GPU line of `kubectl describe node`. You can also explain why the kubelet can still reject a pod that
# the scheduler placed.
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
# The `options` in the Register request are the `DevicePluginOptions` of the plugin. This plugin answers
# `GetPreferredAllocation`, and it needs no `PreStartContainer` call.
#
# The scheduler never knows more than the last line: `nvidia.com/gpu: 8`. When a pod that asks for GPUs
# lands on the node, the kubelet selects free healthy device IDs and calls `Allocate`. Before it selects
# them, it asks the plugin for a *preferred* set. The NVIDIA plugin has a default `envvar` strategy, and
# with it the answer is one environment variable. At the creation of the container, the NVIDIA Container
# Toolkit reads this variable. Then it injects the device nodes and driver libraries (layer 02 explains
# that half).

# %%
print(kubelet.admit("trainer-0", 4))
print(kubelet.admit("server-0", 2))
print("devices pinned per container:", kubelet.assigned)

# %% [markdown]
# ## Exercise 1.1 — taints and tolerations
#
# GPU nodes have a taint, so that ordinary pods stay off high-cost nodes. GKE uses
# `nvidia.com/gpu=present:NoSchedule` (verify). A pod can land on a tainted node only if one of its
# tolerations *tolerates* the taint. Write the upstream rule as `tolerates(tol, taint)`:
#
# 1. If the toleration names an effect, it must equal the effect of the taint.
# 2. If the toleration names a key, it must equal the key of the taint. An empty key matches every key.
# 3. Then the operator `Exists` matches any value. The operator `Equal` (or an empty operator) needs
#    equal values.

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
# When the **ExtendedResourceToleration** admission plugin is on, nobody writes that toleration by hand.
# The plugin adds `{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}` to every pod that
# *requests* `nvidia.com/gpu`. The function `gpu_pod()` in this package does the same.
#
# The plugin is **off by default** in kube-apiserver. GKE turns it on (verify). On kubeadm or kind, you
# turn it on (`--enable-admission-plugins=...,ExtendedResourceToleration`), or you add the toleration to
# every GPU pod yourself. If you do neither, GPU pods stay Pending with `untolerated taint(s)`. With the
# plugin on, the taint does exactly one job. It keeps pods that do **not** ask for GPUs off GPU nodes.

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
# `nvidia.com/gpu` is an *extended resource*, that is, a name with a domain prefix outside
# `kubernetes.io`. For these resources, the API server applies three rules and one default:
#
# * The quantity must be an integer. The message is `must be an integer`.
# * If a container sets both a request and a limit, they must be equal. The message is
#   `must be equal to nvidia.com/gpu limit of N`.
# * The API server rejects a request without a limit. The message is
#   `Limit must be set for non overcommitable resources`.
# * If a container sets a limit and no request, the request gets the value of the limit by default.
#
# Write `gpu_request(requests, limits)`. It returns the effective GPU request as an integer, or 0 if the
# container asks for none. When a rule rejects the request, the function raises `ValueError`, and the
# message contains the upstream text.

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
# **Why no fractions?** The scheduler and the kubelet only count. Thus, to share a GPU between several
# pods, the device plugin must *advertise it as several units* (time-slicing, MPS). Another possibility
# is to *divide it into real partitions* (MIG, where each partition is its own device). See primer §9,
# and layer 02 for the mechanics.
#
# Dynamic Resource Allocation (DRA, GA in Kubernetes 1.34) replaces the counter with a structured claim.
# With DRA, you write this claim instead of `limits: {nvidia.com/gpu: 1}`:

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
# `cloud.google.com/gke-accelerator` itself. On other clusters, NVIDIA's GPU Feature Discovery (part of
# the GPU Operator) writes labels such as `nvidia.com/gpu.product` and `nvidia.com/gpu.memory`. Notebook
# 03 uses the topology labels (block / sub-block / host).

# %%
fleet = Cluster([gpu_node("l4-0", 1, accelerator="nvidia-l4"), gpu_node("l4-1", 1, accelerator="nvidia-l4"),
                 gpu_node("h100-0", 8, topology=("b0", "b0-s0", "h100-0")),
                 Node("cpu-0", {"cpu": 32000, "memory": 131072})])
print(fleet.nodes["h100-0"].labels)
fleet.bind(gpu_pod("busy", 1), "l4-0")

# %% [markdown]
# ## Exercise 1.3 — where can this pod run?
#
# Write `where_can_it_run(pod, cluster)`. It returns the names of the nodes where the pod passes all
# three checks that the filters of the scheduler make:
#
# * Every `node_selector` label matches.
# * The pod tolerates every `NoSchedule`/`NoExecute` taint.
# * Every requested resource fits in `node.free(resource)`.
#
# Use your `tolerates()`.

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
# Go back to the first node. `trainer-0` holds 4 GPUs and `server-0` holds 2 (6 requested). Now the
# health check of the plugin marks device `GPU-fake-0006` (one of the two free devices) **Unhealthy**.
# For example, the cause is an XID 79, "GPU has fallen off the bus". Predict these values *before* you
# run anything:
#
# * `capacity` and `allocatable` in the node status.
# * How many GPUs the scheduler now counts as free on this node
#   ($\mathit{allocatable} - \mathit{requested}$).
# * If the kubelet can admit a new 2-GPU pod.

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
# After the status update, the scheduler does not *send* a 2-GPU pod here. The rejection in the previous
# cell occurs in the window between the decision of the scheduler and the admission of the kubelet. The
# two keep separate accounts, and the decision of the kubelet is final. The pod fails with the reason
# `UnexpectedAdmissionError`, and its controller must create it again.
#
# ## Exercise 1.5 — keep a container's GPUs on one NVLink island
#
# In a node, not all GPU pairs are equal. GPUs on the same NVLink island (or at least on the same NUMA
# node) exchange data much faster. This is why the device-plugin API has `GetPreferredAllocation`. Write
# the plugin side. The inputs are the free device IDs, a map `island_of[id]`, and a size. Return `size`
# IDs from the **tightest** island that can hold them all.
#
# The tightest island has the fewest free devices, and the island number breaks a tie. If no single
# island can hold them, return the first `size` free IDs.

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
# ## Sharing: time-slicing advertises more integers, not more GPUs
#
# The scheduler only counts. Thus, to share a GPU, the device plugin must advertise *more units*. For
# time-slicing, you configure the NVIDIA device plugin with `replicas: N`. Then `ListAndWatch` lists
# every physical GPU $N$ times (IDs `<uuid>::0 ... ::N-1`), and allocatable increases $N$-fold.
#
# The preferred allocation of the plugin takes replicas one at a time from the physical GPU with the
# fewest replicas already allocated. This is the default *distributed* policy of the plugin. A replica
# has no memory limit and no guaranteed share of compute. The GPU divides its time equally between all the processes
# on it.

# %%
shared_plugin = DevicePlugin(make_gpus(8), replicas=10)
shared_kubelet = Kubelet()
shared_kubelet.register(shared_plugin)
print("first IDs:", [d["ID"] for d in shared_plugin.list_and_watch()[:3]], "...")
print("node status:", shared_kubelet.node_status())

# %% [markdown]
# ## Exercise 1.6 — what does a time-sliced request really get?
#
# A node has 8 GPUs and `replicas: 10`. Predict these values:
#
# * `allocatable` for `nvidia.com/gpu`.
# * On the **fresh** node, how many **physical** GPUs supply a container that requests
#   `nvidia.com/gpu: 2`.
# * The same number after the kubelet admits sixteen 1-replica pods (the plugin spreads them, two per
#   GPU) and the two pods on `GPU-fake-0003` finish.
# * If the kubelet admits a 2-"GPU" container when the plugin sets `failRequestsGreaterThanOne: true`.
#   This NVIDIA option treats a shared request as "access to a GPU", so a request for more than one is
#   an error.

# %% exercise
# predicted_shared_allocatable = ...     an int
# predicted_physical_fresh = ...         an int
# predicted_physical_loaded = ...        an int
# predicted_admitted_with_limit = ...    True or False
### BEGIN SOLUTION
predicted_shared_allocatable = 8 * 10          # every GPU advertised ten times
predicted_physical_fresh = 2                   # least-loaded first: GPU 0, then GPU 1 (now fewer than GPU 0)
predicted_physical_loaded = 1                  # GPU 3 holds 0 replicas, the rest 2: both picks land on GPU 3
predicted_admitted_with_limit = False          # more than one shared replica is refused at Allocate
### END SOLUTION

# %% check
def physical_gpus(kubelet, pod, count):
    visible = kubelet.admit(pod, count)["envs"]["NVIDIA_VISIBLE_DEVICES"]
    print(f"{pod}: NVIDIA_VISIBLE_DEVICES = {visible}")
    return len(visible.split(","))

fresh_kubelet = Kubelet()
fresh_kubelet.register(DevicePlugin(make_gpus(8), replicas=10))
assert predicted_shared_allocatable == fresh_kubelet.node_status()["allocatable"][GPU]
assert predicted_physical_fresh == physical_gpus(fresh_kubelet, "fresh-two", 2)
loaded_kubelet = Kubelet()
loaded_kubelet.register(DevicePlugin(make_gpus(8), replicas=10))
for i in range(16):
    loaded_kubelet.admit(f"small-{i}", 1)
for pod in [p for p, ids in loaded_kubelet.assigned.items() if ids[0].startswith("GPU-fake-0003")]:
    del loaded_kubelet.assigned[pod]                                  # the pods on GPU 3 finish
assert predicted_physical_loaded == physical_gpus(loaded_kubelet, "loaded-two", 2)
strict_kubelet = Kubelet()
strict_kubelet.register(DevicePlugin(make_gpus(8), replicas=10, fail_requests_greater_than_one=True))
try:
    strict_kubelet.admit("wants-two", 2)
    admitted_with_limit = True
except AdmissionError as e:
    admitted_with_limit = False
    print("kubelet:", e)
assert predicted_admitted_with_limit == admitted_with_limit
print("✅ time-slicing multiplies the integer, not the hardware")

# %% [markdown]
# That was the default `distributed` policy of the plugin. The `--shared-devices-allocation-policy` flag
# of the NVIDIA plugin also offers `packed` (v0.20.0 and later). On the main branch of the plugin, the
# flag also offers `spread` (verify, and see primer §9). The model takes the same names as
# `DevicePlugin(allocation_policy=...)`. The next cell sends the same 2-"GPU" request to a fresh node
# under each policy:

# %%
for policy in ("distributed", "packed", "spread"):
    k = Kubelet()
    k.register(DevicePlugin(make_gpus(8), replicas=10, allocation_policy=policy))
    print(f"{policy:12} -> {k.admit('two', 2)['envs']['NVIDIA_VISIBLE_DEVICES']}")

# %% [markdown]
# GKE has its own device plugin, and it applies the same rule on time-sharing nodes. There, a container
# can request at most one `nvidia.com/gpu`. MIG is the other way to share a GPU. It makes real
# partitions, each its own device, with memory and fault isolation (primer §9, and layer 02 for the
# mechanics).
#
# ## In a design review
#
# **Two-minute version.** "A GPU becomes schedulable through three hops. First, the NVIDIA device plugin
# registers `nvidia.com/gpu` with the kubelet and sends the device health as a stream. The GPU Operator
# installs the plugin, or GKE itself installs it. Then the kubelet turns the registration and the
# device health into capacity and allocatable on the Node. Last, the scheduler sees an integer and
# never overcommits it.
#
# "Requests are whole GPUs, and requests are equal to limits. We put a taint on GPU nodes, so only GPU
# pods land there. The ExtendedResourceToleration admission plugin adds the toleration to GPU pods. It
# is on in GKE, and we turn it on in self-built clusters. We select GPU types with node labels.
#
# "If we need fractions, we select MIG for isolation or time-slicing for density. We know that
# time-slicing gives no memory or fault isolation. We also know that two slices can be one GPU. A GPU
# that becomes unhealthy decreases allocatable, but it does not evict the pods that run. Thus health
# alerts and node drains are part of the design."
#
# **Drill questions.**
#
# 1. *A pod asks for `nvidia.com/gpu: 0.5`. What happens?* The API server rejects it, because extended
#    resources must be integers. To share a GPU, advertise replicas (time-slicing/MPS) or MIG devices.
# 2. *`kubectl describe node` shows capacity 8, allocatable 7. What does that mean?* The plugin reports
#    one device Unhealthy (for example, after an XID). The pods that use it continue to run. New pods
#    see 7.
# 3. *On GKE, why do CPU-only pods never land on our GPU nodes, although nobody wrote tolerations?* GPU
#    nodes have a `nvidia.com/gpu` NoSchedule taint. Only pods that *request* the resource get the
#    toleration that matches, from the ExtendedResourceToleration admission plugin. That plugin is off
#    by default upstream. Thus, on a self-built cluster, turn it on or write the toleration.
# 4. *A team asks for two time-sliced GPUs to "get twice the compute". What do they get?* They get two
#    replicas that can be slices of the same physical GPU, with no guaranteed share of it. Set
#    `failRequestsGreaterThanOne` (or use GKE time-sharing, which permits one), so that the request
#    fails with a visible error.
