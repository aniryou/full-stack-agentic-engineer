# %% [markdown]
# # 03 · Why is my pod Pending? Reading the control plane's evidence
#
# **Tier:** T0 — seventeen fixtures in the formats `kubectl get -o json` returns (pods, events,
# nodes, Kueue Workloads), labelled *illustrative*. If the kind cluster from notebook 02 is up,
# the last section creates the s6 "zoo" and diagnoses the real pods.
#
# ## The one-minute version
# A GPU pod passes up to four gates, and each leaves a different trace:
#
# | gate | where the reason lives | typical GPU causes |
# |---|---|---|
# | **Kueue** | the *Workload*'s `QuotaReserved` / admission checks; the Job is suspended (no pods) or its pods carry a `kueue.x-k8s.io/admission` gate | quota, topology, DWS capacity, preemption |
# | **kube-scheduler** | pod condition `PodScheduled=False, Unschedulable`: `0/N nodes are available: …` | GPU taint, wrong accelerator, too big, fragmented, busy |
# | **cluster autoscaler** | pod events `TriggeredScaleUp` / `NotTriggerScaleUp` / (GKE) `FailedScaleUp` | pool at max size, stockout, cloud quota |
# | **kubelet** | the pod is bound but not running: events and container states | volume mount (GCS FUSE), device allocation, probes killing a slow load |
#
# Read the gates in that order and you never debug the scheduler for a Kueue problem again.
# Primer §3 *The scheduling cycle*, §6 *Queues, quotas and multi-tenancy with Kueue*, §7
# *Getting capacity*, §8 *Startup latency*.

# %%
from k8sgpu import kindlab, pending

for name in pending.fixture_names():
    print(f"{name:28} {pending.load_fixture(name)['description']}")

# %% [markdown]
# ## The scheduler's message is a histogram
# The kube-scheduler runs its filters in a fixed order (NodeUnschedulable, NodeName,
# TaintToleration, NodeAffinity, NodePorts, NodeResourcesFit, … DynamicResources) and records,
# for each node, the **first** filter that rejected it. The message counts nodes per reason,
# sorted as strings; then it reports what preemption could do: *not helpful* (the failure is
# not about resources other pods hold) or *no victims* (nothing of lower priority to evict).

# %%
fx = pending.load_fixture("gpu-taint-not-tolerated")
msg = fx["pod"]["status"]["conditions"][0]["message"]
print(msg, "\n")
fe = pending.parse_fit_error(msg)
for reason, count in fe.reasons.items():
    cat, blocking, meaning = pending.classify_reason(reason)
    print(f"{count} x {reason}\n    -> {cat}{'' if blocking else ' (expected, ignore)'}: {meaning}")
print("\npreemption:", fe.preemption)

# %% [markdown]
# ## Exercise 3.1 — parse the histogram
# Write `histogram(message)` returning `{reason: node_count}` for the **filter** part only (before
# `" preemption: "`). Careful: reasons contain dots (`Insufficient nvidia.com/gpu`), so you cannot
# split on `"."`; the items are separated by `", "` and the part ends with a single `"."`.

# %% exercise
def histogram(message: str) -> dict[str, int]:
    ### BEGIN SOLUTION
    main = message.split(" preemption: ")[0]
    body = main.split("nodes are available: ", 1)[1].rstrip()
    if body.endswith("."):
        body = body[:-1]
    out = {}
    for item in body.split(", "):
        count, reason = item.split(" ", 1)
        out[reason] = out.get(reason, 0) + int(count)
    return out
    ### END SOLUTION

# %% check
for name in ("gpu-taint-not-tolerated", "wrong-accelerator", "fragmented-gpus", "all-gpus-busy"):
    m = pending.load_fixture(name)["pod"]["status"]["conditions"][0]["message"]
    assert histogram(m) == pending.parse_fit_error(m).reasons, name
assert histogram(pending.load_fixture("too-big-for-any-node")["pod"]["status"]["conditions"][0]["message"])[
    "Insufficient nvidia.com/gpu"] == 4
print("✅ histogram parses the kube-scheduler's FitError format")

# %% [markdown]
# ## Same message, different problems
# `too-big-for-any-node` and `fragmented-gpus` print the *same* filter histogram — `4
# Insufficient nvidia.com/gpu` — yet one can never be fixed by waiting and the other is a packing
# problem. Two tells: the preemption part (`Preemption is not helpful` means the request exceeds
# the node's **allocatable**, not just what is free) and the node inventory.

# %%
for name in ("too-big-for-any-node", "fragmented-gpus"):
    b = pending.load_fixture(name)
    print(name, "| preemption:", pending.parse_fit_error(b["pod"]["status"]["conditions"][0]["message"]).preemption)
    print("   free GPUs per node:", pending.free_gpus_per_node(b["nodes"], b["pods"]),
          "| request:", pending.pod_gpus(b["pod"]))

# %% [markdown]
# ## Exercise 3.2 — too big, fragmented, or just busy?
# Write `gpu_shortage(free_per_node, allocatable_per_node, request)` returning:
# * `"too-big"` if no node could ever hold `request` GPUs (compare with allocatable);
# * `"fragmented"` if the free GPUs add up to `request` or more but no single node has that many;
# * `"busy"` otherwise (the GPUs exist on some node shape but are in use).

# %% exercise
def gpu_shortage(free_per_node: dict[str, int], allocatable_per_node: dict[str, int], request: int) -> str:
    ### BEGIN SOLUTION
    if request > max(allocatable_per_node.values(), default=0):
        return "too-big"
    if sum(free_per_node.values()) >= request and max(free_per_node.values(), default=0) < request:
        return "fragmented"
    return "busy"
    ### END SOLUTION

# %% check
alloc = {"w2": 4, "w3": 4, "w4": 4, "w5": 4}
assert gpu_shortage({"w2": 1, "w3": 1, "w4": 1, "w5": 1}, alloc, 5) == "too-big"
assert gpu_shortage({"w2": 1, "w3": 1, "w4": 1, "w5": 1}, alloc, 2) == "fragmented"
assert gpu_shortage({"w2": 0, "w3": 0, "w4": 0, "w5": 0}, alloc, 1) == "busy"
b = pending.load_fixture("fragmented-gpus")
assert gpu_shortage(pending.free_gpus_per_node(b["nodes"], b["pods"]), alloc, pending.pod_gpus(b["pod"])) == "fragmented"
print("✅ the same '4 Insufficient nvidia.com/gpu' can mean three different fixes")

# %% [markdown]
# ## Gate 1: Kueue speaks through the Workload
# A queued Job has **no pods** until Kueue admits it, so `kubectl describe pod` shows nothing.
# `kubectl get workloads -n <ns>` and the `QuotaReserved` condition do. In Kueue v0.19 the
# condition's reason is `Pending` and the message comes from the flavor assigner or TAS:

# %%
for name in ("kueue-exceeds-max-quota", "kueue-waiting-for-quota", "kueue-topology-no-fit", "kueue-preempted",
             "gke-dws-waiting"):
    wl = pending.load_fixture(name)["workload"]
    conds = {c["type"]: c for c in wl["status"]["conditions"]}
    print(f"{name}:\n   QuotaReserved={conds['QuotaReserved']['status']}  {conds['QuotaReserved']['message'][:150]}")

# %% [markdown]
# ## Exercise 3.3 — what is Kueue waiting for?
# Write `kueue_blocker(workload)` returning one of `"preempted"`, `"admission-check"`,
# `"exceeds-max-quota"`, `"waiting-for-quota"`, `"topology"` from the Workload's conditions:
# a `Preempted=True` condition wins; `QuotaReserved=True` without `Admitted=True` means admission
# checks; otherwise read the `QuotaReserved` message (`maximum capacity` / `insufficient unused
# quota` / `topology … fit`).

# %% exercise
def kueue_blocker(workload: dict) -> str:
    conds = {c["type"]: c for c in workload["status"]["conditions"]}
    ### BEGIN SOLUTION
    if conds.get("Preempted", {}).get("status") == "True":
        return "preempted"
    if conds.get("QuotaReserved", {}).get("status") == "True":
        return "admission-check"
    msg = conds.get("QuotaReserved", {}).get("message", "")
    if "maximum capacity" in msg:
        return "exceeds-max-quota"
    if "insufficient unused quota" in msg:
        return "waiting-for-quota"
    return "topology"
    ### END SOLUTION

# %% check
for name in ("kueue-exceeds-max-quota", "kueue-waiting-for-quota", "kueue-topology-no-fit", "kueue-preempted",
             "gke-dws-waiting"):
    fx = pending.load_fixture(name)
    assert kueue_blocker(fx["workload"]) == fx["expect"]["category"], name
print("✅ five Kueue blockers, five different conversations with the queue owner")

# %% [markdown]
# ## The whole diagnosis, every fixture
# `pending.diagnose()` walks the four gates in order and names the fix. The autoscaler's events
# override the scheduler's verdict when they explain it (a pool at max size, a stockout). Each
# diagnosis is printed under its fixture's provenance: simulated by the predictor, or written by
# hand in the documented format — none of it is output recorded from a cluster.

# %%
for name in pending.fixture_names():
    fx = pending.load_fixture(name)
    print(name, pending.fixture_label(fx))
    print(pending.diagnose(fx), "\n")

# %% [markdown]
# ## Exercise 3.4 — pick the fix
# For each fixture, choose the fix that actually unblocks it:
#
# * **a** add `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]`
# * **b** split the pod across nodes (TP/PP over an LWS or JobSet) or add a bigger GPU shape
# * **c** raise the node pool's maximum size
# * **d** add fallbacks: other zones or shapes, on-demand, DWS flex-start, or a reservation
# * **e** add a startupProbe sized to the weight load
# * **f** grant the pod's ServiceAccount `roles/storage.objectViewer` on the bucket (Workload Identity)
# * **g** nothing is broken: wait for quota, or raise the job's priority / the queue's borrowingLimit
#
# Fill `fixes` with one letter per fixture.

# %% exercise
fixes = {
    "gpu-taint-not-tolerated": None,
    "too-big-for-any-node": None,
    "gke-autoscaler-max-size": None,
    "gke-stockout": None,
    "liveness-kills-model-load": None,
    "gcsfuse-permission-denied": None,
    "kueue-waiting-for-quota": None,
}
### BEGIN SOLUTION
fixes = {"gpu-taint-not-tolerated": "a", "too-big-for-any-node": "b", "gke-autoscaler-max-size": "c",
         "gke-stockout": "d", "liveness-kills-model-load": "e", "gcsfuse-permission-denied": "f",
         "kueue-waiting-for-quota": "g"}
### END SOLUTION

# %% check
key = {"gpu-taint": "a", "too-big": "b", "max-size": "c", "stockout": "d", "probe-kills-startup": "e",
       "volume-mount": "f", "waiting-for-quota": "g"}
for name, letter in fixes.items():
    assert letter == key[pending.diagnose(pending.load_fixture(name)).category], (name, letter)
print("✅ seven Pending pods, seven different fixes")

# %% [markdown]
# ## Live: the s6 zoo on your kind cluster
# With the cluster from notebook 02 up, this creates the zoo (fillers leave one free GPU per
# node, then six workloads that cannot start) and diagnoses every Pending pod with
# `pending.diagnose_live`, which gathers the same JSON documents with `kubectl`. Offline it
# prints the commands.

# %%
READY, why = kindlab.cluster_status()
if READY:
    kindlab.run_scenario("s6")
    k = kindlab.Kubectl(echo=lambda s: None)
    for ns in ("zoo", "team-a"):
        for p in k.get_json("pods", "-n", ns)["items"]:
            if p["status"].get("phase") == "Pending":
                print(p["metadata"]["name"], "->", pending.diagnose_live(p["metadata"]["name"], ns, k), "\n")
    print("Queued Jobs without pods: kubectl get workloads -n team-a  (then k8sgpu pending on the Workload JSON)")
else:
    print("no cluster (" + why + "); on a live cluster you would run:")
    for cmd in ("python -m k8sgpu kind run s6",
                "kubectl get pods -n zoo",
                "kubectl get pod <pod> -n zoo -o json > pod.json",
                "kubectl get events -n zoo --field-selector involvedObject.name=<pod> -o json > events.json",
                "python -m k8sgpu pending pod.json --events events.json",
                "python -m k8sgpu pending --live <pod> -n zoo",
                "kubectl get workloads -n team-a -o wide"):
        print("  $", cmd)

# %% [markdown]
# ## In a design review
# *"A training job has been Pending for an hour — how do you find out why?"* — in two minutes:
# first ask whether it is queued. If Kueue manages it, the Job is suspended and the answer is on
# the Workload: quota (wait, priority, borrowing limit), topology (no domain big enough), an
# admission check (DWS still looking for capacity) or a preemption. If the pods exist, read
# `PodScheduled`: the histogram says which filter rejected each node — taint, selector,
# resources — and the preemption line says whether waiting could ever help. Then the
# autoscaler's events: is a node coming, is the pool at its maximum, did the zone stock out?
# Only a *bound* pod that is not Running is a kubelet problem: images, volumes, device
# allocation, probes.
#
# **Drill 1.** *`0/6 nodes are available: 4 Insufficient nvidia.com/gpu` — scale up?* Not yet:
# check whether the request exceeds any node's allocatable (then no scale-up helps) or whether
# free GPUs are fragmented across nodes (then pack, don't buy).
#
# **Drill 2.** *The pod has a `kueue.x-k8s.io/admission` scheduling gate.* It belongs to a group
# (LWS, pod group) that Kueue has not admitted; look at that group's Workload.
#
# **Drill 3.** *`NotTriggerScaleUp: 1 max node group size reached`.* The only pool that can run
# the pod is full; raise its maximum or queue the work.
