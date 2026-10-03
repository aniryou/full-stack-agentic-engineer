# %% [markdown]
# # 03 · Why is my pod Pending? Reading the control plane's evidence
#
# **Tier:** T0. The notebook uses seventeen fixtures in the formats that `kubectl get -o json` returns
# (pods, events, nodes, Kueue Workloads), labelled *illustrative*. If the kind cluster from notebook 02 is
# up, the last section creates the s6 "zoo" and diagnoses the real pods.
#
# ## The one-minute version
# A GPU pod goes through up to four gates. Each gate leaves a different trace:
#
# | gate | where the reason lives | typical GPU causes |
# |---|---|---|
# | **Kueue** | the *Workload*'s `QuotaReserved` / admission checks. The Job is suspended (no pods), or its pods have a `kueue.x-k8s.io/admission` gate | quota, topology, DWS capacity, preemption |
# | **kube-scheduler** | pod condition `PodScheduled=False, Unschedulable`: `0/N nodes are available: …` | GPU taint, incorrect accelerator, too large, fragmented, busy |
# | **cluster autoscaler** | pod events `TriggeredScaleUp` / `NotTriggerScaleUp` / (GKE) `FailedScaleUp` | pool at max size, stockout, cloud quota |
# | **kubelet** | the pod is bound but does not run: events and container states | volume mount (GCS FUSE), device allocation, probes that stop a slow load |
#
# Read the gates in that order. Then you never again debug the scheduler for a Kueue problem.
# Primer §3 *The scheduling cycle*, §6 *Queues, quotas and multi-tenancy with Kueue*, §7
# *Getting capacity*, §8 *Startup latency*.

# %%
from k8sgpu import kindlab, pending

for name in pending.fixture_names():
    print(f"{name:28} {pending.load_fixture(name)['description']}")

# %% [markdown]
# ## The scheduler's message is a histogram
# The kube-scheduler always runs its filters in the same order (NodeUnschedulable, NodeName, TaintToleration,
# NodeAffinity, NodePorts, NodeResourcesFit, … DynamicResources). For each node, it records the **first**
# filter that rejected the node. The message counts nodes per reason, sorted as strings.
#
# Then the message tells what preemption can do. One result is *not helpful*: the failure is not about
# resources that other pods hold. The other result is *no victims*: there is nothing of lower priority to evict.

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
# Write `histogram(message)`. It returns `{reason: node_count}` for the **filter** part only (before
# `" preemption: "`). Be careful: reasons contain dots (`Insufficient nvidia.com/gpu`). Thus you cannot
# split on `"."`. A `", "` separates the items, and the part ends with a single `"."`.

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
# `too-big-for-any-node` and `fragmented-gpus` print the *same* filter histogram:
# `4 Insufficient nvidia.com/gpu`. But a wait can never repair the first one, and the second one is a
# packing problem. Two signs tell them apart. The first sign is the preemption part. `Preemption is not helpful`
# means that the request is more than the **allocatable** of the node, not only more than the free GPUs.
# The second sign is the node inventory.

# %%
for name in ("too-big-for-any-node", "fragmented-gpus"):
    b = pending.load_fixture(name)
    print(name, "| preemption:", pending.parse_fit_error(b["pod"]["status"]["conditions"][0]["message"]).preemption)
    print("   free GPUs per node:", pending.free_gpus_per_node(b["nodes"], b["pods"]),
          "| request:", pending.pod_gpus(b["pod"]))

# %% [markdown]
# ## Exercise 3.2 — too big, fragmented, or just busy?
# Write `gpu_shortage(free_per_node, allocatable_per_node, request)`. It returns:
# * `"too-big"` if no node can ever hold `request` GPUs. Compare with allocatable.
# * `"fragmented"` if the free GPUs add up to `request` or more, but no single node has that many.
# * `"busy"` in all other cases. The GPUs exist on some node shape, but they are in use.

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
# A queued Job has **no pods** until Kueue admits it. Thus `kubectl describe pod` shows nothing.
# `kubectl get workloads -n <ns>` and the `QuotaReserved` condition show the reason. In Kueue v0.19, the
# reason of the condition is `Pending`. The message comes from the flavor assigner or from TAS:

# %%
for name in ("kueue-exceeds-max-quota", "kueue-waiting-for-quota", "kueue-topology-no-fit", "kueue-preempted",
             "gke-dws-waiting"):
    wl = pending.load_fixture(name)["workload"]
    conds = {c["type"]: c for c in wl["status"]["conditions"]}
    print(f"{name}:\n   QuotaReserved={conds['QuotaReserved']['status']}  {conds['QuotaReserved']['message'][:150]}")

# %% [markdown]
# ## Exercise 3.3 — what is Kueue waiting for?
# Write `kueue_blocker(workload)`. From the conditions of the Workload, it returns one of `"preempted"`,
# `"admission-check"`, `"exceeds-max-quota"`, `"waiting-for-quota"`, `"topology"`. Use these rules:
#
# * A `Preempted=True` condition wins.
# * `QuotaReserved=True` without `Admitted=True` means admission checks.
# * In all other cases, read the `QuotaReserved` message (`maximum capacity` /
#   `insufficient unused quota` / `topology … fit`).

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
# `pending.diagnose()` goes through the four gates in order and gives the repair. The events of the
# autoscaler override the result of the scheduler when they explain it (a pool at max size, a stockout). The
# next cell prints each diagnosis under the provenance of its fixture. The predictor simulated the fixture,
# or a person wrote it by hand in the documented format. None of it is output recorded from a cluster.

# %%
for name in pending.fixture_names():
    fx = pending.load_fixture(name)
    print(name, pending.fixture_label(fx))
    print(pending.diagnose(fx), "\n")

# %% [markdown]
# ## Exercise 3.4 — pick the fix
# For each fixture, select the repair that really unblocks it:
#
# * **a** add `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]`
# * **b** divide the pod across nodes (TP/PP over an LWS or JobSet), or add a larger GPU shape
# * **c** increase the maximum size of the node pool
# * **d** add fallbacks: other zones or shapes, on-demand, DWS flex-start, or a reservation
# * **e** add a startupProbe sized to the weight load
# * **f** grant the pod's ServiceAccount `roles/storage.objectViewer` on the bucket (Workload Identity)
# * **g** nothing is broken. Wait for quota, or increase the priority of the job / the borrowingLimit of the queue
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
# If the cluster from notebook 02 is up, the next cell creates the zoo. Fillers leave one free GPU per node,
# then six workloads that cannot start arrive. Then the cell diagnoses every Pending pod with
# `pending.diagnose_live`. This function gets the same JSON documents with `kubectl`. Offline, the cell
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
# *"A training job has been Pending for an hour — how do you find out why?"* The answer, in two minutes:
#
# * First, ask if the job is queued. If Kueue manages it, the Job is suspended, and the answer is on the
#   Workload. The cause is quota (wait, priority, borrowing limit), topology (no domain is sufficiently
#   large), an admission check (DWS still looks for capacity) or a preemption.
# * If the pods exist, read `PodScheduled`. The histogram tells which filter rejected each node: taint,
#   selector or resources. The preemption line tells if a wait can ever help.
# * Then read the events of the autoscaler. Is a node on its way? Is the pool at its maximum? Did the zone
#   have a stockout?
# * Only a *bound* pod that is not Running is a kubelet problem: images, volumes, device allocation, probes.
#
# **Drill 1.** *`0/6 nodes are available: 4 Insufficient nvidia.com/gpu`: do you scale up?* Not yet.
# First, find out if the request is more than the allocatable of any node (then no scale-up helps).
# Also find out if the free GPUs are fragmented across nodes (then pack, do not buy).
#
# **Drill 2.** *The pod has a `kueue.x-k8s.io/admission` scheduling gate.* The pod is part of a group
# (LWS, pod group) that Kueue did not admit yet. Look at the Workload of that group.
#
# **Drill 3.** *`NotTriggerScaleUp: 1 max node group size reached`.* The only pool that can run the pod is
# full. Increase its maximum, or put the work in a queue.
