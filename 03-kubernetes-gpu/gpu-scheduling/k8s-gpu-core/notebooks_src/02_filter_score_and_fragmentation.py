# %% [markdown]
# # 02 · Filter, score and fragmentation
#
# **Tier:** T0 — a pure-Python scheduler that uses kube-scheduler's plugin names, formulas and
# messages. Reading real `FailedScheduling` events is lab notebook `03_why_is_my_pod_pending`.
#
# ## The one-minute version
#
# kube-scheduler places **one pod at a time**: sort the queue by priority, **filter** out nodes
# that cannot run the pod (each with a reason), **score** the rest, bind to the best. When no node
# passes, it tries **preemption** of lower-priority pods and otherwise emits
# `0/N nodes are available: ...`. The score step decides GPU **fragmentation**: the default
# `NodeResourcesFit` strategy is `LeastAllocated` over **cpu and memory** — it spreads pods and
# never looks at GPUs — so a stream of 1-GPU pods lands one per node and an 8-GPU pod then fits
# nowhere although half the GPUs are free. `MostAllocated` with a GPU weight packs instead. After
# this notebook you can read a FailedScheduling message, compute a node score by hand, measure
# stranded GPUs, and say which scoring strategy a GPU cluster wants and why.
#
# Primer: §3 *The scheduling cycle* in `../../PRIMER.md`.

# %%
from gpusched import (GPU, Cluster, Node, Pod, Scheduler, fit_error, fragmentation, gpu_node, gpu_pod,
                      least_allocated, make_cluster, most_allocated, run_filters, select_victims, stranded_gpus)

cluster = make_cluster(hosts=4)                          # 4 nodes x 8 GPUs, cpu 96 cores, 768 GiB each
sched = Scheduler(cluster)                               # upstream defaults: LeastAllocated on cpu+memory
d = sched.schedule_one(gpu_pod("first", 1))
print("scores (0-100, higher wins):", d.scores, "->", d.node)
d = sched.schedule_one(gpu_pod("second", 1))
print("scores:", d.scores, "->", d.node, "  (the loaded node now scores lower: spread)")

# %% [markdown]
# ## A stream of small pods, then a big one
#
# Sixteen 1-GPU inference pods arrive (8 cores and 64 GiB each), then one 8-GPU training pod.

# %%
spread = make_cluster(hosts=4)
s = Scheduler(spread)
s.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(16)])
s.run()
print(spread.show())
big = s.schedule_one(gpu_pod("train", 8))
print("\ntrain ->", big.node, "|", big.message)
print("free GPUs:", spread.free(), "| stranded for 8-GPU pods:", stranded_gpus(spread, 8))

# %% [markdown]
# Sixteen GPUs are free and an 8-GPU pod still cannot run: every node has 4 free, and a pod cannot
# span nodes. The GPUs are **stranded** — free, but in pieces too small for the pending shape.
#
# ## Exercise 2.1 — the MostAllocated score
#
# kube-scheduler's `MostAllocated` strategy scores a node as a weighted average over the configured
# resources of `requested * 100 // allocatable`, where *requested* already includes the incoming
# pod, in integer arithmetic:
#
# `score = sum(weight_r * (min(requested_r, alloc_r) * 100 // alloc_r)) // sum(weight_r)`
#
# Write `most_allocated_score(rows)` where `rows` is a list of `(requested_after, allocatable,
# weight)` tuples.

# %% exercise
def most_allocated_score(rows: list) -> int:
    ### BEGIN SOLUTION
    if not rows:
        return 0
    total = sum(w * (min(r, a) * 100 // a) for r, a, w in rows)
    return total // sum(w for _, _, w in rows)
    ### END SOLUTION

# %% check
assert most_allocated_score([(7, 8, 1)]) == 87                  # 6 GPUs busy + 1 incoming, of 8
assert most_allocated_score([(7, 8, 5), (16000, 96000, 1)]) == (5 * 87 + 16) // 6 == 75
node = gpu_node("n", 8)
node.pods.append(gpu_pod("busy", 6))
assert most_allocated_score([(7, 8, 1)]) == most_allocated(gpu_pod("new", 1), node, ((GPU, 1),))
print("✅ score = weighted, integer, and includes the incoming pod")

# %% [markdown]
# Now pack instead of spread: score with `MostAllocated` and give the GPU a weight. In a real
# cluster that is a `KubeSchedulerConfiguration` profile (a scheduler you run yourself, or a second
# profile selected with `schedulerName`):
#
# ```yaml
# apiVersion: kubescheduler.config.k8s.io/v1
# kind: KubeSchedulerConfiguration
# profiles:
# - schedulerName: gpu-binpack
#   pluginConfig:
#   - name: NodeResourcesFit
#     args:
#       scoringStrategy:
#         type: MostAllocated
#         resources: [{name: cpu, weight: 1}, {name: memory, weight: 1}, {name: nvidia.com/gpu, weight: 5}]
# ```

# %%
packed = make_cluster(hosts=4)
p = Scheduler(packed, strategy="MostAllocated", resources=(("cpu", 1), ("memory", 1), (GPU, 5)))
p.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(16)])
p.run()
print(packed.show())
print("\ntrain ->", p.schedule_one(gpu_pod("train", 8)).node)

# %% [markdown]
# ## Exercise 2.2 — measure fragmentation
#
# Define **stranded GPUs** for a pod shape of `k` GPUs as the free GPUs that cannot host one more
# such pod: on each node, `free mod k`. Write `stranded(free_per_node, k)` and `frag(free_per_node,
# k)` = stranded / total free (0.0 when nothing is free).

# %% exercise
def stranded(free_per_node: list, k: int) -> int:
    ### BEGIN SOLUTION
    return sum(f % k for f in free_per_node)
    ### END SOLUTION

def frag(free_per_node: list, k: int) -> float:
    ### BEGIN SOLUTION
    total = sum(free_per_node)
    return stranded(free_per_node, k) / total if total else 0.0
    ### END SOLUTION

# %% check
assert stranded([4, 4, 4, 4], 8) == 16 and frag([4, 4, 4, 4], 8) == 1.0
assert stranded([4, 4, 4, 4], 4) == 0 and frag([0, 0, 0], 8) == 0.0
assert stranded([7, 3, 8], 4) == 3 + 3 + 0
free_now = [n.free() for n in spread.nodes.values()]
assert stranded(free_now, 8) == stranded_gpus(spread, 8) and frag(free_now, 8) == fragmentation(spread, 8)
print("✅ spread cluster:", free_now, "-> stranded", stranded(free_now, 8), "of", sum(free_now))

# %% [markdown]
# ## Exercise 2.3 — pick the scoring strategy for a mixed GPU cluster
#
# A cluster of four 8-GPU nodes serves twelve 1-GPU inference replicas and must also start two
# 8-GPU fine-tuning pods that arrive afterwards. Choose `strategy` (`"LeastAllocated"` or
# `"MostAllocated"`) and the `resources` weights for `NodeResourcesFit`, and write one sentence in
# `why`. The check replays the arrivals with your choice.

# %% exercise
### BEGIN SOLUTION
strategy = "MostAllocated"
resources = (("cpu", 1), ("memory", 1), (GPU, 5))
why = ("Pack small GPU pods onto as few nodes as possible so whole nodes stay free for 8-GPU pods; "
       "weight the GPU because it is the scarce resource and the default score ignores it.")
### END SOLUTION

# %% check
mixed = make_cluster(hosts=4)
m = Scheduler(mixed, strategy=strategy, resources=resources)
m.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(12)])
m.run()
placed = [m.schedule_one(gpu_pod(f"tune-{i}", 8)).node for i in range(2)]
assert all(placed), f"a fine-tuning pod stayed Pending: {placed}\n{mixed.show()}"
assert isinstance(why, str) and len(why) > 20
print(mixed.show())
print("✅ both 8-GPU pods placed; GPUs still stranded for 8-GPU pods:", stranded_gpus(mixed, 8))

# %% [markdown]
# Packing has a price, which belongs in the same review: replicas of one service pile onto one node
# (one node failure takes several down — add `topologySpreadConstraints` for those), and other
# default score plugins (`NodeResourcesBalancedAllocation`, the default `PodTopologySpread`
# constraints for Deployment pods) pull toward spreading, so measure the result.
#
# ## Reading FailedScheduling
#
# Each node fails at the *first* filter that rejects it; the message is a histogram of those
# reasons, sorted as strings.

# %%
shop = Cluster([gpu_node("h100-0", 8), gpu_node("h100-1", 8, accelerator="nvidia-h100-80gb"),
                gpu_node("l4-0", 1, accelerator="nvidia-l4"), Node("cpu-0", {"cpu": 32000, "memory": 131072})])
shop.nodes["h100-1"].unschedulable = True            # cordoned for maintenance
shop.bind(gpu_pod("resident", 4), "h100-0")
pod = gpu_pod("tp8", 8, node_selector={"cloud.google.com/gke-accelerator": "nvidia-h100-80gb"})
for n in shop.nodes.values():
    print(f"{n.name:<8}", run_filters(pod, n))

# %% [markdown]
# ## Exercise 2.4 — predict the event text
#
# Write the exact message kube-scheduler (and `Scheduler.schedule_one`) produces for `tp8` in the
# `shop` cluster above, *without preemption* — i.e. the text before any `preemption:` suffix. The
# format is `0/<nodes> nodes are available: <count> <reason>, ... .` with the `"<count> <reason>"`
# strings sorted.

# %% exercise
### BEGIN SOLUTION
predicted = ("0/4 nodes are available: 1 Insufficient nvidia.com/gpu, 1 node(s) were unschedulable, "
             "2 node(s) didn't match Pod's node affinity/selector.")
### END SOLUTION

# %% check
actual = fit_error(4, {n.name: run_filters(pod, n) for n in shop.nodes.values()})
assert predicted == actual, actual
print("✅", actual)

# %% [markdown]
# Why does `cpu-0` say *affinity/selector* and not *Insufficient nvidia.com/gpu*? Filters run in
# order — unschedulable, taints, node affinity, then resources — and the first failure is the one
# reported. The fix differs per reason: *Insufficient* means capacity (wait, preempt, scale up);
# *affinity/selector* and *taint* mean the pod asked for a node shape that does not exist here.
#
# ## Priority and preemption
#
# Pods carry a priority (from a `PriorityClass`). When no node passes the filters, the
# `DefaultPreemption` plugin looks for nodes where evicting **lower-priority** pods would make room,
# evicts as few and as unimportant pods as it can, and binds the preemptor there.

# %%
c = Cluster([gpu_node("a", 8), gpu_node("b", 8)])
c.bind(gpu_pod("batch-old", 4, priority=0), "a")
c.bind(gpu_pod("batch-new", 4, priority=0), "a")
c.bind(gpu_pod("eval", 8, priority=500), "b")
print(c.show())
d = Scheduler(c).schedule_one(gpu_pod("serve", 4, priority=1000))
print("\n", d.node, d.message)

# %% [markdown]
# ## Exercise 2.5 — choose the victims on one node
#
# Implement the per-node step of `DefaultPreemption`: remove **all** pods with lower priority than
# the preemptor; if the preemptor still does not fit, return `None`. Otherwise add them back one at
# a time, *most important first* (higher priority, then earlier `started`), and keep out only those
# the preemptor cannot fit without. Use `fits(node)` below, which runs the filters.

# %% exercise
def victims_on(node: Node, preemptor: Pod) -> list | None:
    fits = lambda n: not run_filters(preemptor, n)
    original = list(node.pods)
    try:
        ### BEGIN SOLUTION
        lower = [p for p in original if p.priority < preemptor.priority]
        if not lower:
            return None
        node.pods[:] = [p for p in original if p not in lower]
        if not fits(node):
            return None
        victims = []
        for p in sorted(lower, key=lambda p: (-p.priority, p.started)):
            node.pods.append(p)
            if not fits(node):
                node.pods.remove(p)
                victims.append(p)
        return victims
        ### END SOLUTION
    finally:
        node.pods[:] = original

# %% check
x = Cluster([gpu_node("x", 8)])
for name, g, prio in [("old-low", 4, 0), ("mid", 2, 50), ("new-low", 2, 0)]:
    x.bind(gpu_pod(name, g, priority=prio), "x")
want = gpu_pod("want", 4, priority=100)
got = victims_on(x.nodes["x"], want)
assert [v.name for v in got] == [v.name for v in select_victims(want, x.nodes["x"])]
assert [v.name for v in got] == ["old-low"]       # mid is reprieved first, then old-low does not fit back
assert victims_on(x.nodes["x"], gpu_pod("low", 4, priority=0)) is None        # nobody below priority 0
assert len(x.nodes["x"].pods) == 3                                             # nothing was really evicted
print("✅ victims:", [v.name for v in got], "- reprieve in order of importance, evict what does not fit back")

# %% [markdown]
# Notice what preemption does *not* do: it never evicts a higher-priority pod, never moves pods to
# consolidate free GPUs (no defragmentation), and it looks at one pod at a time — preempting for
# the first member of an 8-pod training job gains nothing if the other seven cannot follow.
# Notebook 03 is about that last problem.
#
# ## In a design review
#
# **Two-minute version.** "The scheduler filters, scores and binds one pod at a time. For GPUs the
# default score is wrong for us: it is LeastAllocated over cpu and memory, so it spreads 1-GPU pods
# across every node and strands GPUs in pieces too small for our 8-GPU jobs. We run a bin-packing
# profile — MostAllocated with the GPU weighted — for GPU node pools, keep inference replicas apart
# with topology spread constraints where availability matters, and track stranded GPUs per pod
# shape as a capacity metric next to utilisation. Priority classes put serving above batch;
# preemption evicts the fewest, least important pods, but it never defragments and it does not
# understand jobs made of many pods — that is what gang scheduling and Kueue are for."
#
# **Drill questions.**
#
# 1. *Half the GPUs are free but an 8-GPU pod is Pending with "Insufficient nvidia.com/gpu". Why?*
#    — Fragmentation: free GPUs are spread across nodes in pieces smaller than 8; a pod cannot span
#    nodes. Pack (MostAllocated + GPU weight), dedicate a pool to 8-GPU shapes, or preempt.
# 2. *Does the default scheduler consider GPUs when scoring?* — No. NodeResourcesFit scores cpu and
#    memory by default; GPUs only matter in the filter (the integer must fit).
# 3. *Why can bin-packing hurt an inference service?* — Replicas concentrate on few nodes, so one
#    node or GPU failure removes several replicas; balance it with topology spread constraints.
