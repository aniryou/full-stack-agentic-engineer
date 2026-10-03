# %% [markdown]
# # 02 · Filter, score and fragmentation
#
# **Tier:** T0. It is a pure-Python scheduler that uses the plugin names, formulas and messages of
# kube-scheduler. Lab notebook `03_why_is_my_pod_pending` reads real `FailedScheduling` events.
#
# ## The one-minute version
#
# kube-scheduler places **one pod at a time**. It sorts the queue by priority. In the **filter** step,
# it removes the nodes that cannot run the pod, each with a reason. In the **score** step, it gives a
# score to the other nodes, and then it binds the pod to the best node. When no node passes, it tries
# **preemption** of lower-priority pods. If preemption does not help, it reports
# `0/N nodes are available: ...`.
#
# The score step decides GPU **fragmentation**. The default `NodeResourcesFit` strategy is
# `LeastAllocated` over **cpu and memory**. It spreads pods and never looks at GPUs. Thus a stream of
# 1-GPU pods lands one per node. Then an 8-GPU pod fits nowhere, although half the GPUs are free.
# `MostAllocated` with a GPU weight packs the pods instead.
#
# After this notebook, you can read a FailedScheduling message and calculate a node score by hand. You
# can measure stranded GPUs. You can also say which scoring strategy a GPU cluster wants, and why.
#
# Primer: §3 *The scheduling cycle* in `../../PRIMER.md`.

# %%
from gpusched import (GPU, Cluster, Node, Pod, Scheduler, Toleration, fit_error, fragmentation, gpu_node,
                      gpu_pod, make_cluster, most_allocated, run_filters, select_victims, stranded_gpus)

cluster = make_cluster(hosts=4)                          # 4 nodes x 8 GPUs, cpu 96 cores, 768 GiB each
sched = Scheduler(cluster)                               # upstream defaults: LeastAllocated on cpu+memory
d = sched.schedule_one(gpu_pod("first", 1))
print("scores (0-100, higher wins):", d.scores, "->", d.node)
d = sched.schedule_one(gpu_pod("second", 1))
print("scores:", d.scores, "->", d.node, "  (the loaded node now scores lower: spread)")

# %% [markdown]
# ## A stream of small pods, then a big one
#
# Sixteen 1-GPU inference pods arrive (8 cores and 64 GiB each). Then one 8-GPU training pod arrives.

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
# Sixteen GPUs are free, but an 8-GPU pod still cannot run. Every node has 4 free GPUs, and a pod cannot
# use more than one node. The GPUs are **stranded**. They are free, but in pieces too small for the
# pending shape.
#
# ## Exercise 2.1 — the MostAllocated score
#
# The `MostAllocated` strategy of kube-scheduler gives a node a score in integer arithmetic. The score
# is a weighted average of $\lfloor \mathit{requested} \cdot 100 / \mathit{allocatable} \rfloor$ over
# the configured resources. Here, *requested* already includes the pod that arrives:
#
# $$
# \mathit{score} = \left\lfloor \frac{\sum_r \mathit{weight}_r \cdot \lfloor \min(\mathit{requested}_r, \mathit{alloc}_r) \cdot 100 / \mathit{alloc}_r \rfloor}{\sum_r \mathit{weight}_r} \right\rfloor
# $$
#
# Write `most_allocated_score(rows)`. Here, `rows` is a list of `(requested_after, allocatable, weight)`
# tuples.

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
# Now pack the pods, and do not spread them. Use `MostAllocated` for the score, and give the GPU a
# weight. In a real cluster, this is a `KubeSchedulerConfiguration` profile. The profile is in a
# scheduler that you run yourself, or it is a second profile that pods select with `schedulerName`:
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
# For a pod shape of $k$ GPUs, the **stranded GPUs** are the free GPUs that cannot hold one more such
# pod. On each node, this is $\mathit{free} \bmod k$. Write `stranded(free_per_node, k)`. Also write
# `frag(free_per_node, k)` = stranded / total free (0.0 when nothing is free).

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
# There are four 8-GPU nodes (96 cores, 768 GiB each). Some pods already run. On `h0`, a CPU-heavy
# data-prep pod runs (48 cores, 384 GiB, **no GPU**), and it tolerates the GPU taint. On `h1` and on
# `h2`, one 4-GPU embedding server runs on each. Now eight 1-GPU inference replicas arrive (8 cores, 64
# GiB each). Then two 8-GPU fine-tuning pods arrive.
#
# Select `strategy` (`"LeastAllocated"` or `"MostAllocated"`) and the `resources` weights for
# `NodeResourcesFit`. Write one sentence in `why`. The check replays the arrivals with your choice.
# Before you answer, find where plain `MostAllocated` on cpu and memory sends the first replica.

# %% exercise
# strategy = ...     "LeastAllocated" or "MostAllocated"
# resources = ...    a tuple of (resource name, weight) pairs, e.g. (("cpu", 1), ("memory", 1))
# why = ...          one sentence
### BEGIN SOLUTION
strategy = "MostAllocated"
resources = (("cpu", 1), ("memory", 1), (GPU, 5))
why = ("Pack small GPU pods onto the nodes whose GPUs are already in use so whole nodes stay free for "
       "8-GPU pods; weight the GPU above cpu and memory, or the CPU-heavy pod on h0 attracts them.")
### END SOLUTION

# %% check
def replay(strategy, resources):
    c = make_cluster(hosts=4)
    c.bind(Pod("data-prep", {"cpu": 48000, "memory": 393216},
               tolerations=[Toleration(GPU, "Exists", effect="NoSchedule")]), "b0-s0-h0")
    c.bind(gpu_pod("embed-0", 4), "b0-s0-h1")
    c.bind(gpu_pod("embed-1", 4), "b0-s0-h2")
    s = Scheduler(c, strategy=strategy, resources=resources)
    s.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(8)])
    s.run()
    return c, [s.schedule_one(gpu_pod(f"tune-{i}", 8)).node for i in range(2)]

mixed, placed = replay(strategy, resources)
assert all(placed), f"a fine-tuning pod stayed Pending: {placed}\n{mixed.show()}"
assert stranded_gpus(mixed, 8) == 0
assert isinstance(why, str) and len(why) > 20
print(mixed.show())
for label, strat, res in (("default LeastAllocated", "LeastAllocated", (("cpu", 1), ("memory", 1))),
                          ("MostAllocated, cpu+memory only", "MostAllocated", (("cpu", 1), ("memory", 1))),
                          ("MostAllocated, GPU weight 1", "MostAllocated", (("cpu", 1), ("memory", 1), (GPU, 1)))):
    print(f"{label:<32} fine-tuning pods ->", replay(strat, res)[1])
print("✅ both 8-GPU pods placed: only a score that weights the GPU keeps two whole nodes free")

# %% [markdown]
# Bin-packing has a price, and this price belongs in the same review. First, the replicas of one service
# collect on one node. If that node fails, several replicas stop. Thus, add `topologySpreadConstraints`
# for those replicas. Also, other default score plugins pull the placement toward a spread:
# `NodeResourcesBalancedAllocation`, and the default `PodTopologySpread` constraints for Deployment
# pods. Thus, measure the result.
#
# ## Reading FailedScheduling
#
# Each node fails at the *first* filter that rejects it. The message is a histogram of those reasons,
# sorted as strings.

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
# Write the exact message that kube-scheduler (and `Scheduler.schedule_one`) gives for `tp8` in the
# `shop` cluster of the previous cell. Write it *without preemption*, that is, the text before any
# `preemption:` suffix. The format is `0/<nodes> nodes are available: <count> <reason>, ... .`, and the
# `"<count> <reason>"` strings are in sorted order.

# %% exercise
# predicted = "0/4 nodes are available: ..."
### BEGIN SOLUTION
predicted = ("0/4 nodes are available: 1 Insufficient nvidia.com/gpu, 1 node(s) were unschedulable, "
             "2 node(s) didn't match Pod's node affinity/selector.")
### END SOLUTION

# %% check
actual = fit_error(4, {n.name: run_filters(pod, n) for n in shop.nodes.values()})
assert predicted == actual, actual
print("✅", actual)

# %% [markdown]
# Why does `cpu-0` say *affinity/selector* and not *Insufficient nvidia.com/gpu*? The filters run in
# this order: unschedulable, taints, node affinity, then resources. The scheduler reports the first
# failure. The correction is different for each reason. *Insufficient* means capacity (wait, preempt,
# scale up). *affinity/selector* and *taint* mean that the pod asked for a node shape that does not
# exist here.
#
# ## Priority and preemption
#
# Pods have a priority (from a `PriorityClass`). When no node passes the filters, the
# `DefaultPreemption` plugin looks for nodes where it can make room if it evicts **lower-priority**
# pods. It evicts as few pods as it can, and the least important pods that it can. Then it **nominates**
# that node (`status.nominatedNodeName`). The preemptor binds in a later cycle, after the graceful
# termination of the victims. The simulator does all of this in one step and binds the pod at once.

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
# Write the per-node step of `DefaultPreemption`. First, remove **all** pods with lower priority than
# the preemptor. If the preemptor still does not fit, return `None`. If it fits, add the pods back one
# at a time, *most important first* (higher priority, then earlier `started`). Keep out only the pods
# without which the preemptor cannot fit. Run the filters with `fits(node)` in the next cell.

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
# Note what preemption does *not* do:
#
# * It never evicts a higher-priority pod.
# * It never moves pods to collect free GPUs together (no defragmentation).
# * It looks at one pod at a time. A preemption for the first member of an 8-pod training job gains
#   nothing if the other seven cannot follow.
#
# Notebook 03 is about that last problem.
#
# ## In a design review
#
# **Two-minute version.** "The scheduler does the filter, score and bind steps for one pod at a time.
# For GPUs, the default score is incorrect for us. It is LeastAllocated over cpu and memory. Thus it
# spreads 1-GPU pods across every node. It leaves GPUs stranded in pieces too small for our 8-GPU jobs.
#
# "For GPU node pools, we run a bin-packing profile: MostAllocated with a weight on the GPU. Where
# availability is important, we keep inference replicas apart with topology spread constraints. We
# monitor stranded GPUs per pod shape as a capacity metric, next to utilisation.
#
# "Priority classes put serving above batch. Preemption evicts the fewest and least important pods. But
# it never does a defragmentation, and it does not understand jobs that have many pods. Gang scheduling
# and Kueue are for that problem."
#
# **Drill questions.**
#
# 1. *Half the GPUs are free but an 8-GPU pod is Pending with "Insufficient nvidia.com/gpu". Why?*
#    Fragmentation. The free GPUs are spread across nodes in pieces smaller than 8. A pod cannot use
#    more than one node. Pack the pods (MostAllocated + GPU weight), use a dedicated pool for 8-GPU
#    shapes, or preempt.
# 2. *Does the default scheduler consider GPUs when scoring?* No. By default, NodeResourcesFit uses cpu
#    and memory for the score. GPUs are important only in the filter (the integer must fit).
# 3. *Why can bin-packing hurt an inference service?* Replicas collect on few nodes. Thus one node
#    failure or GPU failure removes several replicas. Balance it with topology spread constraints.
