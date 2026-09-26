# %% [markdown]
# # 03 · Gangs and topology
#
# **Tier:** T0 — pure Python. The same workloads as real JobSet / LeaderWorkerSet objects with Kueue
# topology annotations are built and run on kind in lab notebooks `01_manifests_and_the_linter`
# and `02_kind_with_fake_gpus_and_kueue`.
#
# ## The one-minute version
#
# A distributed training job or a multi-host inference replica is useful only when **all** of its
# pods run: workers block at the first collective until every rank has joined. The default
# scheduler places pods one at a time, so two such jobs arriving together can each get *part* of
# what they need — every GPU allocated, no job making progress: a **deadlock**. A **gang** is placed
# all-or-nothing: Kueue admits a whole workload or none of it (the coscheduling plugin, Volcano and
# the new Kubernetes Workload API do the same at the pod level). And "all" is not enough: a job's
# pods must also be **close** — inside one NVLink domain, sub-block or block — because the
# collectives run at the speed of the slowest link they cross. Kueue's Topology-Aware Scheduling
# picks the tightest domain that holds the whole gang. After this notebook you can show the
# deadlock, explain the fix, and choose a topology constraint for a workload.
#
# Primer: §4 *Gangs* and §5 *Topology-aware placement* in `../../PRIMER.md`.

# %%
from gpusched import (Cluster, Scheduler, admit_gangs, best_fit, gpu_pod, interleave, least_free_capacity,
                      make_cluster, place_gang, run_filters, running)

# Three nodes with 4 GPUs each (12 GPUs). Two jobs, A and B, each 4 workers x 2 GPUs = 8 GPUs.
cluster = make_cluster(hosts=3, gpus=4)
A = [gpu_pod(f"a{i}", 2, gang="A") for i in range(4)]
B = [gpu_pod(f"b{i}", 2, gang="B") for i in range(4)]
s = Scheduler(cluster)
s.submit(*interleave(A, B))          # both job controllers create pods at the same time
s.run()
print(cluster.show())
print("\nA running:", running(A), "| B running:", running(B), "| free GPUs:", cluster.free(),
      "| pending:", [p.name for p in s.pending])

# %% [markdown]
# Twelve GPUs allocated, zero jobs running, and nothing will change: each job's three placed
# workers wait for their fourth, which waits for GPUs held by the other job's workers. Preemption
# does not help (equal priority), and a higher-priority job would evict pods one at a time for its
# *first* worker only. The fix is to decide for the whole group at once.
#
# ## Exercise 3.1 — all or nothing
#
# Write `all_or_nothing(cluster, pods)`: try to bind every pod (to the first node where
# `run_filters` passes); if any pod finds no node, **evict the ones you placed** and return
# `False`; else return `True`.

# %% exercise
def all_or_nothing(cluster: Cluster, pods: list) -> bool:
    placed = []
    ### BEGIN SOLUTION
    for pod in pods:
        node = next((n for n in cluster.nodes.values() if not run_filters(pod, n)), None)
        if node is None:
            for p in placed:
                cluster.evict(p)
            return False
        cluster.bind(pod, node.name)
        placed.append(pod)
    return True
    ### END SOLUTION

# %% check
fresh = make_cluster(hosts=3, gpus=4)
A2 = [gpu_pod(f"a{i}", 2, gang="A") for i in range(4)]
B2 = [gpu_pod(f"b{i}", 2, gang="B") for i in range(4)]
assert all_or_nothing(fresh, A2) is True and running(A2)
assert all_or_nothing(fresh, B2) is False
assert not any(p.node for p in B2) and fresh.free() == 4          # B holds nothing while it waits
print(fresh.show())
print("✅ A runs on 8 GPUs; B waits holding nothing and starts when A finishes")

# %% [markdown]
# That is what Kueue gives you for a Job or JobSet: the workload stays **suspended** until its whole
# request fits the queue's quota (and, with TAS or a ProvisioningRequest, the nodes), then all its
# pods are created at once. `admit_gangs` in the core is the same loop over a FIFO of gangs:

# %%
again = make_cluster(hosts=3, gpus=4)
print("admitted:", admit_gangs(again, [[gpu_pod(f"a{i}", 2, gang="A") for i in range(4)],
                                       [gpu_pod(f"b{i}", 2, gang="B") for i in range(4)]]))

# %% [markdown]
# ## Topology: *where* the gang runs matters
#
# A data-center GPU fleet is a tree: GPUs in a host share NVLink; hosts in a **sub-block** share leaf
# switches (full bandwidth inside); sub-blocks form a **block**; blocks talk through oversubscribed
# spine links. A tensor-parallel all-reduce that crosses a slow level runs at that level's speed
# (layer 01 has the numbers). GCE exposes the tree as node labels
# (`cloud.google.com/gce-topology-block`, `-subblock`, `-host` — verify); Kueue's `Topology` object lists
# them as levels, and a pod template asks for one with an annotation:
#
# ```yaml
# metadata:
#   annotations:
#     kueue.x-k8s.io/podset-required-topology: cloud.google.com/gce-topology-subblock   # or -preferred-
# ```
#
# Below: 2 blocks x 2 sub-blocks x 4 hosts x 8 GPUs, partly busy. The table counts how many more
# **8-GPU pods** each host and sub-block can take — a host with even one GPU in use counts zero.

# %%
fleet = make_cluster(blocks=2, subblocks=2, hosts=4)
busy = {"b0-s0-h0": 8, "b0-s0-h1": 8, "b0-s0-h2": 8, "b0-s1-h0": 8, "b1-s0-h0": 8, "b1-s0-h1": 1}
for host, gpus in busy.items():
    fleet.bind(gpu_pod(f"x-{host}", gpus), host)
room = {}
for n in fleet.nodes.values():
    room[n.topology[1]] = room.get(n.topology[1], 0) + (1 if n.free() == 8 else 0)
print("free 8-GPU slots per sub-block:", room)
spread = place_gang(fleet, [gpu_pod(f"u{i}", 8) for i in range(3)])            # no constraint
print("unconstrained 3-node gang:", spread)

# %% [markdown]
# Unconstrained, Kueue fills the smallest gaps first (`LeastFreeCapacity`, good against
# fragmentation) and the 3-node job straddles sub-blocks. With a topology constraint it uses
# **BestFit**: among domains that can hold the whole gang, take the *tightest*; when a gang has to
# be split across child domains, take the ones with the most room first and choose the last one as
# the tightest that holds the remainder (that keeps big holes intact for big jobs).
#
# ## Exercise 3.2 — Kueue's BestFit
#
# Write `bestfit(capacity, n)`: `capacity` maps sibling domains to how many pods each can take.
# Return `{domain: pods}` placing all `n`, or `None` if they do not fit. Rule: repeatedly, if some
# remaining domain can hold everything that is left, pick the **smallest** such domain (ties by
# name) and stop; otherwise take the domain with the **most** room (ties by name) and fill it.

# %% exercise
def bestfit(capacity: dict, n: int) -> dict | None:
    ### BEGIN SOLUTION
    left, out = n, {}
    pool = sorted((d for d, c in capacity.items() if c > 0), key=lambda d: (-capacity[d], d))
    while left > 0 and pool:
        fits = [d for d in pool if capacity[d] >= left]
        d = min(fits, key=lambda d: (capacity[d], d)) if fits else pool[0]
        out[d] = min(capacity[d], left)
        left -= out[d]
        pool.remove(d)
    return out if left == 0 else None
    ### END SOLUTION

# %% check
rack = {"n1": 3, "n2": 3, "n3": 2, "n4": 1}                 # the example in Kueue's TAS design (KEP-2724)
assert bestfit(rack, 7) == {"n1": 3, "n2": 3, "n4": 1}      # 3 + 3, then the exact 1-slot node
assert bestfit(rack, 2) == {"n3": 2}
assert bestfit(rack, 10) is None
for n in range(1, 10):
    assert bestfit(rack, n) == best_fit(rack, n), n
print("✅ BestFit:", bestfit(rack, 7), "| LeastFreeCapacity would take", least_free_capacity(rack, 7))

# %% [markdown]
# ## Exercise 3.3 — predict the placement
#
# Using the `room` table printed above (free 8-GPU slots per sub-block), predict where
# `place_gang(fleet, gang, required="subblock")` puts a gang of **3** and a gang of **2** 8-GPU pods
# (answer with the sub-block name), and which **block** a gang of **5** lands in with
# `preferred="subblock"` (no sub-block holds 5, so it widens to a block).

# %% exercise
# subblock_for_3 = ...   a sub-block name such as "b0-s0"
# subblock_for_2 = ...
# block_for_5 = ...      a block name such as "b0"
### BEGIN SOLUTION
subblock_for_3 = "b0-s1"         # the only sub-block with exactly 3 (b1-s1 has 4: not the tightest)
subblock_for_2 = "b1-s0"         # h0 full, h1 has 1 GPU busy: 2 whole hosts left
block_for_5 = "b1"               # b0 has 1 + 3 = 4 < 5; b1 has 2 + 4 = 6
### END SOLUTION

# %% check
def subblocks(placement):
    return {fleet.nodes[n].topology[1] for n in placement.values()}
g3 = place_gang(fleet, [gpu_pod(f"g{i}", 8) for i in range(3)], required="subblock")
g2 = place_gang(fleet, [gpu_pod(f"h{i}", 8) for i in range(2)], required="subblock")
g5 = place_gang(fleet, [gpu_pod(f"k{i}", 8) for i in range(5)], preferred="subblock")
assert subblocks(g3) == {subblock_for_3} and subblocks(g2) == {subblock_for_2}
assert {fleet.nodes[n].topology[0] for n in g5.values()} == {block_for_5}
assert place_gang(fleet, [gpu_pod(f"r{i}", 8) for i in range(5)], required="subblock") is None
print("✅ gang of 5 (preferred) ->", sorted(subblocks(g5)), "= 4 in the empty sub-block + 1 in the tightest")

# %% [markdown]
# ## Exercise 3.4 — choose a constraint per workload
#
# For each workload pick `("required", level)`, `("preferred", level)` or `None` (unconstrained),
# with level one of `"host"`, `"subblock"`, `"block"`:
#
# * **serving** — one replica of a large model served by a LeaderWorkerSet group of 2 nodes x 8 GPUs;
#   the shards exchange activations every layer, every token.
# * **pretraining** — a 32-node data-parallel + pipeline-parallel JobSet running for weeks.
# * **batch-eval** — 200 independent 1-GPU pods scoring a dataset; no communication at all.

# %% exercise
# choices = {"serving": ..., "pretraining": ..., "batch-eval": ...}
### BEGIN SOLUTION
choices = {
    "serving": ("required", "subblock"),     # a slow replica is worse than a late one: wait for a good spot
    "pretraining": ("preferred", "block"),   # locality matters, but waiting for a perfect block may be forever
    "batch-eval": None,                      # no traffic: fill the gaps others leave (LeastFreeCapacity)
}
### END SOLUTION

# %% check
assert choices["serving"] == ("required", "subblock")
assert choices["pretraining"][0] == "preferred" and choices["pretraining"][1] in ("block", "subblock")
assert choices["batch-eval"] is None
print("✅", choices)

# %% [markdown]
# How the choices look in manifests (built and linted in lab notebook `01_manifests_and_the_linter`):
#
# * **JobSet** (`jobset.x-k8s.io/v1alpha2`): one ReplicatedJob of 32 indexed pods; put the Kueue
#   annotation on the pod template, or use JobSet's own `alpha.jobset.sigs.k8s.io/exclusive-topology`
#   to give each child Job a whole topology domain.
# * **LeaderWorkerSet** (`leaderworkerset.x-k8s.io/v1`): `leaderWorkerTemplate.size: 2`, one group
#   per replica, `leaderworkerset.sigs.k8s.io/exclusive-topology` to keep a group inside one domain,
#   `restartPolicy: RecreateGroupOnPodRestart` so a failed shard restarts the whole group.
# * **Plain Job**: `parallelism = completions = N` plus the queue label `kueue.x-k8s.io/queue-name`;
#   Kueue suspends it until the whole pod set is admitted.
#
# ## In a design review
#
# **Two-minute version.** "Our training jobs and multi-host inference replicas are gangs: no rank
# does useful work until all ranks run. Scheduling their pods one at a time lets two jobs split the
# cluster between them and deadlock, holding every GPU while neither runs. So every multi-pod GPU
# workload goes through Kueue: jobs stay suspended until the whole pod set fits, then start
# together, with a PodsReady timeout that evicts and requeues a job whose pods do not all come up.
# Placement is topology-aware: Kueue TAS reads the block, sub-block and host labels and puts a gang
# in the tightest domain that holds it — required for inference groups whose shards talk every
# token, preferred for long training jobs, unconstrained for embarrassingly parallel batch."
#
# **Drill questions.**
#
# 1. *Twelve GPUs are allocated but nothing is training. What happened?* — Partial placement of two
#    gangs: each holds some GPUs waiting for the rest. Admit gangs atomically (Kueue, coscheduling).
# 2. *Required or preferred topology for a 2-node tensor-parallel serving group?* — Required at the
#    smallest multi-node domain: a replica split across a slow link is slow for its whole life.
# 3. *Why does Kueue use LeastFreeCapacity for unconstrained pod sets but BestFit otherwise?* —
#    Without a constraint the goal is to fill small gaps and keep large domains whole for jobs that
#    need them; with one, it is to find the tightest single domain that holds the gang.
