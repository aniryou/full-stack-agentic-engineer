# %% [markdown]
# # 03 · Gangs and topology
#
# **Tier:** T0. It is pure Python. Lab notebooks `01_manifests_and_the_linter` and
# `02_kind_with_fake_gpus_and_kueue` build the same workloads as real JobSet / LeaderWorkerSet objects
# with Kueue topology annotations, and run them on kind.
#
# ## The one-minute version
#
# A distributed training job or a multi-host inference replica is useful only when **all** of its pods
# run. The workers block at the first collective until every rank joins.
#
# The default scheduler places pods one at a time. Thus, when two such jobs arrive together, each job
# can get *part* of what it needs. Then every GPU is allocated, but no job makes progress. This is a
# **deadlock**.
#
# A **gang** gets an all-or-nothing placement. Kueue admits a whole workload or none of it. The
# coscheduling plugin, Volcano and the new Kubernetes Workload API do the same at the pod level.
#
# But "all" is not sufficient. The pods of a job must also be **near** each other: inside one NVLink
# domain, sub-block or block. The reason is that the collectives run at the speed of the slowest link
# that they cross. Kueue's Topology-Aware Scheduling selects the tightest domain that holds the whole
# gang.
#
# After this notebook, you can show the deadlock and explain the solution. You can also select a
# topology constraint for a workload.
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
# Twelve GPUs are allocated and zero jobs run, and nothing will change. The three placed workers of each
# job wait for their fourth worker. That worker waits for GPUs that the workers of the other job hold.
# Preemption does not help (equal priority). If a job has a higher priority, it evicts pods one at a
# time, for its *first* worker only. The solution is to make one decision for the whole group at once.
#
# ## Exercise 3.1 — all or nothing
#
# Write `all_or_nothing(cluster, pods)`. Try to bind every pod to the first node where `run_filters`
# passes. If a pod finds no node, **evict the pods that you placed** and return `False`. If all pods
# find a node, return `True`.

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
# Kueue gives you this all-or-nothing admission for a Job or JobSet. The workload stays **suspended**
# until its whole request fits the quota of the queue. With TAS or a ProvisioningRequest, the request
# must also fit the nodes. Then Kueue permits the creation of all of its pods at once. The function
# `admit_gangs` in the core does the same loop over a FIFO of gangs:

# %%
again = make_cluster(hosts=3, gpus=4)
print("admitted:", admit_gangs(again, [[gpu_pod(f"a{i}", 2, gang="A") for i in range(4)],
                                       [gpu_pod(f"b{i}", 2, gang="B") for i in range(4)]]))

# %% [markdown]
# ## Topology: *where* the gang runs matters
#
# A data-center GPU fleet is a tree:
#
# * The GPUs in a host share NVLink.
# * The hosts in a **sub-block** share leaf switches (full bandwidth inside).
# * Sub-blocks make a **block**.
# * Blocks exchange data through oversubscribed spine links.
#
# A tensor-parallel all-reduce that crosses a slow level runs at the speed of that level (layer 01 has
# the numbers). GCE shows the tree as node labels: `cloud.google.com/gce-topology-block`, `-subblock`
# and `-host` (verify). The `Topology` object of Kueue lists them as levels. A pod template asks for one
# level with an annotation:
#
# ```yaml
# metadata:
#   annotations:
#     kueue.x-k8s.io/podset-required-topology: cloud.google.com/gce-topology-subblock   # or -preferred-
# ```
#
# The next cell has 2 blocks x 2 sub-blocks x 4 hosts x 8 GPUs, partly busy. The table counts how many
# more **8-GPU pods** each host and sub-block can take. A host with even one GPU in use counts zero.

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
# With no constraint, Kueue puts all hosts in one flat list, tightest first (`LeastFreeCapacity`). It
# takes the tightest single host that holds the whole gang. When no host does, it fills the smallest
# gaps first (here, every free host holds one 8-GPU pod). Thus the 3-node job goes across sub-blocks.
#
# With a topology constraint, Kueue uses **BestFit**. Among the domains that can hold the whole gang, it
# takes the *tightest*. When it must divide a gang across child domains, it first takes the domains with
# the most room. It selects the last domain as the tightest one that holds the remainder. This selection
# keeps large holes intact for large jobs. Then Kueue does the selection again one level down, over the
# children of *all* the domains that it selected.
#
# The gangs of the simulator have one pod shape. They have no leader pod set, no slices and no balanced
# placement.
#
# ## Exercise 3.2 — Kueue's BestFit
#
# Write `bestfit(capacity, n)`. The `capacity` argument maps sibling domains to the number of pods that
# each domain can take. Return `{domain: pods}` with all `n` pods placed, or `None` if they do not fit.
# Use this rule again and again. If an unused domain can hold all the pods that are left, select the
# **smallest** such domain (ties by name) and stop. If not, take the domain with the **most** room (ties
# by name) and fill it.

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
# Use the `room` table that an earlier cell printed (free 8-GPU slots per sub-block). Predict where
# `place_gang(fleet, gang, required="subblock")` puts a gang of **3** and a gang of **2** 8-GPU pods.
# Answer with the sub-block name. Also predict which **block** a gang of **5** lands in with
# `preferred="subblock"`. No sub-block holds 5, so the placement widens to a block.

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
print("✅ gang of 5 (preferred) ->", sorted(fleet.nodes[n].name for n in g5.values()))
split = " + ".join(f"{sum(fleet.nodes[n].topology[1] == sb for n in g5.values())} in {sb}"
                  for sb in sorted(subblocks(g5)))
print("   the sub-block pass picks 4 + 1, but the host pass re-runs BestFit over all six free hosts of both\n"
      "   sub-blocks, which tie at one slot each and are taken in name order:", split)

# %% [markdown]
# ## Exercise 3.4 — choose a constraint per workload
#
# For each workload, select `("required", level)`, `("preferred", level)` or `None` (unconstrained). The
# level is one of `"host"`, `"subblock"`, `"block"`:
#
# * **serving**: one replica of a large model, which a LeaderWorkerSet group of 2 nodes x 8 GPUs serves.
#   The shards exchange activations every layer, every token.
# * **pretraining**: a 32-node data-parallel + pipeline-parallel JobSet that runs for weeks.
# * **batch-eval**: 200 independent 1-GPU pods that calculate scores over a dataset. There is no
#   communication at all.

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
import hashlib


def digest(key, value):  # the check compares digests, so the blank does not print the answer
    value = tuple(value) if isinstance(value, (list, tuple)) else value
    return hashlib.sha256(f"{key}={value!r}".encode()).hexdigest()[:10]


ACCEPTED = {  # more than one digest: more than one defensible answer
    "serving": {'fa891141ec'},
    "pretraining": {'64f19d6c88', '9092837345'},
    "batch-eval": {'79c8c55899'},
}
HINTS = {
    "serving": "a replica that talks every layer, every token: is a slow placement acceptable, or worth waiting for?",
    "pretraining": "32 nodes for weeks: locality matters, but can you afford to wait for a perfect domain?",
    "batch-eval": "no communication at all: what does any topology constraint buy?",
}
for workload, ok in ACCEPTED.items():
    assert digest(workload, choices.get(workload)) in ok, f"{workload}: not quite. {HINTS[workload]}"
print("✅", choices)

# %% [markdown]
# The choices look like this in manifests. Lab notebook `01_manifests_and_the_linter` builds and lints
# them:
#
# * **JobSet** (`jobset.x-k8s.io/v1alpha2`): one ReplicatedJob of 32 indexed pods. Put the Kueue
#   annotation on the pod template. Or use JobSet's own `alpha.jobset.sigs.k8s.io/exclusive-topology` to
#   give each child Job a whole topology domain.
# * **LeaderWorkerSet** (`leaderworkerset.x-k8s.io/v1`): `leaderWorkerTemplate.size: 2`, with one group
#   per replica. Use `leaderworkerset.sigs.k8s.io/exclusive-topology` to keep a group inside one domain.
#   Use `restartPolicy: RecreateGroupOnPodRestart`, so that a failed shard restarts the whole group.
# * **Plain Job**: `parallelism = completions = N`, plus the queue label `kueue.x-k8s.io/queue-name`.
#   Kueue suspends the Job until it admits the whole pod set.
#
# ## In a design review
#
# **Two-minute version.** "Our training jobs and multi-host inference replicas are gangs. No rank does
# useful work until all ranks run. If the scheduler places their pods one at a time, two jobs can divide
# the cluster between them and get a deadlock. They hold every GPU, and neither job runs.
#
# "Thus every multi-pod GPU workload goes through Kueue. Jobs stay suspended until the whole pod set
# fits, and then they start together. A PodsReady timeout evicts a job and puts it back in the queue
# when not all of its pods become ready.
#
# "Placement is topology-aware. Kueue TAS reads the block, sub-block and host labels, and puts a gang in
# the tightest domain that holds it. We use a required constraint for inference groups whose shards
# exchange data every token. We use a preferred constraint for long training jobs, and no constraint
# for embarrassingly parallel batch."
#
# **Drill questions.**
#
# 1. *Twelve GPUs are allocated but nothing is training. What happened?* A partial placement of two
#    gangs. Each gang holds some GPUs and waits for the rest. Admit gangs atomically (Kueue,
#    coscheduling).
# 2. *Required or preferred topology for a 2-node tensor-parallel serving group?* Required, at the
#    smallest multi-node domain. A replica that goes across a slow link is slow for its whole life.
# 3. *Why does Kueue use LeastFreeCapacity for unconstrained pod sets but BestFit otherwise?* Without a
#    constraint, the goal is to use the tightest host, or to fill small gaps. The goal is also to keep
#    large domains whole for the jobs that need them. With a constraint, the goal is to find the
#    tightest single domain that holds the gang.
