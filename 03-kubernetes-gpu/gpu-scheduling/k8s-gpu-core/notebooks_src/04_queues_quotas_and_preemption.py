# %% [markdown]
# # 04 · Queues, quotas and preemption
#
# **Tier:** T0. It is a Kueue-style quota simulator that uses the field names and admission rules of
# Kueue. Lab notebook `02_kind_with_fake_gpus_and_kueue` applies real Kueue v0.19 objects
# (ResourceFlavor, ClusterQueue, LocalQueue, WorkloadPriorityClass) to a kind cluster. The simulator
# leaves out some parts of Kueue:
#
# * It has one resource group per ClusterQueue.
# * It has flat cohorts (no Cohort objects).
# * It has classic preemption only (no Fair Sharing).
# * It has no admission checks.
# * It admits a preemptor in the same step that it evicts the victims. Real Kueue marks the victims
#   Evicted and admits the preemptor in a later cycle, when their quota is free again.
#
# ## The one-minute version
#
# The scheduler decides **where** a pod runs. A quota system decides **if a job can start at all**.
# Kueue keeps a job *suspended* until its whole request fits the quota of its **ClusterQueue**. The job
# gets to its ClusterQueue through a namespaced **LocalQueue**. Then Kueue permits the creation of all
# of its pods.
#
# Quotas are per **ResourceFlavor** (for example, H100 on-demand or H100 Spot) and per resource.
# ClusterQueues in a **cohort** lend their *unused* quota to each other. `borrowingLimit` sets the
# maximum that one queue can take. `lendingLimit` sets the maximum that one queue gives away.
#
# Borrowed GPUs are loans. With `reclaimWithinCohort`, the owner **preempts** borrowers when it needs
# its quota back. `withinClusterQueue` lets high-priority jobs preempt low-priority jobs in the same
# queue.
#
# After this notebook, you can calculate what a team can use at this time and predict which workloads
# Kueue preempts. You can also write the ClusterQueue for a serving/batch split.
#
# Primer: §6 *Queues, quotas and multi-tenancy with Kueue* in `../../PRIMER.md`.

# %%
from gpusched import GPU, ClusterQueue, Kueue, Quota, Workload

def research_cohort(**team_a_policy):
    team_a = ClusterQueue("team-a-cq", {"h100": {GPU: Quota(16)}}, cohort="research", **team_a_policy)
    team_b = ClusterQueue("team-b-cq", {"h100": {GPU: Quota(16)}}, cohort="research")
    return Kueue([team_a, team_b], local_queues={"team-a/gpus": "team-a-cq", "team-b/gpus": "team-b-cq"})

kueue = research_cohort()
kueue.submit(*[Workload(f"b{i}", "team-b/gpus", {GPU: 8}) for i in (1, 2, 3)])   # team-a is idle
for event in kueue.schedule():
    print(event)
print(kueue.table())

# %% [markdown]
# The third job of team B **borrowed** 8 of the idle GPUs of team A. It fits because the cohort has 32
# GPUs of nominal quota, and only 16 of B's own GPUs plus 8 more are in use. This is the quota
# arithmetic behind `available` (Kueue's `resource_node.go`, for a flat cohort):
#
# * $\text{guaranteed} =$ $\text{nominal} - \texttt{lendingLimit}$ (0 when there is no lending limit).
#   The queue never lends this part.
# * The cohort pool = the sum over the members of $\text{nominal} - \text{guaranteed}$. The pool usage =
#   the sum of $\max(0, \text{usage} - \text{guaranteed})$.
# * $\text{available} =$ $\max(0, \text{guaranteed} - \text{usage}) +{}$ $(\text{pool} - \text{pool
#   usage})$. If the queue sets a `borrowingLimit`, the cohort part has a maximum of $(\text{nominal} -
#   \text{guaranteed}) -{}$ $\max(0, \text{usage} - \text{guaranteed}) +{}$ $\texttt{borrowingLimit}$.
#
# ## Exercise 4.1 — how many GPUs can this ClusterQueue use right now?
#
# Write this function. `me` and each entry of `others` are dicts with the keys `nominal`, `usage`,
# `borrowing_limit` and `lending_limit`. The limits can be `None`.

# %% exercise
def available(me: dict, others: list) -> int:
    ### BEGIN SOLUTION
    def guaranteed(q):
        return max(0, q["nominal"] - q["lending_limit"]) if q["lending_limit"] is not None else 0
    members = [me] + others
    pool = sum(q["nominal"] - guaranteed(q) for q in members)
    used = sum(max(0, q["usage"] - guaranteed(q)) for q in members)
    g = guaranteed(me)
    from_cohort = pool - used
    if me["borrowing_limit"] is not None:
        from_cohort = min(from_cohort, (me["nominal"] - g) - max(0, me["usage"] - g) + me["borrowing_limit"])
    return max(0, g - me["usage"]) + from_cohort
    ### END SOLUTION

# %% check
def q(nominal, usage=0, borrowing_limit=None, lending_limit=None):
    return dict(nominal=nominal, usage=usage, borrowing_limit=borrowing_limit, lending_limit=lending_limit)
assert available(q(9), [q(12)]) == 21                           # Kueue docs: 9 + 12
assert available(q(9, borrowing_limit=1), [q(12)]) == 10        # docs: 9 + 1
assert available(q(9), [q(12, lending_limit=1)]) == 10          # docs: 9 + 1
assert available(q(16), [q(16, usage=24)]) == 8                 # team A above: 8 of its 16 are lent out
assert available(q(16, lending_limit=8), [q(16, usage=24)]) == 8  # 8 stay home, the other 8 are borrowed
a, b = kueue.cqs["team-a-cq"], kueue.cqs["team-b-cq"]
assert available(q(16), [q(16, usage=24)]) == kueue.available(a, "h100", GPU)
print("✅ available() matches Kueue's arithmetic and its documented examples")

# %% [markdown]
# ## When the owner comes back
#
# Now team A submits a 16-GPU job. It has a right to 16 GPUs, but 8 of its GPUs are on loan.

# %%
a1 = Workload("a1", "team-a/gpus", {GPU: 16})
kueue.submit(a1)
print("events:", kueue.schedule())
print(kueue.explain(a1))

# %% [markdown]
# Nothing occurs. With the default `reclaimWithinCohort: Never`, **nominal quota is not a guarantee**.
# Team A waits until the job of team B that borrowed GPUs finishes by itself, for as long as that takes:

# %%
kueue.finish("b3")
print("after b3 finishes:", kueue.schedule())

# %% [markdown]
# Now turn on reclaim. Then replay the same arrivals:

# %%
kueue = research_cohort(reclaim_within_cohort="Any")
kueue.submit(*[Workload(f"b{i}", "team-b/gpus", {GPU: 8}) for i in (1, 2, 3)])
kueue.schedule()
kueue.submit(Workload("a1", "team-a/gpus", {GPU: 16}))
for event in kueue.schedule():
    print(event)
print(kueue.table())

# %% [markdown]
# ## Exercise 4.2 — who is preempted first?
#
# The classic preemption of Kueue first makes a list of candidates. These are the workloads in the
# preemptor's own ClusterQueue that its `withinClusterQueue` policy permits. They are also the workloads
# of the ClusterQueues in the cohort that *borrow*, where `reclaimWithinCohort` permits it. Then Kueue
# removes the candidates in this order until the preemptor fits: **other ClusterQueues first**, then
# **lowest priority**, then **most recently admitted**. Write the sort. Each candidate is a dict with
# `cq`, `priority` and `admitted` (the admission order, where a larger value is more recent).

# %% exercise
def preemption_order(candidates: list, my_cq: str) -> list:
    ### BEGIN SOLUTION
    return sorted(candidates, key=lambda c: (c["cq"] == my_cq, c["priority"], -c["admitted"]))
    ### END SOLUTION

# %% check
cands = [dict(name="mine-low", cq="a", priority=0, admitted=9), dict(name="b-old", cq="b", priority=0, admitted=1),
         dict(name="b-new", cq="b", priority=0, admitted=5), dict(name="b-high", cq="b", priority=100, admitted=7)]
assert [c["name"] for c in preemption_order(cands, "a")] == ["b-new", "b-old", "b-high", "mine-low"]
print("✅ reclaim from borrowers first, cheapest first, newest first (it has done the least work)")

# %% [markdown]
# Then Kueue *minimises* the list. It goes through the selected targets in reverse order and gives back
# each target that it did not need to remove. Kueue takes a candidate from another ClusterQueue only
# while that queue still borrows. When team B is back at its nominal quota, the jobs that team B still
# has are safe.
#
# ## Exercise 4.3 — StrictFIFO or BestEffortFIFO?
#
# There is one empty ClusterQueue with 16 GPUs of quota. Three workloads arrive in this order: `small-1`
# (8 GPUs), `big` (16), `small-2` (8). Predict which workloads Kueue admits under each queueing
# strategy.

# %% exercise
# admitted_strict = [...]        workload names, in admission order
# admitted_best_effort = [...]
### BEGIN SOLUTION
admitted_strict = ["small-1"]                  # big cannot fit behind small-1 and blocks small-2
admitted_best_effort = ["small-1", "small-2"]  # small-2 may pass the blocked big job
### END SOLUTION

# %% check
for strategy, predicted in (("StrictFIFO", admitted_strict), ("BestEffortFIFO", admitted_best_effort)):
    k = Kueue([ClusterQueue("cq", {"h100": {GPU: Quota(16)}}, queueing_strategy=strategy)])
    k.submit(Workload("small-1", "cq", {GPU: 8}), Workload("big", "cq", {GPU: 16}), Workload("small-2", "cq", {GPU: 8}))
    assert [e[1] for e in k.schedule()] == predicted, strategy
print("✅ StrictFIFO keeps order at the cost of idle quota; BestEffortFIFO fills quota but can starve big jobs")

# %% [markdown]
# ## Exercise 4.4 — write the ClusterQueues for a serving/batch split
#
# There is one cohort `shared` and one flavor `h100`. This is the policy:
#
# * `serving-cq` owns 16 GPUs. It can lend at most 8 of them. It must preempt borrowers to get lent GPUs
#   back, whatever their priority. It never preempts its own workloads.
# * `batch-cq` owns 16 GPUs and can borrow at most 8. Inside the queue, a higher-priority job can
#   preempt lower-priority jobs.
#
# Make `serving_cq` and `batch_cq` with `ClusterQueue(...)` and `Quota(...)`.

# %% exercise
# serving_cq = ClusterQueue("serving-cq", {"h100": {GPU: Quota(...)}}, cohort="shared", ...)
# batch_cq = ClusterQueue("batch-cq", ...)
### BEGIN SOLUTION
serving_cq = ClusterQueue("serving-cq", {"h100": {GPU: Quota(16, lending_limit=8)}}, cohort="shared",
                          reclaim_within_cohort="Any", within_cluster_queue="Never")
batch_cq = ClusterQueue("batch-cq", {"h100": {GPU: Quota(16, borrowing_limit=8)}}, cohort="shared",
                        within_cluster_queue="LowerPriority")
### END SOLUTION

# %% check
assert serving_cq.guaranteed("h100", GPU) == 8 and serving_cq.within_cluster_queue == "Never"
assert batch_cq.quota("h100", GPU).borrowing_limit == 8
k = Kueue([serving_cq, batch_cq])
k.submit(*[Workload(f"batch-{i}", "batch-cq", {GPU: 8}, priority=0) for i in range(5)])
assert [e[1] for e in k.schedule()] == ["batch-0", "batch-1", "batch-2"]        # 16 own + 8 borrowed
assert k.available(serving_cq, "h100", GPU) == 8                                 # 8 never left home
k.submit(Workload("serve", "serving-cq", {GPU: 16}))
assert ("preempted", "batch-2", "batch-cq", "serve") in k.schedule()             # the loan is recalled
k.submit(Workload("urgent-batch", "batch-cq", {GPU: 8}, priority=100))
assert k.schedule()[0][:2] == ("preempted", "batch-1")                           # newest low-priority job
print(k.table())
print("✅ serving keeps 8 GPUs at home, recalls its loan at once, and batch sorts itself out by priority")

# %% [markdown]
# This is the same policy as Kueue objects (`kueue.x-k8s.io/v1beta2`). The lab applies them on kind:
#
# ```yaml
# apiVersion: kueue.x-k8s.io/v1beta2
# kind: ClusterQueue
# metadata: {name: serving-cq}
# spec:
#   cohortName: shared
#   namespaceSelector: {}
#   preemption: {reclaimWithinCohort: Any, withinClusterQueue: Never}
#   resourceGroups:
#   - coveredResources: ["nvidia.com/gpu"]
#     flavors:
#     - name: h100
#       resources: [{name: "nvidia.com/gpu", nominalQuota: 16, lendingLimit: 8}]
# ```
#
# A `ResourceFlavor` with the name `h100` holds the node labels (and tolerations) that Kueue injects
# into the admitted pods. Thus the pods land on H100 nodes. A `LocalQueue` in each team namespace points
# at its ClusterQueue. A job joins Kueue with the label `kueue.x-k8s.io/queue-name`, and gets a priority
# from `kueue.x-k8s.io/priority-class` (a `WorkloadPriorityClass`). This priority puts the workloads in
# the queue in order and decides their preemption. It does not change the pods' own scheduling priority.

# %% [markdown]
# ## Exercise 4.5 — predict a reclaim
#
# There are two ClusterQueues in one cohort, and one flavor `f`. Team A has a nominal quota of **12**
# GPUs and `reclaimWithinCohort: Any`, and it is idle. Team B has a nominal quota of **4**. It runs
# `b-big` (8 GPUs, priority 5, admitted first) and `b-small` (2 GPUs, priority 0). It borrows 6 of A's
# GPUs. Team A submits a **10**-GPU job.
#
# Use the rules from exercise 4.2 and the note after it. Which candidates does the greedy pass remove,
# and in which order, until the job of team A fits? Which candidates does the reverse pass give back?
#
# Then look at a second cohort, with the queues `a`, `b` and `c` and a nominal quota of 8 each. `a` sets
# reclaim to `Any`. `b` runs `b1` (8) and `b2` (4, which borrows). `c` runs `c1` (8), admitted last. `a`
# submits 8. Which workload does Kueue preempt?

# %% exercise
# preempted_first = [...]    names preempted for team A's 10-GPU job
# preempted_second = [...]   names preempted for a's 8-GPU job
### BEGIN SOLUTION
preempted_first = ["b-big"]     # greedy: b-small (lowest priority) then b-big; in reverse b-small fits back
preempted_second = ["b2"]       # c1 is the newest, but c is not borrowing, so it is never a target
### END SOLUTION

# %% check
def reclaim_events():
    a = ClusterQueue("a", {"f": {GPU: Quota(12)}}, cohort="c", reclaim_within_cohort="Any")
    b = ClusterQueue("b", {"f": {GPU: Quota(4)}}, cohort="c")
    k = Kueue([a, b])
    k.submit(Workload("b-big", "b", {GPU: 8}, priority=5))
    k.schedule()
    k.submit(Workload("b-small", "b", {GPU: 2}, priority=0))
    k.schedule()
    k.submit(Workload("a1", "a", {GPU: 10}))
    return k.schedule()

def three_queue_events():
    qs = [ClusterQueue(n, {"f": {GPU: Quota(8)}}, cohort="c", **kw)
          for n, kw in (("a", {"reclaim_within_cohort": "Any"}), ("b", {}), ("c", {}))]
    k = Kueue(qs)
    k.submit(Workload("b1", "b", {GPU: 8}), Workload("b2", "b", {GPU: 4}))
    k.schedule()
    k.submit(Workload("c1", "c", {GPU: 8}))
    k.schedule()
    k.submit(Workload("a1", "a", {GPU: 8}))
    return k.schedule()

first, second = reclaim_events(), three_queue_events()
print(first, second, sep="\n")
assert sorted(preempted_first) == sorted(e[1] for e in first if e[0] == "preempted")
assert sorted(preempted_second) == sorted(e[1] for e in second if e[0] == "preempted")
print("✅ reclaim takes only from borrowers, cheapest first, then gives back what it did not need")

# %% [markdown]
# ## In a design review
#
# **Two-minute version.** "Every team gets a LocalQueue that points at its ClusterQueue, with nominal
# GPU quota per flavor: on-demand H100, Spot H100, L4. ClusterQueues share a cohort, so other queues
# borrow idle quota, and none of it goes to waste. With borrowing limits, one team cannot take
# everything. Lending limits keep a minimum at home for latency-sensitive work.
#
# "The reclaimWithinCohort setting recalls borrowed quota, so nominal quota really is a guarantee.
# Without it, nominal quota is only a hope. Inside a queue, WorkloadPriorityClasses let urgent jobs
# preempt the newest, lowest-priority jobs.
#
# "Admission is all-or-nothing per job, and this also makes Kueue our gang admitter. BestEffortFIFO
# keeps GPUs busy, and we look for large jobs that starve behind small jobs."
#
# **Drill questions.**
#
# 1. *Team A has 16 GPUs of nominal quota, uses none, and its 16-GPU job is Pending. Why?* The cohort
#    borrowed its idle quota, and `reclaimWithinCohort` is `Never`. Turn on reclaim or set a
#    `lendingLimit`.
# 2. *What is the difference between `borrowingLimit` and `lendingLimit`?* The borrowing limit sets the
#    maximum that a queue can take above its nominal quota. The lending limit sets the maximum that it
#    gives away (the rest is reserved).
# 3. *Does Kueue place pods on nodes?* No. It admits whole workloads against quota (and, with TAS, it
#    assigns topology domains). kube-scheduler still binds each pod.
