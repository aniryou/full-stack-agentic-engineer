# %% [markdown]
# # 02 · kind with fake GPUs and Kueue: quota, gangs, topology and preemption for real
#
# **Tier:** T0. With no cluster, this notebook runs the bundled predictor and prints the exact `kubectl`
# commands. With a laptop and Docker ($0), run `deploy/kind/up.sh` first. Then the same cells apply each
# scenario to a real kind cluster and compare the result with the prediction.
#
# ## The one-minute version
# Kubernetes schedules a GPU job in **two decisions in series**:
#
# 1. **Kueue** decides *if* the job can start. It looks at the quota in its ClusterQueue and at what the
#    queue can borrow from the cohort. It also looks at preemption by priority or by reclaim. With Topology-Aware Scheduling,
#    Kueue also decides *which topology domain* its pods go to. A queued Job is **suspended** (no pods) until
#    Kueue admits it.
# 2. The **kube-scheduler** binds each pod to a node. It looks at taints, selectors and free
#    `nvidia.com/gpu`.
#
# On kind, the GPUs are fake. Each one is an extended resource that a patch writes into the node status. But
# everything that *decides* is real: kube-scheduler, Kueue v0.19, JobSet, LeaderWorkerSet. The simulation
# covers only the device (no `/dev/nvidia*`, no CUDA). Primer §6 *Queues, quotas and
# multi-tenancy with Kueue*, §4 *Gangs*, §5 *Topology-aware placement*, §10 *Learning locally*.

# %%
import json

from k8sgpu import kindlab, kindsim, scenarios
from k8sgpu import manifests as m

READY, why = kindlab.cluster_status()
LIVE = READY          # set to False to stay on the predictor even when the cluster is up
print(("LIVE on " if LIVE else "T0 (predictor only): ") + why)


def scenario(sid: str) -> None:
    """Predict a scenario; on a live cluster also run it and compare, step by step."""
    if LIVE:
        kindlab.run_scenario(sid)            # deletes the previous scenario's workloads first
    else:
        kindlab.run_scenario(sid, dry_run=True, do_reset=False)

# %% [markdown]
# ## The cluster: 16 fake L4s in GCE-style topology
# Both `fake-gpus.sh` and the predictor read `deploy/kind/topology.txt`. Four workers become 4-GPU "hosts"
# in two subblocks of one block. A fifth worker (with no taint) is the system pool, where Kueue and the
# other add-ons run.

# %%
for n in kindsim.read_topology():
    where = " > ".join(n.labels.get(k, "-") for k in m.GKE_TOPOLOGY_LEVELS[:3])
    taints = ",".join(f"{t['key']}={t['value']}" for t in n.taints) or "-"
    print(f"{n.name:22} gpus={n.gpus}  {where:32}  taints={taints}")

# %% [markdown]
# ## How a GPU becomes schedulable without a GPU
# Usually, a device plugin advertises `nvidia.com/gpu` through the kubelet. Here, we write the number
# directly into the node status. We send one JSON-patch per node to the `status` subresource. In a JSON
# pointer, you write `~1` for a `/` inside a key.
#
# ## Exercise 2.1 — write the patch
# Return the JSON-patch (a list of two `add` operations) that sets both
# `status.capacity["nvidia.com/gpu"]` and `status.allocatable["nvidia.com/gpu"]` to `n`.
# Write `n` as a string, because Kubernetes stores quantities as strings.

# %% exercise
def gpu_capacity_patch(n: int) -> list[dict]:
    ### BEGIN SOLUTION
    key = m.GPU.replace("~", "~0").replace("/", "~1")
    return [{"op": "add", "path": f"/status/{field}/{key}", "value": str(n)}
            for field in ("capacity", "allocatable")]
    ### END SOLUTION

# %% check
patch = gpu_capacity_patch(4)
script = (kindsim.KIND_DIR / "fake-gpus.sh").read_text()
assert json.dumps(patch, separators=(",", ":")).replace('"4"', '"$gpus"') in script.replace('\\"', '"')
print("✅ the same patch fake-gpus.sh sends:", json.dumps(patch))

# %% [markdown]
# ## s1 — a fake GPU is still a scheduling unit
# One Job goes directly to the kube-scheduler (no queue label). The other Job goes through Kueue. All four
# GPU nodes have the same scheduler scores, thus the first Job goes to a random node. Kueue TAS places the
# second Job. TAS packs a workload with **no** topology request where the least free capacity still fits
# (*LeastFreeCapacity*). Thus the second Job goes next to the first.

# %%
scenario("s1")

# %% [markdown]
# ## Quota: nominal, borrowing, and the cohort
# The ClusterQueue of each team owns 8 GPUs (`nominalQuota`). It can borrow up to 4 more (`borrowingLimit`)
# from the *unused* quota of the other team. The two queues are in cohort `gpu-lab`
# (`deploy/kind/manifests/20-kueue-queues.yaml`). Kueue admits a workload if its request fits the
# **available** quota of the queue:
#
# $$
# \begin{aligned}
# \text{available} = \min\big(&\text{nominal} + \texttt{borrowingLimit} - \text{usage}, \\
# &\text{cohort nominal total} - \text{cohort usage}\big)
# \end{aligned}
# $$
#
# (This lab has no lending limits.) If the workload needs more than the unused nominal quota of its own
# queue, it *borrows*.
#
# ## Exercise 2.2 — admit, borrow or wait?
# Write `admission(usage, request, nominal, borrowing_limit, cohort_unused)`. It returns
# `"fits"` (within nominal), `"borrows"` (it fits only if the queue borrows) or `"waits"`.
# `cohort_unused` is the unused nominal quota across the whole cohort. This includes the quota of
# this queue.

# %% exercise
def admission(usage: int, request: int, nominal: int, borrowing_limit: int, cohort_unused: int) -> str:
    ### BEGIN SOLUTION
    available = min(nominal + borrowing_limit - usage, cohort_unused)
    if request > available:
        return "waits"
    return "fits" if usage + request <= nominal else "borrows"
    ### END SOLUTION

# %% check
assert admission(0, 4, 8, 4, 16) == "fits"
assert admission(8, 4, 8, 4, 8) == "borrows"      # s5: a-job-3 borrows team-b's idle GPUs
assert admission(12, 4, 8, 4, 4) == "waits"       # s5: a-job-4, team-a at nominal + borrowingLimit
assert admission(8, 4, 8, 4, 0) == "waits"        # s4's a-high: team-a at nominal, cohort full -> preempt instead
sim = kindsim.new_sim()
assert sim.available(sim.cfg.cqs["team-a-cq"]) == min(8 + 4 - 0, 16)
print("✅ the admission arithmetic Kueue applies before it looks at nodes")

# %% [markdown]
# ## s2 — gangs land whole, inside one topology domain
# A plain Job holds 2 GPUs on `host-a1-1`. Kueue does not manage this Job, but TAS still counts it.
# Then come three JobSets with `kueue.x-k8s.io/podset-required-topology`:
#
# * a 4-pod gang that must share a **host**,
# * an 8-pod gang that must share a **subblock**,
# * a third 4-pod subblock gang that waits until the scenario deletes the blocker. Its quota is sufficient,
#   but no subblock has 4 free GPUs.
#
# For required and preferred levels, TAS uses **BestFit**. Among the domains at that level that can hold
# the whole gang, TAS takes the domain with the *least* room. This keeps large domains whole for large
# gangs. If two domains are equal, the lowest label value wins.

# %%
scenario("s2")

# %% [markdown]
# ### Admission is not placement — except where Kueue checks placement
# Kueue admits against **quota**. Two things cause an admitted gang to also *fit*:
#
# * a **TAS flavor** (the `gpu-l4` flavor of this kind lab). Here, admission selects a domain with free
#   capacity for every pod.
# * a **ProvisioningRequest admission check** (GKE's `l4-flex` flavor, notebook 04). Here, admission waits
#   until DWS has created every node.
#
# A **plain-quota flavor**, such as GKE's `l4-spot` in `deploy/gke/10-kueue-gke.yaml`, does neither.
# Admission means "the quota is yours". Then the pods wait for the cluster autoscaler, and the nodes arrive
# one at a time. A Spot stockout can leave half of the gang Running, and this half holds GPUs while the rest
# is Pending.
#
# The safety net of Kueue is **`waitForPodsReady`**. If not all the pods of an admitted workload are Ready
# within `timeout`, Kueue evicts the workload. This releases the GPUs. Then Kueue requeues the workload with
# exponential backoff.
#
# In Kueue v0.19, it is **on by default**. The v1beta2 Configuration sets these defaults: a 30 min
# `timeout`, a `recoveryTimeout` equal to it, and `blockAdmission: false`
# (`apis/config/v1beta2/defaults.go` at v0.19.6). The shipped config file shows the setting only as a
# comment, as an example of the knobs. To adjust it, edit the `kueue-manager-config` ConfigMap
# (`kueue-system` namespace, key `controller_manager_config.yaml`). Then run
# `kubectl -n kueue-system rollout restart deployment/kueue-controller-manager`:
#
# ```yaml
# waitForPodsReady:
#   timeout: 10m              # a GPU node from zero plus the driver takes minutes: not too short
#   blockAdmission: true      # admit one gang at a time until its pods are Ready
#   requeuingStrategy: {backoffLimitCount: 5}
# ```

# %% [markdown]
# ## Exercise 2.3 — BestFit
# Write `best_fit(capacity, need)` for one level. `capacity` maps a domain name to the number of pods that
# the domain can still take. Return the domain that Kueue selects, or `None` if no single domain fits.

# %% exercise
def best_fit(capacity: dict[str, int], need: int) -> str | None:
    ### BEGIN SOLUTION
    fitting = [(c, name) for name, c in capacity.items() if c >= need]
    return min(fitting)[1] if fitting else None
    ### END SOLUTION

# %% check
assert best_fit({"host-a1-1": 2, "host-a1-2": 4, "host-a2-1": 4, "host-a2-2": 4}, 4) == "host-a1-2"   # s2 gang-host
assert best_fit({"subblock-a1": 2, "subblock-a2": 8}, 8) == "subblock-a2"                             # s2 gang-subblock
assert best_fit({"subblock-a1": 2, "subblock-a2": 0}, 4) is None                                      # s2 gang-waits
assert best_fit({"s1": 8, "s2": 5, "s3": 6}, 5) == "s2"
print("✅ BestFit: the tightest domain that still holds the whole gang")

# %% [markdown]
# ## s3 — LeaderWorkerSet: multi-host replicas, admitted a group at a time
# One replica is a leader pod and a worker pod, with 4 GPUs each (a model sharded across two hosts). The two
# templates have the same `podset-group-name`, thus Kueue places them together in one subblock. A scale-up
# to 2 replicas creates a *second group*: another 8 GPUs. The queue of team-b has no quota left and can
# borrow only 4, thus the whole group waits. Its pods exist, but they have the `kueue.x-k8s.io/admission` scheduling
# gate.

# %%
scenario("s3")

# %% [markdown]
# ## s4 — priority preemption, inside the queue
# Both teams fill their nominal quota. A job with the `high` WorkloadPriorityClass arrives in team-a. It
# cannot borrow, because team-b uses all of its own quota. Thus Kueue looks for victims. With
# `withinClusterQueue: LowerPriority`, the candidates are the lower-priority workloads of team-a.
#
# Kueue v0.19 puts the candidates in this order (`preemption/common/ordering.go`, `CandidatesOrdering`):
#
# 1. workloads that are **already in eviction**. They give their quota back in any case.
# 2. the workloads of **other queues**.
# 3. lower LocalQueue usage, but only with admission fair sharing.
# 4. the **lowest priority**.
# 5. the **most recently admitted**.
#
# (The predictor evicts at once, thus the first rule never separates its candidates. Fair sharing is off in
# this lab.)
#
# ## Exercise 2.4 — order the candidates
# Write `order_victims(candidates, preemptor_queue)`. Each candidate is a dict with `name`, `queue`,
# `priority`, `admitted` and `evicted`. `admitted` is a clock value, and a larger value is later.
# `evicted` is a bool. Return the names in the order that Kueue examines them (fair sharing off).

# %% exercise
def order_victims(candidates: list[dict], preemptor_queue: str) -> list[str]:
    ### BEGIN SOLUTION
    ranked = sorted(candidates, key=lambda c: (not c["evicted"], c["queue"] == preemptor_queue,
                                               c["priority"], -c["admitted"]))
    return [c["name"] for c in ranked]
    ### END SOLUTION

# %% check
s4 = [{"name": "a-low-1", "queue": "team-a-cq", "priority": 100, "admitted": 3, "evicted": False},
      {"name": "a-low-2", "queue": "team-a-cq", "priority": 100, "admitted": 4, "evicted": False}]
assert order_victims(s4, "team-a-cq")[0] == "a-low-2"                  # the most recent low goes first
s5 = [{"name": "a-job-1", "queue": "team-a-cq", "priority": 0, "admitted": 1, "evicted": False},
      {"name": "a-job-3", "queue": "team-a-cq", "priority": 0, "admitted": 3, "evicted": False},
      {"name": "b-other", "queue": "team-b-cq", "priority": 0, "admitted": 5, "evicted": False}]
assert order_victims(s5, "team-b-cq") == ["a-job-3", "a-job-1", "b-other"]   # reclaim: the borrower's newest
s5[1]["evicted"] = False; s5[2]["evicted"] = True                            # b-other is already on its way out
assert order_victims(s5, "team-b-cq")[0] == "b-other"
print("✅ Kueue's victim order: already evicting, other queues, lowest priority, newest admission")

# %%
scenario("s4")

# %% [markdown]
# ## Exercise 2.5 — borrow or preempt?
# Change one fact in s4: team-b is **idle** (do not run its two fill jobs). team-a is at its nominal 8
# with two low-priority jobs. Then `a-high` (4 GPUs) arrives. Does Kueue preempt a low job, or does it let
# `a-high` borrow? Set `answer` to `"preempt"` or `"borrow"`.
#
# Then use the simulator to make sure that your answer is correct. Replay steps 3-5 of s4 on a new
# `kindsim.new_sim()`. Read the placement of `a-high` and the state of `a-low-2`.

# %% exercise
### BEGIN SOLUTION
answer = "borrow"      # a workload that fits by borrowing is admitted; preemption is only tried when it does not fit
### END SOLUTION
sim = kindsim.new_sim()
sc = scenarios.scenario("s4")
for i in (2, 3, 4):   # a-low-1, a-low-2, a-high
    kindsim.run_step(sim, sc, i)
out = sim.outcome()

# %% check
assert answer in ("preempt", "borrow"), 'answer "preempt" or "borrow"'
assert out["Job/team-a/a-high"]["state"] == "admitted"
happened = "borrow" if out["Job/team-a/a-low-2"]["state"] == "admitted" else "preempt"
assert answer == happened, f"the simulator disagrees: a-low-2 is {out['Job/team-a/a-low-2']['state']}"
print(f"✅ {happened}: a-high {out['Job/team-a/a-high']}, a-low-2 {out['Job/team-a/a-low-2']['state']}")

# %% [markdown]
# ## s5 — borrowing is a loan: the lender takes it back
# team-a borrows the idle GPUs of team-b, up to its `borrowingLimit`. When team-b submits work that fits in
# *its own* nominal quota, `reclaimWithinCohort: Any` lets Kueue preempt the borrower, even at equal
# priority. The borrower is the most recently admitted workload of team-a.

# %%
scenario("s5")

# %% [markdown]
# ## s6 — a zoo of Pending pods
# Fillers leave exactly one free GPU on every node. Then come six workloads that cannot start, each for a
# different reason. They are the raw material for notebook 03. The predictor makes the
# `0/N nodes are available: ...` message of the kube-scheduler, in the format of the scheduler.

# %%
scenario("s6")

# %% [markdown]
# ## k1 — placement at fleet scale with KWOK (optional)
# `deploy/kind/kwok.sh` adds 32 fake 8-GPU H100 nodes (2 blocks × 4 subblocks × 4 hosts). No kubelet runs
# these nodes. A 16-host gang takes a whole block. A 4-host gang takes a whole subblock of the other block.
# 8 single-GPU pods with no topology request go together onto one host.

# %%
print(kindsim.describe(kindsim.predict("k1")))

# %% [markdown]
# ## In a design review
# *"How do you share 16 GPUs between two teams and still run 8-GPU gangs?"* The answer, in two minutes:
#
# * Give each team a ClusterQueue with a nominal quota in a shared cohort. Then a team can borrow idle GPUs,
#   and the owner reclaims them when it needs them (`reclaimWithinCohort`).
# * Use priorities only *within* a team.
# * Submit multi-pod jobs as JobSets/LWS through Kueue, so that Kueue admits them whole.
# * Turn on Topology-Aware Scheduling with the placement labels of the cloud. Then a gang goes into one host
#   or subblock (BestFit keeps large domains free).
# * Keep the job of the kube-scheduler simple. The ResourceFlavor adds the tolerations and selectors.
#
# **Drill 1.** *A queued Job shows no pods at all. Is it broken?* No. Kueue keeps it suspended until it
# admits the Job. Read the `QuotaReserved` condition of the Workload.
#
# **Drill 2.** *Why does a 4-pod subblock gang wait while the cluster has 6 free GPUs?* Because of the
# required topology. No single subblock has 4 free GPUs, and TAS does not divide the gang.
#
# **Drill 3.** *Kueue preempted a job of team-a, but nobody in team-a had a higher priority.* The job ran on
# borrowed quota. The lender reclaimed it (`InCohortReclamation`).
#
# **Drill 4.** *Kueue admitted an 8-node gang on a Spot pool. Three pods run, and five are still Pending
# after 20 minutes because of a stockout. What occurs next, and what can prevent this state?*
#
# The flavor is plain quota, thus admission never examined capacity. `waitForPodsReady` (on by default,
# 30 min) evicts the gang, frees the GPUs of the three nodes and requeues the gang with backoff. An
# admission that examines capacity prevents the half-started state. There are two kinds: a
# ProvisioningRequest check (DWS flex-start: every node or none) or, on a fleet with a constant size, TAS.
