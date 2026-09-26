# %% [markdown]
# # 02 · kind with fake GPUs and Kueue: quota, gangs, topology and preemption for real
#
# **Tier:** T0 — with no cluster this notebook runs the bundled predictor and prints the exact
# `kubectl` commands. With a laptop and Docker ($0) run `deploy/kind/up.sh` first: the same cells
# then apply each scenario to a real kind cluster and compare what happened with the prediction.
#
# ## The one-minute version
# Scheduling a GPU job on Kubernetes is **two decisions in series**:
#
# 1. **Kueue** decides *whether* it may start — quota in its ClusterQueue, borrowing from the
#    cohort, preemption by priority or by reclaim — and, with Topology-Aware Scheduling, *which
#    topology domain* its pods go to. A queued Job is **suspended** (no pods) until admitted.
# 2. The **kube-scheduler** binds each pod to a node: taints, selectors, free `nvidia.com/gpu`.
#
# On kind the GPUs are fake — an extended resource patched into node status — but everything
# that *decides* is real: kube-scheduler, Kueue v0.19, JobSet, LeaderWorkerSet. What is
# simulated is only the device (no `/dev/nvidia*`, no CUDA). Primer §6 *Queues, quotas and
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
# `deploy/kind/topology.txt` is read by both `fake-gpus.sh` and the predictor. Four workers
# become 4-GPU "hosts" in two subblocks of one block; a fifth (untainted) worker is the system
# pool where Kueue and friends run.

# %%
for n in kindsim.read_topology():
    where = " > ".join(n.labels.get(k, "-") for k in m.GKE_TOPOLOGY_LEVELS[:3])
    taints = ",".join(f"{t['key']}={t['value']}" for t in n.taints) or "-"
    print(f"{n.name:22} gpus={n.gpus}  {where:32}  taints={taints}")

# %% [markdown]
# ## How a GPU becomes schedulable without a GPU
# A device plugin normally advertises `nvidia.com/gpu` through the kubelet. Here we write the
# number into the node's status directly — one JSON-patch per node, sent to the `status`
# subresource. In a JSON pointer, `/` inside a key is spelled `~1`.
#
# ## Exercise 2.1 — write the patch
# Return the JSON-patch (a list of two `add` operations) that sets both
# `status.capacity["nvidia.com/gpu"]` and `status.allocatable["nvidia.com/gpu"]` to `n`
# (as a string, the way Kubernetes stores quantities).

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
# One Job goes straight to the kube-scheduler (no queue label); one goes through Kueue. All four
# GPU nodes tie on the scheduler's scores, so the first lands on a random one. The second is
# placed by Kueue TAS, and a workload with **no** topology request is packed where the least
# free capacity still fits (*LeastFreeCapacity*): next to the first.

# %%
scenario("s1")

# %% [markdown]
# ## Quota: nominal, borrowing, and the cohort
# Each team's ClusterQueue owns 8 GPUs (`nominalQuota`) and may borrow up to 4 more
# (`borrowingLimit`) from the other team's *unused* quota; both sit in cohort `gpu-lab`
# (`deploy/kind/manifests/20-kueue-queues.yaml`). Kueue admits a workload if its request fits
# the queue's **available** quota:
#
# `available = min(nominal + borrowingLimit - usage,  cohort nominal total - cohort usage)`
#
# (no lending limits here). If it needs more than its own unused nominal quota, it *borrows*.
#
# ## Exercise 2.2 — admit, borrow or wait?
# Write `admission(usage, request, nominal, borrowing_limit, cohort_unused)` returning
# `"fits"` (within nominal), `"borrows"` (fits only by borrowing) or `"waits"`.
# `cohort_unused` = unused nominal quota across the whole cohort, including this queue's own.

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
# A plain Job pins 2 GPUs on `host-a1-1` (Kueue does not manage it, but TAS still counts it).
# Then three JobSets with `kueue.x-k8s.io/podset-required-topology`:
# a 4-pod gang that must share a **host**, an 8-pod gang that must share a **subblock**, and a
# third 4-pod subblock gang that waits — quota is fine, but no subblock has 4 free GPUs — until
# the blocker is deleted.
#
# For required and preferred levels TAS uses **BestFit**: among the domains at that level that
# can hold the whole gang, take the one with the *least* room (keep big domains whole for big
# gangs); ties go to the lowest label value.

# %%
scenario("s2")

# %% [markdown]
# ### Admission is not placement — except where Kueue checks placement
# Kueue admits against **quota**. Two things make an admitted gang also *fit*: a **TAS flavor**
# (this kind lab's `gpu-l4`), where admission picks a domain with free capacity for every pod,
# and a **ProvisioningRequest admission check** (GKE's `l4-flex` flavor, notebook 04), where
# admission waits until DWS has created every node. A **plain-quota flavor** — GKE's `l4-spot` in
# `deploy/gke/10-kueue-gke.yaml` — promises neither: admission means "the quota is yours", the
# pods then wait for the cluster autoscaler, nodes arrive one at a time, and a Spot stockout can
# leave half the gang Running and holding GPUs while the rest is Pending.
#
# Kueue's safety net is **`waitForPodsReady`**: if an admitted workload's pods are not all Ready
# within `timeout`, Kueue evicts it (releasing the GPUs) and requeues it with exponential backoff.
# In Kueue v0.19 it is **on by default** — the v1beta2 Configuration defaults it to a 30 min
# `timeout`, `recoveryTimeout` equal to it and `blockAdmission: false`
# (`apis/config/v1beta2/defaults.go` at v0.19.6; the shipped config file only shows it commented
# out, as an example of the knobs). Tune it in the `kueue-manager-config` ConfigMap
# (`kueue-system` namespace, key `controller_manager_config.yaml`), then
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
# Write `best_fit(capacity, need)` for one level: `capacity` maps a domain name to how many
# pods it can still take; return the domain Kueue picks, or `None` if no single domain fits.

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
# One replica = a leader and a worker pod, 4 GPUs each (a model sharded across two hosts). Both
# templates carry the same `podset-group-name`, so Kueue places them together in one subblock.
# Scaling to 2 replicas creates a *second group*: another 8 GPUs. team-b has none left and may
# borrow only 4, so the whole group waits — its pods exist but carry the
# `kueue.x-k8s.io/admission` scheduling gate.

# %%
scenario("s3")

# %% [markdown]
# ## s4 — priority preemption, inside the queue
# Both teams fill their nominal quota. A `high` WorkloadPriorityClass job arrives in team-a: it
# cannot borrow (team-b uses all of its own quota), so Kueue looks for victims. With
# `withinClusterQueue: LowerPriority`, candidates are team-a's lower-priority workloads. Kueue
# v0.19 orders candidates (`preemption/common/ordering.go`, `CandidatesOrdering`): workloads
# **already being evicted** first (they are giving their quota back anyway), then **other
# queues'** workloads, then — only with admission fair sharing — lower LocalQueue usage, then
# **lowest priority**, then **most recently admitted**. (The predictor evicts instantly, so the
# first rule never separates its candidates, and fair sharing is off in this lab.)
#
# ## Exercise 2.4 — order the candidates
# Write `order_victims(candidates, preemptor_queue)` where each candidate is a dict with
# `name`, `queue`, `priority`, `admitted` (a clock value; larger = later) and `evicted` (bool).
# Return the names in the order Kueue considers them (fair sharing off).

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
# Change one fact in s4: team-b is **idle** (skip its two fill jobs). team-a is at its nominal 8
# with two low-priority jobs; `a-high` (4 GPUs) arrives. Does Kueue preempt a low job or let
# `a-high` borrow? Set `answer` to `"preempt"` or `"borrow"`, then confirm with the simulator:
# replay steps 3-5 of s4 on a fresh `kindsim.new_sim()` and read `a-high`'s placement and
# `a-low-2`'s state.

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
# team-a borrows team-b's idle GPUs up to its `borrowingLimit`. When team-b submits work that
# fits in *its own* nominal quota, `reclaimWithinCohort: Any` lets Kueue preempt the borrower —
# the most recently admitted of team-a's workloads — even at equal priority.

# %%
scenario("s5")

# %% [markdown]
# ## s6 — a zoo of Pending pods
# Fillers leave exactly one free GPU on every node; then six workloads that cannot start, each for
# a different reason — the raw material for notebook 03. The predictor produces the kube-
# scheduler's `0/N nodes are available: ...` message in the scheduler's own format.

# %%
scenario("s6")

# %% [markdown]
# ## k1 — placement at fleet scale with KWOK (optional)
# `deploy/kind/kwok.sh` adds 32 fake 8-GPU H100 nodes (2 blocks × 4 subblocks × 4 hosts) that
# no kubelet runs. A 16-host gang takes a whole block, a 4-host gang a whole subblock of the
# other block, and 8 single-GPU pods with no topology request are packed onto one host.

# %%
print(kindsim.describe(kindsim.predict("k1")))

# %% [markdown]
# ## In a design review
# *"How do you share 16 GPUs between two teams and still run 8-GPU gangs?"* — in two minutes:
# give each team a ClusterQueue with a nominal quota in a shared cohort, so idle GPUs are
# borrowed but reclaimed when the owner needs them (`reclaimWithinCohort`), and use priorities
# only *within* a team. Submit multi-pod jobs as JobSets/LWS through Kueue so they are admitted
# whole; turn on Topology-Aware Scheduling with the cloud's placement labels so a gang lands in
# one host or subblock (BestFit keeps big domains free). Keep the kube-scheduler's job simple:
# tolerations and selectors are injected by the ResourceFlavor.
#
# **Drill 1.** *A queued Job shows no pods at all. Broken?* No — Kueue keeps it suspended until
# admitted; read the Workload's `QuotaReserved` condition.
#
# **Drill 2.** *Why does a 4-pod subblock gang wait while the cluster has 6 free GPUs?* Required
# topology: no single subblock has 4 free; TAS will not split the gang.
#
# **Drill 3.** *team-a's job was preempted though nobody in team-a had higher priority.* It was
# running on borrowed quota; the lender reclaimed it (`InCohortReclamation`).
#
# **Drill 4.** *Kueue admitted an 8-node gang on a Spot pool, three pods run and five have been
# Pending for 20 minutes on a stockout. What happens next, and what would have avoided it?* The
# flavor is plain quota, so admission never checked capacity. `waitForPodsReady` (on by default,
# 30 min) evicts the gang, frees the three nodes' GPUs and requeues it with backoff. Admission
# that checks capacity avoids the half-started state: a ProvisioningRequest check (DWS
# flex-start: every node or none) or, on a fixed fleet, TAS.
