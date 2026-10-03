# %% [markdown]
# # 05 · Autoscaling and obtainability
#
# **Tier:** T0. This notebook is a discrete-time simulation of the cluster autoscaler. Durations,
# prices, stockout rates and preemption rates are **inputs that you select** (the defaults are
# illustrative), and every output is simulated. Lab notebook `04_gke_pools_dws_and_computeclasses`
# (T3) does the real thing: GKE node pools from zero, Spot, DWS flex-start queued provisioning and
# ComputeClass fallbacks. Offline, that notebook plans and inspects. Current prices and
# obtainability are in `COMPUTE.md` at the repo root.
#
# ## The one-minute version
#
# A Pending GPU pod is a request for a machine. The **cluster autoscaler** simulates the pending
# pods on the *template node* of each node pool. It estimates how many nodes each pool needs for
# these pods (bin-packing). Then it lets an **expander** select a pool, and it asks the cloud for
# nodes. These nodes become useful only after some minutes, because of the VM, the driver, the
# device plugin, the image pull and the model weights. The autoscaler removes a node after the node
# stays idle for a time (10 minutes by default).
#
# GPUs add two problems that CPUs rarely have. First, **capacity does not always exist** (stockouts
# and Spot reclaims). Second, **gangs need every node at once**: a job that needs 16 nodes and
# receives 11 pays for 11 idle nodes. Queued, all-or-nothing provisioning (Kueue
# ProvisioningRequest, GKE DWS flex-start) is the solution to the second problem. You manage the
# first problem when you select the capacity type for each workload (on-demand, Spot, reservation,
# flex-start).
#
# After this notebook, you can estimate the time-to-capacity and the idle cost. You can also select
# a capacity type for each workload.
#
# Primer: §7 *Getting capacity* and §8 *Startup latency* in `../../PRIMER.md`.

# %%
from gpusched import (Job, NodePool, expected_runtime_h, gang_survival, least_waste, nodes_needed, provision,
                      simulate, startup_latency)

pools = [NodePool("l4-x1", gpus_per_node=1, max_nodes=16), NodePool("l4-x4", gpus_per_node=4, max_nodes=4),
         NodePool("h100-x8", gpus_per_node=8, max_nodes=2)]
for pending in ([1, 1, 1], [3, 3, 3], [4, 2, 1, 1], [8, 8, 8]):
    pool = least_waste(pools, pending)
    print(f"pending GPUs {pending} -> pool {pool.name if pool else None}",
          f"({nodes_needed(pending, pool.gpus_per_node)} node(s))" if pool else "(no pool can take them)")

# %% [markdown]
# Look at the third line: least-waste compares *idle resources*, not prices. For 4 + 2 + 1 + 1 GPUs,
# one 8-GPU H100 node and two 4-GPU L4 nodes have the same waste (zero idle GPUs). The tie-break
# (fewer nodes) selects the H100. The real expander ranks idle CPU, then memory, and it does not
# rank dollars. Thus pods select a GPU type with a node selector. Cost preferences go into
# a `price` or `priority` expander, or into a GKE ComputeClass.
#
# ## Exercise 5.1 — how many new nodes?
#
# The estimate of the autoscaler is a bin-packing of the pending pods onto empty copies of the
# template node of the pool. Implement **first-fit decreasing**:
#
# 1. Sort the pods by GPU count, largest first.
# 2. Put each pod on the first node that has sufficient free GPUs.
# 3. If no node has sufficient free GPUs, open a new node.
#
# If a pod can never fit on one node, raise `ValueError`.

# %% exercise
def ffd_nodes(pod_gpus: list, gpus_per_node: int) -> int:
    ### BEGIN SOLUTION
    free = []
    for g in sorted(pod_gpus, reverse=True):
        if g > gpus_per_node:
            raise ValueError(f"{g} > {gpus_per_node}")
        for i, f in enumerate(free):
            if f >= g:
                free[i] -= g
                break
        else:
            free.append(gpus_per_node - g)
    return len(free)
    ### END SOLUTION

# %% check
for pods, per in (([4, 4, 2, 2, 1, 1, 1, 1], 8), ([5, 5, 5], 8), ([1] * 9, 8), ([3, 3, 3], 4), ([2, 6, 3, 5], 8)):
    assert ffd_nodes(pods, per) == nodes_needed(pods, per), (pods, per)
assert ffd_nodes([5, 5, 5], 8) == 3                   # three 5-GPU pods strand 3 GPUs per node
try:
    ffd_nodes([16], 8)
    raise AssertionError("a 16-GPU pod must be rejected on 8-GPU nodes")
except ValueError:
    pass
print("✅ pod shapes that do not divide the node waste GPUs before anything is even scheduled")

# %% [markdown]
# ## Scale from zero, and back
#
# One L4 node pool has zero nodes. A 1-hour inference job arrives at $t = 0$. The node needs 300 s
# (illustrative) from creation to *allocatable GPUs*.
#
# The autoscaler removes a node after the node stays unneeded for 600 s (its default
# `--scale-down-unneeded-time`). It also does not remove a node within 600 s of the last scale-up
# (`--scale-down-delay-after-add`). `simulate()` models both. Here, the only scale-up is at
# $t = 0$. Thus the unneeded time is the limit that controls the removal. `simulate()` does not
# model the utilisation threshold or several pools.

# %%
l4 = NodePool("l4", gpus_per_node=1, max_nodes=4, boot_s=300, price_per_node_hr=0.70)   # illustrative price
run = simulate(l4, [Job("chat-replica", nodes=1, hours=1.0)], until_s=3 * 3600)
for t, what in run["log"]:
    print(f"t={t:>5}s  {what}")
print(f"billed {run['node_h']:.2f} node-h, busy {run['busy_node_h']:.2f} node-h, cost ${run['cost']:.3f} (simulated)")

# %% [markdown]
# ## Exercise 5.2 — predict the bill
#
# The pool is the same, but the job runs for **2 hours** and the node boots in **420 s**. Predict
# when the job starts. Predict when the autoscaler removes the idle node (here, the autoscaler
# does a check every 30 s). Also predict the billed node-hours.

# %% exercise
# predicted_start_s = ...     seconds
# predicted_removed_s = ...   seconds
# predicted_node_h = ...      node-hours billed
### BEGIN SOLUTION
predicted_start_s = 420                        # waits for the node to boot
predicted_removed_s = 420 + 7200 + 600         # finish + unneeded time (the delay after the t=0 add is long over)
predicted_node_h = predicted_removed_s / 3600  # billed from creation (t=0) to removal
### END SOLUTION

# %% check
pool = NodePool("l4", gpus_per_node=1, max_nodes=4, boot_s=420, price_per_node_hr=0.70)
r = simulate(pool, [Job("chat-replica", nodes=1, hours=2.0)], until_s=4 * 3600)
removed = [t for t, what in r["log"] if what == "idle node removed"]
assert r["jobs"]["chat-replica"][0] == predicted_start_s
assert removed == [predicted_removed_s]
assert abs(r["node_h"] - predicted_node_h) < 1e-9
print(f"✅ {r['node_h']:.3f} node-h billed for {r['busy_node_h']:.1f} h of work - boot and idle tail are overhead")

# %% [markdown]
# ## Spot and gangs
#
# Spot capacity is low-cost because the provider can reclaim it. For one node, a reclaim causes an
# occasional restart. For a gang, the effect multiplies. If the provider reclaims *any* node, the
# collective breaks. Without checkpoints, the whole job then starts again from zero.
#
# Let the reclaims be independent, at rate $\lambda$ per node-hour. Then a gang of $N$ nodes
# survives $T$ hours with probability $e^{-N \cdot \lambda \cdot T}$. If the job loses all of its
# work at each restart, the expected wall-clock time to finish $T$ hours of work is
#
# $$
# \left(e^{N \cdot \lambda \cdot T} - 1\right) \cdot \left(\frac{1}{N \cdot \lambda} + R\right)
# $$
#
# for a restart overhead $R$. The rate in the next cell is an illustrative input, not a measured
# Spot statistic.

# %%
rate = 0.005   # reclaims per node-hour (illustrative)
for n in (1, 4, 16):
    print(f"{n:>2} nodes x 24 h: survives {gang_survival(n, 24, rate):6.1%}, "
          f"expected wall-clock {expected_runtime_h(24, n, rate, restart_h=0.25):5.1f} h")
spot = NodePool("h100-spot", gpus_per_node=8, max_nodes=8, boot_s=300, spot=True)
r = simulate(spot, [Job("finetune", nodes=4, hours=6.0)], until_s=72 * 3600, spot_rate_per_node_hr=0.02, seed=2)
print("\nsimulated 4-node job at 0.02/node-h:", r["jobs"]["finetune"], "(start s, end s, restarts)")
print([w for _, w in r["log"] if "Spot" in w])

# %% [markdown]
# ## Exercise 5.3 — the cost of restarting a gang
#
# Implement `expected_hours(work_h, nodes, rate, restart_h)` from the formula in *Spot and gangs*.
# When the rate is 0, return `work_h`. Then use the function for this question. A **16-node**,
# 24-hour job runs at `rate = 0.005`, with a 15-minute restart overhead. How many times longer
# than the work itself is its wall-clock time? Put the ratio in `slowdown_16`.

# %% exercise
import math

def expected_hours(work_h: float, nodes: int, rate: float, restart_h: float = 0.0) -> float:
    ### BEGIN SOLUTION
    lam = nodes * rate
    if lam == 0:
        return work_h
    return (math.exp(lam * work_h) - 1) * (1 / lam + restart_h)
    ### END SOLUTION

# slowdown_16 = ...   a ratio: expected wall-clock / work hours
### BEGIN SOLUTION
slowdown_16 = expected_hours(24, 16, 0.005, 0.25) / 24
### END SOLUTION

# %% check
assert abs(expected_hours(10, 4, 0.01) - 12.2956) < 1e-3
assert expected_hours(5, 3, 0.0) == 5
assert abs(expected_hours(24, 16, 0.005, 0.25) - expected_runtime_h(24, 16, 0.005, 0.25)) < 1e-9
assert 3.0 < slowdown_16 < 3.2
print(f"✅ a 16-node gang on Spot takes {slowdown_16:.2f}x its work time without checkpoints - "
      "checkpoint often (layer 01, section 7) or run it on non-preemptible capacity")

# %% [markdown]
# ## All-or-nothing provisioning
#
# A 16-node training job asks for capacity when the free capacity of the zone is low. The provider
# grants each node that the job still needs with 3% probability per minute (a stockout model,
# simulated). An
# **ordinary** pool creates each node immediately when the provider grants it. The pool also bills
# the node while the node waits for the other nodes. A **queued** pool (Kueue ProvisioningRequest,
# or DWS flex-start with queued provisioning on GKE) holds the request until it can create all 16
# nodes together.

# %%
kw = dict(gpus_per_node=8, boot_s=300, stockout=0.97)
ordinary = provision(NodePool("a3", **kw), 16, tick_s=60, seed=0)
queued = provision(NodePool("a3-queued", queued=True, **kw), 16, tick_s=60, seed=0)
for name, p in (("ordinary", ordinary), ("queued", queued)):
    print(f"{name:>8}: gang starts at {p['gang_start_s'] / 3600:.2f} h, billed while waiting "
          f"{p['waiting_node_h']:5.1f} node-h = {p['waiting_gpu_h']:6.1f} GPU-h   (one run, seed 0)")
runs = [provision(NodePool("a3", **kw), 16, tick_s=60, seed=seed) for seed in range(200)]
print(f"ordinary, mean of 200 seeds: gang starts at {sum(r['gang_start_s'] for r in runs) / 200 / 3600:.2f} h, "
      f"billed while waiting {sum(r['waiting_node_h'] for r in runs) / 200:.1f} node-h")

# %% [markdown]
# One run is one draw of a random process. Thus, look at the mean also. In this model, the two
# pools have the same capacity and the same start time. The difference is who pays for the wait
# (the queued pool pays only for the boot).
#
# Queued provisioning also prevents a second failure
# mode. If the `max_nodes` of an ordinary pool is less than the gang size, the pool scales up
# *part* of the gang. These nodes then stay idle forever.

# %%
for q in (False, True):
    r = simulate(NodePool("a3", gpus_per_node=8, max_nodes=3, queued=q), [Job("train", 4, 1.0)], until_s=7200)
    print("queued  " if q else "ordinary", "->", r["log"][:2], f"billed {r['node_h']:.1f} node-h")

# %% [markdown]
# ## Exercise 5.4 — billed while waiting
#
# You have the creation times of the nodes of a gang (seconds) and the boot time. Calculate the
# node-hours billed before the gang can start. The gang starts when the **last** node is Ready
# (`max(created) + boot_s`). Each node bills from its own creation until that time.

# %% exercise
def waiting_node_hours(created_s: list, boot_s: int) -> float:
    ### BEGIN SOLUTION
    start = max(created_s) + boot_s
    return sum(start - c for c in created_s) / 3600
    ### END SOLUTION

# %% check
assert waiting_node_hours([0, 0], 300) == 600 / 3600
assert waiting_node_hours([0, 3600], 300) == (3900 + 300) / 3600
assert abs(waiting_node_hours(ordinary["created_s"], 300) - ordinary["waiting_node_h"]) < 1e-9
assert abs(waiting_node_hours(queued["created_s"], 300) - 16 * 300 / 3600) < 1e-9
print("✅ the ordinary pool paid for", round(ordinary["waiting_node_h"] - queued["waiting_node_h"], 1),
      "extra node-hours of idle H100s")

# %% [markdown]
# ## Exercise 5.5 — pick a capacity type
#
# For each workload in the list, select one of `"on-demand"`, `"spot"`, `"flex-start"` or
# `"reservation"`. `"flex-start"` is DWS: queued, all at once, time-bounded. `"reservation"` is
# capacity held for you, and you pay for it if you use it or not. The workloads are:
#
# * **chat**: interactive inference. It scales from 0 to 6 L4 replicas with the traffic. The
#   replicas are stateless.
# * **finetune**: 64 H100s for 3 days. It starts at some time this week. It writes a checkpoint
#   every hour.
# * **embeddings**: a nightly batch over a corpus. You can retry any pod. The deadline is morning.
# * **flagship**: 24/7 serving at steady high utilisation for a year.

# %% exercise
# capacity = {"chat": ..., "finetune": ..., "embeddings": ..., "flagship": ...}
### BEGIN SOLUTION
capacity = {
    "chat": "on-demand",          # must appear in minutes when traffic comes; can add a Spot tier behind it
    "finetune": "flex-start",     # a big gang with a flexible start: atomic, time-bounded, no idle partial nodes
    "embeddings": "spot",         # interruptible, retryable, cheapest
    "flagship": "reservation",    # steady load: guaranteed capacity, pay for it and use it
}
### END SOLUTION

# %% check
import hashlib


def digest(key, value):  # the check compares digests, so the blank does not print the answer
    value = tuple(value) if isinstance(value, (list, tuple)) else value
    return hashlib.sha256(f"{key}={value!r}".encode()).hexdigest()[:10]


ACCEPTED = {'chat': 'fa5a7fda80', 'finetune': 'af396b8373', 'embeddings': '139b974c15', 'flagship': '882d122ea0'}
HINTS = {
    "chat": "it must scale up within minutes when traffic arrives, and it cannot wait in a queue",
    "finetune": "a big gang whose start can slip a few days: which type avoids paying for half-provisioned nodes?",
    "embeddings": "retryable pods and a deadline hours away: which type is cheapest when it can be taken back?",
    "flagship": "steady high utilisation for a year: what guarantees the capacity is there every day?",
}
assert set(capacity) == set(ACCEPTED), "answer for exactly: " + ", ".join(ACCEPTED)
for workload, want in ACCEPTED.items():
    assert digest(workload, capacity[workload]) == want, f"{workload}: not quite. {HINTS[workload]}"
print("✅", capacity)

# %% [markdown]
# On GKE, a **custom ComputeClass** encodes such a preference list for one workload class. An
# example order is reservation, then Spot, then on-demand, then flex-start. Node auto-provisioning
# then creates pools that match the list. The field names are in the `deploy/gke/` manifests of
# the lab. Make sure that they agree with the current CRD.
#
# ## From Pending to serving
#
# Even when capacity exists, a new replica is not useful until it completes every stage in the
# next cell. The inputs are illustrative. Measure your own values. Lab notebook
# `04_gke_pools_dws_and_computeclasses` shows where each stage appears in the node events and the
# pod events.

# %%
# sizes in GB, rates in GB/s (gigabytes: a 10 Gbit/s link moves at most 1.25 GB/s)
cold = startup_latency(node_s=150, driver_s=90, image_gb=12, pull_GBps=0.25, weights_gb=16, load_GBps=0.5, warmup_s=60)
warm = startup_latency(image_gb=12, pull_GBps=2.0, weights_gb=16, load_GBps=4.0, warmup_s=60)
for name, s in (("cold node, plain pull + download", cold), ("warm node, streamed image + fast weights", warm)):
    print(f"{name:<42}", {k: round(v) for k, v in s.items()})

# %% [markdown]
# Each lever applies to a stage:
#
# * A warm pool or a low `min_nodes` removes `node`/`driver`.
# * Image streaming, or a secondary boot disk with the image preloaded, decreases `image`.
# * Weights from a nearby cache (GCS FUSE with caching, Hyperdisk ML, a model streamer) decrease
#   `weights`.
# * Startup probes keep traffic away until `warmup` is complete.
#
# Primer §8 has the details.
#
# ## In a design review
#
# **Two-minute version.** "Our GPU pools scale from zero with the cluster autoscaler. Thus a
# Pending pod must wait for node creation, driver install, image pull and weight load before it
# serves. This wait is minutes, not seconds. We set min nodes, image streaming and weight caching for our
# cold-start budget. Scale-down waits for ten minutes of idle time, and we pay for that time.
#
# "We have no guarantee of capacity. Stateless inference runs on on-demand capacity, with a Spot tier
# behind it. Interruptible batch runs on Spot. Large multi-node jobs go through Kueue with a
# ProvisioningRequest. Thus they get all their nodes at once from DWS flex-start, and they do not
# pay for a partial gang. Steady baseline load stays on a reservation.
#
# "A ComputeClass gives that fallback order for each workload class."
#
# **Drill questions.**
#
# 1. *Why can a 16-node training job not use an autoscaling Spot pool?* Any reclaim
#    restarts the gang. Survival decreases as $e^{-N \cdot \lambda \cdot T}$. Partial scale-ups
#    also bill idle nodes. Use queued provisioning (flex-start) or reservations. Also write
#    checkpoints.
# 2. *A replica needs 9 minutes from Pending to Ready. Where do you look first?* Divide the time
#    by stage: node provisioning and driver, image pull, weight download, warm-up. Then repair the
#    largest stage.
# 3. *What does all-or-nothing provisioning not repair?* It does not create capacity. The gang
#    still waits for the provider. It only stops two things: the payment for the partial set, and
#    the fragmentation of that set.
