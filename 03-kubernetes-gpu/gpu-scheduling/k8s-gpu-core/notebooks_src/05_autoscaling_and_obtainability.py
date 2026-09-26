# %% [markdown]
# # 05 · Autoscaling and obtainability
#
# **Tier:** T0 — a discrete-time cluster-autoscaler simulation. Durations, prices, stockout and
# preemption rates are **inputs you choose** (defaults are illustrative), and every output is
# simulated. The real thing — GKE node pools from zero, Spot, DWS flex-start queued provisioning,
# ComputeClass fallbacks — is lab notebook `04_gke_pools_dws_and_computeclasses` (T3; offline it
# plans and inspects). Current prices and obtainability live in `COMPUTE.md` at the repo root.
#
# ## The one-minute version
#
# A Pending GPU pod is a request for a machine. The **cluster autoscaler** simulates the pending
# pods on each node pool's *template node*, estimates how many nodes each pool would need
# (bin-packing), lets an **expander** pick a pool, and asks the cloud for nodes. Those nodes take
# minutes to become useful — VM, driver, device plugin, image pull, model weights — and are removed
# after sitting idle (10 minutes by default). GPUs add two problems CPUs rarely have: **capacity may
# not exist** (stockouts; Spot reclaims) and **gangs need every node at once** — a job needing 16
# nodes that receives 11 pays for 11 idle nodes. Queued, all-or-nothing provisioning (Kueue
# ProvisioningRequest, GKE DWS flex-start) fixes the second; choosing the capacity type per workload
# (on-demand, Spot, reservation, flex-start) is how you manage the first. After this notebook you can
# estimate time-to-capacity and idle cost, and pick a capacity type per workload.
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
# Look at the third line: least-waste compares *idle resources*, not prices. For 4 + 2 + 1 + 1 GPUs one
# 8-GPU H100 node and two 4-GPU L4 nodes are equally wasteful (zero idle GPUs), and the tie-break (fewer
# nodes) picks the H100. (The real expander ranks idle CPU, then memory; either way, not dollars.) Pods
# therefore select a GPU type with a node selector, and cost preferences go into a `price` or `priority`
# expander or a GKE ComputeClass.
#
# ## Exercise 5.1 — how many new nodes?
#
# The autoscaler's estimate is a bin-packing of the pending pods onto empty copies of the pool's
# template node. Implement **first-fit decreasing**: sort pods by GPU count, largest first; put each
# on the first node with enough free GPUs, else open a new node. Raise `ValueError` if a pod can never
# fit one node.

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
# One L4 node pool at zero nodes, a 1-hour inference job arrives at t = 0. The node takes 300 s
# (illustrative) from creation to *allocatable GPUs*; the autoscaler removes a node after it has
# been unneeded for 600 s (its default `--scale-down-unneeded-time`), and not within 600 s of the last
# scale-up (`--scale-down-delay-after-add`). `simulate()` models both; here the only scale-up is at
# t = 0, so the unneeded time is what binds. (Not modelled: the utilisation threshold, several pools.)

# %%
l4 = NodePool("l4", gpus_per_node=1, max_nodes=4, boot_s=300, price_per_node_hr=0.70)   # illustrative price
run = simulate(l4, [Job("chat-replica", nodes=1, hours=1.0)], until_s=3 * 3600)
for t, what in run["log"]:
    print(f"t={t:>5}s  {what}")
print(f"billed {run['node_h']:.2f} node-h, busy {run['busy_node_h']:.2f} node-h, cost ${run['cost']:.3f} (simulated)")

# %% [markdown]
# ## Exercise 5.2 — predict the bill
#
# Same pool, but the job runs **2 hours** and the node boots in **420 s**. Predict when the job
# starts, when the idle node is removed (the autoscaler checks every 30 s here), and the billed
# node-hours.

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
# Spot capacity is cheap because the provider can reclaim it. For one node that is an occasional
# restart. For a gang it compounds: if *any* node is reclaimed the collective breaks and, without
# checkpoints, the whole job starts over. With independent reclaims at rate λ per node-hour, a gang
# of N nodes survives T hours with probability `exp(-N·λ·T)`, and the expected wall-clock time to
# finish T hours of work restarting from scratch is `(exp(N·λ·T) − 1)·(1/(N·λ) + R)` for a restart
# overhead R. The rate below is an illustrative input, not a measured Spot statistic.

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
# Implement `expected_hours(work_h, nodes, rate, restart_h)` from the formula above (return
# `work_h` when the rate is 0), then use it: how many times longer than the work itself does a
# **16-node**, 24-hour job take at `rate = 0.005`, with a 15-minute restart overhead? Store it in
# `slowdown_16`.

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
# A 16-node training job asks for capacity while the zone is tight: each missing node is granted
# with 3% probability per minute (a stockout model, simulated). An **ordinary** pool creates each
# node as soon as it is granted — and bills it while it waits for the rest. A **queued** pool
# (Kueue ProvisioningRequest; on GKE, DWS flex-start with queued provisioning) holds the request
# until all 16 can be created together.

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
# One run is one draw of a random process, so look at the mean too. Same capacity, same start time in
# this model — the difference is who pays for the wait (the queued pool pays only the boot). There is
# a second failure mode queued provisioning prevents: an ordinary pool whose `max_nodes` is below
# the gang size scales up *part* of the gang, and those nodes idle forever.

# %%
for q in (False, True):
    r = simulate(NodePool("a3", gpus_per_node=8, max_nodes=3, queued=q), [Job("train", 4, 1.0)], until_s=7200)
    print("queued  " if q else "ordinary", "->", r["log"][:2], f"billed {r['node_h']:.1f} node-h")

# %% [markdown]
# ## Exercise 5.4 — billed while waiting
#
# Given the creation times of a gang's nodes (seconds) and the boot time, compute the node-hours
# billed before the gang can start. The gang starts when the **last** node is Ready
# (`max(created) + boot_s`); every node bills from its own creation until then.

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
# For each workload choose one of `"on-demand"`, `"spot"`, `"flex-start"` (DWS: queued, all at once,
# time-bounded), `"reservation"` (capacity held for you, paid whether used or not):
#
# * **chat** — interactive inference, scales 0 → 6 L4 replicas with traffic; replicas are stateless.
# * **finetune** — 64 H100s for 3 days, starts sometime this week, checkpoints hourly.
# * **embeddings** — nightly batch over a corpus; any pod can be retried; deadline is morning.
# * **flagship** — 24/7 serving at steady high utilisation for a year.

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
# On GKE a **custom ComputeClass** encodes such a preference list for one workload class —
# e.g. reservation → Spot → on-demand → flex-start — and node auto-provisioning creates pools to
# match (field names in the lab's `deploy/gke/` manifests; verify against the current CRD).
#
# ## From Pending to serving
#
# Even when capacity exists, a new replica is not useful until every stage below is done. The
# inputs are illustrative; measure yours (lab notebook `04_gke_pools_dws_and_computeclasses`
# shows where each shows up in node and pod events).

# %%
# sizes in GB, rates in GB/s (gigabytes: a 10 Gbit/s link moves at most 1.25 GB/s)
cold = startup_latency(node_s=150, driver_s=90, image_gb=12, pull_GBps=0.25, weights_gb=16, load_GBps=0.5, warmup_s=60)
warm = startup_latency(image_gb=12, pull_GBps=2.0, weights_gb=16, load_GBps=4.0, warmup_s=60)
for name, s in (("cold node, plain pull + download", cold), ("warm node, streamed image + fast weights", warm)):
    print(f"{name:<42}", {k: round(v) for k, v in s.items()})

# %% [markdown]
# The levers map to stages: a warm pool or low `min_nodes` removes `node`/`driver`; image streaming
# or a secondary boot disk with the image preloaded shrinks `image`; weights from a nearby cache
# (GCS FUSE with caching, Hyperdisk ML, a model streamer) shrink `weights`; startup probes keep
# traffic away until `warmup` is done. Primer §8 has the details.
#
# ## In a design review
#
# **Two-minute version.** "GPU pools scale from zero with the cluster autoscaler, so a Pending pod
# costs node creation, driver install, image pull and weight loading before it serves — minutes,
# not seconds — and we size min nodes, image streaming and weight caching to our cold-start budget.
# Scale-down waits ten minutes of idleness, which we pay for. Capacity is not guaranteed: stateless
# inference runs on on-demand with a Spot tier behind it; interruptible batch runs on Spot; big
# multi-node jobs go through Kueue with a ProvisioningRequest so they get all their nodes at once
# from DWS flex-start instead of paying for a partial gang; steady baseline load sits on a
# reservation. A ComputeClass expresses that fallback order per workload class."
#
# **Drill questions.**
#
# 1. *Why can't a 16-node training job just use an autoscaling Spot pool?* — Any reclaim restarts
#    the gang; survival falls as exp(−N·λ·T), and partial scale-ups bill idle nodes. Use queued
#    provisioning (flex-start) or reservations, and checkpoint.
# 2. *A replica takes 9 minutes from Pending to Ready. Where do you look first?* — Split it by stage:
#    node provisioning and driver, image pull, weight download, warm-up; fix the largest.
# 3. *What does all-or-nothing provisioning not fix?* — It does not create capacity: the gang still
#    waits for the provider; it only stops you paying for, and fragmenting, the partial set.
