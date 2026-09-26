# %% [markdown]
# # 05 · Sharing a GPU, and reading its health
#
# **Tier:** T0 (CPU, simulated). On a real cluster, the lab's `06_gpu_sharing_and_dcgm_on_gke`
# (T3) configures MIG and time-sharing node pools and reads DCGM metrics. Offline, it parses
# sample exporter output.
#
# ## The one-minute version
# * There are three ways to share one GPU. **MIG** partitions it into up to 7 isolated instances
#   with fixed shapes. **MPS** runs several processes' kernels on the SMs at the same time: it
#   is efficient but weakly isolated. **Time-slicing** gives each process the whole GPU in turns:
#   there is no isolation, and latency multiplies with the number of busy tenants.
# * MIG shapes are constrained: 7 compute slices, 8 memory slices, and fixed start positions.
#   `3g + 3g + 1g` does **not** fit, even though 3 + 3 + 1 = 7.
# * For LLM serving, the best "sharing" is usually **one engine per GPU batching many requests**.
#   Continuous batching shares the weight reads, which no GPU-level mechanism can do.
# * **GPU utilization** is the fraction of *time* any kernel is running. One small kernel on 8
#   of 132 SMs reads 100%. Judge load by SM active, tensor active and DRAM active.
# * **XIDs** are triaged by who acts: the app owner (13, 31, 43), the node operator (63, 94) or
#   the hardware path (48, 79, 95).
#
# Primer: §7 *Sharing a GPU* and §8 *Health and observability* (`../../PRIMER.md`).

# %%
import itertools

import numpy as np

from gpusim import health as H
from gpusim import sharing as S

gpu = "H100-80GB"
print(f"MIG profiles on {gpu} (verify with `nvidia-smi mig -lgipp`):")
for p in S.MIG[gpu]:
    print(f"  {p.name:>8}: {p.compute}/7 compute, {p.mem_slices}/8 memory, may start at slices {p.starts}, max {p.max_count}")
for mix in (["4g.40gb", "2g.20gb", "1g.10gb"], ["3g.40gb", "2g.20gb", "1g.10gb", "1g.10gb"], ["3g.40gb", "3g.40gb", "1g.10gb"]):
    pl = S.pack(gpu, mix)
    print(f"\n{' + '.join(mix)} ->", "does not fit" if pl is None else S.layout(gpu, pl))

# %% [markdown]
# Why does `3g + 3g + 1g` fail? A 3g instance takes 3 compute slices but **4 memory slices**, and
# it may only start at slice 0 or 4. Two of them use all 8 memory slices, so the seventh compute
# slice has no memory left to pair with. Creation order matters too: create small instances
# first, without choosing where they go, and they can block the big ones:

# %%
arrivals = ["1g.10gb", "1g.10gb", "1g.10gb", "4g.40gb"]
ff = S.first_fit(gpu, arrivals)
print("first-fit, in arrival order:", ff)
print("  ", S.layout(gpu, ff))
print("planned for the whole mix  :", S.pack(gpu, arrivals))
print("  ", S.layout(gpu, S.pack(gpu, arrivals)))

# %% [markdown]
# ## Exercise 5.1: validate and search MIG layouts
#
# Write `valid_layout(gpu, placement)` for a list of `(profile_name, start)` pairs. Every start
# must be one the profile allows (`p.starts`), and no two instances may share a memory slice
# (instance `p` at `s` occupies memory slices `s .. s + p.mem_slices - 1`). Then write
# `fits(gpu, requests)`, which brute-forces every combination of allowed starts and returns
# True if any is valid.

# %% exercise
profiles = {g: {p.name: p for p in S.MIG[g]} for g in S.MIG}

def valid_layout(gpu, placement):
    ### BEGIN SOLUTION
    used = set()
    for name, start in placement:
        p = profiles[gpu][name]
        if start not in p.starts:
            return False
        span = set(range(start, start + p.mem_slices))
        if used & span:
            return False
        used |= span
    return True
    ### END SOLUTION

def fits(gpu, requests):
    ### BEGIN SOLUTION
    options = [profiles[gpu][n].starts for n in requests]
    return any(valid_layout(gpu, list(zip(requests, starts))) for starts in itertools.product(*options))
    ### END SOLUTION

# %% check
assert valid_layout(gpu, [("4g.40gb", 0), ("3g.40gb", 4)])
assert not valid_layout(gpu, [("4g.40gb", 4)])                       # 4g may only start at 0
assert not valid_layout(gpu, [("2g.20gb", 0), ("1g.20gb", 0)])       # overlap
mixes = [["3g.40gb", "3g.40gb", "1g.10gb"], ["4g.40gb", "4g.40gb"], ["2g.20gb"] * 3 + ["1g.10gb"],
         ["1g.20gb"] * 4 + ["1g.10gb"], ["1g.10gb"] * 7, ["1g.10gb"] * 8, ["3g.40gb", "2g.20gb", "1g.20gb"],
         ["4g.40gb", "1g.20gb", "1g.10gb", "1g.10gb"], ["7g.80gb", "1g.10gb"]]
for m in mixes:
    assert fits(gpu, m) == (S.pack(gpu, m) is not None), m
print("✅ your brute force agrees with the planner on", len(mixes), "mixes")

# %% [markdown]
# ## Turns, overlap or partitions: what latency does a tenant see?
# The model is idealised: it counts SM capacity only, while memory bandwidth and L2 contention
# would make MPS and time-slicing worse. One request needs 10 ms of GPU time when alone. `util`
# is the share of the GPU its kernels can fill. Small-batch decode of a small model is far below
# 1, and a big prefill is about 1. There are four busy tenants:

# %%
for util in (0.2, 1.0):
    lat = {m: S.shared_latency(10.0, 4, m, util=util) for m in ("exclusive", "time_slicing", "mps", "mig")}
    print(f"util={util}: " + "  ".join(f"{m} {v:5.1f} ms" for m, v in lat.items()))
print("\nwhat each mechanism guarantees:")
for mode, t in S.TRAITS.items():
    print(f"  {mode:>12}: memory {t['memory isolation']}; faults {t['fault isolation']}")

# %% [markdown]
# * With **small kernels** (util 0.2), MPS overlaps four tenants at no cost, because together
#   they still fit on the SMs. Time-slicing makes each wait for the others' turns (about 3.8x).
#   MIG gives each tenant a fixed 1/7 of the GPU (1.4x) but a *guaranteed* one.
# * With **saturating kernels** (util 1.0), nothing can make four tenants faster than 4x. MIG's
#   fixed slice (7x) is the price of isolation.
#
# ## Exercise 5.2: time-slicing latency from first principles
#
# The GPU runs one context at a time, round-robin, for a quantum `q` each, and pays `s` for every
# context switch. Our request needs `W` of GPU time, so `k = ceil(W/q)` quanta. The other `N - 1`
# tenants always have work. Between two of our quanta, each of them runs a quantum, with a switch
# before each, plus one more switch back to us. If our first quantum starts right away, when do
# we finish? Write `timeslice_best(W, N, q, s)`.

# %% exercise
def timeslice_best(W, N, q=2.0, s=0.05):
    ### BEGIN SOLUTION
    if N <= 1:
        return W
    k = int(np.ceil(W / q))
    gap = (N - 1) * (q + s) + s
    return W + (k - 1) * gap
    ### END SOLUTION

# %% check
for W in (0.5, 2.0, 7.0, 10.0, 33.0):
    for N in (1, 2, 4, 8):
        assert np.isclose(timeslice_best(W, N), S.timeslice_latency(W, N)["best"]), (W, N)
print(f"✅ 10 ms of work with 4 busy tenants finishes after {timeslice_best(10, 4):.1f} ms at best:"
      " latency scales with the number of busy tenants")

# %% [markdown]
# ## Exercise 5.3: pick the sharing mode
#
# For each situation pick `"mig"`, `"mps"`, `"time_slicing"` or `"none"` (whole GPUs, no
# sharing). Remember that MIG exists only on A100/A30/H100/H200/B200-class GPUs, and not on L4
# or T4.
#
# * **a.** Three customers' inference endpoints with latency SLOs on one H100. A noisy customer
#   must not affect the others.
# * **b.** One team runs six small embedding-model replicas on an L4, each keeping about 10% of
#   the SMs busy. They trust each other and want throughput.
# * **c.** Ten students' Jupyter notebooks on one T4. They sit mostly idle, and each wants "a GPU".
# * **d.** A 70B model serving 200 concurrent chats on 8 H100s.

# %% exercise
choice = {"a": "?", "b": "?", "c": "?", "d": "?"}
### BEGIN SOLUTION
choice.update(a="mig",            # hard memory, fault and performance isolation, with guaranteed slices
              b="mps",            # small kernels overlap on the SMs; there is no MIG on an L4
              c="time_slicing",   # idle tenants cost nothing; no MIG on a T4; isolation not needed
              d="none")           # the engine batches 200 chats itself; TP across the 8 GPUs
### END SOLUTION

# %% check
assert choice == {"a": "mig", "b": "mps", "c": "time_slicing", "d": "none"}, choice
print("✅ isolation needs MIG, cooperative small kernels want MPS, idle tenants tolerate time-slicing,")
print("   and a big model shares best inside its engine (continuous batching), not on the GPU")

# %% [markdown]
# ## Health: GPU utilization is a time measure
# A decode loop at small batch launches one kernel after another, each filling only 16 of the
# H100's 132 SMs, with short gaps between launches. The timeline is simulated:

# %%
kernels = [(t, t + 20.0, 16) for t in np.arange(0, 1000, 25.0)]      # 20 us kernels every 25 us, 16 SMs
c = H.util_counters(kernels, n_sms=132, window=1000.0)
print(f"GPU util {c['gpu_util']:.0%}   SM active {c['sm_active']:.1%}")
sample = {"DCGM_FI_DEV_GPU_UTIL": 80, "DCGM_FI_PROF_SM_ACTIVE": 0.097, "DCGM_FI_PROF_SM_OCCUPANCY": 0.05,
          "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": 0.03, "DCGM_FI_PROF_DRAM_ACTIVE": 0.62,
          "DCGM_FI_DEV_FB_USED": 76000, "DCGM_FI_DEV_FB_FREE": 1900, "DCGM_FI_DEV_CLOCKS_EVENT_REASONS": 0x4}
print("illustrative sample (not a measurement) ->")
for finding in H.diagnose(sample):
    print("  *", finding)

# %% [markdown]
# ## Exercise 5.4: compute the two counters yourself
#
# Write `counters(kernels, n_sms, window)` for kernels given as `(start, end, sms_busy)` with
# **integer** microsecond times. `gpu_util` is the share of the window in which at least one
# kernel runs. `sm_active` is the average over the window of `min(total SMs busy, n_sms) / n_sms`.
# Kernels may overlap. (Stepping through the window 1 us at a time is fine.)

# %% exercise
def counters(kernels, n_sms, window):
    ### BEGIN SOLUTION
    busy_time = sm_time = 0.0
    for t in range(int(window)):
        active = sum(s for (t0, t1, s) in kernels if t0 <= t < t1)
        if active:
            busy_time += 1
            sm_time += min(active, n_sms) / n_sms
    return {"gpu_util": busy_time / window, "sm_active": sm_time / window}
    ### END SOLUTION

# %% check
cases = [([(0, 1000, 8)], 132, 1000),                              # tiny kernel, always on
         ([(0, 500, 132), (500, 750, 66)], 132, 1000),               # a full kernel, then a half one
         ([(0, 600, 100), (400, 1000, 100)], 132, 1000),             # overlap, capped at 132
         ([(int(t), int(t) + 20, 16) for t in range(0, 1000, 25)], 132, 1000)]
for k, n_, w in cases:
    mine, ref = counters(k, n_, w), H.util_counters(k, n_, w)
    assert np.isclose(mine["gpu_util"], ref["gpu_util"]) and np.isclose(mine["sm_active"], ref["sm_active"]), (k, mine, ref)
print("✅ the first case reads GPU util 100% with 6% of the SMs busy; alert on SM active, not GPU util")

# %% [markdown]
# ## Triage: throttling and XIDs
# `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` is a bitmask. Power capping under load is normal, and
# hardware slowdowns are not. XIDs are the driver's error codes in the kernel log (`dmesg`) and
# in DCGM. What matters is **who has to act**:

# %%
for mask in (0x4, 0x44, 0x88):
    print(f"clocks event mask {mask:#06x}: {H.decode_throttle(mask)}")
for code in (13, 31, 48, 63, 79, 94, 95):
    x = H.triage_xid(code)
    print(f"XID {code:>3} [{x.owner:>8}] {x.meaning}: {x.action}")

# %% [markdown]
# ## Exercise 5.5: triage a night of events
#
# Here is a fleet's overnight log of XIDs (`(node, xid)`). Return the set of nodes to **drain**
# (any XID whose owner is `"hardware"`, or XID 63, whose pending row remap needs a GPU reset)
# and the set of nodes whose **app owners to notify** (owner `"app"`). Use `H.triage_xid`.

# %% exercise
events = [("gpu-a1", 13), ("gpu-a2", 79), ("gpu-a3", 31), ("gpu-a3", 45), ("gpu-b1", 63),
          ("gpu-b2", 94), ("gpu-b3", 95), ("gpu-b3", 13), ("gpu-c1", 43)]
### BEGIN SOLUTION
drain = {n for n, x in events if H.triage_xid(x).owner == "hardware" or x == 63}
notify = {n for n, x in events if H.triage_xid(x).owner == "app"}
### END SOLUTION

# %% check
assert drain == {"gpu-a2", "gpu-b1", "gpu-b3"}, drain
assert notify == {"gpu-a1", "gpu-a3", "gpu-b3", "gpu-c1"}, notify
print("✅ drain", sorted(drain), "| notify app owners on", sorted(notify))
print("   gpu-b3 is on both lists: with an uncontained ECC error (95) on the node, treat its app error (13)")
print("   as a likely symptom and fix the hardware first")

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "LLM replicas get whole GPUs. The engine's continuous batching is
# our sharing mechanism, because it shares the weight reads across requests. Small models and
# multi-tenant endpoints go on MIG slices when tenants need isolation (A100/H100 class), with the
# layout planned up front, since profiles have fixed placements. Cooperative small workloads
# share through MPS, and idle-heavy dev notebooks through time-slicing. We alert on DCGM SM
# active, tensor active and DRAM active, never on GPU util. Hardware XIDs (48, 79, 95) and
# pending row remaps drain the node automatically, while app XIDs (13, 31, 43) page the service
# owner."
#
# **Drill questions**
#
# 1. *Why not time-slice an H100 across seven customers' endpoints?* There is no memory or fault
#    isolation (one tenant can OOM the others), and latency grows with the number of busy
#    tenants. Use seven `1g.10gb` MIG instances: fixed, isolated, predictable.
# 2. *The dashboard says 100% GPU utilization but throughput is poor. Where do you look?* GPU
#    util only says *some* kernel was running. Check SM active and tensor active. Small-batch
#    decode, launch gaps or tiny grids show high util with low SM active. Batch more, capture
#    CUDA graphs.
# 3. *A node logs XID 79. Whose problem is it?* The hardware path: the GPU fell off the PCIe bus.
#    Drain, reboot, run diagnostics, and RMA if it repeats. It is not an application bug.
