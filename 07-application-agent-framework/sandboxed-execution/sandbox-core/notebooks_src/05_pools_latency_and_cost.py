# %% [markdown]
# # 05 · Pools, latency and cost per action
#
# **Tier:** T0: a laptop, a Colab CPU or CI, free, less than a minute. The lab measures the real startup latency of
# containers, kind and GKE (`../sandbox-lab`, `bench.py`). The queueing model and the cost arithmetic are here, and
# this notebook measures the two process-level latencies live on your machine.
#
# ## The one-minute version
# A sandbox per execution has a **cold start**:
#
# - fork/exec takes milliseconds,
# - a container takes 100–500 ms,
# - a microVM boots in ~125 ms,
# - a full VM takes tens of seconds,
# - a GKE pod on a busy cluster can take 40–50 s.
#
# Thus you calculate the number of sandboxes that you keep ready with queueing arithmetic.
#
# In a **replace-after-use** warm pool, each sandbox runs one execution, and then the pool destroys it. A replacement
# starts in the background and becomes warm. Thus every execution holds a slot for its run *and* for the cold start of
# its replacement. **Little's law** gives the *mean* occupancy λ·(t_exec + t_cold). That is the offered load: a floor,
# not a size, because with exactly that many slots the queue never clears. **Erlang C** on that load turns a target
# "fraction that waits for a warm sandbox" into a slot count.
#
# The arrival rate ties to the scaling primer: 27.1 tool calls/s at peak. If a fifth of them are `run_code`,
# λ ≈ 5.4/s.
#
# **Cost per action** closes the loop. A warm pool moves the cold start off the latency path, but not off the bill.
# The cost is sandbox-seconds per execution × $/s, then actions/turn × cost/action.
#
# Primer: `../PRIMER.md` §6 (latency, throughput and cost). This notebook uses the rates (§3.2) and the
# cost-per-action view (§1) of the scaling primer again. All latencies other than the two that this notebook measures
# are inputs, and each $ figure is `(verify)`.

# %%
from sandboxcore import pool

# %% [markdown]
# ## Worked example 1 — cold start by isolation level (two measured here, the rest labelled inputs)
# This notebook measures the first two rows **on this machine now** (median of 15 runs). The other rows come from
# upstream specs, or they are `(verify)` inputs. The table is the menu that §6 of the primer walks through.

# %%
import statistics
import subprocess
import time

from sandboxcore import ExecutionRequest, ProcessSandbox


def median_s(fn, n=15):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


fork_exec = median_s(lambda: subprocess.run(["true"]))
sb = ProcessSandbox()
py_sandbox = median_s(lambda: sb.run(ExecutionRequest(code="pass")))
levels = [
    ("fork/exec (true)", fork_exec, "measured on this machine now"),
    ("ProcessSandbox round trip", py_sandbox, "measured on this machine now (Python start + limits + sweep)"),
    ("container (runc)", 0.3, "(verify) 100-500 ms"),
    ("gVisor (runsc)", 0.4, "(verify) runc + tens-hundreds ms"),
    ("Firecracker microVM", 0.125, "Firecracker SPECIFICATION.md (<=125 ms to init)"),
    ("full VM (GCE)", 20.0, "(verify) tens of seconds"),
    ("GKE pod, busy cluster", 45.0, "agent-sandbox performance-tuning.md, self-run cluster, p50"),
]
print(f"{'level':27} {'cold start (s)':>14}   provenance")
for name, cold, src in levels:
    print(f"{name:27} {cold:14.4f}   {src}")

# %% [markdown]
# ## Worked example 2 — Little's law gives the mean, not the pool (FACTS §12)
# Let $\lambda = 5$ executions/s, with a run of 2 s each and a cold start of 3 s. On average, 10 sandboxes are busy
# and 15 are replacements in warm-up. That gives **25 slots occupied on average**. That is the offered load $a$ in
# Erlangs. But a pool of exactly 25 slots has nothing spare for the moments when many arrivals come at the same time.

# %%
lam, t_exec, t_cold = 5, 2, 3
a = pool.mean_occupancy(lam, t_exec, t_cold)
print(f"busy = λ·t_exec = {pool.busy_sandboxes(lam, t_exec):.0f}, warming = λ·t_cold = "
      f"{pool.warming_sandboxes(lam, t_cold):.0f}, mean occupancy a = {a:.0f} Erlangs")
print(f"P(wait) with c = 25 slots: {pool.erlang_c(a, 25):.3f}  (the queue never clears)")

# %% [markdown]
# ## Worked example 3 — Erlang C: how many slots for a wait target
# Replace-after-use holds a slot for $t_{\text{exec}} + t_{\text{cold}} = 5$ s, so $a = 25$. A **reuse** pool holds
# it for $t_{\text{exec}}$ only, $a = 10$. In a reuse pool, one warm sandbox serves execution after execution. That is
# faster and lower in cost, but state carries over. The tables give $P(\text{wait})$ and the mean wait in the queue by
# slot count.

# %%
print("replace-after-use (a = 25):")
print(f"{'slots':>7} {'P(wait)':>9} {'E[Wq] s':>9}")
for c in (26, 28, 30, 31, 33, 35):
    print(f"{c:7d} {pool.erlang_c(25, c):9.3f} {pool.expected_wait_s(lam, t_exec + t_cold, c):9.3f}")
print("slots for P(wait) <= 0.2:", pool.slots_for_wait_target(lam, t_exec, t_cold, 0.2))
print("reuse pool (a = 10): slots for P(wait) <= 0.2:",
      pool.slots_for_wait_target(lam, t_exec, t_cold, 0.2, replace_after_use=False))

# %% [markdown]
# ## Worked example 4 — the simulator checks it (SIMULATED)
# This is a discrete-event run of a pool of constant size, in three modes:
#
# - **replace_after_use** at 25 slots is unstable. At 31, it mostly serves warm. That is slightly better than
#   Erlang C, because a constant cold start varies less than the exponential that the formula assumes.
# - **cold_on_demand** has no pool, and every request starts its own sandbox from cold. It pays the 3 s cold start
#   on every request, whatever the slot count.
# - **reuse** at 12 sits near M/M/c.

# %%
for label, c, mode in (("replace-after-use", 25, "replace_after_use"), ("replace-after-use", 31, "replace_after_use"),
                       ("cold-on-demand", 31, "cold_on_demand"), ("reuse", 12, "reuse")):
    r = pool.simulate(lam, t_exec, t_cold, c, n=30000, mode=mode, seed=1)
    print(f"{label:18} c={c:2d}: P(wait)={r.frac_waited:.3f} mean wait={r.mean_wait_s:6.3f}s "
          f"p95={r.p95_wait_s:6.3f}s cold on path={r.cold_on_path}  (SIMULATED)")

# %% [markdown]
# ## Worked example 5 — cost per action, tied to the scaling primer
# Every one-shot sandbox (pod per call, replace-after-use, pay-per-use) spends a cold start that someone pays for:
# 2 + 3 = 5 sandbox-seconds per execution. The fleet view adds the idle headroom that the wait target buys:
# 31 slots ÷ 5/s = 6.2 sandbox-seconds per execution. Only a reuse pool amortises the cold start (2 s), and it pays
# in state. Then actions/turn × cost/action is the code-execution bill per turn (scaling primer §1).

# %%
node_per_s = 1e-4     # (verify) a small CPU node's per-second share; COMPUTE.md has no CPU-only price yet
one_shot = pool.cost_per_execution(t_exec, t_cold, node_per_s)
reused = pool.cost_per_execution(t_exec, t_cold, node_per_s, reuse=True)
fleet = pool.pool_cost_per_execution(lam, pool.slots_for_wait_target(lam, t_exec, t_cold, 0.2), node_per_s)
print(f"cost/execution: one-shot ${one_shot:.6f}  fleet of 31 ${fleet:.6f}  reuse ${reused:.6f}  (verify the price)")
print(f"at 1.3 run_code calls/turn (scaling primer): ${pool.actions_cost(1.3, fleet):.6f}/turn for the fleet")

# %% [markdown]
# ## Exercise 5.1 — predict the slot count, then check it
# The run_code rate of the scaling primer is $\lambda = 27.1 \times 0.2 = 5.42$/s. Use $t_{\text{exec}} = 2$ s,
# $t_{\text{cold}} = 3$ s and a target of at most 20% of requests that wait for a warm sandbox. First, **predict**
# the replace-after-use slot count (hint: it is not the Little's-law mean, 27.1). Then implement
# `slots(lam, t_exec, t_cold, target)` with `pool.erlang_c`. It returns the smallest $c$ whose
# $P(\text{wait}) \le \text{target}$, when each execution holds a slot for $t_{\text{exec}} + t_{\text{cold}}$.

# %% exercise
import math

predicted_slots = None
### BEGIN SOLUTION
predicted_slots = 34
### END SOLUTION


def slots(lam, t_exec, t_cold, target):
    ### BEGIN SOLUTION
    a = lam * (t_exec + t_cold)
    c = math.floor(a) + 1
    while pool.erlang_c(a, c) > target:
        c += 1
    return c
    ### END SOLUTION

# %% check
lam_peak = 27.1 * 0.2
got = slots(lam_peak, 2, 3, 0.2)
for args in ((5, 2, 3, 0.2), (5, 2, 3, 0.05), (2, 1, 8, 0.1), (lam_peak, 2, 3, 0.2)):
    assert slots(*args) == pool.slots_for_wait_target(*args), args
print(f"you predicted {predicted_slots}; Erlang C says {got} (mean occupancy {pool.mean_occupancy(lam_peak, 2, 3):.1f})")
sim = pool.simulate(lam_peak, 2, 3, got, n=20000, mode="replace_after_use", seed=2)
print(f"simulated at {got} slots: P(wait) = {sim.frac_waited:.3f} (SIMULATED)")
assert predicted_slots == got, "the Little's-law mean is the floor; the target needs headroom above it"
assert sim.frac_waited <= 0.2 + 0.02
print("✅ size a pool with Erlang C on (exec + cold); Little's law only tells you the floor")

# %% [markdown]
# ## Exercise 5.2 — Erlang C from scratch
# Implement `p_wait(a, c)` for offered load `a` and `c` servers (the Erlang C formula). Return 1.0 when
# `c <= a` (the queue never clears). Calculate each term from the term before it, in a loop. Do not calculate
# $a^c / c!$ directly, because it overflows a float long before the loads that a busy fleet sees (try $a = 160$).

# %% exercise
def p_wait(a, c):
    ### BEGIN SOLUTION
    if c <= a:
        return 1.0
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    last = term * (a / c) * (c / (c - a))
    return last / (s + last)
    ### END SOLUTION

# %% check
for a_, c_ in ((10, 12), (10, 16), (25, 31), (25, 33), (160.0, 175), (162.6, 186)):
    assert abs(p_wait(a_, c_) - pool.erlang_c(a_, c_)) < 1e-9, (a_, c_)
assert p_wait(10, 10) == 1.0
print("✅ Erlang C matches, including at a = 162.6 Erlangs where a**c / c! would overflow")

# %% [markdown]
# ## Exercise 5.3 — choose an isolation level for a latency budget
# A tool must return within a **2-second** p95 and start from cold each time (no warm pool). From the levels
# table, set `viable` to the set of level names whose cold start alone fits inside 1 s. That leaves ~1 s for the
# code. The point: on a busy GKE cluster, pod-per-execution cold starts go over the budget. You need a warm pool or
# a lighter boundary.

# %% exercise
### BEGIN SOLUTION
viable = {name for name, cold, _ in levels if cold <= 1.0}
### END SOLUTION

# %% check
assert "GKE pod, busy cluster" not in viable and "full VM (GCE)" not in viable
assert "Firecracker microVM" in viable and "container (runc)" in viable
assert viable == {name for name, cold, _ in levels if cold <= 1.0}
print("✅ viable cold-start levels under a 2 s budget:", sorted(viable))
print("   a 45 s GKE cold start needs a warm pool or exec-into-a-running-pod, not pod-per-call")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A sandbox per execution has a cold start, so I calculate the size of the fleet with
# queueing arithmetic. In a replace-after-use pool, each execution holds a slot for its run and for the warm-up of its
# replacement. Take 5 executions a second, 2-second runs and a 3-second cold start. Little's law says that these keep
# 25 slots busy or in warm-up on average. That is the floor, not the size.
#
# "Erlang C on that load of 25 slots says that 31 slots keep the fraction that waits under 20%, and a simulation
# agrees. The arrival rate comes from the workload. The scaling primer has 27 tool calls a second at peak. A fifth of
# them are code. That code rate is about 5.4 a second, and it needs 34 slots.
#
# "Cost per action is sandbox-seconds times the node price. A warm pool hides the cold start from latency, but it
# still pays for it. That is five sandbox-seconds per execution plus the idle headroom, slightly more than six with
# the fleet. Only the reuse of sandboxes amortises it, and that trades away isolation between executions.
#
# "The lever that matters most is the cold start of the isolation level. A process takes milliseconds, and a microVM
# ~125 ms. But a fresh GKE pod on a busy cluster takes 40-plus seconds. That is why interactive tools use warm pools,
# not a pod per call."
#
# **Drill questions**
# 1. *$\lambda = 5$/s, 2 s runs, 3 s cold start: how many replace-after-use slots?* The mean occupancy is
#    $\lambda(t_{\text{exec}} + t_{\text{cold}}) = 25$, and that is unstable as a size. Erlang C on $a = 25$ gives 31
#    for $P(\text{wait}) \le 0.2$.
# 2. *Where does the arrival rate come from?* From the workload. For example, the scaling primer has 27.1 tool
#    calls/s at peak. The run_code fraction (say 20%) gives ~5.4/s.
# 3. *Does a warm pool decrease the cost of each execution?* Not a replace-after-use pool. Every execution still
#    uses the cold start of a sandbox (5 sandbox-seconds here) plus idle headroom. Only reuse amortises it, at the
#    cost of state between executions.
# 4. *Your p95 budget is 2 s and pods have a cold start of 45 s. What do you change?* Do not start a pod per call.
#    Keep a warm pool with a size from Erlang C, and give out ready sandboxes (sub-second). Or use a lighter
#    boundary: gVisor, or a microVM restored from a snapshot. Never restore one snapshot into two tenants.
