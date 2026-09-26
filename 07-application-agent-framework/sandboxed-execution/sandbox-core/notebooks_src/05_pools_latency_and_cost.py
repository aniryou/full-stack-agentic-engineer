# %% [markdown]
# # 05 · Pools, latency and cost per action
#
# **Tier:** T0 — laptop / Colab CPU / CI, free, seconds. Real startup-latency measurement by isolation level
# is the lab (`../sandbox-lab`, `bench.py`); the queueing model and cost arithmetic are here.
#
# ## The one-minute version
# A sandbox per execution has a **cold start** (fork/exec is milliseconds; a container is 100–500 ms; a
# microVM boots in ~125 ms; a full VM is tens of seconds; a GKE pod on a busy cluster can be 40–50 s). So
# the number of sandboxes you keep ready follows queueing arithmetic. **Little's law** sizes the busy set
# (arrival rate × execution time); a replace-after-use warm pool also holds a warming set (rate × cold
# start). **Erlang C** turns a target "fraction that waits" into a server count. The arrival rate ties to
# the scaling primer: 27.1 tool calls/s at peak, and if a fifth are `run_code`, λ ≈ 5.4/s. **Cost per
# action** closes the loop: sandbox-seconds held × $/s, then actions/turn × cost/action. After this you can
# size a pool and put a number on what a code tool costs. Primer: `../PRIMER.md` §6 (latency, throughput and
# cost). Reuses the scaling primer's rates (§3.2) and cost-per-action framing (§1); latencies are inputs and
# any $ figure is `(verify)` against `COMPUTE.md`.

# %%
from sandboxcore import pool

# %% [markdown]
# ## Worked example 1 — cold start by isolation level (inputs, labelled)
# These are the levels you choose between. The milliseconds are **(verify)** except the two measured on this
# build host; a microVM's boot is from Firecracker's own spec. The table is the menu §6 of the primer walks.

# %%
levels = [
    ("fork/exec (process)", 0.0014, "measured on this host"),
    ("Python interpreter/exec", 0.013, "measured on this host"),
    ("container (runc)", 0.3, "(verify) 100-500 ms"),
    ("gVisor (runsc)", 0.4, "(verify) runc + tens-hundreds ms"),
    ("Firecracker microVM", 0.125, "Firecracker SPECIFICATION.md"),
    ("full VM (GCE)", 20.0, "(verify) tens of seconds"),
    ("GKE pod, busy cluster", 45.0, "agent-sandbox docs, p50"),
]
print(f"{'level':26} {'cold start (s)':>14}   provenance")
for name, cold, src in levels:
    print(f"{name:26} {cold:14.4f}   {src}")

# %% [markdown]
# ## Worked example 2 — Little's law and the warm pool (FACTS §12)
# λ = 5 executions/s, each running 2 s, cold start 3 s. Busy = 10, warming = 15, so a replace-after-use pool
# holds **25** to never make a request wait on a cold start.

# %%
lam, t_exec, t_cold = 5, 2, 3
print(f"busy = λ·t_exec = {pool.busy_sandboxes(lam, t_exec):.0f}")
print(f"warming = λ·t_cold = {pool.warming_sandboxes(lam, t_cold):.0f}")
print(f"pool size = {pool.pool_size(lam, t_exec, t_cold)}")

# %% [markdown]
# ## Worked example 3 — Erlang C: how many slots for a wait target
# Offered load a = λ·t_exec = 10 Erlangs. The table is P(wait) and mean queueing wait by server count — the
# curve from "usually waits" to "rarely waits".

# %%
print(f"{'servers':>7} {'P(wait)':>9} {'E[Wq] s':>9}")
for c in (11, 12, 13, 14, 16):
    print(f"{c:7d} {pool.erlang_c(10, c):9.3f} {pool.expected_wait_s(lam, t_exec, c):9.3f}")
print("smallest c with P(wait) <= 0.2:", pool.servers_for_wait_target(lam, t_exec, 0.2))

# %% [markdown]
# ## Worked example 4 — the simulator agrees (SIMULATED)
# A discrete-event run of a fixed pool checks the closed form. A warm pool (c=12) sits near P(wait)=0.449;
# tearing a sandbox down after each use so every request pays the cold start pushes the same load into
# instability — the reason warm pools exist.

# %%
warm = pool.simulate(lam, t_exec, t_cold, 12, n=30000, warm_pool=True, seed=1)
print("warm pool c=12:", warm.as_row(), "(SIMULATED)")
cold = pool.simulate(lam, t_exec, t_cold, 30, n=30000, warm_pool=False, seed=1)
print("replace-after-use c=30:", cold.as_row(), "(SIMULATED; needs far more slots for the same load)")

# %% [markdown]
# ## Worked example 5 — cost per action, tied to the scaling primer
# Cost per execution = sandbox-seconds held × $/s (+ fixed). A warm pool amortises the cold start; pay-per-use
# pays it every time. Then actions/turn × cost/action is the per-turn code-execution bill (scaling primer §1).

# %%
node_per_s = 1e-4     # (verify) a small CPU node's per-second share; COMPUTE.md has no CPU-only price yet
warm_cost = pool.cost_per_execution(t_exec, t_cold, node_per_s, warm_share=0.0)
payg_cost = pool.cost_per_execution(t_exec, t_cold, node_per_s, warm_share=1.0)
print(f"cost/execution: warm ${warm_cost:.6f}  pay-per-use ${payg_cost:.6f}  (verify the node price)")
print(f"at 1.3 run_code calls/turn (scaling primer): warm ${pool.actions_cost(1.3, warm_cost):.6f}/turn")

# %% [markdown]
# ## Exercise 5.1 — size the pool
# Implement `size(lam, t_exec, t_cold)` returning the replace-after-use pool size (busy + warming, rounded
# up). Then compute the pool for λ = 5.4/s (the scaling primer's run_code rate), t_exec = 1.5 s, t_cold = 3 s.

# %%
import math

# %% exercise
def size(lam, t_exec, t_cold):
    ### BEGIN SOLUTION
    return math.ceil(lam * t_exec + lam * t_cold)
    ### END SOLUTION

# %% check
assert size(5, 2, 3) == 25
peak = size(5.4, 1.5, 3)
assert peak == pool.pool_size(5.4, 1.5, 3)
print(f"✅ pool for λ=5.4/s, t_exec=1.5s, t_cold=3s -> {peak} sandboxes")

# %% [markdown]
# ## Exercise 5.2 — Erlang C from scratch
# Implement `p_wait(a, c)` for offered load `a` and `c` servers (the Erlang C formula). Return 1.0 when
# `c <= a` (the queue never clears). Check it against the FACTS table (a=10).

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
assert abs(p_wait(10, 12) - 0.449) < 0.001
assert abs(p_wait(10, 16) - 0.057) < 0.001
assert p_wait(10, 10) == 1.0
assert abs(p_wait(10, 13) - pool.erlang_c(10, 13)) < 1e-9
print("✅ Erlang C matches: c=12 -> 0.449, c=16 -> 0.057")

# %% [markdown]
# ## Exercise 5.3 — choose an isolation level for a latency budget
# A tool must return within a **2-second** p95 and start from cold each time (no warm pool). From the levels
# table, set `viable` to the set of level names whose cold start alone fits inside 2 s (leaving ~1 s for the
# code). The point: on a busy GKE cluster, pod-per-execution cold starts blow the budget — you need a warm
# pool or a lighter boundary.

# %% exercise
### BEGIN SOLUTION
viable = {name for name, cold, _ in levels if cold <= 1.0}
### END SOLUTION

# %% check
assert "GKE pod, busy cluster" not in viable and "full VM (GCE)" not in viable
assert "Firecracker microVM" in viable and "container (runc)" in viable
print("✅ viable cold-start levels under a 2 s budget:", sorted(viable))
print("   a 45 s GKE cold start needs a warm pool or exec-into-a-running-pod, not pod-per-call")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A sandbox per execution has a cold start, so I size the fleet with queueing
# arithmetic. Little's law gives the busy count — arrival rate times execution time — and a replace-after-use
# pool also holds a warming set of rate times cold start, so at 5 executions a second, 2-second runs and a
# 3-second cold start I keep about 25 sandboxes ready. Erlang C turns a wait target into a slot count. The
# arrival rate comes from the workload: the scaling primer's 27 tool calls a second at peak, a fifth of them
# code, is about 5 a second. Cost per action is sandbox-seconds held times the node's per-second price, so a
# warm pool that amortises the cold start is cheaper per action as well as faster. The lever that matters
# most is the isolation level's cold start: a process is milliseconds, a microVM ~125 ms, but a fresh GKE pod
# on a busy cluster is 40-plus seconds — which is why production uses warm pools and exec-into-a-running-pod,
# not a pod per call."
#
# **Drill questions**
# 1. *How many sandboxes to never wait on a cold start at λ=5/s, 2 s runs, 3 s cold?* — busy 10 + warming 15
#    = 25 (Little's law on both the running and warming sets).
# 2. *Where does the arrival rate come from?* — The workload: e.g. the scaling primer's 27.1 tool calls/s at
#    peak; the run_code fraction (say 20%) gives ~5.4/s.
# 3. *Your p95 budget is 2 s and pods cold-start in 45 s. What do you change?* — Do not start a pod per call:
#    keep a warm pool and exec into a ready sandbox (sub-second), or use a lighter boundary (microVM/gVisor)
#    with snapshot/restore.
