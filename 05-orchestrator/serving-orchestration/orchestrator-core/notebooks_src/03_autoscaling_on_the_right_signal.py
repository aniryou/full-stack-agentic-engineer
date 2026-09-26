# %% [markdown]
# # 03 · Autoscaling on the right signal
#
# **Tier:** T0 — CPU only, about 20 seconds, no network. Every number below is **simulated** by `fleetsim`; the HPA
# logic is the Kubernetes controller's (`fleetsim/autoscale.py` mirrors `pkg/controller/podautoscaler`).
#
# ## The one-minute version
# The Horizontal Pod Autoscaler is a proportional controller: every 15 s it sets
# `desired = ceil(current x metric / target)`, ignores changes within a 10 % tolerance band, remembers its
# recommendations for 5 minutes before scaling down, and rate-limits scale-up (+100 % or +4 pods per 15 s). That
# rule only works if the metric **grows in proportion to load per replica**, **rises before latency does**, and
# **is not capped**. For an LLM engine:
#
# * **GPU utilisation** fails all three: continuous batching keeps a kernel running whenever *any* request is in
#   flight, so "utilisation" reads 100 % at a fraction of capacity. The HPA either pins the fleet at max or never moves.
# * **Queue depth alone** is zero until the cliff, then explodes; once capacity catches up it reads zero again and the
#   HPA scales back down into the next cliff — a sawtooth.
# * **KV-cache usage** and **running requests** are proportional but capped (100 %, `max_num_seqs`), and not-ready
#   pods count as 0 on a scale-up, so they climb out of a big step one cold start at a time.
# * **In-flight work** (running + waiting, including requests held at the gateway) is proportional and uncapped. It
#   is what llm-d's KEDA path scales on (EPP flow-control queue + running requests).
#
# And whatever the signal, the **cold start** (node, image, weights, warm-up) decides how much traffic queues while
# capacity arrives. Primer §4.

# %%
from fleetsim import (HPA, L4_8B, Autoscaler, ColdStart, Fleet, PowerOfTwo, burst, chat, pods_metric_replicas,
                      sparkline, table)
import math

p = L4_8B


def chats(rate, duration=300, seed=5):
    return chat(rate, duration, seed=seed, system=1000, user=300, output=150)


rows = []
for rate in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
    res = Fleet(p, 1, PowerOfTwo(seed=1), sample_s=15.0).run(chats(rate))
    s = res.summary(ttft_slo=1.0, tpot_slo=0.15)
    tl = [r for r in res.timeline if 60 <= r["t"] <= 300]
    avg = lambda k: sum(r[k] for r in tl) / len(tl)          # noqa: E731
    rows.append({"req/s": rate, "gpu_util": avg("gpu_util"), "running": avg("running"), "kv": avg("kv"),
                 "waiting": avg("waiting"), "ttft_p95": s["ttft_p95"], "tpot_p50": s["tpot_p50"]})
sweep = rows
print(table(rows, title="simulated: one L4 replica, chat 1,300 in / 150 out"))

# %% [markdown]
# "GPU utilisation" here is what `nvidia-smi` reports: the fraction of time *a* kernel is running. It is pinned at
# 1.0 from the second row on, while p95 TTFT stays inside a 1 s SLO up to seven times that load. Running requests and
# KV usage grow with load; the waiting queue stays near zero until the engine runs out of room — then it jumps, and
# that jump *is* the latency cliff.
#
# ## Exercise 3.1 — the HPA's core rule
# Write `hpa_desired(current, metric, target, tolerance=0.1)`: return `current` if `metric / target` is within
# `[1 - tolerance, 1 + tolerance]`, else `ceil(current * metric / target)`. (`metric` is the average over ready pods.)

# %% exercise
def hpa_desired(current, metric, target, tolerance=0.1):
    ### BEGIN SOLUTION
    ratio = metric / target
    if 1 - tolerance <= ratio <= 1 + tolerance:
        return current
    return math.ceil(current * ratio)
    ### END SOLUTION

# %% check
assert hpa_desired(3, 200, 100) == 6            # the docs' example: 200m against a 100m target doubles
assert hpa_desired(4, 50, 100) == 2             # 50m halves
assert hpa_desired(5, 108, 100) == 5            # inside the 10 % band: no action
assert hpa_desired(5, 125, 100) == 7            # ceil(6.25)
assert hpa_desired(5, 125, 100) == pods_metric_replicas([1.25] * 5, 1.0, 5)   # same as the controller model
print("✅ hpa_desired works")

# %% [markdown]
# ## Exercise 3.2 — the scale-down stabilization window
# Before scaling **down**, the controller takes the **maximum** of every recommendation it made in the last
# `window` seconds (default 300) — a rolling max that stops it removing pods it will need again a minute later.
# Write `stabilized_down(history, now, window, desired)`: `history` is a list of `(time, recommendation)`;
# include only samples strictly newer than `now - window`, plus the new `desired`.

# %% exercise
def stabilized_down(history, now, window, desired):
    ### BEGIN SOLUTION
    return max([d for t, d in history if t > now - window] + [desired])
    ### END SOLUTION

# %% check
hist = [(0, 8), (60, 6), (120, 3), (180, 2)]
assert stabilized_down(hist, 200, 300, 2) == 8          # the 8 from t=0 still holds the fleet up
assert stabilized_down(hist, 330, 300, 2) == 6          # t=0 aged out; the 6 from t=60 is the max now
assert stabilized_down(hist, 450, 300, 1) == 2          # only the t=180 sample is left
assert stabilized_down(hist, 500, 300, 1) == 1          # everything aged out: follow the metric
print("✅ stabilized_down works — scale-down follows the max of the last 5 minutes")

# %% [markdown]
# ## Exercise 3.3 — how fast can it scale up?
# The default scale-up behaviour allows, per 15 s period, **the larger of +100 % and +4 pods** (`selectPolicy: Max`),
# with no stabilization window. Starting from 3 replicas with a desired count of 40 (and max 50), predict the replica
# count after each of the first four 15-second syncs.

# %% exercise
predicted = []      # four integers
### BEGIN SOLUTION
predicted = [7, 14, 28, 40]     # max(3+4, 6) = 7; max(7+4, 14) = 14; 28; then capped by the desired 40
### END SOLUTION

# %% check
hpa, n, seen = HPA(1, 50), 3, []
for k in range(4):
    n = hpa.step(15.0 * k, n, 40)
    seen.append(n)
assert predicted == seen, seen
print("✅", seen, "— doubling per sync, so a 1 -> 40 jump takes about a minute of controller time")

# %% [markdown]
# ## Worked example — the same traffic step, five signals
# One replica serves 1.5 req/s; at t = 60 s traffic jumps to 7.5 req/s for nine minutes. Each run is an HPA
# (min 1, max 8, default behaviour) on a different signal; new replicas take 30 s to become ready (a warm node and
# cached weights). The sparkline is the replica count every 15 s.

# %%
traffic = burst(1.5, 7.5, 60, 600)


def step_traffic():
    return chat(traffic, 800, seed=7, system=1000, user=300, output=150)


signals = {
    "gpu_util, target 0.7": lambda: Autoscaler(HPA(1, 8), "gpu_util", 0.7),
    "gpu_util, target 0.95": lambda: Autoscaler(HPA(1, 8), "gpu_util", 0.95),
    "waiting per pod, target 2": lambda: Autoscaler(HPA(1, 8), "waiting", 2),
    "kv usage per pod, target 0.6": lambda: Autoscaler(HPA(1, 8), "kv", 0.6),
    "in-flight total, 40 per pod": lambda: Autoscaler(HPA(1, 8), "inflight", 40, kind="external"),
}
rows, lines = [], []
for name, make in signals.items():
    res = Fleet(p, 1, PowerOfTwo(seed=1), autoscaler=make(), cold_start_s=30).run(step_traffic(), horizon=900)
    s = res.summary(ttft_slo=1.0, tpot_slo=0.15)
    reps = [r["replicas"] for r in res.timeline]
    rows.append({"signal": name, "ttft_p95": s["ttft_p95"], "slo_attainment": s["slo_attainment"],
                 "gpu_hours": s["gpu_hours"], "max_replicas": max(reps)})
    lines.append(f"{name:30s} {sparkline(reps, 0, 8)}")
print(table(rows, title="simulated: 1.5 -> 7.5 req/s step, 30 s cold start"))
print("\n".join(lines))

# %% [markdown]
# * **GPU utilisation** at a 0.7 target scales to the maximum and never comes down (utilisation stays above target
#   even at 8 replicas); at 0.95 it never scales at all. Neither is a policy; both are accidents.
# * **Waiting per pod** scales up hard, then sees an empty queue, scales down after the 5-minute window — and walks
#   into the cliff again: the sawtooth in the middle of the peak.
# * **KV usage** is proportional but capped at 1.0, so each round can at most multiply the fleet by 1/0.6, and the
#   not-yet-ready pods count as 0 — it climbs one cold start at a time.
# * **In-flight total** (running + waiting + held at the gateway, averaged per replica) tracks demand directly: the best
#   SLO attainment of the signals that also scale back down, at half the GPU-hours of the utilisation policy. What is
#   left of its tail is the 30 s cold start, which the next table isolates.
#
# ## Worked example — the cold start decides the tail
# Same traffic, the in-flight signal, and a longer cold start; the last row keeps three replicas warm instead.

# %%
rows = []
for label, cold, lo in (("0 s (already warm)", 0, 1), ("30 s", 30, 1), ("120 s (new node + weights)", 120, 1),
                        ("120 s, minReplicas 3", 120, 3)):
    auto = Autoscaler(HPA(lo, 8), "inflight", 40, kind="external")
    s = Fleet(p, lo, PowerOfTwo(seed=1), autoscaler=auto, cold_start_s=cold).run(step_traffic(), horizon=900).summary(
        ttft_slo=1.0, tpot_slo=0.15)
    rows.append({"cold start": label, "ttft_p95": s["ttft_p95"], "slo_attainment": s["slo_attainment"],
                 "gpu_hours": s["gpu_hours"]})
print(table(rows, title="simulated: in-flight signal, 1.5 -> 7.5 req/s"))

# %% [markdown]
# ## Exercise 3.4 — what queues during a cold start?
# While new capacity starts, the excess arrival rate piles up: `backlog = max(0, peak - capacity_now) x cold_start`.
# Write `cold_start_backlog(peak_rps, capacity_rps, cold_start_s)`, then use `ColdStart.estimate` to price a
# realistic start for an 8B bf16 model: a free GPU node, a 10 GB image pulled at 0.25 GB/s, 16 GB of weights read at
# 0.5 GB/s and 30 s of engine warm-up (all four numbers are assumptions to replace with your own measurements).

# %% exercise
def cold_start_backlog(peak_rps, capacity_rps, cold_start_s):
    ### BEGIN SOLUTION
    return max(0.0, peak_rps - capacity_rps) * cold_start_s
    ### END SOLUTION


start = None      # a ColdStart for the assumptions above
### BEGIN SOLUTION
start = ColdStart.estimate(weight_gb=16, load_gb_s=0.5, image_gb=10, pull_gb_s=0.25, node_s=0, init_s=30)
### END SOLUTION

# %% check
cap = max(r["req/s"] for r in sweep if r["ttft_p95"] <= 1.0)          # one replica's capacity at the SLO
assert cold_start_backlog(7.5, cap, 120) == (7.5 - cap) * 120 and cold_start_backlog(1.0, 2.0, 60) == 0
assert abs(start.total - (40 + 32 + 30)) < 1e-9
print(f"one replica holds the SLO up to {cap} req/s; a {start.total:.0f} s cold start "
      f"(pull {start.pull_s:.0f} + weights {start.load_s:.0f} + warm-up {start.init_s:.0f}) queues "
      f"{cold_start_backlog(7.5, cap, start.total):.0f} requests at 7.5 req/s")
print("✅ the fixes differ per term: image streaming, faster weight reads, cached compilation, warm headroom")

# %% [markdown]
# ## Exercise 3.5 — pick the target from the load test
# The target is not a guess: it is the per-replica value of the signal at the highest load that still meets the SLO,
# minus a margin that buys time for the cold start. Write `pick_target(sweep, signal, slo_ttft_s, margin)` that finds
# the sweep rows with `ttft_p95 <= slo_ttft_s`, takes the **largest** value of `signal` among them, and multiplies it
# by `1 - margin`.

# %% exercise
def pick_target(sweep, signal, slo_ttft_s, margin):
    ### BEGIN SOLUTION
    return max(r[signal] for r in sweep if r["ttft_p95"] <= slo_ttft_s) * (1 - margin)
    ### END SOLUTION

# %% check
ok = [r for r in sweep if r["ttft_p95"] <= 1.0]
assert pick_target(sweep, "running", 1.0, 0.2) == max(r["running"] for r in ok) * 0.8
assert pick_target(sweep, "kv", 1.0, 0.0) == max(r["kv"] for r in ok)
print(f"✅ running-requests target {pick_target(sweep, 'running', 1.0, 0.2):.0f} per replica, "
      f"KV target {pick_target(sweep, 'kv', 1.0, 0.2):.2f} (20 % margin)")

# %% [markdown]
# ## Exercise 3.6 — is scale-to-zero worth it?
# An internal tool gets 6 bursts a day, each 20 minutes long at 1 req/s, and nothing in between. With
# `minReplicas: 1` one L4 idles all day; with `minReplicas: 0` (an External metric — KEDA, or the HPA's
# `HPAScaleToZero`) the fleet drops to zero 5 minutes (the stabilization window) after each burst, and the requests
# of the first `cold_start_s` of each burst wait for a cold start. Write `scale_to_zero(bursts, burst_min, rps,
# cold_start_s, usd_per_hour)` returning `(usd_saved_per_day, requests_delayed_per_day)`. Idle hours saved per day =
# `24 - bursts x (burst_min + 5) / 60 - bursts x cold_start_s / 3600`.

# %% exercise
def scale_to_zero(bursts, burst_min, rps, cold_start_s, usd_per_hour):
    ### BEGIN SOLUTION
    idle_h = 24 - bursts * (burst_min + 5) / 60 - bursts * cold_start_s / 3600
    return idle_h * usd_per_hour, bursts * rps * cold_start_s
    ### END SOLUTION

# %% check
usd, delayed = scale_to_zero(6, 20, 1.0, 102, 0.70)   # $0.70/h: approximate L4 on-demand, verify for your region
assert abs(usd - (24 - 2.5 - 0.17) * 0.70) < 1e-9 and delayed == 612
print(f"✅ saves ${usd:.2f}/day per idle L4, delays {delayed:.0f} requests/day by ~{102} s each")

# %% [markdown]
# For an internal batch-ish tool that is a good trade; for a customer-facing chat it is not — keep one replica warm
# (or use a sleep/wake mechanism that keeps the weights resident) and scale *from one*.
#
# ## In a design review
# **Two-minute version.** "I'd scale the model servers with an HPA — via KEDA so it can go to zero where that makes
# sense — on in-flight work per replica: running plus waiting, including what the gateway is holding, with the target
# taken from a load test at the SLO minus a margin. Not GPU utilisation: continuous batching pegs it at a fraction of
# capacity. Not the queue alone: it reads zero whenever we have enough capacity and the fleet saws up and down.
# Scale-down keeps the 5-minute stabilization window; scale-up stays fast. Then I'd attack the cold start term by
# term — image streaming, faster weight loading, cached compilation — because on a traffic step the cold start, not
# the signal, decides the tail."
#
# **Drills**
# 1. *Why does the HPA scale a fleet to max when targeting 70 % GPU utilisation?* The engine is "busy" whenever any
#    request is in flight, so utilisation sits near 100 % at every replica count; the ratio never drops below 1.1.
# 2. *Our HPA on `num_requests_waiting` oscillates during peaks. Why, and the fix?* Waiting is ~0 whenever capacity
#    suffices, so the HPA sees "no load" and scales down into the next cliff; add running requests (or KV usage) as a
#    second metric — the HPA takes the max over metrics.
# 3. *What does `minReplicas: 0` cost?* Every burst's first requests wait for a full cold start; it needs an Object or
#    External metric (a pod metric cannot be read from zero pods), which is why KEDA is the usual route.
