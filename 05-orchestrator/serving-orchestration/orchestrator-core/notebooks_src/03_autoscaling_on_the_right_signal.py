# %% [markdown]
# # 03 · Autoscaling on the right signal
#
# **Tier:** T0. CPU only, about 25 seconds, no network. `fleetsim` **simulates** every number in this notebook. The
# HPA logic is the logic of the Kubernetes controller (`fleetsim/autoscale.py` mirrors
# `pkg/controller/podautoscaler`).
#
# ## The one-minute version
# The Horizontal Pod Autoscaler is a proportional controller. It does these four things:
#
# - Every 15 s, it sets $\text{desired} = \lceil \text{current} \times \text{metric} / \text{target} \rceil$ (more
#   exactly: the sum of the samples that the pods reported, divided by the target).
# - It ignores changes in a 10 % tolerance band.
# - It remembers its recommendations for 5 minutes before it scales down.
# - It rate-limits scale-up (+100 % or +4 pods per 15 s).
#
# The rule of the HPA works only if the metric **grows in proportion to load per replica**, **rises before latency
# does**, and **has no cap**. For an LLM engine:
#
# * **GPU utilisation** fails all three. Continuous batching keeps a kernel in operation whenever *any* request is
#   in flight. Thus "utilisation" reads 100 % at a fraction of capacity. The HPA either holds the fleet at max or
#   never moves.
# * **Queue depth alone** is zero until the cliff, and then it jumps to a large value. When capacity catches up, the
#   queue depth reads zero again. Then the HPA scales back down into the next cliff. The result is a sawtooth.
# * **KV-cache usage** and **running requests** are proportional, but they have a cap (100 %, `max_num_seqs`). The
#   controller multiplies the average of the pods that report by their number. Thus a capped metric grows the fleet
#   at most $\text{cap}/\text{target}$-fold per round. The fleet climbs out of a large step one cold start at a time.
# * **In-flight requests** (running + waiting, and also the requests that the gateway holds) are proportional and
#   have no cap. The queue-based KEDA path of llm-d scales on this signal. But this signal counts *requests*. A
#   target from a chat load test is incorrect for RAG, because a RAG request carries five times the prefill. The
#   token-aware path of llm-d counts *work*: seconds of prefill backlog (in-flight uncached tokens ÷ prefill rate),
#   together with KV occupancy for decode.
#
# And for any signal, the **cold start** (node, image, weights, warm-up) decides how much traffic waits in a queue
# while the capacity arrives. Scale-to-zero puts all of the cold start, node included, on every burst. Primer §4.

# %%
from fleetsim import (HPA, L4_8B, Autoscaler, ColdStart, Fleet, PowerOfTwo, burst, chat, mix, pods_metric_replicas,
                      rag, sparkline, table)
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
    rows.append({"req/s": rate, "gpu_util": avg("gpu_util"), "running": avg("running"), "waiting": avg("waiting"),
                 "inflight": avg("running") + avg("waiting"), "kv": avg("kv"), "ttft_p95": s["ttft_p95"],
                 "tpot_p50": s["tpot_p50"]})
sweep = rows
print(table(rows, title="simulated: one L4 replica, chat 1,300 in / 150 out"))

# %% [markdown]
# "GPU utilisation" in this table is the value that `nvidia-smi` reports: the fraction of time in which *a* kernel
# runs. It stays at 1.0 from the second row on. But p95 TTFT stays inside a 1 s SLO up to seven times that load.
# Running requests and KV usage grow with load. The waiting queue stays near zero until the engine runs out of room.
# Then the queue jumps, and that jump *is* the latency cliff.
#
# ## Exercise 3.1 — the HPA's core rule
# Write `hpa_desired(current, metric, target, tolerance=0.1)`. If $\text{metric} / \text{target}$ is in
# $[1 - \text{tolerance}, 1 + \text{tolerance}]$, return `current`. Else, return
# $\lceil \text{current} \times \text{metric} / \text{target} \rceil$. (`metric` is the average over ready pods.)

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
# Before the controller scales **down**, it takes the **maximum** of all the recommendations that it made in the
# last `window` seconds (default 300). This max moves with time. It prevents the removal of pods that the controller
# will need again a minute later. Write `stabilized_down(history, now, window, desired)`. `history` is a list of
# `(time, recommendation)`. Include only the samples that are strictly newer than $\text{now} - \text{window}$, and
# also the new `desired`.

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
# The default scale-up behaviour permits **the larger of +100 % and +4 pods** per 15 s period (`selectPolicy: Max`),
# with no stabilization window. Start from 3 replicas with a desired count of 40 (and max 50). Predict the replica
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
# ## Exercise 3.4 — pods that have not reported yet
# Two ready replicas each report 10 queued requests against a target of 2 per pod. Thus the HPA asked for
# $\lceil 20 / 2 \rceil = 10$ replicas. Eight of them are still Pending, because they pull the image.
#
# At the next sync, the controller calculates the average of the two samples **and eight zeros**. On a scale-up,
# Pending pods count as 0. If that new ratio is inside the 10 % band, or no longer above 1, the controller keeps the
# current count. Else, it asks for $\lceil \text{new ratio} \times \text{pods counted} \rceil$.
#
# **Predict** the recommendation (current = 10) when the two ready pods report **(a)** 10 each, **(b)** 10.5 each
# and **(c)** 30 each. Also predict **(d)**: what does $\lceil \text{sum of the samples} / \text{target} \rceil$
# give for (b), if you ignore the Pending pods fully? Then write `with_pending(values, target, current, pending)`
# for this scale-up path.

# %% exercise
predicted = {"a": None, "b": None, "c": None, "d": None}
### BEGIN SOLUTION
predicted = {"a": 10,     # (20 + 0 x 8) / 10 / 2 = 1.0: inside the band, hold
             "b": 10,     # 21 / 10 / 2 = 1.05: still inside the band, hold
             "c": 30,     # 60 / 10 / 2 = 3.0 -> ceil(3.0 x 10) = 30
             "d": 11}     # ceil(21 / 2): the zeros only changed the answer through the tolerance band
### END SOLUTION


def with_pending(values, target, current, pending):
    ### BEGIN SOLUTION
    counted = list(values) + [0.0] * pending
    new_ratio = sum(counted) / len(counted) / target
    if 0.9 <= new_ratio <= 1.1 or new_ratio < 1.0:
        return current
    return max(current, math.ceil(new_ratio * len(counted)))
    ### END SOLUTION

# %% check
truth = {k: pods_metric_replicas(v, 2, 10, unready=8) for k, v in (("a", [10, 10]), ("b", [10.5, 10.5]),
                                                                   ("c", [30, 30]))}
assert predicted == {**truth, "d": math.ceil(21 / 2)}, (predicted, truth)
for vals, pending in (([10, 10], 8), ([10.5, 10.5], 8), ([30, 30], 8), ([2.3, 2.3], 2), ([4, 6], 1), ([12, 9], 3),
                      ([3, 3, 3], 3), ([5, 7, 9], 4)):
    cur = len(vals) + pending
    assert with_pending(vals, 2, cur, pending) == pods_metric_replicas(vals, 2, cur, unready=pending), (vals, pending)
print("✅ the zeros never raise the count — desired is ceil(sum / target) either way — they only make a small or "
      "direction-flipping change fall inside the band. What bounds a scale-up is the pods that reported.")

# %% [markdown]
# ## Worked example — take the target from the load test
# The reference load is the highest load that still meets the SLO (TTFT $\le$ 1 s in the earlier sweep). A target is
# the per-replica value of the signal at that load, minus a margin. The margin gives time for the cold start. Here
# the margin is one third.

# %%
def pick_target(sweep, signal, slo_ttft_s, margin):
    return max(r[signal] for r in sweep if r["ttft_p95"] <= slo_ttft_s) * (1 - margin)


inflight_target = round(pick_target(sweep, "inflight", 1.0, 1 / 3))
kv_target = round(pick_target(sweep, "kv", 1.0, 1 / 3), 2)
print(f"in-flight target {inflight_target} per replica, KV target {kv_target} "
      f"(waiting would give {pick_target(sweep, 'waiting', 1.0, 1 / 3):.2f}: a queue is ~0 until the cliff)")

# %% [markdown]
# ## Worked example — the same traffic step, five signals
# One replica serves 1.5 req/s. At $t = 60$ s, the traffic jumps to 7.5 req/s for nine minutes. Each run is an HPA
# (min 1, max 8, default behaviour) on a different signal. New replicas take 30 s to become ready (a warm node and
# cached weights). The sparkline is the replica count every 15 s.

# %%
traffic = burst(1.5, 7.5, 60, 600)


def step_traffic():
    return chat(traffic, 800, seed=7, system=1000, user=300, output=150)


signals = {
    "gpu_util, target 0.7": lambda: Autoscaler(HPA(1, 8), "gpu_util", 0.7),
    "gpu_util, target 0.95": lambda: Autoscaler(HPA(1, 8), "gpu_util", 0.95),
    "waiting per pod, target 2": lambda: Autoscaler(HPA(1, 8), "waiting", 2),
    f"kv usage per pod, target {kv_target}": lambda: Autoscaler(HPA(1, 8), "kv", kv_target),
    f"in-flight total, {inflight_target} per pod": lambda: Autoscaler(HPA(1, 8), "inflight", inflight_target,
                                                                      kind="external"),
}
five_rows, lines = [], []
for name, make in signals.items():
    res = Fleet(p, 1, PowerOfTwo(seed=1), autoscaler=make(), cold_start_s=30).run(step_traffic(), horizon=900)
    s = res.summary(ttft_slo=1.0, tpot_slo=0.15)
    reps = [r["replicas"] for r in res.timeline]
    five_rows.append({"signal": name, "ttft_p95": s["ttft_p95"], "slo_attainment": s["slo_attainment"],
                      "gpu_hours": s["gpu_hours"], "max_replicas": max(reps)})
    lines.append(f"{name:30s} {sparkline(reps, 0, 8)}")
print(table(five_rows, title="simulated: 1.5 -> 7.5 req/s step, 30 s cold start"))
print("\n".join(lines))

# %% [markdown]
# * **GPU utilisation** at a 0.7 target scales to the maximum and never comes down. The utilisation stays above the
#   target even at 8 replicas. At 0.95, it never scales at all. Neither is a policy. Both are accidents.
# * **Waiting per pod** scales up strongly, and then it sees an empty queue. It scales down after the 5-minute
#   window and goes into the cliff again. This is the sawtooth in the middle of the peak.
# * **KV usage** is proportional, but it has a cap of 1.0. Each round multiplies the *ready* pods by at most
#   $1/\text{target}$. The pods that the HPA asked for count only when they are ready and report values. Thus the
#   fleet climbs one cold start at a time.
# * **In-flight total** (running + waiting + held at the gateway, averaged per replica) moves directly with demand.
#   Of the signals that also scale back down, it has the best SLO attainment, at half the GPU-hours of the
#   utilisation policy. The rest of its tail comes from the 30 s cold start, which the next table isolates.
#
# ## Worked example — the cold start decides the tail
# This example uses the same traffic and the in-flight signal, with a longer cold start. The last row keeps three
# replicas warm instead. While the new capacity starts, the excess arrival rate builds a backlog:
#
# $$
# \text{backlog} = \max(0, \text{peak} - \text{capacity now}) \times \text{cold start}.
# $$

# %%
cold_rows = []
for label, cold, lo in (("0 s (already warm)", 0, 1), ("30 s", 30, 1), ("120 s (new node + weights)", 120, 1),
                        ("120 s, minReplicas 3", 120, 3)):
    auto = Autoscaler(HPA(lo, 8), "inflight", inflight_target, kind="external")
    s = Fleet(p, lo, PowerOfTwo(seed=1), autoscaler=auto, cold_start_s=cold).run(step_traffic(), horizon=900).summary(
        ttft_slo=1.0, tpot_slo=0.15)
    cold_rows.append({"cold start": label, "ttft_p95": s["ttft_p95"], "slo_attainment": s["slo_attainment"],
                      "gpu_hours": s["gpu_hours"]})
print(table(cold_rows, title="simulated: in-flight signal, 1.5 -> 7.5 req/s"))


def cold_start_backlog(peak_rps, capacity_rps, cold_start_s):
    return max(0.0, peak_rps - capacity_rps) * cold_start_s


start = ColdStart.estimate(weight_gb=16, load_gb_s=0.5, image_gb=10, pull_gb_s=0.25, node_s=0, init_s=30)
cap = max(r["req/s"] for r in sweep if r["ttft_p95"] <= 1.0)          # one replica's capacity at the SLO
print(f"one replica holds the SLO up to {cap} req/s; a {start.total:.0f} s cold start on a free node "
      f"(pull {start.pull_s:.0f} + weights {start.load_s:.0f} + warm-up {start.init_s:.0f}; assumptions to replace "
      f"with measurements) queues {cold_start_backlog(7.5, cap, start.total):.0f} requests at 7.5 req/s")

# %% [markdown]
# Warm headroom (`minReplicas: 3`) beat the reactive fleet on latency *and* on GPU-hours. The reactive fleet had to
# over-scale to drain the backlog that its cold start built. Each term has a different solution: image streaming,
# faster weight reads, cached compilation, warm headroom.
#
# ## Worked example — a request count does not transfer across traffic
# The in-flight target came from a chat load test. Now the same step arrives as a mix: half the rate as chat, plus
# RAG requests with ~6,000-token prompts. The RAG requests are about 15 % of the requests, but they carry three
# quarters of the prefill that the prefix cache does not cover.

# %%
def mixed_step():
    return mix(chat(burst(0.8, 4.0, 60, 600), 800, seed=7, system=1000, user=300, output=150),
               rag(burst(0.15, 0.75, 60, 600), 800, seed=8, docs=4, doc=1400, corpus=5000, zipf=0.5, output=200))


def scorecard(autoscaler, make_traffic, ttft_slo):
    res = Fleet(p, 1, PowerOfTwo(seed=1), autoscaler=autoscaler, cold_start_s=30).run(make_traffic(), horizon=900)
    s = res.summary(ttft_slo=ttft_slo, tpot_slo=0.15)
    return {"ttft_p95": s["ttft_p95"], "slo_attainment": s["slo_attainment"], "gpu_hours": s["gpu_hours"],
            "replicas": sparkline([r["replicas"] for r in res.timeline][::2], 0, 8)}



in_flight = lambda: Autoscaler(HPA(1, 8), "inflight", inflight_target, kind="external")     # noqa: E731
mixed_rows = [{"traffic": "chat step (SLO 1 s)", **scorecard(in_flight(), step_traffic, 1.0)},
              {"traffic": "chat + RAG step (SLO 2 s)", **scorecard(in_flight(), mixed_step, 2.0)}]
print(table(mixed_rows, title=f"simulated: in-flight target {inflight_target} per pod on two traffic mixes"))

# %% [markdown]
# On the mix, the same target scales too late and by too small a quantity. A RAG request counts as one request, but
# it brings five times the prefill. Thus the fleet looks lightly loaded while the prefill queue grows. A new target
# from the new mix repairs this, until the mix changes again. **Request counts are a safe signal only when requests
# are alike.**
#
# The token-aware path of llm-d measures work in its own units instead. The first signal is the prefill backlog in
# seconds (EPP in-flight uncached tokens ÷ `peakPrefillThroughput`), compared with a share of the TTFT SLO. The
# second signal is KV occupancy for the decode side. The HPA takes the larger of the two recommendations. In
# `fleetsim`, the prefill backlog is the metric `"backlog_s"`, and a second metric goes through
# `Autoscaler(..., also=[(metric, target, kind)])`.
#
# ## Exercise 3.5 — one autoscaler for both mixes
# Write `make_autoscaler()`. It returns a new `Autoscaler` (min 1, max 8, any metric or metrics from
# `Autoscaler.METRICS`). It must meet both budgets, with no new adjustment between the two mixes:
#
# - On the chat step: SLO attainment **>= 0.94 within 1.3 GPU-hours**.
# - On the chat + RAG step: attainment **>= 0.90**.
#
# Every target must come from the load test, not from trial and error. For a signal that the sweep measured, use its
# per-replica value at the 1 s TTFT SLO, minus a margin of 20-50 % (`pick_target`). For `backlog_s`, use a share
# (10-50 %) of the tightest TTFT SLO. Be ready to say where each target comes from.

# %% exercise
def make_autoscaler():
    ### BEGIN SOLUTION
    # prefill work: a quarter of the tightest TTFT SLO in seconds of backlog; decode: the load-test KV target
    return Autoscaler(HPA(1, 8), "backlog_s", 0.25, "external", also=[("kv", kv_target, "pods")])
    ### END SOLUTION

# %% check
def derived_range(metric):  # what the load test (and the SLO) allow for this metric's target
    if metric == "backlog_s":
        return 0.1 * 1.0, 0.5 * 1.0
    if metric in sweep[0] and metric != "gpu_util":   # gpu_util is pinned at 1.0: no target comes out of it
        return pick_target(sweep, metric, 1.0, 0.5), pick_target(sweep, metric, 1.0, 0.2)
    return None


for metric, target, kind in make_autoscaler().specs:
    allowed = derived_range(metric)
    assert allowed, f"{metric}: the load test gives no target for it — which signal did it show tracking load?"
    assert allowed[0] <= target <= allowed[1], (f"{metric} target {target}: outside {allowed[0]:.2f}-{allowed[1]:.2f}, "
                                                "what the load test gives with a 20-50 % margin (a share of the SLO "
                                                "for backlog_s)")
yours = [{"traffic": "chat step (SLO 1 s)", **scorecard(make_autoscaler(), step_traffic, 1.0)},
         {"traffic": "chat + RAG step (SLO 2 s)", **scorecard(make_autoscaler(), mixed_step, 2.0)}]
print(table(yours, title="simulated: your autoscaler"))
assert yours[0]["slo_attainment"] >= 0.94 and yours[0]["gpu_hours"] <= 1.3, yours[0]
assert yours[1]["slo_attainment"] >= 0.90, yours[1]
print("✅ one configuration, two traffic mixes: scale on work (prefill seconds, KV), not on request counts")

# %% [markdown]
# ## Exercise 3.6 — is scale-to-zero worth it?
# An internal tool gets 6 bursts a day. Each burst is 20 minutes long at 1 req/s, and nothing comes between the
# bursts. With `minReplicas: 1`, you pay for one L4 node all day. This node is a `g2-standard-4` VM, and its price
# includes the GPU. With `minReplicas: 0` (KEDA, or the `HPAScaleToZero` gate of the HPA), the pod goes 5 minutes
# after a burst (the stabilization window). That saves nothing while the node stays.
#
# You save the money only when the cluster autoscaler removes the empty GPU node. It does this after 10 minutes with
# no need for the node (`--scale-down-unneeded-time`, layer 03 §7.1). Thus the next burst waits for a new node and
# also for the pod. That is ~300 s to allocatable GPUs (the assumption of layer 03), plus the 102 s cold start. The
# worked example "the cold start decides the tail" calculates that 102 s.
#
# Write `scale_to_zero(bursts, burst_min, rps, cold_start_s, usd_per_hour, idle_min=15)`. It returns
# `(usd_saved_per_day, requests_delayed_per_day, mean_added_wait_s)`. You pay for the node from the first request of
# a burst until `idle_min` after the burst ends. The delayed requests are the requests that arrive during the cold
# start. Each of them waits for the rest of the cold start at the time of its arrival. Ignore the backlog drain
# after that.

# %% exercise
def scale_to_zero(bursts, burst_min, rps, cold_start_s, usd_per_hour, idle_min=15):
    ### BEGIN SOLUTION
    billed_h = bursts * (burst_min + idle_min) / 60
    return (24 - billed_h) * usd_per_hour, bursts * rps * cold_start_s, cold_start_s / 2
    ### END SOLUTION

# %% check
with_node = ColdStart.estimate(weight_gb=16, load_gb_s=0.5, image_gb=10, pull_gb_s=0.25, node_s=300, init_s=30)
usd, delayed, wait = scale_to_zero(6, 20, 1.0, with_node.total, 0.70)   # $0.70/h: approximate L4 VM, verify
assert with_node.total == 402 and abs(usd - (24 - 6 * 35 / 60) * 0.70) < 1e-9
assert delayed == 6 * 402 and wait == 201
print(f"✅ saves ${usd:.2f}/day per L4 node; {delayed:,.0f} requests/day wait for a {with_node.total:.0f} s cold "
      f"start — {wait:.0f} s on average, up to {with_node.total:.0f} s, plus the backlog drain")

# %% [markdown]
# For an internal tool with batch-like traffic, that is a good trade. For a chat that customers use, it is not a
# good trade. Keep one replica warm (or use a sleep/wake mechanism that keeps the weights resident), and scale *from
# one*. If you scale only the pod to zero and the GPU node stays, you save nothing, unless something else uses that
# GPU. A platform that bills per second and owns the node pool (Cloud Run with GPUs) removes the node term from your
# bill. But it does not remove the model load from the first request.
#
# ## In a design review
# **Two-minute version.** "I scale the model servers with an HPA, through KEDA, so that it can go to zero where that
# makes sense. The HPA scales on the work in flight. If our traffic is uniform, that is running plus waiting
# requests, and also the requests that the router holds. If prompt sizes are not uniform, that is seconds of prefill
# backlog plus KV occupancy, because a request count calibrated on chat under-scales RAG. The target comes from a
# load test at the SLO, minus a margin.
#
# "Not GPU utilisation: continuous batching puts it at its maximum at a fraction of capacity. Not the queue alone:
# it reads zero whenever we have sufficient capacity, and the fleet goes up and down in a sawtooth. Scale-down keeps
# the 5-minute stabilization window. Scale-up stays fast. Then I attack the cold start term by term: node, image,
# weights, compilation. The reason is that on a traffic step, the cold start, not the signal, decides the tail."
#
# **Drills**
# 1. *Why does the HPA scale a fleet to max when its target is 70 % GPU utilisation?* The engine is "busy" whenever
#    any request is in flight. Thus utilisation stays near 100 % at every replica count, and the ratio never goes
#    below 1.1.
# 2. *Our HPA on `num_requests_waiting` oscillates during peaks. Why, and what is the solution?* Waiting is ~0
#    whenever capacity is sufficient. Thus the HPA sees "no load" and scales down into the next cliff. Add running
#    requests (or KV usage) as a second metric. The HPA takes the max over the metrics.
# 3. *2 ready pods at 10 queued each (target 2) and 8 Pending: what does the HPA do?* It holds at 10:
#    $(20 + 0 \times 8) / 10 / 2 = 1.0$ is inside the band. Without the zeros, it also asks for
#    $\lceil 20 / 2 \rceil = 10$. The zeros only stop a small change. At 10.5 each, it still holds, but a
#    calculation that ignores the Pending pods gives 11.
# 4. *We adjusted our in-flight target on chat, and then the RAG traffic doubled. What breaks?* The count gives too
#    small a weight to RAG. Here, SLO attainment decreased from 0.94 to about 0.7. Scale on seconds of prefill backlog
#    plus KV, or calculate the target again for every mix.
# 5. *What does `minReplicas: 0` cost?* The first requests of every burst wait for the full cold start. After
#    the cluster autoscaler removes the idle GPU node, the cold start also includes a new node. Then the full cold
#    start is ~400 s here.
#    `minReplicas: 0` also needs an Object or External metric, because zero pods give no pod metric. This is why KEDA is the usual route.
