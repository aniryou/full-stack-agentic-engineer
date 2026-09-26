# %% [markdown]
# # 03 · The autoscaling recommender, exactly
#
# **Tier:** T0 — CPU only. The HPA math is a line-by-line port of kube-controller-manager v1.34
# (`igwlab/autoscale.py`); the live part scrapes the in-process fake backends (emulated timing);
# the long-horizon part is a *simulated* fluid model of a pool with cold starts.
#
# ## The one-minute version
#
# The HorizontalPodAutoscaler is a proportional controller with guard rails:
# `desired = ceil(ready_pods × current_average / target)`, skipped when the ratio is within ±10%;
# then **stabilization** (scale down no lower than the highest recommendation of the last 300 s),
# then **rate limits** (by default at most +4 pods or +100% per 15 s up, −100% per 15 s down), then
# min/max. For LLM servers the question is *which metric*: GPU utilization reads 100% as soon as a
# continuous-batching engine has any work, so it cannot tell 3 replicas from 30. Use demand signals
# the engine exposes — `vllm:num_requests_waiting` (queue) **and** `vllm:num_requests_running`
# (occupied batch slots). Queue alone is a trap: once capacity catches up the queue drains to 0, the
# HPA proposes the minimum, and the pool oscillates. Cold starts of minutes (node + image + weights)
# make every scale-up late, so overshoot is the norm and rate limits trade overshoot for drain time.
# Background: [PRIMER §4 Autoscaling](../../PRIMER.md).

# %%
import threading
import time

from igwlab.autoscale import (DEFAULT_BEHAVIOR, Behavior, FluidPool, HPARecommender, PodSample, ScalingPolicy,
                              ScalingRules, hpa_manifest, plain_metric_replicas, simulate)
from igwlab.bench import agentic_sessions, ascii_bars, run_bench
from igwlab.promtext import Families
from igwlab.stack import LocalStack

# %% [markdown]
# ## The core formula, and what "ready pods" means
#
# For a `type: Pods` metric with an `AverageValue` target, the controller averages the metric over the
# pods that reported it (in integer milli-units), and proposes `ceil(ratio × pods)`:

# %%
def pods(*values):
    return [PodSample(f"pod-{i}", v) for i, v in enumerate(values)]

print("3 pods, 12 waiting each, target 5  ->", plain_metric_replicas(pods(12, 12, 12), 3, 5))   # ceil(3 * 2.4) = 8
print("3 pods, 5.4 waiting each, target 5 ->", plain_metric_replicas(pods(5.4, 5.4, 5.4), 3, 5), "(within 10%: no change)")
print("2 ready + 1 starting, 12 waiting    ->",
      plain_metric_replicas(pods(12, 12) + [PodSample("new", None, phase="Pending")], 3, 5),
      "(the starting pod counts as 0 on a scale-up)")

# %% [markdown]
# The last line is how the HPA avoids stampeding while new pods start: an unready pod is assumed to
# carry **zero** load on a scale-up (and a pod with *missing* metrics is assumed to be exactly at
# target on a scale-down). With multi-minute LLM cold starts this damping matters.
#
# ## Exercise 3.1 — the proportional step
#
# Implement the common case (every pod ready and reporting): `desired(current, avg, target, tol)`
# returns `current` if `1 - tol <= avg/target <= 1 + tol`, else `ceil(current × avg / target)`.
# (Write the band exactly like that: `abs(1 - ratio) <= tol` differs at the edge in floating point —
# `abs(1 - 1.1)` is `0.10000000000000009` — and the controller keeps the replica count at exactly 10%.)

# %% exercise
import math

def desired(current: int, avg: float, target: float, tol: float = 0.1) -> int:
    ### BEGIN SOLUTION
    ratio = avg / target
    if 1.0 - tol <= ratio <= 1.0 + tol:
        return current
    return math.ceil(current * ratio)
    ### END SOLUTION

# %% check
import random
rng = random.Random(3)
for _ in range(500):
    n = rng.randint(1, 12)
    avg = round(rng.uniform(0, 30), 1)
    tgt = rng.choice([1, 2, 5, 8])
    assert desired(n, avg, tgt) == plain_metric_replicas(pods(*[avg] * n), n, tgt)[0], (n, avg, tgt)
print("✅ desired() agrees with the controller port on 500 random cases")

# %% [markdown]
# ## Stabilization and rate limits
#
# Two HPAs see the same proposals: `behavior` unset (the legacy path: scale-up capped at
# `max(2 × current, 4)` per sync, scale-down stabilized over 300 s) and `behavior` set to the API
# defaults (scale-up `max(+4 pods, +100%)` per 15 s, same 300 s scale-down window).

# %%
proposals = [(0, 3, 20), (15, None, 20), (30, None, 20), (45, None, 2), (120, None, 2), (340, None, 2), (360, None, 2)]
for label, hpa in (("behavior unset", HPARecommender(1, 12)), ("default behavior", HPARecommender(1, 12, behavior=DEFAULT_BEHAVIOR))):
    cur = None
    print(f"--- {label}")
    for t, start, prop in proposals:
        cur = start if start is not None else cur
        step = hpa.reconcile(t, cur, prop)
        print("  ", step)
        cur = step.desired

# %% [markdown]
# ## Exercise 3.2 — the scale-down stabilization window
#
# Legacy path: the controller keeps every recommendation it made (seeded with the replica count at
# its first reconcile) and returns `max(desired, max of recommendations recorded at t >= now - window)`.
# Implement it; `history` is a list of `(t, replicas)` recorded *before* this reconcile.

# %% exercise
def stabilized_down(history, now: float, desired: int, window: float = 300) -> int:
    ### BEGIN SOLUTION
    recent = [r for t, r in history if t >= now - window]
    return max([desired] + recent)
    ### END SOLUTION

# %% check
hpa = HPARecommender(1, 20)
rng = random.Random(7)
cur, t = 6, 0
for i in range(60):
    prop = rng.randint(1, 12)
    history = [tuple(r) for r in hpa.recommendations] or [(t, cur)]
    step = hpa.reconcile(t, cur, prop)
    assert step.stabilized == stabilized_down(history, t, prop), (t, history, prop)
    cur = step.desired
    t += rng.choice([15, 15, 30, 120])
print("✅ stabilized_down reproduces the controller's window on 60 random reconciles")

# %% [markdown]
# ## Exercise 3.3 — the default scale-up limit
#
# With `behavior` set, the default scale-up rules are `Pods: 4 per 15 s` and `Percent: 100 per 15 s`
# with `selectPolicy: Max`. Each policy starts from the replica count at the *start of its period*
# (`current - added in the last 15 s`); `Pods` adds its value, `Percent` multiplies and rounds **up**;
# `Max` takes the larger. Write `scale_up_limit(current, added_last_15s)`.

# %% exercise
def scale_up_limit(current: int, added_last_15s: int = 0) -> int:
    ### BEGIN SOLUTION
    start = current - added_last_15s
    return max(start + 4, math.ceil(start * 2))
    ### END SOLUTION

# %% check
assert scale_up_limit(1) == 5 and scale_up_limit(3) == 7 and scale_up_limit(10) == 20
assert scale_up_limit(5, added_last_15s=4) == 5          # 4 already added this period: start = 1 -> limit 5
for cur in range(1, 30):
    for added in range(0, cur):
        want = HPARecommender._up_limit(cur, [[100.0, added, False]] if added else [], [], DEFAULT_BEHAVIOR.scale_up, 101.0)
        assert scale_up_limit(cur, added) == want
print("✅ scale_up_limit matches the controller's default policies")

# %% [markdown]
# ## Live: what the engines' metrics say under load
#
# Now the signals themselves. 40 agent sessions hit 3 fake backends with 8 batch slots each (24
# slots), so requests queue. Every 0.2 s we scrape the backends — exactly what Managed Prometheus
# would do every 15 s — and compute what an HPA would propose from each signal (current = 3 pods).
# "gpu busy" is the duty cycle `nvidia-smi` reports: 1.0 whenever anything runs.

# %%
samples = []
sessions = agentic_sessions(n_sessions=40, turns=3, n_agents=4, system_words=1200, tool_words=400, seed=1)
with LocalStack(3, "default-weighted") as s:
    runner = threading.Thread(target=lambda: samples.append(("bench", run_bench(s.router_url, sessions, stagger_s=0.2))))
    t0 = time.perf_counter()
    runner.start()
    while runner.is_alive():
        fams = [Families.from_text(txt) for txt in s.backend_metrics().values()]
        w = [f.sum("vllm:num_requests_waiting") for f in fams]
        r = [f.sum("vllm:num_requests_running") for f in fams]
        kv = [f.max("vllm:kv_cache_usage_perc") for f in fams]
        samples.append((time.perf_counter() - t0, w, r, kv))
        time.sleep(0.2)
    runner.join()

print(f"{'t(s)':>5} {'waiting/pod':>12} {'running/pod':>12} {'kv/pod':>7} {'gpu busy':>9} | proposal: waiting@2 running@6 kv@0.6")
for t, w, r, kv in [x for x in samples if x[0] != "bench"][::2]:
    avg = lambda xs: sum(xs) / len(xs)
    busy = sum(1.0 for x in r if x > 0) / len(r)
    p_w = plain_metric_replicas(pods(*w), 3, 2)[0]
    p_r = plain_metric_replicas(pods(*r), 3, 6)[0]
    p_kv = plain_metric_replicas(pods(*kv), 3, 0.6)[0]
    print(f"{t:>5.1f} {avg(w):>12.2f} {avg(r):>12.2f} {avg(kv):>7.3f} {busy:>9.2f} | {p_w:>17} {p_r:>9} {p_kv:>6}")

# %% [markdown]
# While the burst lasts, the queue and running-slot signals ask for more replicas and relax as the
# sessions finish; "gpu busy" is 1.0 the whole time and would have asked for the maximum forever.
# KV usage stays low here because the fake backend's KV pool is large relative to these prompts — on
# a real L4 with long agent contexts it is often the first signal to saturate.
#
# ## Exercise 3.4 — a queue target from a latency budget
#
# On one replica, prefills run (roughly) one after another, so a request that finds `N` requests
# waiting ahead of it pays about `N × prefill_time` of queueing before its own prefill. If the TTFT
# SLO leaves `budget_s` for queueing and an average prefill takes `prefill_s`, the HPA should keep
# the average waiting count per pod at most `floor(budget_s / prefill_s)` — but never below 1.

# %% exercise
def waiting_target(budget_s: float, prefill_s: float) -> int:
    ### BEGIN SOLUTION
    return max(1, math.floor(budget_s / prefill_s))
    ### END SOLUTION

# %% check
assert waiting_target(0.3, 0.06) == 5
assert waiting_target(0.05, 0.08) == 1
assert waiting_target(2.0, 0.25) == 8
print("✅ a 300 ms queueing budget with 60 ms prefills -> target 5 waiting per pod (deploy/gke/hpa.yaml uses 5)")

# %% [markdown]
# ## A simulated pool with cold starts
#
# The fluid model (`FluidPool`, *simulated*): each replica completes 2 requests/s, each request holds
# a batch slot for 4 s (8 slots per replica), and a new replica is ready **120 s** after it is
# requested. Load steps from 3 to 11 requests/s at t = 120 s and back to 3 at t = 900 s — 11 rps
# needs at least 5.5 replicas. First, the HPA on **queue length only**:

# %%
class StepLoad:
    horizon = 1500

    def __call__(self, t):
        return 3.0 if t < 120 else 11.0 if t < 900 else 3.0


def timeline(rows, every=4):
    print(f"{'t':>6} {'load':>5} {'ready':>6} {'repl.':>6} {'wait/pod':>9} {'run/pod':>8}  why")
    for r in rows[::every]:
        print(f"{r['t']:>6.0f} {r['load_rps']:>5.0f} {r['ready']:>6} {r['replicas']:>6} {r['waiting_per_pod']:>9.1f} "
              f"{r['running_per_pod']:>8.1f}  {r['why']}")

queue_only = simulate(StepLoad(), {"waiting": 5}, pool=FluidPool(ready=2))
timeline(queue_only)

# %% [markdown]
# Look at t ≈ 700 s: the backlog is gone, so `waiting/pod = 0`, the proposal is `ceil(0 × n) = 0`, and
# after the 300 s window the HPA scales to `minReplicas` — at full load. The queue explodes, the HPA
# scales back up (2+ minutes of cold start), and the cycle repeats. A queue measures *excess* demand,
# not demand.
#
# ## Exercise 3.5 — add the signal that measures demand
#
# Return the `targets` dict for `simulate()` — `{"waiting": ..., "running": ...}` — so that the pool
# no longer collapses. `running` is the average number of occupied batch slots per pod (8 slots per
# replica here); give it a target below 8 so there is headroom. The check requires: at least 6 ready
# replicas for the whole of t = 600…900 s, no more than 3 replicas at the end, and a `running` target
# that fits in the 8 slots.

# %% exercise
def my_targets() -> dict:
    ### BEGIN SOLUTION
    return {"waiting": 5, "running": 6}      # 75% of the slots: lam*S/6 = 11*4/6 -> 8 replicas at peak
    ### END SOLUTION

# %% check
tg = my_targets()
assert 0 < tg.get("running", 0) <= 8, "add a running-slots target that fits in 8 slots"
rows = simulate(StepLoad(), tg, pool=FluidPool(ready=2))
steady = [r["ready"] for r in rows if 600 <= r["t"] < 900]
assert min(steady) >= 6, f"pool collapsed to {min(steady)} ready replicas under full load"
assert rows[-1]["replicas"] <= 3, "the pool did not scale back down after the load dropped"
timeline(rows, every=6)
print("✅ queue + running: steady", min(steady), "-", max(steady), "ready replicas at 11 rps; back to", rows[-1]["replicas"])

# %% [markdown]
# Both signals still overshoot the ~8 replicas needed, because the backlog built during the 120 s
# cold start keeps the queue proposal high until the new replicas are ready. The levers: faster
# cold starts (image streaming, weight caching — layer 03), a buffer (`minReplicas` above the
# trough), or rate limits (`behavior.scaleUp.policies`) that trade overshoot for a longer drain.

# %%
slow_up = Behavior(scale_up=ScalingRules(0, "Max", (ScalingPolicy("Pods", 2, 60),)))
for label, hpa in (("default behavior", HPARecommender(1, 10, behavior=DEFAULT_BEHAVIOR)),
                   ("scaleUp 2 pods / 60 s", HPARecommender(1, 10, behavior=slow_up))):
    rows = simulate(StepLoad(), my_targets(), hpa=hpa, pool=FluidPool(ready=2))
    drained = next((r["t"] for r in rows if r["t"] > 150 and r["waiting_per_pod"] < 1 and r["load_rps"] > 5), None)
    print(f"{label:<24} peak replicas {max(r['replicas'] for r in rows):>2} | backlog drained at t={drained:.0f}s | "
          f"max waiting/pod {max(r['waiting_per_pod'] for r in rows):.0f}")

# %% [markdown]
# ## The manifest
#
# The same policy as an `autoscaling/v2` HPA. On GKE the metric name is the Managed Prometheus series
# as the Custom Metrics Stackdriver Adapter exposes it (`deploy/gke/hpa.yaml`; naming marked VERIFY).

# %%
import yaml
m = hpa_manifest("vllm-qwen", "vllm-qwen", "prometheus.googleapis.com|vllm:num_requests_waiting|gauge", 5,
                 min_replicas=1, max_replicas=4, behavior=Behavior(
                     scale_up=ScalingRules(0, "Max", (ScalingPolicy("Pods", 1, 60),)),
                     scale_down=ScalingRules(300, "Max", (ScalingPolicy("Pods", 1, 120),))))
m["spec"]["metrics"].append({"type": "Pods", "pods": {"metric": {"name": "prometheus.googleapis.com|vllm:num_requests_running|gauge"},
                                                      "target": {"type": "AverageValue", "averageValue": "6"}}})
print(yaml.safe_dump(m, sort_keys=False))
try:
    import kubernetes_validate
    kubernetes_validate.validate(m, "1.34.0", strict=True)
    print("valid autoscaling/v2 for Kubernetes 1.34")
except ImportError:
    print("(pip install kubernetes-validate to check the manifest offline)")

# %% [markdown]
# **Scale to zero.** `minReplicas: 0` needs the `HPAScaleToZero` feature gate, still alpha and off
# in Kubernetes 1.34 — and with 0 pods there is no per-pod metric to scale *up* from anyway (the
# controller reports `ScalingDisabled` when the target is at 0). Scale-from-zero needs a signal that
# exists without pods — a router-side queue (e.g. the EPP's flow-control queue) exported as an
# `External` metric — and a controller that owns the 0↔1 step, such as KEDA. See `usage_ratio_replicas`
# for the controller's from-zero arithmetic (`ceil(usage / target)`).
#
# ## In a design review
#
# **Two-minute walkthrough.** "We scale vLLM on two per-pod metrics from its own `/metrics`: waiting
# requests (target 5, from a 300 ms queueing budget and ~60 ms prefills) and running requests (target
# 6 of 8 batch slots). The HPA takes the larger proposal: the queue reacts to bursts, the running
# count holds capacity once the queue has drained — queue alone collapses the pool at full load. Not
# GPU utilization: it is 100% whenever any request runs. Scale-up is rate-limited to one pod per
# minute because each new L4 node needs minutes to pull the image and load weights, and faster
# scaling only overshoots; scale-down waits 300 s and removes one pod per two minutes. Minimum is one
# replica; scale-to-zero would need a router-side queue metric and KEDA."
#
# **Drill questions**
#
# 1. *Current 4 pods, average waiting 9, target 5 — what does the HPA propose, and what does the
#    default behavior allow in one step?* `ceil(4 × 1.8) = 8`; default limit `max(4+4, 8) = 8`, so 8.
# 2. *Why did the pool drop to one replica while load was still high?* The only metric was the queue;
#    with enough capacity the queue is 0, which proposes the minimum. Add a demand signal (running
#    slots, KV usage, or router in-flight requests).
# 3. *Why don't new, still-loading pods make the HPA scale up even more?* On a scale-up the
#    controller counts unready pods as zero load, which lowers the average and damps the proposal.
