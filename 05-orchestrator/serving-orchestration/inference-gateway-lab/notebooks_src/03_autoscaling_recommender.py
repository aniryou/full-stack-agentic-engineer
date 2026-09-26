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
# make every scale-up late, so overshoot is the norm; scale-up rate limits trade GPU-hours and
# overshoot for a longer backlog. The targets come from the engine you deploy — its batch slots and
# its time per request — not from a rule of thumb. The basic rule, stabilization and the default
# policies are exercises in the core's notebook 03; here you work the parts the core leaves out:
# missing and starting pods, legacy vs `behavior`, live scrapes, and a queue target from a latency
# budget. Background: [PRIMER §4 Autoscaling](../../PRIMER.md).

# %%
import pathlib
import threading
import time

from igwlab.autoscale import (DEFAULT_BEHAVIOR, Behavior, FluidPool, HPARecommender, PodSample, ScalingPolicy,
                              ScalingRules, hpa_manifest, plain_metric_replicas, pods_from_scrapes,
                              recommend_from_scrapes, simulate)
from igwlab.bench import agentic_sessions, run_bench
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
# The last line is how the HPA avoids stampeding while new pods start. The rules
# (`calcPlainMetricReplicas`, ported in `plain_metric_replicas`):
#
# 1. The average and the proposal `ceil(ratio × n)` use only the **ready pods that reported** the metric.
# 2. If some pods are **not ready** and the ratio says *scale up*, recompute with each of them at **0**.
# 3. If some pods have **no metric**: on a scale-*down* assume each is exactly **at target**, on a
#    scale-*up* assume **0**; recompute.
# 4. After 2–3, keep the current count if the new ratio is within the ±10 % band **or points the other
#    way** (the fill flipped the direction), or if the new count would move against the direction.
#
# With multi-minute LLM cold starts, rules 2–4 are what stop a pool from over- or under-shooting on
# pods that are still loading weights or have not been scraped yet.
#
# ## Exercise 3.1 — predict the controller
#
# Predict the proposal for each case (target 5 waiting per pod). `None` = the pod has not reported
# the metric (a new pod GMP has not scraped yet); `"Pending"` = not ready (still loading weights).
# Write the four integers into `predicted`; a naive `ceil(current × average / target)` gets three
# of them wrong.
#
# | case | pods (waiting per pod) | current replicas |
# |---|---|---|
# | a | 1, 1, `None` | 3 |
# | b | 6, `None` | 2 |
# | c | 10 (target **2** here), `Pending`, `Pending`, `Pending` | 4 |
# | d | 3.9, 3.9, 3.9, 3.9, `None` | 5 |

# %% exercise
import math

predicted = []          # four integers: cases a, b, c, d
### BEGIN SOLUTION
predicted = [2,         # a: 0.2 -> scale-down; None at target: (1+1+5)/3 = 2.33 -> ceil(0.467 x 3) = 2 (naive: 1)
             2,         # b: 1.2 -> scale-up; None at 0: 3/5 = 0.6 flips the direction -> keep 2 (naive: 3)
             5,         # c: 5.0 -> scale-up; Pending at 0: 10/4 = 2.5 -> ceil(1.25 x 4) = 5 (naive: 20)
             5]         # d: 0.78 -> scale-down; None at 5: 4.12 -> ceil(0.824 x 5) = 5, no change (naive: 4)
### END SOLUTION

# %% check
def _pods(*values):
    return [PodSample(f"pending-{i}", None, phase="Pending") if v == "Pending" else PodSample(f"pod-{i}", v)
            for i, v in enumerate(values)]
cases = [(_pods(1, 1, None), 3, 5), (_pods(6, None), 2, 5), (_pods(10, "Pending", "Pending", "Pending"), 4, 2),
         (_pods(3.9, 3.9, 3.9, 3.9, None), 5, 5)]
want = [plain_metric_replicas(p, cur, tgt)[0] for p, cur, tgt in cases]
assert predicted == want, f"controller says {want}"
naive = [math.ceil(cur * (sum(x.value for x in p if x.value is not None) / sum(x.value is not None for x in p)) / tgt)
         for p, cur, tgt in cases]
print("controller:", want, "| naive ceil(current x avg / target):", naive)
print("✅ missing pods damp scale-downs, starting pods damp scale-ups, and a flipped direction means no change")

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
# ## Exercise 3.2 — legacy path or `behavior`?
#
# An HPA without a `behavior` block takes the *legacy* path: one sync may scale up to
# `max(2 × current, 4)`. Setting `behavior` — even to the API defaults — switches to the policies:
# `max(+4 pods, +100 %)` per 15 s, counted from the replica count at the start of the period. The
# two agree once the pool has 4 or more replicas, and differ below. Both HPAs start at **2**
# replicas (min 1, max 50) and see a proposal of **30** at t = 0, 15, 30 and 45 s. Predict the
# replica count each one sets at each sync.

# %% exercise
legacy_seq, behavior_seq = [], []          # four integers each
### BEGIN SOLUTION
legacy_seq = [4, 8, 16, 30]                # max(2x2, 4) = 4, then doubling, then the proposal caps it
behavior_seq = [6, 12, 24, 30]             # max(2+4, 2x2) = 6: +4 pods wins at small counts; then +100 %
### END SOLUTION

# %% check
for label, beh, mine in (("legacy", None, legacy_seq), ("behavior", DEFAULT_BEHAVIOR, behavior_seq)):
    h, cur, seen = HPARecommender(1, 50, behavior=beh), 2, []
    for t in (0, 15, 30, 45):
        cur = h.reconcile(t, cur, 30).desired
        seen.append(cur)
    assert mine == seen, (label, seen)
print("✅ legacy", legacy_seq, "vs behavior", behavior_seq, "— the same 30 is reached in the same four syncs,"
      " but from 2 replicas the policies add 4 pods where the legacy path adds 2")

# %% [markdown]
# ## Live: what the engines' metrics say under load
#
# Now the signals themselves. 40 agent sessions hit 3 fake backends with 8 batch slots each (24
# slots), so requests queue. Every 0.2 s we scrape the backends — what Managed Prometheus would do
# every 15 s — and `recommend_from_scrapes` computes what an HPA with one `type: Pods` metric per
# signal would propose (current = 3 pods; the HPA takes the largest valid proposal).
# "gpu busy" is the duty cycle `nvidia-smi` reports: 1.0 whenever anything runs.

# %%
samples = []
sessions = agentic_sessions(n_sessions=40, turns=3, n_agents=4, system_words=1200, tool_words=400, seed=1)
# targets sized for THIS fake engine (8 batch slots): running 6 = 75 % of them. A real deployment
# derives its own from its --max-num-seqs and measured request times (Exercise 3.4, deploy/gke/hpa.yaml).
targets = {"vllm:num_requests_waiting": 2, "vllm:num_requests_running": 6, "vllm:kv_cache_usage_perc": 0.6}
with LocalStack(3, "default-weighted") as s:
    runner = threading.Thread(target=lambda: run_bench(s.router_url, sessions, stagger_s=0.2))
    t0 = time.perf_counter()
    runner.start()
    while runner.is_alive():
        samples.append((time.perf_counter() - t0, s.backend_metrics()))     # {pod: /metrics text}
        time.sleep(0.2)
    runner.join()

print(f"{'t(s)':>5} {'waiting/pod':>12} {'running/pod':>12} {'kv/pod':>7} {'gpu busy':>9} | proposals (current 3): "
      "waiting@2 running@6 kv@0.6 -> HPA")
for t, scrapes in samples[::2]:
    best, per = recommend_from_scrapes(scrapes, targets, 3)
    avg = {m: sum(p.value for p in pods_from_scrapes(scrapes, m)) / 3 for m in targets}
    busy = sum(p.value > 0 for p in pods_from_scrapes(scrapes, "vllm:num_requests_running")) / 3
    print(f"{t:>5.1f} {avg['vllm:num_requests_waiting']:>12.2f} {avg['vllm:num_requests_running']:>12.2f} "
          f"{avg['vllm:kv_cache_usage_perc']:>7.3f} {busy:>9.2f} | "
          + " ".join(f"{per[m][0]:>9}" for m in targets) + f" -> {best}")

# %% [markdown]
# While the burst lasts, the queue and running-slot signals ask for more replicas and relax as the
# sessions finish; "gpu busy" is 1.0 the whole time and would have asked for the maximum forever.
# KV usage stays low here because the fake backend's KV pool is large relative to these prompts — on
# a real L4 with long agent contexts it is often the first signal to saturate.
#
# ## Exercise 3.3 — from raw scrapes to the HPA's proposal
#
# This is the path Managed Prometheus + the custom-metrics adapter + the HPA take, in one function.
# Write `hpa_proposal(scrapes, targets, current)`: `scrapes` is `{pod: /metrics text}`, `targets` is
# `{vLLM metric name: AverageValue target}`. Per metric, each pod's value is the sum of its series
# (one per engine rank) or `None` if the pod does not export it; get the metric's proposal from
# `plain_metric_replicas` (it raises `MetricError` when no pod reported); the HPA takes the **largest
# valid** proposal and ignores failed metrics — `None` if none is valid.

# %% exercise
from igwlab.autoscale import MetricError
from igwlab.promtext import Families

def hpa_proposal(scrapes: dict, targets: dict, current: int):
    ### BEGIN SOLUTION
    valid = []
    for metric, target in targets.items():
        pods_ = [PodSample(name, Families.from_text(text).sum(metric)) for name, text in sorted(scrapes.items())]
        try:
            valid.append(plain_metric_replicas(pods_, current, target)[0])
        except MetricError:
            continue
    return max(valid) if valid else None
    ### END SOLUTION

# %% check
for t, scrapes in samples:
    assert hpa_proposal(scrapes, targets, 3) == recommend_from_scrapes(scrapes, targets, 3)[0], t
page = 'vllm:num_requests_waiting{engine="0"} 9\nvllm:num_requests_waiting{engine="1"} 3\n'
assert hpa_proposal({"p0": page, "p1": "# not exporting yet\n"}, {"vllm:num_requests_waiting": 5,
                                                                    "vllm:kv_cache_usage_perc": 0.6}, 2) == 3
print(f"✅ hpa_proposal matches the controller on all {len(samples)} live scrapes (and on a pod that exports nothing)")

# %% [markdown]
# ## Exercise 3.4 — a queue target from a latency budget
#
# `vllm:num_requests_waiting` counts requests that have **no batch slot yet**. Once all `slots`
# (vLLM's `--max-num-seqs`) are busy, a slot frees whenever a running request finishes; if each
# request holds its slot for `service_s` seconds, slots free at `slots / service_s` per second (Little's
# law). A request that finds `N` waiting ahead of it therefore waits about `N × service_s / slots`
# for its slot. If the TTFT SLO leaves `budget_s` for that wait, keep the average waiting count per
# pod at most `floor(budget_s × slots / service_s)` — never below 1. (Waiting is not a queue of
# prefills: it is a queue for slots, which free at the pace of whole requests, decode included.)

# %% exercise
def waiting_target(budget_s: float, service_s: float, slots: int) -> int:
    ### BEGIN SOLUTION
    return max(1, math.floor(budget_s * slots / service_s + 1e-9))
    ### END SOLUTION

# %% check
import yaml
from igwlab import autoscale
assert waiting_target(0.5, 3.0, 32) == 5                     # floor(5.33)
assert waiting_target(0.05, 4.0, 8) == 1                     # never below 1
for args in ((0.3, 0.25, 8), (1.0, 2.2, 16), (0.2, 0.9, 256)):
    assert waiting_target(*args) == autoscale.waiting_target(*args), args
# the fake engine, measured in the live run above (emulated timing): mean time a request holds a slot
last = Families.from_text("".join(samples[-1][1].values()))
e2e = last.sum("vllm:e2e_request_latency_seconds_sum") / last.sum("vllm:e2e_request_latency_seconds_count")
queued = last.sum("vllm:request_queue_time_seconds_sum") / last.sum("vllm:request_queue_time_seconds_count")
print(f"fake engine (emulated): S = {e2e - queued:.3f} s per request in the batch, 8 slots "
      f"-> 0.3 s budget gives target {waiting_target(0.3, e2e - queued, 8)}")
# the GKE deployment: slots from vllm.yaml, S = 3 s ASSUMED for Qwen2.5-1.5B on an L4 (verify by measuring)
LAB = pathlib.Path.cwd().resolve()
while not (LAB / "igwlab").exists():
    LAB = LAB.parent
vllm_args = next(yaml.safe_load_all((LAB / "deploy/gke/vllm.yaml").read_text()))["spec"]["template"]["spec"]["containers"][0]["args"]
gke_slots = int(next(a.split("=")[1] for a in vllm_args if a.startswith("--max-num-seqs=")))
gke_hpa = yaml.safe_load((LAB / "deploy/gke/hpa.yaml").read_text())
gke_waiting = float(gke_hpa["spec"]["metrics"][0]["pods"]["target"]["averageValue"])
assert gke_waiting == waiting_target(0.5, 3.0, gke_slots)
print(f"✅ GKE: {gke_slots} slots, 0.5 s budget, S ~ 3 s (assumed) -> target {waiting_target(0.5, 3.0, gke_slots)}"
      " = deploy/gke/hpa.yaml")

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
# trough), or scale-up rate limits (`behavior.scaleUp.policies`). What a rate limit buys, with room
# to overshoot (maxReplicas 16):

# %%
def gpu_hours(rows, sync_s=15):
    return sum(r["replicas"] for r in rows) * sync_s / 3600

limits = {"default behavior": DEFAULT_BEHAVIOR,
          "scaleUp 2 pods / 60 s": Behavior(scale_up=ScalingRules(0, "Max", (ScalingPolicy("Pods", 2, 60),))),
          "scaleUp 1 pod / 60 s": Behavior(scale_up=ScalingRules(0, "Max", (ScalingPolicy("Pods", 1, 60),)))}
runs = {}
for label, beh in limits.items():
    rows = simulate(StepLoad(), my_targets(), hpa=HPARecommender(1, 16, behavior=beh), pool=FluidPool(ready=2))
    drained = next(r["t"] for r in rows if r["t"] > 150 and r["waiting_per_pod"] < 1 and r["load_rps"] > 5)
    runs[label] = (max(r["replicas"] for r in rows), gpu_hours(rows), drained, max(r["waiting_per_pod"] for r in rows))
    print(f"{label:<22} peak replicas {runs[label][0]:>2} | GPU-hours {runs[label][1]:.2f} | backlog drained at "
          f"t={drained:.0f}s | max waiting/pod {runs[label][3]:.0f}   (simulated)")
peaks, hours, drains, worst = zip(*runs.values())
assert list(peaks) == sorted(peaks, reverse=True) and list(hours) == sorted(hours, reverse=True)
assert list(drains) == sorted(drains) and len(set(worst)) == 1

# %% [markdown]
# Slower scale-up means fewer replicas at the peak and fewer GPU-hours, paid for with a backlog that
# drains minutes later. It does **not** change the worst queue: that builds during the first cold
# start, before any new replica exists, whatever the policy. Only a faster cold start or a buffer
# of warm capacity touches it.

# %% [markdown]
# ## The manifest
#
# The policy as an `autoscaling/v2` HPA for the GKE deployment: both metrics, with targets derived
# from that deployment's 32 batch slots (running: 75 % of them; waiting: Exercise 3.4), not the fake
# engine's 8. On GKE the metric names are the Managed Prometheus series as the Custom Metrics
# Stackdriver Adapter exposes them (naming marked VERIFY). The cell checks that this is exactly what
# `deploy/gke/hpa.yaml` ships.

# %%
gp = "prometheus.googleapis.com|{}|gauge"
m = hpa_manifest("vllm-qwen", "vllm-qwen", gp.format("vllm:num_requests_waiting"), waiting_target(0.5, 3.0, gke_slots),
                 extra_metrics={gp.format("vllm:num_requests_running"): int(0.75 * gke_slots)},
                 min_replicas=1, max_replicas=2, behavior=Behavior(
                     scale_up=ScalingRules(0, "Max", (ScalingPolicy("Pods", 1, 60),)),
                     scale_down=ScalingRules(300, "Max", (ScalingPolicy("Pods", 1, 120),))))
print(yaml.safe_dump(m, sort_keys=False))
assert m["spec"] == gke_hpa["spec"], "deploy/gke/hpa.yaml drifted from the derivation"
try:
    import kubernetes_validate
    kubernetes_validate.validate(m, "1.34.0", strict=True)
    print("valid autoscaling/v2 for Kubernetes 1.34")
except ImportError:
    print("(pip install kubernetes-validate to check the manifest offline)")

# %% [markdown]
# **Scale to zero.** `minReplicas: 0` needs the `HPAScaleToZero` feature gate, still alpha and off
# in Kubernetes 1.34 — and with 0 pods there is no per-pod metric to scale *up* from anyway (with
# `minReplicas` ≥ 1 the controller reports `ScalingDisabled` when the target has been scaled to 0;
# with the gate and `minReplicas: 0` it computes from Object/External metrics). Scale-from-zero needs a signal that
# exists without pods — a router-side queue (e.g. the EPP's flow-control queue) exported as an
# `External` metric — and a controller that owns the 0↔1 step, such as KEDA. See `usage_ratio_replicas`
# for the controller's from-zero arithmetic (`ceil(usage / target)`).
#
# ## In a design review
#
# **Two-minute walkthrough.** "We scale vLLM on two per-pod metrics from its own `/metrics`, with
# targets derived from the engine we deploy: 32 batch slots (`--max-num-seqs`, set explicitly).
# Running requests, target 24 — 75 % of the slots — holds capacity once the queue has drained;
# queue alone proposes the minimum at full load and collapses the pool. Waiting requests, target 5
# — Little's law: a waiting request needs a slot, slots free at 32 per ~3 s (an assumption we
# re-measure), so 5 waiting cost ~0.5 s of our TTFT budget. The HPA takes the larger proposal. Not
# GPU utilization: it is 100% whenever any request runs. Scale-up adds one pod a minute: each pod
# needs a new L4 Spot node, which quota and Spot obtainability limit anyway, and a slower ramp buys
# fewer GPU-hours for a longer backlog — it cannot shorten the queue built during the first cold
# start. Scale-down waits 300 s and removes one pod per two minutes. Minimum is one replica;
# scale-to-zero would need a router-side queue metric and KEDA."
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
