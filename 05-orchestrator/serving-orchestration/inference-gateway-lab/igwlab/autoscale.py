"""The Kubernetes HPA recommender, exactly — and what to feed it when the pods are LLM engines.

The one idea: the HPA is a proportional controller with guard rails.

    desired = ceil(ready_pods x current_average / target_average)          (per metric; max over metrics)
      unless |1 - current/target| <= 10%            -> keep current           (tolerance)
    then stabilize: scale-down may not go below the *highest* recommendation of the last 300 s,
                    scale-up may not go above the *lowest* of the scale-up window (default 0 s)
    then rate-limit: default scale-up allows max(+4 pods, +100%) per 15 s; scale-down 100% per 15 s
    then clamp to [minReplicas, maxReplicas]

This module is a line-by-line port of kube-controller-manager v1.34
(pkg/controller/podautoscaler/{horizontal.go, replica_calculator.go}): the milli-unit integer
average, the special cases for missing and not-yet-ready pods, both normalization paths
(`behavior` unset -> legacy 2x/4-pod limit; `behavior` set -> policies), the recommendation
and scale-event history. Time is passed in explicitly, so you can replay a trace.

What to scale an LLM server on: a *demand* signal the engine exposes — `vllm:num_requests_waiting`
(queue per pod), KV-cache usage, or router-side in-flight/queued requests — not GPU utilization,
which reads ~100% as soon as a continuous-batching engine has any work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Tolerances", "ScalingPolicy", "ScalingRules", "Behavior", "DEFAULT_BEHAVIOR", "PodSample",
           "MetricError", "milli", "plain_metric_replicas", "usage_ratio_replicas", "external_per_pod_replicas",
           "HPARecommender", "Step", "hpa_manifest", "FluidPool", "simulate"]

SYNC_PERIOD_S = 15                   # --horizontal-pod-autoscaler-sync-period
DOWNSCALE_STABILIZATION_S = 300      # --horizontal-pod-autoscaler-downscale-stabilization
DEFAULT_TOLERANCE = 0.1              # --horizontal-pod-autoscaler-tolerance
SCALE_UP_LIMIT_FACTOR, SCALE_UP_LIMIT_MINIMUM = 2.0, 4.0   # legacy path (behavior unset)
INT32_MAX, INT32_MIN = 2**31 - 1, -(2**31)


class MetricError(ValueError):
    """The controller could not compute a proposal from this metric (e.g. no pods reported it)."""


@dataclass(frozen=True)
class Tolerances:
    down: float = DEFAULT_TOLERANCE
    up: float = DEFAULT_TOLERANCE

    def is_within(self, ratio: float) -> bool:
        return (1.0 - self.down) <= ratio <= (1.0 + self.up)


def milli(v: float) -> int:
    """resource.Quantity.MilliValue(): the value in milli-units, rounded up."""
    return math.ceil(round(v * 1000, 6))


# ------------------------------------------------------------------ proposals from one metric
@dataclass(frozen=True)
class PodSample:
    """One pod as the replica calculator sees it."""
    name: str
    value: float | None               # None = no metric sample for this pod
    phase: str = "Running"            # Running | Pending | Failed
    deleting: bool = False


def plain_metric_replicas(pods, current_replicas: int, target: float, tol: Tolerances = Tolerances()) -> tuple[int, float]:
    """`type: Pods` (target AverageValue) — calcPlainMetricReplicas. Returns (replicas, average usage)."""
    if not pods:
        raise MetricError("no pods returned by selector while calculating replica count")
    ready, unready, missing, metrics = 0, [], [], {}
    for p in pods:                                         # groupPods, non-CPU branch
        if p.deleting or p.phase == "Failed":
            continue                                       # ignored
        if p.phase == "Pending":
            unready.append(p.name)
            continue
        if p.value is None:
            missing.append(p.name)
            continue
        metrics[p.name] = milli(p.value)
        ready += 1
    if not metrics:
        raise MetricError("did not receive metrics for targeted pods (pods might be unready)")
    t = milli(target)
    usage = sum(metrics.values()) // len(metrics)          # integer milli average, as in GetMetricUsageRatio
    ratio = usage / t
    scale_up_with_unready = bool(unready) and ratio > 1.0
    if not scale_up_with_unready and not missing:
        if tol.is_within(ratio):
            return current_replicas, usage / 1000
        return math.ceil(ratio * ready), usage / 1000
    if missing:
        fill = t if ratio < 1.0 else 0 if ratio > 1.0 else None   # scale-down: assume at target; scale-up: assume 0
        if fill is not None:
            for n in missing:
                metrics[n] = fill
    if scale_up_with_unready:
        for n in unready:
            metrics[n] = 0
    new_ratio = (sum(metrics.values()) // len(metrics)) / t
    if tol.is_within(new_ratio) or (ratio < 1.0 < new_ratio) or (ratio > 1.0 > new_ratio):
        return current_replicas, usage / 1000                 # too small a change, or direction flipped
    new = math.ceil(new_ratio * len(metrics))
    if (new_ratio < 1.0 and new > current_replicas) or (new_ratio > 1.0 and new < current_replicas):
        return current_replicas, usage / 1000
    return new, usage / 1000


def usage_ratio_replicas(current_replicas: int, usage: float, target: float, ready_pods: int,
                         tol: Tolerances = Tolerances()) -> int:
    """`type: Object`/`External` with target Value — getUsageRatioReplicaCount.
    Note the scale-from-zero branch: with 0 replicas, desired = ceil(usage/target)."""
    ratio = milli(usage) / milli(target)
    if current_replicas != 0:
        if tol.is_within(ratio):
            return current_replicas
        return math.ceil(ratio * ready_pods)
    return math.ceil(ratio)


def external_per_pod_replicas(status_replicas: int, values, target_average: float,
                              tol: Tolerances = Tolerances()) -> int:
    """`type: External` with target AverageValue — GetExternalPerPodMetricReplicas."""
    usage = sum(milli(v) for v in values)
    ratio = usage / (milli(target_average) * status_replicas)
    if not tol.is_within(ratio):
        return min(INT32_MAX, math.ceil(usage / milli(target_average)))
    return status_replicas


# ------------------------------------------------------------------ behavior (autoscaling/v2)
@dataclass(frozen=True)
class ScalingPolicy:
    type: str                 # "Pods" | "Percent"
    value: int
    period_s: int


@dataclass(frozen=True)
class ScalingRules:
    stabilization_window_s: int | None = None      # None on scaleDown -> the controller flag (300 s)
    select_policy: str = "Max"                     # Max | Min | Disabled
    policies: tuple = ()


@dataclass(frozen=True)
class Behavior:
    scale_up: ScalingRules = ScalingRules(0, "Max", (ScalingPolicy("Pods", 4, 15), ScalingPolicy("Percent", 100, 15)))
    scale_down: ScalingRules = ScalingRules(None, "Max", (ScalingPolicy("Percent", 100, 15),))


DEFAULT_BEHAVIOR = Behavior()       # what the API server fills in when you set *any* part of behavior


@dataclass
class Step:
    t: float
    current: int
    proposal: int | None
    desired: int
    reason: str
    stabilized: int | None = None
    limited: str = ""

    def __str__(self):
        return (f"t={self.t:>6.0f}s current={self.current} proposal={self.proposal} -> desired={self.desired}"
                f"  [{self.reason}{'; ' + self.limited if self.limited else ''}]")


def _changes_in_period(period_s: int, events, now: float) -> int:
    cutoff = now - period_s
    return sum(e[1] for e in events if e[0] > cutoff)       # timestamp.After(cutoff)


class HPARecommender:
    """One HorizontalPodAutoscaler's control loop state, replayable with explicit timestamps."""

    def __init__(self, min_replicas: int = 1, max_replicas: int = 10, behavior: Behavior | None = None,
                 downscale_stabilization_s: int = DOWNSCALE_STABILIZATION_S):
        self.min, self.max, self.behavior = min_replicas, max_replicas, behavior
        self.downscale_window = downscale_stabilization_s
        self.recommendations: list[list] = []      # [timestamp, replicas]
        self.up_events: list[list] = []            # [timestamp, change, outdated]
        self.down_events: list[list] = []
        self.history: list[Step] = []

    # -- the two normalization paths ------------------------------------------------------
    def _stabilize_legacy(self, desired: int, now: float) -> int:
        cutoff = now - self.downscale_window
        best, old = desired, None
        for i, rec in enumerate(self.recommendations):
            if rec[0] < cutoff:
                old = i
            elif rec[1] > best:
                best = rec[1]
        if old is not None:
            self.recommendations[old] = [now, desired]
        else:
            self.recommendations.append([now, desired])
        return best

    def _stabilize_behavior(self, current: int, desired: int, now: float) -> int:
        b = self.behavior
        up_w = b.scale_up.stabilization_window_s or 0
        down_w = b.scale_down.stabilization_window_s
        down_w = self.downscale_window if down_w is None else down_w
        up_cut, down_cut = now - up_w, now - down_w
        up_rec = down_rec = desired
        old = None
        for i, rec in enumerate(self.recommendations):
            if rec[0] > up_cut:
                up_rec = min(rec[1], up_rec)
            if rec[0] > down_cut:
                down_rec = max(rec[1], down_rec)
            if rec[0] < up_cut and rec[0] < down_cut:
                old = i
        rec = current
        if rec < up_rec:
            rec = up_rec
        if rec > down_rec:
            rec = down_rec
        if old is not None:
            self.recommendations[old] = [now, desired]
        else:
            self.recommendations.append([now, desired])
        return rec

    @staticmethod
    def _up_limit(current, up_events, down_events, rules: ScalingRules, now) -> int:
        if rules.select_policy == "Disabled":
            return current
        result, pick = (INT32_MAX, min) if rules.select_policy == "Min" else (INT32_MIN, max)
        for p in rules.policies:
            start = current - _changes_in_period(p.period_s, up_events, now) + _changes_in_period(p.period_s, down_events, now)
            proposed = start + p.value if p.type == "Pods" else math.ceil(start * (1 + p.value / 100))
            result = pick(result, proposed)
        return result

    @staticmethod
    def _down_limit(current, up_events, down_events, rules: ScalingRules, now) -> int:
        if rules.select_policy == "Disabled":
            return current
        result, pick = (INT32_MIN, max) if rules.select_policy == "Min" else (INT32_MAX, min)
        for p in rules.policies:
            start = current - _changes_in_period(p.period_s, up_events, now) + _changes_in_period(p.period_s, down_events, now)
            proposed = start - p.value if p.type == "Pods" else int(start * (1 - p.value / 100))
            result = pick(result, proposed)
        return result

    def _rate_limit(self, current: int, desired: int, now: float) -> tuple[int, str]:
        b = self.behavior
        if desired > current:
            limit = max(current, self._up_limit(current, self.up_events, self.down_events, b.scale_up, now))
            allowed, why = (limit, "ScaleUpLimit") if self.max > limit else (self.max, "TooManyReplicas")
            if desired > allowed:
                return allowed, why
        elif desired < current:
            limit = min(current, self._down_limit(current, self.up_events, self.down_events, b.scale_down, now))
            allowed, why = (limit, "ScaleDownLimit") if self.min < limit else (self.min, "TooFewReplicas")
            if desired < allowed:
                return allowed, why
        return desired, ""

    def _legacy_rules(self, current: int, desired: int) -> tuple[int, str]:
        up_limit = int(max(SCALE_UP_LIMIT_FACTOR * current, SCALE_UP_LIMIT_MINIMUM))
        hi, why = (up_limit, "ScaleUpLimit") if self.max > up_limit else (self.max, "TooManyReplicas")
        if desired < self.min:
            return self.min, "TooFewReplicas"
        if desired > hi:
            return hi, why
        return desired, ""

    def _store_event(self, prev: int, new: int, now: float) -> None:
        if self.behavior is None:
            return
        rules, events, change = ((self.behavior.scale_up, self.up_events, new - prev) if new > prev
                                 else (self.behavior.scale_down, self.down_events, prev - new))
        longest = max((p.period_s for p in rules.policies), default=0)
        for e in events:
            if e[0] < now - longest:
                e[2] = True                                     # outdated -> reusable slot
        old = None
        for i, e in enumerate(events):
            if e[2]:
                old = i                                         # the controller reuses the *last* outdated slot
        if old is not None:
            events[old] = [now, change, False]
        else:
            events.append([now, change, False])

    # -- one reconcile ----------------------------------------------------------------------
    def reconcile(self, now: float, current: int, proposal: int | None) -> Step:
        """One pass of reconcileAutoscaler. `proposal` = max over metric proposals (None = metrics invalid)."""
        if not self.recommendations:
            self.recommendations = [[now, current]]            # recordInitialRecommendation
        if current == 0 and self.min != 0:
            step = Step(now, current, proposal, 0, "ScalingDisabled: the target is scaled to zero")
            step.desired = current
        elif current > self.max:
            step = Step(now, current, proposal, self.max, "Current number of replicas above Spec.MaxReplicas")
        elif current < self.min:
            step = Step(now, current, proposal, self.min, "Current number of replicas below Spec.MinReplicas")
        elif proposal is None:
            step = Step(now, current, None, current, "FailedGetMetrics: keeping current replicas")
        else:
            if self.behavior is None:
                stab = self._stabilize_legacy(proposal, now)
                desired, why = self._legacy_rules(current, stab)
            else:
                stab = self._stabilize_behavior(current, proposal, now)
                desired, why = self._rate_limit(current, stab, now)
            reason = ("above target" if proposal > current else "below target" if proposal < current else "at target")
            step = Step(now, current, proposal, desired, reason, stabilized=stab, limited=why)
        if step.desired != current:
            self._store_event(current, step.desired, now)
        self.history.append(step)
        return step


# ------------------------------------------------------------------ manifests
def hpa_manifest(name: str, deployment: str, metric: str, target_average_value, min_replicas: int = 1,
                 max_replicas: int = 4, behavior: Behavior | None = None, namespace: str | None = None,
                 metric_type: str = "Pods") -> dict:
    """An autoscaling/v2 HorizontalPodAutoscaler scaling `deployment` on a per-pod metric."""
    target = {"type": "AverageValue", "averageValue": str(target_average_value)}
    if metric_type == "Pods":
        metrics = [{"type": "Pods", "pods": {"metric": {"name": metric}, "target": target}}]
    elif metric_type == "External":
        metrics = [{"type": "External", "external": {"metric": {"name": metric}, "target": target}}]
    else:
        raise ValueError("metric_type must be Pods or External")
    spec = {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment},
            "minReplicas": min_replicas, "maxReplicas": max_replicas, "metrics": metrics}
    if behavior is not None:
        def rules(r: ScalingRules):
            out = {"selectPolicy": r.select_policy,
                   "policies": [{"type": p.type, "value": p.value, "periodSeconds": p.period_s} for p in r.policies]}
            if r.stabilization_window_s is not None:
                out = {"stabilizationWindowSeconds": r.stabilization_window_s, **out}
            return out
        spec["behavior"] = {"scaleUp": rules(behavior.scale_up), "scaleDown": rules(behavior.scale_down)}
    meta = {"name": name}
    if namespace:
        meta["namespace"] = namespace
    return {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler", "metadata": meta, "spec": spec}


# ------------------------------------------------------------------ a fluid model to drive the loop (simulated)
@dataclass
class FluidPool:
    """A deliberately simple *simulated* serving pool: each ready replica completes `mu` req/s;
    unserved requests queue; new replicas become ready `cold_start_s` after being requested."""
    mu: float = 2.0
    cold_start_s: float = 120.0
    ready: int = 1
    backlog: float = 0.0
    pending: list = field(default_factory=list)       # ready-at timestamps

    def replicas(self) -> int:
        return self.ready + len(self.pending)

    def scale_to(self, n: int, now: float) -> None:
        cur = self.replicas()
        if n > cur:
            self.pending += [now + self.cold_start_s] * (n - cur)
        elif n < cur:
            drop = cur - n
            while drop and self.pending:              # cancel not-yet-ready pods first
                self.pending.pop()
                drop -= 1
            self.ready -= drop

    def advance(self, now: float, arrival_rate: float, dt: float) -> None:
        self.ready += sum(1 for t in self.pending if t <= now)
        self.pending = [t for t in self.pending if t > now]
        served = min(self.backlog + arrival_rate * dt, self.ready * self.mu * dt)
        self.backlog = max(0.0, self.backlog + arrival_rate * dt - served)

    def waiting_per_pod(self) -> float:
        return self.backlog / max(1, self.ready)

    def busy_fraction(self, arrival_rate: float) -> float:
        """A GPU-utilization-like signal: 1.0 whenever there is a queue or demand >= capacity."""
        if self.backlog > 0:
            return 1.0
        return min(1.0, arrival_rate / max(1e-9, self.ready * self.mu))


def simulate(load, target_waiting: float = 5.0, hpa: HPARecommender | None = None, pool: FluidPool | None = None,
             dt: float = 1.0, sync_s: int = SYNC_PERIOD_S):
    """Replay `load(t) -> requests/s` through FluidPool + HPA on `vllm:num_requests_waiting`
    (AverageValue target). Returns a list of dict rows, one per HPA sync (simulated)."""
    hpa = hpa or HPARecommender(1, 10)
    pool = pool or FluidPool()
    rows, t = [], 0.0
    horizon = getattr(load, "horizon", 1800)
    while t <= horizon:
        lam = load(t)
        pool.advance(t, lam, dt)
        if int(t) % sync_s == 0:
            pods = [PodSample(f"p{i}", pool.waiting_per_pod()) for i in range(pool.ready)]
            pods += [PodSample(f"pending{i}", None, phase="Pending") for i in range(len(pool.pending))]
            cur = pool.replicas()
            try:
                proposal, avg = plain_metric_replicas(pods, cur, target_waiting)
            except MetricError:
                proposal, avg = None, float("nan")
            step = hpa.reconcile(t, cur, proposal)
            if step.desired != cur:
                pool.scale_to(step.desired, t)
            rows.append({"t": t, "load_rps": lam, "ready": pool.ready, "replicas": pool.replicas(),
                         "waiting_per_pod": avg, "busy": pool.busy_fraction(lam), "proposal": proposal,
                         "desired": step.desired, "why": step.limited or step.reason})
        t += dt
    return rows
