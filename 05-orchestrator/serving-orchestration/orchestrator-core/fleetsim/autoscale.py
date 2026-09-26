"""Autoscaling: how many replicas — the Kubernetes HPA algorithm, faithfully, plus what makes LLM serving different
(which signal to scale on, the cold start, scale to zero).

Every 15 s (sync period) the HPA controller does, per metric:

    desired = ceil(current x currentMetric / target)         skip if the ratio is within tolerance (0.1)
    -> stabilization: scale-down uses the MAX recommendation of the last 300 s, scale-up the MIN of the last 0 s
    -> policies: default scale-up max(+4 pods, +100%) per 15 s; scale-down -100% per 15 s
    -> clamp to [minReplicas, maxReplicas]

This mirrors kube-controller-manager (pkg/controller/podautoscaler: replica_calculator.go, horizontal.go):
milli-unit averaging, strict time windows, and its treatment of pods without a usable sample. For a Pods metric
(anything but CPU) the controller sorts pods into: *unready* = phase Pending (scheduling, image pull) — dropped on
a scale-down, counted as 0 on a scale-up; *missing* = Running with no sample (a vLLM still loading weights exports
no /metrics) — counted at the target on a scale-down, 0 on a scale-up. It checks the Ready condition only for CPU.
Note what this does NOT do: the zeros never raise the result (desired is ceil(sum of samples / target) either
way); they only make a small or direction-flipping change fall inside the tolerance and be skipped. A scale-up is
bounded because the ratio is multiplied by the number of pods that reported, not by the current replica count.
The Fleet treats a starting replica as Pending for its whole cold start (the same on a scale-up; on a scale-down
real pods past their image pull would count at the target instead of being dropped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def within(ratio: float, up: float = 0.1, down: float = 0.1) -> bool:
    """The tolerance band: no scaling while 1 - down <= ratio <= 1 + up (default 0.1 each way)."""
    return (1.0 - down) <= ratio <= (1.0 + up)


def pods_metric_replicas(values, target, current, *, unready=0, missing=0, tol_up=0.1, tol_down=0.1) -> int:
    """Desired replicas for a per-pod metric with an AverageValue target (calcPlainMetricReplicas).
    values: the samples of the pods that reported; unready: Pending pods; missing: Running pods with no sample."""
    if not values:
        return current                                   # no metrics at all: the controller skips this metric
    tgt, ms = round(target * 1000), [round(v * 1000) for v in values]        # Kubernetes quantities: milli-units
    ratio = (sum(ms) // len(ms)) / tgt
    up_with_unready = unready > 0 and ratio > 1.0
    if not up_with_unready and not missing:
        return current if within(ratio, tol_up, tol_down) else math.ceil(ratio * len(ms))
    if ratio != 1.0:                                     # missing: at target on a scale-down, 0 on a scale-up
        ms += [tgt if ratio < 1.0 else 0] * missing
    ms += [0] * (unready if up_with_unready else 0)      # Pending pods count as idle on a scale-up only
    new_ratio = (sum(ms) // len(ms)) / tgt
    if within(new_ratio, tol_up, tol_down) or (ratio < 1.0 < new_ratio) or (ratio > 1.0 > new_ratio):
        return current                                   # too small a change, or the direction flipped
    new = math.ceil(new_ratio * len(ms))
    if (new_ratio < 1.0 and new > current) or (new_ratio > 1.0 and new < current):
        return current
    return new


def external_metric_replicas(total, target_per_pod, current, *, tol_up=0.1, tol_down=0.1) -> int:
    """Desired replicas for an External metric with an AverageValue target — how KEDA drives an HPA:
    ceil(total / target), unless total / (target x current) is within tolerance. Works from zero replicas."""
    usage, tgt = round(total * 1000), round(target_per_pod * 1000)
    if current and within(usage / (tgt * current), tol_up, tol_down):
        return current
    return math.ceil(usage / tgt)


@dataclass(frozen=True)
class Policy:
    kind: str          # "Pods" or "Percent"
    value: int
    period_s: int


@dataclass(frozen=True)
class Rules:
    window_s: float                 # stabilizationWindowSeconds
    policies: tuple
    select: str = "Max"             # selectPolicy: Max (most change) | Min | Disabled
    tolerance: float = 0.1          # behavior.scale{Up,Down}.tolerance (GA in Kubernetes 1.37; verify yours)


DEFAULT_UP = Rules(0, (Policy("Percent", 100, 15), Policy("Pods", 4, 15)))
DEFAULT_DOWN = Rules(300, (Policy("Percent", 100, 15),))


class HPA:
    """One HorizontalPodAutoscaler's controller state: past recommendations and past scale events."""

    def __init__(self, min_replicas=1, max_replicas=10, up: Rules = DEFAULT_UP, down: Rules = DEFAULT_DOWN,
                 sync_s: float = 15.0):
        self.min_replicas, self.max_replicas, self.up, self.down, self.sync_s = min_replicas, max_replicas, up, down, sync_s
        self.recs: list[tuple[float, int]] = []          # (time, unstabilized desired)
        self.up_events: list[tuple[float, int]] = []     # (time, pods added)
        self.down_events: list[tuple[float, int]] = []   # (time, pods removed)

    @staticmethod
    def _changed(events, now, period):
        return sum(c for t, c in events if t > now - period)

    def _limit(self, now, current, rules, up: bool) -> int:
        """The replica bound the policies allow now (calculateScale{Up,Down}Limit...)."""
        if rules.select == "Disabled":
            return current
        props = []
        for p in rules.policies:
            start = current - self._changed(self.up_events, now, p.period_s) + self._changed(self.down_events, now, p.period_s)
            if up:
                props.append(start + p.value if p.kind == "Pods" else math.ceil(start * (1 + p.value / 100)))
            else:
                props.append(start - p.value if p.kind == "Pods" else int(start * (1 - p.value / 100)))
        most_change = max if up else min
        return most_change(props) if rules.select == "Max" else (min if up else max)(props)

    def step(self, now: float, current: int, desired: int) -> int:
        """One reconcile: the raw metric recommendation -> the replica count the controller sets."""
        if current > self.max_replicas or current < self.min_replicas:
            new = min(max(current, self.min_replicas), self.max_replicas)
        else:
            up_rec = down_rec = desired
            for t, d in self.recs:
                if t > now - self.up.window_s:
                    up_rec = min(up_rec, d)
                if t > now - self.down.window_s:
                    down_rec = max(down_rec, d)
            new = min(max(current, up_rec), down_rec)    # hold between the two stabilized bounds
            horizon = max(self.up.window_s, self.down.window_s)
            self.recs = [(t, d) for t, d in self.recs if t >= now - horizon] + [(now, desired)]
            if new > current:
                new = min(new, self.max_replicas, max(self._limit(now, current, self.up, True), current))
            elif new < current:
                new = max(new, self.min_replicas, min(self._limit(now, current, self.down, False), current))
        if new != current:
            events = self.up_events if new > current else self.down_events
            events.append((now, abs(new - current)))
            longest = max(p.period_s for r in (self.up, self.down) for p in r.policies)
            self.up_events = [e for e in self.up_events if e[0] > now - longest]
            self.down_events = [e for e in self.down_events if e[0] > now - longest]
        return new


class Autoscaler:
    """An HPA bound to one fleet signal, or several (`also=[(metric, target, kind), ...]`): the controller computes
    a replica count per metric and takes the largest.

    metric: 'waiting' (vllm:num_requests_waiting), 'running', 'inflight' (waiting + running), 'kv'
    (vllm:kv_cache_usage_perc), 'gpu_util' (fraction of time a kernel ran — what nvidia-smi calls utilisation) or
    'backlog_s' (seconds of prefill work owed: the replica's uncached prompt tokens not yet prefilled / prefill
    tokens/s — llm-d's token-aware path computes it as EPP in-flight tokens / peakPrefillThroughput and pairs it
    with KV occupancy for decode). 'backlog_s' counts work, not requests; the others count requests or memory.
    kind='pods' averages per ready pod; kind='external' scales on the pool total including requests held at the
    gateway or in the router's queue (the KEDA pattern) — the only kind that may scale to zero."""

    METRICS = ("waiting", "running", "inflight", "kv", "gpu_util", "backlog_s")

    def __init__(self, hpa: HPA, metric="waiting", target=4.0, kind="pods", also=()):
        self.hpa, self.metric, self.target, self.kind = hpa, metric, target, kind
        self.specs = [(metric, target, kind)] + list(also)
        for m, _, k in self.specs:
            if m not in self.METRICS or k not in ("pods", "external"):
                raise ValueError(f"metric must be one of {self.METRICS}; kind 'pods' or 'external'")
            if hpa.min_replicas == 0 and k != "external":
                raise ValueError("minReplicas: 0 needs an Object or External metric (the API server rejects it)")

    @staticmethod
    def value(r, util: float, metric: str) -> float:
        """One replica's sample of `metric` (`util` = its busy fraction over the last sync period)."""
        if metric == "backlog_s":
            return r.prefill_backlog() / r.p.compute_tok_s
        return {"waiting": len(r.waiting), "running": len(r.running), "inflight": len(r.waiting) + len(r.running),
                "kv": r.pool.usage(), "gpu_util": util}[metric]

    @staticmethod
    def held(reqs, profile, metric: str) -> float:
        """What requests not yet on any replica add to an External metric's pool total."""
        if metric == "backlog_s":
            return sum(q.prompt for q in reqs) / profile.compute_tok_s
        return len(reqs) if metric in ("waiting", "inflight") else 0.0

    def decide(self, now, current, ready, util, unready, held_reqs, profile) -> int:
        """One sync: a recommendation per metric, the largest wins, then stabilization and policies."""
        tol = dict(tol_up=self.hpa.up.tolerance, tol_down=self.hpa.down.tolerance)
        desired = 0
        for metric, target, kind in self.specs:
            vals = [self.value(r, util[r.rid], metric) for r in ready]
            if kind == "external":
                total = sum(vals) + self.held(held_reqs, profile, metric)
                desired = max(desired, external_metric_replicas(total, target, current, **tol))
            else:
                desired = max(desired, pods_metric_replicas(vals, target, current, unready=unready, **tol))
        return self.hpa.step(now, current, desired)


@dataclass(frozen=True)
class ColdStart:
    """From "the HPA said +1" to "the replica serves" — each term has a different fix (layer 03 §8)."""
    node_s: float = 0.0     # get a GPU node: 0 if one is free; minutes if the cluster autoscaler must add one
    pull_s: float = 0.0     # pull the serving image (image streaming, preloaded boot disks)
    load_s: float = 0.0     # read the weights: bytes / storage bandwidth (local SSD, parallel/streamed loads)
    init_s: float = 0.0     # engine start: KV allocation, CUDA-graph capture, compilation (cacheable)

    @property
    def total(self) -> float:
        return self.node_s + self.pull_s + self.load_s + self.init_s

    @classmethod
    def estimate(cls, weight_gb, load_gb_s, image_gb=10.0, pull_gb_s=0.25, node_s=0.0, init_s=30.0):
        return cls(node_s, image_gb / pull_gb_s, weight_gb / load_gb_s, init_s)
