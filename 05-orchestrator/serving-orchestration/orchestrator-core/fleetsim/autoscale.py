"""Autoscaling: how many replicas — the Kubernetes HPA algorithm, faithfully, plus what makes LLM serving different
(which signal to scale on, the cold start, scale to zero).

Every 15 s (sync period) the HPA controller does, per metric:

    desired = ceil(current x currentMetric / target)         skip if the ratio is within tolerance (0.1)
    -> stabilization: scale-down uses the MAX recommendation of the last 300 s, scale-up the MIN of the last 0 s
    -> policies: default scale-up max(+4 pods, +100%) per 15 s; scale-down -100% per 15 s
    -> clamp to [minReplicas, maxReplicas]

This mirrors kube-controller-manager (pkg/controller/podautoscaler: replica_calculator.go, horizontal.go):
milli-unit averaging, strict time windows, and the conservative treatment of pods that are not ready yet
(counted as using 0 on a scale-up: replicas still loading weights count as capacity already on its way).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def within(ratio: float, up: float = 0.1, down: float = 0.1) -> bool:
    """The tolerance band: no scaling while 1 - down <= ratio <= 1 + up (default 0.1 each way)."""
    return (1.0 - down) <= ratio <= (1.0 + up)


def pods_metric_replicas(values, target, current, *, unready=0, missing=0, tol_up=0.1, tol_down=0.1) -> int:
    """Desired replicas for a per-pod metric with an AverageValue target (calcPlainMetricReplicas).
    values: the metric of each ready pod; unready: pods not ready yet; missing: running pods with no sample."""
    if not values:
        return current                                   # no metrics at all: the controller skips this metric
    tgt, ms = round(target * 1000), [round(v * 1000) for v in values]        # Kubernetes quantities: milli-units
    ratio = (sum(ms) // len(ms)) / tgt
    up_with_unready = unready > 0 and ratio > 1.0
    if not up_with_unready and not missing:
        return current if within(ratio, tol_up, tol_down) else math.ceil(ratio * len(ms))
    ms += [tgt if ratio < 1.0 else 0] * missing          # missing: at target on a scale-down, 0 on a scale-up
    ms += [0] * (unready if up_with_unready else 0)      # not-yet-ready pods count as idle on a scale-up
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
    """An HPA bound to one fleet signal.

    metric: 'waiting' (vllm:num_requests_waiting), 'running', 'inflight' (waiting + running), 'kv'
    (vllm:kv_cache_usage_perc) or 'gpu_util' (fraction of time a kernel ran — what nvidia-smi calls utilisation).
    kind='pods' averages per ready pod; kind='external' scales on the pool total including requests held at the
    gateway while nothing is ready (the KEDA pattern) — the only kind that may scale to zero."""

    METRICS = ("waiting", "running", "inflight", "kv", "gpu_util")

    def __init__(self, hpa: HPA, metric="waiting", target=4.0, kind="pods"):
        if metric not in self.METRICS or kind not in ("pods", "external"):
            raise ValueError(f"metric must be one of {self.METRICS}; kind 'pods' or 'external'")
        if hpa.min_replicas == 0 and kind != "external":
            raise ValueError("minReplicas: 0 needs an Object or External metric (the API server rejects it)")
        self.hpa, self.metric, self.target, self.kind = hpa, metric, target, kind

    def value(self, r, util: float) -> float:
        """This replica's sample of the metric (`util` = its busy fraction over the last sync period)."""
        return {"waiting": len(r.waiting), "running": len(r.running), "inflight": len(r.waiting) + len(r.running),
                "kv": r.pool.usage(), "gpu_util": util}[self.metric]

    def decide(self, now, current, values, unready, pool_total) -> int:
        tol = dict(tol_up=self.hpa.up.tolerance, tol_down=self.hpa.down.tolerance)
        if self.kind == "external":
            desired = external_metric_replicas(pool_total, self.target, current, **tol)
        else:
            desired = pods_metric_replicas(values, self.target, current, unready=unready, **tol)
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
