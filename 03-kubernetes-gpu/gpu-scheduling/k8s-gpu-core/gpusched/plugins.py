"""Filter and score plugins, and the preemption fallback, the way kube-scheduler runs them.

The one idea: placement is two different questions. FILTER asks "can the pod run on this
node?"; each node passes, or fails with the reason of the first plugin that said no. SCORE asks
"which feasible node is best?" (0-100 per plugin, weighted). GPU fragmentation is decided in
the score step, and the default NodeResourcesFit score - LeastAllocated over cpu and memory -
does not look at GPUs at all.
"""
from __future__ import annotations

from .cluster import Node, Pod, Taint, is_extended

MAX_NODE_SCORE = 100
DEFAULT_RESOURCES = (("cpu", 1), ("memory", 1))   # kube-scheduler's default scoring resources
UNSCHEDULABLE = Taint("node.kubernetes.io/unschedulable", "", "NoSchedule")


# -- filters: [] means "passes"; otherwise the reasons, worded as kube-scheduler words them ------
def node_unschedulable(pod: Pod, node: Node) -> list:
    if node.unschedulable and not any(t.tolerates(UNSCHEDULABLE) for t in pod.tolerations):
        return ["node(s) were unschedulable"]
    return []


def taint_toleration(pod: Pod, node: Node) -> list:
    for taint in node.taints:                       # PreferNoSchedule only affects scoring
        if taint.effect in ("NoSchedule", "NoExecute") and not any(t.tolerates(taint) for t in pod.tolerations):
            return ["node(s) had untolerated taint(s)"]
    return []


def node_affinity(pod: Pod, node: Node) -> list:
    if all(node.labels.get(k) == v for k, v in pod.node_selector.items()):
        return []
    return ["node(s) didn't match Pod's node affinity/selector"]


def node_resources_fit(pod: Pod, node: Node) -> list:
    order = sorted(pod.requests, key=lambda r: (is_extended(r), r))       # cpu, memory, then extended
    return [f"Insufficient {r}" for r in order if 0 < pod.requests[r] and pod.requests[r] > node.free(r)]


FILTERS = (node_unschedulable, taint_toleration, node_affinity, node_resources_fit)


def run_filters(pod: Pod, node: Node, filters=FILTERS) -> list:
    """Run filters in order and stop at the first failure, as the framework does."""
    for f in filters:
        reasons = f(pod, node)
        if reasons:
            return reasons
    return []


# -- scores: NodeResourcesFit's two scoring strategies, integer arithmetic as upstream -------------
def _scored(pod: Pod, node: Node, resources) -> list:
    """(requested-after-placement, allocatable, weight) for each resource that is actually scored."""
    rows = []
    for name, weight in resources:
        req = pod.requests.get(name, 0)
        alloc = node.allocatable.get(name, 0)
        if (req == 0 and is_extended(name)) or alloc == 0:   # upstream skips these
            continue
        rows.append((node.requested(name) + req, alloc, weight))
    return rows


def least_allocated(pod: Pod, node: Node, resources=DEFAULT_RESOURCES) -> int:
    """Spread: sum(weight * (alloc - requested) * 100 // alloc) // sum(weight)."""
    rows = _scored(pod, node, resources)
    if not rows:
        return 0
    total = sum(w * ((a - r) * MAX_NODE_SCORE // a if r <= a else 0) for r, a, w in rows)
    return total // sum(w for _, _, w in rows)


def most_allocated(pod: Pod, node: Node, resources=DEFAULT_RESOURCES) -> int:
    """Bin-pack: sum(weight * min(requested, alloc) * 100 // alloc) // sum(weight)."""
    rows = _scored(pod, node, resources)
    if not rows:
        return 0
    total = sum(w * (min(r, a) * MAX_NODE_SCORE // a) for r, a, w in rows)
    return total // sum(w for _, _, w in rows)


SCORERS = {"LeastAllocated": least_allocated, "MostAllocated": most_allocated}


# -- preemption (the DefaultPreemption PostFilter) -----------------------------------------------
def select_victims(pod: Pod, node: Node, filters=FILTERS) -> list | None:
    """Remove every lower-priority pod; if the preemptor then fits, add them back most important
    first (higher priority, then earlier start) and keep only those it cannot fit without."""
    lower = [p for p in node.pods if p.priority < pod.priority]
    if not lower:
        return None
    saved = list(node.pods)
    node.pods[:] = [p for p in saved if p not in lower]
    try:
        if run_filters(pod, node, filters):
            return None
        victims = []
        for p in sorted(lower, key=lambda p: (-p.priority, p.started)):
            node.pods.append(p)
            if run_filters(pod, node, filters):
                node.pods.remove(p)
                victims.append(p)
        return victims
    finally:
        node.pods[:] = saved


def pick_preemption_node(candidates: dict) -> str:
    """Upstream order (PodDisruptionBudgets not modelled): lowest highest-victim priority, lowest
    sum of (priority + 2**31) - so fewer victims first - fewest victims, latest start of the
    highest-priority victims, then the first node."""
    def key(item):
        name, victims = item
        top = max(p.priority for p in victims)
        earliest_top = min(p.started for p in victims if p.priority == top)
        return (top, sum(p.priority + 2**31 for p in victims), len(victims), -earliest_top, name)
    return min(candidates.items(), key=key)[0]
