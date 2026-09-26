"""The scheduling cycle: queue sort -> filter -> score -> bind, with preemption as the fallback.

The one idea: kube-scheduler places ONE pod at a time. It does not look ahead at the next pod,
does not treat a job's pods as a unit, and by default ignores GPUs when choosing among feasible
nodes. GPU fragmentation (free GPUs no pending pod can use) and partially placed jobs follow
directly from that - which is what the rest of this package fixes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .cluster import GPU, Cluster, Pod
from .plugins import DEFAULT_RESOURCES, FILTERS, SCORERS, pick_preemption_node, run_filters, select_victims


@dataclass
class Decision:
    pod: Pod
    node: str | None = None
    message: str = ""                               # FailedScheduling-style text when node is None
    scores: dict = field(default_factory=dict)
    victims: list = field(default_factory=list)


def fit_error(n_nodes: int, reasons_by_node: dict) -> str:
    """The FailedScheduling message: a histogram of per-node reasons, sorted as strings."""
    counts = Counter(r for reasons in reasons_by_node.values() for r in reasons)
    return f"0/{n_nodes} nodes are available: " + ", ".join(sorted(f"{c} {r}" for r, c in counts.items())) + "."


class Scheduler:
    """A queue of pending pods and a cycle that places them one at a time.

    strategy: "LeastAllocated" (upstream default: spread) or "MostAllocated" (bin-pack);
    resources: the (name, weight) pairs the score looks at - upstream default is cpu and memory.
    Preemption is collapsed to one step: victims are evicted and the preemptor bound at once
    (upstream nominates the node and binds after the victims' graceful termination)."""

    def __init__(self, cluster: Cluster, strategy: str = "LeastAllocated", resources=DEFAULT_RESOURCES,
                 preemption: bool = True, filters=FILTERS):
        self.cluster, self.scorer, self.resources = cluster, SCORERS[strategy], tuple(resources)
        self.preemption, self.filters = preemption, filters
        self.pending: list[Pod] = []

    def submit(self, *pods: Pod) -> None:
        self.pending.extend(pods)

    def schedule_one(self, pod: Pod) -> Decision:
        nodes = list(self.cluster.nodes.values())
        reasons = {n.name: run_filters(pod, n, self.filters) for n in nodes}
        feasible = [n for n in nodes if not reasons[n.name]]
        if feasible:
            scores = {n.name: self.scorer(pod, n, self.resources) for n in feasible}
            best = min(scores, key=lambda name: (-scores[name], name))   # upstream breaks ties randomly
            self.cluster.bind(pod, best)
            return Decision(pod, best, scores=scores)
        message = fit_error(len(nodes), reasons)
        if self.preemption and pod.preemption_policy != "Never":
            candidates, why = {}, {}
            for n in nodes:
                resolvable = reasons[n.name][0].startswith("Insufficient") and all(
                    q <= n.allocatable.get(r, 0) for r, q in pod.requests.items())
                victims = select_victims(pod, n, self.filters) if resolvable else None
                if victims:
                    candidates[n.name] = victims
                elif not resolvable:
                    why[n.name] = ["Preemption is not helpful for scheduling"]
                elif not any(p.priority < pod.priority for p in n.pods):
                    why[n.name] = ["No preemption victims found for incoming pod"]
                else:
                    why[n.name] = reasons[n.name]
            if candidates:
                target = pick_preemption_node(candidates)
                for v in candidates[target]:
                    self.cluster.evict(v)
                self.cluster.bind(pod, target)
                return Decision(pod, target, f"preempted {[v.name for v in candidates[target]]} on {target}",
                                victims=candidates[target])
            message += " preemption: " + fit_error(len(nodes), why)
        return Decision(pod, None, message)

    def run(self, max_passes: int = 100) -> list:
        """Drain the queue in PrioritySort order (priority desc, then creation) until a pass binds
        nothing. Preempted pods rejoin the queue, as if their controller recreated them."""
        decisions = []
        for _ in range(max_passes):
            progress = False
            for pod in sorted(self.pending, key=lambda p: (-p.priority, p.seq)):
                d = self.schedule_one(pod)
                decisions.append(d)
                if d.node:
                    self.pending.remove(pod)
                    self.pending.extend(d.victims)
                    progress = True
            if not progress or not self.pending:
                break
        return decisions


def stranded_gpus(cluster: Cluster, gpus_per_pod: int) -> int:
    """Free GPUs that cannot host one more pod of this size (they sit in too-small pieces)."""
    return sum(n.free(GPU) % gpus_per_pod for n in cluster.nodes.values())


def fragmentation(cluster: Cluster, gpus_per_pod: int) -> float:
    """Share of the free GPUs that are stranded for pods of this size (0 = none, 1 = all)."""
    free = cluster.free(GPU)
    return stranded_gpus(cluster, gpus_per_pod) / free if free else 0.0
