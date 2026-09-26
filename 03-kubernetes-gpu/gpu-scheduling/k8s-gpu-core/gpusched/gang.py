"""Gangs and topology: place every pod of a group, inside one topology domain, or none of them.

The one idea: a distributed job is useful only when ALL of its pods run, and fast only when they
run close together (one NVLink domain, one sub-block, one block). A pod-at-a-time scheduler can
hand half of a job its GPUs and deadlock the cluster; a gang scheduler reserves the whole group
atomically and picks the smallest domain that holds it. The domain choice follows Kueue's
Topology-Aware Scheduling: BestFit for required/preferred levels, LeastFreeCapacity otherwise.
"""
from __future__ import annotations

from collections import defaultdict

from .cluster import _clock, Cluster, Node, Pod
from .plugins import FILTERS, run_filters


def pods_that_fit(pod: Pod, node: Node) -> int:
    """How many more copies of `pod` this node can take (0 if a non-resource filter rejects it)."""
    if run_filters(pod, node, FILTERS[:-1]):
        return 0
    per_resource = [node.free(r) // q for r, q in pod.requests.items() if q > 0]
    return max(0, min(per_resource)) if per_resource else 0


def best_fit(capacity: dict, n: int) -> dict | None:
    """Kueue TAS BestFit over sibling domains: take the domains with the most room first, but choose
    the last one as the tightest domain that holds what is left. None if they cannot hold n."""
    left, out = n, {}
    pool = sorted((k for k, c in capacity.items() if c > 0), key=lambda k: (-capacity[k], k))
    while left > 0 and pool:
        fits = [k for k in pool if capacity[k] >= left]
        k = min(fits, key=lambda k: (capacity[k], k)) if fits else pool[0]
        out[k] = min(capacity[k], left)
        left -= out[k]
        pool.remove(k)
    return out if left == 0 else None


def least_free_capacity(capacity: dict, n: int) -> dict | None:
    """Kueue TAS LeastFreeCapacity (its default for unconstrained pod sets): fill the smallest gaps."""
    left, out = n, {}
    for k in sorted((k for k, c in capacity.items() if c > 0), key=lambda k: (capacity[k], k)):
        if left == 0:
            break
        out[k] = min(capacity[k], left)
        left -= out[k]
    return out if left == 0 else None


def place_gang(cluster: Cluster, pods: list, required: str | None = None, preferred: str | None = None,
               algorithm=None) -> dict | None:
    """{pod name: node name} for ALL pods, or None. Binds nothing.

    required="subblock": every pod inside one sub-block, or wait. preferred="subblock": try one
    sub-block, then one block, then spread over the cluster. Neither: anywhere (unconstrained)."""
    shape, n = pods[0], len(pods)
    fit = algorithm or (best_fit if (required or preferred) else least_free_capacity)
    room = {node.topology: pods_that_fit(shape, node) for node in cluster.nodes.values()}
    node_at = {node.topology: node.name for node in cluster.nodes.values()}

    def capacity(depth: int, inside: tuple = ()) -> dict:
        caps = defaultdict(int)
        for path, k in room.items():
            if path[:len(inside)] == inside:
                caps[path[:depth]] += k
        return caps

    def split(domain: tuple, count: int) -> dict:      # push `count` pods down the tree, level by level
        if len(domain) == len(cluster.levels):
            return {node_at[domain]: count}
        out = {}
        for child, k in fit(capacity(len(domain) + 1, domain), count).items():
            out.update(split(child, k))
        return out

    level = required or preferred
    start = cluster.levels.index(level) + 1 if level else 0
    for depth in ([start] if required else range(start, -1, -1)):   # preferred: widen until it fits
        caps = capacity(depth)
        fitting = [d for d, c in caps.items() if c >= n]
        if fitting:
            domain = min(fitting, key=lambda d: (caps[d], d))       # one domain, the tightest that fits
            names = [name for name, k in split(domain, n).items() for _ in range(k)]
            return {p.name: name for p, name in zip(pods, names)}
    return None


def bind_gang(cluster: Cluster, pods: list, placement: dict) -> None:
    for p in pods:
        cluster.bind(p, placement[p.name])


def admit_gangs(cluster: Cluster, gangs: list, **placement_kw) -> list:
    """All-or-nothing in FIFO order: bind a gang only if every pod fits; return the names bound."""
    admitted = []
    for gang in gangs:
        placement = place_gang(cluster, gang, **placement_kw)
        if placement:
            bind_gang(cluster, gang, placement)
            admitted.append(gang[0].gang)
    return admitted


def interleave(*gangs) -> list:
    """Round-robin creation order (A1, B1, A2, B2, ...), as when two job controllers create pods at
    the same time; re-stamps each pod's creation order to match."""
    longest = max(len(g) for g in gangs)
    pods = [g[i] for i in range(longest) for g in gangs if i < len(g)]
    for p in pods:
        p.seq = next(_clock)
    return pods


def running(gang: list) -> bool:
    """A gang makes progress only when every member is placed."""
    return all(p.node for p in gang)
