"""What the scheduler sees: nodes with integer GPU counters, labels, taints and a topology path.

The one idea: to Kubernetes a GPU is an *extended resource*, an opaque integer on the node
(`status.allocatable["nvidia.com/gpu"]: 8`). A container asks for whole GPUs, its request must
equal its limit, and the scheduler never hands out more than allocatable. Everything else in
this package (filters, scores, gangs, quotas, autoscaling) is arithmetic on those integers
plus labels, taints and a topology path.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

GPU = "nvidia.com/gpu"
ACCELERATOR_LABEL = "cloud.google.com/gke-accelerator"          # GKE's GPU-type node label
TOPOLOGY_LABELS = ("cloud.google.com/gce-topology-block",       # GCE topology labels (verify)
                   "cloud.google.com/gce-topology-subblock",
                   "cloud.google.com/gce-topology-host")
_clock = itertools.count(1)          # creation / start order; stands in for timestamps


@dataclass(frozen=True)
class Taint:
    key: str
    value: str = ""
    effect: str = "NoSchedule"       # NoSchedule | PreferNoSchedule | NoExecute


@dataclass(frozen=True)
class Toleration:
    key: str = ""                    # empty key + Exists tolerates every taint
    operator: str = "Equal"          # Equal | Exists
    value: str = ""
    effect: str = ""                 # empty effect matches every effect

    def tolerates(self, taint: Taint) -> bool:
        """core/v1 Toleration.ToleratesTaint: effect, then key, then operator."""
        if self.effect and self.effect != taint.effect:
            return False
        if self.key and self.key != taint.key:
            return False
        if self.operator == "Exists":
            return True
        return self.operator in ("", "Equal") and self.value == taint.value


GPU_TAINT = Taint(GPU, "present", "NoSchedule")       # how GKE taints GPU nodes (verify)


def is_extended(resource: str) -> bool:
    """Extended resources have a domain prefix outside kubernetes.io (nvidia.com/gpu, ...)."""
    return "/" in resource and "kubernetes.io/" not in resource and not resource.startswith("requests.")


def effective_requests(requests: dict, limits: dict) -> dict:
    """The API server's rules for a container's extended resources; raises ValueError like it does.

    Integers only; if both are set they must be equal (no overcommit); a request with no limit is
    rejected; a limit with no request defaults the request to the limit."""
    out = dict(requests)
    for res in sorted(set(requests) | set(limits)):
        if not is_extended(res):
            continue
        for q in (requests.get(res), limits.get(res)):
            if q is not None and q != int(q):
                raise ValueError(f"{res}: Invalid value: {q}: must be an integer")
        if res not in limits:
            raise ValueError(f"limits[{res}]: Required value: Limit must be set for non overcommitable resources")
        if res in requests and requests[res] != limits[res]:
            raise ValueError(f"requests[{res}]: Invalid value: {requests[res]}: "
                             f"must be equal to {res} limit of {limits[res]}")
        out[res] = int(limits[res])
    return out


@dataclass(eq=False)
class Pod:
    name: str
    requests: dict = field(default_factory=dict)   # {"nvidia.com/gpu": 1, "cpu": 8000 (m), "memory": 65536 (MiB)}
    tolerations: list = field(default_factory=list)
    node_selector: dict = field(default_factory=dict)
    priority: int = 0
    preemption_policy: str = "PreemptLowerPriority"   # or "Never"
    gang: str | None = None                            # pods sharing a gang id run all-or-nothing
    node: str | None = None                            # set when bound
    started: int = 0                                   # bind order (the preemption tie-breaks use it)
    seq: int = field(default_factory=lambda: next(_clock))   # creation order (queue tie-break)

    @property
    def gpus(self) -> int:
        return self.requests.get(GPU, 0)


def extended_resource_toleration(pod: Pod) -> Pod:
    """What the ExtendedResourceToleration admission plugin does: tolerate a taint keyed by each
    extended resource the pod requests (operator Exists, effect NoSchedule)."""
    for res in sorted(r for r in pod.requests if is_extended(r) and pod.requests[r] > 0):
        tol = Toleration(res, "Exists", effect="NoSchedule")
        if tol not in pod.tolerations:
            pod.tolerations.append(tol)
    return pod


def gpu_pod(name: str, gpus: int = 1, *, cpu: int = 8000, memory: int = 65536, **kw) -> Pod:
    """A pod asking for whole GPUs, with the toleration the admission plugin would add."""
    return extended_resource_toleration(Pod(name, {GPU: gpus, "cpu": cpu, "memory": memory}, **kw))


@dataclass(eq=False)
class Node:
    name: str
    allocatable: dict                               # {"nvidia.com/gpu": 8, "cpu": 96000, "memory": 786432}
    labels: dict = field(default_factory=dict)
    taints: list = field(default_factory=list)
    topology: tuple = ()                            # coarse -> fine, e.g. ("b0", "b0-s1", "b0-s1-h3")
    unschedulable: bool = False
    pods: list = field(default_factory=list)

    def requested(self, resource: str) -> int:
        return sum(p.requests.get(resource, 0) for p in self.pods)

    def free(self, resource: str = GPU) -> int:
        return self.allocatable.get(resource, 0) - self.requested(resource)


def gpu_node(name: str, gpus: int = 8, *, topology: tuple = (), accelerator: str = "nvidia-h100-80gb",
             cpu: int = 96000, memory: int = 786432, labels: dict | None = None, taint: bool = True) -> Node:
    """A GKE-style GPU node: accelerator + topology labels, and the GPU taint. Sizes are illustrative."""
    lab = {ACCELERATOR_LABEL: accelerator, "kubernetes.io/hostname": name}
    lab.update(zip(TOPOLOGY_LABELS, topology))
    lab.update(labels or {})
    return Node(name, {GPU: gpus, "cpu": cpu, "memory": memory}, lab,
                [GPU_TAINT] if taint else [], tuple(topology))


class Cluster:
    """Nodes plus the names of the topology levels their `topology` paths use."""

    def __init__(self, nodes=(), levels=("block", "subblock", "host")):
        self.levels = tuple(levels)
        self.nodes: dict[str, Node] = {}
        for n in nodes:
            if len(n.topology) != len(self.levels):     # unlabelled nodes share one domain per level
                n.topology = ("-",) * (len(self.levels) - 1) + (n.name,)
            self.nodes[n.name] = n

    def bind(self, pod: Pod, node_name: str) -> None:
        pod.node, pod.started = node_name, next(_clock)
        self.nodes[node_name].pods.append(pod)

    def evict(self, pod: Pod) -> None:
        self.nodes[pod.node].pods.remove(pod)
        pod.node = None

    def free(self, resource: str = GPU) -> int:
        return sum(n.free(resource) for n in self.nodes.values())

    def show(self, resource: str = GPU) -> str:
        """One line per node: `#` = allocated, `.` = free."""
        rows = []
        for n in self.nodes.values():
            used, alloc = n.requested(resource), n.allocatable.get(resource, 0)
            rows.append(f"{n.name:<12} {'#' * used}{'.' * max(0, alloc - used)}  {used}/{alloc}")
        return "\n".join(rows)


def make_cluster(blocks: int = 1, subblocks: int = 1, hosts: int = 4, gpus: int = 8, **kw) -> Cluster:
    """blocks x subblocks x hosts identical GPU nodes; each node's topology path names its domains."""
    nodes = []
    for b, s, h in itertools.product(range(blocks), range(subblocks), range(hosts)):
        path = (f"b{b}", f"b{b}-s{s}", f"b{b}-s{s}-h{h}")
        nodes.append(gpu_node(path[-1], gpus, topology=path, **kw))
    return Cluster(nodes)
