"""Predict what Kueue and the kube-scheduler will do with the kind lab's manifests.

The one idea: GPU scheduling on Kubernetes is two decisions in series. **Kueue** decides
*whether* a workload may start (quota in its ClusterQueue, borrowing within the cohort,
preemption) and — with Topology-Aware Scheduling — *which topology domain* its pods go to.
Only then does the **kube-scheduler** bind each pod to a node (filters: taints, selectors,
free ``nvidia.com/gpu``). Pods that bypass Kueue meet only the second decision.

This is a small, readable model of both, sized for the kind lab (one GPU flavor per queue,
whole-GPU requests, one resource that matters). It follows Kueue v0.19's documented rules:
classic preemption (candidates already being evicted first, then other queues, then lowest
priority, then most recently admitted; greedy, then minimised), TAS *BestFit* for required/preferred levels and
*LeastFreeCapacity* for workloads with no topology request, and the kube-scheduler's
``0/N nodes are available: ...`` message format. It is a predictor, not an emulator: where
the real system breaks ties randomly (the default scheduler among equal nodes) it says so.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import manifests as m
from .scenarios import FLAVOR, SCENARIOS, Scenario, scenario as get_scenario

LAB_ROOT = Path(__file__).resolve().parents[1]
KIND_DIR = LAB_ROOT / "deploy" / "kind"
CP_TAINT = {"key": "node-role.kubernetes.io/control-plane", "value": "", "effect": "NoSchedule"}
GPU_TAINT = dict(m.GPU_TAINT)


# ---- the cluster ------------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    labels: dict[str, str]
    taints: list[dict] = field(default_factory=list)
    gpus: int = 0                # allocatable nvidia.com/gpu
    used: int = 0                # requested by pods bound here

    @property
    def free(self) -> int:
        return self.gpus - self.used

    @property
    def host(self) -> str:
        return self.labels.get(m.TOPOLOGY_HOST, self.name)


def read_topology(path: Path | None = None, cluster: str = "gpu-lab") -> list[Node]:
    """The kind cluster from ``deploy/kind/topology.txt`` — the same file fake-gpus.sh reads."""
    path = path or KIND_DIR / "topology.txt"
    nodes = [Node(f"{cluster}-control-plane", {m.HOSTNAME: f"{cluster}-control-plane",
                                               "node-role.kubernetes.io/control-plane": ""}, [dict(CP_TAINT)])]
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        suffix, pool, gpus, accel, block, sub, host = line.split()
        name = f"{cluster}-{suffix}"
        labels = {m.HOSTNAME: name, m.GKE_NODEPOOL: pool}
        taints: list[dict] = []
        if int(gpus):
            labels.update({m.GKE_ACCELERATOR: accel, m.TOPOLOGY_BLOCK: block,
                           m.TOPOLOGY_SUBBLOCK: sub, m.TOPOLOGY_HOST: host})
            taints = [dict(GPU_TAINT)]
        nodes.append(Node(name, labels, taints, int(gpus)))
    return nodes


def kwok_nodes(blocks: int = 2, subblocks: int = 4, hosts: int = 4, gpus: int = 8) -> list[Node]:
    """The optional KWOK fleet exactly as kwok.sh names and labels it."""
    out = []
    for b in range(1, blocks + 1):
        for s in range(1, subblocks + 1):
            for h in range(1, hosts + 1):
                name = f"kwok-b{b}-s{s}-h{h}"
                out.append(Node(name, {m.HOSTNAME: name, "type": "kwok", m.GKE_NODEPOOL: "kwok-h100",
                                       m.GKE_ACCELERATOR: "nvidia-h100-80gb",
                                       m.TOPOLOGY_BLOCK: f"kwok-b{b}", m.TOPOLOGY_SUBBLOCK: f"kwok-b{b}-s{s}",
                                       m.TOPOLOGY_HOST: name},
                                [dict(GPU_TAINT), {"key": "kwok.x-k8s.io/node", "value": "fake", "effect": "NoSchedule"}],
                                gpus))
    return out


def nodes_from_k8s(items: Iterable[dict]) -> list[Node]:
    """Nodes from ``kubectl get nodes -o json`` (used when a live cluster is reachable)."""
    out = []
    for n in items:
        alloc = (n.get("status") or {}).get("allocatable") or {}
        out.append(Node(n["metadata"]["name"], dict(n["metadata"].get("labels") or {}),
                        [dict(t, value=t.get("value", "")) for t in (n.get("spec") or {}).get("taints") or []],
                        int(alloc.get(m.GPU, 0) or 0)))
    return out


# ---- matching helpers (the kube-scheduler's filters, simplified) ---------------------------------
def tolerates(taint: dict, tolerations: list[dict]) -> bool:
    """Filter semantics: only NoSchedule/NoExecute taints keep a pod off a node; matching is
    Kubernetes' own rule (``manifests.toleration_tolerates``)."""
    if taint.get("effect") not in ("NoSchedule", "NoExecute"):
        return True
    return m.tolerates(taint, tolerations)


def untolerated(node: Node, tolerations: list[dict]) -> dict | None:
    return next((t for t in node.taints if not tolerates(t, tolerations)), None)


def selector_matches(node: Node, selector: dict) -> bool:
    return all(node.labels.get(k) == v for k, v in (selector or {}).items())


# ---- workloads -------------------------------------------------------------------------------------
@dataclass
class PodSet:
    name: str
    count: int
    gpus: int
    selector: dict
    tolerations: list[dict]
    mode: str | None = None          # "required" | "preferred" | "unconstrained" | None (implied)
    level: str | None = None
    group: str | None = None


@dataclass
class Workload:
    key: str                          # "JobSet/team-a/gang-host", "LeaderWorkerSet/team-b/x#1"
    namespace: str
    queue: str | None                 # LocalQueue; None = not managed by Kueue
    priority: int
    podsets: list[PodSet]
    seq: int
    state: str = "pending"            # pending | admitted | preempted | running | unschedulable
    admitted_at: int = -1
    placement: dict[str, int] = field(default_factory=dict)   # node -> pods
    gpu_use: dict[str, int] = field(default_factory=dict)     # node -> GPUs held
    message: str = ""
    preempted: str = ""
    any_node: bool = False            # the default scheduler chose among equal nodes
    evicting: bool = False            # evicted but still holding quota (never True here: eviction is instant)

    @property
    def gpus(self) -> int:
        return sum(ps.count * ps.gpus for ps in self.podsets)


def _podsets(o: dict) -> list[PodSet]:
    kind, spec = o["kind"], o.get("spec") or {}

    def mk(name: str, count: int, tmpl: dict) -> PodSet:
        ps_spec = tmpl.get("spec") or {}
        ann = (tmpl.get("metadata") or {}).get("annotations") or {}
        mode = level = None
        for key, md in ((m.TAS_REQUIRED, "required"), (m.TAS_PREFERRED, "preferred")):
            if key in ann:
                mode, level = md, ann[key]
        if m.TAS_UNCONSTRAINED in ann:
            mode = "unconstrained"
        gpus = sum(m.gpu_count(c) for c in ps_spec.get("containers") or [])
        return PodSet(name, count, gpus, dict(ps_spec.get("nodeSelector") or {}),
                      list(ps_spec.get("tolerations") or []), mode, level, ann.get(m.TAS_GROUP))

    if kind == "Job":
        return [mk("main", int(spec.get("parallelism", 1)), spec["template"])]
    if kind == "JobSet":
        return [mk(rj["name"], int(rj.get("replicas", 1)) * int(rj["template"]["spec"].get("parallelism", 1)),
                   rj["template"]["spec"]["template"]) for rj in spec["replicatedJobs"]]
    if kind == "LeaderWorkerSet":
        lwt = spec["leaderWorkerTemplate"]
        size = int(lwt.get("size", 1))
        if lwt.get("leaderTemplate"):
            return [mk("leader", 1, lwt["leaderTemplate"]), mk("worker", size - 1, lwt["workerTemplate"])]
        return [mk("main", size, lwt["workerTemplate"])]
    raise ValueError(f"unsupported kind {kind}")


# ---- Kueue configuration ---------------------------------------------------------------------------
@dataclass
class ClusterQueue:
    name: str
    flavor: str
    nominal: int
    borrowing_limit: int | None
    lending_limit: int | None
    cohort: str | None
    namespaces: list[str] | None
    within: str = "Never"
    reclaim: str = "Never"


@dataclass
class KueueConfig:
    cqs: dict[str, ClusterQueue]
    lqs: dict[tuple[str, str], str]           # (namespace, name) -> ClusterQueue
    flavors: dict[str, dict]
    topologies: dict[str, list[str]]
    priorities: dict[str, int]

    @classmethod
    def from_objects(cls, objs: Iterable[dict]) -> "KueueConfig":
        cqs, lqs, flavors, topos, prios = {}, {}, {}, {}, {}
        for o in objs:
            kind, name, spec = o.get("kind"), o["metadata"]["name"], o.get("spec") or {}
            if kind == "ClusterQueue":
                fl = spec["resourceGroups"][0]["flavors"][0]
                gpu = next(r for r in fl["resources"] if r["name"] == m.GPU)
                ns = None
                for e in (spec.get("namespaceSelector") or {}).get("matchExpressions") or []:
                    if e["key"] == "kubernetes.io/metadata.name" and e["operator"] == "In":
                        ns = list(e["values"])
                pre = spec.get("preemption") or {}
                cqs[name] = ClusterQueue(name, fl["name"], int(gpu["nominalQuota"]),
                                         None if gpu.get("borrowingLimit") is None else int(gpu["borrowingLimit"]),
                                         None if gpu.get("lendingLimit") is None else int(gpu["lendingLimit"]),
                                         spec.get("cohortName"), ns, pre.get("withinClusterQueue", "Never"),
                                         pre.get("reclaimWithinCohort", "Never"))
            elif kind == "LocalQueue":
                lqs[(o["metadata"]["namespace"], name)] = spec["clusterQueue"]
            elif kind == "ResourceFlavor":
                flavors[name] = spec
            elif kind == "Topology":
                topos[name] = [lv["nodeLabel"] for lv in spec["levels"]]
            elif kind == "WorkloadPriorityClass":
                prios[name] = int(o["value"])
        return cls(cqs, lqs, flavors, topos, prios)

    @classmethod
    def from_dir(cls, path: Path | None = None) -> "KueueConfig":
        path = path or KIND_DIR / "manifests"
        return cls.from_objects(o for f in sorted(path.glob("*.yaml")) for o in m.load_all(f))


# ---- the simulator -------------------------------------------------------------------------------
class Sim:
    """Cluster state + the two schedulers. ``apply``/``delete``/``scale`` mirror kubectl."""

    def __init__(self, nodes: list[Node], config: KueueConfig):
        self.nodes = nodes
        self.cfg = config
        self.workloads: dict[str, Workload] = {}
        self.objects: dict[str, dict] = {}
        self.seq = 0
        self.clock = 0
        self.log: list[str] = []

    # -- bookkeeping --------------------------------------------------------------------------------
    def node(self, name: str) -> Node:
        return next(n for n in self.nodes if n.name == name)

    def _bind(self, w: Workload, placement: dict[str, int], gpu_use: dict[str, int]) -> None:
        for node, gpus in gpu_use.items():
            self.node(node).used += gpus
        w.placement, w.gpu_use = dict(placement), dict(gpu_use)

    def _unbind(self, w: Workload) -> None:
        for node, gpus in w.gpu_use.items():
            self.node(node).used -= gpus
        w.placement, w.gpu_use = {}, {}

    def usage(self, cq: str, without: Iterable[Workload] = ()) -> int:
        skip = {id(x) for x in without}
        return sum(w.gpus for w in self.workloads.values()
                   if w.state == "admitted" and self.cq_of(w) == cq and id(w) not in skip)

    def cq_of(self, w: Workload) -> str | None:
        return self.cfg.lqs.get((w.namespace, w.queue)) if w.queue else None

    # -- kubectl verbs ----------------------------------------------------------------------------
    def apply(self, o: dict) -> None:
        kind, meta = o["kind"], o["metadata"]
        key = f"{kind}/{meta.get('namespace', 'default')}/{meta['name']}"
        self.objects[key] = o
        labels = meta.get("labels") or {}
        queue = labels.get(m.QUEUE_LABEL)
        prio = self.cfg.priorities.get(labels.get(m.PRIORITY_LABEL, ""), 0)
        groups = int((o.get("spec") or {}).get("replicas", 1)) if kind == "LeaderWorkerSet" else 1
        for g in range(groups):
            self._new_workload(key + (f"#{g}" if kind == "LeaderWorkerSet" else ""), o, queue, prio)
        self.settle()

    def _new_workload(self, key: str, o: dict, queue: str | None, prio: int) -> None:
        self.seq += 1
        w = Workload(key, o["metadata"].get("namespace", "default"), queue, prio, _podsets(o), self.seq)
        self.workloads[key] = w
        if not queue:
            self._schedule_plain(w)

    def delete(self, key: str) -> None:
        for k in [k for k in self.workloads if k == key or k.startswith(key + "#")]:
            w = self.workloads.pop(k)
            self._unbind(w)
        self.objects.pop(key, None)
        self.settle()

    def scale(self, key: str, replicas: int) -> None:
        o = self.objects[key]
        o.setdefault("spec", {})["replicas"] = replicas
        labels = o["metadata"].get("labels") or {}
        existing = sorted(int(k.split("#")[1]) for k in self.workloads if k.startswith(key + "#"))
        for g in range(len(existing), replicas):
            self._new_workload(f"{key}#{g}", o, labels.get(m.QUEUE_LABEL),
                               self.cfg.priorities.get(labels.get(m.PRIORITY_LABEL, ""), 0))
        for g in existing[replicas:]:
            w = self.workloads.pop(f"{key}#{g}")
            self._unbind(w)
        self.settle()

    # -- the kube-scheduler (pods Kueue does not manage) ----------------------------------------------
    def filter_nodes(self, ps: PodSet) -> tuple[list[Node], Counter, dict[str, bool]]:
        """Filter plugins in the default order: TaintToleration, NodeAffinity, NodeResourcesFit.
        Returns feasible nodes, the reason histogram and, per node, whether it was unresolvable."""
        feasible, reasons, unresolvable = [], Counter(), {}
        for n in self.nodes:
            t = untolerated(n, ps.tolerations)
            if t is not None:
                reasons[f"node(s) had untolerated taint {{{t['key']}: {t.get('value', '')}}}"] += 1
                unresolvable[n.name] = True
            elif not selector_matches(n, ps.selector):
                reasons["node(s) didn't match Pod's node affinity/selector"] += 1
                unresolvable[n.name] = True
            elif ps.gpus > n.free:
                reasons[f"Insufficient {m.GPU}"] += 1
                unresolvable[n.name] = ps.gpus > n.gpus
            else:
                feasible.append(n)
        return feasible, reasons, unresolvable

    def _schedule_plain(self, w: Workload) -> None:
        ps = w.podsets[0]
        placement, gpu_use = Counter(w.placement), Counter(w.gpu_use)
        for _ in range(ps.count - sum(placement.values())):
            feasible, reasons, unresolvable = self.filter_nodes(ps)
            if not feasible:
                w.state, w.message = "unschedulable", fit_error(len(self.nodes), reasons, unresolvable)
                break
            if len(feasible) > 1:
                w.any_node = True   # equal scores: the real scheduler picks at random
            n = feasible[0]
            n.used += ps.gpus
            placement[n.name] += 1
            gpu_use[n.name] += ps.gpus
        else:
            w.state, w.message = "running", ""
        w.placement, w.gpu_use = dict(placement), dict(gpu_use)

    # -- Kueue ------------------------------------------------------------------------------------------
    def cohort_members(self, cq: ClusterQueue) -> list[ClusterQueue]:
        return [c for c in self.cfg.cqs.values() if cq.cohort and c.cohort == cq.cohort]

    def available(self, cq: ClusterQueue, without: Iterable[Workload] = ()) -> int:
        """GPUs this ClusterQueue can still admit: its own unused quota plus what it may borrow."""
        without = list(without)
        use = self.usage(cq.name, without)
        if not cq.cohort:
            return cq.nominal - use
        members = self.cohort_members(cq)
        guaranteed = {c.name: c.nominal - (c.lending_limit if c.lending_limit is not None else c.nominal) for c in members}
        lendable = sum(c.lending_limit if c.lending_limit is not None else c.nominal for c in members)
        borrowed = sum(max(0, self.usage(c.name, without) - guaranteed[c.name]) for c in members)
        local = max(0, guaranteed[cq.name] - use)
        cap = math.inf if cq.borrowing_limit is None else cq.nominal + cq.borrowing_limit - use
        return int(min(local + lendable - borrowed, cap))

    def max_capacity(self, cq: ClusterQueue) -> int:
        """The most this ClusterQueue could ever hold (nominal + what it may borrow), usage ignored."""
        if not cq.cohort:
            return cq.nominal
        others = sum(c.lending_limit if c.lending_limit is not None else c.nominal
                     for c in self.cohort_members(cq) if c.name != cq.name)
        borrow = others if cq.borrowing_limit is None else min(cq.borrowing_limit, others)
        return cq.nominal + borrow

    def settle(self) -> None:
        """Kueue scheduling passes until nothing else can be admitted."""
        for w in self.workloads.values():
            if not w.queue and w.state == "unschedulable":
                self._schedule_plain(w)
        while True:
            pending = [w for w in self.workloads.values() if w.queue and w.state in ("pending", "preempted")]
            # workloads that fit within their queue's nominal quota go first, then priority, then age
            pending.sort(key=lambda w: (not self._within_nominal(w), -w.priority, w.seq))
            if not any(self._try_admit(w) for w in pending):
                return

    def _within_nominal(self, w: Workload) -> bool:
        cq = self.cfg.cqs.get(self.cq_of(w) or "")
        return bool(cq) and self.usage(cq.name) + w.gpus <= cq.nominal

    def _quota_message(self, w: Workload, text: str) -> str:
        """The same reason on every pod set (a flavor mismatch or a TAS failure)."""
        names = [ps.name for ps in w.podsets]
        return "; ".join(f"couldn't assign flavors to pod set {n}: {text}" for n in names)

    def _unused_quota_message(self, w: Workload, cq: ClusterQueue, avail: int) -> str:
        """Kueue's flavor assigner (v0.19 ``assignFlavors``/``fitsResourceQuota``): pod sets are
        assigned in order, pod sets that share a ``podset-group-name`` together (their requests
        summed, one status for all members); each is checked as *assumed usage so far + its
        request* against what is available, and ``Assignment.Message()`` lists only the pod sets
        that do not fit. Assignment stops at the first group that cannot fit."""
        groups: dict[str, list[PodSet]] = {}
        for ps in w.podsets:
            groups.setdefault(ps.group or f"#{ps.name}", []).append(ps)
        assumed, parts = 0, []
        for members in groups.values():
            val = assumed + sum(p.count * p.gpus for p in members)
            if val > avail:
                text = f"insufficient unused quota for {m.GPU} in flavor {cq.flavor}, {val - avail} more needed"
                parts += [f"couldn't assign flavors to pod set {p.name}: {text}" for p in members]
                break
            assumed = val
        return "; ".join(parts)

    def _try_admit(self, w: Workload) -> bool:
        cq = self.cfg.cqs.get(self.cq_of(w) or "")
        if cq is None:
            w.message = f"LocalQueue {w.namespace}/{w.queue} doesn't exist"
            return False
        flavor = self.cfg.flavors[cq.flavor]
        for ps in w.podsets:   # the pod's own selector must agree with the flavor's node labels
            for k, v in ps.selector.items():
                if k in (flavor.get("nodeLabels") or {}) and flavor["nodeLabels"][k] != v:
                    w.message = self._quota_message(w, f"flavor {cq.flavor} doesn't match node affinity")
                    return False
        maxcap, assumed = self.max_capacity(cq), 0
        groups: dict[str, list[PodSet]] = {}      # same grouping as _unused_quota_message
        for ps in w.podsets:
            groups.setdefault(ps.group or f"#{ps.name}", []).append(ps)
        for members in groups.values():
            req = sum(p.count * p.gpus for p in members)
            if assumed + req > maxcap:
                text = (f"insufficient quota for {m.GPU} in flavor {cq.flavor}, previously considered podsets "
                        f"requests ({assumed}) + current podset request ({req}) > maximum capacity ({maxcap})")
                w.message = "; ".join(f"couldn't assign flavors to pod set {p.name}: {text}" for p in members)
                return False
            assumed += req
        victims: list[Workload] = []
        avail = self.available(cq)
        if w.gpus > avail:
            quota_msg = self._unused_quota_message(w, cq, avail)
            victims = self._find_victims(w, cq) if w.gpus <= cq.nominal else []
            if not victims:
                w.message = quota_msg + self._preempted_suffix(w)
                return False
        saved = [(v, dict(v.placement), dict(v.gpu_use)) for v in victims]
        for v in victims:   # evict first so topology placement sees the freed capacity
            self._unbind(v)
        placement, gpu_use, tas_msg = self._place_tas(w, cq, flavor)
        if placement is None:
            for v, pl, gu in saved:   # nothing gets preempted if the preemptor cannot be placed
                self._bind(v, pl, gu)
            w.message = self._quota_message(w, tas_msg) + self._preempted_suffix(w)
            return False
        self.clock += 1
        for v in victims:
            reason = "InClusterQueue" if self.cq_of(v) == cq.name else "InCohortReclamation"
            v.state, v.preempted, v.admitted_at = "preempted", reason, -1
            v.message = f"Preempted ({reason}) to admit {w.key}"
            self.log.append(f"{v.key}: preempted ({reason}) by {w.key}")
        self._bind(w, placement, gpu_use)
        w.state, w.admitted_at, w.message = "admitted", self.clock, ""
        self.log.append(f"{w.key}: admitted -> {self.hosts(w)}")
        return True

    def _preempted_suffix(self, w: Workload) -> str:
        return f" (evicted earlier: Preempted, {w.preempted})" if w.preempted else ""

    # -- classic preemption --------------------------------------------------------------------------
    def _find_victims(self, w: Workload, cq: ClusterQueue) -> list[Workload]:
        admitted = [v for v in self.workloads.values() if v.state == "admitted" and v.queue]
        cands = []
        for v in admitted:
            vcq = self.cfg.cqs[self.cq_of(v)]
            if vcq.name == cq.name:
                if cq.within == "LowerPriority" and v.priority < w.priority:
                    cands.append(v)
                elif cq.within == "LowerOrNewerEqualPriority" and (v.priority < w.priority or
                                                                   (v.priority == w.priority and v.seq > w.seq)):
                    cands.append(v)
            elif cq.cohort and vcq.cohort == cq.cohort and self.usage(vcq.name) > vcq.nominal \
                    and self.usage(cq.name) + w.gpus <= cq.nominal:   # reclaim only what is ours
                if cq.reclaim == "Any" or (cq.reclaim == "LowerPriority" and v.priority < w.priority):
                    cands.append(v)
        if not cands:
            return []
        # Kueue's CandidatesOrdering (v0.19 preemption/common/ordering.go): workloads already being
        # evicted first, then other queues, then (admission fair sharing only) lower LocalQueue
        # usage, then lowest priority, then newest admission. Eviction is instant in this model,
        # so the first criterion never separates candidates here, and fair sharing is off.
        cands.sort(key=lambda v: (not v.evicting, self.cq_of(v) == cq.name, v.priority, -v.admitted_at))
        if all(self.cq_of(v) == cq.name for v in cands):
            targets = self._greedy(w, cq, cands, allow_borrow=True)
        elif self.usage(cq.name) < cq.nominal:
            targets = self._greedy(w, cq, cands, allow_borrow=False)
        else:
            targets = self._greedy(w, cq, [v for v in cands if self.cq_of(v) == cq.name], allow_borrow=True)
        if not targets:
            return []
        for v in reversed(list(targets)):   # minimise: give back any target we did not need
            trial = [t for t in targets if t is not v]
            if self._fits(w, cq, trial, allow_borrow=True):
                targets = trial
        return targets

    def _fits(self, w: Workload, cq: ClusterQueue, removed: list[Workload], allow_borrow: bool) -> bool:
        if not allow_borrow and self.usage(cq.name, removed) + w.gpus > cq.nominal:
            return False
        return w.gpus <= self.available(cq, removed)

    def _greedy(self, w: Workload, cq: ClusterQueue, cands: list[Workload], allow_borrow: bool) -> list[Workload]:
        removed: list[Workload] = []
        for v in cands:
            vcq = self.cfg.cqs[self.cq_of(v)]
            if vcq.name != cq.name and self.usage(vcq.name, removed) <= vcq.nominal:
                continue   # that queue is no longer borrowing: nothing left to reclaim there
            removed.append(v)
            if self._fits(w, cq, removed, allow_borrow):
                return removed
        return []

    # -- Topology-Aware Scheduling -------------------------------------------------------------------
    def _place_tas(self, w: Workload, cq: ClusterQueue,
                   flavor: dict) -> tuple[dict[str, int] | None, dict[str, int], str]:
        topo_name = flavor.get("topologyName")
        if not topo_name:   # no TAS: pods go to the kube-scheduler; model as unconstrained first-fit
            topo_name, levels = "", [m.HOSTNAME]
        else:
            levels = self.cfg.topologies[topo_name]
        groups: dict[str, list[PodSet]] = {}
        for ps in w.podsets:
            groups.setdefault(ps.group or ps.name, []).append(ps)
        total: Counter = Counter()
        taken: Counter = Counter()        # GPUs claimed by earlier pod-set groups of this workload
        for pss in groups.values():
            gpus = pss[0].gpus
            if any(p.gpus != gpus for p in pss):
                raise NotImplementedError("pod sets grouped for TAS must request the same GPUs per pod here")
            count = sum(p.count for p in pss)
            ps = pss[-1]
            tolerations = list(ps.tolerations) + list(flavor.get("tolerations") or [])
            selector = {**(flavor.get("nodeLabels") or {}), **ps.selector}
            leaves = [n for n in self.nodes if selector_matches(n, flavor.get("nodeLabels") or {})
                      and all(lv in n.labels for lv in levels)]
            cap: dict[str, int] = {}
            excluded = Counter()
            for n in leaves:
                if not selector_matches(n, selector):
                    excluded["nodeSelector"] += 1
                    cap[n.name] = 0
                elif untolerated(n, tolerations) is not None:
                    t = untolerated(n, tolerations)
                    excluded[f'taint "{t["key"]}={t.get("value", "")}:{t["effect"]}"'] += 1
                    cap[n.name] = 0
                else:
                    free = n.free - taken[n.name]
                    cap[n.name] = free // gpus if gpus else 10 ** 6
                    if cap[n.name] == 0:
                        excluded[f'resource "{m.GPU}"'] += 1
            mode, level = ps.mode, ps.level
            if mode is None or mode == "unconstrained":
                mode, level = "unconstrained", levels[-1]
            result = tas_assign(leaves, levels, cap, count, mode, level)
            if isinstance(result, tuple):
                fit_count = result[1]
                verb = (f"allows to fit only {fit_count} out of {count} pod(s)" if fit_count
                        else f"doesn't allow to fit any of {count} pod(s)")
                msg = f'topology "{topo_name}" {verb}'
                if excluded:
                    msg += f". Total nodes: {len(leaves)}; excluded: " + ", ".join(
                        sorted(f"{k}: {v}" for k, v in excluded.items()))
                return None, {}, msg
            for node, pods in result.items():
                taken[node] += pods * gpus
                total[node] += pods
        return dict(total), dict(taken), ""

    # -- reading results -----------------------------------------------------------------------------
    def hosts(self, w: Workload) -> dict[str, int]:
        out: Counter = Counter()
        for node, pods in w.placement.items():
            out[self.node(node).host] += pods
        return dict(sorted(out.items()))

    def outcome(self) -> dict[str, dict]:
        res = {}
        for key, w in self.workloads.items():
            state = w.state
            d: dict = {"state": state}
            if state in ("admitted", "running"):
                d["hosts"] = "any" if w.any_node else self.hosts(w)
            if w.message:
                d["message"] = w.message
            res[key] = d
        return res


def fit_error(num_nodes: int, reasons: Counter, unresolvable: dict[str, bool]) -> str:
    """The kube-scheduler's FailedScheduling message (framework ``FitError.Error()`` format)."""
    msg = f"0/{num_nodes} nodes are available: " + ", ".join(sorted(f"{c} {r}" for r, c in reasons.items())) + "."
    not_helpful = sum(1 for v in unresolvable.values() if v)
    no_victims = len(unresolvable) - not_helpful   # this lab's pods all have priority 0: nothing to evict
    parts = []
    if not_helpful:
        parts.append(f"{not_helpful} Preemption is not helpful for scheduling")
    if no_victims:
        parts.append(f"{no_victims} No preemption victims found for incoming pod")
    return msg + f" preemption: 0/{num_nodes} nodes are available: " + ", ".join(sorted(parts)) + "."


# ---- TAS placement (Kueue's BestFit / LeastFreeCapacity), exposed for the notebooks ----------------
def _domains(leaves: list[Node], levels: list[str], idx: int, cap: dict[str, int]) -> dict[tuple, int]:
    out: Counter = Counter()
    for n in leaves:
        out[tuple(n.labels[lv] for lv in levels[: idx + 1])] += cap.get(n.name, 0)
    return dict(out)


def best_fit(domains: list[tuple[tuple, int]], need: int) -> tuple | None:
    """First domain (in the given order) with the smallest capacity that still fits ``need``."""
    best = None
    for dom, c in domains:
        if c >= need and (best is None or c < best[1]):
            best = (dom, c)
    return best[0] if best else None


def _distribute(sorted_domains: list[tuple[tuple, int]], need: int, bestfit_last: bool) -> list[tuple[tuple, int]]:
    """Kueue's updateCountsToMinimum: fill domains in order; the last one by best fit."""
    out, remaining = [], need
    for i, (dom, c) in enumerate(sorted_domains):
        if remaining <= 0:
            break
        if bestfit_last and c >= remaining:
            dom = best_fit(sorted_domains[i:], remaining)
            out.append((dom, remaining))
            return out
        take = min(c, remaining)
        if take:
            out.append((dom, take))
        remaining -= take
    return out if remaining <= 0 else []


def tas_assign(leaves: list[Node], levels: list[str], cap: dict[str, int], count: int,
               mode: str, level: str) -> dict[str, int] | tuple[None, int]:
    """Assign ``count`` pods to nodes. Returns {node: pods} or (None, pods_that_fit_in_best_domain)."""
    idx = levels.index(level)
    unconstrained = mode == "unconstrained"

    def order(doms: dict[tuple, int]) -> list[tuple[tuple, int]]:
        if unconstrained:   # LeastFreeCapacity: smallest gaps first
            return sorted(doms.items(), key=lambda kv: (kv[1], kv[0]))
        return sorted(doms.items(), key=lambda kv: (-kv[1], kv[0]))   # BestFit: biggest first

    chosen: list[tuple[tuple, int]] = []
    search = idx
    while True:
        doms = order(_domains(leaves, levels, search, cap))
        if not doms:
            return None, 0
        if unconstrained:
            single = next(((d, c) for d, c in doms if c >= count), None)
            if single:
                chosen = [(single[0], count)]
                break
        elif doms[0][1] >= count:
            chosen = [(best_fit(doms, count), count)]
            break
        if mode == "required":
            return None, doms[0][1]
        if search > 0 and not unconstrained:
            search -= 1     # preferred: try one level up
            continue
        chosen = _distribute(doms, count, bestfit_last=not unconstrained)
        if not chosen:
            return None, sum(c for _, c in doms)
        break
    # descend level by level to the nodes, filling the fewest domains
    for lv in range(search + 1, len(levels)):
        prefixes = {d for d, _ in chosen}
        children = {d: c for d, c in _domains(leaves, levels, lv, cap).items() if d[:lv] in prefixes}
        need = sum(n for _, n in chosen)
        chosen = _distribute(order(children), need, bestfit_last=not unconstrained)
    by_path = {tuple(n.labels[lv] for lv in levels): n.name for n in leaves}
    return {by_path[d]: pods for d, pods in chosen if pods}


# ---- scenario driver ---------------------------------------------------------------------------------
def new_sim(sc: Scenario | None = None, nodes: list[Node] | None = None) -> Sim:
    if nodes is None:
        nodes = read_topology() + (kwok_nodes() if sc is not None and sc.kwok else [])
    objs = [o for f in sorted((KIND_DIR / "manifests").glob("*.yaml")) for o in m.load_all(f)]
    return Sim(nodes, KueueConfig.from_objects(objs))


def run_step(sim: Sim, sc: Scenario, i: int) -> None:
    st = sc.steps[i]
    if st.action == "apply":
        for o in m.load_all(KIND_DIR / "workloads" / st.file):
            sim.apply(o)
    elif st.action == "delete":
        for o in m.load_all(KIND_DIR / "workloads" / st.file):
            meta = o["metadata"]
            sim.delete(f"{o['kind']}/{meta.get('namespace', 'default')}/{meta['name']}")
    elif st.action == "scale":
        sim.scale(st.target, st.replicas)


def predict(key: str, *, upto: int | None = None, nodes: list[Node] | None = None) -> dict[str, dict]:
    """Outcome after ``upto`` steps (default: all) of a scenario, from the committed manifests."""
    sc = get_scenario(key)
    sim = new_sim(sc, nodes)
    for i in range(len(sc.steps) if upto is None else upto):
        run_step(sim, sc, i)
    return sim.outcome()


def predict_all_steps(key: str, nodes: list[Node] | None = None) -> list[dict[str, dict]]:
    sc = get_scenario(key)
    sim = new_sim(sc, nodes)
    out = []
    for i in range(len(sc.steps)):
        run_step(sim, sc, i)
        out.append(sim.outcome())
    return out


def check_expectation(outcome: dict[str, dict], expected: dict[str, dict]) -> list[str]:
    """Differences between an outcome (predicted or observed) and the answer key ([] = match)."""
    problems = []
    for key, exp in expected.items():
        got = outcome.get(key)
        if got is None:
            problems.append(f"{key}: missing")
            continue
        if got["state"] != exp["state"]:
            problems.append(f"{key}: state {got['state']!r}, expected {exp['state']!r} ({got.get('message', '')})")
            continue
        hosts = exp.get("hosts")
        if isinstance(hosts, dict) and got.get("hosts") != hosts:
            problems.append(f"{key}: hosts {got.get('hosts')}, expected {hosts}")
        elif isinstance(hosts, str) and hosts.startswith("same-as:"):
            other = outcome.get(hosts.split(":", 1)[1], {})
            if other.get("hosts") not in ("any", None) and got.get("hosts") != other.get("hosts") \
                    and got.get("hosts") != "any":
                problems.append(f"{key}: hosts {got.get('hosts')}, expected the same as {hosts[8:]}")
        says = exp.get("says")
        if says and says not in got.get("message", ""):
            problems.append(f"{key}: message {got.get('message', '')!r} lacks {says!r}")
    return problems


def describe(outcome: dict[str, dict]) -> str:
    lines = []
    for key, d in outcome.items():
        where = d.get("hosts", "")
        where = "" if where == "" else (" on any GPU node (scheduler's choice)" if where == "any" else f" on {where}")
        lines.append(f"{key:44} {d['state']:13}{where}")
        if d.get("message") and d["state"] not in ("admitted", "running"):
            lines.append(f"{'':46}{d['message']}")
    return "\n".join(lines)


__all__ = ["Node", "Sim", "KueueConfig", "read_topology", "kwok_nodes", "nodes_from_k8s", "predict",
           "predict_all_steps", "check_expectation", "describe", "tas_assign", "best_fit", "fit_error",
           "SCENARIOS", "FLAVOR"]
