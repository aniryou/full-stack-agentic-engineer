"""Why is my GPU pod Pending? Read the evidence in the order the control plane produced it.

The one idea: a GPU pod passes up to four gates, and each leaves a different trace.
  1. **Kueue** (if the job is queued): the Job is *suspended* and has no pods at all; the
     reason lives on the Workload (``QuotaReserved=False``: quota, topology, flavor) or in its
     admission checks (a ProvisioningRequest waiting for DWS capacity). Grouped pods (LWS, pod
     groups) exist but carry a ``kueue.x-k8s.io/admission`` scheduling gate.
  2. **kube-scheduler**: ``PodScheduled=False, reason=Unschedulable`` with a histogram message
     ``0/N nodes are available: 4 Insufficient nvidia.com/gpu, ...`` — one reason per node,
     the first filter that rejected it — plus what preemption could (not) do.
  3. **cluster autoscaler**: events on the pod — ``TriggeredScaleUp``, ``NotTriggerScaleUp``
     (max size, backoff), and on GKE ``FailedScaleUp`` (stockout, quota).
  4. **kubelet**: the pod is bound but not running — image pull, volume mount (GCS FUSE),
     device allocation, or probes killing a slow model load.
``diagnose()`` walks those gates and returns a ``Diagnosis`` naming the gate, the category and
the fix. Inputs are plain ``kubectl ... -o json`` documents, so it works on fixtures offline
(``fixtures/pending/*.json``) and on a live cluster (``diagnose_live``).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .manifests import GPU, gpu_count

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pending"
EXPECTED_TAINTS = ("node-role.kubernetes.io/control-plane", "node-role.kubernetes.io/master")


# ---- parsing the scheduler's message ------------------------------------------------------------
@dataclass
class FitError:
    total_nodes: int
    reasons: dict[str, int]              # filter reason -> number of nodes
    preemption: dict[str, int]           # preemption verdict -> number of nodes
    prefilter: str = ""                  # a PreFilter failure applies to every node

    def nodes_with(self, pattern: str) -> int:
        return sum(c for r, c in self.reasons.items() if re.search(pattern, r))


_HEAD = re.compile(r"^0/(\d+) nodes are available:\s*")


def _histogram(body: str) -> tuple[dict[str, int], str]:
    body = body.strip()
    if body.endswith("."):
        body = body[:-1]
    reasons: dict[str, int] = {}
    prefilter = []
    for item in [x.strip() for x in body.split(", ") if x.strip()]:
        mt = re.match(r"^(\d+) (.+)$", item)
        if mt:
            reasons[mt.group(2)] = reasons.get(mt.group(2), 0) + int(mt.group(1))
        else:
            prefilter.append(item)
    return reasons, ", ".join(prefilter)


def parse_fit_error(message: str) -> FitError:
    """Parse ``0/N nodes are available: <reasons>. preemption: 0/N nodes are available: <verdicts>.``"""
    message = message.strip()
    main, _, pre = message.partition(" preemption: ")
    mt = _HEAD.match(main)
    if not mt:
        raise ValueError(f"not a scheduler FitError message: {message[:80]!r}")
    reasons, prefilter = _histogram(main[mt.end():])
    preemption: dict[str, int] = {}
    if pre:
        pm = _HEAD.match(pre.strip())
        preemption, _ = _histogram(pre.strip()[pm.end():] if pm else pre)
    return FitError(int(mt.group(1)), reasons, preemption, prefilter)


# ---- classifying reasons ---------------------------------------------------------------------------
CATEGORIES = [
    # (regex on the reason, category, blocking?, what it means)
    (r"^Insufficient nvidia\.com/gpu$", "insufficient-gpu", True, "not enough free GPUs on the node"),
    (r"^Insufficient (cpu|memory|ephemeral-storage)$", "insufficient-host-resource", True,
     "the node's CPU/memory is used up (often by pods that need no GPU)"),
    (r"^Insufficient ", "insufficient-other", True, "not enough of an extended resource"),
    (r"^node\(s\) had untolerated taint \{nvidia\.com/gpu: ?", "gpu-taint", True,
     "GPU nodes are tainted and the pod has no matching toleration"),
    (r"^node\(s\) had untolerated taint \{node-role\.kubernetes\.io/(control-plane|master): ?", "control-plane", False,
     "control-plane nodes never run workloads: expected, ignore"),
    (r"^node\(s\) had untolerated taint \{cloud\.google\.com/gke-spot", "spot-taint", True,
     "Spot nodes are tainted; tolerate them to run on Spot"),
    (r"^node\(s\) had untolerated taint \{cloud\.google\.com/gke-queued", "queued-taint", True,
     "queued-provisioning nodes accept only the workload they were provisioned for"),
    (r"^node\(s\) had untolerated taint \{node\.kubernetes\.io/(not-ready|unreachable)", "node-not-ready", True,
     "nodes are not ready (still booting, or failing)"),
    (r"^node\(s\) had untolerated taint", "other-taint", True, "a taint the pod does not tolerate"),
    (r"^node\(s\) didn't match Pod's node affinity/selector$", "selector", True,
     "the pod's nodeSelector/affinity matches no (free) node"),
    (r"^node\(s\) were unschedulable$", "cordoned", True, "nodes are cordoned (maintenance, drain, upgrade)"),
    (r"topology spread constraints", "spread", True, "topologySpreadConstraints cannot be satisfied"),
    (r"anti-affinity|pod affinity", "affinity", True, "inter-pod (anti-)affinity cannot be satisfied"),
    (r"volume node affinity conflict|available persistent volumes", "volume", True,
     "the volume lives in another zone/node than the free GPUs"),
    (r"^Too many pods$", "max-pods", True, "the node's pod limit is reached"),
    (r"cannot allocate all claims|resourceclaim|device class", "dra", True,
     "no device satisfies the ResourceClaim (DRA): class, CEL selectors or count"),
]


def classify_reason(reason: str) -> tuple[str, bool, str]:
    """(category, blocking, meaning) for one scheduler filter reason."""
    for pattern, cat, blocking, meaning in CATEGORIES:
        if re.search(pattern, reason):
            return cat, blocking, meaning
    return "unknown", True, "see the scheduler plugin that emitted it"


# ---- cluster facts (optional) -----------------------------------------------------------------------
def free_gpus_per_node(nodes: list[dict], pods: list[dict]) -> dict[str, int]:
    """Allocatable GPUs minus GPUs requested by pods bound to the node (not finished)."""
    used: Counter = Counter()
    for p in pods:
        node = (p.get("spec") or {}).get("nodeName")
        if node and (p.get("status") or {}).get("phase") not in ("Succeeded", "Failed"):
            used[node] += sum(gpu_count(c) for c in p["spec"].get("containers") or [])
    out = {}
    for n in nodes:
        alloc = int(((n.get("status") or {}).get("allocatable") or {}).get(GPU, 0) or 0)
        if alloc:
            out[n["metadata"]["name"]] = alloc - used[n["metadata"]["name"]]
    return out


def is_fragmented(free_per_node: list[int] | dict[str, int], request: int) -> bool:
    """Enough GPUs in total, but no single node has ``request`` of them."""
    vals = list(free_per_node.values()) if isinstance(free_per_node, dict) else list(free_per_node)
    return sum(vals) >= request and (max(vals) if vals else 0) < request


def pod_gpus(pod: dict) -> int:
    return sum(gpu_count(c) for c in (pod.get("spec") or {}).get("containers") or [])


def _selector_hint(pod: dict, nodes: list[dict]) -> str:
    sel = (pod.get("spec") or {}).get("nodeSelector") or {}
    missing = []
    for k, v in sel.items():
        values = sorted({(n["metadata"].get("labels") or {}).get(k) for n in nodes} - {None})
        if v not in values:
            missing.append(f"{k}={v} matches no node (nodes have: {', '.join(values) or 'no such label'})")
    return "; ".join(missing)


# ---- the diagnosis ----------------------------------------------------------------------------------------
@dataclass
class Diagnosis:
    gate: str                       # kueue | kube-scheduler | cluster-autoscaler | kubelet | none
    category: str
    headline: str
    evidence: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"[{self.gate} / {self.category}] {self.headline}"]
        lines += [f"  evidence: {e}" for e in self.evidence]
        lines += [f"  fix:      {f}" for f in self.fixes]
        return "\n".join(lines)


def _cond(obj: dict | None, ctype: str) -> dict:
    for c in ((obj or {}).get("status") or {}).get("conditions") or []:
        if c.get("type") == ctype:
            return c
    return {}


def diagnose_workload(wl: dict) -> Diagnosis | None:
    """Kueue's view: why a queued job has not started (None if Kueue is not the blocker)."""
    name = wl["metadata"]["name"]
    admitted, reserved, preempted = _cond(wl, "Admitted"), _cond(wl, "QuotaReserved"), _cond(wl, "Preempted")
    if admitted.get("status") == "True":
        return None
    if preempted.get("status") == "True":
        why = {"InClusterQueue": "a higher-priority workload in the same ClusterQueue needed the quota",
               "InCohortReclamation": "the lending ClusterQueue took its nominal quota back (you were borrowing)",
               "InCohortFairSharing": "fair sharing rebalanced the cohort"}.get(preempted.get("reason", ""), "")
        return Diagnosis("kueue", "preempted", f"Workload {name} was preempted ({preempted.get('reason')}): {why}",
                         [preempted.get("message", "")[:240]],
                         ["it is requeued automatically; checkpoint so the restart is cheap",
                          "raise its WorkloadPriorityClass, or stop borrowing (run within nominal quota)"])
    if reserved.get("status") == "True":
        checks = (wl.get("status") or {}).get("admissionChecks") or []
        waiting = [c for c in checks if c.get("state") != "Ready"]
        ev = [f"admission check {c.get('name')}: {c.get('state')} - {c.get('message', '')}" for c in waiting]
        return Diagnosis("kueue", "admission-check",
                         f"Workload {name} has quota but waits on admission checks (e.g. a ProvisioningRequest for DWS capacity)",
                         ev, ["queued provisioning is all-or-nothing: it waits until every node can be created together",
                              "check `kubectl get provisioningrequests -A` and the autoscaler's events",
                              "shrink the gang, allow another zone/machine, or use a reservation for a deadline"])
    msg = reserved.get("message", "")
    ev = [f"QuotaReserved=False ({reserved.get('reason', '')}): {msg}"] if msg else []
    if "maximum capacity" in msg:
        return Diagnosis("kueue", "exceeds-max-quota",
                         "the job asks for more than this ClusterQueue can ever hold (nominal + borrowing limit): it will never start",
                         ev, ["split it or lower its parallelism", "raise nominalQuota/borrowingLimit",
                              "submit to a LocalQueue whose ClusterQueue is big enough"])
    if "insufficient unused quota" in msg:
        return Diagnosis("kueue", "waiting-for-quota",
                         "the queue's quota (plus what it may borrow) is in use: the job waits its turn",
                         ev, ["wait: it is admitted when running work finishes or the cohort frees quota",
                              "a higher WorkloadPriorityClass preempts lower-priority work in the same queue "
                              "(withinClusterQueue: LowerPriority)",
                              "raise borrowingLimit if the cohort often has idle quota"])
    if "topology" in msg and "fit" in msg:
        return Diagnosis("kueue", "topology",
                         "quota is available, but no single topology domain at the required level has room for the whole gang",
                         ev, ["wait for a domain to drain (Kueue requeues when capacity frees)",
                              "use podset-preferred-topology instead of required if spread is tolerable",
                              "smaller gangs, or consolidate small jobs so whole hosts stay free"])
    if "doesn't match node affinity" in msg or "untolerated taint" in msg:
        return Diagnosis("kueue", "flavor-mismatch", "no ResourceFlavor matches the pod's nodeSelector/tolerations",
                         ev, ["align the pod's nodeSelector with a flavor's nodeLabels, or drop it and let Kueue choose"])
    if "unavailable in ClusterQueue" in msg:
        return Diagnosis("kueue", "uncovered-resource",
                         "the pod requests a resource the ClusterQueue does not cover (e.g. ephemeral-storage)",
                         ev, ["add it to coveredResources with a quota, or stop requesting it"])
    if "LocalQueue" in msg or "ClusterQueue" in msg:
        return Diagnosis("kueue", "queue-misconfigured", "the LocalQueue/ClusterQueue is missing or inactive", ev,
                         ["check `kubectl get localqueue,clusterqueue -A` (Active condition)"])
    return Diagnosis("kueue", "pending", f"Workload {name} is not admitted yet", ev,
                     ["kubectl describe workload -n <ns> <name>"])


def _autoscaler(events: list[dict]) -> Diagnosis | None:
    by_reason: dict[str, dict] = {}
    for e in events:
        by_reason[e.get("reason", "")] = e
    fs = by_reason.get("FailedScaleUp")
    if fs:
        msg = fs.get("message", "")
        if "out of resources" in msg or "STOCKOUT" in msg:
            return Diagnosis("cluster-autoscaler", "stockout",
                             "the autoscaler asked for a GPU VM and the zone had none (stockout)", [msg],
                             ["allow more zones/regions or machine shapes (ComputeClass priorities)",
                              "fall back from Spot to on-demand, or queue for capacity with DWS flex-start",
                              "a reservation is the only guarantee"])
        if "quota" in msg.lower():
            return Diagnosis("cluster-autoscaler", "cloud-quota",
                             "the project's GPU quota is exhausted (e.g. GPUS_ALL_REGIONS or a per-model regional quota)",
                             [msg], ["request a quota increase (GPU quotas often start at 0)", "use a smaller shape"])
        return Diagnosis("cluster-autoscaler", "scale-up-failed", "a scale-up attempt failed", [msg],
                         ["kubectl get events -A --field-selector reason=FailedScaleUp"])
    nt = by_reason.get("NotTriggerScaleUp")
    if nt:
        msg = nt.get("message", "")
        if "max node group size reached" in msg:
            return Diagnosis("cluster-autoscaler", "max-size",
                             "every node pool that could run it is already at its maximum size", [msg],
                             ["raise the pool's max nodes (e.g. gpu_max_nodes)", "or free capacity / queue the job"])
        if "backoff" in msg:
            return Diagnosis("cluster-autoscaler", "backoff",
                             "the matching node pool is in backoff after a failed scale-up (stockout or quota)", [msg],
                             ["look for FailedScaleUp events; add fallbacks (zones, shapes, flex-start)"])
        return Diagnosis("cluster-autoscaler", "no-matching-pool",
                         "no node pool template could run this pod even if scaled up", [msg],
                         ["the reasons listed are per node group: fix the selector/taint/size mismatch"])
    tu = by_reason.get("TriggeredScaleUp")
    if tu:
        return Diagnosis("cluster-autoscaler", "scaling-up",
                         "a GPU node is being created for it: wait (VM boot + driver install take minutes)",
                         [tu.get("message", "")], ["nothing to fix; budget for cold start or keep a warm node"])
    return None


def _kubelet(pod: dict, events: list[dict]) -> Diagnosis | None:
    status = pod.get("status") or {}
    if status.get("reason") == "UnexpectedAdmissionError" or str(status.get("reason", "")).startswith("OutOf"):
        return Diagnosis("kubelet", "device-allocation",
                         "the kubelet rejected the pod: the device plugin could not allocate the GPUs the scheduler counted",
                         [status.get("message", "")],
                         ["check the device plugin pod and `nvidia-smi` on the node (unhealthy/XID GPUs are withdrawn)",
                          "delete the failed pod; its controller recreates it"])
    for e in events:
        if e.get("reason") == "FailedMount":
            msg = e.get("message", "")
            hint = ("the pod's KSA lacks storage access: grant roles/storage.objectViewer to its Workload Identity principal"
                    if "403" in msg or "PermissionDenied" in msg or "denied" in msg else "check the volume definition")
            return Diagnosis("kubelet", "volume-mount", "the pod is bound but its volume cannot be mounted", [msg[:300]], [hint])
    for cs in status.get("containerStatuses") or []:
        waiting = (cs.get("state") or {}).get("waiting") or {}
        if waiting.get("reason") in ("ErrImagePull", "ImagePullBackOff"):
            return Diagnosis("kubelet", "image-pull", f"image cannot be pulled: {waiting.get('message', '')[:200]}",
                             [], ["check the image name/tag and registry access (Artifact Registry permissions)"])
    kills = [e for e in events if e.get("reason") in ("Killing",) and "liveness" in e.get("message", "")]
    unhealthy = [e for e in events if e.get("reason") == "Unhealthy"]
    restarts = sum(int(cs.get("restartCount", 0)) for cs in status.get("containerStatuses") or [])
    if kills or (unhealthy and restarts):
        has_startup = any("startupProbe" in c for c in (pod.get("spec") or {}).get("containers") or [])
        return Diagnosis("kubelet", "probe-kills-startup",
                         f"the container keeps being killed by its liveness probe while loading ({restarts} restarts)",
                         [e.get("message", "") for e in (kills or unhealthy)[:2]],
                         ["add a startupProbe sized to the weight-load time (liveness waits until it succeeds)"
                          if not has_startup else "raise the startupProbe failureThreshold x periodSeconds"])
    if (pod.get("spec") or {}).get("nodeName") and status.get("phase") == "Pending":
        return Diagnosis("kubelet", "starting", "bound to a node and starting (image pull / volume setup)",
                         [f"node {pod['spec']['nodeName']}"], ["kubectl describe pod: watch the events"])
    return None


def diagnose(bundle: dict) -> Diagnosis:
    """Diagnose from ``{"pod", "events", "nodes", "pods", "workload"}`` (any may be missing)."""
    pod, events = bundle.get("pod"), bundle.get("events") or []
    nodes, pods, wl = bundle.get("nodes") or [], bundle.get("pods") or [], bundle.get("workload")
    # gate 1: Kueue
    if pod is None:
        if wl is None:
            return Diagnosis("none", "no-pod", "no pod and no Kueue workload: is the Job suspended or never created?",
                             [], ["kubectl get job -o yaml (spec.suspend) and kubectl get workloads -n <ns>"])
        d = diagnose_workload(wl)
        return d or Diagnosis("kueue", "admitted", "admitted: pods should appear shortly", [], [])
    gates = [g.get("name", "") for g in (pod.get("spec") or {}).get("schedulingGates") or []]
    if gates:
        if "kueue.x-k8s.io/admission" in gates:
            d = diagnose_workload(wl) if wl else None
            base = "the pod exists but is gated until Kueue admits its group (all-or-nothing)"
            if d:
                d.headline = f"{base}; {d.headline}"
                return d
            return Diagnosis("kueue", "gated", base, [f"schedulingGates: {gates}"],
                             ["kubectl get workloads -n <ns>: find why the group is not admitted"])
        if "kueue.x-k8s.io/topology" in gates:
            return Diagnosis("kueue", "topology-gate", "admitted; Kueue is ungating pods onto their assigned domain",
                             [f"schedulingGates: {gates}"], ["transient: wait a few seconds"])
        return Diagnosis("controller", "gated", f"pod is scheduling-gated by {gates}", [],
                         ["the controller that added the gate decides when it is removed"])
    # gate 4 first if already bound: the scheduler is done with it
    if (pod.get("spec") or {}).get("nodeName") or (pod.get("status") or {}).get("phase") == "Failed":
        d = _kubelet(pod, events)
        return d or Diagnosis("none", "running", "the pod is bound and not blocked", [], [])
    # gate 2: the scheduler
    cond = _cond(pod, "PodScheduled")
    msg = cond.get("message", "") if cond.get("reason") == "Unschedulable" else ""
    if not msg:
        msg = next((e.get("message", "") for e in reversed(events) if e.get("reason") == "FailedScheduling"), "")
    auto = _autoscaler(events)
    if not msg:
        return auto or Diagnosis("kube-scheduler", "not-yet-scheduled", "no scheduling verdict yet", [], ["wait / describe"])
    fe = parse_fit_error(msg)
    d = _explain_fit_error(fe, pod, nodes, pods)
    d.evidence.insert(0, f"FailedScheduling: {msg}")
    if auto:   # the autoscaler's verdict decides whether waiting can help
        d.evidence.append(f"autoscaler: {auto.headline}")
        d.fixes = auto.fixes + d.fixes
        if auto.category in ("stockout", "cloud-quota", "max-size", "backoff", "scaling-up"):
            d.gate, d.category, d.headline = auto.gate, auto.category, auto.headline + f" (scheduler: {d.headline})"
    return d


def _explain_fit_error(fe: FitError, pod: dict, nodes: list[dict], pods: list[dict]) -> Diagnosis:
    if fe.prefilter:
        cat, _, meaning = classify_reason(fe.prefilter)
        return Diagnosis("kube-scheduler", cat, f"rejected before filtering any node: {fe.prefilter} ({meaning})", [],
                         ["for DRA: check the DeviceClass exists and the claim's CEL selectors match a ResourceSlice"
                          if cat == "dra" else "fix the pod-level problem named above"])
    blocking = [(r, c, *classify_reason(r)) for r, c in fe.reasons.items()]
    relevant = [b for b in blocking if b[3]]           # drop expected noise (control-plane taint)
    relevant.sort(key=lambda b: -b[1])
    ev = [f"{c} node(s): {r}  -> {meaning}" for r, c, cat, blk, meaning in blocking]
    fixes: list[str] = []
    cats = {b[2] for b in relevant}
    request = pod_gpus(pod)
    if "gpu-taint" in cats:
        head, cat = "GPU nodes are tainted nvidia.com/gpu and the pod does not tolerate it", "gpu-taint"
        fixes.append("tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}] "
                     "(GKE adds it for GPU requests via ExtendedResourceToleration; kind and many clusters do not)")
    elif "insufficient-gpu" in cats:
        cat = "insufficient-gpu"
        unresolvable = fe.preemption.get("Preemption is not helpful for scheduling", 0)
        free = free_gpus_per_node(nodes, pods) if nodes else {}
        biggest = max((int(((n.get("status") or {}).get("allocatable") or {}).get(GPU, 0) or 0) for n in nodes), default=0)
        if nodes and request > biggest:
            head, cat = f"no node has {request} GPUs (largest has {biggest}): waiting cannot help", "too-big"
            fixes += ["split the pod (tensor/pipeline parallel across pods: LWS/JobSet)",
                      "or add a node pool with a bigger GPU shape"]
        elif free and is_fragmented(free, request):
            head, cat = (f"{sum(free.values())} GPUs are free but no node has {request} of them: fragmentation "
                         f"(free per node: {sorted(free.values(), reverse=True)})"), "fragmentation"
            fixes += ["bin-pack small jobs (MostAllocated scoring, Kueue TAS unconstrained packing)",
                      "request whole-node shapes for big pods; drain/consolidate the stragglers"]
        elif not nodes and unresolvable >= fe.nodes_with(r"^Insufficient nvidia"):
            head, cat = "the request exceeds every GPU node's allocatable: waiting cannot help", "too-big"
            fixes += ["split the pod or add a bigger GPU shape"]
        else:
            head = "all GPU nodes are busy: wait, scale the pool up, or preempt lower-priority work"
            fixes += ["let the autoscaler add a node (check max size) or queue it with Kueue",
                      "give it a higher PriorityClass if lower-priority pods may be preempted"]
    elif "selector" in cats and (not relevant or relevant[0][2] == "selector"):
        cat = "selector"
        hint = _selector_hint(pod, nodes) if nodes else ""
        head = "the pod's nodeSelector/affinity matches no node" + (f": {hint}" if hint else "")
        fixes.append("fix the selector value (e.g. cloud.google.com/gke-accelerator) or add a pool with that accelerator")
    elif relevant:
        r, c, cat, blk, meaning = relevant[0]
        head = f"{c} node(s): {meaning}"
        fixes.append({"insufficient-host-resource": "taint GPU nodes so CPU-only pods stay off; right-size requests",
                      "cordoned": "wait for the upgrade/drain or uncordon",
                      "volume": "keep the PV and the GPU pool in the same zone (WaitForFirstConsumer storage class)",
                      "dra": "check ResourceSlices (`kubectl get resourceslices`) against the claim's selectors",
                      "node-not-ready": "check node health; GPU nodes may still be installing the driver",
                      }.get(cat, "see the reason above"))
    else:
        head, cat = "only expected rejections (control plane)", "unknown"
    return Diagnosis("kube-scheduler", cat, head, ev, fixes)


# ---- fixtures and live use ---------------------------------------------------------------------------
def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def diagnose_live(pod_name: str, namespace: str, kubectl=None) -> Diagnosis:
    """Collect pod, events, nodes, pods and (if any) the Kueue workload with kubectl, then diagnose."""
    from .kindlab import Kubectl
    k = kubectl or Kubectl(echo=lambda s: None)
    pod = json.loads(k.run("get", "pod", pod_name, "-n", namespace, "-o", "json", quiet=True))
    events = k.get_json("events", "-n", namespace, "--field-selector", f"involvedObject.name={pod_name}")["items"]
    bundle = {"pod": pod, "events": events, "nodes": k.get_json("nodes")["items"],
              "pods": k.get_json("pods", "-A")["items"]}
    wl_name = (pod["metadata"].get("annotations") or {}).get("kueue.x-k8s.io/workload")
    if wl_name:
        bundle["workload"] = json.loads(k.run("get", "workloads.kueue.x-k8s.io", wl_name, "-n", namespace,
                                              "-o", "json", quiet=True, check=False) or "null")
    return diagnose(bundle)
