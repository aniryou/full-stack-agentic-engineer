"""A linter for GPU pod specs: the mistakes that cost GPU-hours, caught before ``kubectl apply``.

The one idea: most GPU scheduling incidents are pod-spec bugs, not scheduler bugs. The API
server rejects some of them outright (a GPU request without a limit, a fractional GPU); worse,
a CRD such as a JobSet or LeaderWorkerSet *accepts* the same pod template and the error only
surfaces minutes later when its controller tries to create pods. Others are legal but
expensive: no toleration for the GPU taint (Pending forever on clusters that taint GPU
nodes), no startup probe (the kubelet kills the pod halfway through loading weights), a CPU
request that strands the node's other GPUs, a GPU on a sidecar.

Each rule is a small function; ``lint()`` walks every pod template in an object (see
``manifests.iter_pod_templates``) and returns ``Finding``s. Severity: ``error`` = the API
server or kubelet will reject or kill it; ``warning`` = it will schedule badly or waste GPUs;
``info`` = worth knowing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import machines
from .manifests import (GKE_ACCELERATOR, GKE_COMPUTE_CLASS, GKE_NODEPOOL, GPU, QUEUE_LABEL,
                        iter_pod_templates, load_all)

# Node labels that pin a pod to a GPU model or pool (GKE, GPU Feature Discovery, instance type).
ACCELERATOR_KEYS = (GKE_ACCELERATOR, GKE_NODEPOOL, GKE_COMPUTE_CLASS, "nvidia.com/gpu.product",
                    "node.kubernetes.io/instance-type", "cloud.google.com/gke-nodepool")
SERVING_KINDS = ("Deployment", "StatefulSet", "LeaderWorkerSet")


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str        # "error" | "warning" | "info"
    where: str           # e.g. "Job/train spec.template.spec.containers[0]"
    message: str
    fix: str = ""

    def __str__(self) -> str:
        s = f"{self.severity.upper():7} {self.rule:26} {self.where}: {self.message}"
        return s + (f"\n        fix: {self.fix}" if self.fix else "")


@dataclass
class Context:
    """Everything a rule may look at for one pod template."""
    obj: dict
    path: str
    template: dict
    machine: machines.Machine | None = None
    expected_load_s: float | None = None

    @property
    def kind(self) -> str:
        return self.obj.get("kind", "?")

    @property
    def name(self) -> str:
        return (self.obj.get("metadata") or {}).get("name", "?")

    @property
    def spec(self) -> dict:
        return self.template.get("spec") or {}

    def where(self, suffix: str = "") -> str:
        return f"{self.kind}/{self.name} {self.path}.spec" + (f".{suffix}" if suffix else "")

    def containers(self) -> Iterator[tuple[str, dict]]:
        for i, c in enumerate(self.spec.get("containers") or []):
            yield f"containers[{i}]", c

    def init_containers(self) -> Iterator[tuple[str, dict]]:
        for i, c in enumerate(self.spec.get("initContainers") or []):
            yield f"initContainers[{i}]", c

    def gpu_containers(self) -> list[tuple[str, dict]]:
        return [(p, c) for p, c in self.containers() if _gpu_quantity(c) != 0]

    @property
    def pod_gpus(self) -> int:
        return sum(_as_int(_gpu_quantity(c)) for _, c in self.containers())

    @property
    def kueue_managed(self) -> bool:
        return QUEUE_LABEL in ((self.obj.get("metadata") or {}).get("labels") or {})

    @property
    def replicas_in_gang(self) -> int:
        """Pods that must talk to each other over NCCL (multi-pod gangs)."""
        spec = self.obj.get("spec") or {}
        if self.kind == "LeaderWorkerSet":
            return int((spec.get("leaderWorkerTemplate") or {}).get("size", 1))
        if self.kind == "JobSet":
            return sum(int(rj.get("replicas", 1)) * int(((rj.get("template") or {}).get("spec") or {}).get("parallelism", 1))
                       for rj in spec.get("replicatedJobs", []))
        if self.kind == "Job":
            return int(spec.get("parallelism", 1))
        return 1


# ---- quantities -----------------------------------------------------------------------------
def _gpu_quantity(c: dict, which: str | None = None):
    res = c.get("resources") or {}
    if which:
        return (res.get(which) or {}).get(GPU)
    lim = (res.get("limits") or {}).get(GPU)
    return lim if lim is not None else ((res.get("requests") or {}).get(GPU) or 0)


def parse_quantity(q) -> Fraction:
    """Exact value of a Kubernetes quantity for integer checks ('2' -> 2, '500m' -> 1/2)."""
    s = str(q).strip()
    suffixes = {"m": Fraction(1, 1000), "k": 1000, "M": 10 ** 6, "G": 10 ** 9,
                "Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3}
    for suf in sorted(suffixes, key=len, reverse=True):
        if s.endswith(suf) and s[: -len(suf)].replace(".", "", 1).isdigit():
            return Fraction(s[: -len(suf)]) * suffixes[suf]
    return Fraction(s)


def _as_int(q) -> int:
    try:
        return int(parse_quantity(q))
    except (ValueError, ZeroDivisionError):
        return 0


# ---- rules ------------------------------------------------------------------------------------
Rule = Callable[[Context], Iterable[Finding]]
RULES: dict[str, Rule] = {}


def rule(fn: Rule) -> Rule:
    RULES[fn.__name__.replace("_", "-")] = fn
    return fn


def check_gpu_resources(resources: dict) -> list[str]:
    """What the API server says about one container's GPU resources ([] = accepted).

    Extended resources are integers, cannot be overcommitted, and so need a limit; a request,
    if given, must equal the limit. (kube-apiserver ``validateResourceRequirements``.)
    """
    req = (resources.get("requests") or {}).get(GPU)
    lim = (resources.get("limits") or {}).get(GPU)
    problems = []
    for label, q in (("requests", req), ("limits", lim)):
        if q is not None and parse_quantity(q).denominator != 1:
            problems.append(f"{label}[{GPU}]: {q!r} must be an integer")
    if req is not None and lim is None:
        problems.append(f"limits: Limit must be set for non overcommitable resources ({GPU})")
    elif req is not None and parse_quantity(req) != parse_quantity(lim):
        problems.append(f"requests: {req} must be equal to {GPU} limit of {lim}")
    return problems


@rule
def gpu_request_equals_limit(ctx: Context):
    for containers in (ctx.containers(), ctx.init_containers()):
        for p, c in containers:
            for problem in check_gpu_resources(c.get("resources") or {}):
                yield Finding("gpu-request-equals-limit", "error", ctx.where(p), problem,
                              f"set only `limits: {{{GPU}: N}}` (the request is copied from it) with a whole N")


BATCH_KINDS = ("Job", "JobSet", "CronJob")
LONG_RUNNING_KINDS = ("Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "LeaderWorkerSet")


@rule
def restart_policy(ctx: Context):
    """Job pods must be Never/OnFailure; Deployment/StatefulSet (and so LWS) pods must be Always.
    The API server enforces this for built-in kinds, but a CRD's embedded template is only
    checked when its controller creates the pods — i.e. after you walked away."""
    rp = ctx.spec.get("restartPolicy")
    if ctx.kind in BATCH_KINDS and rp in (None, "Always"):
        yield Finding("restart-policy", "error", ctx.where("restartPolicy"),
                      f"{ctx.kind} pods need restartPolicy Never or OnFailure (got {rp or 'unset = Always'})",
                      "restartPolicy: Never (let the Job/JobSet failure policy handle retries)")
    elif ctx.kind in LONG_RUNNING_KINDS and rp not in (None, "Always"):
        owner = "StatefulSets" if ctx.kind == "LeaderWorkerSet" else f"{ctx.kind}s"
        yield Finding("restart-policy", "error", ctx.where("restartPolicy"),
                      f"{owner} only accept restartPolicy Always (got {rp}); the pods will never be created",
                      "remove restartPolicy (defaults to Always)")


@rule
def gpu_on_sidecar(ctx: Context):
    for p, c in ctx.init_containers():
        if _gpu_quantity(c):
            native = c.get("restartPolicy") == "Always"
            yield Finding("gpu-on-sidecar", "error" if native else "info", ctx.where(p),
                          ("a native sidecar holds its GPUs for the life of the pod, on top of the app containers"
                           if native else "an init container's GPUs are reused by the app containers afterwards"),
                          "request GPUs only on the container that runs the model")
    gpu_cs = ctx.gpu_containers()
    if len(gpu_cs) > 1:
        yield Finding("gpu-on-sidecar", "warning", ctx.where(),
                      f"{len(gpu_cs)} containers request GPUs; each gets its own devices and they cannot share them",
                      "put GPUs on one container; proxies, log shippers and loaders need none")


@rule
def gpu_toleration(ctx: Context):
    if not ctx.pod_gpus:
        return
    for t in ctx.spec.get("tolerations") or []:
        if (t.get("key") == GPU or (not t.get("key") and t.get("operator") == "Exists")) \
                and t.get("effect") in (None, "", "NoSchedule"):
            return
    yield Finding("gpu-toleration", "warning", ctx.where("tolerations"),
                  f"no toleration for the `{GPU}` taint: Pending on clusters that taint GPU nodes "
                  "unless an admission plugin (ExtendedResourceToleration) or a Kueue ResourceFlavor adds it",
                  f"tolerations: [{{key: {GPU}, operator: Exists, effect: NoSchedule}}]")


def _pins_accelerator(spec: dict) -> bool:
    if any(k in ACCELERATOR_KEYS for k in (spec.get("nodeSelector") or {})):
        return True
    na = ((spec.get("affinity") or {}).get("nodeAffinity") or {}).get("requiredDuringSchedulingIgnoredDuringExecution") or {}
    for term in na.get("nodeSelectorTerms") or []:
        if any(e.get("key") in ACCELERATOR_KEYS for e in term.get("matchExpressions") or []):
            return True
    return bool(spec.get("resourceClaims"))   # a DRA claim selects devices by attribute instead


@rule
def gpu_node_selector(ctx: Context):
    if not ctx.pod_gpus or _pins_accelerator(ctx.spec):
        return
    if ctx.kueue_managed:
        yield Finding("gpu-node-selector", "info", ctx.where("nodeSelector"),
                      "no accelerator selector; Kueue adds the ResourceFlavor's nodeLabels at admission",
                      f"add `{GKE_ACCELERATOR}` anyway so the manifest means the same without Kueue")
    else:
        yield Finding("gpu-node-selector", "warning", ctx.where("nodeSelector"),
                      "no accelerator selector: in a mixed fleet this pod can land on any GPU model",
                      f"nodeSelector: {{{GKE_ACCELERATOR}: nvidia-l4}} (or nvidia.com/gpu.product with GFD)")


def probe_budget_s(probe: dict) -> int:
    """Seconds a probe tolerates failure before the kubelet gives up (restarts the container)."""
    return int(probe.get("initialDelaySeconds", 0)) + \
        int(probe.get("failureThreshold", 3)) * int(probe.get("periodSeconds", 10))


def _is_server(ctx: Context, c: dict) -> bool:
    return ctx.kind in SERVING_KINDS or bool(c.get("ports")) or "readinessProbe" in c


@rule
def startup_probe(ctx: Context):
    for p, c in ctx.gpu_containers():
        if not _is_server(ctx, c):
            continue
        sp, lp, rp = c.get("startupProbe"), c.get("livenessProbe"), c.get("readinessProbe")
        if sp and ctx.expected_load_s and probe_budget_s(sp) < ctx.expected_load_s:
            yield Finding("startup-probe", "error", ctx.where(p + ".startupProbe"),
                          f"startup budget {probe_budget_s(sp)} s < expected model load {ctx.expected_load_s:.0f} s: "
                          "the kubelet restarts the container before it is ready, forever",
                          f"failureThreshold >= {math.ceil(ctx.expected_load_s * 1.5 / int(sp.get('periodSeconds', 10)))}")
        elif not sp and lp:
            budget = probe_budget_s(lp)
            yield Finding("startup-probe", "warning", ctx.where(p),
                          f"liveness probe with no startup probe: the container is killed after ~{budget} s "
                          "of loading weights",
                          "add a startupProbe (e.g. /health every 10 s, failureThreshold sized to the load time)")
        elif not sp and not rp:
            yield Finding("startup-probe", "info", ctx.where(p),
                          "no startup or readiness probe: traffic can arrive before the model is loaded",
                          "add readiness + startup probes on the server's health endpoint")


@rule
def cpu_memory_requests(ctx: Context):
    for p, c in ctx.gpu_containers():
        req = (c.get("resources") or {}).get("requests") or {}
        missing = [r for r in ("cpu", "memory") if r not in req]
        if missing:
            yield Finding("cpu-memory-requests", "warning", ctx.where(p),
                          f"GPU container without {' / '.join(missing)} request: the scheduler cannot "
                          "reserve the host resources that feed the GPU",
                          "request cpu and memory explicitly (and a memory limit)")
        if "cpu" in ((c.get("resources") or {}).get("limits") or {}):
            yield Finding("cpu-memory-requests", "info", ctx.where(p),
                          "CPU limit on a GPU container: CFS throttling stalls the tokenizer/data loader",
                          "drop the CPU limit, keep the request")


@rule
def strands_gpus(ctx: Context):
    if not ctx.machine or not ctx.pod_gpus:
        return
    cpu = sum(machines.parse_cpu(((c.get("resources") or {}).get("requests") or {}).get("cpu", 0))
              for _, c in ctx.containers())
    mem = sum(machines.parse_memory_gib(((c.get("resources") or {}).get("requests") or {}).get("memory", 0))
              for _, c in ctx.containers())
    alloc = machines.allocatable(ctx.machine)
    n = machines.pods_per_node(alloc, pod_cpu=cpu, pod_mem_gib=mem, pod_gpus=ctx.pod_gpus)
    if n == 0:
        yield Finding("strands-gpus", "error", ctx.where(),
                      f"requests (cpu {cpu:g}, memory {mem:.1f} GiB, {ctx.pod_gpus} GPU) do not fit "
                      f"{ctx.machine.name} at all (allocatable cpu {alloc.cpu:g}, memory {alloc.memory_gib:.1f} GiB)",
                      "shrink the requests or pick a bigger machine")
        return
    stranded = alloc.gpus - n * ctx.pod_gpus
    if stranded:
        share = machines.per_gpu_share(ctx.machine)
        yield Finding("strands-gpus", "warning", ctx.where(),
                      f"{n} such pod(s) fit a {ctx.machine.name}; {stranded} of {alloc.gpus} GPUs stay idle "
                      f"(per-GPU share is cpu {share.cpu:.2f}, memory {share.memory_gib:.1f} GiB)",
                      "size cpu/memory requests to at most the per-GPU share")


@rule
def shm_size(ctx: Context):
    multi = ctx.pod_gpus > 1 or (ctx.pod_gpus and ctx.replicas_in_gang > 1 and ctx.kind in ("JobSet", "LeaderWorkerSet"))
    if not multi:
        return
    for v in ctx.spec.get("volumes") or []:
        if (v.get("emptyDir") or {}).get("medium") == "Memory":
            mounted = any(m.get("mountPath") == "/dev/shm" and m.get("name") == v.get("name")
                          for _, c in ctx.containers() for m in c.get("volumeMounts") or [])
            if mounted:
                return
    yield Finding("shm-size", "warning", ctx.where("volumes"),
                  "multi-GPU / multi-pod workload without a memory-backed /dev/shm: NCCL and PyTorch "
                  "workers exhaust the 64 MiB default",
                  "emptyDir {medium: Memory, sizeLimit: 2Gi} mounted at /dev/shm")


@rule
def rollout_surge(ctx: Context):
    if ctx.kind != "Deployment" or not ctx.pod_gpus:
        return
    strat = (ctx.obj.get("spec") or {}).get("strategy") or {}
    ru = strat.get("rollingUpdate") or {}
    if strat.get("type") == "Recreate" or str(ru.get("maxSurge", "25%")) in ("0", "0%"):
        return
    yield Finding("rollout-surge", "info", f"{ctx.kind}/{ctx.name} spec.strategy",
                  "a rolling update surges extra GPU pods (default maxSurge 25%); with no spare GPUs the "
                  "rollout stalls with the new pod Pending",
                  "maxSurge: 0, maxUnavailable: 1 — or keep headroom / a ComputeClass that can scale up")


# ---- entry points ------------------------------------------------------------------------------
def lint(obj: dict, *, machine: str | machines.Machine | None = None,
         expected_load_s: float | None = None, rules: Iterable[str] | None = None) -> list[Finding]:
    """Lint one object (Pod, Job, Deployment, JobSet, LeaderWorkerSet, ...)."""
    m = machines.CATALOG[machine] if isinstance(machine, str) else machine
    selected = [RULES[r] for r in rules] if rules else list(RULES.values())
    findings: list[Finding] = []
    for path, template in iter_pod_templates(obj):
        ctx = Context(obj, path, template, machine=m, expected_load_s=expected_load_s)
        for fn in selected:
            findings.extend(fn(ctx))
    return findings


def lint_docs(objs: Iterable[dict], **kw) -> list[Finding]:
    out: list[Finding] = []
    for o in objs:
        out.extend(lint(o, **kw))
    return out


def lint_file(path: str | Path, **kw) -> list[Finding]:
    return lint_docs(load_all(path), **kw)


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "no findings"
    order = {"error": 0, "warning": 1, "info": 2}
    lines = [str(f) for f in sorted(findings, key=lambda f: (order[f.severity], f.where, f.rule))]
    counts = {s: sum(f.severity == s for f in findings) for s in order}
    lines.append(f"-- {counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info")
    return "\n".join(lines)
