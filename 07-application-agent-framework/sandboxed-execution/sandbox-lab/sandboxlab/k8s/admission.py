"""admission.py — what the API server will say about a sandbox pod, before you send it.

One idea: the controls on a sandbox pod are only real if the cluster *refuses* pods without them.
Three admission steps do that, in this order (PRIMER §5 "Sandboxes on Kubernetes"):

1. **RuntimeClass admission** merges the class's ``scheduling`` into the pod (node selector —
   a conflict rejects the pod — and tolerations) and adds ``overhead.podFixed``.
2. **Pod Security Admission** ``restricted`` (namespace labels) — the upstream baseline for any
   untrusted workload. ``enforce`` rejects *Pods*, not Jobs: a violating Job is accepted and its
   Pods are rejected later, which you only see in the Job's events.
3. **A ValidatingAdmissionPolicy** for what PSS does not know about sandboxes: the runtime class,
   no service-account token, a read-only root, sized volumes only, no secrets in env, and — on
   Jobs — a deadline, no retries and a TTL.

``RULES`` is the single source for step 3: each rule carries its CEL expression (rendered into the
policy by ``policy.py``) and the same predicate in Python (``evaluate``), so a notebook can predict
the API server's answer offline, and the tests check that both agree (with ``cel-python`` when it
is installed). API-server defaulting happens *before* admission (``backoffLimit`` is already 6
when the policy sees an unset one); ``evaluate`` applies the same defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CONTAINER_LISTS = ("containers", "initContainers", "ephemeralContainers")
PSS_VOLUME_TYPES = {"configMap", "csi", "downwardAPI", "emptyDir", "ephemeral", "persistentVolumeClaim",
                    "projected", "secret"}
BASELINE_CAPS = {"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE",
                 "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT"}
JOB_DEFAULT_BACKOFF_LIMIT = 6           # batch/v1 default, applied before admission


# ---- small helpers over plain dicts -----------------------------------------------------------------
def _get(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def all_containers(spec: dict, lists: tuple[str, ...] = CONTAINER_LISTS) -> list[tuple[str, dict]]:
    return [(kind, c) for kind in lists for c in (spec.get(kind) or [])]


# ---- the sandbox rules: CEL for the API server, Python for the notebook ----------------------------------
def _cel_all(pred: str, lists: tuple[str, ...] = CONTAINER_LISTS) -> str:
    """``pred`` (over ``c``) for every container in each list; lists are variables (typed per list,
    so CEL never has to concatenate Container and EphemeralContainer lists)."""
    return " && ".join(f"variables.{l}.all(c, {pred})" for l in lists)


CEL_VARIABLES = [
    {"name": "containers", "expression": "object.spec.containers"},
    {"name": "initContainers", "expression": "has(object.spec.initContainers) ? object.spec.initContainers : []"},
    {"name": "ephemeralContainers",
     "expression": "has(object.spec.ephemeralContainers) ? object.spec.ephemeralContainers : []"},
]


@dataclass(frozen=True)
class Rule:
    name: str
    resource: str                       # "pods" or "jobs"
    cel: str
    message: str
    check: Callable[[dict, dict], bool]  # (object, params) -> passes?


def _p_runtime(o, prm):
    return _get(o, "spec", "runtimeClassName") in prm["runtime_classes"]


def _p_token(o, prm):
    return _get(o, "spec", "automountServiceAccountToken") is False


def _p_host_ns(o, prm):
    return not any(_get(o, "spec", k) for k in ("hostNetwork", "hostPID", "hostIPC"))


def _p_nonroot(o, prm):
    sc = _get(o, "spec", "securityContext") or {}
    return sc.get("runAsNonRoot") is True and sc.get("runAsUser") not in (None, 0)


def _p_seccomp(o, prm):
    return _get(o, "spec", "securityContext", "seccompProfile", "type") in ("RuntimeDefault", "Localhost")


def _each(pred, lists=CONTAINER_LISTS):
    return lambda o, prm: all(pred(c) for _, c in all_containers(o.get("spec") or {}, lists))


def _c_no_esc(c):
    return _get(c, "securityContext", "allowPrivilegeEscalation") is False


def _c_ro(c):
    return _get(c, "securityContext", "readOnlyRootFilesystem") is True


def _c_caps(c):
    caps = _get(c, "securityContext", "capabilities") or {}
    return "ALL" in (caps.get("drop") or []) and not caps.get("add")


def _c_limits(c):
    lim = _get(c, "resources", "limits") or {}
    return "cpu" in lim and "memory" in lim


def _c_no_secret_env(c):
    return not c.get("envFrom") and all(not _get(e, "valueFrom", "secretKeyRef") for e in c.get("env") or [])


def _p_volumes(o, prm):
    return all((_get(v, "emptyDir", "sizeLimit") is not None) or ("configMap" in v) for v in _get(o, "spec", "volumes") or [])


def _p_job_deadline(o, prm):
    d = _get(o, "spec", "activeDeadlineSeconds")
    return d is not None and d <= prm["max_deadline_s"]


def _p_job_backoff(o, prm):
    return _get(o, "spec", "backoffLimit", default=JOB_DEFAULT_BACKOFF_LIMIT) == 0


def _p_job_ttl(o, prm):
    return _get(o, "spec", "ttlSecondsAfterFinished") is not None


def rules(runtime_classes: list[str], max_deadline_s: int = 600) -> list[Rule]:
    rc_list = "[" + ", ".join(f"'{r}'" for r in runtime_classes) + "]"
    return [
        Rule("runtime-class", "pods",
             f"has(object.spec.runtimeClassName) && object.spec.runtimeClassName in {rc_list}",
             f"sandbox pods must set runtimeClassName to one of {runtime_classes}", _p_runtime),
        Rule("no-service-account-token", "pods",
             "has(object.spec.automountServiceAccountToken) && object.spec.automountServiceAccountToken == false",
             "sandbox pods must set automountServiceAccountToken: false (no Kubernetes API credentials inside)", _p_token),
        Rule("no-host-namespaces", "pods",
             "!(has(object.spec.hostNetwork) && object.spec.hostNetwork) && !(has(object.spec.hostPID) && "
             "object.spec.hostPID) && !(has(object.spec.hostIPC) && object.spec.hostIPC)",
             "sandbox pods must not share the node's network, PID or IPC namespace", _p_host_ns),
        Rule("run-as-non-root", "pods",
             "has(object.spec.securityContext) && has(object.spec.securityContext.runAsNonRoot) && "
             "object.spec.securityContext.runAsNonRoot == true && has(object.spec.securityContext.runAsUser) && "
             "object.spec.securityContext.runAsUser != 0",
             "sandbox pods must set pod-level runAsNonRoot: true and a numeric non-zero runAsUser", _p_nonroot),
        Rule("seccomp", "pods",
             "has(object.spec.securityContext) && has(object.spec.securityContext.seccompProfile) && "
             "object.spec.securityContext.seccompProfile.type in ['RuntimeDefault', 'Localhost']",
             "sandbox pods must set a pod-level seccompProfile of RuntimeDefault or Localhost", _p_seccomp),
        Rule("no-privilege-escalation", "pods",
             _cel_all("has(c.securityContext) && has(c.securityContext.allowPrivilegeEscalation) && "
                      "c.securityContext.allowPrivilegeEscalation == false"),
             "every container must set allowPrivilegeEscalation: false (it defaults to true)", _each(_c_no_esc)),
        Rule("read-only-root", "pods",
             _cel_all("has(c.securityContext) && has(c.securityContext.readOnlyRootFilesystem) && "
                      "c.securityContext.readOnlyRootFilesystem == true"),
             "every container must set readOnlyRootFilesystem: true (write to the sized emptyDir only)", _each(_c_ro)),
        Rule("drop-all-capabilities", "pods",
             _cel_all("has(c.securityContext) && has(c.securityContext.capabilities) && "
                      "has(c.securityContext.capabilities.drop) && 'ALL' in c.securityContext.capabilities.drop && "
                      "(!has(c.securityContext.capabilities.add) || size(c.securityContext.capabilities.add) == 0)"),
             "every container must drop ALL capabilities and add none", _each(_c_caps)),
        Rule("resource-limits", "pods",
             _cel_all("has(c.resources) && has(c.resources.limits) && 'cpu' in c.resources.limits && "
                      "'memory' in c.resources.limits", ("containers", "initContainers")),
             "every container must have cpu and memory limits (the LimitRange fills defaults before this runs)",
             _each(_c_limits, ("containers", "initContainers"))),
        Rule("no-secret-env", "pods",
             _cel_all("!has(c.envFrom) && (!has(c.env) || c.env.all(e, !has(e.valueFrom) || "
                      "!has(e.valueFrom.secretKeyRef)))"),
             "no Secrets in a sandbox's environment: credentials live in the egress proxy", _each(_c_no_secret_env)),
        Rule("volumes-allowlist", "pods",
             "!has(object.spec.volumes) || object.spec.volumes.all(v, (has(v.emptyDir) && has(v.emptyDir.sizeLimit)) "
             "|| has(v.configMap))",
             "sandbox pods may mount only sized emptyDir volumes and ConfigMaps (no Secrets, no hostPath, no PVCs)",
             _p_volumes),
        Rule("job-deadline", "jobs",
             f"has(object.spec.activeDeadlineSeconds) && object.spec.activeDeadlineSeconds <= {max_deadline_s}",
             f"sandbox Jobs must set activeDeadlineSeconds <= {max_deadline_s}", _p_job_deadline),
        Rule("job-no-retries", "jobs", "object.spec.backoffLimit == 0",
             "sandbox Jobs must set backoffLimit: 0 (the default is 6: model-generated code would run up to 7 times)",
             _p_job_backoff),
        Rule("job-ttl", "jobs", "has(object.spec.ttlSecondsAfterFinished)",
             "sandbox Jobs must set ttlSecondsAfterFinished (finished Jobs and their Pods are deleted)", _p_job_ttl),
    ]


@dataclass
class Violation:
    step: str          # "runtimeclass" | "pss" | "vap"
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.step}] {self.rule}: {self.message}"


def evaluate(obj: dict, runtime_classes: list[str], max_deadline_s: int = 600) -> list[Violation]:
    """The ValidatingAdmissionPolicy's verdict on a Pod or a Job, offline."""
    res = {"Pod": "pods", "Job": "jobs"}.get(obj.get("kind"))
    prm = {"runtime_classes": runtime_classes, "max_deadline_s": max_deadline_s}
    return [Violation("vap", r.name, r.message) for r in rules(runtime_classes, max_deadline_s)
            if r.resource == res and not r.check(obj, prm)]


# ---- Pod Security Standards, restricted (concepts/security/pod-security-standards.md) -------------------------
def pss_restricted(spec: dict) -> list[str]:
    """PSS ``restricted`` (which includes ``baseline``) on a pod spec; empty list = admitted.
    Container fields may be omitted when the pod-level field covers them (runAsNonRoot, seccomp)."""
    out: list[str] = []
    psc = spec.get("securityContext") or {}
    for k in ("hostNetwork", "hostPID", "hostIPC"):
        if spec.get(k):
            out.append(f"host namespaces ({k}=true)")
    for v in spec.get("volumes") or []:
        kinds = set(v) - {"name"}
        bad = kinds - PSS_VOLUME_TYPES
        if bad:
            out.append(f'restricted volume types (volume "{v.get("name")}" uses {", ".join(sorted(bad))})')
    pod_nonroot = psc.get("runAsNonRoot") is True
    pod_seccomp = _get(psc, "seccompProfile", "type")
    if psc.get("runAsUser") == 0:
        out.append("runAsUser=0 (pod must not set runAsUser=0)")
    if pod_seccomp == "Unconfined":
        out.append('seccompProfile (pod must not set securityContext.seccompProfile.type to "Unconfined")')
    for kind, c in all_containers(spec):
        n, sc = c.get("name"), c.get("securityContext") or {}
        if sc.get("privileged"):
            out.append(f'privileged (container "{n}" must not set securityContext.privileged=true)')
        if sc.get("allowPrivilegeEscalation") is not False:
            out.append(f'allowPrivilegeEscalation != false (container "{n}" must set securityContext.allowPrivilegeEscalation=false)')
        caps = sc.get("capabilities") or {}
        if "ALL" not in (caps.get("drop") or []):
            out.append(f'unrestricted capabilities (container "{n}" must set securityContext.capabilities.drop=["ALL"])')
        added = set(caps.get("add") or [])
        if added - {"NET_BIND_SERVICE"}:
            out.append(f'unrestricted capabilities (container "{n}" must not include {sorted(added - {"NET_BIND_SERVICE"})} in securityContext.capabilities.add)')
        c_nonroot = sc.get("runAsNonRoot")
        if c_nonroot is False or (c_nonroot is None and not pod_nonroot):
            out.append(f'runAsNonRoot != true (pod or container "{n}" must set securityContext.runAsNonRoot=true)')
        if sc.get("runAsUser") == 0:
            out.append(f'runAsUser=0 (container "{n}" must not set runAsUser=0)')
        c_seccomp = _get(sc, "seccompProfile", "type")
        eff = c_seccomp or pod_seccomp
        if eff not in ("RuntimeDefault", "Localhost"):
            out.append(f'seccompProfile (pod or container "{n}" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost")')
        for p in c.get("ports") or []:
            if p.get("hostPort"):
                out.append(f'hostPort (container "{n}" uses hostPort {p["hostPort"]})')
    return out


# ---- RuntimeClass admission ---------------------------------------------------------------------------------
def apply_runtime_class(spec: dict, rc: dict | None) -> tuple[dict, list[str]]:
    """What the RuntimeClass admission controller does to a pod: merge ``scheduling.nodeSelector``
    (a key with a different value is a conflict and the pod is rejected), union
    ``scheduling.tolerations``, set ``overhead`` from ``overhead.podFixed``. A RuntimeClass that
    does not exist is *not* an admission error: the pod is created and ends in phase ``Failed``
    (``runtime_failure`` below says so)."""
    out = dict(spec)
    errs: list[str] = []
    if rc is None:
        return out, errs
    sched = rc.get("scheduling") or {}
    ns = dict(spec.get("nodeSelector") or {})
    for k, v in (sched.get("nodeSelector") or {}).items():
        if k in ns and ns[k] != v:
            errs.append(f"conflict with RuntimeClass {rc['metadata']['name']}: nodeSelector {k}={ns[k]} vs {v}")
        ns[k] = v
    if ns:
        out["nodeSelector"] = ns
    tols = list(spec.get("tolerations") or [])
    for t in sched.get("tolerations") or []:
        if t not in tols:
            tols.append(t)
    if tols:
        out["tolerations"] = tols
    if rc.get("overhead"):
        out["overhead"] = dict(rc["overhead"]["podFixed"])
    return out, errs


_UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3}


def quantity(q) -> float:
    """Kubernetes quantity -> float (cores or bytes): '250m' -> 0.25, '320Mi' -> 335544320."""
    s = str(q)
    for suf in ("Ki", "Mi", "Gi", "Ti", "k", "M", "G", "m"):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * _UNITS[suf]
    return float(s)


def effective_requests(spec: dict) -> dict[str, float]:
    """Pod requests as the scheduler and ResourceQuota count them: sum of containers, max with
    each init container, plus the RuntimeClass overhead."""
    tot: dict[str, float] = {}
    for c in spec.get("containers") or []:
        for k, v in (_get(c, "resources", "requests") or {}).items():
            tot[k] = tot.get(k, 0.0) + quantity(v)
    for c in spec.get("initContainers") or []:
        for k, v in (_get(c, "resources", "requests") or {}).items():
            tot[k] = max(tot.get(k, 0.0), quantity(v))
    for k, v in (spec.get("overhead") or {}).items():
        tot[k] = tot.get(k, 0.0) + quantity(v)
    return tot


def runtime_failure(spec: dict, runtime_classes: dict[str, dict], node_handlers: set[str] | None) -> str | None:
    """Why an *admitted* pod would still end in phase Failed: its RuntimeClass does not exist, or no
    node can run the class's handler (kind's containerd has ``runc`` only — no ``runsc``)."""
    name = spec.get("runtimeClassName")
    if not name:
        return None
    rc = runtime_classes.get(name)
    if rc is None:
        return f'RuntimeClass "{name}" not found: the pod is admitted and then Failed'
    if node_handlers is not None and rc["handler"] not in node_handlers:
        return (f'no runtime for "{rc["handler"]}" is configured on the node (handlers: {sorted(node_handlers)}): '
                "the pod sandbox cannot be created and the pod ends Failed")
    return None


@dataclass
class AdmissionResult:
    admitted: bool
    violations: list[Violation]
    spec: dict                       # after RuntimeClass admission
    will_fail: str | None = None     # admitted, but the node cannot run it

    def report(self) -> str:
        if self.admitted:
            return "admitted" + (f" -- but {self.will_fail}" if self.will_fail else "")
        return "rejected:\n" + "\n".join(f"  {v}" for v in self.violations)


def admit(obj: dict, *, runtime_classes: dict[str, dict], allowed_runtime_classes: list[str],
          enforce_pss: bool = True, max_deadline_s: int = 600, node_handlers: set[str] | None = None) -> AdmissionResult:
    """The whole admission chain for one object in a sandbox namespace, in the API server's order.
    For a Job, the pod checks apply to its template and only the Job rules can reject the Job
    itself; PSS ``enforce`` would reject its Pods later, which is reported as a pod-template
    violation here."""
    kind = obj.get("kind")
    violations: list[Violation] = []
    if kind == "Job":
        violations += evaluate(obj, allowed_runtime_classes, max_deadline_s)
        tmpl = obj["spec"]["template"]
        pod = {"kind": "Pod", "metadata": tmpl.get("metadata", {}), "spec": tmpl["spec"]}
        later = admit(pod, runtime_classes=runtime_classes, allowed_runtime_classes=allowed_runtime_classes,
                      enforce_pss=enforce_pss, max_deadline_s=max_deadline_s, node_handlers=node_handlers)
        violations += [Violation(v.step, v.rule, "(its Pods) " + v.message) for v in later.violations]
        return AdmissionResult(not violations, violations, later.spec, later.will_fail)
    spec = obj.get("spec") or {}
    rcname = spec.get("runtimeClassName")
    merged, errs = apply_runtime_class(spec, runtime_classes.get(rcname) if rcname else None)
    violations += [Violation("runtimeclass", "scheduling", e) for e in errs]
    if enforce_pss:
        violations += [Violation("pss", "restricted", m) for m in pss_restricted(merged)]
    violations += evaluate({**obj, "spec": merged}, allowed_runtime_classes, max_deadline_s)
    return AdmissionResult(not violations, violations, merged, runtime_failure(spec, runtime_classes, node_handlers))
