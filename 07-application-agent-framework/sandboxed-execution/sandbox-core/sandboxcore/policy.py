"""Policy as data: what may run, where it may reach, and the budget ceiling — plus the manifests that pin it.

The one idea: the decision "should this code run, and under what limits?" belongs outside the model, as a
small data object you can read, diff and enforce twice — once in the executor (the runtime check) and once
in the cluster (Pod Security, NetworkPolicy, RuntimeClass, quotas). ``SandboxPolicy`` holds the tool tiers
(the identity primer's READ/WRITE/DESTRUCTIVE/EXTERNAL, §4.2), the egress allowlist, the filesystem rule and
the budget ceiling; ``evaluate()`` is the deny-by-default gate; ``render_k8s()`` turns the same policy into
a Namespace, the RuntimeClass, a default-deny-egress NetworkPolicy plus one that opens only the egress proxy
(no DNS: the pod finds the proxy through ``hostAliases`` and a pinned ClusterIP), the proxy's Service, a
ResourceQuota, a LimitRange, a per-execution Pod and Job, and a ValidatingAdmissionPolicy. The YAML is built
as plain dicts (like ``k8sgpu.manifests``) and validates against the Kubernetes 1.34 schemas; a test also
checks what the schema cannot (every request ≤ its limit).

``evaluate()`` reads ``ExecutionRequest.egress`` — the hosts the *model says* its code needs. That is a
policy-review input and an audit signal, never enforcement: code that does not declare a host is not
stopped by it. The network layer (the NetworkPolicy rendered here) is what enforces egress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .contract import Budgets, ExecutionRequest


class Tier(str, Enum):
    """The identity primer's tool tiers (agentsec `Tier`), extended with EXTERNAL for egress tools."""
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Decision:
    effect: Effect
    reasons: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW


@dataclass
class SandboxPolicy:
    name: str = "default"
    # run_code is DESTRUCTIVE by definition (identity primer §6.2): "An 'execute code' tool is
    # DESTRUCTIVE-tier by definition." Running it never needs a human, but leaving the sandbox does.
    tool_tier: Tier = Tier.DESTRUCTIVE
    allowed_principals: frozenset[str] = frozenset()   # empty = any principal (still deny-by-default on tools)
    egress_allowlist: tuple[str, ...] = ()             # hostnames the proxy will reach; () = no egress
    egress_needs_confirm: bool = False                 # ask a human before an execution that requests egress
    filesystem: str = "ephemeral"                      # ephemeral | none (no writable workspace at all)
    max_budgets: Budgets = field(default_factory=Budgets)   # the ceiling; a request may ask for less
    # Kubernetes rendering knobs
    namespace: str = "sandbox"
    runtime_class: str = "gvisor"          # RuntimeClass name the pods select
    runtime_handler: str = "runsc"         # CRI handler for a self-managed gVisor node
    render_runtime_class: bool = True      # False on GKE: GKE creates RuntimeClass "gvisor" itself (verify)
    run_as_uid: int = 65534                # numeric, so the kubelet can verify runAsNonRoot
    proxy_service: str = "egress-proxy"    # the only egress destination the NetworkPolicy opens
    proxy_port: int = 8080
    proxy_cluster_ip: str = "10.96.0.200"  # pinned in the Service; pods reach it via hostAliases (no DNS)
    image: str = "python:3.12-slim"        # (verify tag/digest) the run_code image
    pod_pids_limit: int = 128              # kubelet KubeletConfiguration; there is no per-pod PID field
    startup_allowance_s: int = 120         # scheduling + node scale-up + image pull, before the code runs
    log_headroom_mb: int = 48              # container ephemeral-storage on top of the workspace: logs, layer

    def evaluate(self, req: ExecutionRequest) -> Decision:
        """Deny by default. Check the principal, the budget ceiling, the egress allowlist, then confirm."""
        reasons: list[str] = []
        if self.allowed_principals and req.principal not in self.allowed_principals:
            return Decision(Effect.DENY, [f"principal {req.principal!r} is not allowed to run code"])
        over = req.budgets.exceeding(self.max_budgets)
        if over:
            return Decision(Effect.DENY, [f"requested budget exceeds the policy ceiling: {', '.join(over)}"])
        bad = [h for h in req.egress if h not in self.egress_allowlist]
        if bad:
            return Decision(Effect.DENY, [f"egress to {h} is not on the allowlist" for h in bad])
        if req.egress and self.egress_needs_confirm:
            reasons.append("egress requested; needs human confirmation")
            return Decision(Effect.CONFIRM, reasons)
        return Decision(Effect.ALLOW, reasons or ["within policy"])

    def clamp(self, req: ExecutionRequest) -> ExecutionRequest:
        """Lower any budget field that exceeds the ceiling, so a request can never widen the envelope."""
        b, cap = req.budgets, self.max_budgets
        clamped = Budgets(**{k: min(getattr(b, k), getattr(cap, k)) for k in asdict(b)})
        req.budgets = clamped
        return req

    # ---- Kubernetes rendering: the same policy, enforced by the cluster --------------------
    def render_k8s(self) -> list[dict]:
        """The manifests that make a namespace only able to run conforming sandbox pods.

        Order: Namespace (restricted Pod Security labels) → RuntimeClass (unless the platform creates it) →
        default-deny-egress NetworkPolicy → egress-to-the-proxy-only NetworkPolicy → the proxy's Service at
        a pinned ClusterIP → ResourceQuota → LimitRange → a per-execution Pod and Job → the
        ValidatingAdmissionPolicy and its binding. Each is a plain dict; ``to_yaml`` serialises them.
        """
        return [self._namespace(), *([self.runtime_class_obj()] if self.render_runtime_class else []),
                self._deny_egress(), self._allow_proxy_egress(), self._proxy_service(),
                self._resource_quota(), self._limit_range(), self.pod(), self.job(),
                *self._admission_policy()]

    def _meta(self, name: str, **extra) -> dict:
        m = {"name": name, "namespace": self.namespace}
        m.update(extra)
        return m

    def _namespace(self) -> dict:
        # Pod Security Admission enforces the RESTRICTED level on Pods in this namespace (FACTS §3).
        labels = {f"pod-security.kubernetes.io/{mode}": "restricted"
                  for mode in ("enforce", "audit", "warn")}
        labels.update({f"pod-security.kubernetes.io/{mode}-version": "v1.34"
                       for mode in ("enforce", "audit", "warn")})
        return {"apiVersion": "v1", "kind": "Namespace",
                "metadata": {"name": self.namespace, "labels": labels}}

    def runtime_class_obj(self) -> dict:
        """The RuntimeClass the pods select: the object that ties the policy to its isolation rung.

        Self-managed gVisor: handler ``runsc`` (the containerd runtime name). GKE creates ``gvisor`` itself
        with the first GKE Sandbox node pool (``render_runtime_class=False`` there). Kata would name a Kata
        shim and set ``overhead.podFixed`` so the scheduler and quota count the VMM.
        """
        return {"apiVersion": "node.k8s.io/v1", "kind": "RuntimeClass",
                "metadata": {"name": self.runtime_class}, "handler": self.runtime_handler}

    def _deny_egress(self) -> dict:
        # A default-deny egress policy also blocks DNS (FACTS §3) — and that is kept: no DNS for sandboxes.
        return {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
                "metadata": self._meta("sandbox-default-deny-egress"),
                "spec": {"podSelector": {"matchLabels": {"app": "sandbox-run"}},
                         "policyTypes": ["Egress"]}}

    def _allow_proxy_egress(self) -> dict:
        # The sandbox pod may reach ONLY the egress proxy. No DNS rule: resolving names would re-open a
        # DNS-exfiltration channel (data in query names), so the pod finds the proxy through hostAliases
        # and the proxy resolves the allowlisted names itself (PRIMER §4).
        return {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
                "metadata": self._meta("sandbox-egress-to-proxy"),
                "spec": {"podSelector": {"matchLabels": {"app": "sandbox-run"}},
                         "policyTypes": ["Egress"],
                         "egress": [
                             {"to": [{"podSelector": {"matchLabels": {"app": self.proxy_service}}}],
                              "ports": [{"protocol": "TCP", "port": self.proxy_port}]},
                         ]}}

    def _proxy_service(self) -> dict:
        # The proxy's Service at a pinned ClusterIP, so hostAliases can name it without DNS. Its pods run in
        # this namespace (the Service and the egress rule select them here), so they meet Pod Security
        # restricted and the admission policy too; the lab instead gives the proxy its own namespace.
        return {"apiVersion": "v1", "kind": "Service",
                "metadata": self._meta(self.proxy_service),
                "spec": {"clusterIP": self.proxy_cluster_ip,
                         "selector": {"app": self.proxy_service},
                         "ports": [{"name": "http", "port": self.proxy_port, "targetPort": self.proxy_port,
                                    "protocol": "TCP"}]}}

    def _resource_quota(self) -> dict:
        return {"apiVersion": "v1", "kind": "ResourceQuota",
                "metadata": self._meta("sandbox-quota"),
                "spec": {"hard": {"pods": "50",
                                  "requests.cpu": "10", "requests.memory": "20Gi",
                                  "limits.cpu": "20", "limits.memory": "40Gi",
                                  "count/jobs.batch": "50"}}}

    def _limit_range(self) -> dict:
        # ResourceQuota on cpu/memory rejects pods without requests/limits, so ship defaults (FACTS §3).
        return {"apiVersion": "v1", "kind": "LimitRange",
                "metadata": self._meta("sandbox-defaults"),
                "spec": {"limits": [{"type": "Container",
                                     "default": {"cpu": "500m", "memory": "256Mi",
                                                 "ephemeral-storage": "1Gi"},
                                     "defaultRequest": {"cpu": "250m", "memory": "128Mi",
                                                        "ephemeral-storage": "256Mi"}}]}}

    def security_context(self) -> tuple[dict, dict]:
        """(pod securityContext, container securityContext) meeting Pod Security 'restricted' (FACTS §3)."""
        pod_sc = {"runAsNonRoot": True, "runAsUser": self.run_as_uid, "runAsGroup": self.run_as_uid,
                  "fsGroup": self.run_as_uid, "seccompProfile": {"type": "RuntimeDefault"}}
        ctr_sc = {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                  "runAsNonRoot": True, "runAsUser": self.run_as_uid,
                  "capabilities": {"drop": ["ALL"]},
                  "seccompProfile": {"type": "RuntimeDefault"}}
        return pod_sc, ctr_sc

    def pod_spec(self, code: str | None = None) -> dict:
        """The sandbox pod: restricted, egress only to the proxy by hostAliases, no token, sized volumes.

        The code runs under coreutils ``timeout -s KILL <wall_s>``: the wall budget is enforced *inside*
        the pod, because every Kubernetes deadline also counts scheduling and the image pull.
        """
        pod_sc, ctr_sc = self.security_context()
        b = self.max_budgets
        ephemeral = f"{b.disk_mb + self.log_headroom_mb}Mi"   # the workspace emptyDir counts toward it too
        container = {
            "name": "run",
            "image": self.image,
            "command": ["timeout", "-s", "KILL", f"{int(b.wall_s)}",
                        "python3", "-I", "-c", code or "print('hello from the sandbox')"],
            "env": [{"name": "HOME", "value": "/work"},
                    {"name": "OPENBLAS_NUM_THREADS", "value": "1"},
                    # advisory only (the NetworkPolicy enforces); no HTTPS_PROXY: the proxy refuses CONNECT
                    {"name": "HTTP_PROXY", "value": f"http://{self.proxy_service}:{self.proxy_port}"}],
            # requests <= limits for every resource (the API server rejects a request above its limit)
            "resources": {"requests": {"cpu": "250m", "memory": f"{min(128, b.memory_mb)}Mi",
                                       "ephemeral-storage": ephemeral},
                          "limits": {"cpu": "1", "memory": f"{b.memory_mb}Mi",
                                     "ephemeral-storage": ephemeral}},
            "securityContext": ctr_sc,
            "volumeMounts": [{"name": "work", "mountPath": "/work"},
                             {"name": "tmp", "mountPath": "/tmp"}],
        }
        return {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,     # no cloud credentials via the KSA token (FACTS §3)
            "enableServiceLinks": False,               # no *_SERVICE_HOST variables advertising the cluster
            "runtimeClassName": self.runtime_class,     # gVisor / Kata handle
            "dnsPolicy": "None",                        # no cluster DNS: nothing to exfiltrate through
            "dnsConfig": {"nameservers": ["127.0.0.1"]},
            "hostAliases": [{"ip": self.proxy_cluster_ip, "hostnames": [self.proxy_service]}],
            "securityContext": pod_sc,
            "containers": [container],
            "volumes": [
                {"name": "work", "emptyDir": {"sizeLimit": f"{b.disk_mb}Mi"}},
                {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": f"{b.memory_mb}Mi"}},
            ],
        }

    def pod(self, name: str = "sandbox-run-pod", *, code: str | None = None) -> dict:
        """One execution as a bare Pod (what a warm-pool or custom runner creates). Its deadline counts
        from the kubelet admitting it — before the image pull — so it gets the startup allowance too."""
        spec = self.pod_spec(code)
        spec["activeDeadlineSeconds"] = self.startup_allowance_s + int(self.max_budgets.wall_s)
        return {"apiVersion": "v1", "kind": "Pod",
                "metadata": self._meta(name, labels={"app": "sandbox-run"}), "spec": spec}

    def job(self, name: str = "sandbox-run", *, code: str | None = None) -> dict:
        """One Job per execution: bounded, non-retrying, self-deleting, restricted, egress-gated.

        ``activeDeadlineSeconds`` bounds the *whole* Job from its start — scheduling, node scale-up and
        image pull included — and takes precedence over ``backoffLimit``; so it is the startup allowance
        plus the wall budget, and the wall budget itself is enforced inside the pod (``timeout``).
        ``backoffLimit=0`` and ``restartPolicy=Never`` stop non-idempotent code from re-running;
        ``ttlSecondsAfterFinished`` cleans up (collect logs first — the TTL deletes the Pod and its logs).
        See FACTS §3 and pitfalls 17-18.
        """
        return {"apiVersion": "batch/v1", "kind": "Job",
                "metadata": self._meta(name, labels={"app": "sandbox-run"}),
                "spec": {"backoffLimit": 0, "completions": 1, "parallelism": 1,
                         "activeDeadlineSeconds": self.startup_allowance_s + int(self.max_budgets.wall_s),
                         "ttlSecondsAfterFinished": 300,
                         "template": {"metadata": {"labels": {"app": "sandbox-run"}},
                                      "spec": self.pod_spec(code)}}}

    def _admission_policy(self) -> list[dict]:
        """A ValidatingAdmissionPolicy (stable since 1.30) that rejects sandbox pods missing the controls.

        Requires the gVisor RuntimeClass, runAsNonRoot, dropped capabilities and no privilege escalation.
        The binding must set validationActions; Deny+Warn together is invalid (FACTS §3, pitfall 22).
        """
        expr = "object.spec"
        policy = {
            "apiVersion": "admissionregistration.k8s.io/v1", "kind": "ValidatingAdmissionPolicy",
            "metadata": {"name": "sandbox-pods-must-be-isolated"},
            "spec": {
                "failurePolicy": "Fail",
                "matchConstraints": {"resourceRules": [
                    {"apiGroups": [""], "apiVersions": ["v1"],
                     "operations": ["CREATE", "UPDATE"], "resources": ["pods"]}]},
                "validations": [
                    {"expression": f"{expr}.?runtimeClassName.orValue('') == '{self.runtime_class}'",
                     "message": f"sandbox pods must set runtimeClassName: {self.runtime_class}",
                     "reason": "Forbidden"},
                    {"expression": f"{expr}.?securityContext.runAsNonRoot.orValue(false) == true",
                     "message": "sandbox pods must set securityContext.runAsNonRoot: true",
                     "reason": "Forbidden"},
                    {"expression": (f"{expr}.containers.all(c, "
                                    "c.?securityContext.allowPrivilegeEscalation.orValue(true) == false)"),
                     "message": "every container must set allowPrivilegeEscalation: false",
                     "reason": "Forbidden"},
                    {"expression": (f"{expr}.containers.all(c, "
                                    "'ALL' in c.?securityContext.capabilities.drop.orValue([]))"),
                     "message": "every container must drop ALL capabilities",
                     "reason": "Forbidden"},
                ],
            },
        }
        binding = {
            "apiVersion": "admissionregistration.k8s.io/v1", "kind": "ValidatingAdmissionPolicyBinding",
            "metadata": {"name": "sandbox-pods-must-be-isolated-binding"},
            "spec": {"policyName": "sandbox-pods-must-be-isolated",
                     "validationActions": ["Deny"],
                     "matchResources": {"namespaceSelector": {"matchLabels": {
                         "kubernetes.io/metadata.name": self.namespace}}}},
        }
        return [policy, binding]


# ---- YAML serialisation (mirrors k8sgpu.manifests: keep insertion order) --------------------
def to_yaml(*objs: dict, header: str | None = None) -> str:
    """Serialise objects as a multi-document YAML stream, keeping field order (needs PyYAML)."""
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def _dict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    _Dumper.add_representer(dict, _dict)
    parts = []
    if header:
        parts.append("\n".join(f"# {line}" for line in header.splitlines()))
    for o in objs:
        parts.append(yaml.dump(o, Dumper=_Dumper, default_flow_style=False, sort_keys=False))
    return "---\n".join(parts)


def render_yaml(policy: SandboxPolicy, header: str | None = None) -> str:
    return to_yaml(*policy.render_k8s(), header=header or f"sandbox policy: {policy.name} (generated)")
