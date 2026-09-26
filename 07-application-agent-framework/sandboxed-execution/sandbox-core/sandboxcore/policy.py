"""Policy as data: what may run, where it may reach, and the budget ceiling — plus the manifests that pin it.

The one idea: the decision "should this code run, and under what limits?" belongs outside the model, as a
small data object you can read, diff and enforce twice — once in the executor (the runtime check) and once
in the cluster (Pod Security, NetworkPolicy, RuntimeClass, quotas). ``SandboxPolicy`` holds the tool tiers
(the identity primer's READ/WRITE/DESTRUCTIVE/EXTERNAL, §4.2), the egress allowlist, the filesystem rule and
the budget ceiling; ``evaluate()`` is the deny-by-default gate; ``render_k8s()`` turns the same policy into
a Namespace, a per-execution Job, a default-deny-egress NetworkPolicy that opens only the proxy, a
RuntimeClass reference, a ResourceQuota, a LimitRange and a ValidatingAdmissionPolicy. The YAML is built as
plain dicts (like ``k8sgpu.manifests``) and validates against the Kubernetes 1.34 schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    runtime_class: str = "gvisor"          # RuntimeClass handle; the lab wires runsc / GKE Sandbox
    run_as_uid: int = 65534                # numeric, so the kubelet can verify runAsNonRoot
    proxy_service: str = "egress-proxy"    # the only egress destination the NetworkPolicy opens
    proxy_port: int = 8080
    image: str = "python:3.12-slim"        # (verify tag/digest) the run_code image
    pod_pids_limit: int = 128              # kubelet KubeletConfiguration; there is no per-pod PID field

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
        clamped = Budgets(**{k: min(getattr(b, k), getattr(cap, k))
                             for k in ("cpu_s", "wall_s", "memory_mb", "pids", "file_mb",
                                       "disk_mb", "output_bytes", "open_files")})
        req.budgets = clamped
        return req

    # ---- Kubernetes rendering: the same policy, enforced by the cluster --------------------
    def render_k8s(self) -> list[dict]:
        """The manifests that make a namespace only able to run conforming sandbox pods.

        Order: Namespace (with restricted Pod Security labels) → default-deny-egress NetworkPolicy →
        an egress-to-proxy-and-DNS NetworkPolicy → ResourceQuota → LimitRange → the per-execution Job →
        the ValidatingAdmissionPolicy and its binding. Each is a plain dict; ``to_yaml`` serialises them.
        """
        return [self._namespace(), self._deny_egress(), self._allow_proxy_egress(),
                self._resource_quota(), self._limit_range(), self.job(),
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

    def _deny_egress(self) -> dict:
        # A default-deny egress policy also blocks DNS (FACTS §3); the next policy re-opens DNS + proxy only.
        return {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
                "metadata": self._meta("sandbox-default-deny-egress"),
                "spec": {"podSelector": {"matchLabels": {"app": "sandbox-run"}},
                         "policyTypes": ["Egress"]}}

    def _allow_proxy_egress(self) -> dict:
        # The sandbox pod may reach ONLY the egress proxy (and DNS). Everything else is dropped.
        return {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
                "metadata": self._meta("sandbox-egress-to-proxy"),
                "spec": {"podSelector": {"matchLabels": {"app": "sandbox-run"}},
                         "policyTypes": ["Egress"],
                         "egress": [
                             {"to": [{"podSelector": {"matchLabels": {"app": self.proxy_service}}}],
                              "ports": [{"protocol": "TCP", "port": self.proxy_port}]},
                             {"to": [{"namespaceSelector": {}}],
                              "ports": [{"protocol": "UDP", "port": 53},
                                        {"protocol": "TCP", "port": 53}]},
                         ]}}

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

    def job(self, name: str = "sandbox-run", *, code: str | None = None) -> dict:
        """One Job per execution: bounded, non-retrying, self-deleting, restricted, egress-gated.

        activeDeadlineSeconds bounds the whole Job and takes precedence over backoffLimit; backoffLimit=0
        and restartPolicy=Never stop non-idempotent code from re-running; ttlSecondsAfterFinished cleans
        up (collect logs first — the TTL deletes the Pod and its logs). See FACTS §3 and pitfalls 17-18.
        """
        pod_sc, ctr_sc = self.security_context()
        container = {
            "name": "run",
            "image": self.image,
            "command": ["python3", "-I", "-c", code or "print('hello from the sandbox')"],
            "env": [{"name": "HOME", "value": "/work"},
                    {"name": "OPENBLAS_NUM_THREADS", "value": "1"},
                    {"name": "HTTP_PROXY", "value": f"http://{self.proxy_service}:{self.proxy_port}"},
                    {"name": "HTTPS_PROXY", "value": f"http://{self.proxy_service}:{self.proxy_port}"}],
            "resources": {"requests": {"cpu": "250m", "memory": "128Mi", "ephemeral-storage": "256Mi"},
                          "limits": {"cpu": "1", "memory": f"{self.max_budgets.memory_mb}Mi",
                                     "ephemeral-storage": f"{self.max_budgets.disk_mb}Mi"}},
            "securityContext": ctr_sc,
            "volumeMounts": [{"name": "work", "mountPath": "/work"},
                             {"name": "tmp", "mountPath": "/tmp"}],
        }
        pod_spec = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,     # no cloud credentials via the KSA token (FACTS §3)
            "runtimeClassName": self.runtime_class,     # gVisor / Kata handle
            "activeDeadlineSeconds": int(self.max_budgets.wall_s) + 5,
            "securityContext": pod_sc,
            "containers": [container],
            "volumes": [
                {"name": "work", "emptyDir": {"sizeLimit": f"{self.max_budgets.disk_mb}Mi"}},
                {"name": "tmp", "emptyDir": {"medium": "Memory",
                                             "sizeLimit": f"{self.max_budgets.memory_mb}Mi"}},
            ],
        }
        return {"apiVersion": "batch/v1", "kind": "Job",
                "metadata": self._meta(name, labels={"app": "sandbox-run"}),
                "spec": {"backoffLimit": 0, "completions": 1, "parallelism": 1,
                         "activeDeadlineSeconds": int(self.max_budgets.wall_s) + 10,
                         "ttlSecondsAfterFinished": 300,
                         "template": {"metadata": {"labels": {"app": "sandbox-run"}},
                                      "spec": pod_spec}}}

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
