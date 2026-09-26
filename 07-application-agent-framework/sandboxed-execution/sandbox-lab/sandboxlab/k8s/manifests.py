"""Typed builders for the objects a sandbox platform is made of.

The one idea: a Kubernetes sandbox is ordinary Kubernetes with a handful of load-bearing fields —
a ``runtimeClassName``, a locked-down ``securityContext``, ``automountServiceAccountToken:
false``, a sized ``emptyDir``, a Job deadline — plus the namespace-level objects that make those
fields mandatory rather than hoped-for: Pod Security labels, a default-deny NetworkPolicy, a
ResourceQuota/LimitRange and a ValidatingAdmissionPolicy. Each builder returns a plain ``dict``
that serialises to exactly the YAML you would apply (the style of layer 03's
``k8sgpu.manifests``), so the function is the documentation of every field that matters.

API versions are pinned for Kubernetes 1.34 (``kubernetes-validate --strict -k 1.34.0``) and
agent-sandbox v1.0.2 (verify); see ``API_VERSIONS``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

API_VERSIONS: dict[str, str] = {
    "Namespace": "v1",
    "Pod": "v1",
    "ServiceAccount": "v1",
    "ConfigMap": "v1",
    "Service": "v1",
    "ResourceQuota": "v1",
    "LimitRange": "v1",
    "Job": "batch/v1",
    "Deployment": "apps/v1",
    "Role": "rbac.authorization.k8s.io/v1",
    "RoleBinding": "rbac.authorization.k8s.io/v1",
    "RuntimeClass": "node.k8s.io/v1",
    "NetworkPolicy": "networking.k8s.io/v1",
    "ValidatingAdmissionPolicy": "admissionregistration.k8s.io/v1",          # GA since 1.30
    "ValidatingAdmissionPolicyBinding": "admissionregistration.k8s.io/v1",
    # kubernetes-sigs/agent-sandbox (SIG Apps) CRDs, v1.0.2 (verify)
    "Sandbox": "agents.x-k8s.io/v1beta1",
    "SandboxTemplate": "extensions.agents.x-k8s.io/v1beta1",
    "SandboxWarmPool": "extensions.agents.x-k8s.io/v1beta1",
    "SandboxClaim": "extensions.agents.x-k8s.io/v1beta1",
}
CORE_KINDS = {k for k, v in API_VERSIONS.items() if "x-k8s.io" not in v}

NAME_LABEL = "kubernetes.io/metadata.name"      # immutable, set by the API server on every namespace
PSS_LEVELS = ("privileged", "baseline", "restricted")


def _meta(name: str, namespace: str | None = None, labels: dict | None = None,
          annotations: dict | None = None) -> dict:
    m: dict[str, Any] = {"name": name}
    if namespace:
        m["namespace"] = namespace
    if labels:
        m["labels"] = dict(labels)
    if annotations:
        m["annotations"] = dict(annotations)
    return m


def obj(kind: str, name: str, namespace: str | None = None, *, labels: dict | None = None,
        annotations: dict | None = None, **body: Any) -> dict:
    """Any object: apiVersion from ``API_VERSIONS``, metadata, then the body fields in order."""
    out: dict[str, Any] = {"apiVersion": API_VERSIONS[kind], "kind": kind,
                           "metadata": _meta(name, namespace, labels, annotations)}
    out.update({k: v for k, v in body.items() if v is not None})
    return out


# ---- namespaces and identity -----------------------------------------------------------------------
def pss_labels(level: str = "restricted", version: str = "v1.34") -> dict:
    """Pod Security Admission labels: enforce (rejects Pods), audit and warn (also look at
    workload templates, so a bad Job gets a warning at create time even though only its Pods are
    rejected later)."""
    if level not in PSS_LEVELS:
        raise ValueError(f"level must be one of {PSS_LEVELS}")
    out = {}
    for mode in ("enforce", "audit", "warn"):
        out[f"pod-security.kubernetes.io/{mode}"] = level
        out[f"pod-security.kubernetes.io/{mode}-version"] = version
    return out


def namespace(name: str, labels: dict | None = None) -> dict:
    return obj("Namespace", name, labels=labels)


def service_account(name: str, namespace: str, *, automount: bool = False, annotations: dict | None = None) -> dict:
    return obj("ServiceAccount", name, namespace, annotations=annotations, automountServiceAccountToken=automount)


def role(name: str, namespace: str, rules: list[dict]) -> dict:
    return obj("Role", name, namespace, rules=rules)


def role_binding(name: str, namespace: str, role_name: str, sa: str, sa_namespace: str | None = None) -> dict:
    return obj("RoleBinding", name, namespace,
               roleRef={"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": role_name},
               subjects=[{"kind": "ServiceAccount", "name": sa, "namespace": sa_namespace or namespace}])


def rule(api_groups: list[str], resources: list[str], verbs: list[str], resource_names: list[str] | None = None) -> dict:
    r: dict[str, Any] = {"apiGroups": api_groups, "resources": resources, "verbs": verbs}
    if resource_names:
        r["resourceNames"] = resource_names
    return r


# ---- containers and pods -------------------------------------------------------------------------
def pod_security_context(uid: int = 65534, *, fs_group: bool = True) -> dict:
    """Pod-level: non-root by number (the kubelet cannot check a *named* image USER against
    runAsNonRoot), the runtime's default seccomp profile, and a group for the emptyDir volumes."""
    sc: dict[str, Any] = {"runAsNonRoot": True, "runAsUser": uid, "runAsGroup": uid,
                          "seccompProfile": {"type": "RuntimeDefault"}}
    if fs_group:
        sc["fsGroup"] = uid
    return sc


def container_security_context(read_only_root: bool = True) -> dict:
    """Container-level: the four fields restricted PSS checks per container, plus a read-only root.
    ``allowPrivilegeEscalation`` defaults to *true* when unset, so it is always written out.
    ``appArmorProfile`` is deliberately absent: set explicitly, it makes the Pod unadmittable on
    nodes without AppArmor (kind on some hosts); unset, the runtime default applies where it exists."""
    return {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": read_only_root,
            "runAsNonRoot": True, "capabilities": {"drop": ["ALL"]}}


def resources(cpu: str, memory: str, ephemeral: str | None = None, *, cpu_limit: bool = True) -> dict:
    """requests == limits (Guaranteed QoS for CPU and memory) — a sandbox should not burst into
    its neighbours. Ephemeral storage is limited too: the container's writable layer and logs."""
    req: dict[str, Any] = {"cpu": cpu, "memory": memory}
    lim: dict[str, Any] = {"memory": memory}
    if cpu_limit:
        lim["cpu"] = cpu
    if ephemeral:
        req["ephemeral-storage"] = ephemeral
        lim["ephemeral-storage"] = ephemeral
    return {"requests": req, "limits": lim}


def container(name: str, image: str, *, command: list[str] | None = None, args: list[str] | None = None,
              env: dict[str, str] | None = None, env_from_secret: dict[str, tuple[str, str]] | None = None,
              res: dict | None = None, security_context: dict | None = None, volume_mounts: list[dict] | None = None,
              ports: list[dict] | None = None, readiness_probe: dict | None = None, working_dir: str | None = None,
              stdin: bool = False) -> dict:
    c: dict[str, Any] = {"name": name, "image": image}
    if command:
        c["command"] = list(command)
    if args:
        c["args"] = list(args)
    if working_dir:
        c["workingDir"] = working_dir
    e = [{"name": k, "value": v} for k, v in (env or {}).items()]
    e += [{"name": k, "valueFrom": {"secretKeyRef": {"name": s, "key": key}}} for k, (s, key) in (env_from_secret or {}).items()]
    if e:
        c["env"] = e
    if ports:
        c["ports"] = ports
    if res:
        c["resources"] = res
    if readiness_probe:
        c["readinessProbe"] = readiness_probe
    c["securityContext"] = security_context if security_context is not None else container_security_context()
    if volume_mounts:
        c["volumeMounts"] = volume_mounts
    if stdin:
        c["stdin"] = True
    return c


def empty_dir(name: str, size: str, medium: str | None = None) -> dict:
    """``sizeLimit`` is mandatory here: an unsized emptyDir can fill the node's disk (or, with
    ``medium: Memory``, count against the container's memory limit without a cap of its own)."""
    ed: dict[str, Any] = {"sizeLimit": size}
    if medium:
        ed["medium"] = medium
    return {"name": name, "emptyDir": ed}


def config_map_volume(name: str, config_map: str, mode: int = 0o444) -> dict:
    return {"name": name, "configMap": {"name": config_map, "defaultMode": mode}}


def pod_spec(containers: list[dict], *, runtime_class: str | None = None, service_account: str | None = None,
             automount_token: bool = False, security_context: dict | None = None, volumes: list[dict] | None = None,
             restart_policy: str | None = "Never", termination_grace_s: int | None = 0,
             active_deadline_s: int | None = None, host_aliases: list[dict] | None = None,
             dns_none: bool = False, enable_service_links: bool = False, node_selector: dict | None = None,
             tolerations: list[dict] | None = None, init_containers: list[dict] | None = None) -> dict:
    """A pod spec with no ambient authority: no token, no service-link env vars, optional
    ``dnsPolicy: None`` (resolution fails fast; only ``hostAliases`` names resolve)."""
    spec: dict[str, Any] = {}
    if runtime_class:
        spec["runtimeClassName"] = runtime_class
    if service_account:
        spec["serviceAccountName"] = service_account
    spec["automountServiceAccountToken"] = automount_token
    spec["enableServiceLinks"] = enable_service_links
    if restart_policy:
        spec["restartPolicy"] = restart_policy
    if termination_grace_s is not None:
        spec["terminationGracePeriodSeconds"] = termination_grace_s
    if active_deadline_s is not None:
        spec["activeDeadlineSeconds"] = active_deadline_s
    if node_selector:
        spec["nodeSelector"] = dict(node_selector)
    if tolerations:
        spec["tolerations"] = list(tolerations)
    if host_aliases:
        spec["hostAliases"] = host_aliases
    if dns_none:
        spec["dnsPolicy"] = "None"
        spec["dnsConfig"] = {"nameservers": ["127.0.0.1"], "options": [{"name": "ndots", "value": "1"}]}
    spec["securityContext"] = security_context if security_context is not None else pod_security_context()
    if init_containers:
        spec["initContainers"] = init_containers
    spec["containers"] = containers
    if volumes:
        spec["volumes"] = volumes
    return spec


def pod_template(spec: dict, labels: dict | None = None, annotations: dict | None = None) -> dict:
    meta: dict[str, Any] = {}
    if labels:
        meta["labels"] = dict(labels)
    if annotations:
        meta["annotations"] = dict(annotations)
    return {"metadata": meta, "spec": spec}


def pod(name: str, namespace: str, spec: dict, labels: dict | None = None) -> dict:
    return obj("Pod", name, namespace, labels=labels, spec=spec)


def job(name: str, namespace: str, template: dict, *, backoff_limit: int = 0, active_deadline_s: int | None = None,
        ttl_after_finished_s: int | None = None, labels: dict | None = None, annotations: dict | None = None,
        pod_failure_policy: dict | None = None) -> dict:
    """One execution = one Job. ``backoffLimit`` defaults to 6 upstream: model-generated code is
    not idempotent, so it is 0 here. ``activeDeadlineSeconds`` bounds the whole Job (start-up
    included) and wins over ``backoffLimit``; ``ttlSecondsAfterFinished`` deletes the Job *and its
    logs*, so collect the result first."""
    spec: dict[str, Any] = {"backoffLimit": backoff_limit}
    if active_deadline_s is not None:
        spec["activeDeadlineSeconds"] = active_deadline_s
    if ttl_after_finished_s is not None:
        spec["ttlSecondsAfterFinished"] = ttl_after_finished_s
    if pod_failure_policy:
        spec["podFailurePolicy"] = pod_failure_policy
    spec["template"] = template
    return obj("Job", name, namespace, labels=labels, annotations=annotations, spec=spec)


def deployment(name: str, namespace: str, template: dict, *, replicas: int = 1, labels: dict | None = None) -> dict:
    sel = dict((template.get("metadata") or {}).get("labels") or {"app": name})
    return obj("Deployment", name, namespace, labels=labels,
               spec={"replicas": replicas, "selector": {"matchLabels": sel}, "template": template})


def service(name: str, namespace: str, selector: dict, port: int, target_port: int | None = None, *,
            cluster_ip: str | None = None) -> dict:
    spec: dict[str, Any] = {}
    if cluster_ip:
        spec["clusterIP"] = cluster_ip
    spec["selector"] = dict(selector)
    spec["ports"] = [{"name": "http", "port": port, "targetPort": target_port or port, "protocol": "TCP"}]
    return obj("Service", name, namespace, spec=spec)


def config_map(name: str, namespace: str, data: dict[str, str], labels: dict | None = None) -> dict:
    return obj("ConfigMap", name, namespace, labels=labels, data=data)


# ---- runtime, network, quota -----------------------------------------------------------------------
def runtime_class(name: str, handler: str, *, node_selector: dict | None = None, tolerations: list[dict] | None = None,
                  overhead: dict | None = None, labels: dict | None = None) -> dict:
    """``handler`` names the CRI runtime handler on the node (``runsc`` for a self-managed gVisor,
    ``runc`` for the default). ``scheduling`` is merged into every pod that names this class, so
    pods do not need to know which node pool runs the sandbox runtime; ``overhead.podFixed`` is
    added to the pod's requests by the scheduler, ResourceQuota and the pod cgroup."""
    out = obj("RuntimeClass", name, labels=labels, handler=handler)
    sched: dict[str, Any] = {}
    if node_selector:
        sched["nodeSelector"] = dict(node_selector)
    if tolerations:
        sched["tolerations"] = list(tolerations)
    if sched:
        out["scheduling"] = sched
    if overhead:
        out["overhead"] = {"podFixed": dict(overhead)}
    return out


def network_policy(name: str, namespace: str, *, pod_selector: dict | None = None, policy_types: list[str],
                   ingress: list[dict] | None = None, egress: list[dict] | None = None) -> dict:
    """Policy types are always explicit: omitted, ``Egress`` is only implied when egress rules exist,
    which silently turns a 'deny all egress' policy with no rules into 'deny nothing'."""
    spec: dict[str, Any] = {"podSelector": pod_selector or {}, "policyTypes": policy_types}
    if ingress is not None:
        spec["ingress"] = ingress
    if egress is not None:
        spec["egress"] = egress
    return obj("NetworkPolicy", name, namespace, spec=spec)


def peer(namespace: str | None = None, pod_labels: dict | None = None) -> dict:
    """One ``to``/``from`` entry. Namespace and pod selectors in *one* entry are ANDed (these pods in
    that namespace); separate entries are ORed — the most common NetworkPolicy mistake."""
    p: dict[str, Any] = {}
    if namespace:
        p["namespaceSelector"] = {"matchLabels": {NAME_LABEL: namespace}}
    if pod_labels:
        p["podSelector"] = {"matchLabels": dict(pod_labels)}
    return p


def ports(*items: tuple[str, int]) -> list[dict]:
    return [{"protocol": proto, "port": port} for proto, port in items]


def resource_quota(name: str, namespace: str, hard: dict, scopes: list[str] | None = None) -> dict:
    spec: dict[str, Any] = {"hard": dict(hard)}
    if scopes:
        spec["scopes"] = scopes
    return obj("ResourceQuota", name, namespace, spec=spec)


def limit_range(name: str, namespace: str, *, default: dict, default_request: dict, max_: dict | None = None,
                min_: dict | None = None) -> dict:
    """Defaults so that a quota on cpu/memory does not reject every pod that forgot its requests;
    ``max`` so that no single sandbox asks for the whole quota. Applied at admission only."""
    item: dict[str, Any] = {"type": "Container", "default": dict(default), "defaultRequest": dict(default_request)}
    if max_:
        item["max"] = dict(max_)
    if min_:
        item["min"] = dict(min_)
    return obj("LimitRange", name, namespace, spec={"limits": [item]})


# ---- admission policy ----------------------------------------------------------------------------------
def validating_admission_policy(name: str, *, resources: list[str], validations: list[dict],
                                variables: list[dict] | None = None, api_groups: list[str] | None = None,
                                operations: tuple[str, ...] = ("CREATE", "UPDATE"),
                                match_conditions: list[dict] | None = None, failure_policy: str = "Fail") -> dict:
    spec: dict[str, Any] = {
        "failurePolicy": failure_policy,
        "matchConstraints": {"resourceRules": [{"apiGroups": api_groups or [""], "apiVersions": ["v1"],
                                                "operations": list(operations), "resources": resources}]},
    }
    if match_conditions:
        spec["matchConditions"] = match_conditions
    if variables:
        spec["variables"] = variables
    spec["validations"] = validations
    return obj("ValidatingAdmissionPolicy", name, spec=spec)


def vap_binding(name: str, policy: str, *, namespaces: list[str], actions: tuple[str, ...] = ("Deny",)) -> dict:
    """Bind to namespaces by their immutable name label. ``validationActions``: Deny rejects,
    Audit annotates the audit log, Warn returns a warning; Deny and Warn together are invalid."""
    if "Deny" in actions and "Warn" in actions:
        raise ValueError("validationActions cannot contain both Deny and Warn")
    sel = {"matchExpressions": [{"key": NAME_LABEL, "operator": "In", "values": list(namespaces)}]}
    return obj("ValidatingAdmissionPolicyBinding", name,
               spec={"policyName": policy, "validationActions": list(actions),
                     "matchResources": {"namespaceSelector": sel}})


# ---- agent-sandbox (kubernetes-sigs/agent-sandbox) -----------------------------------------------------
def sandbox_template(name: str, namespace: str, pod_template_: dict, *, network_policy_egress: list[dict] | None = None) -> dict:
    """A SandboxTemplate: the pod template a warm pool stamps out. An empty egress list in the
    template's managed NetworkPolicy means default deny (sandboxtemplate_types.go)."""
    spec: dict[str, Any] = {"podTemplate": pod_template_}
    if network_policy_egress is not None:
        spec["networkPolicyManagement"] = "Managed"
        spec["networkPolicy"] = {"egress": network_policy_egress}
    return obj("SandboxTemplate", name, namespace, spec=spec)


def sandbox_warm_pool(name: str, namespace: str, template: str, replicas: int) -> dict:
    return obj("SandboxWarmPool", name, namespace, spec={"replicas": replicas, "sandboxTemplateRef": {"name": template}})


def sandbox_claim(name: str, namespace: str, warm_pool: str) -> dict:
    return obj("SandboxClaim", name, namespace, spec={"warmPoolRef": {"name": warm_pool}})


# ---- YAML in and out --------------------------------------------------------------------------------
class _Dumper(yaml.SafeDumper):
    """Block-style YAML that keeps insertion order and indents lists under their key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: D401
        return super().increase_indent(flow, False)


def _str_presenter(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_presenter)


def to_yaml(*objs: dict, header: str | None = None) -> str:
    """Multi-document YAML; ``header`` becomes a ``#`` comment block at the top."""
    docs = [yaml.dump(o, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=110).rstrip()
            for o in objs]
    text = "\n---\n".join(docs) + "\n"
    if header:
        text = "".join(f"# {line}".rstrip() + "\n" for line in header.strip().splitlines()) + text
    return text


def loads_all(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if d]


def load_all(path: str | Path) -> list[dict]:
    return loads_all(Path(path).read_text())


def iter_pod_templates(o: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(path, podTemplateSpec)`` for every pod template inside a workload object."""
    kind = o.get("kind")
    spec = o.get("spec") or {}
    if kind == "Pod":
        yield "spec", {"metadata": o.get("metadata", {}), "spec": spec}
    elif kind in ("Job", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"):
        if "template" in spec:
            yield "spec.template", spec["template"]
    elif kind in ("Sandbox", "SandboxTemplate"):
        if "podTemplate" in spec:
            yield "spec.podTemplate", spec["podTemplate"]
