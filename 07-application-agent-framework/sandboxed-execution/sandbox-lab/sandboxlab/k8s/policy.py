"""policy.py — one sandbox policy, as data, rendered into every object that enforces it.

One idea: "sandbox pods run as 65534 under gVisor, reach only the egress proxy, and get 500m CPU
for at most two minutes" is a *policy*; the Pod securityContext, the RuntimeClass, the
NetworkPolicies, the ResourceQuota/LimitRange, the ValidatingAdmissionPolicy and the proxy's
allowlist are its *enforcement points*. Writing the policy once and rendering all of them from it
keeps the enforcement points from drifting apart (PRIMER §3 "The execution contract" and
§5 "Sandboxes on Kubernetes"). ``render_all(target)`` returns ``{filename: (header, objects)}``;
``render.py`` writes them under ``deploy/kind/`` and ``deploy/gke/``.

Two targets share everything except what the platform decides:

| | kind (T0 + Docker) | GKE Sandbox (T3) |
|---|---|---|
| RuntimeClass | ``sandbox-runc`` (handler ``runc``) pinned to a tainted worker | ``gvisor`` (created by GKE with the sandbox node pool) |
| kernel isolation | none: runc, the host kernel | gVisor's Sentry |
| NetworkPolicy | kindnetd (fails open) | Dataplane V2 |
| images | Docker Hub ``python:3.12-slim`` | Artifact Registry copy (private nodes, no NAT) |
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import admission
from . import manifests as m

PKG = Path(__file__).resolve().parents[1]
WRAPPER_SRC = PKG / "wrapper.py"
PROXY_SRC = PKG / "proxy" / "server.py"
STUB_SRC = PKG / "proxy" / "stub.py"

SANDBOX_NS, EGRESS_NS, UPSTREAM_NS, CONTROL_NS, DEMO_NS = (
    "sandbox", "sandbox-egress", "sandbox-upstream", "sandbox-control", "sandbox-demo")
POOL_LABEL = "sandboxlab/pool"
EXEC_ID_LABEL = "sandboxlab/execution-id"
IDEMPOTENCY_ANNOTATION = "sandboxlab/idempotency-key"
STATE_LABEL = "sandboxlab/state"
PROXY_PORT, STUB_PORT = 8080, 8081
AR_PLACEHOLDER = "LOCATION-docker.pkg.dev/PROJECT_ID/sandbox"
GKE_SANDBOX_TAINT = {"key": "sandbox.gke.io/runtime", "operator": "Equal", "value": "gvisor", "effect": "NoSchedule"}


@dataclass(frozen=True)
class SandboxPolicy:
    target: str = "kind"                          # "kind" | "gke"
    runtime_class: str = "sandbox-runc"
    runtime_handler: str = "runc"
    image: str = "python:3.12-slim"
    uid: int = 65534
    cpu: str = "500m"
    memory: str = "320Mi"                         # the wrapper's RLIMIT_AS (256 MiB) + the wrapper itself
    ephemeral: str = "128Mi"
    workspace: str = "64Mi"                       # emptyDir /work sizeLimit
    tmp: str = "16Mi"
    budgets: dict = field(default_factory=lambda: {"cpu_s": 2.0, "wall_s": 10.0, "memory_mib": 256,
                                                  "file_mib": 8, "nofile": 64, "output_bytes": 65536,
                                                  "output_kill_bytes": 1048576})
    job_deadline_s: int = 120                     # whole Job: scheduling + image pull + run
    job_ttl_s: int = 300                          # collect logs before this
    max_deadline_s: int = 600
    proxy_cluster_ip: str = "10.96.0.200"         # inside the Service CIDR; sandboxes reach it via hostAliases
    allow_internet_from_proxy: bool = False
    pod_pids_limit: int = 128                     # kubelet podPidsLimit (kind patch / GKE node_config)
    warm_replicas: int = 2
    quota_pods: int = 20
    node_pool_selector: dict = field(default_factory=lambda: {POOL_LABEL: "sandbox"})
    node_pool_toleration: dict = field(default_factory=lambda: {"key": POOL_LABEL, "operator": "Equal",
                                                               "value": "sandbox", "effect": "NoSchedule"})

    @classmethod
    def kind(cls, **kw) -> "SandboxPolicy":
        return cls(**kw)

    @classmethod
    def gke(cls, **kw) -> "SandboxPolicy":
        base = dict(target="gke", runtime_class="gvisor", runtime_handler="gvisor",
                    image=f"{AR_PLACEHOLDER}/python:3.12-slim", proxy_cluster_ip="10.30.0.200",
                    node_pool_selector={"sandbox.gke.io/runtime": "gvisor"}, node_pool_toleration=GKE_SANDBOX_TAINT)
        base.update(kw)
        return cls(**base)

    def with_(self, **kw) -> "SandboxPolicy":
        return replace(self, **kw)

    @property
    def allowed_runtime_classes(self) -> list[str]:
        return [self.runtime_class]


# ---- the sandbox pod ----------------------------------------------------------------------------------------
def sandbox_labels(extra: dict | None = None) -> dict:
    return {"app.kubernetes.io/name": "sandbox", "app.kubernetes.io/part-of": "sandboxlab", **(extra or {})}


def wrapper_command(p: SandboxPolicy, source: str) -> list[str]:
    """``source``: ``--code-env SANDBOX_CODE`` (Job) or ``--stdin`` (``kubectl exec`` into a warm pod)."""
    return ["python3", "-I", "/opt/sandbox/wrapper.py", *source.split(), "--workspace", "/work",
            "--budgets", json.dumps(p.budgets, separators=(",", ":")), "--pass-env", "SANDBOX_PROXY_URL"]


def sandbox_pod_spec(p: SandboxPolicy, *, code: str | None = None, command: list[str] | None = None,
                     restart_policy: str = "Never") -> dict:
    env = {"SANDBOX_PROXY_URL": f"http://egress-proxy:{PROXY_PORT}"}
    if code is not None:
        env = {"SANDBOX_CODE": code, **env}
    c = m.container("run", p.image, command=command or wrapper_command(p, "--code-env SANDBOX_CODE"), env=env,
                    res=m.resources(p.cpu, p.memory, p.ephemeral), working_dir="/work",
                    volume_mounts=[{"name": "work", "mountPath": "/work"}, {"name": "tmp", "mountPath": "/tmp"},
                                   {"name": "wrapper", "mountPath": "/opt/sandbox", "readOnly": True}])
    return m.pod_spec([c], runtime_class=p.runtime_class, service_account="sandbox-exec", automount_token=False,
                      security_context=m.pod_security_context(p.uid), restart_policy=restart_policy,
                      termination_grace_s=0,
                      host_aliases=[{"ip": p.proxy_cluster_ip, "hostnames": ["egress-proxy"]}], dns_none=True,
                      volumes=[m.empty_dir("work", p.workspace), m.empty_dir("tmp", p.tmp),
                               m.config_map_volume("wrapper", "sandbox-wrapper")])


def run_code_job(p: SandboxPolicy, code: str, execution_id: str, idempotency_key: str | None = None) -> dict:
    """One execution as one Job. The Job's *name* is derived from the idempotency key, so a retried
    request collides with ``AlreadyExists`` instead of running the code twice."""
    labels = sandbox_labels({EXEC_ID_LABEL: execution_id, "sandboxlab/mode": "one-shot"})
    tmpl = m.pod_template(sandbox_pod_spec(p, code=code), labels=labels)
    return m.job(f"run-{execution_id}", SANDBOX_NS, tmpl, backoff_limit=0, active_deadline_s=p.job_deadline_s,
                 ttl_after_finished_s=p.job_ttl_s, labels=labels,
                 annotations={IDEMPOTENCY_ANNOTATION: idempotency_key} if idempotency_key else None)


def warm_pool_deployment(p: SandboxPolicy) -> dict:
    """Pods that are already scheduled, pulled and started, idling until ``kubectl exec`` hands one
    a piece of code; the runner deletes each pod after one use (no state carries over) and the
    Deployment replaces it off the critical path."""
    idle = ["python3", "-I", "-c", "import time\nwhile True:\n    time.sleep(3600)"]
    labels = sandbox_labels({"app": "sandbox-warm", STATE_LABEL: "idle"})
    spec = sandbox_pod_spec(p, command=idle, restart_policy="Always")
    tmpl = m.pod_template(spec, labels=labels)
    d = m.deployment("sandbox-warm", SANDBOX_NS, tmpl, replicas=p.warm_replicas, labels=sandbox_labels())
    d["spec"]["selector"] = {"matchLabels": {"app": "sandbox-warm"}}   # relabelling to "claimed" keeps ownership
    return d


# ---- namespace-level objects ---------------------------------------------------------------------------------
def namespaces(p: SandboxPolicy) -> list[dict]:
    rs = m.pss_labels("restricted")
    return [m.namespace(SANDBOX_NS, {**rs, "sandboxlab/role": "sandboxes"}),
            m.namespace(EGRESS_NS, {**rs, "sandboxlab/role": "egress"}),
            m.namespace(UPSTREAM_NS, {**rs, "sandboxlab/role": "upstream-stand-in"}),
            m.namespace(CONTROL_NS, {**rs, "sandboxlab/role": "agent-runtime"})]


def runtime_class(p: SandboxPolicy) -> dict:
    return m.runtime_class(p.runtime_class, p.runtime_handler, node_selector=dict(p.node_pool_selector),
                           tolerations=[dict(p.node_pool_toleration)], labels={"app.kubernetes.io/part-of": "sandboxlab"})


def rbac(p: SandboxPolicy) -> list[dict]:
    """Two identities. ``sandbox-exec`` runs the code and can do nothing (no Role, no token, and on
    GKE no IAM binding). ``agent-runner`` (the agent runtime, in its own namespace) may create Jobs,
    read their logs and exec into warm pods — in the sandbox namespace only."""
    rules = [m.rule(["batch"], ["jobs"], ["create", "get", "list", "watch", "delete"]),
             m.rule([""], ["pods"], ["get", "list", "watch", "patch", "delete"]),
             m.rule([""], ["pods/log"], ["get"]),
             m.rule([""], ["pods/exec"], ["create"])]
    return [m.service_account("sandbox-exec", SANDBOX_NS, automount=False),
            m.service_account("agent-runner", CONTROL_NS, automount=True),
            m.role("sandbox-runner", SANDBOX_NS, rules),
            m.role_binding("agent-runner", SANDBOX_NS, "sandbox-runner", "agent-runner", CONTROL_NS)]


def quota(p: SandboxPolicy) -> list[dict]:
    n = p.quota_pods
    per_cpu = admission.quantity(p.cpu)
    per_mem = admission.quantity(p.memory) / 2**20
    hard = {"pods": str(n), "requests.cpu": f"{per_cpu * n:g}", "limits.cpu": f"{per_cpu * n:g}",
            "requests.memory": f"{int(per_mem * n)}Mi", "limits.memory": f"{int(per_mem * n)}Mi",
            "requests.ephemeral-storage": "4Gi", "limits.ephemeral-storage": "4Gi", "count/jobs.batch": str(5 * n)}
    lr = m.limit_range("sandbox-defaults", SANDBOX_NS,
                       default={"cpu": p.cpu, "memory": p.memory, "ephemeral-storage": p.ephemeral},
                       default_request={"cpu": p.cpu, "memory": p.memory, "ephemeral-storage": p.ephemeral},
                       max_={"cpu": "1", "memory": "1Gi", "ephemeral-storage": "512Mi"})
    return [m.resource_quota("sandbox-quota", SANDBOX_NS, hard), lr]


def network_policies(p: SandboxPolicy) -> list[dict]:
    """Default deny everywhere, then three allows: sandbox -> proxy:8080, proxy -> upstream:8081
    (+ DNS for the proxy alone), and optionally proxy -> the internet on 443 (public ranges only)."""
    proxy_pods, stub_pods = {"app": "egress-proxy"}, {"app": "api-stub"}
    deny = lambda ns: m.network_policy("default-deny", ns, policy_types=["Ingress", "Egress"])  # noqa: E731
    out = [deny(SANDBOX_NS),
           m.network_policy("egress-to-proxy-only", SANDBOX_NS, policy_types=["Egress"],
                            egress=[{"to": [m.peer(EGRESS_NS, proxy_pods)], "ports": m.ports(("TCP", PROXY_PORT))}]),
           deny(EGRESS_NS),
           m.network_policy("proxy-from-sandboxes", EGRESS_NS, pod_selector={"matchLabels": proxy_pods},
                            policy_types=["Ingress"],
                            ingress=[{"from": [m.peer(SANDBOX_NS)], "ports": m.ports(("TCP", PROXY_PORT))}])]
    egress = [{"to": [m.peer(UPSTREAM_NS, stub_pods)], "ports": m.ports(("TCP", STUB_PORT))},
              {"to": [m.peer("kube-system", {"k8s-app": "kube-dns"})], "ports": m.ports(("UDP", 53), ("TCP", 53))}]
    if p.allow_internet_from_proxy:
        egress.append({"to": [{"ipBlock": {"cidr": "0.0.0.0/0",
                                           "except": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
                                                      "100.64.0.0/10"]}}],
                       "ports": m.ports(("TCP", 443))})
    out.append(m.network_policy("proxy-egress", EGRESS_NS, pod_selector={"matchLabels": proxy_pods},
                                policy_types=["Egress"], egress=egress))
    out += [deny(UPSTREAM_NS),
            m.network_policy("stub-from-proxy", UPSTREAM_NS, pod_selector={"matchLabels": stub_pods},
                             policy_types=["Ingress"],
                             ingress=[{"from": [m.peer(EGRESS_NS, proxy_pods)], "ports": m.ports(("TCP", STUB_PORT))}])]
    return out


def proxy_config(p: SandboxPolicy) -> dict:
    return {"routes": {"api-stub": {"upstream": f"http://api-stub.{UPSTREAM_NS}.svc.cluster.local:{STUB_PORT}",
                                    "methods": ["GET", "POST"],
                                    "inject": {"header": "Authorization", "value_from": "file:/etc/egress/token",
                                               "format": "Bearer {}"}}},
            "forward_allow": [], "allow_connect": False, "block_private": True,
            "max_request_bytes": 65536, "max_response_bytes": 1048576, "timeout_s": 10}


def _service_pod(name: str, ns: str, p: SandboxPolicy, *, args: list[str], code_cm: str, file: str,
                 port: int, secret: str, secret_mount: str, extra_volumes: list[dict] | None = None,
                 extra_mounts: list[dict] | None = None) -> dict:
    labels = {"app": name, "app.kubernetes.io/part-of": "sandboxlab"}
    c = m.container(name, p.image, command=["python3", "-I", f"/app/{file}", *args],
                    res=m.resources("100m", "128Mi", "64Mi"), ports=[{"name": "http", "containerPort": port}],
                    readiness_probe={"tcpSocket": {"port": port}, "periodSeconds": 5},
                    volume_mounts=[{"name": "code", "mountPath": "/app", "readOnly": True},
                                   {"name": "credential", "mountPath": secret_mount, "readOnly": True}] + (extra_mounts or []))
    spec = m.pod_spec([c], service_account=None, automount_token=False, security_context=m.pod_security_context(p.uid),
                      restart_policy="Always", termination_grace_s=5,
                      volumes=[m.config_map_volume("code", code_cm),
                               {"name": "credential", "secret": {"secretName": secret, "defaultMode": 0o440}}]
                      + (extra_volumes or []))
    return m.deployment(name, ns, m.pod_template(spec, labels=labels), replicas=1, labels=labels)


def egress_proxy(p: SandboxPolicy) -> list[dict]:
    """The proxy (its code and config from ConfigMaps, the credential from a Secret mounted into the
    proxy only), its Service at a fixed ClusterIP, and the stand-in upstream. The Secrets are
    created by the deploy scripts from a random value; they are never in these files."""
    proxy_dep = _service_pod("egress-proxy", EGRESS_NS, p,
                             args=["--config", "/etc/proxy/proxy.json", "--port", str(PROXY_PORT)],
                             code_cm="egress-proxy-code", file="server.py", port=PROXY_PORT,
                             secret="egress-credentials", secret_mount="/etc/egress",
                             extra_volumes=[m.config_map_volume("config", "egress-proxy-config")],
                             extra_mounts=[{"name": "config", "mountPath": "/etc/proxy", "readOnly": True}])
    proxy_dep["spec"]["template"]["spec"]["containers"][0]["readinessProbe"] = {
        "httpGet": {"path": "/healthz", "port": PROXY_PORT}, "periodSeconds": 5}
    stub_dep = _service_pod("api-stub", UPSTREAM_NS, p, args=["--port", str(STUB_PORT), "--token-file", "/etc/stub/token"],
                            code_cm="api-stub-code", file="stub.py", port=STUB_PORT, secret="api-stub-token",
                            secret_mount="/etc/stub")
    return [m.config_map("egress-proxy-code", EGRESS_NS, {"server.py": PROXY_SRC.read_text()}),
            m.config_map("egress-proxy-config", EGRESS_NS, {"proxy.json": json.dumps(proxy_config(p), indent=2) + "\n"}),
            proxy_dep,
            m.service("egress-proxy", EGRESS_NS, {"app": "egress-proxy"}, PROXY_PORT, cluster_ip=p.proxy_cluster_ip),
            m.config_map("api-stub-code", UPSTREAM_NS, {"stub.py": STUB_SRC.read_text()}),
            stub_dep,
            m.service("api-stub", UPSTREAM_NS, {"app": "api-stub"}, STUB_PORT)]


def admission_policies(p: SandboxPolicy) -> list[dict]:
    rs = admission.rules(p.allowed_runtime_classes, p.max_deadline_s)
    pod_rules = [{"expression": r.cel, "message": r.message, "reason": "Forbidden"} for r in rs if r.resource == "pods"]
    job_rules = [{"expression": r.cel, "message": r.message, "reason": "Forbidden"} for r in rs if r.resource == "jobs"]
    return [
        m.validating_admission_policy("sandboxlab-pod-hardening", resources=["pods", "pods/ephemeralcontainers"],
                                      variables=admission.CEL_VARIABLES, validations=pod_rules),
        m.vap_binding("sandboxlab-pod-hardening", "sandboxlab-pod-hardening", namespaces=[SANDBOX_NS]),
        m.validating_admission_policy("sandboxlab-job-budgets", resources=["jobs"], api_groups=["batch"],
                                      validations=job_rules, operations=("CREATE",)),
        m.vap_binding("sandboxlab-job-budgets", "sandboxlab-job-budgets", namespaces=[SANDBOX_NS]),
    ]


def wrapper_config_map(p: SandboxPolicy) -> dict:
    return m.config_map("sandbox-wrapper", SANDBOX_NS, {"wrapper.py": WRAPPER_SRC.read_text()})


# ---- examples that must fail --------------------------------------------------------------------------------
def rejected_pod(p: SandboxPolicy) -> dict:
    """A pod that 'just runs python': no securityContext, the default runtime, a token. PSA and the
    policy both reject it; the API server returns every violated message."""
    c = m.container("run", p.image, command=["python3", "-c", "print('hello')"], security_context={})
    del c["securityContext"]
    spec = {"restartPolicy": "Never", "containers": [c]}
    return m.pod("naive-run-code", SANDBOX_NS, spec, labels=sandbox_labels())


def rejected_job(p: SandboxPolicy) -> dict:
    """A well-formed sandbox pod template in a Job that forgot its budgets (no deadline, default
    retries, no TTL): the job policy rejects the Job itself at create time."""
    tmpl = m.pod_template(sandbox_pod_spec(p, code="print('hello')"), labels=sandbox_labels())
    job = m.job("unbounded-run", SANDBOX_NS, tmpl, labels=sandbox_labels())
    del job["spec"]["backoffLimit"]
    return job


def gvisor_in_kind() -> list[dict]:
    """kind has no gVisor: a RuntimeClass whose handler is ``runsc`` is accepted, and every pod that
    uses it ends ``Failed`` (the node's containerd has no such handler). In its own namespace, so
    the sandbox policy (which would reject the runtime class) does not hide the lesson."""
    rc = m.runtime_class("gvisor", "runsc", labels={"app.kubernetes.io/part-of": "sandboxlab"})
    c = m.container("probe", "busybox:1.38.0", command=["sh", "-c", "dmesg | head -1; uname -a"],
                    res=m.resources("50m", "32Mi"))
    spec = m.pod_spec([c], runtime_class="gvisor", automount_token=False, security_context=m.pod_security_context())
    return [m.namespace(DEMO_NS, m.pss_labels("restricted")), rc, m.pod("needs-gvisor", DEMO_NS, spec)]


# ---- agent-sandbox (optional, GKE) --------------------------------------------------------------------------
def agent_sandbox_objects(p: SandboxPolicy) -> list[dict]:
    """The same warm pod as a kubernetes-sigs/agent-sandbox SandboxTemplate + SandboxWarmPool, and a
    claim. The controller adopts a warm pod per claim; the namespace's own NetworkPolicies still apply
    (``networkPolicyManagement: Unmanaged``)."""
    warm = warm_pool_deployment(p)["spec"]["template"]
    warm["metadata"]["labels"] = sandbox_labels({"app": "agent-sandbox"})
    tmpl = m.sandbox_template("python-sandbox", SANDBOX_NS, warm)
    tmpl["spec"]["networkPolicyManagement"] = "Unmanaged"
    return [tmpl, m.sandbox_warm_pool("python-sandbox-pool", SANDBOX_NS, "python-sandbox", p.warm_replicas),
            m.sandbox_claim("example-claim", SANDBOX_NS, "python-sandbox-pool")]


# ---- kind's cluster config -----------------------------------------------------------------------------------
def kind_config(p: SandboxPolicy) -> dict:
    """1 control plane + 1 system worker + 1 sandbox worker (tainted by up.sh). The kubelet patch sets
    ``podPidsLimit`` — the only per-pod process limit Kubernetes has — and kubeadm reads it from the
    first node for all nodes."""
    return {"kind": "Cluster", "apiVersion": "kind.x-k8s.io/v1alpha4",
            "networking": {"serviceSubnet": "10.96.0.0/16"},
            "nodes": [{"role": "control-plane",
                       "kubeadmConfigPatches": [f"kind: KubeletConfiguration\npodPidsLimit: {p.pod_pids_limit}\n"]},
                      {"role": "worker", "labels": {POOL_LABEL: "system"}},
                      {"role": "worker", "labels": {POOL_LABEL: "sandbox"}}]}


def render_all(p: SandboxPolicy) -> dict[str, tuple[str, list[dict]]]:
    """``{relative file name: (header, objects)}`` for one target, in apply order."""
    files: dict[str, tuple[str, list[dict]]] = {
        "00-namespaces.yaml": ("Namespaces: sandboxes, the egress proxy, the stand-in upstream, the agent runtime. "
                               "Pod Security 'restricted' (enforce, audit, warn) on all four.", namespaces(p)),
        "10-rbac.yaml": ("Identities: sandbox-exec (runs the code; no Role, no token) and agent-runner "
                         "(creates Jobs, reads logs, execs into warm pods; sandbox namespace only).", rbac(p)),
        "20-quota.yaml": ("ResourceQuota and LimitRange for the sandbox namespace.", quota(p)),
        "30-network-policy.yaml": ("Default deny in every namespace; sandbox -> proxy:8080; proxy -> upstream:8081 and DNS.",
                                   network_policies(p)),
        "40-egress-proxy.yaml": ("The egress proxy and the stand-in upstream. Create the Secrets first "
                                 "(deploy scripts: egress-credentials and api-stub-token, one random value).",
                                 egress_proxy(p)),
        "50-admission-policy.yaml": ("ValidatingAdmissionPolicies: sandbox pod hardening, and budgets on sandbox Jobs.",
                                     admission_policies(p)),
        "60-wrapper.yaml": ("The execution-contract wrapper (sandboxlab/wrapper.py), mounted read-only into sandbox pods.",
                            [wrapper_config_map(p)]),
    }
    if p.target == "kind":
        files["05-runtimeclass.yaml"] = ("RuntimeClass sandbox-runc: handler runc (kind has no gVisor), pinned to the "
                                         "tainted sandbox worker through its scheduling field.", [runtime_class(p)])
    return dict(sorted(files.items()))


def workloads(p: SandboxPolicy) -> dict[str, tuple[str, list[dict]]]:
    example = ("import json, os, urllib.request\n"
               "print(sum(range(10)))\n"
               "print(json.load(urllib.request.urlopen(os.environ['SANDBOX_PROXY_URL'] + '/api-stub/whoami')))\n")
    return {
        "10-run-code-job.yaml": ("One execution as one Job (what the Runner creates). Its pod reaches only the proxy.",
                                 [run_code_job(p, example, "example-0001", "turn-1:step-1:call-0:9f2c")]),
        "20-warm-pool.yaml": ("A warm pool: idle sandbox pods for kubectl exec; each is deleted after one use.",
                              [warm_pool_deployment(p)]),
        "90-rejected-pod.yaml": ("MUST FAIL: a pod with no hardening. Pod Security and the policy both reject it.",
                                 [rejected_pod(p)]),
        "91-rejected-job.yaml": ("MUST FAIL: a Job without a deadline, with default retries and no TTL.",
                                 [rejected_job(p)]),
    }
