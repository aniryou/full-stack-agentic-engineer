"""Typed builders for the objects a GPU platform team writes every day.

The one idea: a GPU workload is ordinary Kubernetes plus a handful of load-bearing fields —
an integer ``nvidia.com/gpu`` limit, a toleration for the GPU taint, a node selector for the
accelerator, and (for gangs) a queue label and a topology annotation that Kueue reads. Each
builder here returns a plain ``dict`` that serialises to exactly the YAML you would apply, so
you can read the function and see every field that matters, and nothing else.

API versions are the ones pinned for this lab (Kubernetes 1.34, Kueue v0.19.6, JobSet v0.12.0,
LeaderWorkerSet v0.11.0); see ``API_VERSIONS``. GKE ``ComputeClass`` field names are marked
(verify) — that CRD is not published outside GKE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

# ---- API versions pinned for this lab (FACTS, Sep 2026) ----------------------------------
API_VERSIONS: dict[str, str] = {
    "Namespace": "v1",
    "Pod": "v1",
    "Job": "batch/v1",
    "Deployment": "apps/v1",
    "Service": "v1",
    "ServiceAccount": "v1",
    "PriorityClass": "scheduling.k8s.io/v1",
    "ResourceClaimTemplate": "resource.k8s.io/v1",   # DRA, GA in Kubernetes 1.34
    "DeviceClass": "resource.k8s.io/v1",
    "JobSet": "jobset.x-k8s.io/v1alpha2",
    "LeaderWorkerSet": "leaderworkerset.x-k8s.io/v1",
    "ResourceFlavor": "kueue.x-k8s.io/v1beta2",
    "ClusterQueue": "kueue.x-k8s.io/v1beta2",
    "LocalQueue": "kueue.x-k8s.io/v1beta2",
    "WorkloadPriorityClass": "kueue.x-k8s.io/v1beta2",
    "Topology": "kueue.x-k8s.io/v1beta2",
    "AdmissionCheck": "kueue.x-k8s.io/v1beta2",
    "ProvisioningRequestConfig": "kueue.x-k8s.io/v1beta2",
    "ComputeClass": "cloud.google.com/v1",            # GKE-only CRD (verify)
}

# ---- the names that carry meaning ---------------------------------------------------------
GPU = "nvidia.com/gpu"                                  # the device plugin's extended resource
GPU_TOLERATION = {"key": GPU, "operator": "Exists", "effect": "NoSchedule"}
GKE_ACCELERATOR = "cloud.google.com/gke-accelerator"   # GKE node label: nvidia-l4, nvidia-h100-80gb, ...
GKE_NODEPOOL = "cloud.google.com/gke-nodepool"
GKE_COMPUTE_CLASS = "cloud.google.com/compute-class"
TOPOLOGY_BLOCK = "cloud.google.com/gce-topology-block"       # (verify) GCE placement labels
TOPOLOGY_SUBBLOCK = "cloud.google.com/gce-topology-subblock"
TOPOLOGY_HOST = "cloud.google.com/gce-topology-host"
HOSTNAME = "kubernetes.io/hostname"
GKE_TOPOLOGY_LEVELS = [TOPOLOGY_BLOCK, TOPOLOGY_SUBBLOCK, TOPOLOGY_HOST, HOSTNAME]

QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
TAS_REQUIRED = "kueue.x-k8s.io/podset-required-topology"
TAS_PREFERRED = "kueue.x-k8s.io/podset-preferred-topology"
TAS_UNCONSTRAINED = "kueue.x-k8s.io/podset-unconstrained-topology"
TAS_GROUP = "kueue.x-k8s.io/podset-group-name"
PROVREQ_PREFIX = "provreq.kueue.x-k8s.io/"             # job annotations passed to the ProvisioningRequest

DEFAULT_IMAGE = "busybox:1.38.0"


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


# ---- containers and pods -----------------------------------------------------------------
@dataclass
class GPUContainer:
    """One container that asks for whole GPUs.

    ``gpus`` becomes an integer *limit* on ``nvidia.com/gpu`` (the API server fills the request
    from the limit; extended resources cannot be overcommitted, so request == limit always).
    CPU and memory get explicit requests; memory also gets a limit, CPU deliberately does not
    (a CPU limit throttles the data-loader or tokenizer that feeds the GPU).
    """
    name: str = "main"
    image: str = DEFAULT_IMAGE
    command: list[str] | None = None
    args: list[str] | None = None
    gpus: int = 1
    cpu: str = "100m"
    memory: str = "64Mi"
    env: dict[str, str] = field(default_factory=dict)
    env_from_field: dict[str, str] = field(default_factory=dict)   # NAME -> fieldPath
    ports: list[int] = field(default_factory=list)
    startup_probe: dict | None = None
    readiness_probe: dict | None = None
    liveness_probe: dict | None = None
    volume_mounts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        if int(self.gpus) != self.gpus or self.gpus < 0:
            raise ValueError(f"GPUs are whole devices: got {self.gpus!r}")
        c: dict[str, Any] = {"name": self.name, "image": self.image}
        if self.command:
            c["command"] = list(self.command)
        if self.args:
            c["args"] = list(self.args)
        env = [{"name": k, "value": v} for k, v in self.env.items()]
        env += [{"name": k, "valueFrom": {"fieldRef": {"fieldPath": p}}} for k, p in self.env_from_field.items()]
        if env:
            c["env"] = env
        if self.ports:
            c["ports"] = [{"containerPort": p} for p in self.ports]
        requests: dict[str, Any] = {"cpu": self.cpu, "memory": self.memory}
        limits: dict[str, Any] = {"memory": self.memory}
        if self.gpus:
            requests[GPU] = int(self.gpus)
            limits[GPU] = int(self.gpus)
        c["resources"] = {"requests": requests, "limits": limits}
        for key, probe in (("startupProbe", self.startup_probe), ("readinessProbe", self.readiness_probe),
                           ("livenessProbe", self.liveness_probe)):
            if probe:
                c[key] = probe
        if self.volume_mounts:
            c["volumeMounts"] = list(self.volume_mounts)
        return c


def http_probe(path: str, port: int, *, period_s: int = 10, failure_threshold: int = 3,
               initial_delay_s: int = 0, timeout_s: int = 1) -> dict:
    p: dict[str, Any] = {"httpGet": {"path": path, "port": port}, "periodSeconds": period_s,
                         "failureThreshold": failure_threshold}
    if initial_delay_s:
        p["initialDelaySeconds"] = initial_delay_s
    if timeout_s != 1:
        p["timeoutSeconds"] = timeout_s
    return p


def startup_probe_for(load_seconds: float, path: str = "/health", port: int = 8000, *,
                      period_s: int = 10, margin: float = 1.5) -> dict:
    """A startup probe that tolerates a model load of ``load_seconds`` (times a safety margin).

    Budget = failureThreshold x periodSeconds. A liveness probe does not run until the startup
    probe succeeds, so this is the knob that stops kubelet killing a pod mid-weight-load.
    """
    threshold = max(1, math.ceil(load_seconds * margin / period_s))
    return http_probe(path, port, period_s=period_s, failure_threshold=threshold)


def pod_spec(containers: list[GPUContainer | dict], *, accelerator: str | None = "nvidia-l4",
             tolerate_gpu: bool = True, node_selector: dict | None = None,
             restart_policy: str | None = "Never", termination_grace_s: int | None = 5,
             priority_class: str | None = None, volumes: list[dict] | None = None,
             shm_size: str | None = None, service_account: str | None = None,
             init_containers: list[dict] | None = None, resource_claims: list[dict] | None = None,
             tolerations: list[dict] | None = None) -> dict:
    """A pod spec with the GPU plumbing in place: toleration, accelerator selector, /dev/shm."""
    cs = [c.to_dict() if isinstance(c, GPUContainer) else dict(c) for c in containers]
    spec: dict[str, Any] = {}
    if restart_policy:
        spec["restartPolicy"] = restart_policy
    if termination_grace_s is not None:
        spec["terminationGracePeriodSeconds"] = termination_grace_s
    if priority_class:
        spec["priorityClassName"] = priority_class
    if service_account:
        spec["serviceAccountName"] = service_account
    selector = dict(node_selector or {})
    if accelerator:
        selector.setdefault(GKE_ACCELERATOR, accelerator)
    if selector:
        spec["nodeSelector"] = selector
    tols = [dict(GPU_TOLERATION)] if tolerate_gpu else []
    tols += list(tolerations or [])
    if tols:
        spec["tolerations"] = tols
    vols = list(volumes or [])
    if shm_size:   # PyTorch/NCCL/vLLM workers use /dev/shm; the container default is 64 MiB
        vols.append({"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": shm_size}})
        for c in cs:
            c.setdefault("volumeMounts", []).append({"name": "dshm", "mountPath": "/dev/shm"})
    if init_containers:
        spec["initContainers"] = list(init_containers)
    spec["containers"] = cs
    if vols:
        spec["volumes"] = vols
    if resource_claims:
        spec["resourceClaims"] = list(resource_claims)
    return spec


def topology_annotations(*, required: str | None = None, preferred: str | None = None,
                         unconstrained: bool = False, group: str | None = None) -> dict:
    """Kueue Topology-Aware Scheduling request for one PodSet (set on the pod template)."""
    if sum(bool(x) for x in (required, preferred, unconstrained)) > 1:
        raise ValueError("pick one of required / preferred / unconstrained")
    a: dict[str, str] = {}
    if required:
        a[TAS_REQUIRED] = required
    if preferred:
        a[TAS_PREFERRED] = preferred
    if unconstrained:
        a[TAS_UNCONSTRAINED] = "true"
    if group:
        a[TAS_GROUP] = group
    return a


def pod_template(spec: dict, *, labels: dict | None = None, annotations: dict | None = None) -> dict:
    t: dict[str, Any] = {}
    meta = {}
    if labels:
        meta["labels"] = dict(labels)
    if annotations:
        meta["annotations"] = dict(annotations)
    if meta:
        t["metadata"] = meta
    t["spec"] = spec
    return t


def _kueue_labels(queue: str | None, priority_class: str | None, extra: dict | None) -> dict:
    labels = dict(extra or {})
    if queue:
        labels[QUEUE_LABEL] = queue
    if priority_class:
        labels[PRIORITY_LABEL] = priority_class
    return labels


# ---- workloads ---------------------------------------------------------------------------
def job(name: str, namespace: str, template: dict, *, parallelism: int = 1, completions: int | None = None,
        queue: str | None = None, priority_class: str | None = None, backoff_limit: int = 0,
        indexed: bool = False, labels: dict | None = None, annotations: dict | None = None,
        active_deadline_s: int | None = None, ttl_after_finished_s: int | None = None) -> dict:
    """A batch/v1 Job. With ``queue`` set, Kueue suspends it at creation and unsuspends on admission.
    ``active_deadline_s`` also bounds time spent Pending — a cheap guard against a job that waits
    all weekend for a GPU that never comes."""
    spec: dict[str, Any] = {"parallelism": parallelism, "completions": completions or parallelism,
                            "backoffLimit": backoff_limit}
    if active_deadline_s:
        spec["activeDeadlineSeconds"] = active_deadline_s
    if ttl_after_finished_s is not None:
        spec["ttlSecondsAfterFinished"] = ttl_after_finished_s
    if indexed:
        spec["completionMode"] = "Indexed"
    spec["template"] = template
    return obj("Job", name, namespace, labels=_kueue_labels(queue, priority_class, labels) or None,
               annotations=annotations, spec=spec)


def replicated_job(name: str, template: dict, *, replicas: int = 1, parallelism: int = 1,
                   completions: int | None = None, backoff_limit: int = 0) -> dict:
    return {"name": name, "replicas": replicas,
            "template": {"spec": {"parallelism": parallelism, "completions": completions or parallelism,
                                  "backoffLimit": backoff_limit, "completionMode": "Indexed",
                                  "template": template}}}


def jobset(name: str, namespace: str, replicated_jobs: list[dict], *, queue: str | None = None,
           priority_class: str | None = None, max_restarts: int | None = None,
           labels: dict | None = None) -> dict:
    """A JobSet: several Jobs managed (and, through Kueue, admitted) as one unit."""
    spec: dict[str, Any] = {}
    if max_restarts is not None:
        spec["failurePolicy"] = {"maxRestarts": max_restarts}
    spec["replicatedJobs"] = replicated_jobs
    return obj("JobSet", name, namespace, labels=_kueue_labels(queue, priority_class, labels) or None,
               spec=spec)


def leader_worker_set(name: str, namespace: str, *, size: int, worker_template: dict,
                      leader_template: dict | None = None, replicas: int = 1, queue: str | None = None,
                      priority_class: str | None = None, labels: dict | None = None) -> dict:
    """A LeaderWorkerSet: ``replicas`` groups of ``size`` pods; each group is one multi-host replica.

    Kueue admits each group as its own Workload (all-or-nothing per group), so scaling up adds a
    whole group or nothing.
    """
    if len(name) > 39:   # Kueue appends a group suffix into a 63-char label (Kueue LWS docs)
        raise ValueError("keep LeaderWorkerSet names <= 39 characters when Kueue manages them")
    lwt: dict[str, Any] = {"size": size}
    if leader_template:
        lwt["leaderTemplate"] = leader_template
    lwt["workerTemplate"] = worker_template
    lwt["restartPolicy"] = "RecreateGroupOnPodRestart"
    return obj("LeaderWorkerSet", name, namespace, labels=_kueue_labels(queue, priority_class, labels) or None,
               spec={"replicas": replicas, "leaderWorkerTemplate": lwt})


# ---- Kueue --------------------------------------------------------------------------------
def topology(name: str, levels: list[str]) -> dict:
    if HOSTNAME in levels and levels[-1] != HOSTNAME:
        raise ValueError("kubernetes.io/hostname can only be the lowest topology level")
    return obj("Topology", name, spec={"levels": [{"nodeLabel": lv} for lv in levels]})


def resource_flavor(name: str, *, node_labels: dict | None = None, tolerations: list[dict] | None = None,
                    node_taints: list[dict] | None = None, topology_name: str | None = None) -> dict:
    """A ResourceFlavor maps quota to a set of nodes (by label). ``tolerations`` are injected into
    admitted pods; ``topology_name`` turns on Topology-Aware Scheduling for this flavor."""
    spec: dict[str, Any] = {}
    if node_labels:
        spec["nodeLabels"] = dict(node_labels)
    if node_taints:
        spec["nodeTaints"] = list(node_taints)
    if tolerations:
        spec["tolerations"] = list(tolerations)
    if topology_name:
        if not node_labels:
            raise ValueError("a TAS flavor needs nodeLabels (Kueue CRD validation)")
        spec["topologyName"] = topology_name
    return obj("ResourceFlavor", name, spec=spec)


def quota(nominal: Any, borrowing_limit: Any = None, lending_limit: Any = None) -> dict:
    q: dict[str, Any] = {"nominalQuota": nominal}
    if borrowing_limit is not None:
        q["borrowingLimit"] = borrowing_limit
    if lending_limit is not None:
        q["lendingLimit"] = lending_limit
    return q


def cluster_queue(name: str, *, flavors: dict[str, dict[str, Any]], cohort: str | None = None,
                  namespaces: list[str] | None = None, preemption: dict | None = None,
                  queueing_strategy: str | None = None,
                  admission_checks: list[str | tuple[str, list[str]]] | None = None) -> dict:
    """A ClusterQueue with one resource group.

    ``flavors`` maps flavor name -> {resource: nominal | quota(...)}; every flavor must list the
    same resources in the same order (Kueue validation), and those become ``coveredResources``.
    ``namespaces=None`` admits from every namespace (``namespaceSelector: {}``).
    """
    covered: list[str] | None = None
    fl = []
    for fname, resources in flavors.items():
        names = list(resources)
        if covered is None:
            covered = names
        elif names != covered:
            raise ValueError(f"flavor {fname} must list {covered} in the same order")
        fl.append({"name": fname, "resources": [
            {"name": r, **(q if isinstance(q, dict) else {"nominalQuota": q})} for r, q in resources.items()]})
        if cohort is None and any(isinstance(q, dict) and "borrowingLimit" in q for q in resources.values()):
            raise ValueError("borrowingLimit must be nil when the ClusterQueue has no cohort")
    spec: dict[str, Any] = {}
    if cohort:
        spec["cohortName"] = cohort   # v1beta2 name (v1beta1 called it `cohort`)
    if namespaces is None:
        spec["namespaceSelector"] = {}
    else:
        spec["namespaceSelector"] = {"matchExpressions": [
            {"key": "kubernetes.io/metadata.name", "operator": "In", "values": list(namespaces)}]}
    if queueing_strategy:
        spec["queueingStrategy"] = queueing_strategy
    if preemption:
        spec["preemption"] = preemption
    spec["resourceGroups"] = [{"coveredResources": covered or [], "flavors": fl}]
    if admission_checks:
        acs = []
        for ac in admission_checks:
            if isinstance(ac, tuple):
                acs.append({"name": ac[0], "onFlavors": list(ac[1])})
            else:
                acs.append({"name": ac})
        spec["admissionChecksStrategy"] = {"admissionChecks": acs}   # v1beta2 shape
    return obj("ClusterQueue", name, spec=spec)


def local_queue(name: str, namespace: str, cluster_queue_name: str) -> dict:
    return obj("LocalQueue", name, namespace, spec={"clusterQueue": cluster_queue_name})


def workload_priority_class(name: str, value: int, description: str = "") -> dict:
    o = obj("WorkloadPriorityClass", name, value=value)
    if description:
        o["description"] = description
    return o


def priority_class(name: str, value: int, description: str = "", *,
                   preemption_policy: str | None = None) -> dict:
    o = obj("PriorityClass", name, value=value, globalDefault=False)
    if description:
        o["description"] = description
    if preemption_policy:
        o["preemptionPolicy"] = preemption_policy
    return o


def admission_check(name: str, *, config_name: str,
                    controller: str = "kueue.x-k8s.io/provisioning-request") -> dict:
    return obj("AdmissionCheck", name, spec={
        "controllerName": controller,
        "parameters": {"apiGroup": "kueue.x-k8s.io", "kind": "ProvisioningRequestConfig", "name": config_name}})


def provisioning_request_config(name: str, *, provisioning_class: str = "queued-provisioning.gke.io",
                                managed_resources: list[str] | None = None,
                                node_selector_updates: dict[str, str] | None = None,
                                backoff_limit: int | None = None) -> dict:
    """How Kueue asks the cluster autoscaler for capacity before admitting (DWS on GKE)."""
    spec: dict[str, Any] = {"provisioningClassName": provisioning_class,
                            "managedResources": list(managed_resources or [GPU])}
    if node_selector_updates:
        spec["podSetUpdates"] = {"nodeSelector": [
            {"key": k, "valueFromProvisioningClassDetail": v} for k, v in node_selector_updates.items()]}
    if backoff_limit is not None:
        spec["retryStrategy"] = {"backoffLimitCount": backoff_limit}
    return obj("ProvisioningRequestConfig", name, spec=spec)


# ---- DRA and GKE ComputeClass --------------------------------------------------------------
def resource_claim_template(name: str, namespace: str, *, device_class: str = "gpu.nvidia.com",
                            count: int = 1, cel: list[str] | None = None, request_name: str = "gpu") -> dict:
    """DRA (resource.k8s.io/v1): 'give me ``count`` devices of this class matching these CEL
    expressions'. The pod references it in ``spec.resourceClaims`` and each container that uses
    the devices lists it in ``resources.claims``."""
    exactly: dict[str, Any] = {"deviceClassName": device_class, "allocationMode": "ExactCount", "count": count}
    if cel:
        exactly["selectors"] = [{"cel": {"expression": e}} for e in cel]
    return obj("ResourceClaimTemplate", name, namespace,
               spec={"spec": {"devices": {"requests": [{"name": request_name, "exactly": exactly}]}}})


def compute_class_priority(*, machine_type: str | None = None, machine_family: str | None = None,
                           gpu_type: str | None = None, gpu_count: int | None = None,
                           spot: bool | None = None, flex_start: bool = False,
                           reservation: str | None = None, capacity_check_wait_s: int | None = None,
                           node_recycling_lead_s: int | None = None) -> dict:
    """One rung of a GKE ComputeClass fallback ladder (field names: verify against your cluster's CRD)."""
    p: dict[str, Any] = {}
    if machine_family:
        p["machineFamily"] = machine_family
    if machine_type:
        p["machineType"] = machine_type
    if gpu_type:
        p["gpu"] = {"type": gpu_type, "count": gpu_count or 1}
    if spot is not None:
        p["spot"] = spot
    if flex_start:
        fs: dict[str, Any] = {"enabled": True}
        if node_recycling_lead_s:
            fs["nodeRecycling"] = {"leadTimeSeconds": node_recycling_lead_s}
        p["flexStart"] = fs
    if reservation:
        p["reservations"] = {"affinity": "Specific", "specific": [{"name": reservation}]}
    if capacity_check_wait_s is not None:
        p["capacityCheckWaitTimeSeconds"] = capacity_check_wait_s
    return p


def compute_class(name: str, priorities: list[dict], *, auto_create: bool = True,
                  when_unsatisfiable: str = "DoNotScaleUp", optimize_rule_priority: bool = True) -> dict:
    """A GKE custom ComputeClass: an ordered list of ways to get a node, tried top to bottom."""
    spec: dict[str, Any] = {"priorities": priorities}
    if auto_create:
        spec["nodePoolAutoCreation"] = {"enabled": True}
    if optimize_rule_priority:
        spec["activeMigration"] = {"optimizeRulePriority": True}
    spec["whenUnsatisfiable"] = when_unsatisfiable
    return obj("ComputeClass", name, spec=spec)


def namespace(name: str, labels: dict | None = None) -> dict:
    return obj("Namespace", name, labels=labels)


# ---- YAML in and out ------------------------------------------------------------------------
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
    """Parse YAML text into a list of objects, skipping empty documents."""
    return [d for d in yaml.safe_load_all(text) if d]


def load_all(path: str | Path) -> list[dict]:
    """Parse a YAML file into a list of objects, skipping empty documents."""
    return loads_all(Path(path).read_text())


# ---- walking pod templates -----------------------------------------------------------------
def iter_pod_templates(o: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(path, podTemplateSpec)`` for every pod template inside a workload object.

    Knows Pod, Job, Deployment/StatefulSet/DaemonSet/ReplicaSet, JobSet and LeaderWorkerSet —
    the shapes a GPU linter must look inside, because a CRD accepts an invalid pod template and
    the error only appears later, when its controller tries to create the pods.
    """
    kind = o.get("kind")
    spec = o.get("spec") or {}
    if kind == "Pod":
        yield "spec", {"metadata": o.get("metadata", {}), "spec": spec}
    elif kind in ("Job", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"):
        if "template" in spec:
            yield "spec.template", spec["template"]
    elif kind == "JobSet":
        for i, rj in enumerate(spec.get("replicatedJobs", [])):
            t = ((rj.get("template") or {}).get("spec") or {}).get("template")
            if t is not None:
                yield f"spec.replicatedJobs[{i}].template.spec.template", t
    elif kind == "LeaderWorkerSet":
        lwt = spec.get("leaderWorkerTemplate") or {}
        if lwt.get("leaderTemplate"):
            yield "spec.leaderWorkerTemplate.leaderTemplate", lwt["leaderTemplate"]
        if lwt.get("workerTemplate"):
            yield "spec.leaderWorkerTemplate.workerTemplate", lwt["workerTemplate"]


def gpu_count(container: dict) -> int:
    """GPUs a container asks for (limit wins; the API server copies it into requests)."""
    res = container.get("resources") or {}
    val = (res.get("limits") or {}).get(GPU, (res.get("requests") or {}).get(GPU, 0))
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
