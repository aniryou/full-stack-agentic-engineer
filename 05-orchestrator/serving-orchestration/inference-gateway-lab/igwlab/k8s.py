"""The Kubernetes-native serving stack as objects: build the manifests, then check them offline.

The one idea: in Gateway mode the routing policy you exercised in-process becomes a handful of
declarative objects, each owned by a different persona:

    Gateway (platform)  --HTTPRoute (app team)-->  InferencePool (inference platform)
                                                     selector -> model-server pods (targetPorts)
                                                     endpointPickerRef -> EPP Service :9002 (ext-proc)
    InferenceObjective (workload owner): priority of a class of requests in that pool
    PodMonitoring / HPA (operator): scrape vLLM metrics, scale the Deployment on them

Kubernetes validates CRD objects against the CRD's openAPIV3Schema at apply time; `check()`
does the same offline against schema snapshots in igwlab/crds/ (upstream CRDs, descriptions
stripped): types, required fields, enums, patterns, bounds and — like `kubectl apply` with
strict field validation — unknown fields. CEL rules are not evaluated generically; the two
InferencePool rules are checked by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

CRD_DIR = Path(__file__).resolve().parent / "crds"

# apiVersion/kind pairs this lab emits (FACTS, Sep 2026)
INFERENCE_POOL = ("inference.networking.k8s.io/v1", "InferencePool")
INFERENCE_OBJECTIVE = ("llm-d.ai/v1alpha2", "InferenceObjective")
GATEWAY = ("gateway.networking.k8s.io/v1", "Gateway")
HTTP_ROUTE = ("gateway.networking.k8s.io/v1", "HTTPRoute")
POD_MONITORING = ("monitoring.googleapis.com/v1", "PodMonitoring")
GKE_GATEWAY_CLASSES = ("gke-l7-regional-external-managed", "gke-l7-rilb")   # (verify) regional external / internal

__all__ = ["inference_pool", "inference_objective", "gateway", "http_route", "pod_monitoring", "to_yaml",
           "load_schemas", "validate", "check", "check_file", "INFERENCE_POOL", "INFERENCE_OBJECTIVE",
           "GATEWAY", "HTTP_ROUTE", "POD_MONITORING", "GKE_GATEWAY_CLASSES"]


def _meta(name, namespace=None, labels=None):
    m = {"name": name}
    if namespace:
        m["namespace"] = namespace
    if labels:
        m["labels"] = dict(labels)
    return m


# ------------------------------------------------------------------ builders
def inference_pool(name: str, match_labels: dict, target_ports=(8000,), epp_service: str | None = None,
                   epp_port: int = 9002, failure_mode: str = "FailClose", app_protocol: str | None = None,
                   namespace: str | None = None) -> dict:
    """InferencePool v1: which pods serve one base model, and which EPP picks among them."""
    spec = {"selector": {"matchLabels": dict(match_labels)},
            "targetPorts": [{"number": int(p)} for p in target_ports],
            "endpointPickerRef": {"name": epp_service or f"{name}-epp", "port": {"number": int(epp_port)},
                                  "failureMode": failure_mode}}
    if app_protocol:
        spec["appProtocol"] = app_protocol
    return {"apiVersion": INFERENCE_POOL[0], "kind": INFERENCE_POOL[1], "metadata": _meta(name, namespace), "spec": spec}


def inference_objective(name: str, pool: str, priority: int | None = None, namespace: str | None = None) -> dict:
    """InferenceObjective (llm-d.ai/v1alpha2): a request class in a pool; higher priority is served
    first under contention; negative priorities are sheddable. Requests select one with the
    header `x-llm-d-inference-objective: <name>`."""
    spec = {"poolRef": {"group": "inference.networking.k8s.io", "name": pool}}
    if priority is not None:
        spec["priority"] = int(priority)
    return {"apiVersion": INFERENCE_OBJECTIVE[0], "kind": INFERENCE_OBJECTIVE[1], "metadata": _meta(name, namespace),
            "spec": spec}


def gateway(name: str, gateway_class: str = GKE_GATEWAY_CLASSES[0], port: int = 80, namespace: str | None = None) -> dict:
    return {"apiVersion": GATEWAY[0], "kind": GATEWAY[1], "metadata": _meta(name, namespace),
            "spec": {"gatewayClassName": gateway_class,
                     "listeners": [{"name": "http", "protocol": "HTTP", "port": int(port)}]}}


def http_route(name: str, gateway_name: str, pool: str, path_prefix: str = "/", timeout: str | None = None,
               namespace: str | None = None) -> dict:
    """HTTPRoute sending `path_prefix` to an InferencePool (instead of a Service)."""
    rule = {"matches": [{"path": {"type": "PathPrefix", "value": path_prefix}}],
            "backendRefs": [{"group": "inference.networking.k8s.io", "kind": "InferencePool", "name": pool}]}
    if timeout:
        rule["timeouts"] = {"request": timeout}
    return {"apiVersion": HTTP_ROUTE[0], "kind": HTTP_ROUTE[1], "metadata": _meta(name, namespace),
            "spec": {"parentRefs": [{"name": gateway_name}], "rules": [rule]}}


def pod_monitoring(name: str, match_labels: dict, port, path: str = "/metrics", interval: str = "15s",
                   namespace: str | None = None) -> dict:
    """Google Cloud Managed Service for Prometheus scrape config (monitoring.googleapis.com/v1)."""
    return {"apiVersion": POD_MONITORING[0], "kind": POD_MONITORING[1], "metadata": _meta(name, namespace),
            "spec": {"selector": {"matchLabels": dict(match_labels)},
                     "endpoints": [{"port": port, "path": path, "interval": interval}]}}


def to_yaml(*objs) -> str:
    return "---\n".join(yaml.safe_dump(o, sort_keys=False) for o in objs)


# ------------------------------------------------------------------ validation
def load_schemas(directory: Path = CRD_DIR) -> dict:
    """(apiVersion, kind) -> openAPIV3Schema for every snapshot in `directory`."""
    out = {}
    for p in sorted(Path(directory).glob("*.json")):
        d = json.loads(p.read_text())
        out[(f"{d['group']}/{d['version']}", d["kind"])] = d["schema"]
    return out


def _type_ok(v, t) -> bool:
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
            "boolean": isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool)}.get(t, True)


def validate(obj, schema: dict, path: str = "") -> list[str]:
    """Structural validation of `obj` against an openAPIV3Schema node; returns error strings."""
    errs: list[str] = []
    here = path or "<root>"
    if obj is None:
        return [] if schema.get("nullable") else [f"{here}: null not allowed"]
    if "anyOf" in schema:
        if not any(not validate(obj, s, path) for s in schema["anyOf"]):
            if not (schema.get("x-kubernetes-int-or-string") and isinstance(obj, (int, str)) and not isinstance(obj, bool)):
                errs.append(f"{here}: matches none of anyOf")
    if schema.get("x-kubernetes-int-or-string"):
        if not (isinstance(obj, (int, str)) and not isinstance(obj, bool)):
            errs.append(f"{here}: must be an integer or a string")
    elif "type" in schema and not _type_ok(obj, schema["type"]):
        return errs + [f"{here}: expected {schema['type']}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{here}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, str):
        if "pattern" in schema and not re.search(schema["pattern"], obj):
            errs.append(f"{here}: {obj!r} does not match {schema['pattern']}")
        if len(obj) > schema.get("maxLength", 1 << 30) or len(obj) < schema.get("minLength", 0):
            errs.append(f"{here}: length {len(obj)} out of bounds")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{here}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{here}: {obj} > maximum {schema['maximum']}")
    if isinstance(obj, list):
        if len(obj) < schema.get("minItems", 0) or len(obj) > schema.get("maxItems", 1 << 30):
            errs.append(f"{here}: {len(obj)} items out of bounds [{schema.get('minItems', 0)}, {schema.get('maxItems', 'inf')}]")
        if "items" in schema:
            for i, it in enumerate(obj):
                errs += validate(it, schema["items"], f"{path}[{i}]")
    if isinstance(obj, dict):
        if len(obj) < schema.get("minProperties", 0) or len(obj) > schema.get("maxProperties", 1 << 30):
            errs.append(f"{here}: {len(obj)} properties out of bounds")
        for r in schema.get("required", []):
            if r not in obj:
                errs.append(f"{here}: missing required field {r!r}")
        props = schema.get("properties")
        addl = schema.get("additionalProperties")
        for k, v in obj.items():
            sub = f"{path}.{k}" if path else k
            if props and k in props:
                errs += validate(v, props[k], sub)
            elif isinstance(addl, dict):
                errs += validate(v, addl, sub)
            elif props is not None and not schema.get("x-kubernetes-preserve-unknown-fields") and addl is not True:
                errs.append(f"{sub}: unknown field")
    return errs


def _inference_pool_cel(obj) -> list[str]:
    errs = []
    spec = obj.get("spec", {})
    ref = spec.get("endpointPickerRef") or {}
    if ref and ref.get("kind", "Service") == "Service" and "port" not in ref:
        errs.append("spec.endpointPickerRef: port is required when kind is 'Service' or unspecified")
    nums = [p.get("number") for p in spec.get("targetPorts", [])]
    if len(nums) != len(set(nums)):
        errs.append("spec.targetPorts: port number must be unique")
    return errs


def check(obj: dict, schemas: dict | None = None) -> list[str]:
    """Validate one manifest. Kinds without a bundled schema return ['no schema ...']."""
    schemas = schemas if schemas is not None else load_schemas()
    key = (obj.get("apiVersion"), obj.get("kind"))
    if key not in schemas:
        return [f"no schema for {key}"]
    errs = validate(obj, schemas[key])
    if not obj.get("metadata", {}).get("name"):
        errs.append("metadata.name is required")
    if key == INFERENCE_POOL:
        errs += _inference_pool_cel(obj)
    return errs


def check_file(path, schemas: dict | None = None) -> dict:
    """{(kind, name): [errors]} for every document in a YAML file that has a bundled schema."""
    schemas = schemas if schemas is not None else load_schemas()
    out = {}
    for doc in yaml.safe_load_all(Path(path).read_text()):
        if not doc:
            continue
        if (doc.get("apiVersion"), doc.get("kind")) in schemas:
            out[(doc["kind"], doc.get("metadata", {}).get("name"))] = check(doc, schemas)
    return out
