#!/usr/bin/env python3
"""Validate every CRD-kind manifest under deploy/ against the pinned upstream CRD schemas.

    python3 tools/check_crds.py            # needs network once (raw.githubusercontent.com); cached after

Kueue, JobSet and LeaderWorkerSet publish their CRDs; this downloads the ones matching
deploy/versions.env, turns each CRD's openAPIV3Schema into a JSON Schema that rejects unknown
fields (what `kubectl apply` does with server-side field validation), and validates. CEL rules
(x-kubernetes-validations) are not evaluated here — the pytest suite checks the few that
matter. GKE's ComputeClass CRD is not published; tests check its field names instead.
Requires `jsonschema` (pip install -e ".[dev]").
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("K8SGPU_CRD_CACHE", Path.home() / ".cache" / "k8sgpu-crds"))


def versions() -> dict[str, str]:
    out = {}
    for line in (ROOT / "deploy" / "versions.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def crd_sources(v: dict[str, str]) -> dict[tuple[str, str], str]:
    kueue = f"https://raw.githubusercontent.com/kubernetes-sigs/kueue/{v['KUEUE_VERSION']}/config/components/crd/bases"
    srcs = {("kueue.x-k8s.io", k): f"{kueue}/kueue.x-k8s.io_{p}.yaml" for k, p in [
        ("ClusterQueue", "clusterqueues"), ("LocalQueue", "localqueues"), ("ResourceFlavor", "resourceflavors"),
        ("WorkloadPriorityClass", "workloadpriorityclasses"), ("Topology", "topologies"),
        ("AdmissionCheck", "admissionchecks"), ("ProvisioningRequestConfig", "provisioningrequestconfigs")]}
    srcs[("jobset.x-k8s.io", "JobSet")] = (f"https://raw.githubusercontent.com/kubernetes-sigs/jobset/{v['JOBSET_VERSION']}"
                                          "/config/components/crd/bases/jobset.x-k8s.io_jobsets.yaml")
    srcs[("leaderworkerset.x-k8s.io", "LeaderWorkerSet")] = (
        f"https://raw.githubusercontent.com/kubernetes-sigs/lws/{v['LWS_VERSION']}"
        "/config/crd/bases/leaderworkerset.x-k8s.io_leaderworkersets.yaml")
    return srcs


def fetch(url: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / url.replace("https://", "").replace("/", "_")
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310 - pinned https URL
            path.write_bytes(r.read())
    return yaml.safe_load(path.read_text())


def to_json_schema(s: dict) -> dict:
    """OpenAPI v3 (structural CRD schema) -> JSON Schema that rejects unknown fields."""
    if not isinstance(s, dict):
        return s
    out = {k: v for k, v in s.items() if not k.startswith("x-kubernetes") and k not in ("nullable", "format")}
    if "pattern" in out:   # CRD patterns are RE2; drop the few Python's re cannot compile
        try:
            re.compile(out["pattern"])
        except re.error:
            out.pop("pattern")
    if s.get("nullable"):
        t = out.get("type")
        if t:
            out["type"] = [t, "null"]
    if "properties" in s:
        out["properties"] = {k: to_json_schema(v) for k, v in s["properties"].items()}
        if not s.get("x-kubernetes-preserve-unknown-fields") and "additionalProperties" not in s:
            out["additionalProperties"] = False
    if isinstance(s.get("additionalProperties"), dict):
        out["additionalProperties"] = to_json_schema(s["additionalProperties"])
    if "items" in s:
        out["items"] = to_json_schema(s["items"])
    for key in ("anyOf", "allOf", "oneOf"):
        if key in s:
            out[key] = [to_json_schema(x) for x in s[key]]
    if s.get("x-kubernetes-int-or-string") and "anyOf" not in s:
        out["anyOf"] = [{"type": "integer"}, {"type": "string"}]
        out.pop("type", None)
    return out


def main() -> int:
    try:
        import jsonschema
    except ImportError:
        print("pip install jsonschema (or pip install -e '.[dev]')")
        return 2
    srcs = crd_sources(versions())
    schemas: dict[tuple[str, str, str], dict] = {}
    bad = checked = 0
    for f in sorted((ROOT / "deploy").rglob("*.yaml")):
        for doc in yaml.safe_load_all(f.read_text()):
            if not doc or "/" not in doc.get("apiVersion", ""):
                continue
            group, version = doc["apiVersion"].split("/")
            key = (group, doc["kind"])
            if key not in srcs:
                continue
            if (group, doc["kind"], version) not in schemas:
                crd = fetch(srcs[key])
                ver = next((x for x in crd["spec"]["versions"] if x["name"] == version), None)
                if ver is None or not ver.get("served"):
                    print(f"FAIL {f.relative_to(ROOT)}: {doc['apiVersion']} is not served by the pinned CRD")
                    bad += 1
                    continue
                schema = to_json_schema(ver["schema"]["openAPIV3Schema"])
                schema["properties"]["metadata"] = {"type": "object"}
                schemas[(group, doc["kind"], version)] = schema
            errors = sorted(jsonschema.Draft7Validator(schemas[(group, doc["kind"], version)]).iter_errors(doc),
                            key=lambda e: list(e.path))
            checked += 1
            name = f"{doc['kind']}/{doc['metadata'].get('name')}"
            if errors:
                bad += 1
                for e in errors[:5]:
                    print(f"FAIL {f.relative_to(ROOT)} {name}: {'.'.join(map(str, e.path))}: {e.message[:200]}")
            else:
                print(f"ok   {f.relative_to(ROOT)} {name}")
    print(f"\n{checked - bad}/{checked} CRD objects valid against the pinned upstream CRDs")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
