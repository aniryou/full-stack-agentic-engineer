"""Deploy assets are correct by construction: generated YAML is current, every object uses a pinned
API version, CRD objects use only fields the pinned upstream CRDs define (and all required ones),
core kinds and embedded pod templates validate strictly for Kubernetes 1.34, scripts are safe and
dry-runnable, and the Terraform pins match the lab's contract."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from k8sgpu import manifests as m
from k8sgpu import render

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
CRD = json.loads((ROOT / "tests" / "data" / "crd_paths.json").read_text())["kinds"]
CORE_KINDS = {"Namespace", "Pod", "Job", "Deployment", "Service", "ServiceAccount", "PriorityClass",
              "ResourceClaimTemplate", "DeviceClass"}
SCRIPTS = sorted(DEPLOY.rglob("*.sh"))


def all_objects():
    for f in sorted(DEPLOY.rglob("*.yaml")):
        if f.name == "kind-config.yaml":   # kind's own config, not a Kubernetes object
            continue
        for o in m.load_all(f):
            yield f.relative_to(ROOT), o


def test_generated_manifests_are_up_to_date():
    assert render.stale() == [], "run: python3 tools/render_manifests.py"


def test_every_object_uses_a_pinned_api_version():
    objs = list(all_objects())
    assert len(objs) > 50
    for f, o in objs:
        assert o["apiVersion"] == m.API_VERSIONS[o["kind"]], f


def _paths(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else k
            yield p
            yield from _paths(v, p)
    elif isinstance(node, list):
        for item in node:
            yield from _paths(item, prefix + "[]")


def _allowed(path: str, allowed: set[str]) -> bool:
    if path in allowed:
        return True
    parts = path.split(".")
    return any(".".join(parts[:i]) + ".*" in allowed for i in range(len(parts), 0, -1))


def _nodes(obj, path: str):
    nodes = [obj]
    for part in [p for p in path.split(".") if p]:
        arr, key = part.endswith("[]"), part.rstrip("[]")
        nxt = []
        for n in nodes:
            if isinstance(n, dict) and key in n:
                nxt.extend(n[key] if arr and isinstance(n[key], list) else [] if arr else [n[key]])
        nodes = nxt
    return nodes


def test_crd_objects_match_the_pinned_upstream_crds():
    checked = 0
    for f, o in all_objects():
        key = f"{o['apiVersion']}/{o['kind']}"
        if key not in CRD:
            continue
        checked += 1
        allowed = set(CRD[key]["paths"])
        body = {k: v for k, v in o.items() if k not in ("apiVersion", "kind", "metadata")}
        unknown = [p for p in _paths(body) if not _allowed(p, allowed)]
        assert not unknown, f"{f} {o['kind']}: fields not in the {key} CRD: {unknown[:5]}"
        for req in CRD[key]["required"]:
            parent, _, leaf = req.rpartition(".")
            for node in _nodes(body, parent) if parent else [body]:
                if isinstance(node, dict):
                    assert leaf in node, f"{f} {o['kind']}: missing required {req}"
    assert checked >= 20


def test_kueue_cel_rules_that_matter():
    for f, o in all_objects():
        spec = o.get("spec") or {}
        if o["kind"] == "ResourceFlavor" and "topologyName" in spec:
            assert spec.get("nodeLabels"), f"{f}: TAS flavor needs nodeLabels"
        if o["kind"] == "Topology":
            levels = [lv["nodeLabel"] for lv in spec["levels"]]
            assert m.HOSTNAME not in levels[:-1] and len(set(levels)) == len(levels)
        if o["kind"] == "ClusterQueue":
            pre = spec.get("preemption") or {}
            assert not (pre.get("reclaimWithinCohort", "Never") == "Never"
                        and (pre.get("borrowWithinCohort") or {}).get("policy", "Never") != "Never")
            if "cohortName" not in spec:
                assert "borrowingLimit" not in json.dumps(spec)


def test_compute_class_uses_documented_fields():
    top = {"priorities", "nodePoolAutoCreation", "activeMigration", "whenUnsatisfiable", "nodePoolConfig",
           "priorityDefaults", "autoscalingPolicy"}
    rung = {"machineFamily", "machineType", "minCores", "minMemoryGb", "spot", "gpu", "tpu", "flexStart",
            "reservations", "location", "nodepools", "capacityCheckWaitTimeSeconds", "storage"}
    for f, o in all_objects():
        if o["kind"] == "ComputeClass":
            assert set(o["spec"]) <= top, f
            for p in o["spec"]["priorities"]:
                assert set(p) <= rung, (f, p)
            assert o["spec"]["whenUnsatisfiable"] in ("DoNotScaleUp", "ScaleUpAnyway")


def test_core_kinds_and_embedded_pod_templates_validate_for_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    n = 0
    for f, o in all_objects():
        if o["kind"] in CORE_KINDS:
            kv.validate(o, "1.34.0", strict=True)
            n += 1
        elif o["kind"] in ("JobSet", "LeaderWorkerSet"):
            for _, t in m.iter_pod_templates(o):   # CRDs defer pod validation: check the templates as Pods
                meta = {"name": "template", **(t.get("metadata") or {})}
                kv.validate({"apiVersion": "v1", "kind": "Pod", "metadata": meta, "spec": t["spec"]}, "1.34.0", strict=True)
                n += 1
    assert n > 40


def test_dra_builder_output_validates_for_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    kv.validate(m.resource_claim_template("one-l4", "default",
                                          cel=['device.attributes["gpu.nvidia.com"].productName == "NVIDIA L4"']),
                "1.34.0", strict=True)
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"},
           "spec": m.pod_spec([{"name": "c", "image": "i", "resources": {"claims": [{"name": "gpu"}]}}], accelerator=None,
                              tolerate_gpu=False, resource_claims=[{"name": "gpu", "resourceClaimTemplateName": "one-l4"}])}
    kv.validate(pod, "1.34.0", strict=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(DEPLOY)))
def test_scripts_are_strict_and_parse(script):
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    assert os.access(script, os.X_OK)
    if script.name != "lib.sh":
        assert 'lib.sh"' in text and "run " in text, "every script prints its commands through run()"
    subprocess.run(["bash", "-n", str(script)], check=True)


def _dry(cmd: list[str], **env) -> str:
    e = {**os.environ, "DRY_RUN": "1", **env}
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_kind_up_dry_run_prints_the_whole_procedure():
    out = _dry(["deploy/kind/up.sh"])
    assert "kind create cluster --name gpu-lab --image kindest/node:v1.34.11@sha256:" in out
    assert out.count("--subresource=status") == 4 and "nvidia.com~1gpu" in out
    assert out.count("taint node") == 4 and "cloud.google.com/gce-topology-host=host-a2-2" in out
    assert "kueue/releases/download/v0.19.6/manifests.yaml" in out and "lws/releases/download/v0.11.0" in out
    assert "20-kueue-queues.yaml" in out and "wait --for=condition=Active clusterqueue/team-a-cq" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_other_scripts_dry_run():
    kw = _dry(["deploy/kind/kwok.sh"])
    assert "name: kwok-b1-s1-h1" in kw and "32 fake nodes x 8 GPUs" in kw
    assert "kwok/releases/download/v0.8.0/kwok.yaml" in kw and "delete stage pod-complete" in kw
    assert "kind delete cluster --name gpu-lab" in _dry(["deploy/kind/down.sh"])
    gk = _dry(["deploy/gke/install-addons.sh"], KUBE_CONTEXT="gke_p_z_gpu-lab")
    assert "--context gke_p_z_gpu-lab apply --server-side" in gk and "10-kueue-gke.yaml" in gk
    assert "00-smoke-l4.yaml" in _dry(["deploy/gke/apply-examples.sh", "smoke"], KUBE_CONTEXT="gke_p_z_gpu-lab")


def test_versions_are_pinned_once():
    env = dict(line.split("=", 1) for line in (DEPLOY / "versions.env").read_text().splitlines()
               if "=" in line and not line.startswith("#"))
    assert env["KUEUE_VERSION"] == "v0.19.6" and env["JOBSET_VERSION"] == "v0.12.0" and env["LWS_VERSION"] == "v0.11.0"
    assert env["KIND_NODE_IMAGE"].startswith("kindest/node:v1.34.") and "@sha256:" in env["KIND_NODE_IMAGE"]
    for s in SCRIPTS:   # no script hard-codes a release
        assert not re.search(r"v0\.19\.6|v0\.12\.0|v0\.11\.0", s.read_text()), s


def test_terraform_contract():
    tf = DEPLOY / "gcp" / "terraform"
    versions = (tf / "versions.tf").read_text()
    assert 'required_version = ">= 1.9"' in versions and 'version = ">= 8.0"' in versions
    variables = (tf / "variables.tf").read_text()
    declared = set(re.findall(r'variable "([a-z0-9_]+)"', variables))
    assert variables.count("description") >= len(declared)
    example = set(re.findall(r"^([a-z_]+)\s*=", (tf / "terraform.tfvars.example").read_text(), re.M))
    assert example <= declared
    pools = (tf / "node_pools.tf").read_text()
    for needle in ("spot            = var.gpu_spot", "min_node_count = 0", "gpu_driver_version = var.gpu_driver_version",
                   "flex_start      = true", "queued_provisioning", 'consume_reservation_type = "NO_RESERVATION"'):
        assert needle in pools, needle
