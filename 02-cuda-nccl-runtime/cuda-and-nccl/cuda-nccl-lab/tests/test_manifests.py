"""The GKE manifests: parse, GPU-pod hygiene, the sharing selectors, and schema validation (core kinds)."""
import pathlib

import pytest
import yaml

GKE = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "gke"
ALLOWED = {("v1", "Namespace"), ("batch/v1", "Job"), ("apps/v1", "Deployment"),
           ("monitoring.googleapis.com/v1", "ClusterRules")}  # the last is a GMP CRD: checked here, not by schema


def docs():
    out = []
    for path in sorted(GKE.glob("0*.yaml")):
        out += [(path.name, d) for d in yaml.safe_load_all(path.read_text()) if d]
    return out


def pod_specs():
    for name, d in docs():
        if d["kind"] in ("Job", "Deployment"):
            yield name, d["spec"]["template"]["spec"]


def test_every_manifest_parses_and_uses_known_kinds():
    kinds = {(d["apiVersion"], d["kind"]) for _, d in docs()}
    assert kinds <= ALLOWED and len(docs()) == 7


def test_gpu_pods_limit_gpus_tolerate_the_taint_and_select_an_accelerator():
    for name, spec in pod_specs():
        for c in spec["containers"]:
            res = c["resources"]
            gpus = res["limits"]["nvidia.com/gpu"]
            assert gpus >= 1, name
            assert res.get("requests", {}).get("nvidia.com/gpu", gpus) == gpus, name  # extended resources: requests == limits
        assert any(t["key"] == "nvidia.com/gpu" for t in spec["tolerations"]), name
        assert "cloud.google.com/gke-accelerator" in spec["nodeSelector"], name


def test_sharing_and_topology_selectors():
    specs = dict(pod_specs())
    ts = specs["04-time-sharing.yaml"]["nodeSelector"]
    assert ts["cloud.google.com/gke-gpu-sharing-strategy"] == "time-sharing"
    assert ts["cloud.google.com/gke-max-shared-clients-per-gpu"] == "2"
    assert specs["05-mig.yaml"]["nodeSelector"]["cloud.google.com/gke-gpu-partition-size"] == "1g.5gb"
    nccl = specs["03-nccl-tests-2gpu.yaml"]
    assert nccl["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert nccl["nodeSelector"]["node.kubernetes.io/instance-type"] == "g2-standard-24"
    assert any(v.get("emptyDir", {}).get("medium") == "Memory" for v in nccl["volumes"])  # /dev/shm for NCCL


def test_core_kinds_pass_strict_schema_validation():
    kv = pytest.importorskip("kubernetes_validate")
    for name, d in docs():
        if d["apiVersion"] == "monitoring.googleapis.com/v1":
            continue  # CRD: no bundled schema
        kv.validate(d, "1.34.0", strict=True)


def test_alertmanager_example_routes_every_rule_severity():
    from gpurt import dcgm

    cfg = yaml.safe_load((GKE / "alertmanager" / "alertmanager.example.yaml").read_text())
    receivers = {r["name"] for r in cfg["receivers"]}
    route = cfg["route"]
    assert route["receiver"] in receivers and all(r["receiver"] in receivers for r in route["routes"])
    routed = {r["matchers"][0].split('"')[1]: r["receiver"] for r in route["routes"]}
    assert routed == {"critical": "page", "warning": "ticket"}  # info falls through to the default: notify
    assert {r.severity for r in dcgm.RULES} <= set(routed) | {"info"}


def test_one_gpu_jobs_pin_the_one_gpu_pool():
    specs = dict(pod_specs())
    for name in ("01-gpu-smoke.yaml", "02-cuda-vectoradd.yaml"):  # not the 2-GPU or time-shared nodes
        assert specs[name]["nodeSelector"]["gpu-lab/pool"] == "l4", name
