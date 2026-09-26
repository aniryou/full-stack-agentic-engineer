"""Deploy assets are correct by construction: CRD objects validate against upstream schemas,
values files agree with each other and with the lab presets, scripts are safe to dry-run."""
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from igwlab.fakebackend import EngineProfile
from igwlab.k8s import GKE_GATEWAY_CLASSES, check_file, load_schemas
from igwlab.router import load_config

LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"
PRESET = yaml.safe_load((LAB / "igwlab/configs/default-weighted.yaml").read_text())
EXPECTED_API = {"InferencePool": "inference.networking.k8s.io/v1", "InferenceObjective": "llm-d.ai/v1alpha2",
                "Gateway": "gateway.networking.k8s.io/v1", "HTTPRoute": "gateway.networking.k8s.io/v1",
                "PodMonitoring": "monitoring.googleapis.com/v1", "HorizontalPodAutoscaler": "autoscaling/v2",
                "Deployment": "apps/v1", "Service": "v1"}
VLLM_METRICS = {"vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc"}


def docs(path):
    return [d for d in yaml.safe_load_all(Path(path).read_text()) if d]


def all_k8s_docs():
    out = []
    for p in sorted(DEPLOY.rglob("*.yaml")):
        for d in docs(p):
            if isinstance(d, dict) and "apiVersion" in d and "kind" in d and d["kind"] != "Cluster":
                out.append((p, d))
    return out


def test_api_versions_match_the_upstream_apis():
    kinds = set()
    for p, d in all_k8s_docs():
        if d["kind"] in EXPECTED_API:
            assert d["apiVersion"] == EXPECTED_API[d["kind"]], (p, d["kind"])
            kinds.add(d["kind"])
    assert {"InferencePool", "InferenceObjective", "Gateway", "HTTPRoute", "PodMonitoring",
            "HorizontalPodAutoscaler", "Deployment"} <= kinds


def test_crd_objects_validate_against_upstream_schemas():
    schemas = load_schemas()
    checked = 0
    for p in sorted(DEPLOY.rglob("*.yaml")):
        for key, errs in check_file(p, schemas).items():
            assert errs == [], (p, key, errs)
            checked += 1
    assert checked >= 6


def test_core_objects_validate_with_kubernetes_validate():
    kv = pytest.importorskip("kubernetes_validate")
    n = 0
    for p, d in all_k8s_docs():
        if d["kind"] in ("Deployment", "Service", "HorizontalPodAutoscaler"):
            kv.validate(d, "1.34.0", strict=True)
            n += 1
    assert n >= 4


def _epp_config(values: dict) -> dict:
    epp = values["router"]["epp"]
    return yaml.safe_load(epp["pluginsCustomConfig"][epp["pluginsConfigFile"]])


@pytest.mark.parametrize("values_file", ["kind/router-values.yaml", "gke/epp-values.yaml"])
def test_helm_values_embed_the_lab_preset_and_valid_enums(values_file):
    v = yaml.safe_load((DEPLOY / values_file).read_text())
    assert _epp_config(v) == PRESET                       # the policy tuned in notebook 02 is what the EPP runs
    load_config(_epp_config(v))
    pool_schema = load_schemas()[("inference.networking.k8s.io/v1", "InferencePool")]
    enum = pool_schema["properties"]["spec"]["properties"]["endpointPickerRef"]["properties"]["failureMode"]["enum"]
    assert v["router"]["inferencePool"]["failureMode"] in enum
    assert re.fullmatch(r"v\d+\.\d+\.\d+", v["router"]["epp"]["image"]["tag"])
    for o in v["router"].get("inferenceObjectives", []):
        assert isinstance(o["priority"], int)


def test_gke_rendered_objects_agree_with_values_route_and_workload():
    v = yaml.safe_load((DEPLOY / "gke/epp-values.yaml").read_text())
    pool = docs(DEPLOY / "gke/rendered/inferencepool.yaml")[0]
    route = [d for d in docs(DEPLOY / "gke/gateway.yaml") if d["kind"] == "HTTPRoute"][0]
    gw = [d for d in docs(DEPLOY / "gke/gateway.yaml") if d["kind"] == "Gateway"][0]
    release = route["spec"]["rules"][0]["backendRefs"][0]["name"]
    assert pool["metadata"]["name"] == release and pool["spec"]["endpointPickerRef"]["name"] == f"{release}-epp"
    assert pool["spec"]["selector"]["matchLabels"] == v["router"]["modelServers"]["matchLabels"]
    assert [p["number"] for p in pool["spec"]["targetPorts"]] == [p["number"] for p in v["router"]["modelServers"]["targetPorts"]]
    assert pool["spec"]["endpointPickerRef"]["failureMode"] == v["router"]["inferencePool"]["failureMode"]
    objs = {d["metadata"]["name"]: d["spec"]["priority"] for d in docs(DEPLOY / "gke/rendered/inferenceobjectives.yaml")}
    assert objs == {o["name"]: o["priority"] for o in v["router"]["inferenceObjectives"]}
    assert route["spec"]["parentRefs"][0]["name"] == gw["metadata"]["name"]
    assert gw["spec"]["gatewayClassName"] in GKE_GATEWAY_CLASSES
    dep = docs(DEPLOY / "gke/vllm.yaml")[0]
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert all(labels.get(k) == val for k, val in pool["spec"]["selector"]["matchLabels"].items())
    ports = {p["name"]: p["containerPort"] for p in dep["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert ports["http"] == pool["spec"]["targetPorts"][0]["number"]
    pm = docs(DEPLOY / "gke/podmonitoring-vllm.yaml")[0]
    assert pm["spec"]["endpoints"][0]["port"] in ports and pm["spec"]["selector"]["matchLabels"] == {"app": labels["app"]}


def test_hpa_scales_the_vllm_deployment_on_a_vllm_queue_metric():
    hpa = docs(DEPLOY / "gke/hpa.yaml")[0]
    dep = docs(DEPLOY / "gke/vllm.yaml")[0]
    assert hpa["spec"]["scaleTargetRef"]["name"] == dep["metadata"]["name"]
    metric = hpa["spec"]["metrics"][0]["pods"]["metric"]["name"]
    assert any(m in metric for m in VLLM_METRICS) and hpa["spec"]["minReplicas"] >= 1


def test_simulator_flags_match_the_fake_backend_profile():
    want = EngineProfile().sim_args("lab/llm", 8000)
    as_eq = []
    for i in range(0, len(want), 1):
        if want[i].startswith("--"):
            nxt = want[i + 1] if i + 1 < len(want) and not want[i + 1].startswith("--") else None
            as_eq.append(f"{want[i]}={nxt}" if nxt is not None else f"{want[i]}=true")
    kind_args = docs(DEPLOY / "kind/sim-deployment.yaml")[0]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert sorted(kind_args) == sorted(as_eq)
    compose = yaml.safe_load((DEPLOY / "local/docker-compose.yaml").read_text())
    cmd = compose["services"]["sim-a"]["command"]
    flags = {c.split("=")[0] for c in cmd if c.startswith("--")}
    assert flags == {a.split("=")[0] for a in as_eq}
    router_cmd = compose["services"]["router"]["command"]
    backends = [router_cmd[i + 1].split("=")[0] for i, c in enumerate(router_cmd) if c == "--backend"]
    assert backends == ["sim-a", "sim-b", "sim-c"] and all(b in compose["services"] for b in backends)


def test_images_are_pinned():
    for p in sorted(DEPLOY.rglob("*.yaml")):
        for img in re.findall(r"image:\s*([^\s#]+)", p.read_text()):
            if img.startswith(("{", "igwlab:")):
                continue
            assert ":" in img and not img.endswith(":latest"), (p, img)


def test_shell_scripts_are_strict_and_dry_runnable():
    scripts = sorted(DEPLOY.rglob("*.sh"))
    assert len(scripts) >= 7
    for s in scripts:
        text = s.read_text()
        assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text, s
        subprocess.run(["bash", "-n", str(s)], check=True)
    out = subprocess.run(["bash", str(DEPLOY / "kind/up.sh")], env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, check=True).stdout
    assert out.index("kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension") \
        < out.index("helm upgrade --install")                       # CRDs before the chart that creates a pool


def test_terraform_has_the_cost_conscious_shape():
    tf = "\n".join(p.read_text() for p in sorted((DEPLOY / "gcp/terraform").glob("*.tf")))
    assert 'required_version = ">= 1.9"' in tf and 'version = ">= 8.0"' in tf
    for needle in ('purpose       = "REGIONAL_MANAGED_PROXY"', "gateway_api_config", "managed_prometheus",
                   "min_node_count = 0", 'gpu_driver_version = "DEFAULT"', "spot         = var.gpu_spot",
                   "location = var.zone"):
        assert needle in tf, needle
    assert (DEPLOY / "gcp/terraform/terraform.tfvars.example").exists()
