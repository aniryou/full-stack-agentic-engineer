"""Deploy assets: scripts are strict and dry-runnable, manifests validate for Kubernetes 1.34 and target the
l4x2 pool that layer 02's Terraform creates, and the vLLM pin is the same everywhere. No Terraform here."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"
REPO = LAB.parents[2]
L02_TF = REPO / "02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform"
SCRIPTS = sorted(DEPLOY.rglob("*.sh"))
MANIFESTS = sorted((DEPLOY / "gke").glob("*.yaml"))


def docs():
    for f in MANIFESTS:
        for d in yaml.safe_load_all(f.read_text()):
            if d:
                yield f, d


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(DEPLOY)))
def test_scripts_are_strict_and_parse(script):
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    assert os.access(script, os.X_OK)
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def _dry(script, *args, **env):
    e = {**os.environ, "DRY_RUN": "1", "MODE": "pip", **env}
    return subprocess.run([str(script), *args], capture_output=True, text=True, env=e, timeout=60)


@pytest.mark.parametrize("layout,flags", [("single", []), ("tp", ["--tensor-parallel-size 2"]),
                                          ("tp_ep", ["--tensor-parallel-size 2", "--enable-expert-parallel"]),
                                          ("dp_ep", ["--data-parallel-size 2", "--enable-expert-parallel"])])
def test_serve_dry_run_prints_the_layout_flags(layout, flags):
    out = _dry(DEPLOY / "any-gpu/serve_moe.sh", LAYOUT=layout, ROUTED="1")
    assert out.returncode == 0, out.stderr
    cmd = [l for l in out.stdout.splitlines() if l.startswith("+ vllm serve")][0]
    assert all(f in cmd for f in flags) and "--enable-return-routed-experts" in cmd
    from moelab import ep
    if layout != "single":
        assert " ".join(ep.vllm_flags(layout)) in cmd


def test_serve_rejects_unknown_layout():
    assert _dry(DEPLOY / "any-gpu/serve_moe.sh", LAYOUT="pp").returncode == 2


def test_bench_dry_run_creates_nothing(tmp_path):
    out = _dry(DEPLOY / "any-gpu/bench_layouts.sh", OUT=str(tmp_path / "res"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("+ vllm serve") == 3 and out.stdout.count("vllm bench serve") == 6
    assert not (tmp_path / "res").exists()


def test_gke_run_dry_run():
    for args in (["up"], ["layout", "dp_ep"], ["bench", "8"], ["clean"]):
        out = _dry(DEPLOY / "gke/run.sh", *args)
        assert out.returncode == 0, out.stderr
    patch = _dry(DEPLOY / "gke/run.sh", "layout", "tp").stderr
    value = json.loads(re.search(r"-p (\[.*\])", patch).group(1))[0]["value"]
    assert "--tensor-parallel-size=2" in value and "--enable-expert-parallel" not in value
    assert _dry(DEPLOY / "gke/run.sh", "layout", "pp").returncode != 0


def test_manifests_validate_for_kubernetes_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    n = 0
    for _, d in docs():
        kv.validate(d, "1.34.0", strict=True)
        n += 1
    assert n == 4


def test_deployment_targets_the_l4x2_pool_of_layer_02():
    dep = next(d for _, d in docs() if d["kind"] == "Deployment")
    pod = dep["spec"]["template"]["spec"]
    c = pod["containers"][0]
    gpus = c["resources"]["limits"]["nvidia.com/gpu"]
    assert c["resources"]["requests"]["nvidia.com/gpu"] == gpus == "2"
    tp = int(next(a.split("=")[1] for a in c["args"] if a.startswith("--tensor-parallel-size")))
    dp = int(next((a.split("=")[1] for a in c["args"] if a.startswith("--data-parallel-size")), 1))
    assert tp * dp == int(gpus) and "--enable-expert-parallel" in c["args"]
    if not L02_TF.is_dir():
        pytest.skip("layer 02 lab not in this checkout")
    tf = "\n".join(p.read_text() for p in L02_TF.glob("*.tf"))
    machine = re.search(r'variable "multi_gpu_machine_type"[^}]*default\s*=\s*"([^"]+)"', tf, re.S).group(1)
    count = int(re.search(r'variable "multi_gpu_count"[^}]*default\s*=\s*(\d+)', tf, re.S).group(1))
    gpu_type = re.search(r'variable "gpu_type"[^}]*default\s*=\s*"([^"]+)"', tf, re.S).group(1)
    assert "l4x2" in tf
    assert pod["nodeSelector"] == {"cloud.google.com/gke-accelerator": gpu_type, "node.kubernetes.io/instance-type": machine}
    assert count == int(gpus)
    assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in pod["tolerations"]
    assert c["startupProbe"]["httpGet"]["path"] == "/health"
    assert dep["spec"]["strategy"]["type"] == "Recreate"
    svc = next(d for _, d in docs() if d["kind"] == "Service")
    assert svc["spec"]["selector"] == dep["spec"]["template"]["metadata"]["labels"]


def test_bench_job_points_at_the_service_and_the_same_model():
    dep = next(d for _, d in docs() if d["kind"] == "Deployment")
    job = next(d for _, d in docs() if d["kind"] == "Job")
    args = job["spec"]["template"]["spec"]["containers"][0]["args"]
    model = dep["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert f"--model={model}" in args and "--base-url=http://vllm-moe.moe-lab.svc.cluster.local:8000" in args
    assert "nvidia.com/gpu" not in json.dumps(job)                      # the client needs no GPU


def test_vllm_pin_is_the_same_everywhere():
    tags = set()
    for p in list(DEPLOY.rglob("*")) + list((LAB / "moelab").glob("*.py")) + [LAB / "pyproject.toml"]:
        if p.is_file() and p.suffix in (".sh", ".yaml", ".md", ".py", ".toml"):
            t = p.read_text()
            tags |= set(re.findall(r"vllm/vllm-openai:v([\d.]+)", t)) | set(re.findall(r"vllm==([\d.]+)", t))
    assert tags == {"0.30.0"}, tags


def test_no_terraform_in_a_foundations_lab():
    assert not list(LAB.rglob("*.tf")), "labs in 00-foundations point at layer 02's Terraform instead"


def test_deploy_readmes_say_cost_and_cleanup():
    for readme in (DEPLOY / "any-gpu/README.md", DEPLOY / "gke/README.md", DEPLOY / "README.md"):
        t = readme.read_text().lower()
        assert "cost" in t and ("clean" in t or "terminate" in t)
