"""Deploy assets: the Terraform keeps its cheap defaults, manifests are consistent, scripts are safe."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def test_cloud_run_terraform_defaults_are_the_cheap_ones():
    tf = "\n".join(p.read_text() for p in (DEPLOY / "gcp/cloud-run/terraform").glob("*.tf"))
    assert 'required_version = ">= 1.9"' in tf and 'version = ">= 8.0"' in tf
    assert 'accelerator = "nvidia-l4"' in tf and '"nvidia.com/gpu" = "1"' in tf
    assert "gpu_zonal_redundancy_disabled = true" in tf
    assert re.search(r'variable "min_instances"[^}]*default\s*=\s*0', tf, re.S)      # scale to zero
    assert re.search(r'variable "max_instances"[^}]*default\s*=\s*1', tf, re.S)
    assert 'path = "/health"' in tf and "container_port = 8000" in tf
    assert "allUsers" not in tf                                                     # never public
    assert (DEPLOY / "gcp/cloud-run/terraform/terraform.tfvars.example").exists()


def test_gke_manifests_are_consistent():
    docs = list(yaml.safe_load_all((DEPLOY / "gcp/gke/vllm.yaml").read_text()))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    svc = next(d for d in docs if d["kind"] == "Service")
    assert dep["apiVersion"] == "apps/v1" and svc["apiVersion"] == "v1"
    pod = dep["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["resources"]["requests"]["nvidia.com/gpu"] == c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["nodeSelector"]["cloud.google.com/gke-accelerator"] == "nvidia-l4"
    assert c["startupProbe"]["httpGet"]["path"] == "/health"
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert svc["spec"]["selector"] == labels
    # PodMonitoring is a CRD (Managed Service for Prometheus): check it by hand
    pm = yaml.safe_load((DEPLOY / "gcp/gke/podmonitoring.yaml").read_text())
    assert (pm["apiVersion"], pm["kind"]) == ("monitoring.googleapis.com/v1", "PodMonitoring")
    assert pm["spec"]["selector"]["matchLabels"].items() <= labels.items()
    ep = pm["spec"]["endpoints"][0]
    assert ep["path"] == "/metrics" and ep["port"] == c["ports"][0]["name"]


def test_image_tag_is_pinned_the_same_everywhere():
    tags = set()
    for p in DEPLOY.rglob("*"):
        if p.suffix in (".tf", ".sh", ".yaml"):
            tags |= set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", p.read_text()))
    assert len(tags) == 1, tags


@pytest.mark.parametrize("script", sorted(p.relative_to(DEPLOY).as_posix() for p in DEPLOY.rglob("*.sh")))
def test_shell_scripts_are_strict_and_support_dry_run(script):
    text = (DEPLOY / script).read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(DEPLOY / script)]).returncode == 0
