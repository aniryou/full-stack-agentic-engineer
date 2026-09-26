"""Deploy assets are safe and consistent: strict, dry-runnable scripts that print the whole procedure;
versions pinned in one place; the Terraform keeps the sandbox contract and its cheapest defaults; every
deploy README says what it costs and how to clean up."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sandboxlab import gke

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
SCRIPTS = sorted(DEPLOY.rglob("*.sh"))
TF = DEPLOY / "gcp" / "terraform"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(DEPLOY)))
def test_scripts_are_strict_and_parse(script):
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    assert os.access(script, os.X_OK)
    if script.name != "lib.sh":
        assert 'lib.sh"' in text and "run " in text and "DRY_RUN=1" in text, "prints commands through run(); documents DRY_RUN"
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def _dry(cmd: list[str], **env) -> str:
    e = {**os.environ, "DRY_RUN": "1", **env}
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_kind_dry_runs_print_the_procedure_in_order():
    out = _dry(["deploy/kind/up.sh"])
    assert "kind create cluster --name sandbox-lab --image kindest/node:v1.34.11@sha256:" in out
    assert "taint node sandbox-lab-worker2 sandboxlab/pool=sandbox:NoSchedule" in out
    order = [out.index(s) for s in ("30-network-policy.yaml", "create secret generic egress-credentials",
                                    "40-egress-proxy.yaml", "50-admission-policy.yaml")]
    assert order == sorted(order)                         # policies before pods; secrets before the proxy
    assert "--from-file=token=" in out and "--from-literal" not in out
    ex = _dry(["deploy/kind/run-examples.sh"])
    assert "wait --for=condition=complete job/run-example-0001" in ex and "92-gvisor-in-kind.yaml" in ex
    assert "93-egress-must-fail.yaml" in ex and "job/run-egress-check-0001" in ex    # NetworkPolicy checked, not assumed
    assert "kind delete cluster --name sandbox-lab" in _dry(["deploy/kind/down.sh"])


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_docker_and_gke_dry_runs():
    hard = _dry(["deploy/docker/run-hardened.sh", "deploy/docker/examples/fetch_weather.py"], RUNTIME="runsc")
    assert "--runtime runsc" in hard and "--network none" in hard and hard.rstrip().endswith("--nproc")
    px = _dry(["deploy/docker/run-with-proxy.sh", "deploy/docker/examples/fetch_weather.py"])
    assert "--unix /tmp/sandboxlab-egress/proxy.sock" in px and "SANDBOX_PROXY_URL=unix:/run/egress/proxy.sock" in px
    gv = _dry(["deploy/docker/install-gvisor.sh"])
    assert "apt-get install -y runsc" in gv and "runsc install" in gv
    g = _dry(["deploy/gke/apply.sh"])
    assert "get-credentials sandbox-lab --zone us-central1-a" in g and "get runtimeclass gvisor" in g
    assert g.index("30-network-policy.yaml") < g.index("40-egress-proxy.yaml") < g.index("10-run-code-job.yaml")
    assert "docker push us-central1-docker.pkg.dev/PROJECT_ID/sandbox/python:3.12-slim" in _dry(["deploy/gcp/mirror-image.sh"])


def test_versions_are_pinned_once():
    env = dict(l.split("=", 1) for l in (DEPLOY / "versions.env").read_text().splitlines() if "=" in l and not l.startswith("#"))
    assert env["KIND_VERSION"] == "v0.33.0" and env["KIND_NODE_IMAGE"].startswith("kindest/node:v1.34.11@sha256:44e222")
    assert env["SANDBOX_IMAGE"] == "python:3.12-slim" and env["AGENT_SANDBOX_VERSION"] == "v1.0.2"
    for s in SCRIPTS:
        assert not re.search(r"kindest/node:|v0\.33\.0|v1\.0\.2", s.read_text()), s


def test_terraform_contract():
    versions = (TF / "versions.tf").read_text()
    assert 'required_version = ">= 1.9"' in versions and 'version = ">= 8.0"' in versions
    variables = (TF / "variables.tf").read_text()
    declared = set(re.findall(r'variable "([a-z0-9_]+)"', variables))
    assert variables.count("description") >= len(declared)
    example = set(re.findall(r"^#?\s*([a-z_]+)\s*=", (TF / "terraform.tfvars.example").read_text(), re.M))
    assert example <= declared
    pools = (TF / "node_pools.tf").read_text()
    assert 'type = "GVISOR"' in pools and 'type = "gvisor"' not in pools     # the provider's validator is case-sensitive
    for needle in ('image_type      = "COS_CONTAINERD"', "spot            = var.sandbox_spot", "min_node_count = 0",
                   'mode = "GKE_METADATA"', "pod_pids_limit = var.pod_pids_limit"):
        assert needle in pools, needle
    cluster = (TF / "cluster.tf").read_text()
    assert 'datapath_provider = "ADVANCED_DATAPATH"' in cluster and "enable_private_nodes    = true" in cluster
    assert "count                              = var.enable_nat ? 1 : 0" in (TF / "network.tf").read_text()
    assert all(c.ok for c in gke.review()), [c.name for c in gke.review() if not c.ok]


def test_gke_cost_helpers():
    assert gke.pods_per_node(2, 8, 0.5, 0.3125) == 3                          # e2-standard-2 after reservations: CPU-bound
    assert abs(gke.cost_per_execution_usd(3600, 0.03, 3) - 0.01) < 1e-12


@pytest.mark.parametrize("readme", sorted(DEPLOY.rglob("README.md")), ids=lambda p: str(p.relative_to(DEPLOY)))
def test_deploy_readmes_state_cost_and_cleanup(readme):
    text = readme.read_text()
    assert re.search(r"\*\*Cost", text) and re.search(r"\*\*Clean ?up", text), readme
