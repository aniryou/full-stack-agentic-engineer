"""Deploy assets: strict, dry-runnable scripts; one pinned image; valid parser names; the GKE manifest is
consistent and schema-valid; the Cloud Run tfvars only uses variables the 04 lab's Terraform declares."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"
REPO = LAB.parents[2]
# vllm030/reasoning_init.py, _REASONING_PARSERS_TO_REGISTER (vLLM v0.30.0) — the names this lab uses
VLLM_PARSERS = {"qwen3", "deepseek_r1", "deepseek_v3", "openai_gptoss", "granite", "mistral", "glm45", "kimi_k2"}


@pytest.mark.parametrize("script", sorted(p.relative_to(DEPLOY).as_posix() for p in DEPLOY.rglob("*.sh")))
def test_shell_scripts_are_strict_and_support_dry_run(script):
    text = (DEPLOY / script).read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(DEPLOY / script)]).returncode == 0


def test_serve_dry_run_picks_parser_and_flags():
    if not shutil.which("bash"):
        pytest.skip("no bash")
    env = {"DRY_RUN": "1", "PATH": "/usr/bin:/bin", "HOME": "/tmp", "MODE": "pip"}
    out = subprocess.run(["bash", str(DEPLOY / "any-gpu/serve.sh")], capture_output=True, text=True, env=env).stdout
    assert "--reasoning-parser qwen3" in out and "--reasoning-config" in out and "--enable-prompt-tokens-details" in out
    out = subprocess.run(["bash", str(DEPLOY / "any-gpu/serve.sh")], capture_output=True, text=True,
                         env={**env, "MODEL": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}).stdout
    assert "--reasoning-parser deepseek_r1" in out
    assert "--enable-reasoning" not in (DEPLOY / "any-gpu/serve.sh").read_text()      # removed from vLLM


def test_image_tag_pinned_and_parsers_valid():
    tags, parsers = set(), set()
    for p in DEPLOY.rglob("*"):
        if p.suffix in (".sh", ".yaml", ".example", ".md"):
            t = p.read_text()
            tags |= set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", t))
            parsers |= set(re.findall(r"--reasoning-parser[= ]\"?([a-z0-9_]+)", t))
            parsers |= set(re.findall(r'"--reasoning-parser",\s*"([a-z0-9_]+)"', t))
    assert tags == {"v0.30.0"}
    assert parsers and parsers <= VLLM_PARSERS, parsers


def test_gke_manifest_is_consistent():
    docs = list(yaml.safe_load_all((DEPLOY / "gcp/gke/vllm-thinking.yaml").read_text()))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    svc = next(d for d in docs if d["kind"] == "Service")
    assert dep["apiVersion"] == "apps/v1" and svc["apiVersion"] == "v1"
    pod = dep["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["resources"]["requests"]["nvidia.com/gpu"] == c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["nodeSelector"]["cloud.google.com/gke-accelerator"] == "nvidia-l4"
    assert "--reasoning-parser=qwen3" in c["args"] and c["startupProbe"]["httpGet"]["path"] == "/health"
    mml = int(next(a.split("=")[1] for a in c["args"] if a.startswith("--max-model-len=")))
    assert mml >= 16384
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert svc["spec"]["selector"] == labels
    pm = yaml.safe_load((DEPLOY / "gcp/gke/podmonitoring-thinking.yaml").read_text())
    assert (pm["apiVersion"], pm["kind"]) == ("monitoring.googleapis.com/v1", "PodMonitoring")
    assert pm["spec"]["selector"]["matchLabels"].items() <= labels.items()
    assert pm["spec"]["endpoints"][0]["port"] == c["ports"][0]["name"]


def test_gke_manifest_schema():
    kv = shutil.which("kubernetes-validate")
    if not kv:
        pytest.skip("kubernetes-validate not installed")
    r = subprocess.run([kv, "--strict", "-k", "1.34.0", str(DEPLOY / "gcp/gke/vllm-thinking.yaml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cloud_run_tfvars_match_the_04_module():
    tfvars = (DEPLOY / "gcp/cloud-run-thinking.tfvars.example").read_text()
    keys = set(re.findall(r"^([a-z_]+)\s*=", tfvars, re.M))
    assert {"model", "extra_args", "max_model_len", "concurrency", "request_timeout", "min_instances"} <= keys
    assert re.search(r'min_instances\s*=\s*0', tfvars) and '"--reasoning-parser", "qwen3"' in tfvars
    seqs = int(re.search(r'"--max-num-seqs", "(\d+)"', tfvars)[1])
    conc = int(re.search(r"^concurrency\s*=\s*(\d+)", tfvars, re.M)[1])
    assert conc <= seqs, "Cloud Run should not send more requests than the engine batches"
    variables = REPO / "04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/variables.tf"
    if not variables.exists():
        pytest.skip("04 serving lab not in this checkout")
    declared = set(re.findall(r'variable "([a-z_]+)"', variables.read_text()))
    assert keys <= declared, keys - declared
