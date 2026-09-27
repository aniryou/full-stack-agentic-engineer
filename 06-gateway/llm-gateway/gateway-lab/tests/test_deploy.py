"""Deploy assets: compose files are consistent with the configs they run, scripts are strict and dry-runnable,
the vLLM pin and flags are the same everywhere, and the GCP pointer's links resolve."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from gwlab.gateway.config import load_config

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SCRIPTS = sorted(p.relative_to(DEPLOY).as_posix() for p in DEPLOY.rglob("*.sh"))


def compose(name):
    return yaml.safe_load((DEPLOY / name / "docker-compose.yaml").read_text())


def test_local_compose_runs_the_lab_config():
    c = compose("local")
    svc = c["services"]
    assert set(svc) == {"gateway", "acme", "bolt"}
    gw = svc["gateway"]
    assert gw["ports"] == ["8080:8080"] and all("ports" not in svc[p] for p in ("acme", "bolt"))   # providers stay internal
    env = gw["environment"]
    cfg = load_config("lab", {"providers": {"acme": {"base_url": env["ACME_URL"], "key": env["ACME_API_KEY"]},
                                            "bolt": {"base_url": env["BOLT_URL"], "key": env["BOLT_API_KEY"]}}})
    assert cfg.providers["acme"].base_url == "http://acme:8101" and "--port" in svc["acme"]["command"]
    assert svc["acme"]["command"][svc["acme"]["command"].index("--port") + 1] == "8101"
    assert svc["bolt"]["command"][svc["bolt"]["command"].index("--dialect") + 1] == "anthropic"
    assert "--fail-rate" in svc["acme"]["command"]                                     # one provider set to fail
    assert svc["acme"]["command"][svc["acme"]["command"].index("--keys") + 1] == env["ACME_API_KEY"]


def test_any_gpu_compose_and_script_agree_on_vllm():
    c = compose("any-gpu")
    v = c["services"]["vllm"]
    assert v["image"] == "vllm/vllm-openai:v0.30.0" and v["ports"] == ["127.0.0.1:8000:8000"]
    args = v["command"]
    assert args[0] == "Qwen/Qwen2.5-0.5B-Instruct" and args[args.index("--served-model-name") + 1] == "lab/llm"
    assert "--enable-prompt-tokens-details" in args
    serve = (DEPLOY / "any-gpu/serve.sh").read_text()
    assert "Qwen/Qwen2.5-0.5B-Instruct" in serve and "--served-model-name lab/llm" in serve
    assert "--enable-prompt-tokens-details" in serve and 'DTYPE="half"' in serve
    cfg = load_config("vllm")
    assert cfg.models["local/llm"].upstream == "lab/llm" and cfg.aliases["chat"].targets[-1] == "acme/fast"
    gw = c["services"]["gateway"]
    assert gw["environment"]["GWLAB_VLLM_URL"] == "http://vllm:8000" and "--config" in gw["command"]


def test_vllm_pin_is_the_same_everywhere():
    tags = set()
    for p in list(DEPLOY.rglob("*")) + [DEPLOY.parent / "README.md"]:
        if p.is_file() and p.suffix in (".sh", ".yaml", ".md"):
            tags |= set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", p.read_text()))
    assert tags == {"v0.30.0"}


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_are_strict_and_dry_runnable(script, tmp_path):
    path = DEPLOY / script
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    if not shutil.which("bash"):
        pytest.skip("no bash")
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    env = {**os.environ, "DRY_RUN": "1", "STATE_DIR": str(tmp_path / "state")}
    r = subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "+ " in r.stdout                                                    # it printed the commands
    assert not (tmp_path / "state").exists()                                   # and ran none of them


def test_gcp_pointer_has_no_terraform_and_its_links_resolve():
    gcp = DEPLOY / "gcp"
    assert not list(gcp.rglob("*.tf"))
    text = (gcp / "README.md").read_text()
    links = re.findall(r"\]\(([^)#]+)", text)
    assert len(links) >= 5
    for rel in links:
        if rel.startswith("http"):
            continue
        assert (gcp / rel).exists(), rel
    for needed in ("vllm-serving-lab/deploy/gcp/cloud-run", "inference-gateway-lab/deploy/gke", "Cost and cleanup"):
        assert needed in text


def test_every_deploy_readme_says_cost_and_cleanup():
    for readme in DEPLOY.rglob("README.md"):
        t = readme.read_text().lower()
        assert "cost" in t and ("cleanup" in t or "clean up" in t), readme
