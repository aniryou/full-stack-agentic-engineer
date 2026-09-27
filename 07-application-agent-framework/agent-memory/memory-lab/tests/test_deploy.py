"""Deploy assets, checked offline: the compose file, the image, the scripts, and what each README promises."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"


def compose():
    return yaml.safe_load((DEPLOY / "local/compose.yaml").read_text())


def test_compose_services_ports_and_images():
    c = compose()
    assert set(c["services"]) == {"memory", "fake-llm", "postgres"}
    for name, svc in c["services"].items():
        for p in svc.get("ports", []):
            assert p.startswith("127.0.0.1:"), (name, p)                 # nothing listens beyond this machine
        assert ":latest" not in svc.get("image", "")
    pg = c["services"]["postgres"]
    assert pg["image"] == "pgvector/pgvector:0.8.6-pg17" and pg["profiles"] == ["pgvector"]
    assert "pg_isready" in " ".join(pg["healthcheck"]["test"])
    mem = c["services"]["memory"]
    assert mem["build"]["dockerfile"] == "deploy/local/Dockerfile" and "memlab-data:/data" in mem["volumes"]
    assert c["services"]["fake-llm"]["command"][:4] == ["python", "-m", "memlab", "fake"]
    assert "--prompt-tokens-details" in c["services"]["fake-llm"]["command"]
    assert set(c["volumes"]) == {"memlab-data", "memlab-pg"}


def test_compose_commands_exist_in_the_cli():
    from memlab.__main__ import main
    with pytest.raises(SystemExit) as e:
        main(["fake", "--help"])
    assert e.value.code == 0
    for svc in compose()["services"].values():
        cmd = svc.get("command")
        if cmd and cmd[:3] == ["python", "-m", "memlab"]:
            assert cmd[3] in ("fake", "serve")


def test_dockerfile_is_non_root_and_pins_its_base():
    text = (DEPLOY / "local/Dockerfile").read_text()
    assert re.search(r"^FROM python:3\.12-slim$", text, re.M) and "USER 10001" in text
    assert '".[pg]"' in text and "memlab" in text and "serve" in text
    assert not re.search(r"(?i)(password|secret|token)\s*=", text)


@pytest.mark.parametrize("script", sorted(p.relative_to(DEPLOY).as_posix() for p in DEPLOY.rglob("*.sh")))
def test_scripts_are_strict_and_support_dry_run(script):
    text = (DEPLOY / script).read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(DEPLOY / script)]).returncode == 0


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_up_without_docker_prints_the_commands():
    env = {**os.environ, "DRY_RUN": "1"}
    r = subprocess.run(["bash", str(DEPLOY / "local/up.sh"), "--pgvector"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 0 and "+ docker compose -f" in r.stdout and "--profile pgvector up -d --build" in r.stdout
    assert "MEMLAB_PG_DSN=postgresql://memlab:" in r.stdout
    r = subprocess.run(["bash", str(DEPLOY / "local/down.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "down --volumes" in r.stdout


@pytest.mark.parametrize("readme", ["local/README.md", "any-gpu/README.md", "gcp/README.md"])
def test_every_deploy_readme_states_tier_cost_and_cleanup(readme):
    text = (DEPLOY / readme).read_text()
    assert "**Tier:**" in text and "## Cost" in text and "## Cleanup" in text


def test_any_gpu_flags_match_vllm_v0_30():
    text = (DEPLOY / "any-gpu/README.md").read_text()
    for flag in ("--enable-auto-tool-choice", "--tool-call-parser hermes", "--enable-prompt-tokens-details",
                 "--runner pooling", "MAX_MODEL_LEN=512"):
        assert flag in text, flag
    assert "--task embed" not in text and "vllm-serving-lab/deploy/any-gpu/serve.sh" in text
    serve = LAB.parents[2] / "04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/serve.sh"
    if serve.exists():
        s = serve.read_text()
        for var in ("MODEL", "PORT", "MAX_MODEL_LEN", "GPU_MEM_UTIL", "EXTRA_ARGS", "DRY_RUN"):
            assert f'{var}="${{{var}:-' in s, var                      # the knobs this README turns exist
        assert "vllm/vllm-openai:v0.30.0" in s


def test_gcp_readme_uses_oauth_for_the_scheduler_and_no_terraform():
    text = (DEPLOY / "gcp/README.md").read_text()
    assert "--oauth-service-account-email" in text and "--oidc-service-account-email" not in text
    assert "jobs/memlab-consolidate:run" in text and "verify" in text
    assert not list(DEPLOY.rglob("*.tf"))
    from memlab.consolidate import gcp_commands
    assert "gcloud scheduler jobs create http memlab-consolidate-nightly" in "\n".join(gcp_commands())
