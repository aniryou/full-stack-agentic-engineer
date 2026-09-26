"""Deploy assets: strict scripts with DRY_RUN, the same image pin as the serving lab, scheme tables in
sync with quantlab.serve, and the serving lab's GKE Deployment still valid with each scheme's args."""
import copy
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from quantlab import serve

LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"
SERVING = LAB.parents[1] / "serving-engine/vllm-serving-lab"
SCRIPTS = sorted(DEPLOY.rglob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(DEPLOY)))
def test_scripts_are_strict_and_parse(script):
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text and "DRY_RUN" in text
    assert os.access(script, os.X_OK)
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def _dry(script, **env):
    e = {**os.environ, "DRY_RUN": "1", **env}
    return subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=60, cwd=LAB)


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_serve_sh_agrees_with_the_python_plan():
    """The case table in serve.sh and quantlab.serve.SCHEMES name the same models and flags."""
    text = (DEPLOY / "any-gpu/serve.sh").read_text()
    for key, s in serve.SCHEMES.items():
        m = re.search(rf'^\s+{re.escape(key)}\)\s+DEFAULT_MODEL="([^"]+)"(.*)$', text, re.M)
        assert m and m.group(1) == s.t1_model, key
        assert ("fp8_per_tensor" in m.group(2)) == ("fp8_per_tensor" in s.flags), key
    out = _dry(DEPLOY / "any-gpu/serve.sh", SCHEME="fp8", GPU_CC="7.5").stdout
    assert "--dtype half" in out and "Marlin" in out
    out = _dry(DEPLOY / "any-gpu/serve.sh", SCHEME="w8a8-int8", GPU_CC="10.0").stdout
    assert "not supported on compute capability >= 10.0" in out
    out = _dry(DEPLOY / "any-gpu/serve.sh", SCHEME="w4a16", GPU_CC="8.9", KV_CACHE_DTYPE="fp8").stdout
    assert "FlashInfer" in out and "--kv-cache-dtype fp8" in out


def test_image_tag_is_the_serving_labs():
    tags = set()
    for p in list(DEPLOY.rglob("*")) + [LAB / "quantlab/serve.py"]:
        if p.suffix in (".sh", ".py", ".md", ".example"):
            tags |= set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", p.read_text()))
    assert tags == {"v0.30.0"}
    if SERVING.exists():
        theirs = set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", (SERVING / "deploy/any-gpu/serve.sh").read_text()))
        assert theirs == tags


def test_tfvars_only_set_variables_the_serving_terraform_declares():
    tf_dir = SERVING / "deploy/gcp/cloud-run/terraform"
    if not tf_dir.exists():
        pytest.skip("serving lab not in this checkout")
    declared = set(re.findall(r'variable "(\w+)"', "\n".join(p.read_text() for p in tf_dir.glob("*.tf"))))
    text = (DEPLOY / "gcp/cloud-run.quantized.tfvars.example").read_text()
    used = set(re.findall(r"^#?\s*(\w+)\s+=", text, re.M))
    assert used and used <= declared, used - declared
    rendered = serve.cloud_run_tfvars(serve.plan("fp8-online", "L4"))
    assert set(re.findall(r"^(\w+)\s+=", rendered, re.M)) <= declared


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_gcp_wrapper_dry_runs_through_the_serving_lab():
    if not SERVING.exists():
        pytest.skip("serving lab not in this checkout")
    out = _dry(DEPLOY / "gcp/deploy-quantized.sh", PROJECT_ID="p", SCHEME="fp8-online", KV_CACHE_DTYPE="fp8")
    assert out.returncode == 0, out.stderr
    assert "--quantization,fp8_per_tensor,--kv-cache-dtype,fp8" in out.stdout and "gcloud run deploy" in out.stdout
    gke = _dry(DEPLOY / "gcp/deploy-quantized.sh", TARGET="gke", SCHEME="w4a16")
    assert gke.returncode == 0 and "kubectl patch deployment vllm" in gke.stdout


def test_serving_lab_deployment_with_each_schemes_args_validates_for_1_34():
    yaml = pytest.importorskip("yaml")
    kv = pytest.importorskip("kubernetes_validate")
    manifest = SERVING / "deploy/gcp/gke/vllm.yaml"
    if not manifest.exists():
        pytest.skip("serving lab not in this checkout")
    dep = next(d for d in yaml.safe_load_all(manifest.read_text()) if d and d["kind"] == "Deployment")
    for scheme in ("bf16", "fp8-online", "w4a16"):
        d = copy.deepcopy(dep)
        d["spec"]["template"]["spec"]["containers"][0]["args"] = serve.gke_container_args(
            serve.plan(scheme, "L4", kv_cache_dtype="fp8"))
        kv.validate(d, "1.34.0", strict=True)
