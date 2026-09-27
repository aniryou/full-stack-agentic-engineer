"""Deploy assets and documented commands: strict, dry-runnable scripts; one pinned vLLM image; the flags each
script emits; every `python -m distillab ...` command in the READMEs, deploy docs, scripts and notebooks parses;
the Cloud Run tfvars only uses variables the 04 lab's Terraform declares."""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from distillab import draft as DR
from distillab.__main__ import build_parser

LAB = Path(__file__).resolve().parents[1]
DEPLOY = LAB / "deploy"
REPO = LAB.parents[2]
VLLM_PARSERS = {"qwen3", "deepseek_r1"}           # vLLM v0.30.0 parser names this lab uses (thinking-lab's fact sheet)
SCRIPTS = sorted(p.relative_to(DEPLOY).as_posix() for p in DEPLOY.rglob("*.sh"))
DOCS = [LAB / "README.md", *sorted(DEPLOY.rglob("*.md")), *sorted((LAB / "notebooks_src").glob("*.py"))]


def dry(script: str, cwd=None, **env) -> str:
    if not shutil.which("bash"):
        pytest.skip("no bash")
    e = {"DRY_RUN": "1", "PATH": os.environ["PATH"], "HOME": "/tmp", "MODE": "pip", **env}
    r = subprocess.run(["bash", str(DEPLOY / script)], capture_output=True, text=True, env=e, timeout=120, cwd=cwd)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_are_strict_and_support_dry_run(script):
    text = (DEPLOY / script).read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text and "DRY_RUN" in text
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(DEPLOY / script)]).returncode == 0


def test_serve_teacher_flags():
    out = dry("any-gpu/serve_teacher.sh")
    assert "--max-logprobs 20" in out and "vllm serve Qwen/Qwen2.5-1.5B-Instruct" in out and "--reasoning-parser" not in out
    assert "--reasoning-parser qwen3" in dry("any-gpu/serve_teacher.sh", MODEL="Qwen/Qwen3-1.7B")
    assert "--reasoning-parser deepseek_r1" in dry("any-gpu/serve_teacher.sh", MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


def test_serve_with_draft_emits_a_valid_speculative_config():
    out = dry("any-gpu/serve_with_draft.sh")
    spec = json.loads(re.search(r"speculative config: (\{.*\})", out).group(1))
    assert DR.speculative_config(spec.pop("model"), spec.pop("num_speculative_tokens"), spec.pop("method"), **spec)
    assert "--speculative-config" not in dry("any-gpu/serve_with_draft.sh", DRAFT="none")


def _spec_and_cmd(out: str):
    spec = json.loads(re.search(r"speculative config: (\{.*\})", out).group(1))
    cmd = next(shlex.split(line[2:]) for line in out.splitlines() if line.startswith("+ "))
    return spec, cmd


def test_a_local_draft_is_passed_where_vllm_will_find_it(tmp_path):
    """Docker: the draft directory is mounted and the spec names the container path (the image's workdir is
    /vllm-workspace, so a relative path would not resolve). pip: an absolute host path, found from the caller's
    directory or from the lab directory."""
    d = tmp_path / "draft-distilled"
    d.mkdir()
    spec, cmd = _spec_and_cmd(dry("any-gpu/serve_with_draft.sh", DRAFT=str(d), MODE="docker"))
    mounts = {m.split(":")[0]: m.split(":")[1] for i, a in enumerate(cmd) if a == "-v" for m in [cmd[i + 1]]}
    assert mounts.get(str(d)) == spec["model"] and spec["model"].startswith("/drafts/")
    spec, cmd = _spec_and_cmd(dry("any-gpu/serve_with_draft.sh", DRAFT="draft-distilled", cwd=tmp_path))
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-4B"] and spec["model"] == str(d)           # relative to the caller
    missing = dry("any-gpu/serve_with_draft.sh", DRAFT="_run_outputs/not-trained-yet")
    assert "no local draft" in missing                                                          # would be a hub id
    assert _spec_and_cmd(dry("any-gpu/serve_with_draft.sh"))[0]["model"] == "Qwen/Qwen3-0.6B"  # a hub id stays one


def _commands_of(out: str):
    return [shlex.split(line[2:]) for line in out.splitlines() if line.startswith("+ ") and "distillab" in line]


@pytest.mark.parametrize("method,batch,seq", [("sft", "4", "1024"), ("kd", "2", "512"), ("gkd", "2", "512")])
def test_train_student_plans_the_run_it_launches(method, batch, seq):
    cmds = _commands_of(dry("any-gpu/train_student.sh", METHOD=method))
    plan = next(c for c in cmds if "memory" in c)
    train = next(c for c in cmds if f"distillab.hf.{method}" in c)
    val = lambda c, flag: c[c.index(flag) + 1]
    assert (val(plan, "--batch"), val(plan, "--seq")) == (batch, seq) == (val(train, "--batch-size"), val(train, "--max-length"))
    big = _commands_of(dry("any-gpu/train_student.sh", METHOD=method, BATCH="1", MAX_LEN="256"))
    assert all(("1" in c and "256" in c) for c in big if "memory" in c or f"distillab.hf.{method}" in c)


def test_train_student_stages_keep_the_teacher_off_the_card_while_training():
    data = dry("any-gpu/train_student.sh", STAGE="data")
    assert "teacher-data" in data and "distillab.hf" not in data and "STAGE=train" in data
    train = dry("any-gpu/train_student.sh", STAGE="train")
    assert "teacher-data" not in train.replace("teacher_", "") and "distillab.hf.sft" in train
    both = dry("any-gpu/train_student.sh", STAGE="all")
    assert both.index("teacher-data") < both.index("distillab.hf.sft")
    assert [l[:6] for l in both.splitlines() if l.startswith(">> ")] == [">> 1/4", ">> 2/4", ">> 3/4", ">> 4/4"]


@pytest.mark.parametrize("method", ["sft", "kd", "gkd"])
def test_train_student_runs_the_lab_modules_with_flags_they_accept(method):
    out = dry("any-gpu/train_student.sh", METHOD=method)
    cmds = [shlex.split(line[2:]) for line in out.splitlines() if line.startswith("+ ") and "distillab" in line]
    assert any(f"distillab.hf.{method}" in c for c in cmds)
    for c in cmds:
        i = c.index("-m")
        mod, args = c[i + 1], c[i + 2:]
        if mod == "distillab":
            build_parser().parse_args(args)
        else:
            r = subprocess.run([sys.executable, "-m", mod, *args, "--dry-run"], cwd=LAB, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, (c, r.stderr)


def _commands(text: str):
    for line in text.splitlines():
        for m in re.finditer(r"python3? -m (distillab(?:\.hf\.\w+)?)((?: +(?:--?[\w-]+|[^\s`|&;#)]+))*)", line):
            yield m.group(1), shlex.split(m.group(2))


def test_every_documented_command_parses():
    seen = 0
    for doc in DOCS:
        for mod, args in _commands(doc.read_text()):
            if any(a in ("...", "<config>") or a.startswith("<") for a in args) or (mod != "distillab" and not args):
                continue                                          # a placeholder, or a module named in prose
            seen += 1
            if mod == "distillab":
                build_parser().parse_args(args or ["env"])
            else:
                r = subprocess.run([sys.executable, "-m", mod, *args, "--dry-run"], cwd=LAB, capture_output=True, text=True, timeout=60)
                assert r.returncode == 0, (doc.name, mod, args, r.stderr)
    assert seen >= 8


def test_image_tag_pinned_and_parser_names_valid():
    tags, parsers = set(), set()
    for p in DEPLOY.rglob("*"):
        if p.suffix in (".sh", ".md", ".example"):
            t = p.read_text()
            tags |= set(re.findall(r"vllm/vllm-openai:(v[\d.]+)", t))
            parsers |= set(re.findall(r'PARSER="([a-z0-9_]+)"', t)) - {"auto", "none"}
            parsers |= set(re.findall(r"--reasoning-parser[= ]\"?([a-z0-9_]+)", t))
    assert tags == {"v0.30.0"} and parsers and parsers <= VLLM_PARSERS, (tags, parsers)


def test_cloud_run_tfvars_match_the_04_module():
    tfvars = (DEPLOY / "gcp/cloud-run-teacher.tfvars.example").read_text()
    keys = set(re.findall(r"^([a-z_]+)\s*=", tfvars, re.M))
    assert {"model", "extra_args", "max_model_len", "concurrency", "request_timeout", "min_instances"} <= keys
    assert re.search(r"min_instances\s*=\s*0", tfvars) and '"--max-logprobs", "20"' in tfvars
    seqs = int(re.search(r'"--max-num-seqs", "(\d+)"', tfvars)[1])
    assert int(re.search(r"^concurrency\s*=\s*(\d+)", tfvars, re.M)[1]) <= seqs
    variables = REPO / "04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/variables.tf"
    if not variables.exists():
        pytest.skip("04 serving lab not in this checkout")
    declared = set(re.findall(r'variable "([a-z_]+)"', variables.read_text()))
    assert keys <= declared, keys - declared
