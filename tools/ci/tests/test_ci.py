"""tools/ci: the lab list covers every test, the workflow stays pinned and T0, the helpers behave."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("ci", REPO / "tools" / "ci" / "ci.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"


def test_labs_json_is_complete_and_consistent():
    labs = ci.load_labs()
    assert len(labs) >= 30
    for lab in labs:
        assert (REPO / lab["dir"]).is_dir(), lab["dir"]
        assert lab["install"] and lab["test"], lab["id"]
        assert lab["python"] in ("3.11", "3.12"), lab["id"]
        assert "torch" not in lab["install"].replace("grep -viE '^[[:space:]]*torch'", ""), lab["id"]
    # every lab runs in its own environment; one package named `scalelab` (the Mistral provider is inside it);
    # lra-gcp runs three times: the default install, with the ADK extra, and on Python 3.12 with the Mistral extra
    ids = {lab["id"] for lab in labs}
    assert {"agentic-scaling-lab", "lra-core", "lra-gcp", "lra-gcp-adk", "lra-gcp-mistral-py312"} <= ids


def test_one_pyproject_declares_each_package_name():
    """Two labs installing the same package name cannot share an environment (the old scalelab clash)."""
    files = subprocess.run(["git", "ls-files", "*pyproject.toml"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    names: dict[str, list[str]] = {}
    for f in files:
        m = re.search(r'(?m)^name\s*=\s*"([^"]+)"', (REPO / f).read_text())
        if m:
            names.setdefault(m.group(1), []).append(f)
    assert {n: fs for n, fs in names.items() if len(fs) > 1} == {}
    assert names.get("scalelab") == ["06-gateway/scaling-admission-cost/agentic-scaling-lab/pyproject.toml"]


def test_check_passes_on_the_committed_tree():
    r = subprocess.run([sys.executable, str(REPO / "tools/ci/ci.py"), "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_uncovered_reports_a_test_outside_every_lab():
    tests = ci.test_files(["a/lab/tests/test_x.py", "b/other/test_y.py", "a/lab/notebooks_src/test_time.py",
                           "a/lab/tests/conftest.py", "tools/site/tests/test_z.py"])
    assert tests == ["a/lab/tests/test_x.py", "b/other/test_y.py", "tools/site/tests/test_z.py"]
    assert ci.uncovered(tests, ["a/lab"]) == ["b/other/test_y.py"]
    assert ci.uncovered(tests, ["a/lab", "b/other"]) == []
    assert ci.uncovered(["a/lab-2/tests/test_q.py"], ["a/lab"]) == ["a/lab-2/tests/test_q.py"]  # prefix is a directory


def test_without_solutions_flags_a_lab_with_a_runner_and_no_solutions_step():
    runners = ["a/lab/tools/run_notebooks.py", "b/lab/scripts/run_notebooks.py", "c/lab/sub/tools/run_notebooks.py"]
    labs = [{"id": "a", "dir": "a/lab"}, {"id": "b", "dir": "b/lab", "solutions": "python scripts/run_notebooks.py"},
            {"id": "c", "dir": "c/lab"}, {"id": "d", "dir": "d/lab"}]
    # a: runner, no step -> flagged; b: has a step; c: the runner belongs to a nested lab; d: no runner
    assert ci.without_solutions(runners, labs) == ["a"]
    assert ci.without_solutions(runners, [{"id": "a2", "dir": "a/lab/"}]) == ["a2"]  # trailing slash


def test_builders_are_all_found():
    found = ci.builders()
    assert "07-application-agent-framework/retrieval-rag/embeddings-lab/build.py" in found
    assert "07-application-agent-framework/retrieval-rag/rag-from-scratch/tools_build_notebooks.py" in found
    # 26 before the durable-agent labs were consolidated and the Mistral copy of agent-core folded into agent-core
    # (each took one builder with it)
    assert len(found) >= 24


def test_matrix_output_is_compact_json():
    out = subprocess.run([sys.executable, str(REPO / "tools/ci/ci.py"), "matrix"], capture_output=True, text=True,
                         check=True).stdout
    assert "\n" not in out.strip()  # one line for $GITHUB_OUTPUT
    rows = json.loads(out)
    assert {"id", "dir", "python"} == set(rows[0])
    sol = json.loads(subprocess.run([sys.executable, str(REPO / "tools/ci/ci.py"), "matrix", "--solutions"],
                                    capture_output=True, text=True, check=True).stdout)
    assert 0 < len(sol) < len(rows)


def test_run_fails_when_the_command_fails(tmp_path, monkeypatch):
    labs = {"labs": [{"id": "x", "dir": "tools/ci", "python": "3.11", "install": "true", "test": "exit 3"}]}
    (tmp_path / "labs.json").write_text(json.dumps(labs))
    monkeypatch.setattr(ci, "LABS_FILE", tmp_path / "labs.json")
    monkeypatch.setattr(ci, "load_labs", lambda path=tmp_path / "labs.json": json.loads(path.read_text())["labs"])
    assert ci.cmd_run(["x", "test"]) == 3
    monkeypatch.setattr(ci, "torch_installed", lambda: False)
    assert ci.cmd_run(["x", "install"]) == 0
    monkeypatch.setattr(ci, "torch_installed", lambda: True)
    assert ci.cmd_run(["x", "install"]) == 1  # T0: an install that brings torch in fails
    assert ci.cmd_run(["x", "solutions"]) == 0  # no solutions step: nothing to do


def test_workflow_actions_are_pinned_and_t0():
    yaml = pytest.importorskip("yaml")
    text = WORKFLOW.read_text()
    uses = re.findall(r"uses:\s*(\S+)(.*)", text)
    assert uses
    for ref, comment in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref
        assert re.search(r"#\s*v\d", comment), f"{ref}: add the release tag as a comment"
    assert "torch" not in text.replace("no torch", "")
    wf = yaml.safe_load(text)
    on = wf.get("on", wf.get(True))
    assert {"pull_request", "push", "workflow_dispatch"} <= set(on)
    jobs = wf["jobs"]
    assert jobs["tests"]["strategy"]["fail-fast"] is False
    assert jobs["solutions"]["if"] == "github.event_name == 'workflow_dispatch'"
    for name in ("notebooks", "docs", "colab-index", "lab-list-check"):
        assert name in jobs
    # the matrix is emitted even when the lab list is incomplete: one missing entry must not skip every lab's tests
    assert not any("ci.py check" in (step.get("run") or "") for step in jobs["matrix"]["steps"])
    assert "ci.py check" in "".join(step.get("run") or "" for step in jobs["lab-list-check"]["steps"])


def test_no_nested_workflows():
    nested = [p for p in subprocess.run(["git", "ls-files", "*/.github/workflows/*"], cwd=REPO, capture_output=True,
                                        text=True, check=True).stdout.splitlines()]
    assert nested == []


def test_shell_helpers_are_committed_executable():
    # the docs say `tools/ci/run_local.sh ...`; a 100644 blob fails with "Permission denied" on a fresh checkout
    staged = subprocess.run(["git", "ls-files", "-s", "tools/ci/*.sh"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.splitlines()
    modes = {line.split("\t", 1)[1]: line.split()[0] for line in staged}
    assert "tools/ci/run_local.sh" in modes
    assert {path: mode for path, mode in modes.items() if mode != "100755"} == {}


def test_bootstrap_status_same_format_stale():
    make_cell = ci.injector().make_cell
    nb = REPO / "tools" / "ci" / "tests" / "_tmp_bootstrap.ipynb"
    rel = nb.parent.relative_to(REPO).as_posix()

    def write(first):
        body = {"cells": [first, {"cell_type": "markdown", "id": "m", "metadata": {}, "source": ["hi"]}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        nb.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        write(make_cell(rel))
        assert ci.bootstrap_status(nb, make_cell) == "same"
        executed = dict(sorted(make_cell(rel).items()))          # re-saved by Jupyter: sorted keys, a count
        executed["execution_count"] = 1
        write(executed)
        assert ci.bootstrap_status(nb, make_cell) == "format"
        old = make_cell(rel)
        old["source"] = old["source"][:-1]                        # an older template
        write(old)
        assert ci.bootstrap_status(nb, make_cell) == "stale"
        elsewhere = make_cell("some/other/folder")                # copied from another folder: wrong chdir
        write(elsewhere)
        assert ci.bootstrap_status(nb, make_cell) == "stale"
        untagged = make_cell(rel)
        untagged["metadata"] = {}
        write(untagged)
        assert ci.bootstrap_status(nb, make_cell) == "stale"
    finally:
        nb.unlink(missing_ok=True)


def test_every_injected_notebook_is_current():
    r = subprocess.run([sys.executable, str(REPO / "tools/ci/ci.py"), "bootstrap-check"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
