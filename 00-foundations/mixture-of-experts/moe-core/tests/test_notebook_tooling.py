"""Regression tests for this lab's notebook tooling (the same file sits in every lab with a notebook builder).

- The builder is deterministic (stable cell ids), so rebuilding unchanged sources leaves the tree clean.
- Every notebook it writes starts with a Colab setup cell. A cell tagged ``colab-bootstrap`` is exactly
  the one ``tools/inject_colab_bootstrap.py`` writes, so running the injector afterwards changes nothing.
- The committed solution-side notebooks are what the builder writes (exercise notebooks are left out:
  learners edit them in place).
- ``run_notebooks.py --expect-fail`` passes a blank that stops at its exercise and reports a missing module,
  a leftover ``...`` or an error before the first exercise as a failure, exiting non-zero.

The builder runs in a temporary copy of the lab, never in the checkout. Standard library + pytest; the
runner tests also need the lab's nbformat, nbclient and ipykernel.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
REPO = next((p for p in LAB.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file()), None)
BUILDER = next(p for d in ("tools", "scripts", "practice") if (p := LAB / d / "build_notebooks.py").is_file())
RUNNER = next((p for d in ("tools", "scripts") if (p := LAB / d / "run_notebooks.py").is_file()), None)
SKIP = {".git", ".venv", "venv", "_run_outputs", ".ipynb_checkpoints", "__pycache__", "node_modules"}


def notebooks(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*.ipynb"))
            if not SKIP & set(p.relative_to(root).parts)}


def build(lab: Path) -> dict[str, bytes]:
    proc = subprocess.run([sys.executable, str(lab / BUILDER.relative_to(LAB))], cwd=lab,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return notebooks(lab)


def injector():
    spec = importlib.util.spec_from_file_location("inject_colab_bootstrap", REPO / "tools" / "inject_colab_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """A fresh build in a copy of the lab at the same repo-relative path, with every notebook deleted first."""
    assert REPO is not None, "run from a checkout of the repo (tools/inject_colab_bootstrap.py not found)"
    repo = tmp_path_factory.mktemp("repo")
    (repo / "tools").mkdir()
    shutil.copy2(REPO / "tools" / "inject_colab_bootstrap.py", repo / "tools")
    lab = repo / LAB.relative_to(REPO)
    shutil.copytree(BUILDER.parent, lab / BUILDER.parent.name, ignore=shutil.ignore_patterns("*.ipynb", "__pycache__"))
    if (LAB / "notebooks_src").is_dir():
        shutil.copytree(LAB / "notebooks_src", lab / "notebooks_src", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in notebooks(LAB):   # a checkout's directory layout, without the notebooks
        (lab / rel).parent.mkdir(parents=True, exist_ok=True)
    return repo, lab, build(lab)


def test_builder_writes_the_committed_notebooks(fresh):
    _, _, built = fresh
    assert built, "the builder wrote no notebooks"
    assert set(built) <= set(notebooks(LAB)), "the builder writes notebooks that are not in the lab"


def test_rebuild_is_a_noop(fresh):
    _, lab, built = fresh
    assert build(lab) == built, "rebuilding over the builder's own output changed it (random cell ids?)"
    for rel in built:
        (lab / rel).unlink()
    assert build(lab) == built, "two builds from scratch differ (random cell ids?)"


def test_cell_ids_are_unique(fresh):
    _, _, built = fresh
    for rel, raw in built.items():
        d = json.loads(raw)
        ids = [c["id"] for c in d["cells"] if "id" in c]
        assert len(set(ids)) == len(ids), rel
        if (d["nbformat"], d["nbformat_minor"]) >= (4, 5):   # cell ids are required from nbformat 4.5
            assert len(ids) == len(d["cells"]), rel


def test_every_notebook_starts_with_a_colab_setup_cell(fresh):
    _, _, built = fresh
    make_cell, lab_rel = injector().make_cell, LAB.relative_to(REPO).as_posix()
    for rel, raw in built.items():
        first = json.loads(raw)["cells"][0]
        src = "".join(first["source"])
        assert first["cell_type"] == "code" and "google.colab" in src, f"{rel}: no Colab setup cell"
        if "colab-bootstrap" in first.get("metadata", {}).get("tags", []):
            nb_dir = (LAB / rel).parent.relative_to(REPO).as_posix()
            assert first == make_cell(nb_dir), f"{rel}: differs from tools/inject_colab_bootstrap.py's cell"
        else:
            assert re.search(r'os\.chdir\(_repo / "([^"]+)"\)', src).group(1) == lab_rel, f"{rel}: Colab cell cds elsewhere"


def test_injector_is_a_noop_on_the_build(fresh):
    repo, lab, built = fresh
    tagged = [rel for rel, raw in built.items()
              if "colab-bootstrap" in json.loads(raw)["cells"][0].get("metadata", {}).get("tags", [])]
    if not tagged:
        pytest.skip("percent-source bootstrap cell: the injector is not run on this lab")
    proc = subprocess.run([sys.executable, "tools/inject_colab_bootstrap.py", lab.relative_to(repo).as_posix()],
                          cwd=repo, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert notebooks(lab) == built, "running the injector after a build changed the notebooks"


def content(raw: bytes):
    """What a rebuild decides: cell types, ids, sources and metadata -- not outputs or execution records."""
    d = json.loads(raw)
    cells = [(c["cell_type"], c.get("id"), "".join(c["source"]) if isinstance(c["source"], list) else c["source"],
              {k: v for k, v in c.get("metadata", {}).items() if k != "execution"}) for c in d["cells"]]
    meta = {k: v for k, v in d["metadata"].items() if k != "language_info"}
    return cells, meta, d["nbformat"], d["nbformat_minor"]


def test_committed_solutions_match_a_rebuild(fresh):
    _, _, built = fresh
    has_solutions = any(rel.startswith("solutions/") for rel in built)
    solution_side = [rel for rel in built if "practice" not in Path(rel).stem
                     and not rel.startswith("notebooks/practice/")
                     and not (has_solutions and rel.startswith("notebooks/"))]
    assert solution_side
    committed = notebooks(LAB)
    stale = [rel for rel in solution_side if content(committed[rel]) != content(built[rel])]
    assert not stale, f"stale against the builder: {stale}; rebuild with {BUILDER.relative_to(LAB)}"


# --- run_notebooks.py --expect-fail --------------------------------------------------------------

CASES = {  # name: (cells, what --expect-fail must report)
    "a_stops_at_exercise": (["x = 1", "# YOUR CODE HERE\nraise NotImplementedError('your turn')", "assert x == 2"], "stop"),
    "b_missing_module": (["import nonexistent_module_xyz", "# YOUR CODE HERE\nraise NotImplementedError('your turn')"], "env"),
    "c_leftover_ellipsis": (["x = 1", "# YOUR CODE HERE\ny = ...\ny.shape"], "env"),
    "c_leftover_ellipsis_arithmetic": (["x = 1", "# YOUR CODE HERE\ny = ...\nz = y + 1"], "env"),
    "c_leftover_ellipsis_len": (["x = 1", "# YOUR CODE HERE\ny = ...\nlen(y)"], "env"),
    "c_leftover_ellipsis_format": (["x = 1", "# YOUR CODE HERE\ny = ...\nprint(f'{y:.2f}')"], "env"),
    "c_leftover_ellipsis_round": (["x = 1", "# YOUR CODE HERE\ny = ...\nround(y, 2)"], "env"),
    "c_leftover_ellipsis_math": (["import math", "# YOUR CODE HERE\ny = ...\nmath.sqrt(y)"], "env"),
    "d_error_before_exercise": (["1 / 0", "# YOUR CODE HERE\nraise NotImplementedError('your turn')"], "fail"),
    "e_runs_clean": (["x = 1", "# YOUR CODE HERE\nx = 2", "assert x == 2"], "fail"),
}
expect_fail = pytest.mark.skipif(RUNNER is None or "--expect-fail" not in RUNNER.read_text(),
                                 reason="this lab's runner executes the worked notebooks only")


def run_notebooks(*args):
    proc = subprocess.run([sys.executable, str(RUNNER), *map(str, args)], cwd=LAB,
                          capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    nbformat = pytest.importorskip("nbformat")
    d = tmp_path_factory.mktemp("notebook-tooling-cases")
    for name, (sources, _) in CASES.items():
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(nb, d / f"{name}.ipynb")
    yield d
    shutil.rmtree(LAB / "_run_outputs" / d.name, ignore_errors=True)   # gcp-agent-platform-lab keeps executed copies


@expect_fail
def test_expect_fail_tells_an_exercise_stop_from_an_environment_failure(cases):
    code, out = run_notebooks(cases, "--expect-fail")
    assert code != 0, out
    lines = {name: next(l for l in out.splitlines() if f"{name}.ipynb" in l) for name in CASES}
    for name, (_, want) in CASES.items():
        line = lines[name].strip()
        if want == "stop":
            assert not line.startswith("FAIL") and re.search(r"STOPS at exercise|stops as expected", line), line
        elif want == "env":
            assert line.startswith("FAIL(env)"), line
        else:
            assert line.startswith("FAIL"), line


@expect_fail
def test_expect_fail_exit_codes(cases):
    code, out = run_notebooks(cases / "a_stops_at_exercise.ipynb", "--expect-fail")
    assert code == 0, out
    code, out = run_notebooks(cases / "b_missing_module.ipynb", "--expect-fail")
    assert code != 0 and "FAIL(env)" in out and "ModuleNotFoundError" in out, out
    code, out = run_notebooks(cases / "b_missing_module.ipynb")
    assert code != 0 and "FAIL(env)" in out, out
