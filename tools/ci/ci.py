#!/usr/bin/env python3
"""Helpers shared by .github/workflows/tests.yml and tools/ci/run_local.sh (standard library only).

    python3 tools/ci/ci.py matrix [--solutions]   # JSON list of {id, dir, python} for the job matrix
    python3 tools/ci/ci.py run <id> <phase>       # phase: install | test | solutions, in the lab directory
    python3 tools/ci/ci.py check                  # every directory with tests is in labs.json, and each exists
    python3 tools/ci/ci.py builders               # every notebook builder, one path per line
    python3 tools/ci/ci.py bootstrap-targets      # notebooks the Colab injector owns (first cell tagged)
    python3 tools/ci/ci.py bootstrap-check        # the injector would not change what their setup cell does

The lab list and the commands live in tools/ci/labs.json, so CI and a laptop run the same thing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABS_FILE = REPO / "tools" / "ci" / "labs.json"
PHASES = ("install", "test", "solutions")
TAG = "colab-bootstrap"
# Test files outside a tests/ folder that are not tests of a lab (notebook sources whose names start with test_).
NOT_TESTS = ("notebooks_src/",)


def load_labs(path: Path = LABS_FILE) -> list[dict]:
    labs = json.loads(path.read_text(encoding="utf-8"))["labs"]
    ids = [lab["id"] for lab in labs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"duplicate lab ids in {path}: {dup}")
    return labs


def tracked(*patterns: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *patterns], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return [p for p in out.splitlines() if p]


def test_files(paths: list[str]) -> list[str]:
    """Paths that pytest would collect as test modules."""
    return [p for p in paths
            if Path(p).name.startswith("test_") and p.endswith(".py")
            and not any(marker in p for marker in NOT_TESTS)]


def uncovered(tests: list[str], lab_dirs: list[str], ignore: tuple[str, ...] = ("tools/ci/", "tools/site/")) -> list[str]:
    """Test files that no lab directory contains (tools/ci and tools/site run in their own jobs)."""
    dirs = [d.rstrip("/") + "/" for d in lab_dirs]
    return [t for t in tests if not t.startswith(ignore) and not any(t.startswith(d) for d in dirs)]


def cmd_matrix(argv: list[str]) -> int:
    labs = load_labs()
    if "--solutions" in argv:
        labs = [lab for lab in labs if lab.get("solutions")]
    print(json.dumps([{"id": lab["id"], "dir": lab["dir"], "python": lab["python"]} for lab in labs],
                     separators=(",", ":")))
    return 0


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def cmd_run(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in PHASES:
        raise SystemExit(f"usage: ci.py run <id> {{{'|'.join(PHASES)}}}")
    lab_id, phase = argv
    lab = next((lab for lab in load_labs() if lab["id"] == lab_id), None)
    if lab is None:
        raise SystemExit(f"no lab {lab_id!r} in {LABS_FILE}")
    command = lab.get(phase)
    if not command:
        print(f"{lab_id}: no {phase} step")
        return 0
    env = dict(os.environ)
    env.setdefault("RUNNER_TEMP", tempfile.gettempdir())
    print(f"$ cd {lab['dir']} && {command}", flush=True)
    rc = subprocess.run(["bash", "-euo", "pipefail", "-c", command], cwd=REPO / lab["dir"], env=env).returncode
    if rc == 0 and phase == "install" and torch_installed():
        print(f"{lab_id}: torch is importable after install; CI runs at T0 (no torch)", file=sys.stderr)
        return 1
    return rc


def cmd_check(argv: list[str]) -> int:
    labs = load_labs()
    problems = [f"missing lab directory: {lab['dir']}" for lab in labs if not (REPO / lab["dir"]).is_dir()]
    problems += [f"{lab['id']}: python {lab['python']} (only 3.11 and 3.12 are used)"
                 for lab in labs if lab["python"] not in ("3.11", "3.12")]
    problems += [f"test file in no lab of tools/ci/labs.json: {t}"
                 for t in uncovered(test_files(tracked("*.py")), [lab["dir"] for lab in labs])]
    for p in problems:
        print(p)
    print(f"{len(labs)} labs, {len(problems)} problems")
    return 1 if problems else 0


def builders() -> list[str]:
    return sorted(tracked("*build_notebooks.py", "*/embeddings-lab/build.py"))


def cmd_builders(argv: list[str]) -> int:
    print("\n".join(builders()))
    return 0


def first_cell_tagged(nb_path: Path) -> bool:
    cells = json.loads(nb_path.read_text(encoding="utf-8")).get("cells") or [{}]
    return TAG in (cells[0].get("metadata", {}).get("tags") or [])


def has_own_bootstrap(nb: str) -> bool:
    """Percent-source labs write their own bootstrap cell: a tools/build_notebooks.py sits in the lab."""
    return any((REPO / parent / "tools" / "build_notebooks.py").is_file() for parent in Path(nb).parents)


def cmd_bootstrap_targets(argv: list[str]) -> int:
    targets, orphans = [], []
    for nb in tracked("*.ipynb"):
        if first_cell_tagged(REPO / nb):
            targets.append(nb)
        elif not has_own_bootstrap(nb):
            orphans.append(nb)
    print("\n".join(targets))
    for nb in orphans:
        print(f"no Colab setup cell (run tools/inject_colab_bootstrap.py on it): {nb}", file=sys.stderr)
    return 1 if orphans else 0


def injector():
    spec = importlib.util.spec_from_file_location("inject_colab_bootstrap", REPO / "tools" / "inject_colab_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bootstrap_status(nb_path: Path, make_cell) -> str:
    """'same': re-running the injector leaves the file byte for byte; 'format': it would only rewrite the cell's
    id, execution_count or key order (the notebook was executed or re-saved after injection); 'stale': the setup
    cell's code or tags differ from what the injector writes today, or there is none."""
    raw = nb_path.read_text(encoding="utf-8")
    d = json.loads(raw)
    cells = d.get("cells") or [{}]
    if TAG not in (cells[0].get("metadata", {}).get("tags") or []):
        return "stale"
    want = make_cell(nb_path.resolve().parent.relative_to(REPO).as_posix())
    have = cells[0]
    src = lambda c: "".join(c["source"]) if isinstance(c.get("source"), list) else c.get("source", "")
    if src(have) != src(want) or have.get("metadata") != want["metadata"] or have.get("outputs"):
        return "stale"
    cells[0] = want
    return "same" if json.dumps(d, indent=1, ensure_ascii=False) + "\n" == raw else "format"


def cmd_bootstrap_check(argv: list[str]) -> int:
    """The injector's no-op check: every notebook it owns already carries today's setup cell."""
    make_cell = injector().make_cell
    targets = [nb for nb in tracked("*.ipynb") if first_cell_tagged(REPO / nb)]
    status = {nb: bootstrap_status(REPO / nb, make_cell) for nb in targets}
    stale = [nb for nb, s in status.items() if s == "stale"]
    fmt = [nb for nb, s in status.items() if s == "format"]
    for nb in fmt:
        print(f"::warning file={nb}::setup cell is current; re-running tools/inject_colab_bootstrap.py would only "
              "rewrite its id, execution_count or key order")
    for nb in stale:
        print(f"::error file={nb}::the Colab setup cell is out of date: python3 tools/inject_colab_bootstrap.py {nb}")
    print(f"{len(targets)} notebooks with the injector's setup cell: {len(targets) - len(stale) - len(fmt)} unchanged "
          f"by a re-run, {len(fmt)} formatting only, {len(stale)} stale")
    return 1 if stale else 0


COMMANDS = {"matrix": cmd_matrix, "run": cmd_run, "check": cmd_check, "builders": cmd_builders,
            "bootstrap-targets": cmd_bootstrap_targets, "bootstrap-check": cmd_bootstrap_check}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
