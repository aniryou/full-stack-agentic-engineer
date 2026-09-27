#!/usr/bin/env python3
"""Execute notebooks and report pass/fail.

    python tools/run_notebooks.py solutions            # all solutions must run clean
    python tools/run_notebooks.py notebooks --expect-fail   # exercises should stop at NotImplementedError

With --expect-fail a blank notebook passes only if it stops at an exercise: the first
error is raised in or after the first `# YOUR CODE HERE` cell. It fails as FAIL(env)
when it never reached one: a missing module (ModuleNotFoundError, ImportError), a
leftover `...` placeholder, or any error in a cell before the first exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
ENV_ERRORS = ("ModuleNotFoundError", "ImportError")


def first_cell(nb, pred):
    return next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and pred(c)), None)


def run(path: Path):
    """Execute one notebook. Returns (ok, env, at_exercise, message): env is True for a missing
    module or a leftover `...`; at_exercise is True when the first error was raised in or after
    the first `# YOUR CODE HERE` cell."""
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        return True, False, False, "ok"
    except CellExecutionError as e:
        msg = f"{e.ename}: {str(e.evalue)[:200]}"
        exercise = first_cell(nb, lambda c: "# YOUR CODE HERE" in c.source)
        failed = first_cell(nb, lambda c: any(o.get("output_type") == "error" for o in c.get("outputs", [])))
        env = e.ename in ENV_ERRORS or "ellipsis" in msg.lower()
        return False, env, None not in (exercise, failed) and failed >= exercise, msg


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = ROOT / argv[1]
    expect_fail = "--expect-fail" in argv
    paths = sorted(target.glob("*.ipynb")) if target.is_dir() else [target]
    bad = 0
    for p in paths:
        ok, env, at_exercise, msg = run(p)
        if expect_fail:
            # a blank exercise notebook must NOT run clean; it stops at the first unsolved
            # cell (NotImplementedError) or the check that depends on it (AssertionError).
            # Anything else -- a missing module, or an error before the first exercise -- is
            # the environment, not the learner's blank.
            env = env or not (ok or at_exercise)
            good = not ok and not env
            label = "FAIL(ran clean?)" if ok else "FAIL(env)" if env else "STOPS at exercise"
        else:
            good = ok
            label = "PASS" if ok else "FAIL(env)" if env else "FAIL"
        print(f"{label:>17}  {p.name}  {'' if ok else msg}")
        bad += 0 if good else 1
    print(f"\n{len(paths) - bad}/{len(paths)} {'behaved' if expect_fail else 'passed'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
