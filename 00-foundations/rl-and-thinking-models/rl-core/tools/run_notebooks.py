#!/usr/bin/env python3
"""Execute notebooks and report pass/fail.

    python tools/run_notebooks.py solutions            # all solutions must run clean
    python tools/run_notebooks.py notebooks --expect-fail   # exercises should stop at NotImplementedError
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path):
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        return True, "ok"
    except CellExecutionError as e:
        return False, f"{e.ename}: {str(e.evalue)[:200]}"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = ROOT / argv[1]
    expect_fail = "--expect-fail" in argv
    paths = sorted(target.glob("*.ipynb")) if target.is_dir() else [target]
    bad = 0
    for p in paths:
        ok, msg = run(p)
        if expect_fail:
            # a blank exercise notebook must NOT run clean; it stops at the first unsolved
            # cell (NotImplementedError) or the check that depends on it (AssertionError).
            good = not ok
            print(f"{'PASS(stops)' if good else 'FAIL(ran clean?)':>16}  {p.name}  {msg if good else ''}")
            bad += 0 if good else 1
        else:
            print(f"{'PASS' if ok else 'FAIL':>12}  {p.name}  {'' if ok else msg}")
            bad += 0 if ok else 1
    print(f"\n{len(paths) - bad}/{len(paths)} {'behaved' if expect_fail else 'passed'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
