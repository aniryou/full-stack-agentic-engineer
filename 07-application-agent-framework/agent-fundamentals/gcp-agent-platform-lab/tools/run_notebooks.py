#!/usr/bin/env python3
"""Execute every notebook in a directory and report failures.

    python tools/run_notebooks.py solutions          # prove all solutions run clean
    python tools/run_notebooks.py notebooks --expect-fail  # exercises should stop at NotImplementedError

Executed copies (with outputs) are written to ``_run_outputs/<dir>/`` so you can
inspect what a solved notebook prints without opening Jupyter.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path, timeout: int = 600) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
        ok, msg = True, "ok"
    except CellExecutionError as e:
        ok, msg = False, f"cell failed: {e.ename}: {str(e.evalue)[:300]}"
    out_dir = ROOT / "_run_outputs" / path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_dir / path.name)
    return ok, msg


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = ROOT / argv[1]
    expect_fail = "--expect-fail" in argv
    paths = sorted(target.glob("*.ipynb")) if target.is_dir() else [target]
    failures = 0
    for p in paths:
        t0 = time.time()
        ok, msg = run(p)
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        if expect_fail:
            status = "PASS(stops as expected)" if not ok and "NotImplementedError" in msg else ("FAIL(ran to completion?)" if ok else f"FAIL({msg})")
            if status.startswith("FAIL"):
                failures += 1
        elif not ok:
            failures += 1
        print(f"{status:>28}  {p.relative_to(ROOT)}  ({dt:.1f}s)  {'' if ok else msg}")
    print(f"\n{len(paths) - failures}/{len(paths)} notebooks {'behaved' if expect_fail else 'passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
