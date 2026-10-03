#!/usr/bin/env python3
"""Execute worked notebooks in place and record their outputs, the way the committed solutions carry them.

    python3 tools/orchestration/nb_exec.py <solution.ipynb> ...

Each notebook runs with its own folder as the working directory (as scripts/run_notebooks.py runs it). Only the
outputs, execution counts and per-cell execution timestamps of code cells, and the notebook's language_info, are
written back; cell sources, ids and the file's JSON layout (`json.dumps(indent=1, ensure_ascii=False)` and a final
newline) stay as they are, so a builder's rebuild-is-a-no-op check still passes. Needs nbformat, nbclient and
ipykernel. A notebook whose metadata lists optional modules (``"lra": {"requires": [...]}``) is skipped when one is
not installed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def execute(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    d = json.loads(raw)
    missing = [m for m in d.get("metadata", {}).get("lra", {}).get("requires", []) if not installed(m)]
    if missing:
        return f"SKIP {path} (needs {', '.join(missing)})"
    nb = nbformat.reads(raw, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    executed = json.loads(nbformat.writes(nb))
    assert len(executed["cells"]) == len(d["cells"])
    for old, new in zip(d["cells"], executed["cells"]):
        if old["cell_type"] != "code":
            continue
        old["outputs"] = new.get("outputs", [])
        old["execution_count"] = new.get("execution_count")
        if "execution" in new.get("metadata", {}):
            old.setdefault("metadata", {})["execution"] = new["metadata"]["execution"]
    if "language_info" in executed.get("metadata", {}):
        d["metadata"]["language_info"] = executed["metadata"]["language_info"]
    out = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    if out != raw:
        path.write_text(out, encoding="utf-8")
        return f"WROTE {path}"
    return f"UNCHANGED {path}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for a in argv:
        try:
            print(execute(Path(a)))
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"FAIL {a}: {type(e).__name__}: {str(e)[:300]}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
