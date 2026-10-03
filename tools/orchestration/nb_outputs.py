#!/usr/bin/env python3
"""Carry recorded outputs over to a rebuilt notebook whose code cells did not change.

    python3 tools/orchestration/nb_outputs.py <with-outputs.ipynb> <rebuilt.ipynb>
    python3 tools/orchestration/nb_outputs.py --from-git HEAD <rebuilt.ipynb> ...   # the source is that git revision's copy

A notebook builder rewrites a changed notebook without outputs. When only Markdown cells changed, the outputs the
committed solution recorded are still the outputs of its code cells, so this copies them back: for each code cell
of the rebuilt notebook whose source is identical to the source of the source notebook's cell at the same position,
the outputs, execution count and execution timestamps are copied, and the notebook's language_info with them. Cells
whose source differs keep no outputs (print which). Written as the repo keeps notebooks (``json.dumps(indent=1,
ensure_ascii=False)`` and a final newline). Standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def source(c: dict) -> str:
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def transplant(src: dict, dst_path: Path) -> str:
    raw = dst_path.read_text(encoding="utf-8")
    dst = json.loads(raw)
    s_code = [c for c in src["cells"] if c["cell_type"] == "code"]
    d_code = [c for c in dst["cells"] if c["cell_type"] == "code"]
    if len(s_code) != len(d_code):
        return f"SKIP {dst_path}: {len(d_code)} code cells, source has {len(s_code)}"
    kept, changed = 0, []
    for i, (a, b) in enumerate(zip(s_code, d_code)):
        if source(a) != source(b):
            changed.append(i)
            continue
        b["outputs"] = a.get("outputs", [])
        b["execution_count"] = a.get("execution_count")
        if "execution" in a.get("metadata", {}):
            b.setdefault("metadata", {})["execution"] = a["metadata"]["execution"]
        kept += 1
    if "language_info" in src.get("metadata", {}):
        dst["metadata"]["language_info"] = src["metadata"]["language_info"]
    out = json.dumps(dst, indent=1, ensure_ascii=False) + "\n"
    note = f" (code changed in cells {changed}: no outputs for them)" if changed else ""
    if out != raw:
        dst_path.write_text(out, encoding="utf-8")
        return f"WROTE {dst_path}: outputs of {kept} code cells{note}"
    return f"UNCHANGED {dst_path}{note}"


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "--from-git":
        rev, targets = argv[1], argv[2:]
        for t in targets:
            rel = subprocess.run(["git", "ls-files", "--full-name", t], capture_output=True, text=True).stdout.strip()
            if not rel:
                print(f"SKIP {t}: not tracked")
                continue
            proc = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"SKIP {t}: {proc.stderr.strip()}")
                continue
            print(transplant(json.loads(proc.stdout), Path(t)))
        return 0
    if len(argv) == 2:
        print(transplant(json.loads(Path(argv[0]).read_text(encoding="utf-8")), Path(argv[1])))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
