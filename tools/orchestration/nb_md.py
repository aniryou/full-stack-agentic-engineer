#!/usr/bin/env python3
"""Dump and apply the Markdown cells of hand-written notebooks (a blank and its solution twin together).

    python3 tools/orchestration/nb_md.py dump  notebooks/01_x.ipynb > cells.txt
    python3 tools/orchestration/nb_md.py apply cells.txt notebooks/01_x.ipynb solutions/01_x.ipynb

`dump` prints every Markdown cell between `=== cell <index> id=<id> ===` markers. `apply` writes the text under
each marker back into the n-th Markdown cell of every notebook given, by position among the Markdown cells (a blank
and its solution have the same Markdown cells, with different ids). Code cells, outputs, ids and metadata are left
alone; the file is written as the repo keeps it (`json.dumps(indent=1, ensure_ascii=False)` and a final newline), so
a round trip of an unchanged dump is a no-op. Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARK = re.compile(r"^=== cell (\d+) id=(\S*) ===\n", re.M)


def md_cells(d: dict) -> list[dict]:
    return [c for c in d["cells"] if c["cell_type"] == "markdown"]


def dump(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(d["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            out.append(f"=== cell {i} id={c.get('id', '')} ===\n{src}\n")
    return "".join(out)


def parse(text: str) -> list[str]:
    parts = MARK.split(text)
    # parts = [preamble, idx, id, body, idx, id, body, ...]
    bodies = parts[3::3]
    return [b[:-1] if b.endswith("\n") else b for b in bodies]


def apply(text_path: Path, targets: list[Path]) -> None:
    bodies = parse(text_path.read_text(encoding="utf-8"))
    for t in targets:
        raw = t.read_text(encoding="utf-8")
        d = json.loads(raw)
        cells = md_cells(d)
        if len(cells) != len(bodies):
            sys.exit(f"{t}: {len(cells)} Markdown cells, but {text_path} has {len(bodies)} blocks")
        for c, body in zip(cells, bodies):
            c["source"] = body.splitlines(keepends=True)
        out = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
        if out != raw:
            t.write_text(out, encoding="utf-8")
            print("wrote", t)
        else:
            print("unchanged", t)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "dump":
        sys.stdout.write(dump(Path(argv[1])))
        return 0
    if len(argv) >= 3 and argv[0] == "apply":
        apply(Path(argv[1]), [Path(p) for p in argv[2:]])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
