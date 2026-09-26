#!/usr/bin/env python3
"""Build exercise + solution notebooks from percent-format sources.

Source cells in ``notebooks_src/NN_name.py``:

    # %% [markdown]      prose (each line prefixed with "# ")
    # %%                 a normal code cell
    # %% exercise        a cell with ### BEGIN SOLUTION / ### END SOLUTION blocks
    # %% check           a self-test cell (kept in both variants)

For each source, writes ``notebooks/NN_name.ipynb`` (solution blocks replaced by
``# YOUR CODE HERE`` + NotImplementedError) and ``solutions/NN_name.ipynb`` (blocks
kept). A bootstrap cell makes ``import gpusim`` work from a fresh checkout.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC, EX, SOL = ROOT / "notebooks_src", ROOT / "notebooks", ROOT / "solutions"
BEGIN, END = "### BEGIN SOLUTION", "### END SOLUTION"

BOOTSTRAP = '''# bootstrap: Colab clone + local import of `gpusim` (auto-inserted)
import sys, pathlib
if "google.colab" in sys.modules:
    import os, subprocess
    _slug = "aniryou/full-stack-agentic-engineer"
    _repo = pathlib.Path("/content/full-stack-agentic-engineer")
    if not _repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{_slug}.git", str(_repo)], check=True)
    os.chdir(_repo / "02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-core")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
_r = pathlib.Path.cwd().resolve()
while _r != _r.parent and not (_r / "gpusim").exists():
    _r = _r.parent
if str(_r) not in sys.path:
    sys.path.insert(0, str(_r))
del _r'''
BANNER = ("> **Exercise cells** contain `# YOUR CODE HERE` — replace it, then run the **Check** cell below it. "
          "A check prints ✅ when it passes. The finished version is in `solutions/`.")


def parse(text: str):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if kind == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in src.splitlines()).strip("\n")
        if src.strip():
            cells.append((kind, src))
    for line in text.splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            flush()
            rest = m.group(1).strip()
            kind = "markdown" if rest.startswith("[markdown]") else rest.split()[0] if rest else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def strip_solution(src: str) -> str:
    out, skip, indent = [], False, ""
    for l in src.splitlines():
        if l.strip() == BEGIN:
            skip, indent = True, l[: len(l) - len(l.lstrip())]
            out += [f"{indent}# YOUR CODE HERE", f'{indent}raise NotImplementedError("your turn")']
        elif l.strip() == END:
            skip = False
        elif not skip:
            out.append(l)
    return "\n".join(out)


def keep_solution(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if l.strip() not in (BEGIN, END))


def build(path: Path):
    cells = parse(path.read_text())
    for variant, out_dir in (("exercise", EX), ("solution", SOL)):
        nb = new_notebook()
        nb.metadata.update(kernelspec={"name": "python3", "display_name": "Python 3", "language": "python"},
                           language_info={"name": "python"})
        nb.cells.append(new_code_cell(BOOTSTRAP))
        shown = False
        for kind, src in cells:
            if kind == "markdown":
                nb.cells.append(new_markdown_cell(src))
                if not shown and variant == "exercise":
                    nb.cells.append(new_markdown_cell(BANNER))
                    shown = True
            elif kind == "exercise":
                nb.cells.append(new_code_cell(strip_solution(src) if variant == "exercise" else keep_solution(src)))
            else:
                nb.cells.append(new_code_cell(keep_solution(src)))
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / f"{path.stem}.ipynb")
    print("built", path.stem)


def main(argv):
    srcs = sorted(SRC.glob("*.py")) if len(argv) < 2 else [Path(a) for a in argv[1:]]
    for s in srcs:
        build(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
