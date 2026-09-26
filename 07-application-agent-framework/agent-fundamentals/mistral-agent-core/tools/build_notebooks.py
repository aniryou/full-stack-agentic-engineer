#!/usr/bin/env python3
"""Build exercise + solution notebooks from percent-format sources.

Source cells in ``notebooks_src/NN_name.py``:

    # %% [markdown]      prose (each line prefixed with "# ")
    # %%                 a normal code cell
    # %% exercise        a cell with ### BEGIN SOLUTION / ### END SOLUTION blocks
    # %% check           a self-test cell (kept in both variants)

For each source, writes ``notebooks/NN_name.ipynb`` (solution blocks replaced by
``# YOUR CODE HERE`` + NotImplementedError) and ``solutions/NN_name.ipynb`` (blocks
kept). Every notebook starts with the repo's Colab setup cell, exactly as
``tools/inject_colab_bootstrap.py`` writes it (so running the injector afterwards
is a no-op), then a bootstrap cell that makes ``import agentcore`` work from a
fresh checkout. Cell ids are stable: rebuilding unchanged sources is a no-op.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC, EX, SOL = ROOT / "notebooks_src", ROOT / "notebooks", ROOT / "solutions"
BEGIN, END = "### BEGIN SOLUTION", "### END SOLUTION"
REPO = next(p for p in ROOT.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file())
_spec = importlib.util.spec_from_file_location("inject_colab_bootstrap", REPO / "tools" / "inject_colab_bootstrap.py")
_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inject)

BOOTSTRAP = (
    "# make `import agentcore` work from anywhere (auto-inserted)\n"
    "import sys, pathlib\n"
    "_r = pathlib.Path.cwd().resolve()\n"
    "while _r != _r.parent and not (_r / 'agentcore').exists():\n"
    "    _r = _r.parent\n"
    "sys.path.insert(0, str(_r))"
)
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


def write_nb(nb, out: Path) -> None:
    """Write ``nb`` with the Colab setup cell first, in the injector's JSON layout."""
    d = json.loads(nbformat.writes(nb))
    d["cells"].insert(0, _inject.make_cell(out.resolve().parent.relative_to(REPO).as_posix()))
    out.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


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
        for i, cell in enumerate(nb.cells):   # stable ids: rebuilding unchanged sources is a no-op
            cell.id = hashlib.sha1(f"{path.stem}/{i}".encode()).hexdigest()[:12]
        out_dir.mkdir(parents=True, exist_ok=True)
        write_nb(nb, out_dir / f"{path.stem}.ipynb")
    print("built", path.stem)


def main(argv):
    srcs = sorted(SRC.glob("*.py")) if len(argv) < 2 else [Path(a) for a in argv[1:]]
    for s in srcs:
        build(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
