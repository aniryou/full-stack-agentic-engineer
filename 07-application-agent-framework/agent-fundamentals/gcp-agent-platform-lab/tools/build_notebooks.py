#!/usr/bin/env python3
"""Build exercise and solution notebooks from percent-format sources.

Source files live in ``notebooks_src/NN_name.py`` and use the "percent" cell
format::

    # %% [markdown]
    # # A heading
    # Some prose.

    # %%
    print("a normal code cell")

    # %% exercise
    def area(r):
        ### BEGIN SOLUTION
        return 3.14159 * r * r
        ### END SOLUTION

    # %% check
    assert round(area(1), 2) == 3.14

Two notebooks are produced per source:

* ``notebooks/NN_name.ipynb``  – the solution blocks are replaced by
  ``# YOUR CODE HERE`` + ``raise NotImplementedError()``; check cells stay so the
  learner gets immediate feedback.
* ``solutions/NN_name.ipynb`` – the same notebook with the solutions in place
  (marker lines removed). ``tools/run_notebooks.py solutions`` executes these
  end to end, which is how the repository proves every exercise is solvable.

A bootstrap cell is inserted at the top of every notebook so ``import agentlab``
works from a fresh checkout without ``pip install -e .``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src"
OUT_EXERCISES = ROOT / "notebooks"
OUT_SOLUTIONS = ROOT / "solutions"

CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")
BEGIN = "### BEGIN SOLUTION"
END = "### END SOLUTION"

BOOTSTRAP = '''# bootstrap: Colab (private clone via GH_TOKEN) + local import of `agentlab` (auto-inserted)
import sys, pathlib
if "google.colab" in sys.modules:
    import os, subprocess
    _slug = "aniryou/full-stack-agentic-engineer"
    _repo = pathlib.Path("/content/full-stack-agentic-engineer")
    if not _repo.exists():
        _tok = ""
        try:
            from google.colab import userdata
            _tok = userdata.get("GH_TOKEN") or ""
        except Exception:
            _tok = ""
        if not _tok:
            print("WARNING: no 'GH_TOKEN' Colab secret; add a GitHub token (repo scope) as Colab secret 'GH_TOKEN', then re-run.")
        _url = (f"https://{_tok}@github.com/{_slug}.git" if _tok else f"https://github.com/{_slug}.git")
        subprocess.run(["git", "clone", "--depth", "1", _url, str(_repo)], check=True)
        subprocess.run(["git", "-C", str(_repo), "remote", "set-url", "origin", f"https://github.com/{_slug}.git"])
    os.chdir(_repo / "07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
_r = pathlib.Path.cwd().resolve()
while _r != _r.parent and not (_r / "agentlab").exists():
    _r = _r.parent
if str(_r) not in sys.path:
    sys.path.insert(0, str(_r))
del _r'''

EXERCISE_BANNER = (
    "> **How to use this notebook.** Run the cells top to bottom. Cells marked **Exercise** contain\n"
    "> `# YOUR CODE HERE` — replace it with your implementation, then run the **Check** cell below it.\n"
    "> A check that passes prints nothing or ✅; a failing assertion tells you what is still wrong.\n"
    "> The fully solved version is in `solutions/` — try the exercise first, then compare.\n"
)


def parse_percent(text: str) -> list[tuple[str, str]]:
    """Return a list of (kind, source) where kind ∈ {markdown, code, exercise, check}."""
    cells: list[tuple[str, str]] = []
    kind: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if kind == "markdown":
            lines = []
            for ln in src.splitlines():
                if ln.startswith("# "):
                    lines.append(ln[2:])
                elif ln.strip() == "#":
                    lines.append("")
                else:
                    lines.append(ln)
            src = "\n".join(lines).strip("\n")
        if src.strip():
            cells.append((kind, src))

    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            rest = m.group("rest").strip()
            if rest.startswith("[markdown]"):
                kind = "markdown"
            elif rest.startswith("exercise"):
                kind = "exercise"
            elif rest.startswith("check"):
                kind = "check"
            else:
                kind = "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def strip_solution(src: str) -> str:
    """Replace BEGIN/END SOLUTION blocks with a placeholder at the same indentation."""
    out: list[str] = []
    in_block = False
    indent = ""
    for ln in src.splitlines():
        if ln.strip() == BEGIN:
            in_block = True
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}# YOUR CODE HERE")
            out.append(f'{indent}raise NotImplementedError("replace this with your implementation")')
            continue
        if ln.strip() == END:
            in_block = False
            continue
        if not in_block:
            out.append(ln)
    return "\n".join(out)


def keep_solution(src: str) -> str:
    return "\n".join(ln for ln in src.splitlines() if ln.strip() not in (BEGIN, END))


def build_one(src_path: Path) -> tuple[Path, Path]:
    cells = parse_percent(src_path.read_text())
    title = src_path.stem
    for variant, out_dir in (("exercise", OUT_EXERCISES), ("solution", OUT_SOLUTIONS)):
        nb = new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["agentlab"] = {"variant": variant, "source": src_path.name}
        first_md_done = False
        nb.cells.append(new_code_cell(BOOTSTRAP, metadata={"tags": ["bootstrap"]}))
        for kind, src in cells:
            if kind == "markdown":
                nb.cells.append(new_markdown_cell(src))
                if not first_md_done and variant == "exercise":
                    nb.cells.append(new_markdown_cell(EXERCISE_BANNER))
                first_md_done = True
            elif kind == "exercise":
                body = strip_solution(src) if variant == "exercise" else keep_solution(src)
                if BEGIN not in src:
                    print(f"  warning: exercise cell without solution markers in {src_path.name}", file=sys.stderr)
                nb.cells.append(new_code_cell(body, metadata={"tags": ["exercise"]}))
            elif kind == "check":
                nb.cells.append(new_code_cell(keep_solution(src), metadata={"tags": ["check"]}))
            else:
                nb.cells.append(new_code_cell(keep_solution(src), metadata={"tags": []}))
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{title}.ipynb"
        nbformat.write(nb, out)
    return OUT_EXERCISES / f"{title}.ipynb", OUT_SOLUTIONS / f"{title}.ipynb"


def main(argv: list[str]) -> int:
    sources = sorted(SRC.glob("*.py")) if len(argv) < 2 else [Path(a) for a in argv[1:]]
    if not sources:
        print("no sources found in", SRC)
        return 1
    for s in sources:
        ex, sol = build_one(s)
        print(f"built {ex.relative_to(ROOT)}  and  {sol.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
