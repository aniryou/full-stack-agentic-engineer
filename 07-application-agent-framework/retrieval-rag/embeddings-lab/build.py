"""Build pipeline: src/*.py (jupytext percent) -> executed notebooks.
Worked notebooks -> notebooks/ ; solved exercises -> solutions/ (executed)
and exercises/ (solutions stripped, unexecuted)."""
import re, sys, pathlib
import jupytext, nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).parent

def strip_solutions(text):
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s == "# >>> SOLUTION":
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + "# " + "=" * 12 + " YOUR CODE HERE " + "=" * 12)
            out.append(indent + "raise NotImplementedError(\"implement me, then re-run\")")
            skip = True
        elif s == "# <<< SOLUTION":
            skip = False
        elif not skip:
            out.append(line)
    return "\n".join(out) + "\n"

def clean(text):
    keep = [l for l in text.splitlines()
            if l.strip() not in ("# >>> SOLUTION", "# <<< SOLUTION")]
    return "\n".join(keep) + "\n"

def execute(nb, cwd):
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(cwd)}}).execute()
    return nb

def build_worked(name):
    text = (ROOT / "src" / f"{name}.py").read_text()
    nb = jupytext.reads(text, fmt="py:percent")
    execute(nb, ROOT / "notebooks")
    nbformat.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
    print(f"  notebooks/{name}.ipynb ok")

def build_exercise(name):
    text = (ROOT / "src" / f"{name}_solved.py").read_text()
    sol = jupytext.reads(clean(text), fmt="py:percent")
    execute(sol, ROOT / "solutions")
    nbformat.write(sol, ROOT / "solutions" / f"{name}_solutions.ipynb")
    ex = jupytext.reads(strip_solutions(text), fmt="py:percent")
    nbformat.write(ex, ROOT / "exercises" / f"{name}.ipynb")
    print(f"  solutions/{name}_solutions.ipynb ok + exercises/{name}.ipynb")

if __name__ == "__main__":
    targets = sys.argv[1:] or ["01_counts_to_vectors", "02_contrastive_bi_encoder",
                               "03_geometry", "04_vector_search",
                               "05_retrieval_pipeline", "06_superposition",
                               "ex01", "ex02", "ex03", "ex04", "ex05", "ex06"]
    for t in targets:
        if t.startswith("ex"):
            build_exercise(t)
        else:
            build_worked(t)
