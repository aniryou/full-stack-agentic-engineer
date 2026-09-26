"""Build pipeline: src/*.py (jupytext percent) -> executed notebooks.
Worked notebooks -> notebooks/ ; solved exercises -> solutions/ (executed)
and exercises/ (solutions stripped, unexecuted).

The build is reproducible: cell ids are derived from the notebook name and the
cell's position, no execution timestamps are recorded, consecutive stream
outputs are merged, and the kernel runs single-threaded BLAS with a fixed hash
seed. Every notebook starts with the repo's Colab setup cell, exactly as
tools/inject_colab_bootstrap.py writes it and in the injector's JSON layout, so
rebuilding unchanged sources (with the versions in
tools/ci/embeddings-lab-build.txt, which CI installs) rewrites the committed
notebooks byte for byte and running the injector afterwards is a no-op."""
import hashlib, importlib.util, json, os, sys, pathlib
import jupytext, nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file())
_spec = importlib.util.spec_from_file_location("inject_colab_bootstrap",
                                               REPO / "tools" / "inject_colab_bootstrap.py")
_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inject)

# The kernel inherits this environment: same numbers and the same output on every run.
for _k, _v in {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

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
    NotebookClient(nb, timeout=1200, kernel_name="python3", record_timing=False,
                   coalesce_streams=True,
                   resources={"metadata": {"path": str(cwd)}}).execute()
    return nb

def write(nb, path):
    """Stable ids, no interpreter version, bootstrap cell first, injector's JSON layout."""
    for i, cell in enumerate(nb.cells):
        cell.id = hashlib.sha1(f"{path.parent.name}/{path.stem}/{i}".encode()).hexdigest()[:12]
    nb.metadata.get("language_info", {}).pop("version", None)
    d = json.loads(nbformat.writes(nb))
    d["cells"].insert(0, _inject.make_cell(path.parent.relative_to(REPO).as_posix()))
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

def build_worked(name):
    text = (ROOT / "src" / f"{name}.py").read_text()
    nb = jupytext.reads(text, fmt="py:percent")
    execute(nb, ROOT / "notebooks")
    write(nb, ROOT / "notebooks" / f"{name}.ipynb")
    print(f"  notebooks/{name}.ipynb ok")

def build_exercise(name):
    text = (ROOT / "src" / f"{name}_solved.py").read_text()
    sol = jupytext.reads(clean(text), fmt="py:percent")
    execute(sol, ROOT / "solutions")
    write(sol, ROOT / "solutions" / f"{name}_solutions.ipynb")
    ex = jupytext.reads(strip_solutions(text), fmt="py:percent")
    write(ex, ROOT / "exercises" / f"{name}.ipynb")
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
