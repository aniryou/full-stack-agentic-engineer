#!/usr/bin/env python3
"""Insert an idempotent, Colab-aware bootstrap cell at the top of notebooks.

Usage (from the mono-repo root):
    python3 tools/inject_colab_bootstrap.py <path> [<path> ...]

<path> is a notebook or a directory (searched recursively, skipping
.ipynb_checkpoints). The injected cell is tagged 'colab-bootstrap'; re-running
REPLACES that cell instead of duplicating it, and never touches a lab's own
cells. Outside Colab the cell is a no-op, so local runs are unchanged.

The repo is public, so on Colab the cell does a plain shallow `git clone`, then
cd's into the notebook's dir and pip-installs the nearest lab.
"""
import json, sys
from pathlib import Path

REPO_SLUG = "aniryou/full-stack-agentic-engineer"
REPO_DIR  = "full-stack-agentic-engineer"
TAG = "colab-bootstrap"

TEMPLATE = '''# --- Colab setup (auto-inserted; no-op outside Colab). tag: colab-bootstrap ---
import sys
if "google.colab" in sys.modules:
    import os, subprocess, pathlib
    _slug = "__REPO_SLUG__"
    _root = pathlib.Path("/content") / "__REPO_DIR__"
    if not _root.exists():
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{_slug}.git", str(_root)], check=True)
    os.chdir(_root / "__NB_REL__")
    for _c in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (_c / "pyproject.toml").exists() or (_c / "setup.py").exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(_c)]); break
        if (_c / "requirements.txt").exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(_c / "requirements.txt")]); break
        if _c == _root:
            break
    if str(pathlib.Path.cwd()) not in sys.path:
        sys.path.insert(0, str(pathlib.Path.cwd()))
'''

def make_cell(nb_rel):
    src = (TEMPLATE.replace("__REPO_SLUG__", REPO_SLUG)
                   .replace("__REPO_DIR__", REPO_DIR)
                   .replace("__NB_REL__", nb_rel))
    return {"cell_type": "code", "id": "colab-setup", "metadata": {"tags": [TAG]},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

def iter_nbs(paths):
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".ipynb":
            yield p
        elif p.is_dir():
            for nb in sorted(p.rglob("*.ipynb")):
                if ".ipynb_checkpoints" not in nb.parts:
                    yield nb

def main(argv):
    root = Path.cwd().resolve()
    n = 0
    for nb in iter_nbs(argv):
        rel = nb.resolve().parent.relative_to(root).as_posix()
        d = json.loads(nb.read_text(encoding="utf-8"))
        cells = d.setdefault("cells", [])
        if cells and TAG in (cells[0].get("metadata", {}).get("tags", []) or []):
            cells[0] = make_cell(rel); act = "replaced"
        else:
            cells.insert(0, make_cell(rel)); act = "inserted"
        nb.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        n += 1
    print(f"done: {n} notebooks")

if __name__ == "__main__":
    main(sys.argv[1:])
