#!/usr/bin/env python3
"""Write per-layer 'Open in Colab' link sections into each layer README, and a
short setup guide to COLAB.md. Idempotent: the per-layer section lives between
<!-- colab-links:start/end --> markers and is replaced on each run.

Run after sorting new content:  python3 tools/gen_colab_index.py
"""
import glob, os, re
from collections import OrderedDict
from pathlib import Path

BRANCH, REPO = "main", "aniryou/full-stack-agentic-engineer"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
START, END = "<!-- colab-links:start -->", "<!-- colab-links:end -->"
# Notebooks with the answers filled in, in every convention the repo uses: a solutions/ or worked/ folder, or a
# name like 01_x_solution(s), 01_x_solved, 01_x_practice_solved. A name ending in _worked (01_x_worked, 01_worked)
# is an answer key only when an exercise twin sits beside it (01_x_worked next to 01_x_practice or 01_x, or the
# same number next to a *_practice/*_exercise notebook) and the folder keeps no solutions/ or worked/ folder of its
# own. Otherwise it is a worked lesson and stays an ordinary notebook: kv-cache's 01_kv_cache_worked comes before
# 02_kv_cache_practice (no twin), and long-running-agents-gcp reads 01..04_*_worked first, then the *_practice
# notebooks, whose answers are in notebooks/solutions/ (the answers live elsewhere).
# Kept identical to tools/site/build_site_content.py.
ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIRS = {"solutions", "worked"}
SOLUTION_STEM = re.compile(r"(?:^|[_\-.])(?:solutions?|solved)(?:$|[_\-.])", re.I)
WORKED_STEM = re.compile(r"(?:^|[_\-.])worked(?:$|[_\-.])", re.I)
EXERCISE_STEM = re.compile(r"(?:^|[_\-.])(?:practice|exercises?)(?:$|[_\-.])", re.I)
INTRO = ("One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Notebooks are listed by folder; "
         "*worked answers* marks the answer key of an exercise (in a `solutions/` or `worked/` folder, named "
         "`*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no "
         "`solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.")


def colab(p): return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{p}"

def _stem(path):
    return path.replace(os.sep, "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _folder(path):
    path = path.replace(os.sep, "/")
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _base(stem, token):
    """The stem with a worked/practice token taken out: 01_kv_cache_worked -> 01_kv_cache."""
    return re.sub(r"^[_\-.]+|[_\-.]+$", "", token.sub("_", stem))


def _number(stem):
    m = re.match(r"^\d+", stem)
    return m.group(0) if m else None


def answers_elsewhere(folder, siblings=None):
    """True when `folder` keeps its exercises' answers in a solutions/ or worked/ folder of its own (on disk, or
    among `siblings`, which may name any file): then a `*_worked` notebook beside them is a lesson."""
    if any((ROOT / folder / d).is_dir() for d in SOLUTION_DIRS):
        return True
    return any(p.replace(os.sep, "/").startswith(f"{folder}/{d}/") for p in siblings or () for d in SOLUTION_DIRS)


def has_exercise_twin(path, siblings=None):
    """True when a `*_worked` notebook has an exercise version in the same folder (see the rule above).
    `siblings` is any iterable of notebook paths; by default the folder is listed on disk."""
    folder = _folder(path)
    if siblings is None:
        d = ROOT / folder
        siblings = [f"{folder}/{n}" for n in os.listdir(d) if n.endswith(".ipynb")] if d.is_dir() else []
    base = _base(_stem(path), WORKED_STEM)
    for other in siblings:
        s = _stem(other)
        if _folder(other) != folder or SOLUTION_STEM.search(s) or WORKED_STEM.search(s):
            continue
        exercise = bool(EXERCISE_STEM.search(s))
        if s == base or (exercise and _base(s, EXERCISE_STEM) == base):
            return True
        if exercise and _number(base) and _number(s) == _number(base):
            return True
    return False


def is_solution(path, siblings=None):
    """True for a notebook with the answers in it (see the rule above SOLUTION_DIRS)."""
    parts = path.replace(os.sep, "/").split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    if SOLUTION_DIRS & set(parts[:-1]) or SOLUTION_STEM.search(stem):
        return True
    return (bool(WORKED_STEM.search(stem)) and has_exercise_twin(path, siblings)
            and not answers_elsewhere(_folder(path), siblings))


def layer_section(layer, nbs):
    body = ["## Run in Colab", "", INTRO, ""]
    if not nbs:
        body.append("_No notebooks yet._")
    else:
        groups = OrderedDict()
        for nb in nbs:
            groups.setdefault(os.path.relpath(os.path.dirname(nb), layer), []).append(nb)
        for rel, items in groups.items():
            body.append(f"**`{rel}/`**" if rel != '.' else "**(layer root)**")
            for nb in items:
                note = " — *worked answers*" if is_solution(nb, nbs) else ""
                body.append(f"- [![Open In Colab]({BADGE})]({colab(nb)}) `{os.path.basename(nb)}`{note}")
            body.append("")
    return START + "\n" + "\n".join(body).rstrip() + "\n" + END


def main():
    total = 0
    for layer in sorted(d for d in glob.glob('[0-9][0-9]-*') if os.path.isdir(d)):
        nbs = sorted(p for p in glob.glob(f'{layer}/**/*.ipynb', recursive=True)
                     if '.ipynb_checkpoints' not in p)
        total += len(nbs)
        section = layer_section(layer, nbs)
        readme = Path(layer) / "README.md"
        t = readme.read_text() if readme.exists() else f"# {layer}\n"
        if START in t and END in t:
            t = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: section, t, flags=re.S)
        else:
            t = t.rstrip() + "\n\n" + section + "\n"
        readme.write_text(t)
        print(f"  {layer}: {len(nbs)} notebooks")

    Path("COLAB.md").write_text('''# Running notebooks in Google Colab

Every notebook in this repo opens directly in Colab: click the **"Open in Colab"** badge next to it in
its layer's `README.md`. The repo is public, so there is nothing to set up. Each notebook's first cell
(tagged `colab-bootstrap`) is a no-op locally; on Colab it clones this repo, `cd`s into the notebook's
folder and pip-installs that lab's dependencies.

## Keeping your work
Colab opens a fresh copy from GitHub each time. To keep your edits, use *File -> Save a copy in Drive*
(or *Save a copy in GitHub* into your own fork).

## Redo an exercise
On Colab, reopen the notebook from its badge. Locally, `git restore <notebook>` returns it to the
committed blank; for percent-source labs, re-run the lab's `python3 tools/build_notebooks.py`.

_Per-layer notebook links are generated by `tools/gen_colab_index.py`._
''')
    print(f"COLAB.md: setup guide only ({total} notebooks linked across layer READMEs)")


if __name__ == "__main__":
    main()
