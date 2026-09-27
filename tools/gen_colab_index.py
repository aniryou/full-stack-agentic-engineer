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
# One notebook layout in every lab (tools/ci/nb_layout.py guards it): <lab>/notebooks/ holds what a learner opens --
# exercise blanks and lessons -- and <lab>/solutions/<name>.ipynb is the worked answer to notebooks/<name>.ipynb.
# A notebook is an answer key exactly when its folder is solutions/. Kept identical to tools/site/build_site_content.py.
ANSWERS_DIR = "solutions"
INTRO = ("One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook "
         "in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer "
         "to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.")
# One name per layer, used everywhere (layer README H1s, the root README, CURRICULUM.md, the site).
# Kept identical to LAYER_TITLES in tools/site/build_site_content.py.
LAYER_NAMES = {
    "00": "00 · Foundations",
    "01": "01 · Hardware and fabric",
    "02": "02 · CUDA, NCCL and runtime",
    "03": "03 · Kubernetes and GPU scheduling",
    "04": "04 · Inference engine",
    "05": "05 · Orchestrator",
    "06": "06 · Gateway",
    "07": "07 · Agents and applications",
}
# The two folders a lab keeps its notebooks in: a notebook is listed under the lab folder above them.
NOTEBOOK_DIRS = {"notebooks", ANSWERS_DIR}


def colab(p): return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{p}"

def _stem(path):
    return path.replace(os.sep, "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _number(stem):
    m = re.match(r"^\d+", stem)
    return m.group(0) if m else None


def is_solution(path):
    """True for a worked answer key: a notebook directly inside a solutions/ folder."""
    parts = path.replace(os.sep, "/").split("/")
    return len(parts) >= 2 and parts[-2] == ANSWERS_DIR


def lab_of(rel_dir):
    """The lab a notebook folder belongs to: the folder above its notebooks/ or solutions/ folder."""
    parts = [p for p in rel_dir.replace(os.sep, "/").split("/") if p not in ("", ".")]
    if parts and parts[-1] in NOTEBOOK_DIRS:
        parts.pop()
    return "/".join(parts)


def _label(rel):
    """A notebook's link text: its path inside the lab without the .ipynb and a leading notebooks/ folder."""
    rel = rel[:-len(".ipynb")] if rel.endswith(".ipynb") else rel
    return rel[len("notebooks/"):] if rel.startswith("notebooks/") else rel


def _answer_labels(rels):
    """Answer keys are labelled by their number when the numbers are unique (01 · 02 · ...), else by name."""
    stems = [_stem(r) for r in rels]
    nums = [_number(s) for s in stems]
    if all(nums) and len(set(nums)) == len(nums):
        return nums
    return stems


def layer_section(layer, nbs):
    body = ["## Run in Colab", "", INTRO, ""]
    if not nbs:
        body.append("_No notebooks yet._")
    else:
        labs = OrderedDict()
        for nb in nbs:
            lab = lab_of(os.path.relpath(os.path.dirname(nb), layer))
            labs.setdefault(lab, []).append(nb)
        for lab, items in sorted(labs.items()):
            base = f"{layer}/{lab}" if lab else layer
            rels = [os.path.relpath(nb, base).replace(os.sep, "/") for nb in items]
            lessons = [(r, nb) for r, nb in zip(rels, items) if not is_solution(nb)]
            answers = [(r, nb) for r, nb in zip(rels, items) if is_solution(nb)]
            line = f"- **`{lab}/`**" if lab else "- **(layer root)**"
            line += " — " + " · ".join(f"[{_label(r)}]({colab(nb)})" for r, nb in lessons) if lessons else ""
            if answers:
                labels = _answer_labels([r for r, _ in answers])
                line += " — *answers:* " + " · ".join(
                    f"[{lab_}]({colab(nb)})" for lab_, (_, nb) in zip(labels, answers))
            body.append(line)
    return START + "\n" + "\n".join(body).rstrip() + "\n" + END


# Folders that hold copies, caches or run outputs, never lessons (gitignored; a lab's tests write executed copies
# of its notebooks to _run_outputs/). Kept in step with SKIP_DIRS in tools/site/build_site_content.py.
SKIP_DIRS = {".ipynb_checkpoints", "_run_outputs", ".venv", "venv", "node_modules", "site_build", ".git"}


def layer_notebooks(layer):
    """The layer's notebooks, sorted, leaving out anything under a SKIP_DIRS folder."""
    return sorted(p for p in glob.glob(f'{layer}/**/*.ipynb', recursive=True)
                  if not SKIP_DIRS & set(p.replace(os.sep, "/").split("/")[:-1]))


def main():
    total, per_layer = 0, []
    for layer in sorted(d for d in glob.glob('[0-9][0-9]-*') if os.path.isdir(d)):
        nbs = layer_notebooks(layer)
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
        per_layer.append((layer, len(nbs)))

    layer_list = "\n".join(
        f"- [{LAYER_NAMES.get(layer[:2], layer)}]({layer}/README.md#run-in-colab) — {n} notebooks"
        for layer, n in per_layer)

    Path("COLAB.md").write_text(f'''# Running notebooks in Google Colab

Every notebook in this repo opens directly in Colab: click its link in the *Run in Colab* section of
its layer's `README.md`. The repo is public, so there is nothing to set up. Each notebook's first cell
(tagged `colab-bootstrap`) is a no-op locally; on Colab it clones this repo, `cd`s into the notebook's
folder and pip-installs that lab's dependencies.

## Where the links are
Each layer README ends with a *Run in Colab* section: one line per lab, exercises and lessons first, then the worked
answers. Every lab has the same two folders: `notebooks/` for what you open (exercise blanks, lessons and
walkthroughs) and `solutions/` for the worked answer to a blank, under the same file name as the blank.

{layer_list}

## Keeping your work
Colab opens a fresh copy from GitHub each time. To keep your edits, use *File -> Save a copy in Drive*
(or *Save a copy in GitHub* into your own fork).

## Redo an exercise
On Colab, reopen the notebook from its link. Locally, `git restore <notebook>` returns it to the
committed blank; for percent-source labs, re-run the lab's `python3 tools/build_notebooks.py`.

_Per-layer notebook links are generated by `tools/gen_colab_index.py`._
''')
    print(f"COLAB.md: setup guide only ({total} notebooks linked across layer READMEs)")


if __name__ == "__main__":
    main()
