#!/usr/bin/env python3
"""The notebook layout every lab follows, and a guard for it (standard library only).

    <lab>/notebooks/<name>.ipynb    what a learner opens: exercise blanks, lessons and walkthroughs
    <lab>/solutions/<name>.ipynb    the worked answer to the blank notebooks/<name>.ipynb (the same file name)

Nothing else: no other notebook folder (no nesting such as notebooks/solutions/), no notebook at a lab's top level,
and no answer key kept beside its blank under a suffix (_solved, _solution(s), or a _worked twin of a blank). A
notebook in notebooks/ without a twin in solutions/ is fine: a lesson, or a blank whose answer is in the lab's tests.

    python3 tools/ci/nb_layout.py            # every tracked notebook: lab, role, twin; then every deviation
    python3 tools/ci/nb_layout.py --check    # deviations only; exit 1 if there is any (run by `ci.py check`)
"""
from __future__ import annotations

import posixpath
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BLANKS, ANSWERS = "notebooks", "solutions"
# Folders, other than notebooks/, where a lab keeps lessons by design: {folder: why}. None today; add one here (and
# say so in the lab's README) rather than teaching the generators another convention.
LESSON_DIRS: dict[str, str] = {}
SKIP = {".ipynb_checkpoints", "_run_outputs"}
ANSWER_SUFFIX = re.compile(r"[_\-.](?:solved|solutions?)$", re.I)
WORKED_SUFFIX = re.compile(r"[_\-.]worked$", re.I)
PRACTICE_SUFFIX = re.compile(r"[_\-.](?:practice|exercises?)$", re.I)


def tracked_notebooks() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", "*.ipynb"], cwd=REPO, check=True, capture_output=True, text=True)
    return sorted(p for p in out.stdout.splitlines() if p and not SKIP & set(p.split("/")[:-1]))


def _stem(path: str) -> str:
    return posixpath.basename(path)[: -len(".ipynb")]


def classify(path: str) -> tuple[str, str]:
    """(lab folder, role) of one notebook path; role is 'notebook', 'solution', 'lesson' or 'misplaced'."""
    folder = posixpath.dirname(path)
    parent, grand = posixpath.basename(folder), posixpath.basename(posixpath.dirname(folder))
    if folder in LESSON_DIRS:
        return posixpath.dirname(folder), "lesson"
    if parent in (BLANKS, ANSWERS) and grand not in (BLANKS, ANSWERS):
        return posixpath.dirname(folder), "notebook" if parent == BLANKS else "solution"
    # Misplaced: name the lab as the folder above any notebook-ish folders, for the inventory's sake.
    parts = folder.split("/")
    while len(parts) > 1 and parts[-1] in {BLANKS, ANSWERS, "practice", "worked", "exercises", "lessons"}:
        parts.pop()
    return "/".join(parts), "misplaced"


def twin(path: str, paths: set[str]) -> str | None:
    """The other half of a blank/answer pair, if the tree has it (lab-relative)."""
    lab, role = classify(path)
    name = posixpath.basename(path)
    other = {"notebook": ANSWERS, "solution": BLANKS}.get(role)
    if other and f"{lab}/{other}/{name}" in paths:
        return f"{other}/{name}"
    return None


def deviations(paths: list[str]) -> list[tuple[str, str]]:
    """[(path, what is wrong)] for every notebook that breaks the layout above."""
    have = set(paths)
    out = []
    for p in paths:
        lab, role = classify(p)
        stem, folder = _stem(p), posixpath.dirname(p)
        if role == "misplaced":
            out.append((p, f"outside {BLANKS}/ and {ANSWERS}/ (lab {lab or '.'})"))
            continue
        if role == "solution" and not twin(p, have):
            out.append((p, f"a solution without a blank of the same name in {BLANKS}/"))
        if ANSWER_SUFFIX.search(stem):
            out.append((p, "an answer key named with a suffix: move it to solutions/ under its blank's name"))
        elif role == "notebook" and WORKED_SUFFIX.search(stem):
            base = WORKED_SUFFIX.sub("", stem)
            if {f"{folder}/{base}.ipynb", f"{folder}/{base}_practice.ipynb"} & have:
                out.append((p, "a _worked twin beside its blank: move it to solutions/ under the blank's name"))
    return out


def inventory(paths: list[str]) -> list[tuple[str, str, str, str]]:
    """[(lab, lab-relative path, role, twin)] for every notebook."""
    have = set(paths)
    rows = []
    for p in paths:
        lab, role = classify(p)
        rows.append((lab, p[len(lab) + 1:] if lab else p, role, twin(p, have) or "-"))
    return rows


def main(argv: list[str]) -> int:
    paths = tracked_notebooks()
    if "--check" not in argv:
        rows = inventory(paths)
        widths = [max(len(r[i]) for r in rows) for i in range(3)]
        for r in rows:
            print(f"{r[0]:<{widths[0]}}  {r[1]:<{widths[1]}}  {r[2]:<{widths[2]}}  {r[3]}")
        print()
    bad = deviations(paths)
    for p, why in bad:
        print(f"notebook layout: {p}: {why}")
    print(f"{len(paths)} notebooks, {len(bad)} deviations from the notebooks/ + solutions/ layout")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
