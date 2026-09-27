"""tools/ci/nb_layout.py: one notebook layout in every lab, and `ci.py check` fails on a deviation."""
from __future__ import annotations

import importlib.util
import posixpath
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("nb_layout", REPO / "tools" / "ci" / "nb_layout.py")
nbl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nbl)
_spec = importlib.util.spec_from_file_location("ci", REPO / "tools" / "ci" / "ci.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)
J = posixpath.join   # the old layouts' folder names are built, so they do not read as paths in the tree


def suffixed(stem: str, suffix: str) -> str:
    return f"{stem}_{suffix}.ipynb"


GOOD = ["07-x/lab/notebooks/01_loop.ipynb", "07-x/lab/solutions/01_loop.ipynb",
        "07-x/lab/notebooks/00_lesson.ipynb",                      # a lesson: no twin needed
        "04-x/kv-cache/notebooks/01_kv_cache_worked.ipynb",        # a worked lesson, no blank twin beside it
        "04-x/kv-cache/notebooks/02_kv_cache_practice.ipynb"]      # a blank whose answers live in tests/code


def test_the_layout_itself_has_no_deviations():
    assert nbl.deviations(GOOD) == []
    rows = {r[1]: r for r in nbl.inventory(GOOD)}
    assert rows["notebooks/01_loop.ipynb"] == ("07-x/lab", "notebooks/01_loop.ipynb", "notebook", "solutions/01_loop.ipynb")
    assert rows["solutions/01_loop.ipynb"][2:] == ("solution", "notebooks/01_loop.ipynb")
    assert rows["notebooks/00_lesson.ipynb"][3] == "-"


@pytest.mark.parametrize("path, why", [
    ("06-x/core/core_walkthrough.ipynb", "outside"),                                   # a lab's top level
    (J("06-x", "lab", "notebooks", "practice", "01_x_practice.ipynb"), "outside"),       # nested in notebooks/
    (J("06-x", "lab", "notebooks", "solutions", "01_x.ipynb"), "outside"),
    (J("07-x", "lab", "exercises", "ex01.ipynb"), "outside"),
    (J("07-x", "lab", "worked", "00_idea.ipynb"), "outside"),
    (J("00-x", "cap", "notebooks", suffixed("01_practice", "solved")), "suffix"),         # an answer beside its blank
    (J("07-x", "lab", "solutions", suffixed("ex01", "solutions")), "suffix"),
    ("07-x/lab/solutions/02_orphan.ipynb", "without a blank"),                          # no notebooks/02_orphan
])
def test_every_old_convention_is_a_deviation(path, why):
    found = nbl.deviations(GOOD + [path])
    assert [p for p, _ in found] and all(p == path for p, _ in found), found
    assert any(why in w for _, w in found), found


def test_a_worked_twin_beside_its_blank_is_a_deviation_but_a_worked_lesson_is_not():
    twin = ["07-x/a/notebooks/01_loop_practice.ipynb", "07-x/a/notebooks/01_loop_worked.ipynb"]
    assert [p for p, _ in nbl.deviations(twin)] == ["07-x/a/notebooks/01_loop_worked.ipynb"]
    assert nbl.deviations(["07-x/a/notebooks/01_loop_worked.ipynb", "07-x/a/notebooks/02_next_practice.ipynb"]) == []


def test_a_documented_lessons_folder_is_accepted(monkeypatch):
    lessons = J("07-x", "lab", "lessons")
    assert nbl.deviations([J(lessons, "01.ipynb")])
    monkeypatch.setattr(nbl, "LESSON_DIRS", {lessons: "why it is kept"})
    assert nbl.deviations([J(lessons, "01.ipynb")]) == []


def test_the_committed_tree_follows_the_layout():
    paths = nbl.tracked_notebooks()
    assert len(paths) > 300
    assert nbl.deviations(paths) == []
    assert all(nbl.classify(p)[1] in ("notebook", "solution") for p in paths)


def test_check_fails_on_a_misplaced_notebook(monkeypatch, capsys):
    monkeypatch.setattr(ci, "nb_layout", lambda: nbl)
    monkeypatch.setattr(nbl, "tracked_notebooks", lambda: GOOD + ["07-x/lab/solutions/03_alone.ipynb"])
    assert ci.cmd_check([]) == 1
    assert "notebook layout: 07-x/lab/solutions/03_alone.ipynb: a solution without a blank" in capsys.readouterr().out
    monkeypatch.setattr(nbl, "tracked_notebooks", lambda: GOOD)
    assert ci.cmd_check([]) == 0
