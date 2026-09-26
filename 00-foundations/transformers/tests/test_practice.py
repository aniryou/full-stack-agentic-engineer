"""The attention practice notebooks: the solutions run, and a blank stops at its own check with a clear
NotImplementedError (not an AttributeError on `...`). Runs the cells in-process; needs numpy and pytest only."""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
PRACTICE = Path(__file__).resolve().parents[1] / "practice"
MARK = "# YOUR CODE HERE"


def code_cells(name):
    return ["".join(c["source"]) for c in json.loads((PRACTICE / name).read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]


def run(cells):
    ns = {}
    for i, src in enumerate(cells):
        try:
            exec(compile(src, f"cell {i}", "exec"), ns)
        except Exception as e:
            return i, e
    return None, None


def test_solutions_run_clean():
    assert run(code_cells("attention_solutions.ipynb")) == (None, None)


def test_blank_stops_at_the_first_check_with_a_clear_message():
    cells = code_cells("attention_practice.ipynb")
    first = next(i for i, s in enumerate(cells) if MARK in s)
    at, err = run(cells)
    assert at == first + 1 and isinstance(err, NotImplementedError), (at, repr(err))
    assert "softmax() still has `...` blanks" in str(err)


def test_each_blank_alone_stops_its_own_check():
    prac, sol = code_cells("attention_practice.ipynb"), code_cells("attention_solutions.ipynb")
    exercises = [i for i, s in enumerate(prac) if MARK in s]
    assert len(exercises) == 7 and len(prac) == len(sol)
    for i in exercises:
        at, err = run(sol[:i] + [prac[i]] + sol[i + 1:])
        assert at == i + 1 and isinstance(err, NotImplementedError), (i, at, repr(err))
