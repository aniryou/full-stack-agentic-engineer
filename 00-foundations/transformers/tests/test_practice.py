"""The attention practice notebooks: the solutions run, and a blank stops at its own check with a clear
NotImplementedError (not an AttributeError on `...`). Runs the cells in-process; needs numpy and pytest only."""
import json
import linecache
from pathlib import Path

import pytest

pytest.importorskip("numpy")
PRACTICE = Path(__file__).resolve().parents[1] / "practice"
MARK = "# YOUR CODE HERE"


def code_cells(name):
    return ["".join(c["source"]) for c in json.loads((PRACTICE / name).read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]


def run(cells):
    """Run cells like a kernel: IPython keeps each cell's source in linecache, so inspect.getsource works."""
    ns = {}
    for i, src in enumerate(cells):
        name = f"<cell {i}: {hash(src)}>"
        linecache.cache[name] = (len(src), None, src.splitlines(True), name)
        try:
            exec(compile(src, name, "exec"), ns)
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


# Right answers that index with numpy's Ellipsis (x[...], x[..., :]) are not blanks: each passes its check.
ELLIPSIS_ANSWERS = {
    "def softmax": """def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x[...])
    return e / e.sum(axis=axis, keepdims=True)""",
    "def causal_mask": """def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))[..., :]""",
    "def split_heads": """def split_heads(x, h):
    n, d = x.shape
    return x.reshape(n, h, d // h).transpose(1, 0, 2)[...]

def merge_heads(x):
    h, n, dh = x.shape
    return x[..., :].transpose(1, 0, 2).reshape(n, h * dh)""",
}


@pytest.mark.parametrize("start", sorted(ELLIPSIS_ANSWERS))
def test_ellipsis_indexing_in_a_right_answer_is_not_a_blank(start):
    sol = code_cells("attention_solutions.ipynb")
    i = next(i for i, s in enumerate(sol) if s.lstrip().startswith(start))
    assert run(sol[:i] + [ELLIPSIS_ANSWERS[start]] + sol[i + 1:]) == (None, None)


def test_a_blank_next_to_ellipsis_indexing_still_stops():
    sol = code_cells("attention_solutions.ipynb")
    i = next(i for i, s in enumerate(sol) if s.lstrip().startswith("def softmax"))
    half = """def softmax(x, axis=-1):
    e = np.exp(x[..., :])
    return ...
"""
    at, err = run(sol[:i] + [half] + sol[i + 1:])
    assert at == i + 1 and isinstance(err, NotImplementedError), (at, repr(err))
