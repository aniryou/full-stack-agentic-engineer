"""tools/ci/nb_sources.py: a heading in a percent-format notebook source belongs to a markdown cell, and
`ci.py check` fails on one that is not (the builders would write it, and the prose under it, as comments at the end
of the preceding code cell)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("nb_sources", REPO / "tools" / "ci" / "nb_sources.py")
nbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nbs)
_spec = importlib.util.spec_from_file_location("ci", REPO / "tools" / "ci" / "ci.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

GOOD = """\
# %% [markdown]
# # A notebook
# **Tier:** T0
#
# ## The one-minute version
# Prose under a heading.

# %%
x = 1   # a code cell

# %% [markdown]
# ## Exercise 1.1 — the first blank
# What to write.

# %% exercise
def f():
    ### BEGIN SOLUTION
    return 2
    ### END SOLUTION

# %% check
assert f() == 2
print("✅")

# %%[markdown]
# ### A heading after a marker without the space
# #### Four
# ##### Five
# ###### Six

# %% [markdown]
# ## In a design review
# The two-minute version.
"""

# The defect this guards against, as it was found: an exercise statement written straight after a check cell.
MISPLACED_AFTER_A_CHECK = GOOD.replace(
    'print("✅")\n\n# %%[markdown]\n# ### A heading after a marker without the space\n',
    'print("✅")\n\n# ## Exercise 1.2 — the second blank\n'
    '# Its statement, as a comment at the end of the check cell.\n\n'
    '# %% exercise\ny = 3\n\n# %% check\nassert y == 3\n\n'
    '# %%[markdown]\n# ### A heading after a marker without the space\n')


def test_cells_split_as_the_builders_do():
    kinds = [(kind, start) for kind, start, _ in nbs.cells(GOOD)]
    assert kinds == [("markdown", 1), ("code", 8), ("markdown", 11), ("exercise", 15), ("check", 21),
                     ("markdown", 25), ("markdown", 31)]
    assert nbs.cell_kind(" [markdown]") == nbs.cell_kind("[markdown]") == "markdown"
    assert nbs.cell_kind(" [markdown] tags=[]") == "markdown"
    assert nbs.cell_kind("") == nbs.cell_kind("   ") == "code"
    assert nbs.cell_kind(" exercise") == "exercise" and nbs.cell_kind(" check ") == "check"
    before, *rest = nbs.cells("# a comment\n\n# %%\nx = 1\n")
    assert before == (nbs.BEFORE_FIRST, 0, ["# a comment", ""]) and [k for k, _, _ in rest] == ["code"]
    assert nbs.cells("# %%\nx = 1\n")[0][0] == "code"     # nothing before the first marker: no pseudo-cell


def test_headings_in_markdown_cells_are_fine():
    assert nbs.misplaced_headings(GOOD) == []


@pytest.mark.parametrize("marker, kind", [("# %%", "code"), ("# %% exercise", "exercise"), ("# %% check", "check")])
def test_a_heading_in_a_code_cell_is_found_with_its_line_and_kind(marker, kind):
    text = f"# %% [markdown]\n# # Title\n\n{marker}\nx = 1\n\n# ## Exercise 1.1 — misplaced\n# Its statement.\n"
    assert nbs.misplaced_headings(text) == [(7, kind, "# ## Exercise 1.1 — misplaced")]


def test_the_defect_as_found_is_reported_once_at_the_heading():
    assert nbs.misplaced_headings(MISPLACED_AFTER_A_CHECK) == [(25, "check", "# ## Exercise 1.2 — the second blank")]
    fixed = MISPLACED_AFTER_A_CHECK.replace("\n# ## Exercise 1.2", "\n# %% [markdown]\n# ## Exercise 1.2")
    assert nbs.misplaced_headings(fixed) == []


def test_a_heading_before_the_first_marker_is_found():
    text = "# # A title the builders would drop\n\n# %% [markdown]\n# ## Fine\n"
    assert nbs.misplaced_headings(text) == [(1, nbs.BEFORE_FIRST, "# # A title the builders would drop")]


def test_every_heading_level_counts_but_other_comments_do_not():
    for level in range(1, 7):
        assert nbs.misplaced_headings(f"# %%\n# {'#' * level} Deep\n") == [(2, "code", f"# {'#' * level} Deep")]
    harmless = "# %%\n### BEGIN SOLUTION\n#### a banner ####\n# noqa: E501\n#comment\n# TODO: later\n#\n# ##no space\n"
    assert nbs.misplaced_headings(harmless) == []


def test_the_committed_sources_are_clean():
    paths = nbs.tracked_sources()
    assert len(paths) >= 150
    assert all(p.endswith(".py") and ("/notebooks_src/" in p or "/embeddings-lab/src/" in p) for p in paths), paths
    assert "06-gateway/llm-gateway/gateway-core/notebooks_src/02_routing_and_fallback_chains.py" in paths
    assert "07-application-agent-framework/retrieval-rag/embeddings-lab/src/01_counts_to_vectors.py" in paths
    assert not any(p.startswith("tools/") for p in paths)   # this file's fixtures carry markers; it is not a source
    assert nbs.findings(paths) == []


def test_findings_name_the_file_the_line_and_the_cell(tmp_path):
    bad = tmp_path / "01_bad.py"
    bad.write_text(MISPLACED_AFTER_A_CHECK, encoding="utf-8")
    (tmp_path / "02_good.py").write_text(GOOD, encoding="utf-8")
    found = nbs.findings([str(bad), str(tmp_path / "02_good.py")], root=tmp_path)
    assert [(p, n) for p, n, _ in found] == [(str(bad), 25)]
    assert "in a check cell" in found[0][2] and "# %% [markdown]" in found[0][2] and "Exercise 1.2" in found[0][2]
    assert nbs.findings(["01_bad.py"], root=tmp_path)[0][0] == "01_bad.py"     # relative paths resolve against root


def test_check_fails_on_a_misplaced_heading(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "01_bad.py"
    bad.write_text(MISPLACED_AFTER_A_CHECK, encoding="utf-8")
    monkeypatch.setattr(ci, "nb_sources", lambda: nbs)
    monkeypatch.setattr(nbs, "tracked_sources", lambda: [str(bad)])
    assert ci.cmd_check([]) == 1
    out = capsys.readouterr().out
    assert f"notebook source: {bad}:25: a heading in a check cell" in out
    bad.write_text(GOOD, encoding="utf-8")
    assert ci.cmd_check([]) == 0


def test_cli_check_passes_on_the_committed_tree():
    r = subprocess.run([sys.executable, str(REPO / "tools/ci/nb_sources.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("0 headings outside a markdown cell")
