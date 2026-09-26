"""Tests for tools/site/build_site_content.py (the guide site's page and nav generator).

Run: python3 -m pytest tools/site/tests   (standard library + pytest; the hook tests also need mkdocs)
"""
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_site_content as b  # noqa: E402


# ---------------------------------------------------------------- solution notebooks

@pytest.mark.parametrize("path", [
    "07-x/lab/solutions/01_tools.ipynb",                          # solutions/
    "06-x/lab/notebooks/solutions/01_scaling_math_solutions.ipynb",  # notebooks/solutions/
    "07-x/lra-gcp/notebooks/worked/00_core_idea.ipynb",           # worked/
    "00-x/gpu-capacity-planning/notebooks/01_capacity_practice_solved.ipynb",  # _solved
    "06-x/agentic-identity-core/core_solution.ipynb",             # *_solution*
    "00-x/transformers/practice/attention_solutions.ipynb",
    "07-x/embeddings-lab/solutions/ex01_solutions.ipynb",
])
def test_is_solution_recognises_every_convention(path):
    assert b.is_solution(path)


@pytest.mark.parametrize("path", [
    "07-x/lab/notebooks/01_tools.ipynb",
    "04-x/kv-cache/02_kv_cache_practice.ipynb",
    "07-x/lra-gcp/notebooks/practice/00_core_idea.ipynb",
    "06-x/agentic-identity-core/core_walkthrough.ipynb",
    "07-x/embeddings-lab/exercises/ex01.ipynb",
    "05-x/lab/notebooks/01_resolution_and_workers.ipynb",         # "solution" inside a word is not a solution
])
def test_is_solution_leaves_exercises_alone(path):
    assert not b.is_solution(path)


# A *_worked notebook is an answer key only beside its exercise twin, and only when the folder keeps no solutions/
# of its own; otherwise it is a worked lesson.
GCP = "07-x/long-running-agents-gcp/notebooks/"
KV = "04-x/kv-cache/"
CORE = "07-x/long-running-agents-core/notebooks/"
FOLDERS = {
    GCP: ["01_durable_loop_practice", "01_durable_loop_worked", "03_hitl_saga_practice", "03_hitl_saga_scheduled_worked"],
    KV: ["01_kv_cache_worked", "02_kv_cache_practice"],
    CORE: ["01_worked", "02_practice"],
    "a/": ["01_x", "01_x_worked"],
    "c/": ["02_z_practice", "02_z_worked", "03_w_exercise", "03_w_scheduled_worked"],
}
SIBLINGS = [f"{d}{n}.ipynb" for d, names in FOLDERS.items() for n in names]
# long-running-agents-gcp keeps the practice answers as Python files in notebooks/solutions/ (README step 3).
SIBLINGS_WITH_FILES = SIBLINGS + [GCP + "solutions/ex1_durable_loop.py"]


@pytest.mark.parametrize("path, answers", [
    ("c/02_z_worked.ipynb", True),                         # same stem as 02_z_practice
    ("c/03_w_scheduled_worked.ipynb", True),               # same number as 03_w_exercise
    ("a/01_x_worked.ipynb", True),                         # the exercise is plain 01_x
    (GCP + "01_durable_loop_worked.ipynb", False),        # twin, but the answers are in notebooks/solutions/
    (GCP + "03_hitl_saga_scheduled_worked.ipynb", False),
    (KV + "01_kv_cache_worked.ipynb", False),             # lesson 01; the exercise is a different notebook, 02
    (CORE + "01_worked.ipynb", False),                    # lesson 01 before 02_practice
    ("b/01_y_worked.ipynb", False),                        # alone in its folder
])
def test_worked_is_an_answer_key_only_beside_its_exercise(path, answers):
    assert b.is_solution(path, SIBLINGS_WITH_FILES) is answers


def test_a_solutions_folder_makes_worked_notebooks_lessons():
    """The same twin is an answer key until its folder gains a solutions/ (or worked/) folder."""
    path = GCP + "01_durable_loop_worked.ipynb"
    assert b.is_solution(path, SIBLINGS) is True
    assert b.is_solution(path, SIBLINGS_WITH_FILES) is False
    assert b.is_solution(path, SIBLINGS + [GCP + "worked/01_durable_loop.ipynb"]) is False


def test_worked_rule_reads_the_folder_on_disk_by_default():
    """With no sibling list the folder is listed; the repo's own notebooks are the fixture. Both are lessons:
    kv-cache's has no twin, and long-running-agents-gcp's README reads the worked notebooks first and keeps the
    practice answers in notebooks/solutions/."""
    root = Path(b.ROOT)
    kv = "04-inference-engine/kv-cache/01_kv_cache_worked.ipynb"
    gcp_dir = ("07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/"
               "notebooks")
    gcp = f"{gcp_dir}/01_durable_loop_worked.ipynb"
    if not ((root / kv).exists() and (root / gcp).exists() and (root / gcp_dir / "solutions").is_dir()):
        pytest.skip("repo notebooks moved")
    assert not b.is_solution(kv)
    assert not b.is_solution(gcp)
    assert b.is_solution("07-application-agent-framework/long-running-durable/lra/lra-gcp/notebooks/worked/"
                         "00_core_idea.ipynb")


def test_worked_lessons_come_before_the_exercises_in_the_nav(monkeypatch):
    """long-running-agents-gcp's reading order: 01..04 worked, then 01..04 practice (README "Read in this order")."""
    folder = "07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/notebooks"
    names = ["01_a_practice", "01_a_worked", "02_b_practice", "02_b_worked"]
    monkeypatch.setattr(b, "pages", {})
    monkeypatch.setattr(b, "notebooks", {f"{folder}/{n}.ipynb": f"layers/{folder}/{n}.ipynb" for n in names})
    monkeypatch.setattr(b, "nb_title", lambda rp: posixpath.basename(rp))
    if not (Path(b.ROOT) / folder / "solutions").is_dir():
        pytest.skip("repo notebooks moved")
    order = [next(iter(e.values())).rsplit("/", 1)[-1] for e in b.nav_for_dir(folder)]
    assert order == ["01_a_worked.ipynb", "02_b_worked.ipynb", "01_a_practice.ipynb", "02_b_practice.ipynb"]


def test_colab_index_uses_the_same_rule():
    """tools/gen_colab_index.py labels the same notebooks as worked answers as the site does."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    for path in ("a/solutions/x.ipynb", "a/worked/x.ipynb", "a/01_x_worked.ipynb", "a/01_x_solved.ipynb",
                 "a/core_solution.ipynb", "a/notebooks/01_x.ipynb", "a/practice/01_x.ipynb",
                 "a/01_resolution.ipynb"):
        assert g.is_solution(path) == b.is_solution(path), path
    for sibs in (SIBLINGS, SIBLINGS_WITH_FILES):
        for path in SIBLINGS + ["b/01_y_worked.ipynb"]:
            assert g.is_solution(path, sibs) == b.is_solution(path, sibs), path
    for path in ("04-inference-engine/kv-cache/01_kv_cache_worked.ipynb",
                 "07-application-agent-framework/long-running-durable/long-running-agentic/long-running-agents-gcp/"
                 "notebooks/01_durable_loop_worked.ipynb"):
        assert g.is_solution(path) == b.is_solution(path), path
    assert (g.SOLUTION_DIRS, g.SOLUTION_STEM.pattern, g.WORKED_STEM.pattern, g.EXERCISE_STEM.pattern) == \
        (b.SOLUTION_DIRS, b.SOLUTION_STEM.pattern, b.WORKED_STEM.pattern, b.EXERCISE_STEM.pattern)


# ---------------------------------------------------------------- math

@pytest.mark.parametrize("text", [
    "tiers unlock at $20 / $100 / $500 / $2,000 of cumulative billing",
    "$5-$10 a month",
    "about $0.50 per million tokens and $2 per hour",
    "US$5 or $ 5 $",
    "$$\\text{softmax}(x)$$",
])
def test_dollar_amounts_stay_text(text):
    assert b.inline_tex_to_parens(text) == text


def test_inline_tex_becomes_parens_markdown_cannot_mangle():
    out = b.inline_tex_to_parens(r"The $\sqrt{d_k}$ keeps it; set to $-\infty$; costs $5 and $x$.")
    assert out == ("The &#92;(&#92;sqrt{d&#95;k}&#92;) keeps it; set to &#92;(-&#92;infty&#92;); "
                   "costs $5 and &#92;(x&#92;).")


def test_math_conversion_skips_code(tmp_path, monkeypatch):
    md = "Shell `echo $x_1$` and\n```\nprint('$a_b$')\n```\nbut $a_b$ is math."
    out = b.rewrite_markdown(md, "00-x/n.ipynb", "layers/00-x/n.ipynb", html=True, math=True)
    assert "`echo $x_1$`" in out and "print('$a_b$')" in out
    assert "but &#92;(a&#95;b&#92;) is math." in out


# ---------------------------------------------------------------- notebook anchors

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny repo with one notebook and one Markdown doc, wired into the generator's globals."""
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 06 · MCP servers\n", "\n",
            "- [Discovery](#1.-Discovery-and-the-401-challenge)\n",      # Jupyter's spelling
            "- [Annotations](#3-toolslist-annotations)\n",               # GitHub's spelling
            "- [Gone](#7-a-section-that-was-removed)\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": "## 1. Discovery and the 401 challenge"},
        {"cell_type": "markdown", "metadata": {}, "source": "## 3. `tools/list` annotations\n```\n# not a heading\n```"},
    ]}
    (tmp_path / "06-x" / "lab").mkdir(parents=True)
    (tmp_path / "06-x" / "lab" / "06_mcp.ipynb").write_text(json.dumps(nb))
    (tmp_path / "06-x" / "lab" / "README.md").write_text(
        "# lab\n\nSee [the challenge](06_mcp.ipynb#1.-Discovery-and-the-401-challenge) and "
        "[nothing](06_mcp.ipynb#no-such-heading).\n")
    monkeypatch.setattr(b, "REPO", tmp_path)
    monkeypatch.setattr(b, "pages", {"06-x/lab/README.md": "layers/06-x/lab/index.md"})
    monkeypatch.setattr(b, "notebooks", {"06-x/lab/06_mcp.ipynb": "layers/06-x/lab/06_mcp.ipynb"})
    monkeypatch.setattr(b, "_anchor_cache", {})
    monkeypatch.setattr(b, "_nb_cache", {})
    monkeypatch.setattr(b, "dropped_anchors", [])
    monkeypatch.setattr(b, "stats", {k: 0 for k in b.stats})
    return tmp_path


def test_notebook_anchor_ids_match_mkdocs_jupyter(repo):
    amap = b.nb_anchors_of("06-x/lab/06_mcp.ipynb")
    assert amap["1.-Discovery-and-the-401-challenge"] == "1-discovery-and-the-401-challenge"
    assert amap["3-toolslist-annotations"] == "3-toolslist-annotations"
    assert "not-a-heading" not in amap


def test_same_page_notebook_anchors_resolve_or_unlink(repo):
    src = "06-x/lab/06_mcp.ipynb"
    cell = "".join(json.loads((repo / src).read_text())["cells"][0]["source"])
    out = b.rewrite_markdown(cell, src, "layers/" + src, html=True, math=True)
    assert "[Discovery](#1-discovery-and-the-401-challenge)" in out
    assert "[Annotations](#3-toolslist-annotations)" in out
    assert "- Gone\n" in out + "\n" and "#7-a-section" not in out     # dead: plain text, not a dead link
    assert (b.stats["nb_anchors_kept"], b.stats["nb_anchors_dropped"]) == (2, 1)


def test_links_into_a_notebook_keep_live_anchors_and_drop_dead_ones(repo):
    src = "06-x/lab/README.md"
    out = b.rewrite_markdown((repo / src).read_text(), src, b.pages[src])
    assert "(06_mcp.ipynb#1-discovery-and-the-401-challenge)" in out
    assert "[nothing](06_mcp.ipynb)" in out
    assert (b.stats["nb_anchors_kept"], b.stats["nb_anchors_dropped"]) == (1, 1)
    assert b.dropped_anchors == ["06-x/lab/README.md -> 06_mcp.ipynb#no-such-heading"]


# ---------------------------------------------------------------- titles and nav

def test_humanize_keeps_spelled_acronyms():
    assert b.humanize("vllm-internals") == "vLLM internals"
    assert b.humanize("k8s-gpu-lab") == "K8s GPU lab"
    assert b.humanize("04_busbw_and_the_alpha_beta_fit") == "04 · Busbw and the alpha beta fit"
    assert b.humanize("long-running-durable") == "Long-running durable"


def test_short_title_cuts_at_the_last_separator_that_fits():
    assert b.short_title("Short enough") == "Short enough"
    long = "01 · The arithmetic of agent scale — Mistral's API or your own GPUs, and what they cost"
    assert b.short_title(long) == "01 · The arithmetic of agent scale"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    files = {
        "07-x/topic/README.md": "# topic-name — the promise of the topic\n",
        "07-x/topic/PRIMER.md": "# A primer\n",
        "07-x/topic/lab-mistral/README.md": "# Agent lab — the core idea\n",
        "07-x/topic/lab-mistral/deploy/README.md": "# deploy\n",
        "07-x/topic/lab-mistral/deploy/gke/README.md": "# gke\n",
        "07-x/topic/wrapper/inner-core/README.md": "# inner-core — only child\n",
    }
    nbs = {
        "07-x/topic/lab-mistral/notebooks/01_loop.ipynb": "# 01 · The loop",
        "07-x/topic/lab-mistral/notebooks/02_tools.ipynb": "# 02 · Tools",
        "07-x/topic/lab-mistral/solutions/01_loop.ipynb": "# 01 · The loop",
        "07-x/topic/lab-mistral/solutions/02_tools.ipynb": "# 02 · Tools",
        "07-x/topic/lab-mistral/more/notebooks/practice/02_x_practice.ipynb": "# 02 · X",
        "07-x/topic/lab-mistral/more/notebooks/practice/03_y_practice.ipynb": "# 03 · Y",
        "07-x/topic/lab-mistral/more/notebooks/worked/02_x.ipynb": "# 02 · X",
        "07-x/topic/lab-mistral/more/notebooks/worked/03_y.ipynb": "# 03 · Y",
        "07-x/topic/vllm-x/notebooks/01_only.ipynb": "# 01 · Only one",
    }
    for rp, text in files.items():
        (tmp_path / rp).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rp).write_text(text)
    for rp, h1 in nbs.items():
        (tmp_path / rp).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rp).write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": h1}]}))
    monkeypatch.setattr(b, "REPO", tmp_path)
    monkeypatch.setattr(b, "pages", {rp: "layers/" + rp.replace("README.md", "index.md") for rp in files})
    monkeypatch.setattr(b, "notebooks", {rp: "layers/" + rp for rp in nbs})
    monkeypatch.setattr(b, "_nb_cache", {})
    monkeypatch.setattr(b, "_dir_titles", {})
    return tmp_path


def test_nav_titles_structure_and_plumbing(tree):
    nav = b.nav_for_dir("07-x/topic")
    flat = json.dumps(nav, ensure_ascii=False)
    assert "deploy" not in flat                                   # plumbing stays out of the nav
    assert nav[0] == "layers/07-x/topic/index.md"
    assert nav[1] == {"Topic name primer": "layers/07-x/topic/PRIMER.md"}
    lab = next(v for e in nav if isinstance(e, dict) for k, v in e.items() if k.startswith("Agent lab"))
    assert "Agent lab (Mistral)" in flat                         # the provider variant says so
    assert {"01 · The loop (solution)": "layers/07-x/topic/lab-mistral/solutions/01_loop.ipynb"} in \
        next(v for e in lab if isinstance(e, dict) for k, v in e.items() if k == "Solutions")
    assert [next(iter(e)) for e in lab if isinstance(e, dict)] == ["More", "Notebooks", "Solutions"]
    # notebooks/ holding only practice/ and worked/ is lifted into its parent
    more = next(v for e in lab if isinstance(e, dict) for k, v in e.items() if k == "More")
    assert [next(iter(e)) for e in more] == ["Practice", "Worked"]
    assert {"02 · X (worked)": "layers/07-x/topic/lab-mistral/more/notebooks/worked/02_x.ipynb"} in more[1]["Worked"]
    # folders around a single entry (vllm-x/ > notebooks/ > one notebook) collapse into it
    assert {"01 · Only one": "layers/07-x/topic/vllm-x/notebooks/01_only.ipynb"} in nav
    # a folder that only wraps one README is a page, titled from the H1's name part, humanised
    assert {"Inner core": "layers/07-x/topic/wrapper/inner-core/index.md"} in nav


def test_mkdocs_yml_blocks_are_rewritten_idempotently(tree, tmp_path, monkeypatch):
    yml = tmp_path / "mkdocs.yml"
    yml.write_text("site_name: x\n# not-in-nav:start\n# not-in-nav:end\nnav:\n  - Home: index.md\n"
                   "  # nav-layers:start\n  # nav-layers:end\n")
    monkeypatch.setattr(b, "MKDOCS_YML", yml)
    assert b.update_mkdocs_yml(["07-x"]) == "updated"
    text = yml.read_text()
    assert "not_in_nav: |\n  /layers/**/client/\n  /layers/**/deploy/" in text
    assert '  - Layers:\n      - "layers/index.md"\n      - "07 · Agents & applications":' in text
    assert b.update_mkdocs_yml(["07-x"]) == "unchanged"


def test_colab_index_marks_worked_answers_and_describes_every_layout():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    out = g.layer_section("07-x", ["07-x/a/notebooks/01_x.ipynb", "07-x/a/notebooks/01_x_worked.ipynb",
                                   "07-x/a/solutions/01_x.ipynb", "07-x/kv/01_kv_worked.ipynb",
                                   "07-x/kv/02_kv_practice.ipynb"])
    lines = out.splitlines()
    assert any(ln.endswith("`01_kv_worked.ipynb`") for ln in lines)   # a worked lesson: a plain Colab link
    assert any(ln.endswith("`01_x.ipynb`") and "/notebooks/" in ln for ln in lines)
    assert any(ln.endswith("`01_x_worked.ipynb` — *worked answers*") for ln in lines)
    assert any(ln.endswith("`01_x.ipynb` — *worked answers*") and "/solutions/" in ln for ln in lines)
    assert "Exercises are under `notebooks/` / `exercises/`" not in out


def test_duplicate_titles_in_provider_variants_name_the_provider():
    nav = [{"Agent core": [{"01 · The loop": "layers/07-x/agent-core/notebooks/01_loop.ipynb"},
                           {"01 · The loop (solution)": "layers/07-x/agent-core/solutions/01_loop.ipynb"}]},
           {"Mistral agent core": [{"01 · The loop": "layers/07-x/mistral-agent-core/notebooks/01_loop.ipynb"},
                                   {"01 · The loop (solution)": "layers/07-x/mistral-agent-core/solutions/01_loop.ipynb"},
                                   {"05 · Going live on Mistral": "layers/07-x/mistral-agent-core/notebooks/05.ipynb"}]},
           {"Long-running": [{"A primer": "layers/07-x/a/primer.md"}, {"A primer": "layers/07-x/b/primer.md"}]},
           {"GCP": [{"On Google Cloud": "layers/07-x/c/primer.md"}, {"On Google Cloud": "layers/07-x/c-gcp/p.md"}]}]
    out = json.dumps(b.mark_variant_duplicates(nav), ensure_ascii=False)
    assert '"01 · The loop": "layers/07-x/agent-core/' in out
    assert '"01 · The loop (Mistral)": "layers/07-x/mistral-agent-core/notebooks' in out
    assert '"01 · The loop (solution, Mistral)"' in out
    assert '"05 · Going live on Mistral"' in out                  # unique titles are left alone
    assert out.count('"A primer"') == 2                           # not a provider variant: unchanged
    assert out.count('"On Google Cloud"') == 2                    # the title already names the provider
