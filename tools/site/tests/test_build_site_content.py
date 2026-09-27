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

# One layout (tools/ci/nb_layout.py): a notebook is an answer key exactly when its folder is solutions/. Names and
# folders that other conventions used for answers are ordinary notebooks now (built with posixpath.join so the
# old folder names do not read as paths in the tree).
J = posixpath.join


def suffixed(stem: str, suffix: str) -> str:
    return f"{stem}_{suffix}.ipynb"


@pytest.mark.parametrize("path", [
    "07-x/lab/solutions/01_tools.ipynb",
    "00-x/gpu-capacity-planning/solutions/01_capacity_practice.ipynb",
    "06-x/agentic-identity-core/solutions/core_practice.ipynb",
    "00-x/transformers/solutions/attention_practice.ipynb",
    "07-x/embeddings-lab/solutions/ex01.ipynb",
    "solutions/x.ipynb",
])
def test_a_solution_is_recognised_under_solutions(path):
    assert b.is_solution(path)


@pytest.mark.parametrize("path", [
    "07-x/lab/notebooks/01_tools.ipynb",
    "04-x/kv-cache/notebooks/02_kv_cache_practice.ipynb",
    "04-x/kv-cache/notebooks/01_kv_cache_worked.ipynb",            # a worked lesson, not an answer key
    J("00-x", "gpu-capacity-planning", "notebooks", suffixed("01_capacity_practice", "solved")),
    J("06-x", "agentic-identity-core", suffixed("core", "solution")),
    J("07-x", "embeddings-lab", "notebooks", suffixed("ex01", "solutions")),
    J("07-x", "lra-gcp", "worked", "00_core_idea.ipynb"),
    J("06-x", "lab", "notebooks", "solutions", "deeper", "01_x.ipynb"),
    "05-x/lab/notebooks/01_resolution_and_workers.ipynb",
    "solutions.ipynb",
])
def test_a_solution_is_recognised_only_under_solutions(path):
    assert not b.is_solution(path)


def test_a_folder_lists_its_notebooks_in_file_name_order(tmp_path, monkeypatch):
    """No reordering by name suffix: a lesson and its practice twin sit side by side, in file-name order."""
    folder = "06-x/lab/notebooks"
    names = ["01_a", "01_a_practice", "02_b", "02_b_practice", "03_c_worked"]
    monkeypatch.setattr(b, "pages", {})
    monkeypatch.setattr(b, "notebooks", {f"{folder}/{n}.ipynb": f"layers/{folder}/{n}.ipynb" for n in names})
    monkeypatch.setattr(b, "nb_title", lambda rp: posixpath.basename(rp))
    order = [next(iter(e.values())).rsplit("/", 1)[-1] for e in b.nav_for_dir(folder)]
    assert order == [f"{n}.ipynb" for n in names]


def test_colab_index_uses_the_same_rule():
    """tools/gen_colab_index.py labels the same notebooks as worked answers as the site does."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    for path in ("a/solutions/x.ipynb", J("a", "worked", "x.ipynb"), "a/notebooks/01_x_worked.ipynb",
                 J("a", "notebooks", suffixed("01_x", "solved")), J("a", suffixed("core", "solution")), "a/notebooks/01_x.ipynb",
                 J("a", "notebooks", "solutions", "x.ipynb"), "a/01_resolution.ipynb", "x.ipynb"):
        assert g.is_solution(path) == b.is_solution(path), path
    assert g.ANSWERS_DIR == b.ANSWERS_DIR == "solutions"
    assert g.NOTEBOOK_DIRS == {"notebooks"} | b.SOLUTION_DIRS and set(b.NOTEBOOK_DIRS) == {"notebooks"}


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


def test_operator_spans_are_tex_but_prices_are_not():
    out = b.inline_tex_to_parens("With $T > 1$, $E/p$, $n-1$ ranks and $k = 2$; a T4 is $0.35/h, $5/$10 a day, $100-$200 a month.")
    assert "&#92;(T &gt; 1&#92;)" in out and "&#92;(E/p&#92;)" in out and "&#92;(n-1&#92;)" in out and "&#92;(k = 2&#92;)" in out
    assert "$0.35/h, $5/$10 a day, $100-$200 a month." in out


def test_inline_tex_becomes_parens_markdown_cannot_mangle():
    out = b.inline_tex_to_parens(r"The $\sqrt{d_k}$ keeps it; set to $-\infty$; costs $5 and $x$.")
    assert out == ("The &#92;(&#92;sqrt{d&#95;k}&#92;) keeps it; set to &#92;(-&#92;infty&#92;); "
                   "costs $5 and &#92;(x&#92;).")


def test_math_conversion_skips_code(tmp_path, monkeypatch):
    md = "Shell `echo $x_1$` and\n```\nprint('$a_b$')\n```\nbut $a_b$ is math."
    out = b.rewrite_markdown(md, "00-x/n.ipynb", "layers/00-x/n.ipynb", html=True, math=True)
    assert "`echo $x_1$`" in out and "print('$a_b$')" in out
    assert "but &#92;(a&#95;b&#92;) is math." in out


def test_markdown_pages_get_plain_parens_for_arithmatex():
    """Pages go through pymdownx.arithmatex, which takes \\( \\) before any other inline rule: no entities needed,
    and $$ display blocks stay as written (arithmatex's dollar block syntax). Prices stay text."""
    md = ("The loss is $\\alpha T^2\\,\\mathrm{KL}(p_T \\| q_T)$ and $x_1$; a T4 is $0.35/h and an L4 $0.7/h, "
          "so $20 / $100 buys little.\n\n$$\nr_t = \\log \\pi_T - \\log \\pi_S\n$$\n\nSee `$HOME` and $5.")
    out = b.rewrite_markdown(md, "00-x/PRIMER.md", "layers/00-x/PRIMER.md", math="page")
    assert "\\(\\alpha T^2\\,\\mathrm{KL}(p_T \\| q_T)\\)" in out and "\\(x_1\\)" in out
    assert "&#92;" not in out
    assert "$0.35/h and an L4 $0.7/h" in out and "$20 / $100" in out and "`$HOME` and $5." in out
    assert "$$\nr_t = \\log \\pi_T - \\log \\pi_S\n$$" in out


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
    assert [next(iter(e)) for e in lab if isinstance(e, dict)] == ["Notebooks", "Solutions"]
    notebooks = next(v for e in lab if isinstance(e, dict) for k, v in e.items() if k == "Notebooks")
    assert notebooks == [{"01 · The loop": "layers/07-x/topic/lab-mistral/notebooks/01_loop.ipynb"},
                         {"02 · Tools": "layers/07-x/topic/lab-mistral/notebooks/02_tools.ipynb"}]
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
    assert '  - Layers:\n      - "layers/index.md"\n      - "07 · Agents and applications":' in text
    assert b.update_mkdocs_yml(["07-x"]) == "unchanged"


def test_colab_index_lists_answers_from_solutions_only_and_says_the_one_layout():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    out = g.layer_section("07-x", ["07-x/a/notebooks/01_x.ipynb", "07-x/a/notebooks/01_x_worked.ipynb",
                                   "07-x/a/solutions/01_x.ipynb", "07-x/kv/notebooks/01_kv_worked.ipynb",
                                   "07-x/kv/notebooks/02_kv_practice.ipynb"])
    lines = out.splitlines()
    lab_a = next(ln for ln in lines if ln.startswith("- **`a/`**"))
    lab_kv = next(ln for ln in lines if ln.startswith("- **`kv/`**"))
    lessons_a, answers_a = lab_a.split(" — *answers:* ")
    assert "[01_x](" in lessons_a and "/a/notebooks/01_x.ipynb" in lessons_a     # the exercise
    assert "[01_x_worked](" in lessons_a                                          # a name is not a folder: a lesson
    assert "/a/solutions/01_x.ipynb" in answers_a and "01_x_worked" not in answers_a
    assert "*answers:*" not in lab_kv                                             # no solutions/: plain links
    assert "[01_kv_worked](" in lab_kv and "[02_kv_practice](" in lab_kv
    assert len([ln for ln in lines if ln.startswith("- **")]) == 2                  # one line per lab
    assert "`notebooks/`" in out and "`solutions/`" in out and "same file name" in out
    for old in ("worked/", "_solved", "_solution`", "exercises/"):
        assert old not in out, old


def test_colab_index_groups_notebook_folders_under_their_lab():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    assert g.lab_of("a/lab/notebooks") == "a/lab"
    assert g.lab_of("a/lab/solutions") == "a/lab"
    assert g.lab_of("kv-cache") == "kv-cache"
    assert g.lab_of(".") == ""
    assert g.lab_of(J("a", "lab", "practice")) == J("a", "lab", "practice")      # not a layout folder: its own line


def test_colab_index_skips_run_outputs_and_checkpoints(tmp_path, monkeypatch):
    """Executed copies a lab's tests write (_run_outputs/) and checkpoints never get a Colab link."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    for rel in ("07-x/lab/notebooks/01_a.ipynb", "07-x/lab/_run_outputs/01_a.ipynb",
                "07-x/lab/notebooks/.ipynb_checkpoints/01_a-checkpoint.ipynb"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert g.layer_notebooks("07-x") == ["07-x/lab/notebooks/01_a.ipynb"]
    assert g.SKIP_DIRS <= b.SKIP_DIRS | {".git"}


def test_layer_names_match_the_colab_index():
    """One name per layer: the site's layer titles and the Colab index's layer names are the same strings."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import gen_colab_index as g
    assert g.LAYER_NAMES == b.LAYER_TITLES


def test_duplicate_titles_in_provider_variants_name_the_provider():
    nav = [{"Agent lab": [{"01 · The loop": "layers/07-x/lab/notebooks/01_loop.ipynb"},
                          {"01 · The loop (solution)": "layers/07-x/lab/solutions/01_loop.ipynb"}]},
           {"Mistral agent lab": [{"01 · The loop": "layers/07-x/mistral-lab/notebooks/01_loop.ipynb"},
                                  {"01 · The loop (solution)": "layers/07-x/mistral-lab/solutions/01_loop.ipynb"},
                                  {"05 · Going live on Mistral": "layers/07-x/mistral-lab/notebooks/05.ipynb"}]},
           {"Long-running": [{"A primer": "layers/07-x/a/primer.md"}, {"A primer": "layers/07-x/b/primer.md"}]},
           {"GCP": [{"On Google Cloud": "layers/07-x/c/primer.md"}, {"On Google Cloud": "layers/07-x/c-gcp/p.md"}]}]
    out = json.dumps(b.mark_variant_duplicates(nav), ensure_ascii=False)
    assert '"01 · The loop": "layers/07-x/lab/' in out
    assert '"01 · The loop (Mistral)": "layers/07-x/mistral-lab/notebooks' in out
    assert '"01 · The loop (solution, Mistral)"' in out
    assert '"05 · Going live on Mistral"' in out                  # unique titles are left alone
    assert out.count('"A primer"') == 2                           # not a provider variant: unchanged
    assert out.count('"On Google Cloud"') == 2                    # the title already names the provider
