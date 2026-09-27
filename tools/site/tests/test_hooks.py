"""Tests for tools/site/hooks.py (needs mkdocs: pip install -r requirements-site.txt)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mkdocs")
from mkdocs.structure.toc import AnchorLink  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hooks  # noqa: E402

NB_HTML = """<style type="text/css">.highlight-ipynb .k { color: red }</style>
<style type="text/css">@charset "UTF-8";.jupyter-wrapper{--x: 1}</style>
<!-- Load mathjax -->
<script src=""> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath: [ ['$','$'] ]}});</script>
<!-- End of mathjax configuration -->
<div class="jupyter-wrapper"><div class="jp-Notebook">
<h1 id="06-mcp-servers">06 · MCP servers</h1>
<p>Tiers unlock at $20 / $100.</p>
<style scoped>.dataframe td { color: blue }</style>
<h2 id="3-toolslist-annotations-and-scope">3. tools/list annotations and scope</h2>
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-OutputArea jp-Cell-outputArea"><pre>42</pre></div>
<div class="clipboard-copy-txt" id="cell-1">echo $$</div>
<pre>echo $$ \\(x\\)</pre>
<p><a href="#3-toolslist-annotations-and-scope">ok</a> <a href="#nowhere">dead</a></p>
</div> <!-- jp-Notebook -->
</div> <!-- jupyter-wrapper -->
<style>
['pre { line-height: 125%; }']
</style>
"""


@pytest.fixture
def fresh(tmp_path):
    cfg = {"docs_dir": str(tmp_path), "extra_css": [], "extra_javascript": [], "site_dir": str(tmp_path / "out")}
    hooks.on_config(cfg)
    return cfg


def nb_page(src="layers/06-x/lab/notebooks/06_mcp.ipynb"):
    toc = [AnchorLink("06 · MCP servers", "06-mcp-servers", 1)]
    toc[0].children = [AnchorLink("3. annotations and scope", "3-annotations-and-scope", 2)]
    return SimpleNamespace(file=SimpleNamespace(src_uri=src), toc=toc, meta={},
                           url=src.replace(".ipynb", "/"))


def test_notebook_css_moves_to_one_shared_file(fresh):
    page = nb_page()
    html = hooks.on_page_content(NB_HTML, page, fresh, None)
    assert ".jupyter-wrapper{--x: 1}" not in html and ".highlight-ipynb" not in html.split("jupyter-wrapper", 1)[0]
    assert "<style scoped>.dataframe td" in html                 # a cell output's own style stays
    assert "['pre" not in html
    out = hooks.on_post_page(f"<html><head></head><body>{html}</body></html>", page, fresh)
    h = hooks._page_css[page.file.src_uri]
    assert f'<link rel="stylesheet" href="../../../../../assets/stylesheets/notebook.{h}.css">' in out
    hooks.on_post_build(fresh)
    css = (Path(fresh["site_dir"]) / f"assets/stylesheets/notebook.{h}.css").read_text()
    assert ".highlight-ipynb .k" in css and ".jupyter-wrapper{--x: 1}" in css and "['pre" not in css


def test_mkdocs_jupyter_mathjax2_config_is_removed_and_no_math_means_no_mathjax(fresh):
    page = nb_page()
    html = hooks.on_page_content(NB_HTML, page, fresh, None)
    assert "inlineMath" not in html and '<script src="">' not in html
    out = hooks.on_post_page(f"<html><head></head><body>{html}</body></html>", page, fresh)
    assert "mathjax" not in out.lower()                           # $$ and \( only inside code: no MathJax


def test_pages_with_math_load_the_config_then_mathjax(fresh):
    page = SimpleNamespace(file=SimpleNamespace(src_uri="layers/00-x/t.ipynb"), toc=[], meta={}, url="layers/00-x/t/")
    html = hooks.on_page_content("<p>The \\(\\sqrt{d_k}\\) keeps</p>", page, fresh, None)
    out = hooks.on_post_page(f"<html><body>{html}</body></html>", page, fresh)
    cfg, cdn = out.index("javascripts/mathjax.js"), out.index("mathjax@3.2.2/es5/tex-mml-chtml.js")
    assert cfg < cdn


def test_notebook_toc_points_at_the_rendered_headings_and_dead_anchors_are_counted(fresh):
    page = nb_page()
    hooks.on_page_content(NB_HTML, page, fresh, None)
    assert page.toc[0].children[0].id == "3-toolslist-annotations-and-scope"
    assert hooks._dead == ["layers/06-x/lab/notebooks/06_mcp.ipynb#nowhere"]


def test_search_skips_solutions_and_code_duplicates(fresh):
    sol = nb_page("layers/06-x/lab/solutions/06_mcp.ipynb")
    html = hooks.on_page_content(NB_HTML, sol, fresh, None)
    assert sol.meta["search"] == {"exclude": True}
    assert '<div class="clipboard-copy-txt" data-search-exclude' in html
    assert '<div class="jp-InputPrompt jp-InputArea-prompt" data-search-exclude' in html
    assert '<div class="jp-OutputArea jp-Cell-outputArea" data-search-exclude' in html
    ex = nb_page()
    hooks.on_page_content(NB_HTML, ex, fresh, None)
    assert "search" not in ex.meta


def test_cache_busting_covers_css_and_js(tmp_path):
    (tmp_path / "stylesheets").mkdir()
    (tmp_path / "stylesheets/extra.css").write_text("a{}")
    (tmp_path / "javascripts").mkdir()
    (tmp_path / "javascripts/tables.js").write_text("1")
    cfg = {"docs_dir": str(tmp_path), "extra_css": ["stylesheets/extra.css"],
           "extra_javascript": ["javascripts/tables.js", "https://cdn.example/x.js"]}
    hooks.on_config(cfg)
    assert cfg["extra_css"][0].startswith("stylesheets/extra.css?v=")
    assert cfg["extra_javascript"][0].startswith("javascripts/tables.js?v=")
    assert cfg["extra_javascript"][1] == "https://cdn.example/x.js"


# ---------------------------------------------------------------- GitHub alerts -> admonitions

def test_github_alerts_become_admonitions():
    md = """Intro.

> [!NOTE]
> The first line, with `code` and \\(x_1\\).
>
> A second paragraph.
Next paragraph, no blank line.

> **Pitfall.** An ordinary blockquote stays a blockquote.

```md
> [!TIP]
> inside a fence: untouched
```

> [!WARNING]
> last block"""
    out = hooks.github_alerts_to_admonitions(md)
    assert ('!!! note "Note"\n    The first line, with `code` and \\(x_1\\).\n\n    A second paragraph.\n\n'
            "Next paragraph, no blank line.") in out
    assert "> **Pitfall.** An ordinary blockquote stays a blockquote." in out
    assert "```md\n> [!TIP]\n> inside a fence: untouched\n```" in out
    assert out.endswith('!!! warning "Warning"\n    last block')
    assert "[!NOTE]" not in out and "[!WARNING]" not in out


def test_on_page_markdown_only_touches_markdown_pages():
    md = "> [!TIP]\n> t"
    page_md = SimpleNamespace(file=SimpleNamespace(src_uri="layers/00-x/PRIMER.md"))
    page_nb = SimpleNamespace(file=SimpleNamespace(src_uri="layers/00-x/notebooks/01.ipynb"))
    assert hooks.on_page_markdown(md, page_md, {}, None).startswith('!!! tip "Tip"')
    assert hooks.on_page_markdown(md, page_nb, {}, None) == md


# ---------------------------------------------------------------- prose tables -> cards

MODULE_TABLE = ("<table>\n<thead>\n<tr>\n<th>Module</th>\n<th>You can …</th>\n<th>Primer</th>\n<th>Hours</th>\n</tr>\n"
                "</thead>\n<tbody>\n<tr>\n<td><strong>00.4.1</strong></td>\n<td>" + " ".join(["word"] * 60) +
                "</td>\n<td>§1</td>\n<td>3</td>\n</tr>\n</tbody>\n</table>")
SPEC_TABLE = ("<table>\n<thead>\n<tr>\n<th>GPU</th>\n<th>Memory</th>\n<th>Bandwidth</th>\n<th>CC</th>\n<th>BF16</th>\n</tr>\n"
              "</thead>\n<tbody>\n<tr>\n<td>T4</td>\n<td>16 GB</td>\n<td>~320 GB/s</td>\n<td>7.5</td>\n<td>no</td>\n</tr>\n"
              "</tbody>\n</table>")


def test_record_tables_become_cards_and_spec_tables_stay_tables():
    out, n = hooks.stack_prose_tables(MODULE_TABLE + "\n" + SPEC_TABLE)
    assert n == 1
    assert '<table class="fse-stacked">' in out and out.count("<table") == 2
    assert '<td data-label="Module" class="fse-cell fse-key"><strong>00.4.1</strong></td>' in out
    assert '<td data-label="You can …" class="fse-cell fse-prose">word' in out
    assert '<td data-label="Hours" class="fse-cell">3</td>' in out
    assert SPEC_TABLE in out                                  # untouched, byte for byte


def test_three_column_tables_stack_only_with_long_cells():
    short = "<table>\n<thead>\n<tr>\n<th>a</th>\n<th>b</th>\n<th>c</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>1</td>\n<td>" + \
        " ".join(["w"] * 50) + "</td>\n<td>3</td>\n</tr>\n</tbody>\n</table>"
    assert hooks.stack_prose_tables(short) == (short, 0)
    long = short.replace(" ".join(["w"] * 50), " ".join(["w"] * 130))
    out, n = hooks.stack_prose_tables(long)
    assert n == 1 and 'class="fse-cell fse-prose"' in out


def test_cards_only_on_markdown_pages(fresh):
    page = SimpleNamespace(file=SimpleNamespace(src_uri="layers/00-x/README.md"), toc=[], meta={}, url="layers/00-x/")
    out = hooks.on_page_content(MODULE_TABLE, page, fresh, None)
    assert "fse-stacked" in out
    nb = nb_page()
    assert "fse-stacked" not in hooks.on_page_content(NB_HTML + MODULE_TABLE, nb, fresh, None)


def test_hand_written_pages_get_inline_tex_converted_too():
    md = "Loss $\\alpha T^2$ costs $5 and $10; `$x_1$` stays.\n\n```\n$a_b$\n```\n\nand $x_1$."
    page_md = SimpleNamespace(file=SimpleNamespace(src_uri="guide/how-to-use.md"))
    out = hooks.on_page_markdown(md, page_md, {}, None)
    assert out == "Loss \\(\\alpha T^2\\) costs $5 and $10; `$x_1$` stays.\n\n```\n$a_b$\n```\n\nand \\(x_1\\)."
    assert hooks.on_page_markdown(out, page_md, {}, None) == out       # idempotent on a generated page


def test_dense_record_tables_become_cards_but_spec_tables_do_not():
    def table(cells):
        head = "".join(f"<th>h{i}</th>" for i in range(len(cells[0])))
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells)
        return f"<table>\n<thead>\n<tr>{head}</tr>\n</thead>\n<tbody>{body}</tbody>\n</table>"
    sentence = " ".join(["w"] * 12)
    records = table([["nb01", sentence, sentence, "—", sentence]] * 4)        # 60% of cells are 12 words
    assert hooks.stack_prose_tables(records)[1] == 1
    notes = [["T4", "16 GB", "~320 GB/s", "7.5", " ".join(["w"] * 6)]] * 5 + [["L4", "24 GB", "~300 GB/s", "8.9", " ".join(["w"] * 20)]]
    assert hooks.stack_prose_tables(table(notes))[1] == 0       # a short notes column with one long note: a table


def test_a_list_after_prose_gets_its_blank_line():
    md = ("**Three fixes:**\n- one\n- two\n\nAlready fine:\n\n- a\n- b\n\n"
          "1. first\n2. second\n\n> Quoted lead\n> - q1\n> - q2\n\n"
          "- item\n  continued line\n- next\n\n```\ntext\n- not a list\n```\n\n## Heading\n- under a heading\n"
          "| a | b |\n|---|---|\n- after a table")
    out = hooks.blank_line_before_lists(md)
    assert "**Three fixes:**\n\n- one\n- two" in out
    assert "Already fine:\n\n- a\n- b" in out                     # unchanged
    assert "> Quoted lead\n>\n> - q1\n> - q2" in out
    assert "- item\n  continued line\n- next" in out              # a continuation line is not prose
    assert "```\ntext\n- not a list\n```" in out
    assert "## Heading\n- under a heading" in out                  # headings and tables already end a block
    assert "|---|---|\n- after a table" in out
    assert hooks.blank_line_before_lists(out) == out                 # idempotent
