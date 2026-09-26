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
    sol = nb_page("layers/06-x/lab/notebooks/solutions/06_mcp_solution.ipynb")
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
