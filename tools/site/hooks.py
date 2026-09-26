"""MkDocs hooks for the guide site (registered under `hooks:` in mkdocs.yml).

1. Cache-busting for the site's own stylesheets and scripts. MkDocs links `extra_css` / `extra_javascript` at a
   fixed path, and GitHub Pages serves them with `Cache-Control: max-age=600`, so right after a deploy a browser can
   pair the new HTML with the previous `extra.css`. A hash of the file's content in the URL makes every change a
   new URL.
2. One notebook stylesheet for the whole site. mkdocs-jupyter inlines ~600 KB of JupyterLab and Pygments CSS into
   every notebook page and has no option to link it instead. The hook moves those <style> blocks into
   `assets/stylesheets/notebook.<hash>.css`, written once, and links it from each notebook page.
3. MathJax only where there is math, with our own configuration. mkdocs-jupyter also emits a MathJax 2 config
   (inline math between single dollars) and an empty <script src="">; the hook removes both. Pages whose content
   has \\( \\), \\[ \\], $$ or arithmatex spans get site/javascripts/mathjax.js (inline math only as \\( \\), so
   "$20 / $100" stays text) and MathJax 3.2.2 from jsDelivr. tools/site/build_site_content.py turns inline $...$
   TeX in notebooks into \\( \\).
4. Notebook tables of contents that point at their headings. mkdocs-jupyter builds a notebook page's table of
   contents from a Markdown copy with inline code removed, but ids the headings from the rendered HTML, so a heading
   with `code` in it gets a TOC link to an id that does not exist. The hook re-points each TOC entry at the heading
   it names (same level, same order), then counts the in-page anchors on notebook pages that still point nowhere
   and reports the total at the end of the build.
5. Search leaves out the answers: solution notebooks (build_site_content.is_solution) are excluded from the index,
   and on every notebook page the duplicate copy of each code cell (the text behind the copy button), cell outputs,
   the `In [ ]:` prompts and the "Copied!" notice are marked `data-search-exclude`.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

from mkdocs.utils import get_relative_url

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_content import is_solution  # noqa: E402

log = logging.getLogger("mkdocs.hooks.site")

MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
MATHJAX_CONFIG = "javascripts/mathjax.js"
NOTEBOOK_CSS = "assets/stylesheets/notebook.{}.css"

STYLE = re.compile(r"<style[^>]*>(.*?)</style>\s*", re.S)
MJ_BLOCK = re.compile(r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->", re.S)
HAS_MATH = re.compile(r"\\\(|\\\[|\$\$|\\begin\{|class=\"arithmatex\"")
CODE = re.compile(r"<pre\b.*?</pre>|<code\b.*?</code>|<div class=\"clipboard-copy-txt\".*?</div>", re.S)
HEADING = re.compile(r"<h([1-6])\b[^>]*\bid=\"([^\"]+)\"")
ID = re.compile(r"\bid=\"([^\"]+)\"")
HREF_FRAG = re.compile(r"\bhref=\"#([^\"]+)\"")
SEARCH_NOISE = re.compile(
    r"<(div|span) class=\"(clipboard-copy-txt|jp-InputPrompt[^\"]*|jp-OutputPrompt[^\"]*|jp-OutputArea [^\"]*|notice)\"")

_css: dict[str, str] = {}          # hash -> stylesheet text
_page_css: dict[str, str] = {}     # page src_uri -> hash of its notebook stylesheet
_math_pages: set[str] = set()
_dead: list[str] = []
_toc_fixed = 0
_config_hash = ""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


def on_config(config, **kwargs):
    global _config_hash, _toc_fixed
    docs = Path(config["docs_dir"])
    for key in ("extra_css", "extra_javascript"):
        busted = []
        for entry in config[key]:
            path = str(entry)
            src = docs / path
            if "?" not in path and "://" not in path and src.is_file():
                path = f"{path}?v={_digest(src.read_bytes())}"
            busted.append(path)
        config[key] = busted
    cfg = docs / MATHJAX_CONFIG
    _config_hash = _digest(cfg.read_bytes()) if cfg.is_file() else ""
    _toc_fixed = 0
    for store in (_css, _page_css):
        store.clear()
    _math_pages.clear()
    _dead.clear()
    return config


def _fix_notebook_toc(html: str, page) -> None:
    """Point each TOC entry at the next heading of its level, in document order."""
    global _toc_fixed
    heads = [(int(level), hid) for level, hid in HEADING.findall(html)]
    pos = 0

    def walk(items):
        nonlocal pos
        global _toc_fixed
        for item in items:
            for j in range(pos, len(heads)):
                if heads[j][0] == item.level:
                    if item.id != heads[j][1]:
                        item.id = heads[j][1]
                        _toc_fixed += 1
                    pos = j + 1
                    break
            walk(item.children)

    walk(page.toc)


def on_page_content(html, page, config, files, **kwargs):
    src = page.file.src_uri
    if src.endswith(".ipynb"):
        # The template's <style> blocks sit before the notebook wrapper and after it; <style> inside the wrapper
        # belongs to a cell's output (a pandas table) and stays. The last block after the wrapper repeats the
        # Pygments CSS as a Python list repr ("['pre {...']"): dropped.
        start, end = html.find('<div class="jupyter-wrapper">'), html.rfind("<!-- jupyter-wrapper -->")
        if start >= 0 and end > start:
            head, body, tail = html[:start], html[start:end], html[end:]
            blocks = [m.group(1).strip() for m in STYLE.finditer(head + tail)]
            css = "".join(b + "\n" for b in blocks if not b.startswith("['"))
            if css:
                h = _digest(css.encode())
                _css.setdefault(h, css)
                _page_css[src] = h
            html = STYLE.sub("", head) + body + STYLE.sub("", tail)
        html = MJ_BLOCK.sub("", html)
        html = SEARCH_NOISE.sub(lambda m: m.group(0) + " data-search-exclude", html)
        _fix_notebook_toc(html, page)
        if src.startswith("layers/") and is_solution(src[len("layers/"):]):
            page.meta["search"] = {**(page.meta.get("search") or {}), "exclude": True}
        ids = set(ID.findall(html))

        def toc_ids(items):
            for item in items:
                yield item.id
                yield from toc_ids(item.children)

        for frag in list(HREF_FRAG.findall(html)) + list(toc_ids(page.toc)):
            if frag not in ids:
                _dead.append(f"{src}#{frag}")
    if HAS_MATH.search(CODE.sub("", html)):
        _math_pages.add(src)
    return html


def on_post_page(output, page, config, **kwargs):
    src = page.file.src_uri
    if src in _page_css:
        href = get_relative_url(NOTEBOOK_CSS.format(_page_css[src]), page.url)
        output = output.replace("</head>", f'<link rel="stylesheet" href="{href}">\n</head>', 1)
    if src in _math_pages:
        cfg = get_relative_url(MATHJAX_CONFIG, page.url) + (f"?v={_config_hash}" if _config_hash else "")
        tags = f'<script src="{cfg}"></script>\n<script async src="{MATHJAX_CDN}"></script>\n'
        output = output.replace("</body>", tags + "</body>", 1)
    return output


def on_post_build(config, **kwargs):
    site = Path(config["site_dir"])
    for h, css in _css.items():
        out = site / NOTEBOOK_CSS.format(h)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(css, encoding="utf-8")
    log.info(f"site hooks: {len(_page_css)} notebook pages share {len(_css)} stylesheet(s); "
             f"{len(_math_pages)} pages load MathJax; {_toc_fixed} notebook TOC entries re-pointed; "
             f"{len(_dead)} dead in-page anchors on notebook pages")
    for d in _dead[:20]:
        log.info(f"  dead anchor: {d}")
