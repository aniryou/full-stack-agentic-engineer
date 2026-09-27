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
6. Wide prose tables become cards. A Markdown table whose rows are records with a paragraph in them (the
   curriculum's module tables: 7 columns, 60–100 words in "You can …"; a README's "What you get": 4 columns, a
   90-word cell) renders as a narrow, very tall grid. The hook marks such a table `fse-stacked` and gives each cell
   its column heading as `data-label`; extra.css lays each row out as a card — the first column as its title, the
   prose column at full width, the short fields in a row beneath. Numeric and short-text tables stay tables.
7. GitHub alerts become admonitions. A Markdown page's blockquote that opens with `> [!NOTE]` (or TIP, IMPORTANT,
   WARNING, CAUTION) is what GitHub renders as a callout; Python-Markdown would show the marker as text. The hook
   rewrites the block into Material's `!!! note "Note"` admonition before the page is rendered, so one source reads
   as a callout on GitHub and on the site. Other blockquotes are left alone (site/stylesheets/extra.css styles them).
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

# GitHub alert markers -> Material admonition type and title (docs.github.com "Alerts"; Material's admonition types).
ALERTS = {"NOTE": ("note", "Note"), "TIP": ("tip", "Tip"), "IMPORTANT": ("info", "Important"),
          "WARNING": ("warning", "Warning"), "CAUTION": ("danger", "Caution")}
ALERT_START = re.compile(r"^[ ]{0,3}>[ ]?\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*$", re.I)
QUOTE_LINE = re.compile(r"^[ ]{0,3}>[ ]?(.*)$")

TABLE = re.compile(r"<table>\s*<thead>\s*<tr>(.*?)</tr>\s*</thead>\s*<tbody>(.*?)</tbody>\s*</table>", re.S)
TH = re.compile(r"<th[^>]*>(.*?)</th>", re.S)
TR = re.compile(r"<tr>(.*?)</tr>", re.S)
TD = re.compile(r"<td([^>]*)>(.*?)</td>", re.S)
TAGS = re.compile(r"<[^>]+>")
STACK_MIN_COLS, STACK_PROSE_MAX, STACK_PROSE_AVG, STACK_3COL_MAX = 4, 40, 12, 120

_css: dict[str, str] = {}          # hash -> stylesheet text
_page_css: dict[str, str] = {}     # page src_uri -> hash of its notebook stylesheet
_math_pages: set[str] = set()
_dead: list[str] = []
_toc_fixed = 0
_stacked = 0
_config_hash = ""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


def on_config(config, **kwargs):
    global _config_hash, _toc_fixed, _stacked
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
    _stacked = 0
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


def _words(html: str) -> int:
    return len(TAGS.sub(" ", html).split())


def stack_prose_tables(html: str) -> tuple[str, int]:
    """Mark record-like tables (see the module docstring, item 6) with class fse-stacked and per-cell data-labels.
    Returns the HTML and the number of tables changed."""
    changed = 0

    def table(m):
        nonlocal changed
        heads = [TAGS.sub("", h).strip() for h in TH.findall(m.group(1))]
        rows = [TD.findall(r) for r in TR.findall(m.group(2))]
        rows = [r for r in rows if r]
        if not heads or not rows or any(len(r) != len(heads) for r in rows):
            return m.group(0)
        cols = len(heads)
        words = [[_words(c[1]) for c in r] for r in rows]
        col_max = [max(w[i] for w in words) for i in range(cols)]
        col_avg = [sum(w[i] for w in words) / len(words) for i in range(cols)]
        prose = [i for i in range(cols) if col_max[i] >= STACK_PROSE_MAX or col_avg[i] >= STACK_PROSE_AVG]
        wide = cols >= STACK_MIN_COLS and bool(prose)
        three = cols == 3 and max(col_max) >= STACK_3COL_MAX
        if not (wide or three):
            return m.group(0)
        changed += 1
        if three:
            prose = [i for i in range(cols) if col_max[i] >= STACK_PROSE_MAX]
        body = []
        for r in rows:
            cells = []
            for i, (attrs, inner) in enumerate(r):
                cls = " fse-key" if i == 0 else (" fse-prose" if i in prose else "")
                label = heads[i].replace('"', "&quot;")
                cells.append(f'<td{attrs} data-label="{label}" class="fse-cell{cls}">{inner}</td>')
            body.append("<tr>" + "".join(cells) + "</tr>")
        return (f'<table class="fse-stacked"><thead><tr>{m.group(1)}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table>')

    return TABLE.sub(table, html), changed


def github_alerts_to_admonitions(markdown: str) -> str:
    """Rewrite `> [!NOTE]` blockquotes (GitHub alerts) as `!!! note "Note"` admonitions; everything else unchanged.
    The alert is the whole run of `>` lines that follows the marker (lazy continuation lines are not GitHub's
    behaviour for alerts either). Fenced code is never touched."""
    out: list[str] = []
    lines = markdown.split("\n")
    i, fence = 0, None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            tok = stripped[:3]
            fence = None if fence == tok else (tok if fence is None else fence)
        m = ALERT_START.match(line) if fence is None else None
        if not m:
            out.append(line)
            i += 1
            continue
        kind, title = ALERTS[m.group(1).upper()]
        body: list[str] = []
        i += 1
        while i < len(lines) and QUOTE_LINE.match(lines[i]):
            body.append(QUOTE_LINE.match(lines[i]).group(1))
            i += 1
        while body and not body[-1].strip():
            body.pop()
        out.append(f'!!! {kind} "{title}"')
        out.extend(("    " + b) if b.strip() else "" for b in body)
        if i < len(lines) and lines[i].strip():
            out.append("")                     # an admonition ends at a blank line
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files, **kwargs):
    if page.file.src_uri.endswith(".md") and "[!" in markdown:
        return github_alerts_to_admonitions(markdown)
    return markdown


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
    elif src.endswith(".md") and "<table>" in html:
        html, n = stack_prose_tables(html)
        global _stacked
        _stacked += n
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
             f"{len(_dead)} dead in-page anchors on notebook pages; {_stacked} prose tables laid out as cards")
    for d in _dead[:20]:
        log.info(f"  dead anchor: {d}")
