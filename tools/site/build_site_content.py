#!/usr/bin/env python3
"""Assemble the guide site's generated pages from the repo (standard library only, idempotent).

Writes, under the MkDocs docs_dir `site/`:
  layers/<layer>/...            every layer's README (as index.md), every Markdown doc, every notebook, and
                                the images those docs reference, with relative links rewritten for the site
  layers/index.md               the stack diagram + one row per layer
  guide/curriculum.md, guide/compute.md, guide/colab.md   from CURRICULUM.md, COMPUTE.md, COLAB.md
and rewrites, in mkdocs.yml, the `nav:` entries between `# nav-layers:start` / `# nav-layers:end` and the
`not_in_nav:` patterns between `# not-in-nav:start` / `# not-in-nav:end` (deploy targets, fixtures and other
plumbing keep their pages, reachable from the READMEs that link them, but stay out of the navigation).

Everything written here is gitignored (site/.gitignore); run it before `mkdocs build` or `mkdocs serve`:
    python3 tools/site/build_site_content.py
Link rules: a relative link to something that has a page on the site points at that page (anchors are
re-slugified the way MkDocs' toc does for Markdown pages and the way mkdocs-jupyter does for notebooks; an anchor
whose heading cannot be found is dropped: the link keeps its page, and a same-page link becomes plain text);
anything else in the repo (code, Terraform, YAML, folders without a README, LICENSE, ...) points at GitHub;
absolute URLs are untouched.
Math: in notebook Markdown, inline TeX written as $...$ becomes \\(...\\), the only inline delimiter the site's
MathJax accepts, so dollar amounts ("$20 / $100") stay text; see site/javascripts/mathjax.js.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
OUT = SITE / "layers"
MKDOCS_YML = REPO / "mkdocs.yml"

# Colab URL convention: identical to tools/gen_colab_index.py
BRANCH, REPO_SLUG = "main", "aniryou/full-stack-agentic-engineer"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
GITHUB = f"https://github.com/{REPO_SLUG}"


def colab(p: str) -> str:
    return f"https://colab.research.google.com/github/{REPO_SLUG}/blob/{BRANCH}/{p}"


LAYER_TITLES = {
    "00": "00 · Foundations",
    "01": "01 · Hardware & fabric",
    "02": "02 · CUDA, NCCL & runtime",
    "03": "03 · Kubernetes & GPUs",
    "04": "04 · Inference engine",
    "05": "05 · Orchestrator",
    "06": "06 · Gateway",
    "07": "07 · Agents & applications",
}
GUIDE_PAGES = {"CURRICULUM.md": "guide/curriculum.md", "COMPUTE.md": "guide/compute.md",
               "COLAB.md": "guide/colab.md"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "node_modules", ".terraform", ".venv", "venv", "_run_outputs", "site_build", ".eggs"}
DATA_DIRS = {"data", "corpus"}          # Markdown under these is data a lab reads, not a document
IMAGE_EXT = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
NOTEBOOK_DIRS = {"notebooks": "Notebooks", "exercises": "Exercises", "practice": "Practice",
                 "lessons": "Lessons"}
# Notebooks with the answers filled in, in every convention the repo uses: a solutions/ or worked/ folder, or a
# name like 01_x_solution(s), 01_x_solved, 01_x_worked, 01_worked. Kept identical to tools/gen_colab_index.py.
SOLUTION_DIRS = {"solutions", "worked"}
SOLUTION_STEM = re.compile(r"(?:^|[_\-.])(?:solutions?|solved|worked)(?:$|[_\-.])", re.I)
# Folders that are plumbing, not lessons: their Markdown still becomes pages (the lab READMEs link them), but they
# are left out of the navigation and listed under `not_in_nav:` in mkdocs.yml.
PLUMBING_DIRS = ("client", "deploy", "fixtures", "infra", "notebooks_src")
# Folders whose name says nothing about the lesson: a primer inside one is named after the nearest real folder.
GENERIC_DIRS = {"docs", "notebooks", "solutions", "worked", "exercises", "practice", "lessons"}

stats = {"pages": 0, "notebooks": 0, "colab": 0, "images": 0, "to_page": 0, "to_github": 0,
         "anchors_kept": 0, "anchors_dropped": 0, "nb_anchors_kept": 0, "nb_anchors_dropped": 0,
         "math": 0, "missing": 0, "skipped": 0}
missing_examples: list[str] = []
dropped_anchors: list[str] = []


def read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"  skip {p.relative_to(REPO)}: {e}", file=sys.stderr)
        stats["skipped"] += 1
        return None


# ---------------------------------------------------------------- inventory

pages: dict[str, str] = {}       # repo path (posix) -> site path of a Markdown page
notebooks: dict[str, str] = {}   # repo path -> site path of a notebook page
images: dict[str, str] = {}      # repo path -> site path of a copied image


def is_solution(repo_path: str) -> bool:
    """True for a notebook with the answers in it (see SOLUTION_DIRS / SOLUTION_STEM)."""
    parts = repo_path.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    return bool(SOLUTION_DIRS & set(parts[:-1])) or bool(SOLUTION_STEM.search(stem))


def is_plumbing(repo_path: str) -> bool:
    return bool(set(PLUMBING_DIRS) & set(repo_path.split("/")[1:-1]))


def inventory() -> list[str]:
    layers = sorted(d.name for d in REPO.iterdir() if d.is_dir() and re.match(r"\d\d-", d.name))
    for layer in layers:
        for root, dirs, files in os.walk(REPO / layer):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            rel_root = Path(root).relative_to(REPO).as_posix()
            in_data = bool(DATA_DIRS & set(rel_root.split("/")))
            for f in sorted(files):
                rp = f"{rel_root}/{f}"
                if f.endswith(".md") and not in_data:
                    name = "index.md" if f == "README.md" else f
                    if f == "index.md" and (Path(root) / "README.md").exists():
                        continue
                    pages[rp] = f"layers/{rel_root}/{name}"
                elif f.endswith(".ipynb"):
                    notebooks[rp] = f"layers/{rp}"
    for src, dst in GUIDE_PAGES.items():
        if (REPO / src).exists():
            pages[src] = dst
    if (SITE / "index.md").exists():          # the landing page (owned by the site design, not generated)
        pages["README.md"] = "index.md"
    return layers


# ---------------------------------------------------------------- slugs

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def heading_text(raw: str) -> str:
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", raw)      # links / images -> text
    t = re.sub(r"<[^>]+>", "", t)                           # html tags
    t = t.replace("`", "").replace("**", "").replace("__", "")
    t = re.sub(r"(?<!\w)[*_](\S[^*_]*?)[*_](?!\w)", r"\1", t)
    t = re.sub(r"\s+#+\s*$", "", t)                          # closing hashes
    return t.replace("&amp;", "&").strip()


def mkdocs_slug(text: str) -> str:
    """markdown.extensions.toc.slugify (MkDocs' default)."""
    v = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    v = re.sub(r"[^\w\s-]", "", v).strip().lower()
    return re.sub(r"[-\s]+", "-", v)


def github_slug(text: str) -> str:
    v = text.strip().lower()
    v = re.sub(r"[^\w\- ]", "", v)
    return v.replace(" ", "-")


_anchor_cache: dict[str, dict[str, str]] = {}


def anchors_of(repo_path: str) -> dict[str, str]:
    """Map every plausible anchor spelling for a Markdown file's headings -> the MkDocs id."""
    if repo_path in _anchor_cache:
        return _anchor_cache[repo_path]
    amap: dict[str, str] = {}
    text = read(REPO / repo_path) or ""
    seen_md: dict[str, int] = {}
    seen_gh: dict[str, int] = {}
    in_fence = None
    for line in text.splitlines():
        m = FENCE.match(line)
        if m:
            tok = m.group(1)
            if in_fence is None:
                in_fence = tok[0] * 3
            elif tok.startswith(in_fence):
                in_fence = None
            continue
        if in_fence:
            continue
        h = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if not h:
            continue
        t = heading_text(h.group(1))
        s = mkdocs_slug(t)
        n = seen_md.get(s, 0)
        seen_md[s] = n + 1
        md_id = s if n == 0 else f"{s}_{n}"
        g = github_slug(t)
        k = seen_gh.get(g, 0)
        seen_gh[g] = k + 1
        gh_id = g if k == 0 else f"{g}-{k}"
        for key in (gh_id, md_id):
            amap.setdefault(key, md_id)
    _anchor_cache[repo_path] = amap
    return amap


_nb_cache: dict[str, dict] = {}


def load_notebook(repo_path: str) -> dict:
    if repo_path not in _nb_cache:
        try:
            _nb_cache[repo_path] = json.loads(read(REPO / repo_path) or "{}")
        except json.JSONDecodeError:
            _nb_cache[repo_path] = {}
    return _nb_cache[repo_path]


def nb_anchors_of(repo_path: str) -> dict[str, str]:
    """Map every plausible anchor spelling for a notebook's headings -> the id mkdocs-jupyter gives the heading.

    mkdocs-jupyter ids a heading slugify(text) (the same slugify as MkDocs' toc) and does not de-duplicate.
    Authors write anchors the way GitHub or Jupyter render the notebook: GitHub's slug, or Jupyter's, which is the
    heading text with spaces turned into hyphens (case and punctuation kept)."""
    key = "nb:" + repo_path
    if key in _anchor_cache:
        return _anchor_cache[key]
    amap: dict[str, str] = {}
    seen_gh: dict[str, int] = {}
    for c in md_cells(load_notebook(repo_path)):
        in_fence = None
        for line in cell_text(c).splitlines():
            m = FENCE.match(line)
            if m:
                tok = m.group(1)
                if in_fence is None:
                    in_fence = tok[0] * 3
                elif tok.startswith(in_fence):
                    in_fence = None
                continue
            if in_fence:
                continue
            h = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
            if not h:
                continue
            t = heading_text(h.group(1))
            nb_id = mkdocs_slug(t)
            if not nb_id:
                continue
            g = github_slug(t)
            k = seen_gh.get(g, 0)
            seen_gh[g] = k + 1
            jup = re.sub(r"\s+", "-", t.strip())
            for spelling in (nb_id, g if k == 0 else f"{g}-{k}", jup, jup.lower()):
                amap.setdefault(spelling, nb_id)
    _anchor_cache[key] = amap
    return amap


def resolve_anchor(repo_path: str, frag: str) -> str | None:
    amap = nb_anchors_of(repo_path) if repo_path.endswith(".ipynb") else anchors_of(repo_path)
    frag = unquote(frag)
    return amap.get(frag) or amap.get(frag.lower())


def count_anchor(repo_path: str, kept: bool, src_repo: str, target: str) -> None:
    prefix = "nb_" if repo_path.endswith(".ipynb") else ""
    stats[prefix + ("anchors_kept" if kept else "anchors_dropped")] += 1
    if not kept:
        dropped_anchors.append(f"{src_repo} -> {target}")


# ---------------------------------------------------------------- link rewriting

SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^//")


def page_url(site_path: str) -> str:
    """Final URL (relative to the site root, directory URLs) of a generated file."""
    if site_path.endswith((".md", ".ipynb")):
        base = site_path.rsplit(".", 1)[0]
        if base == "index":
            return ""
        if base.endswith("/index"):
            return base[: -len("index")]
        return base + "/"
    return site_path


def relative(target_site: str, src_site: str, html: bool) -> str:
    if not html:   # MkDocs resolves links between source files
        return posixpath.relpath(target_site, posixpath.dirname(src_site))
    src_dir = page_url(src_site).rstrip("/") or "."
    tgt = page_url(target_site)
    rel = posixpath.relpath(tgt.rstrip("/") or ".", src_dir)
    is_dir_url = tgt == "" or tgt.endswith("/")
    return (rel + "/") if is_dir_url else rel


def rewrite_target(target: str, src_repo: str, src_site: str, html: bool) -> str | None:
    """The site's version of a link target, or None for a same-page anchor to a heading that does not exist
    (the caller turns that link into plain text)."""
    raw = target.strip()
    if not raw or SCHEME.match(raw):
        return target
    path, _, frag = raw.partition("#")
    path = path.split("?", 1)[0]
    if not path:                                    # same-page anchor
        if frag and src_repo.endswith((".md", ".ipynb")):
            new = resolve_anchor(src_repo, frag)
            count_anchor(src_repo, bool(new), src_repo, target)
            return "#" + new if new else None
        return target
    rp = posixpath.normpath(posixpath.join(posixpath.dirname(src_repo), unquote(path)))
    if rp.startswith(".."):
        stats["missing"] += 1
        missing_examples.append(f"{src_repo} -> {target}")
        return target
    rp = "" if rp == "." else rp
    full = REPO / rp
    if full.is_dir() or path.endswith("/"):
        readme = f"{rp}/README.md" if rp else "README.md"
        if readme in pages:
            stats["to_page"] += 1
            return relative(pages[readme], src_site, html)
        stats["to_github"] += 1
        return f"{GITHUB}/tree/{BRANCH}/{rp}".rstrip("/")
    if rp in pages or rp in notebooks:
        stats["to_page"] += 1
        out = relative(pages[rp] if rp in pages else notebooks[rp], src_site, html)
        if frag:
            new = resolve_anchor(rp, frag)
            count_anchor(rp, bool(new), src_repo, target)
            if new:
                out += "#" + new
        return out
    if rp in images:
        return relative(images[rp], src_site, html)
    if not full.exists():
        stats["missing"] += 1
        missing_examples.append(f"{src_repo} -> {target}")
    stats["to_github"] += 1
    return f"{GITHUB}/blob/{BRANCH}/{rp}" + (f"#{frag}" if frag else "")


# Inline TeX between single dollars, by pandoc's rule (no space inside either dollar, no digit right after the
# closing one), and only when it looks like TeX: a backslash, ^, _ or braces, or a single letter ($x$). "$20 / $100"
# fails the rule; "$5 and $10" too. Display math ($$...$$) is left as it is: MathJax takes $$ as display math.
INLINE_TEX = re.compile(r"(?<![\\$\w])\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<![\s\\])\$(?![\d$])")
TEXISH = re.compile(r"[\\^_{}]|^[A-Za-z]'*$")
MATH_ENTITIES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\\": "&#92;", "_": "&#95;", "*": "&#42;", "`": "&#96;",
                 "[": "&#91;", "]": "&#93;", "|": "&#124;", "~": "&#126;"}


def inline_tex_to_parens(text: str) -> str:
    """$x_1$ -> \\(x_1\\), written with character references so Markdown passes it through untouched."""
    def conv(m):
        body = m.group(1)
        if not TEXISH.search(body):
            return m.group(0)
        stats["math"] += 1
        enc = "".join(MATH_ENTITIES.get(ch, ch) for ch in body)
        return f"&#92;({enc}&#92;)"
    return INLINE_TEX.sub(conv, text)


LINK = re.compile(r"(!?\[((?:[^\[\]]|\[[^\]]*\])*)\])\(\s*(<[^>]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(\s+\"[^\"]*\")?\s*\)")
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:[ \t]*)(\S+)(.*)$", re.M)
HTMLATTR = re.compile(r"\b(src|href)=\"([^\"]+)\"")
CODESPAN = re.compile(r"(`+)(.+?)\1")


def rewrite_segment(s: str, fn) -> str:
    def link(m):
        label, inner, tgt, title = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        if "](" in inner:   # a linked image: rewrite the inner link too
            label = label[0:label.index("[") + 1] + rewrite_segment(inner, fn) + "]"
        bracketed = tgt.startswith("<") and tgt.endswith(">")
        new = fn(tgt[1:-1] if bracketed else tgt)
        if new is None:     # a dead same-page anchor: keep the words, drop the link
            return m.group(0) if label.startswith("!") else label[label.index("[") + 1:-1]
        return f"{label}({'<' + new + '>' if bracketed else new}{title})"
    s = LINK.sub(link, s)
    return HTMLATTR.sub(lambda m: f'{m.group(1)}="{fn(m.group(2)) or m.group(2)}"', s)


def rewrite_markdown(text: str, src_repo: str, src_site: str, html: bool = False, math: bool = False) -> str:
    """Rewrite link targets in Markdown, outside fenced code and inline code; links may span lines.
    With math=True (notebook cells), also turn inline $...$ TeX into \\(...\\) (inline_tex_to_parens)."""
    fn = lambda t: rewrite_target(t, src_repo, src_site, html)  # noqa: E731
    out: list[str] = []
    prose: list[str] = []

    def flush():
        if not prose:
            return
        block = "\n".join(prose)
        prose.clear()
        block = REFDEF.sub(lambda r: r.group(1) + (fn(r.group(2)) or r.group(2)) + r.group(3), block)
        spans: list[str] = []            # mask inline code so links inside it stay as written

        def mask(c):
            spans.append(c.group(0))
            return f"\x00{len(spans) - 1}\x00"
        masked = CODESPAN.sub(mask, block)
        if math:
            masked = inline_tex_to_parens(masked)
        masked = rewrite_segment(masked, fn)
        out.append(re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], masked))

    in_fence = None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if m:
            flush()
            tok = m.group(1)
            if in_fence is None:
                in_fence = tok[0] * 3
            elif tok.startswith(in_fence):
                in_fence = None
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            prose.append(line)
    flush()
    return "\n".join(out)


def collect_images(text: str, src_repo: str) -> None:
    for m in re.finditer(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)|<img[^>]*\bsrc=\"([^\"]+)\"", text):
        t = m.group(1) or m.group(2)
        if SCHEME.match(t):
            continue
        rp = posixpath.normpath(posixpath.join(posixpath.dirname(src_repo), unquote(t.split("#")[0])))
        if (Path(rp).suffix.lower() in IMAGE_EXT and re.match(r"\d\d-", rp) and (REPO / rp).is_file()):
            images.setdefault(rp, f"layers/{rp}")


# ---------------------------------------------------------------- writers

def write(site_path: str, content: str) -> None:
    p = SITE / site_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def front_matter(**meta: str) -> str:
    """YAML front matter for a generated page. site/overrides/main.html turns `source_path` (a repo path) or
    `source_url` into the "View on GitHub" link at the top of the page. JSON strings are valid YAML scalars."""
    return "---\n" + "".join(f"{k}: {json.dumps(v)}\n" for k, v in meta.items()) + "---\n\n"


def md_cells(nb: dict):
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            yield c


def cell_text(c) -> str:
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def build_pages() -> None:
    texts: dict[str, str] = {}
    for rp in pages:
        if rp == "README.md":
            continue
        t = read(REPO / rp)
        if t is not None:
            texts[rp] = t
            collect_images(t, rp)
    nbs: dict[str, dict] = {}
    for rp in notebooks:
        t = read(REPO / rp)
        if t is None:
            continue
        try:
            nbs[rp] = json.loads(t)
        except json.JSONDecodeError as e:
            print(f"  skip {rp}: {e}", file=sys.stderr)
            stats["skipped"] += 1
            continue
        for c in md_cells(nbs[rp]):
            collect_images(cell_text(c), rp)
    for rp, site_path in images.items():
        dst = SITE / site_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rp, dst)
        stats["images"] += 1
    for rp, t in texts.items():
        write(pages[rp], front_matter(source_path=rp) + rewrite_markdown(t, rp, pages[rp]))
        stats["pages"] += 1
    for rp, nb in nbs.items():
        site_path = notebooks[rp]
        for c in md_cells(nb):
            c["source"] = rewrite_markdown(cell_text(c), rp, site_path, html=True, math=True)
        if is_solution(rp):   # no Colab badge (the badge is the call to action for exercises), a plain link
            head = (f"*Worked answers: try the exercise version first.* "
                    f"[View on GitHub]({GITHUB}/blob/{BRANCH}/{rp}) · [Open in Colab]({colab(rp)})")
        else:
            head = (f"[![Open In Colab]({BADGE})]({colab(rp)}) &nbsp; "
                    f"[View on GitHub]({GITHUB}/blob/{BRANCH}/{rp})")
            stats["colab"] += 1
        # The Colab setup cell (always first, a no-op off Colab) is plumbing, not lesson: leave it out of the page.
        cells = nb.get("cells", [])
        if cells and cells[0].get("cell_type") == "code" and "google.colab" in cell_text(cells[0]):
            del cells[0]
        if (nb.get("nbformat", 4), nb.get("nbformat_minor", 0)) >= (4, 5):
            for i, c in enumerate(nb.get("cells", [])):
                c.setdefault("id", f"cell-{i}")
        cell = {"cell_type": "markdown", "metadata": {"tags": ["site-header"]}, "source": head}
        if (nb.get("nbformat", 4), nb.get("nbformat_minor", 0)) >= (4, 5):
            cell["id"] = "site-header"
        nb.setdefault("cells", []).insert(0, cell)
        write(site_path, json.dumps(nb, ensure_ascii=False, indent=1))
        notebooks[rp] = site_path
        stats["notebooks"] += 1


# ---------------------------------------------------------------- layer index + nav

def promise(readme: str) -> str:
    lines = readme.splitlines()
    para: list[str] = []
    started = False
    for ln in lines[1:] if lines and lines[0].startswith("# ") else lines:
        s = ln.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith(("#", "```", "|", ">", "<!--", "- ", "* ")):
            if para:
                break
            if started:
                continue
            continue
        started = True
        para.append(s)
    text = " ".join(para)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text).replace("**", "")
    return re.split(r"(?<=\.)\s+(?=[A-Z(])", text)[0] if text else ""


def stack_diagram() -> str:
    t = read(REPO / "README.md") or ""
    m = re.search(r"## The stack\s*\n+(```.*?```)", t, re.S)
    return m.group(1) if m else ""


def build_layers_index(layers: list[str]) -> None:
    rows = []
    for layer in layers:
        readme = read(REPO / layer / "README.md") if (REPO / layer / "README.md").exists() else None
        nb = sum(1 for rp in notebooks if rp.startswith(layer + "/"))
        pg = sum(1 for rp in pages if rp.startswith(layer + "/"))
        title = LAYER_TITLES.get(layer[:2], layer)
        link = f"[{title}]({layer}/index.md)" if readme is not None else title
        # One heading and paragraph per layer, not a table row: a four-column table with a prose column
        # collapses to a word a line on a phone.
        rows += [f"## {link}", "", promise(readme or ""), "", f"*{pg} pages · {nb} notebooks*", ""]
    body = [
        "# The stack, layer by layer", "",
        "Eight layers, bottom-up: the model-level foundations (00), the hardware (01), and every layer of software "
        "between a GPU and an agent (02–07). Each layer page lists its topics; each topic has a primer, "
        "runnable code and notebooks. The suggested order to work through them is in the "
        "[curriculum](../guide/curriculum.md).", "",
        stack_diagram(), "",
        *rows,
    ]
    write("layers/index.md", front_matter(source_url=f"{GITHUB}/tree/{BRANCH}") + "\n".join(body))


ACRONYMS = {w.lower(): w for w in (
    "LLM LLMs GPU GPUs P2P KV HPA MCP RAG API SIMT NCCL CUDA TP DRA GKE ADK A2A TTFT TPOT HBM vLLM "
    "LoRA MoE GCP DWS OAuth PQ IVF HNSW GraphRAG TF CPU SLO SLOs PD HITL JWT RDMA MIG DCGM RL LRA EP GRPO DPO "
    "Mistral K8s").split()}
COMPOUNDS = ("long-running",)          # hyphenated words that stay hyphenated in a title
SLUGLIKE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
# A folder named for a provider variant (agent-core vs mistral-agent-core) says so in its title.
VARIANTS = {"mistral": ("Mistral",), "gcp": ("GCP", "Google Cloud")}
DIR_TITLES = {"docs": "Docs", "solutions": "Solutions", "worked": "Worked", **NOTEBOOK_DIRS}
TITLE_SPLIT = re.compile(r"\s+[—–]\s+|:\s+")


def humanize(stem: str) -> str:
    m = re.match(r"^(\d+[a-z]?)[_\-. ]+(.*)$", stem)
    num, rest = (m.group(1), m.group(2)) if m else ("", stem)
    for c in COMPOUNDS:
        rest = rest.replace(c, c.replace("-", "\0"))
    words = rest.replace("_", " ").replace("-", " ").replace("\0", "-").split()
    out = [ACRONYMS.get(w.lower(), w) for w in words]
    if out and out[0] == words[0]:      # capitalise the first word unless it is a spelled acronym (vLLM)
        out[0] = out[0][:1].upper() + out[0][1:]
    rest = " ".join(out)
    return f"{num} · {rest}" if num and rest else (num or rest)


def short_title(h: str, limit: int = 60) -> str:
    """A heading short enough for the sidebar: whole if it fits, else cut at the last dash or colon that fits."""
    if len(h) <= limit:
        return h
    cuts = [m.start() for m in TITLE_SPLIT.finditer(h) if 4 <= m.start() <= limit]
    return h[: cuts[-1]].strip() if cuts else h


def first_h1(text: str) -> str | None:
    in_fence = None
    for ln in text.splitlines():
        m = FENCE.match(ln)
        if m:
            tok = m.group(1)
            if in_fence is None:
                in_fence = tok[0] * 3
            elif tok.startswith(in_fence):
                in_fence = None
            continue
        if not in_fence and ln.startswith("# "):
            return heading_text(ln[2:]) or None
    return None


_dir_titles: dict[str, str] = {}


def section_title(repo_dir: str) -> str:
    """Sidebar title of a folder: its layer name, a fixed name for notebooks/solutions/docs folders, or the name part
    of its README's H1 (humanised when that is just the folder name), or else the folder name humanised."""
    if repo_dir in _dir_titles:
        return _dir_titles[repo_dir]
    name = posixpath.basename(repo_dir)
    title = None
    if "/" not in repo_dir:
        title = LAYER_TITLES.get(name[:2], name)
    elif name in DIR_TITLES:
        title = DIR_TITLES[name]
    else:
        readme = f"{repo_dir}/README.md"
        h = first_h1(read(REPO / readme) or "") if readme in pages else None
        if h:
            head = TITLE_SPLIT.split(h)[0].strip()   # README-STYLE H1s are "<name> — <promise>": keep the name
            title = humanize(head) if SLUGLIKE.match(head) else short_title(head if len(head) >= 4 else h)
        title = title or humanize(name)
        tokens = set(re.split(r"[-_]", name.lower()))
        for token, words in VARIANTS.items():
            if token in tokens and not any(w.lower() in title.lower() for w in words):
                title += f" ({words[0]})"
    _dir_titles[repo_dir] = title
    return title


def topic_title(repo_dir: str) -> str:
    """The nearest folder whose name means something (not docs/, notebooks/...), for naming a primer."""
    while posixpath.basename(repo_dir) in GENERIC_DIRS and "/" in repo_dir:
        repo_dir = posixpath.dirname(repo_dir)
    return re.sub(r"\s*\([^)]*\)$", "", section_title(repo_dir))


def md_title(rp: str) -> str:
    name = posixpath.basename(rp)
    if name.upper() == "PRIMER.MD":
        return f"{topic_title(posixpath.dirname(rp))} primer"
    h = first_h1(read(REPO / rp) or "")
    return short_title(h) if h else humanize(name[:-3])


def nb_title(rp: str) -> str:
    """A notebook's sidebar title: its first H1 (shortened), numbered like its file name; a solution says so."""
    stem = posixpath.basename(rp)[:-len(".ipynb")]
    h = None
    for c in md_cells(load_notebook(rp)):
        h = first_h1(cell_text(c))
        if h:
            break
    title = short_title(h) if h else humanize(stem)
    num = re.match(r"^(\d+)[_\-. ]", stem)
    if num and not title[:1].isdigit():
        title = f"{num.group(1)} · {title}"
    if is_solution(rp) and not re.search(r"solution|solved|worked|answer", title, re.I):
        worked = "worked" in rp.split("/")[:-1] or re.search(r"worked", stem, re.I)
        title += " (worked)" if worked else " (solution)"
    return title


def dir_rank(name: str) -> tuple[int, str]:
    if name == "docs":
        return (1, name)
    if name in NOTEBOOK_DIRS:
        return (5, name)
    if name in SOLUTION_DIRS:
        return (6, name)
    if "core" in name:
        return (2, name)
    if "lab" in name:
        return (3, name)
    return (4, name)


def leaves(entries: list[tuple[str, str, str]]) -> list[dict]:
    """[(title, site path, repo path)] -> nav leaves; titles that collide fall back to the file name."""
    seen: dict[str, int] = {}
    for t, _, _ in entries:
        seen[t] = seen.get(t, 0) + 1
    out = []
    for t, site_path, rp in entries:
        if seen[t] > 1:
            t = humanize(posixpath.basename(rp).rsplit(".", 1)[0])
        out.append({t: site_path})
    return out


def nav_for_dir(repo_dir: str) -> list:
    """Nav entries for one directory: README (section index) → primers → other docs → core → lab → notebooks →
    solutions. Plumbing folders (PLUMBING_DIRS) are left out; a folder with nothing but one section or one page
    collapses into it."""
    items: list = []
    index = f"{repo_dir}/README.md"
    if index in pages:
        items.append(pages[index])
    here_md = sorted((rp for rp in pages if posixpath.dirname(rp) == repo_dir and rp != index),
                     key=lambda rp: (0 if "primer" in rp.lower().rsplit("/", 1)[-1] else 1, rp.lower()))
    items += leaves([(md_title(rp), pages[rp], rp) for rp in here_md])
    here_nb = sorted(rp for rp in notebooks if posixpath.dirname(rp) == repo_dir)
    in_solutions_dir = posixpath.basename(repo_dir) in SOLUTION_DIRS
    blanks = [rp for rp in here_nb if in_solutions_dir or not is_solution(rp)]
    sols = [rp for rp in here_nb if not in_solutions_dir and is_solution(rp)]
    subdirs = sorted({rp[len(repo_dir) + 1:].split("/")[0] for rp in list(pages) + list(notebooks)
                      if rp.startswith(repo_dir + "/") and "/" in rp[len(repo_dir) + 1:]}, key=dir_rank)
    subdirs = [d for d in subdirs if d not in PLUMBING_DIRS]

    def add_dir(d: str) -> None:
        child = f"{repo_dir}/{d}"
        sub = nav_for_dir(child)
        if not sub:
            return
        has_readme = f"{child}/README.md" in pages
        if len(sub) == 1 and isinstance(sub[0], str):          # only a README: a page, not a section
            items.append({section_title(child): sub[0]})
        elif len(sub) == 1 and not has_readme:                 # a wrapper folder around one entry
            items.append(sub[0])
        elif (d in NOTEBOOK_DIRS and not has_readme            # notebooks/ holding only practice/ and worked/
              and all(isinstance(e, dict) and isinstance(next(iter(e.values())), list) for e in sub)):
            items.extend(sub)
        else:
            items.append({section_title(child): sub})

    for d in [d for d in subdirs if dir_rank(d)[0] < 5]:
        add_dir(d)
    items += leaves([(nb_title(rp), notebooks[rp], rp) for rp in blanks])
    for d in [d for d in subdirs if dir_rank(d)[0] >= 5]:
        add_dir(d)
    if sols:
        worked_only = all("worked" in posixpath.basename(rp).lower() for rp in sols)
        items.append({"Worked" if worked_only else "Solutions":
                      leaves([(nb_title(rp), notebooks[rp], rp) for rp in sols])})
    return items


def variant_of(repo_path: str) -> tuple[str, ...]:
    """The provider a path's folders name (lab-mistral/, mistral-agent-core/ -> ("Mistral",)), if any."""
    tokens = {t for part in repo_path.split("/")[:-1] for t in re.split(r"[-_]", part.lower())}
    return next((words for token, words in VARIANTS.items() if token in tokens), ())


def mark_variant_duplicates(items: list) -> list:
    """A page title used more than once in the nav (agent-core's and mistral-agent-core's "01 · The agent loop")
    names its provider when its folder is a provider variant, so search results and tabs tell them apart."""
    counts: dict[str, int] = {}

    def count(entries):
        for e in entries:
            if isinstance(e, dict):
                (t, v), = e.items()
                if isinstance(v, str):
                    counts[t] = counts.get(t, 0) + 1
                else:
                    count(v)

    def rename(entries):
        out = []
        for e in entries:
            if isinstance(e, dict):
                (t, v), = e.items()
                if isinstance(v, list):
                    e = {t: rename(v)}
                else:
                    words = variant_of(v[len("layers/"):]) if counts.get(t, 0) > 1 else ()
                    if words and not any(w.lower() in t.lower() for w in words):
                        m = re.match(r"^(.*) \((solution|worked)\)$", t)
                        t = f"{m.group(1)} ({m.group(2)}, {words[0]})" if m else f"{t} ({words[0]})"
                    e = {t: v}
            out.append(e)
        return out

    count(items)
    return rename(items)


def yaml_nav(items: list, indent: int) -> list[str]:
    pad = " " * indent
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(f"{pad}- {json.dumps(it, ensure_ascii=False)}")
        else:
            (title, val), = it.items()
            if isinstance(val, str):
                out.append(f"{pad}- {json.dumps(title, ensure_ascii=False)}: {json.dumps(val, ensure_ascii=False)}")
            else:
                out.append(f"{pad}- {json.dumps(title, ensure_ascii=False)}:")
                out += yaml_nav(val, indent + 4)
    return out


def replace_block(text: str, marker: str, body: list[str]) -> str | None:
    """Replace the lines between `# <marker>:start` and `# <marker>:end` (keeping the markers' indent)."""
    m = re.search(rf"^([ \t]*)# {marker}:start\n.*?^[ \t]*# {marker}:end[ \t]*$", text, re.S | re.M)
    if not m:
        print(f"  mkdocs.yml: {marker} markers not found; left unchanged", file=sys.stderr)
        return None
    pad = m.group(1)
    block = [f"{pad}# {marker}:start"] + body + [f"{pad}# {marker}:end"]
    return text[: m.start()] + "\n".join(block) + text[m.end():]


def update_mkdocs_yml(layers: list[str]) -> str:
    """Rewrite the generated parts of mkdocs.yml: the layer nav and the not_in_nav patterns."""
    text = read(MKDOCS_YML)
    if text is None:
        return "unreadable"
    layer_items: list = ["layers/index.md"]
    for layer in layers:
        sub = nav_for_dir(layer)
        if sub:
            layer_items.append({section_title(layer): sub})
    layer_items = mark_variant_duplicates(layer_items)
    new = replace_block(text, "nav-layers", ["  - Layers:"] + yaml_nav(layer_items, 6))
    if new is None:
        return "markers missing"
    plumbing = ["not_in_nav: |"] + [f"  /layers/**/{d}/" for d in PLUMBING_DIRS]
    new = replace_block(new, "not-in-nav", plumbing)
    if new is None:
        return "markers missing"
    if new == text:
        return "unchanged"
    MKDOCS_YML.write_text(new, encoding="utf-8")
    return "updated"


def nav_stats(items: list, depth: int = 1) -> tuple[int, int, int]:
    """(leaves, sections, max depth) of a nav tree."""
    n_leaf = n_sec = 0
    deepest = depth
    for it in items:
        if isinstance(it, str):
            n_leaf += 1
            continue
        (_, val), = it.items()
        if isinstance(val, str):
            n_leaf += 1
        else:
            a, b, c = nav_stats(val, depth + 1)
            n_leaf, n_sec, deepest = n_leaf + a, n_sec + 1 + b, max(deepest, c)
    return n_leaf, n_sec, deepest


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for dst in GUIDE_PAGES.values():
        (SITE / dst).unlink(missing_ok=True)
    layers = inventory()
    build_pages()
    build_layers_index(layers)
    yml = update_mkdocs_yml(layers)
    for ex in missing_examples[:10]:
        print(f"  missing target: {ex}", file=sys.stderr)
    for ex in dropped_anchors[: None if os.environ.get("SITE_VERBOSE") else 10]:
        print(f"  dropped anchor: {ex}", file=sys.stderr)
    s = stats
    n_leaf, n_sec, deepest = nav_stats([{section_title(layer): nav_for_dir(layer)} for layer in layers])
    print(f"site: {len(layers)} layers, {s['pages']} pages, {s['notebooks']} notebooks "
          f"({s['colab']} with Colab buttons), {s['images']} images; links: {s['to_page']} to site pages, "
          f"{s['to_github']} to GitHub, {s['missing']} missing targets; anchors: Markdown {s['anchors_kept']} kept / "
          f"{s['anchors_dropped']} dropped, notebooks {s['nb_anchors_kept']} kept / {s['nb_anchors_dropped']} "
          f"dropped; {s['math']} inline formulas; nav {n_leaf} pages in {n_sec} sections, depth {deepest}; "
          f"{s['skipped']} files skipped; mkdocs.yml {yml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
