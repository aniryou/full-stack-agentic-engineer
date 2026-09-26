#!/usr/bin/env python3
"""Assemble the guide site's generated pages from the repo (standard library only, idempotent).

Writes, under the MkDocs docs_dir `site/`:
  layers/<layer>/...            every layer's README (as index.md), every Markdown doc, every notebook, and
                                the images those docs reference, with relative links rewritten for the site
  layers/index.md               the stack diagram + one row per layer
  guide/curriculum.md, guide/compute.md, guide/colab.md   from CURRICULUM.md, COMPUTE.md, COLAB.md
and rewrites the `nav:` entries between `# nav-layers:start` / `# nav-layers:end` in mkdocs.yml.

Everything written here is gitignored (site/.gitignore); run it before `mkdocs build` or `mkdocs serve`:
    python3 tools/site/build_site_content.py
Link rules: a relative link to something that has a page on the site points at that page (anchors are
re-slugified the way MkDocs' toc does, or dropped if the heading cannot be found); anything else in the repo
(code, Terraform, YAML, folders without a README, LICENSE, ...) points at GitHub; absolute URLs are untouched.
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
                 "worked": "Worked examples", "lessons": "Lessons"}

stats = {"pages": 0, "notebooks": 0, "colab": 0, "images": 0, "to_page": 0, "to_github": 0,
         "anchors_kept": 0, "anchors_dropped": 0, "missing": 0, "skipped": 0}
missing_examples: list[str] = []


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
    parts = repo_path.split("/")
    return "solutions" in parts[:-1] or bool(re.search(r"solution|solved", parts[-1].lower()))


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


def rewrite_target(target: str, src_repo: str, src_site: str, html: bool) -> str:
    raw = target.strip()
    if not raw or SCHEME.match(raw):
        return target
    path, _, frag = raw.partition("#")
    path = path.split("?", 1)[0]
    if not path:                                    # same-page anchor
        if src_repo.endswith(".md") and frag:
            new = anchors_of(src_repo).get(unquote(frag)) or anchors_of(src_repo).get(unquote(frag).lower())
            if new:
                stats["anchors_kept"] += 1
                return "#" + new
            stats["anchors_dropped"] += 1
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
    if rp in pages:
        stats["to_page"] += 1
        out = relative(pages[rp], src_site, html)
        if frag:
            new = anchors_of(rp).get(unquote(frag)) or anchors_of(rp).get(unquote(frag).lower())
            if new:
                stats["anchors_kept"] += 1
                out += "#" + new
            else:
                stats["anchors_dropped"] += 1
                if os.environ.get("SITE_VERBOSE"):
                    print(f"  dropped anchor: {src_repo} -> {target}", file=sys.stderr)
        return out
    if rp in notebooks:
        stats["to_page"] += 1
        if frag:
            stats["anchors_dropped"] += 1
        return relative(notebooks[rp], src_site, html)
    if rp in images:
        return relative(images[rp], src_site, html)
    if not full.exists():
        stats["missing"] += 1
        missing_examples.append(f"{src_repo} -> {target}")
    stats["to_github"] += 1
    return f"{GITHUB}/blob/{BRANCH}/{rp}" + (f"#{frag}" if frag else "")


LINK = re.compile(r"(!?\[((?:[^\[\]]|\[[^\]]*\])*)\])\(\s*(<[^>]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(\s+\"[^\"]*\")?\s*\)")
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:[ \t]*)(\S+)(.*)$", re.M)
HTMLATTR = re.compile(r"\b(src|href)=\"([^\"]+)\"")
CODESPAN = re.compile(r"(`+)(.+?)\1")


def rewrite_segment(s: str, fn) -> str:
    def link(m):
        label, inner, tgt, title = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        if "](" in inner:   # a linked image: rewrite the inner link too
            label = label[0:label.index("[") + 1] + rewrite_segment(inner, fn) + "]"
        if tgt.startswith("<") and tgt.endswith(">"):
            new = "<" + fn(tgt[1:-1]) + ">"
        else:
            new = fn(tgt)
        return f"{label}({new}{title})"
    s = LINK.sub(link, s)
    return HTMLATTR.sub(lambda m: f'{m.group(1)}="{fn(m.group(2))}"', s)


def rewrite_markdown(text: str, src_repo: str, src_site: str, html: bool = False) -> str:
    """Rewrite link targets in Markdown, outside fenced code and inline code; links may span lines."""
    fn = lambda t: rewrite_target(t, src_repo, src_site, html)  # noqa: E731
    out: list[str] = []
    prose: list[str] = []

    def flush():
        if not prose:
            return
        block = "\n".join(prose)
        prose.clear()
        block = REFDEF.sub(lambda r: r.group(1) + fn(r.group(2)) + r.group(3), block)
        spans: list[str] = []            # mask inline code so links inside it stay as written

        def mask(c):
            spans.append(c.group(0))
            return f"\x00{len(spans) - 1}\x00"
        masked = rewrite_segment(CODESPAN.sub(mask, block), fn)
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
        write(pages[rp], rewrite_markdown(t, rp, pages[rp]))
        stats["pages"] += 1
    for rp, nb in nbs.items():
        site_path = notebooks[rp]
        for c in md_cells(nb):
            c["source"] = rewrite_markdown(cell_text(c), rp, site_path, html=True)
        if is_solution(rp):
            head = (f"*Worked answers.* [View this notebook on GitHub]({GITHUB}/blob/{BRANCH}/{rp}) · "
                    f"try the exercise version first.")
        else:
            head = (f"[![Open In Colab]({BADGE})]({colab(rp)}) &nbsp; "
                    f"[View on GitHub]({GITHUB}/blob/{BRANCH}/{rp})")
            stats["colab"] += 1
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
        desc = promise(readme or "").replace("|", "\\|")
        rows.append(f"| {link} | {desc} | {pg} | {nb} |")
    body = [
        "# The stack, layer by layer", "",
        "Eight layers, bottom-up: the model-level foundations (00), the hardware (01), and every layer of software "
        "between a GPU and an agent (02–07). Each layer page lists its topics; each topic has a primer, "
        "runnable code and notebooks. The suggested order to work through them is in the "
        "[curriculum](../guide/curriculum.md).", "",
        stack_diagram(), "",
        "| Layer | What it covers | Pages | Notebooks |",
        "|---|---|---:|---:|",
        *rows, "",
    ]
    write("layers/index.md", "\n".join(body))


ACRONYMS = {w.lower(): w for w in (
    "LLM LLMs GPU GPUs P2P KV HPA MCP RAG API SIMT NCCL CUDA TP DRA GKE ADK A2A TTFT TPOT HBM vLLM "
    "LoRA MoE GCP DWS OAuth PQ IVF HNSW GraphRAG TF CPU SLO SLOs PD HITL JWT RDMA MIG DCGM").split()}


def humanize(stem: str) -> str:
    m = re.match(r"^(\d+[a-z]?)[_\-. ]+(.*)$", stem)
    num, rest = (m.group(1), m.group(2)) if m else ("", stem)
    rest = rest.replace("_", " ").replace("-", " ").strip()
    rest = " ".join(ACRONYMS.get(w.lower(), w) for w in rest.split())
    rest = rest[:1].upper() + rest[1:]
    return f"{num} · {rest}" if num and rest else (num or rest)


def md_title(rp: str) -> str:
    name = posixpath.basename(rp)
    if name.upper() == "PRIMER.MD":
        return "Primer"
    t = read(REPO / rp) or ""
    for ln in t.splitlines():
        if ln.startswith("# "):
            h = heading_text(ln[2:])
            if len(h) <= 50:
                return h
            short = re.split(r"\s+[—–]\s+|:\s+", h)[0].strip()
            return short if len(short) >= 4 else h
    return humanize(name[:-3])


def dir_rank(name: str) -> tuple[int, str]:
    if name == "docs":
        return (1, name)
    if name in NOTEBOOK_DIRS:
        return (5, name)
    if name == "solutions":
        return (6, name)
    if name == "deploy":
        return (7, name)
    if "core" in name:
        return (2, name)
    if "lab" in name:
        return (3, name)
    return (4, name)


def dir_title(name: str) -> str:
    return {"docs": "Docs", "solutions": "Solutions", "deploy": "Deploy"}.get(name, NOTEBOOK_DIRS.get(name, name))


def nav_for_dir(repo_dir: str) -> list:
    """Nav entries for one directory: README (section index) → primers → other docs → core → lab → notebooks."""
    items: list = []
    index = f"{repo_dir}/README.md"
    if index in pages:
        items.append(pages[index])
    here_md = sorted((rp for rp in pages if posixpath.dirname(rp) == repo_dir and rp != index),
                     key=lambda rp: (0 if "primer" in rp.lower().rsplit("/", 1)[-1] else 1, rp.lower()))
    items += [{md_title(rp): pages[rp]} for rp in here_md]
    here_nb = sorted(rp for rp in notebooks if posixpath.dirname(rp) == repo_dir)
    in_solutions_dir = posixpath.basename(repo_dir) == "solutions"
    blanks = [rp for rp in here_nb if in_solutions_dir or not is_solution(rp)]
    sols = [rp for rp in here_nb if not in_solutions_dir and is_solution(rp)]
    subdirs = sorted({rp[len(repo_dir) + 1:].split("/")[0] for rp in list(pages) + list(notebooks)
                      if rp.startswith(repo_dir + "/") and "/" in rp[len(repo_dir) + 1:]}, key=dir_rank)
    for d in [d for d in subdirs if dir_rank(d)[0] < 5]:
        sub = nav_for_dir(f"{repo_dir}/{d}")
        if sub:
            items.append({dir_title(d): sub})
    items += [{humanize(posixpath.basename(rp)[:-6]): notebooks[rp]} for rp in blanks]
    for d in [d for d in subdirs if dir_rank(d)[0] >= 5]:
        sub = nav_for_dir(f"{repo_dir}/{d}")
        if sub:
            items.append({dir_title(d): sub})
    if sols:
        items.append({"Solutions": [{humanize(posixpath.basename(rp)[:-6]): notebooks[rp]} for rp in sols]})
    return items


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


def update_nav(layers: list[str]) -> bool:
    text = read(MKDOCS_YML)
    if text is None:
        return False
    m = re.search(r"^([ \t]*)# nav-layers:start\n.*?^[ \t]*# nav-layers:end[ \t]*$", text, re.S | re.M)
    if not m:
        print("  mkdocs.yml: nav-layers markers not found; nav left unchanged", file=sys.stderr)
        return False
    pad = m.group(1)
    layer_items: list = ["layers/index.md"]
    for layer in layers:
        sub = nav_for_dir(layer)
        if sub:
            layer_items.append({LAYER_TITLES.get(layer[:2], layer): sub})
    block = [f"{pad}# nav-layers:start", f"{pad}- Layers:"] + yaml_nav(layer_items, len(pad) + 4) + [f"{pad}# nav-layers:end"]
    new = text[: m.start()] + "\n".join(block) + text[m.end():]
    if new != text:
        MKDOCS_YML.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for dst in GUIDE_PAGES.values():
        (SITE / dst).unlink(missing_ok=True)
    layers = inventory()
    build_pages()
    build_layers_index(layers)
    nav_ok = update_nav(layers)
    for ex in missing_examples[:10]:
        print(f"  missing target: {ex}", file=sys.stderr)
    s = stats
    print(f"site: {len(layers)} layers, {s['pages']} pages, {s['notebooks']} notebooks "
          f"({s['colab']} with Colab buttons), {s['images']} images; links: {s['to_page']} to site pages, "
          f"{s['to_github']} to GitHub, anchors {s['anchors_kept']} kept / {s['anchors_dropped']} dropped, "
          f"{s['missing']} missing targets, {s['skipped']} files skipped; nav {'updated' if nav_ok else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
