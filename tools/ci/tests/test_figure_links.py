"""Figure files and the links to them (tools/orchestration/FIGURE-STYLE.md §2 and §8).

- A notebook embeds a figure by its absolute URL on the main branch, because Colab renders neither a Mermaid block
  nor a relative image: ``https://raw.githubusercontent.com/aniryou/full-stack-agentic-engineer/main/<repo path>``.
  Every such URL in a tracked Markdown file, notebook or percent-format source names a file that exists in the tree
  (the link check, tools/orchestration/mdlinks.py, skips absolute URLs, so this is the only guard).
- Every tracked SVG under the layers and the root is referenced by at least one document: an unreferenced figure is a
  dead file that the site never copies (``tools/site/build_site_content.py``, ``collect_images``).

Standard library only.
"""
from __future__ import annotations

import pathlib
import posixpath
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW = "https://raw.githubusercontent.com/aniryou/full-stack-agentic-engineer/main/"
RAW_RE = re.compile(re.escape(RAW) + r"([^\s\"')>\]]+)")
SVG_REF = re.compile(r"([A-Za-z0-9_./%-]+\.svg)")
TEXT_SUFFIXES = (".md", ".ipynb", ".py", ".yml", ".yaml", ".html", ".js", ".css", ".json")


def tracked(*patterns: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *patterns], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [p for p in out.split("\n") if p]


def read(rp: str) -> str:
    try:
        return (ROOT / rp).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def test_every_raw_github_figure_url_names_a_tracked_file():
    files = set(tracked())
    bad = []
    for rp in tracked("*.md", "*.ipynb", "*.py"):
        if rp.startswith("tools/ci/tests/"):
            continue
        for m in RAW_RE.finditer(read(rp)):
            target = m.group(1).split("#")[0].split("?")[0]
            if target not in files:
                bad.append(f"{rp}: {RAW}{target}")
    assert not bad, "raw.githubusercontent.com links to files that are not in the tree:\n  " + "\n  ".join(bad)


def test_every_svg_figure_is_referenced_by_a_document():
    svgs = [p for p in tracked("*.svg") if not p.startswith("site/") and not p.startswith("tools/")]
    referenced: set[str] = set()
    for rp in tracked(*[f"*{s}" for s in TEXT_SUFFIXES]):
        text = read(rp)
        if ".svg" not in text:
            continue
        for m in RAW_RE.finditer(text):
            referenced.add(m.group(1).split("#")[0])
        for m in SVG_REF.finditer(text):
            t = m.group(1)
            referenced.add(posixpath.normpath(posixpath.join(posixpath.dirname(rp), t)))
            referenced.add(posixpath.normpath(t))
    orphans = [s for s in svgs if s not in referenced]
    assert not orphans, "SVG figures no document references (delete them, or embed them):\n  " + "\n  ".join(orphans)
