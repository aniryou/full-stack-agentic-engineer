"""Every figure fits one screen (tools/orchestration/FIGURE-STYLE.md §3).

A figure is shown in a content column of about COL px on a laptop; a wider figure is scaled down to it. The rule:
shown in that column, a figure is at most MAX_SHOWN_H px tall (what a 768 px laptop screen shows below the page
header, without a scroll) and scaled down by no more than MIN_SCALE (16 px labels stay about 11 px).

- An SVG figure is measured from the width and height attributes of its root element (the house style sets both).
- A Mermaid block is measured by ``node tools/orchestration/render_figures.js check``, which renders every block with
  headless Chromium and writes tools/orchestration/figures-manifest.tsv (the sha1 of each block's source and its
  size). A block whose source is not in the manifest, or whose verdict is not "fits", fails here: re-run the check
  after editing a block (prerequisites in the script's header) and commit the manifest with the document.

Standard library only.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tools" / "orchestration" / "figures-manifest.tsv"
COL, MAX_SHOWN_H, MIN_SCALE = 800, 600, 0.7   # the same three numbers as render_figures.js
MERMAID = re.compile(r"^[ \t]{0,3}```mermaid[^\n]*\n(.*?)^[ \t]{0,3}```[ \t]*$", re.S | re.M)   # a fence on a line of its own


def tracked(suffix: str) -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", "--", f"*{suffix}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [ROOT / p for p in out.split("\n") if p]


def verdict(w: float, h: float) -> tuple[bool, str]:
    scale = min(1.0, COL / w)
    shown = round(h * scale)
    why = []
    if shown > MAX_SHOWN_H:
        why.append(f"too tall: {shown} px when shown, the limit is {MAX_SHOWN_H}")
    if scale < MIN_SCALE:
        why.append(f"too wide: shown at {scale:.2f} of its size, the limit is {MIN_SCALE}")
    return not why, "; ".join(why) or "fits"


def normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.split("\n")).strip() + "\n"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def svg_size(path: pathlib.Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    w, h = root.get("width"), root.get("height")
    if w and h and re.fullmatch(r"[\d.]+(px)?", w) and re.fullmatch(r"[\d.]+(px)?", h):
        return float(w.rstrip("px")), float(h.rstrip("px"))
    vb = (root.get("viewBox") or "").split()
    if len(vb) == 4:
        return float(vb[2]), float(vb[3])
    raise AssertionError(f"{path.relative_to(ROOT)}: no numeric width/height and no viewBox on the root <svg>")


def load_manifest() -> dict[str, tuple[str, str]]:
    """sha1 -> (verdict, where) for the Mermaid rows of the manifest."""
    assert MANIFEST.exists(), f"{MANIFEST.relative_to(ROOT)} is missing: run `node tools/orchestration/render_figures.js check`"
    rows: dict[str, tuple[str, str]] = {}
    for line in MANIFEST.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith("#") or line.startswith("kind\t"):
            continue
        kind, doc, block, line_no, digest, w, h, shown, scale, verdict_text = line.split("\t")
        if kind == "mermaid":
            rows[digest] = (verdict_text, f"{doc} block {block} (line {line_no}), {w}x{h}")
    return rows


def test_svg_figures_fit_one_screen():
    failures = []
    for path in tracked(".svg"):
        w, h = svg_size(path)
        ok, why = verdict(w, h)
        if not ok:
            failures.append(f"{path.relative_to(ROOT)} ({w:g}x{h:g}): {why}")
    assert not failures, "SVG figures that do not fit one screen (FIGURE-STYLE.md §3):\n" + "\n".join(failures)


def test_mermaid_blocks_fit_one_screen():
    manifest = load_manifest()
    failures = []
    for path in tracked(".md"):
        text = path.read_text(encoding="utf-8")
        if not MERMAID.search(text):
            continue
        for i, m in enumerate(MERMAID.finditer(text), 1):
            line = text[: m.start()].count("\n") + 1
            digest = sha1(normalize(m.group(1)))
            where = f"{path.relative_to(ROOT)} block {i} (line {line})"
            if digest not in manifest:
                failures.append(f"{where}: not measured; run `node tools/orchestration/render_figures.js check` and commit the manifest")
            elif manifest[digest][0] != "fits":
                failures.append(f"{where}: {manifest[digest][0]}")
    assert not failures, "Mermaid blocks that do not fit one screen (FIGURE-STYLE.md §3):\n" + "\n".join(failures)


def test_manifest_has_no_stale_rows():
    """Every Mermaid row of the manifest still names a block in the tree (a renamed or removed block is re-measured)."""
    if not MANIFEST.exists():
        return
    present = set()
    for path in tracked(".md"):
        text = path.read_text(encoding="utf-8")
        if MERMAID.search(text):
            present |= {sha1(normalize(m.group(1))) for m in MERMAID.finditer(text)}
    stale = [where for digest, (_, where) in load_manifest().items() if digest not in present]
    assert not stale, "manifest rows without a block in the tree; re-run the check:\n" + "\n".join(stale)
