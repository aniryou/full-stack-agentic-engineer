"""Every tracked SVG figure is well-formed XML.

GitHub and the site load a figure as an image, so the browser parses the file as strict XML: a duplicate attribute, an
unclosed tag or an unescaped `&` shows a broken image where the figure should be. `render_figures.js` loads the SVG
the same strict way, but this test runs in CI (the lab-list-check job runs tools/ci/tests) over every SVG the
repository tracks, the hand-written figures of the primers (tools/orchestration/FIGURE-STYLE.md) and the site's assets
included.
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def tracked_svgs() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "*.svg"], cwd=REPO, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p)


SVGS = tracked_svgs()


def test_some_svgs_are_tracked():
    assert len(SVGS) >= 100, SVGS[:5]


@pytest.mark.parametrize("path", SVGS)
def test_svg_is_well_formed_xml_with_a_title(path):
    root = ET.parse(REPO / path).getroot()          # raises on a duplicate attribute, an unclosed tag, a bare &
    assert root.tag == "{http://www.w3.org/2000/svg}svg", path
    assert root.get("viewBox"), f"{path}: no viewBox"
    assert root.find("{http://www.w3.org/2000/svg}title") is not None, f"{path}: no <title> (the alt text)"
    text = (REPO / path).read_text(encoding="utf-8")
    assert "<script" not in text and "xlink:href=\"http" not in text and "url(http" not in text, f"{path}: external resource"
