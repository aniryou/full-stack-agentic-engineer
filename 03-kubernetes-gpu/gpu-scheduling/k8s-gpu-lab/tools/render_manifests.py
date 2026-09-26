#!/usr/bin/env python3
"""Regenerate the YAML under deploy/kind/ and deploy/gke/ from k8sgpu/scenarios.py and k8sgpu/gke.py.

    python3 tools/render_manifests.py          # write files that changed
    python3 tools/render_manifests.py --check  # exit 1 if any generated file is stale
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k8sgpu import render  # noqa: E402


def main(argv: list[str]) -> int:
    if "--check" in argv:
        stale = render.stale(ROOT)
        for rel in stale:
            print("stale:", rel)
        print(f"{len(render.rendered_files()) - len(stale)}/{len(render.rendered_files())} generated files up to date")
        return 1 if stale else 0
    for p in render.write(ROOT):
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
