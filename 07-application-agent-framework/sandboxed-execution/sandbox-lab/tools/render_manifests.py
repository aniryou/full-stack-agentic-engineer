#!/usr/bin/env python3
"""Write every generated deploy file (kind, GKE, Docker seccomp) from sandboxlab/k8s/policy.py.

    python3 tools/render_manifests.py          # write
    python3 tools/render_manifests.py --check  # exit 1 if anything is stale or orphaned
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandboxlab.k8s import render  # noqa: E402

if "--check" in sys.argv:
    bad = render.stale()
    print("\n".join(bad) if bad else "✅ generated deploy files are up to date")
    raise SystemExit(1 if bad else 0)
for p in render.write():
    print("wrote", p.relative_to(render.LAB_ROOT))
print("done")
