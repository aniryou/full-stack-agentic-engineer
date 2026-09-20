"""Test bootstrap: make the repo root importable so ``import minifaiss`` works.

Adds the repo root (the parent of this ``tests`` directory) to ``sys.path`` so
the tests import the in-tree package without an install step.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
