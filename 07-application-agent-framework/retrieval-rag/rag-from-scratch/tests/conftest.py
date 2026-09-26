"""Make the lab root importable so `pytest -q` finds `ragkit` without PYTHONPATH."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
