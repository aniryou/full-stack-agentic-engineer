"""Load the layer's existing single-file implementations from the repo checkout, without copying them.

fa_calculators.py (flash-attention/) and paged_attention_minimal.py (paged-attention/) stay where they
are -- their own tests and notebooks import them there. This package reuses them in place, which needs
a checkout of the repo: install with `pip install -e .` (editable) so this file sits in the tree, or set
KERNCORE_LAYER_DIR to the 04-inference-engine directory.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

_FILES = {
    "fa_calculators": "flash-attention/fa_calculators.py",
    "paged_attention_minimal": "paged-attention/paged_attention_minimal.py",
    "flash_attention_minimal": "flash-attention/flash_attention_minimal.py",
}


def layer_dir() -> Path:
    env = os.environ.get("KERNCORE_LAYER_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def load(name: str):
    """Import one of the layer's reference files by name (cached in sys.modules). Scripts that print
    demos at import (flash_attention_minimal) are imported with their output swallowed."""
    if name in sys.modules:
        return sys.modules[name]
    path = layer_dir() / _FILES[name]
    if not path.is_file():
        raise ImportError(f"{path} not found: kerncore reuses the repo's {_FILES[name]}; install it editable "
                          "from a checkout (pip install -e .) or set KERNCORE_LAYER_DIR")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod
