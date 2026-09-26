"""tinymoe — a tiny MoE transformer trained on a toy task (torch on CPU, T0).

The numpy parts (the toy task, load and specialisation statistics, the bundled curves) import
eagerly; the torch parts (``model``, ``train``) import on first use, so ``import moelab.tinymoe``
works on a machine without torch and the notebooks fall back to the bundled curves there.
"""
from __future__ import annotations

import importlib

from .curves import load_bundled, load_stats, specialisation, table  # noqa: F401
from .data import ToyTask  # noqa: F401

_LAZY = {"model", "train"}


def __getattr__(name):
    if name in _LAZY:
        from .. import env
        env.torch()                                   # a clear error when torch is absent
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
