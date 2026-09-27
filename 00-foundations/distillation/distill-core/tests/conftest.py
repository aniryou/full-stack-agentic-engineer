"""Shared fixtures: the toy teacher, the exposure-bias bench, the RL-trained thinking teacher, and a loader for
the repo modules this topic reproduces (read by path, never installed, no cache left behind)."""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from distillcore import ModLang, ThinkToy, TinyLM, onpolicy as op, reasoning as R, seqkd
from distillcore.tinylm import fit_language

REPO = Path(__file__).resolve().parents[4]


def repo_module(pkg_rel: str, module: str):
    """Import <pkg>.<module> from another topic of this repo under a private alias, without running the
    package's __init__ (only the module and its relative imports). Skips when the file is not in the checkout."""
    pkg = REPO / pkg_rel
    if not (pkg / f"{module}.py").is_file():
        pytest.skip(f"{pkg_rel}/{module}.py not in this checkout")
    alias = "_repo_" + pkg.name
    if alias not in sys.modules:
        m = types.ModuleType(alias)
        m.__path__ = [str(pkg)]
        sys.modules[alias] = m
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(f"{alias}.{module}")
    finally:
        sys.dont_write_bytecode = saved


@pytest.fixture(scope="session")
def lang():
    return ModLang(11, 0.2)


@pytest.fixture(scope="session")
def teacher(lang):
    return fit_language(lang, 64)


@pytest.fixture(scope="session")
def bench(lang, teacher):
    """§3–§4's bench: 20 prompts on two cycles, the greedy teacher's 12-token continuations, and a student
    (H = 16) given supervised KD on those teacher prefixes."""
    o = lang.orbits()
    prompts = np.array(o[0] + o[1])
    greedy = seqkd.teacher_data(teacher, prompts, 1, 12, np.random.default_rng(1), T=0.0)
    kd = TinyLM(11, 16, 8, seed=1)
    op.gkd_train(kd, teacher, prompts, 12, 300, lam=0.0, beta=0.0, data=greedy)
    return {"prompts": prompts, "greedy": greedy, "kd": kd}


@pytest.fixture(scope="session")
def think():
    task = ThinkToy(0.8, 0.1, 32)
    return task, R.reinforce(task, np.random.default_rng(0), 16000)
