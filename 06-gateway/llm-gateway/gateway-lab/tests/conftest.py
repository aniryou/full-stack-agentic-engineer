"""Shared fixtures: a fast in-process stack (fakes with short, fixed timings) and a path loader for other labs."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # the path-import tests load other labs' code: leave no caches in their directories

import ast
import importlib.util
from pathlib import Path

import pytest

from gwlab.fakes import FakeSpec
from gwlab.stack import LocalStack

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]

FAST = dict(ttft_s=0.01, prefill_s_per_token=0.0, itl_s=0.002, output_tokens=12)


def fast_fakes(**kw) -> dict:
    return {"acme": FakeSpec(name="acme", dialect="openai", **{**FAST, **kw.get("acme", {})}),
            "bolt": FakeSpec.anthropic("bolt", **{**FAST, **kw.get("bolt", {})})}


FAST_OVERRIDES = {"providers": {"acme": {"first_byte_timeout_s": 0.5}, "bolt": {"first_byte_timeout_s": 0.5}},
                  "breaker": {"failure_threshold": 3, "recovery_timeout_s": 0.5}}


def make_stack(overrides=None, fakes=None, **kw) -> LocalStack:
    ov = {**FAST_OVERRIDES, **(overrides or {})}
    return LocalStack(fakes=fakes or fast_fakes(), overrides=ov, **kw)


@pytest.fixture(scope="module")
def stack():
    with make_stack() as s:
        yield s


@pytest.fixture
def fresh():
    with make_stack() as s:
        yield s


def load_path(rel: str, name: str, root: str | None = None):
    """Import a module from another lab by path (skip when that lab is not in this checkout); no bytecode left behind."""
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if root:
            sys.path.insert(0, str(REPO / root))
            try:
                return importlib.import_module(name)
            finally:
                sys.path.remove(str(REPO / root))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = saved


def source_function(rel: str, func: str, glb: dict):
    """Exec one top-level function's source from another lab's file (for modules whose imports need extras)."""
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    ns = dict(glb)
    exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns[func]


def source_constants(rel: str, prefix: str) -> dict:
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith(prefix):
                    out[t.id] = n.value.value
    return out
