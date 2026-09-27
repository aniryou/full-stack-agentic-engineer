"""memcore restates formulas that already have a home in this repo; here it reproduces their numbers.

Each test pins the number by hand first, then - where the other lab is present in the checkout - imports the
original by path and checks that the two agree. The core never imports another lab at run time.
"""
import ast
import importlib
import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from memcore import (GPUS, LLMS, HashingEmbedder, PrefixCache, call_cost, compute_ttft, count_tokens,
                     expected_cached_tokens, prefill_seconds, wilson_interval)
from memcore.agent import AuditEvent, args_digest

REPO = next((p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file()), None)


def lab(rel: str) -> Path | None:
    p = REPO / rel if REPO else None
    return p if p is not None and p.exists() else None


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


def import_from(root: Path, module: str):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))


def function_from_source(path: Path, name: str, **globals_):
    """Compile one function out of a file whose package cannot be imported here (e.g. it needs pydantic)."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(globals_)
    exec(compile(ast.Module([fn], []), str(path), "exec"), ns)
    return ns[name]


# --- 07.4 rag-from-scratch: ragkit.embed.HashingEmbedder -------------------------------------------------
def test_hashing_embedder_matches_ragkit():
    ours = HashingEmbedder()
    texts = ["user lives in lisbon", "the user moved to porto", "Where does the user live?", "", "Ümlaut café 42"]
    assert round(float(ours.encode(texts[0]) @ ours.encode(texts[1])), 4) == 0.2236
    root = lab("07-application-agent-framework/retrieval-rag/rag-from-scratch")
    if root is None:
        pytest.skip("rag-from-scratch not in this checkout")
    theirs = import_from(root, "ragkit.embed").HashingEmbedder()
    np.testing.assert_array_equal(ours.encode(texts), theirs.encode(texts))


# --- 04 vllm-serving-lab notebook 04, exercise 4.1: expected_cached_tokens --------------------------------
def test_expected_cached_tokens_notebook_asserts():
    assert expected_cached_tokens(list(range(64)), list(range(64)), 16) == 48
    assert expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30, 16) == 32
    assert expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8) == 16
    src = lab("04-inference-engine/serving-engine/vllm-serving-lab/notebooks_src/04_prefix_caching_for_agents.py")
    if src is None:
        pytest.skip("vllm-serving-lab not in this checkout")
    fn = function_from_source(src, "expected_cached_tokens")
    rng = random.Random(1)
    for _ in range(200):
        a = [rng.randrange(3) for _ in range(rng.randrange(1, 60))]
        b = a[:rng.randrange(0, len(a) + 1)] + [rng.randrange(3) for _ in range(rng.randrange(1, 20))]
        assert fn(a, b, 8) == expected_cached_tokens(a, b, 8)


# --- 04 mini-engine-core: minengine.kv lookup rules and minengine.perf.step_cost -------------------------
def test_prefix_cache_hits_match_minengine_kv():
    root = lab("04-inference-engine/serving-engine/mini-engine-core")
    if root is None:
        pytest.skip("mini-engine-core not in this checkout")
    kv = import_from(root, "minengine.kv")
    rng = random.Random(2)
    for _ in range(100):
        B = rng.choice([4, 16])
        done = [rng.randrange(4) for _ in range(rng.randrange(1, 70))]
        new = done[:rng.randrange(0, len(done) + 1)] + [rng.randrange(4) for _ in range(rng.randrange(1, 30))]
        mgr = kv.KVCacheManager(64, block_size=B)
        mgr.allocate_slots("r0", done, 0, len(done))
        mgr.cache_blocks("r0", done, len(done))             # every full block of `done` computed and published
        ours = PrefixCache(B)
        ours.serve(done + [0])                               # publish the same full blocks (+1: the sampled token)
        assert ours.lookup(new) == len(mgr.lookup(new)) * B


def test_prefill_seconds_matches_minengine_perf():
    pins = {("L4", "qwen2.5-1.5b"): (78.7, 15.1), ("L4", "llama-3.1-8b"): (401.0, 65.6),
            ("H100-SXM", "qwen2.5-1.5b"): (11.4, 3.2), ("H100-SXM", "llama-3.1-8b"): (50.8, 7.7)}
    for (g, m), (cold, warm) in pins.items():
        got = (prefill_seconds(GPUS[g], LLMS[m], 2000, 0), prefill_seconds(GPUS[g], LLMS[m], 2000, 1800))
        assert tuple(round(x * 1e3, 1) for x in got) == (cold, warm)
    root = lab("04-inference-engine/serving-engine/mini-engine-core")
    if root is None:
        pytest.skip("mini-engine-core not in this checkout")
    perf = import_from(root, "minengine.perf")
    for (g, m) in pins:
        for cached, n in [(0, 2000), (1800, 200), (5000, 700), (0, 16)]:
            theirs = perf.step_cost(perf.GPUS[g], perf.LLMS[m], [(cached, n)])["t"]
            assert math.isclose(prefill_seconds(GPUS[g], LLMS[m], cached + n, cached), theirs, rel_tol=1e-12)


# --- 00 capacity planning: capacity.ttft_s ---------------------------------------------------------------
def test_compute_ttft_matches_capacity_ttft():
    assert round(compute_ttft(24, 2000, 1979), 4) == 0.0970
    path = lab("00-foundations/gpu-capacity-planning/capacity.py")
    if path is None:
        pytest.skip("capacity.py not in this checkout")
    cap = load(path, "capacity_for_memcore")
    assert math.isclose(cap.ttft_s(24, 2000, cap.GPUS["H100"]), compute_ttft(24, 2000, cap.GPUS["H100"].fp8_tflops))


# --- 06 scaling lab: scalelab.capacity.cost_per_call; 07.2: agentlab.estimation.calc.token_cost ------------
def test_call_cost_matches_the_scaling_primer_and_the_platform_lab():
    assert round(call_cost(5000, 350, 2700), 6) == 0.007005                 # scaling primer §3.4: ~$0.0070
    assert round(call_cost(5000, 350, 2700, "gemini-3-flash"), 6) == 0.002335
    path = lab("06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/capacity.py")
    if path is not None:
        sc = load(path, "scalelab_capacity_for_memcore")
        assert math.isclose(sc.cost_per_call("gemini-3.5-flash", 5000, 350, 2700), call_cost(5000, 350, 2700))
    path = lab("07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/estimation/calc.py")
    if path is not None:
        calc = load(path, "agentlab_calc_for_memcore")
        assert math.isclose(calc.token_cost(5000, 350, calc.PRICES["gemini-3-flash"], cached_share=0.54),
                            call_cost(5000, 350, 2700, "gemini-3-flash"))


# --- 07.2: agentlab.evals.gate.wilson_interval and agentlab.llm.types.count_tokens --------------------------
def test_wilson_and_count_tokens_match_the_platform_lab():
    assert tuple(round(x, 4) for x in wilson_interval(45, 50)) == (0.7864, 0.9565)
    base = lab("07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab")
    if base is None:
        pytest.skip("gcp-agent-platform-lab not in this checkout")
    theirs = function_from_source(base / "evals" / "gate.py", "wilson_interval", math=math)   # gate.py needs pydantic
    for passes, n in [(45, 50), (0, 20), (20, 20), (7, 13), (0, 0)]:
        assert theirs(passes, n) == wilson_interval(passes, n)
    types = load(base / "llm" / "types.py", "agentlab_types_for_memcore")
    for text in ["", "a", "abcd", "x" * 41, "The user's home city is Lisbon."]:
        assert types.count_tokens(text) == count_tokens(text)


# --- 06 identity lab: the audit event's field names and args_digest --------------------------------------
def test_audit_event_uses_the_identity_labs_field_names():
    path = lab("06-gateway/identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py")
    if path is None:
        pytest.skip("identity lab not in this checkout")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuditEvent")
    theirs = {s.target.id for s in cls.body if isinstance(s, ast.AnnAssign)}
    ours = set(AuditEvent.__dataclass_fields__)
    assert ours <= theirs, ours - theirs                                     # a subset, same names
    digest = function_from_source(path, "args_digest", json=__import__("json"), hashlib=__import__("hashlib"),
                                  Any=object)
    assert digest({"b": 1, "a": [2, 3]}) == args_digest({"b": 1, "a": [2, 3]})
