"""Numbers this lab reproduces from elsewhere in the repo, pinned here and cross-checked by path import where
the other package is present (labs never import each other at run time)."""
import importlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from memlab import cachebench as cb
from memlab import harness as H
from memlab.embedders import HashingEmbedder

LAB = Path(__file__).resolve().parents[1]
REPO = next((p for p in LAB.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file()), None)


def repo_path(rel):
    if REPO is None or not (REPO / rel).exists():
        pytest.skip(f"{rel} not in this checkout")
    return REPO / rel


def import_from(root, name):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(root))


def test_hashing_embedder_equals_ragkit():
    ragkit = import_from(repo_path("07-application-agent-framework/retrieval-rag/rag-from-scratch"), "ragkit.embed")
    theirs, ours = ragkit.HashingEmbedder(dim=1024), HashingEmbedder(1024)
    for text in ("user lives in lisbon", "Where does the user live?", "Home address: 12 Rua das Flores."):
        assert np.array_equal(theirs.encode(text), ours.encode(text))
    assert ragkit.FALLBACK_LABEL == ours.label


def test_rrf_equals_ragkit_reference():
    ref = import_from(repo_path("07-application-agent-framework/retrieval-rag/rag-from-scratch"), "ragkit.reference")
    from memlab.store import rrf
    rankings = [["a", "b", "c"], ["c", "a", "d"]]
    assert [(i, round(s, 12)) for i, s in rrf(rankings)] == [(i, round(s, 12)) for i, s in ref.reciprocal_rank_fusion(rankings, k=60)]


def test_step_cost_equals_minengine_perf():
    perf = import_from(repo_path("04-inference-engine/serving-engine/mini-engine-core"), "minengine.perf")
    for g in ("L4", "H100-SXM", "T4"):
        for m in ("qwen2.5-0.5b", "qwen2.5-1.5b", "llama-3.1-8b"):
            for chunks in ([(0, 2000)], [(1800, 200)], [(4000, 1)]):
                a = perf.step_cost(perf.GPUS[g], perf.LLMS[m], chunks)
                b = cb.step_cost(cb.GPUS[g], cb.LLMS[m], chunks)
                assert math.isclose(a["t"], b["t"], rel_tol=1e-12), (g, m, chunks)


def test_expected_cached_tokens_and_the_fake_cache_equal_minengine_kv():
    kv = import_from(repo_path("04-inference-engine/serving-engine/mini-engine-core"), "minengine.kv")
    from memlab.fakeserver import PrefixCache
    cases = [(list(range(64)), list(range(64))), (list(range(40)), list(range(40)) + [7] * 30),
             ([1] * 20 + [2] * 20, [1] * 20 + [3] * 20), (list(range(200)), list(range(100)) + [5] * 50)]
    for prev, new in cases:
        mgr, fake = kv.KVCacheManager(num_blocks=256, block_size=16), PrefixCache()
        hits = mgr.lookup(prev)
        mgr.allocate_slots("prev", prev, 0, len(prev) - 1, hits=hits)   # the last token is sampled, never fed
        mgr.cache_blocks("prev", prev, len(prev) - 1)
        mgr.free("prev")
        fake.insert(prev)
        want = cb.expected_cached_tokens(prev, new, 16)
        assert len(mgr.lookup(new)) * 16 == fake.lookup(new) == want, (len(prev), len(new))


def test_cost_per_call_equals_scalelab():
    path = repo_path("06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/capacity.py")
    spec = importlib.util.spec_from_file_location("scalelab_capacity", path)
    cap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cap                  # dataclasses look their module up while the class is built
    try:
        spec.loader.exec_module(cap)
    finally:
        sys.modules.pop(spec.name, None)
    assert math.isclose(cap.cost_per_call("gemini-3.5-flash", 5000, 350, 2700), 0.007005)
    inp, out, cached, *_ = cap.PRICES["gemini-3.5-flash"]
    assert (inp, out, cached) == (cb.PRICES["gemini-3.5-flash"].input, cb.PRICES["gemini-3.5-flash"].output,
                                  cb.PRICES["gemini-3.5-flash"].cached_input)
    for args in ((5000, 350, 2700), (1449, 60, 0), (3442, 60, 1088)):
        assert math.isclose(cap.cost_per_call("gemini-3.5-flash", *args),
                            cb.turn_cost(cb.PRICES["gemini-3.5-flash"], args[0], args[1], args[2]))


def test_wilson_interval_equals_agentlab_gate_when_importable():
    root = repo_path("07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab")
    pytest.importorskip("pydantic")
    gate = import_from(root, "agentlab.evals.gate")
    for k, n in ((45, 50), (0, 20), (20, 20), (7, 13)):
        assert all(math.isclose(a, b) for a, b in zip(gate.wilson_interval(k, n), H.wilson_interval(k, n)))


def test_serving_lab_expected_cached_tokens_asserts_are_ours():
    src = repo_path("04-inference-engine/serving-engine/vllm-serving-lab/notebooks_src/04_prefix_caching_for_agents.py").read_text()
    for line in ("expected_cached_tokens(list(range(64)), list(range(64)), 16) == 48",
                 "expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30, 16) == 32",
                 "expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8) == 16"):
        assert line in src
        assert eval(line, {"expected_cached_tokens": cb.expected_cached_tokens})
