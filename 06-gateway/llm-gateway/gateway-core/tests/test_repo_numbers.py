"""Formulas that already have a home in the repo are reproduced here, number for number (SPEC §6b).

Cores never import other labs: each check loads the other module from its path in this checkout and skips when
the path is absent. 07.2's ``agentlab`` needs pydantic to import, so its two checks read the function and the
constants out of the source with ``ast`` instead of importing the package.
"""
import ast
import asyncio
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from gwcore import cache, metering, otel, ratelimit
from gwcore.mcp_authz import s256

REPO = Path(__file__).resolve().parents[4]
SCALING = REPO / "06-gateway/scaling-admission-cost/agentic-scaling-lab"
ROOFLINE = REPO / "01-hardware-gpu-fabric/roofline-and-fabric/roofline-core"
AGENTLAB = REPO / "07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab"
RAGKIT = REPO / "07-application-agent-framework/retrieval-rag/rag-from-scratch"


def _import(root: Path, name: str):
    if not root.exists():
        pytest.skip(f"{root.relative_to(REPO)} not in this checkout")
    saved_path, saved_bc = list(sys.path), sys.dont_write_bytecode
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True                         # leave no cache in the other topic's directory
    try:
        return importlib.import_module(name)
    finally:
        sys.path[:], sys.dont_write_bytecode = saved_path, saved_bc


def test_cost_per_call_equals_scalelab_on_every_price_row():
    cap = _import(SCALING, "scalelab.capacity")
    assert cap.cost_per_call("gemini-3.5-flash", 5000, 350, 2700) == pytest.approx(0.007005)
    for model in cap.PRICES:
        for shape in [(5000, 350, 2700), (5000, 350, 0), (5000, 3500, 2700), (120_000, 900, 100_000)]:
            assert metering.price_call(model, *shape) == pytest.approx(cap.cost_per_call(model, *shape), rel=1e-12)


def test_token_bucket_waits_like_scalelabs_on_the_same_arrivals():
    res = _import(SCALING, "scalelab.resilience")

    class FakeClock:
        t = 0.0
        def now(self):
            return self.t
        async def sleep(self, s):
            self.t += s

    fake, saved = FakeClock(), res.CLOCK
    res.CLOCK = fake
    try:
        theirs, ours = res.TokenBucket(rate=100, capacity=600), ratelimit.TokenBucket(rate=100, capacity=600)
        rng = np.random.default_rng(3)
        for gap, tokens in zip(rng.exponential(2.0, 200), rng.integers(50, 900, 200)):   # some exceed the burst
            fake.t += gap
            now = fake.t
            w_theirs = asyncio.run(theirs.acquire(float(tokens)))
            assert ours.acquire(float(tokens), now) == pytest.approx(w_theirs)
            assert ours.tokens == pytest.approx(theirs.tokens)
    finally:
        res.CLOCK = saved


def test_self_hosted_rows_reproduce_roofline_section_8_1():
    llm, specs, cost = (_import(ROOFLINE, f"roofline.{m}") for m in ("llm", "specs", "cost"))
    m8, h100, l4 = llm.PRESETS["llama-3.1-8b"], specs.DEVICES["h100-sxm"], specs.DEVICES["l4"]
    batch = llm.best_batch_under_itl(m8, h100, 2048, 0.010)
    h_tps, l_tps = llm.decode(m8, h100, batch, 2048).tokens_per_s, llm.decode(m8, l4, llm.max_batch_by_memory(m8, l4, 2048), 2048).tokens_per_s
    for price, tps, want in [(3.7, h_tps, "0.150"), (11.0, h_tps, "0.446"), (0.70, l_tps, "0.660")]:
        ours = metering.self_hosted_per_million(price, tps)
        assert ours == pytest.approx(cost.cost_per_million_tokens(price, tps)) and f"{ours:.3f}" == want
    assert metering.chargeback.__kwdefaults__["decode_tps"] == round(h_tps, 1)     # the chargeback default is this row


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    return path.read_text()


def test_pkce_matches_agentlab_challenge_for():
    tree = ast.parse(_source(AGENTLAB / "auth/oauth.py"))
    ns = {"base64": __import__("base64"), "hashlib": hashlib}
    for fn in ("b64url_encode", "challenge_for"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(compile(ast.Module([node], []), "oauth.py", "exec"), ns)
    for verifier in ["dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk", "x" * 43, "a-b_c.d~e" * 5]:
        assert s256(verifier) == ns["challenge_for"](verifier)


def test_genai_names_match_the_07_2_lab():
    tree = ast.parse(_source(AGENTLAB / "observability/tracing.py"))
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              for t in n.targets if isinstance(t, ast.Name) and t.id.startswith(("GEN_AI_", "TOOL_"))}
    assert consts["GEN_AI_OPERATION_NAME"] == otel.OPERATION_NAME and consts["GEN_AI_REQUEST_MODEL"] == otel.REQUEST_MODEL
    assert consts["GEN_AI_INPUT_TOKENS"] == otel.INPUT_TOKENS and consts["GEN_AI_OUTPUT_TOKENS"] == otel.OUTPUT_TOKENS
    assert consts["GEN_AI_CACHED_TOKENS"] == otel.CACHE_READ and consts["GEN_AI_FINISH_REASONS"] == otel.FINISH_REASONS
    assert consts["TOOL_NAME"] == otel.TOOL_NAME and consts["TOOL_CALL_ID"] == otel.TOOL_CALL_ID


def test_hashing_embedder_matches_ragkit():
    embed = _import(RAGKIT, "ragkit.embed")
    theirs, ours = embed.HashingEmbedder(dim=1024), cache.HashingEmbedder(dim=1024)
    for text in ["How do I reset my password?", "Q3 2025 revenue: $1,234.50 (EUR)", "", "naïve café 42"]:
        np.testing.assert_array_equal(ours.encode(text), theirs.encode(text))
