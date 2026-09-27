"""This lab re-implements, it does not import: these tests pin its numbers to the repo's canonical homes.

Each loads the other lab's code by path (or reads a function's source when its package needs extras such as
pydantic) and skips when that lab is not in this checkout.
"""
import asyncio
import base64
import hashlib

import numpy as np
import pytest

from gwlab.gateway import otel
from gwlab.gateway.adapters import Usage
from gwlab.gateway.cache import HashingEmbedder
from gwlab.gateway.metering import PRICES, cost_per_million_tokens, cost_usd
from gwlab.gateway.ratelimit import Bucket
from gwlab.mcp.client import pkce_challenge

from .conftest import load_path, source_constants, source_function

SCALE = "06-gateway/scaling-admission-cost/agentic-scaling-lab"
PLATFORM = "07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab"


def test_prices_reproduce_scalelab_cost_per_call():
    cap = load_path(f"{SCALE}/scalelab/capacity.py", "scalelab.capacity", root=SCALE)
    assert cap.cost_per_call("gemini-3.5-flash", 5000, 350, 2700) == pytest.approx(0.007005)
    for model in ("gemini-3.5-flash", "gemini-3.5-flash-lite"):
        inp, out, cached, *_ = cap.PRICES[model]
        assert (PRICES[model].input, PRICES[model].output, PRICES[model].cached) == (inp, out, cached)
        for i, o, c in ((5000, 350, 2700), (1200, 40, 0), (80_000, 3500, 64_000)):
            assert cost_usd(Usage(i, o, c), PRICES[model]) == pytest.approx(cap.cost_per_call(model, i, o, c))


def test_self_hosted_rows_reproduce_roofline_cost_and_the_01_primer():
    root = "01-hardware-gpu-fabric/roofline-and-fabric/roofline-core"
    load_path(f"{root}/roofline/cost.py", "roofline", root=root)
    from roofline import cost, llm, specs          # noqa: E402  (on sys.path only inside load_path; now cached)
    m8, h100, l4 = llm.PRESETS["llama-3.1-8b"], specs.DEVICES["h100-sxm"], specs.DEVICES["l4"]
    b16 = llm.best_batch_under_itl(m8, h100, 2048, 0.010)
    h_tps = llm.decode(m8, h100, b16, 2048).tokens_per_s
    l_tps = llm.decode(m8, l4, llm.max_batch_by_memory(m8, l4, 2048), 2048).tokens_per_s
    for price, tps, want in ((3.7, h_tps, 0.150), (11.0, h_tps, 0.446), (0.70, l_tps, 0.660)):
        mine = cost_per_million_tokens(price, tps)
        assert mine == pytest.approx(cost.cost_per_million_tokens(price, tps)) and round(mine, 3) == want


def test_pkce_reproduces_agentlab_challenge_for_and_rfc7636_appendix_b():
    b64url_encode = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()     # noqa: E731
    challenge_for = source_function(f"{PLATFORM}/agentlab/auth/oauth.py", "challenge_for",
                                    {"hashlib": hashlib, "b64url_encode": b64url_encode})
    v = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert pkce_challenge(v) == challenge_for(v) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_span_names_match_the_07_2_lab():
    names = source_constants(f"{PLATFORM}/agentlab/observability/tracing.py", "GEN_AI_")
    mine = {otel.OPERATION, otel.REQUEST_MODEL, otel.INPUT_TOKENS, otel.OUTPUT_TOKENS, otel.CACHE_READ, otel.FINISH_REASONS}
    theirs = {names[k] for k in ("GEN_AI_OPERATION_NAME", "GEN_AI_REQUEST_MODEL", "GEN_AI_INPUT_TOKENS",
                                 "GEN_AI_OUTPUT_TOKENS", "GEN_AI_CACHED_TOKENS", "GEN_AI_FINISH_REASONS")}
    assert mine == theirs
    tools = source_constants(f"{PLATFORM}/agentlab/observability/tracing.py", "TOOL_")
    assert (otel.TOOL_NAME, otel.TOOL_CALL_ID) == (tools["TOOL_NAME"], tools["TOOL_CALL_ID"])


def test_hashing_embedder_matches_ragkit():
    root = "07-application-agent-framework/retrieval-rag/rag-from-scratch"
    load_path(f"{root}/ragkit/embed.py", "ragkit", root=root)
    from ragkit.embed import HashingEmbedder as Ragkit          # noqa: E402
    theirs, mine = Ragkit(1024), HashingEmbedder(1024)
    for text in ("How do I export a report as CSV?", "error E1042 on plan 4", "the 2026-09-01 release notes"):
        assert np.allclose(theirs.encode(text), mine.encode(text))


def test_bucket_follows_scalelab_token_bucket_rule(monkeypatch):
    """Same capacity and rate, same arrivals: our retry_after equals scalelab's wait, and a request larger than
    the burst drives both into the same debt."""
    load_path(f"{SCALE}/scalelab/resilience.py", "scalelab", root=SCALE)
    from scalelab import resilience     # noqa: E402

    class Frozen:                        # scalelab's virtual clock, stopped: time moves only when it sleeps
        t = 0.0

        def now(self):
            return self.t

        async def sleep(self, s):
            self.t += s
    clock = Frozen()
    monkeypatch.setattr(resilience, "CLOCK", clock)
    theirs = resilience.TokenBucket(rate=10.0, capacity=100.0)
    mine = Bucket(100.0, 10.0, clock=clock.now)

    async def go():
        out = []
        for n in (60, 60, 30, 250):
            want = mine.retry_after(n)
            waited = await theirs.acquire(n)
            mine.take(n)
            out.append((want, waited))
        return out
    waits = asyncio.run(go())
    assert [round(w, 9) for w, _ in waits] == [round(w, 9) for _, w in waits] == [0.0, 2.0, 3.0, 10.0]
    assert mine.available() == pytest.approx(theirs.tokens) == -150.0          # 250 > burst: both in debt
