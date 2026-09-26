"""PRIMER §4.2's chat + RAG step, recomputed with fleetsim (notebook 03's `mixed_step`)."""
import pathlib
import re

import pytest

from fleetsim import HPA, L4_8B, Autoscaler, Fleet, PowerOfTwo, burst, chat, mix, rag

PRIMER_PATH = pathlib.Path(__file__).resolve().parents[2] / "PRIMER.md"


def mixed_step():
    return mix(chat(burst(0.8, 4.0, 60, 600), 800, seed=7, system=1000, user=300, output=150),
               rag(burst(0.15, 0.75, 60, 600), 800, seed=8, docs=4, doc=1400, corpus=5000, zipf=0.5, output=200))


def test_rag_share_of_the_mixed_step():
    res = Fleet(L4_8B, 1, PowerOfTwo(seed=1), autoscaler=Autoscaler(HPA(1, 8), "inflight", 40, kind="external"),
                cold_start_s=30).run(mixed_step(), horizon=900)
    reqs = res.requests
    rag_reqs = [q for q in reqs if q.kind == "rag"]
    share = len(rag_reqs) / len(reqs)
    assert share == pytest.approx(0.147, abs=0.005)                               # "~15 % of requests RAG"
    assert sum(q.prompt for q in rag_reqs) / len(rag_reqs) == pytest.approx(6000, rel=0.05)
    uncached = {k: sum(q.prompt - q.cached for q in reqs if q.kind == k) for k in ("chat", "rag")}
    assert uncached["rag"] / sum(uncached.values()) == pytest.approx(0.76, abs=0.02)   # "three quarters"
    assert round(res.summary(ttft_slo=2.0, tpot_slo=0.15)["slo_attainment"], 2) == 0.69
    if PRIMER_PATH.exists():
        text = re.sub(r"\s+", " ", PRIMER_PATH.read_text(encoding="utf-8"))
        assert "~15 % of requests RAG with ~6,000-token prompts, three quarters of the prefill" in text
        assert "reached 0.69 SLO attainment (TTFT ≤ 2 s)" in text
