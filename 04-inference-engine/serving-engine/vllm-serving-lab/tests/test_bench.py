"""Benchmark definitions (TTFT/ITL/TPOT/E2E/throughput/goodput), arrivals and workloads, pinned by hand."""
import math
import random

import pytest

from servelab.bench import (SLO, Lengths, RequestResult, SSEParser, agent_sessions, arrival_times, littles_law,
                            mixed_requests, percentile, random_requests, summarize)
from servelab.textgen import count_tokens


def result(ttft, gaps, out_tokens, prompt=100, ok=True, cached=None):
    """A request sent at t=0 whose chunks arrived at ttft, ttft+g1, ... (seconds)."""
    return RequestResult(ok=ok, start=0.0, ttft=ttft, itl=list(gaps), latency=ttft + sum(gaps),
                         prompt_tokens=prompt, output_tokens=out_tokens, cached_tokens=cached)


def test_per_request_definitions():
    r = result(0.200, [0.02, 0.03, 0.05], 4)
    assert r.latency == pytest.approx(0.300)                          # E2E = last chunk - send
    assert r.tpot == pytest.approx((0.300 - 0.200) / (4 - 1))        # TPOT excludes the first token
    multi = result(0.2, [0.06], 7)                                     # 2 chunks carrying 7 tokens (speculation)
    assert multi.itl == [0.06] and multi.tpot == pytest.approx(0.06 / 6)  # ITL per chunk, TPOT per token
    assert math.isnan(result(0.1, [], 1).tpot)


def test_summary_throughput_and_goodput_by_hand():
    rs = [result(0.10, [0.02] * 9, 10), result(0.50, [0.02] * 9, 10), result(0.20, [0.08] * 9, 10),
          RequestResult(ok=False, error="HTTP 500")]
    s = summarize(rs, duration_s=2.0, slo=SLO(ttft_ms=300, tpot_ms=50))
    assert (s.completed, s.failed) == (3, 1)
    assert s.request_throughput == pytest.approx(1.5)                 # completed / duration
    assert s.output_throughput == pytest.approx(30 / 2.0)
    assert s.total_token_throughput == pytest.approx((300 + 30) / 2.0)
    assert s.goodput == pytest.approx(1 / 2.0)                         # only request 1 meets both SLOs
    assert s.slo_attainment == pytest.approx(1 / 4)                    # failed requests count as missed
    assert s.ttft.median == pytest.approx(200.0) and s.itl.n == 27
    assert s.itl.p[99] == pytest.approx(80.0)


def test_slo_rule_matches_vllm():
    one_token = result(0.1, [], 1)
    assert SLO(ttft_ms=200, tpot_ms=1).ok(one_token)                   # TPOT counts as 0 for 1-token outputs
    assert SLO(ttft_ms=200).ok(result(0.2, [0.1], 2))                  # "<=" : exactly on target is good
    assert not SLO(e2el_ms=250).ok(result(0.2, [0.1], 2))
    assert not SLO().ok(RequestResult(ok=False))


def test_percentile_is_numpys_default():
    np = pytest.importorskip("numpy")
    rng = random.Random(3)
    xs = [rng.expovariate(1.0) for _ in range(101)]
    for p in (0, 1, 50, 90, 99, 99.9, 100):
        assert percentile(xs, p) == pytest.approx(float(np.percentile(xs, p)))
    assert math.isnan(percentile([], 50))


def test_sse_parser_reassembles_split_chunks():
    p = SSEParser()
    wire = b'data: {"a": 1}\n\n: ping\n\ndata: {"b": 2}\n\ndata: [DONE]\n\n'
    got = []
    for i in range(0, len(wire), 5):                                   # arbitrary network boundaries
        got += p.feed(wire[i:i + 5])
    assert got == ['{"a": 1}', '{"b": 2}', "[DONE]"]


def test_arrival_times_follow_vllm_semantics():
    ts = arrival_times(200, rate=10.0, seed=1)
    assert ts[-1] == pytest.approx(200 / 10.0)                          # rescaled: last send at n/rate
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    even = arrival_times(5, rate=2.0, burstiness=math.inf)
    assert even == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    assert arrival_times(3, rate=math.inf) == [0.0, 0.0, 0.0]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    bursty = arrival_times(200, rate=10.0, burstiness=0.25, seed=1)
    bgaps = [b - a for a, b in zip(bursty, bursty[1:])]
    cv = lambda g: (sum((x - sum(g) / len(g)) ** 2 for x in g) / len(g)) ** 0.5 / (sum(g) / len(g))  # noqa: E731
    assert 0.7 < cv(gaps) < 1.3 and cv(bgaps) > 1.5                    # Poisson CV ~1, gamma(0.25) CV ~2


def test_workloads_have_exact_token_counts_and_sharing():
    reqs = random_requests(20, Lengths.uniform(10, 50), Lengths.fixed(8), prefix_tokens=16, seed=0)
    assert all(count_tokens(r.prompt) == r.prompt_tokens for r in reqs)
    assert len({r.prompt.split()[0:16].__str__() for r in reqs}) == 1   # shared 16-token prefix
    mixed = mixed_requests(8, 2, short_in=32, long_in=500, output=4)
    assert sorted(r.tag for r in mixed).count("long") == 2
    assert Lengths.lognormal(100, 1.0, lo=5, hi=400).sample(random.Random(0)) in range(5, 401)


def test_agent_session_layouts():
    stable = agent_sessions(3, turns=3, layout="stable")
    assert stable[0].system(0) == stable[1].system(0) == stable[0].system(2)   # shared and stable
    shuffled = agent_sessions(3, turns=3, layout="shuffled_tools")
    assert shuffled[0].system(0) != shuffled[1].system(0) and shuffled[0].system(0) == shuffled[0].system(1)
    ts = agent_sessions(2, turns=3, layout="timestamp_first")
    assert ts[0].system(0) != ts[0].system(1)                                 # breaks caching every turn
    assert ts[0].system(0).split("\n", 1)[1] == stable[0].system(0)


def test_littles_law():
    assert littles_law(8.0, 12.0) == 96.0     # 8 req/s that live 12 s each -> ~96 in flight
