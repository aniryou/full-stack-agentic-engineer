"""The workload generator and the summary statistics."""
import math
from collections import Counter

import pytest

from igwlab.bench import (BenchResult, Record, agentic_sessions, compare, engine_hit_rate, percentile,
                          session_messages)


def test_percentile_linear_interpolation():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5 and percentile(xs, 0) == 1 and percentile(xs, 100) == 4
    assert percentile(xs, 90) == pytest.approx(3.7)
    assert math.isnan(percentile([], 50)) and percentile([None, 5], 50) == 5


def test_sessions_are_deterministic_and_share_prefixes():
    a = agentic_sessions(12, turns=3, seed=1)
    b = agentic_sessions(12, turns=3, seed=1)
    assert [s.task for s in a] == [s.task for s in b] and len(a[0].tool_outputs) == 2
    assert Counter(s.agent.name for s in a) == Counter({f"agent-{k}": 3 for k in range(4)})
    hot = agentic_sessions(200, hot_share=0.8, seed=0)
    assert 0.7 < sum(s.agent.name == "agent-0" for s in hot) / 200 < 0.9
    m1 = session_messages(a[0], [])
    m3 = session_messages(a[0], ["r1", "r2"])
    assert m3[: len(m1)] == m1 and [m["role"] for m in m3] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_summary_counts_idle_endpoints_and_hit_rate():
    recs = [Record("s", i, "a0", 0.0, ttft=0.01 * (i + 1), e2e=0.1, status=200, endpoint="a",
                   prompt_tokens=100, cached_tokens=50) for i in range(4)]
    r = BenchResult("x", recs, wall_s=2.0, endpoints=["a", "b"])
    s = r.summary()
    assert s["per_endpoint"] == {"a": 4, "b": 0} and s["imbalance"] == 2.0
    assert s["hit_rate"] == 0.5 and s["ttft_p50_ms"] == pytest.approx(25.0) and s["rps"] == 2.0
    assert "x" in compare([r])


def test_hit_rate_is_unavailable_not_zero_when_the_engine_does_not_report_it():
    recs = [Record("s", i, "a0", 0.0, ttft=0.01, e2e=0.1, status=200, endpoint="a", prompt_tokens=100) for i in range(3)]
    s = BenchResult("vllm-without-details", recs, wall_s=1.0, endpoints=["a"]).summary()
    assert s["hit_rate"] is None                                  # vLLM without --enable-prompt-tokens-details
    assert "n/a" in compare([BenchResult("vllm-without-details", recs, wall_s=1.0, endpoints=["a"])])


def test_engine_hit_rate_from_counter_increases():
    page = lambda h, q: f"vllm:prefix_cache_hits_total{{engine=\"0\"}} {h}\nvllm:prefix_cache_queries_total{{engine=\"0\"}} {q}\n"
    before = {"a": page(100, 1000), "b": page(0, 0)}
    after = {"a": page(700, 2000), "b": page(300, 1000)}
    assert engine_hit_rate(before, after) == pytest.approx((600 + 300) / (1000 + 1000))
    assert engine_hit_rate({"a": ""}, {"a": "# nothing\n"}) is None
