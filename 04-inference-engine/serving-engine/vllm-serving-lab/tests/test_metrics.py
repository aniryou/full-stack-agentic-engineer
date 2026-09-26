"""/metrics parsing: names, windows between scrapes, and PromQL-exact histogram quantiles."""
import math
from pathlib import Path

import pytest

from servelab import metrics as M

FIX = Path(__file__).parent / "fixtures"


def load(name, t):
    return M.parse((FIX / name).read_text(), timestamp=t)


def test_parse_gauges_counters_and_labels():
    s = load("vllm_metrics_t1.txt", 10.0)
    assert s.value(M.RUNNING) == 12 and s.value(M.WAITING) == 3 and s.value(M.KV_USAGE) == 0.42
    assert s.value(M.PREFIX_QUERIES) == 30_000            # found via the _total suffix
    assert s.value(M.REQUEST_SUCCESS) == 100              # summed over finished_reason
    assert s.value(M.REQUEST_SUCCESS, finished_reason="stop") == 10
    info = s.find(M.CACHE_CONFIG_INFO)[0]
    assert info.labels["note"] == 'quote " and backslash \\' and info.labels["num_gpu_blocks"] == "64662"
    assert not any(x.name.endswith("_created") for x in s.samples)
    assert s.family_type("vllm:prompt_tokens_total") == "counter"
    assert s.family_type(M.TTFT + "_bucket") == "histogram"


def test_histogram_quantile_by_hand():
    h = load("vllm_metrics_t1.txt", 10.0).histogram(M.TTFT)
    assert h.count == 100 and h.mean == pytest.approx(0.081)
    # p50: rank 50 falls in (0.06, 0.08] holding 30..60 -> 0.06 + 0.02*(50-30)/(60-30)
    assert h.quantile(0.5) == pytest.approx(0.06 + 0.02 * 20 / 30)
    # p99: rank 99 falls in (0.25, 0.5] holding 96..100 -> 0.25 + 0.25*(99-96)/4
    assert h.quantile(0.99) == pytest.approx(0.4375)


def test_histogram_quantile_edge_cases():
    q = M.histogram_quantile
    inf = math.inf
    assert q(0.5, [(1.0, 0), (2.0, 10), (inf, 10)]) == pytest.approx(1.5)
    assert q(0.5, [(1.0, 10), (inf, 10)]) == pytest.approx(0.5)        # first bucket starts at 0
    assert q(0.99, [(1.0, 0), (2.0, 1), (inf, 100)]) == 2.0            # in +Inf: largest finite bound
    assert math.isnan(q(0.5, [(1.0, 0), (inf, 0)]))                    # no observations
    assert math.isnan(q(0.5, [(1.0, 3), (2.0, 5)]))                    # no +Inf bucket
    assert q(-0.1, [(1.0, 1), (inf, 1)]) == -inf and q(1.1, [(1.0, 1), (inf, 1)]) == inf
    assert q(0.5, [(2.0, 4), (1.0, 2), (inf, 4)]) == pytest.approx(1.0)  # unsorted input is fine
    assert q(0.5, [(1.0, 4), (2.0, 3), (inf, 4)]) == pytest.approx(0.5)  # non-monotonic counts are repaired


def test_window_between_two_scrapes():
    t0, t1 = load("vllm_metrics_t0.txt", 0.0), load("vllm_metrics_t1.txt", 10.0)
    d = M.delta(t1, t0)
    h = d.histogram(M.TTFT)
    assert h.count == 60
    assert h.quantile(0.5) == pytest.approx(0.08)                          # rank 30 = top of (0.06, 0.08]
    assert h.quantile(0.9) == pytest.approx(0.1 + 0.15 * (54 - 44) / 12)  # (0.1, 0.25] holds 44..56
    snap = M.snapshot(t1, t0)
    assert snap.prefix_hit_rate == pytest.approx(12_000 / 20_000)        # window, not since start
    assert M.snapshot(t1).prefix_hit_rate == pytest.approx(14_000 / 30_000)
    assert snap.preemptions == 5 and snap.requests_finished == 60
    assert snap.generation_tps == pytest.approx(30_000 / 10.0)
    assert snap.running == 12 and snap.kv_cache_usage == 0.42              # gauges: the later value


def test_counter_reset_uses_later_value():
    a = M.parse("# TYPE x_total counter\nx_total 100\n", timestamp=0)
    b = M.parse("# TYPE x_total counter\nx_total 7\n", timestamp=1)
    assert M.delta(b, a).value("x") == 7


def test_older_metric_names_still_read():
    old = M.parse('vllm:gpu_cache_usage_perc{model_name="m"} 0.3\n')
    assert M.snapshot(old).kv_cache_usage == 0.3


def test_agrees_with_prometheus_client_parser():
    from prometheus_client.parser import text_string_to_metric_families
    text = (FIX / "vllm_metrics_t1.txt").read_text()
    theirs = {(s.name, tuple(sorted(s.labels.items()))): s.value
              for fam in text_string_to_metric_families(text) for s in fam.samples if not s.name.endswith("_created")}
    ours = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in M.parse(text).samples}
    assert ours == theirs
