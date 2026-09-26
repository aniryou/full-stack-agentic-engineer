"""Prometheus text: parse, render, and the PromQL histogram_quantile rule."""
import math

import pytest

from igwlab.promtext import Families, Registry, histogram_quantile, parse


def test_parse_labels_escapes_and_specials():
    s = parse('# HELP x y\nm{a="q\\"z",b="2"} 1e3 1700000000\nn +Inf\nk NaN\n')
    assert s[0].name == "m" and dict(s[0].labels) == {"a": 'q"z', "b": "2"} and s[0].value == 1000.0
    assert math.isinf(s[1].value) and math.isnan(s[2].value)
    with pytest.raises(ValueError):
        parse("not a sample line at all {")


def test_counter_is_exposed_with_total_suffix_and_roundtrips():
    r = Registry()
    c = r.counter("vllm:prompt_tokens", "doc", ["model_name", "engine"])
    c.labels(model_name="m", engine="0").inc(5)
    g = r.gauge("vllm:num_requests_waiting", "doc", ["model_name", "engine"])
    g.labels("m", "0").set(3)
    g.labels("m", "1").set(2)
    f = Families.from_text(r.render())
    assert f.value("vllm:prompt_tokens_total") == 5.0 and not f.has("vllm:prompt_tokens")
    assert f.sum("vllm:num_requests_waiting") == 5.0 and f.max("vllm:num_requests_waiting") == 3.0
    assert f.value("vllm:num_requests_waiting", engine="1") == 2.0


def test_histogram_quantile_matches_promql_hand_values():
    b = [(0.1, 50), (0.5, 90), (math.inf, 100)]
    assert histogram_quantile(0.5, b) == pytest.approx(0.1)        # rank 50 -> top of first bucket
    assert histogram_quantile(0.7, b) == pytest.approx(0.3)        # 0.1 + 0.4 * (20/40)
    assert histogram_quantile(0.9, b) == pytest.approx(0.5)
    assert histogram_quantile(0.95, b) == pytest.approx(0.5)       # lands in +Inf: highest finite bound
    assert math.isnan(histogram_quantile(0.5, [(0.1, 0), (math.inf, 0)]))
    assert math.isnan(histogram_quantile(0.5, [(0.1, 3)]))          # no +Inf bucket


def test_rendered_histogram_buckets_are_cumulative():
    r = Registry()
    h = r.histogram("lat", "doc", [0.01, 0.1, 1.0])
    for v in (0.005, 0.05, 0.5, 5.0):
        h.observe(v)
    f = Families.from_text(r.render())
    assert f.buckets("lat") == [(0.01, 1.0), (0.1, 2.0), (1.0, 3.0), (math.inf, 4.0)]
    assert f.value("lat_count") == 4.0 and f.value("lat_sum") == pytest.approx(5.555)
