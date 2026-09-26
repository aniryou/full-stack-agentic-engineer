"""The Prometheus writer/parser round trip, PromQL's histogram_quantile, and labelled reports."""
import json
import math

from thinklab import metrics as M
from thinklab import report as R


def test_round_trip_and_quantile():
    reg = M.Registry({"model_name": "m", "engine": "0"})
    for v in (0.005, 0.02, 0.02, 0.2):
        reg.observe(M.ITL, "itl", v, M.BUCKETS["itl"])
    reg.counter(M.GENERATION_TOKENS, "gen", 5)
    reg.gauge(M.KV_USAGE, "kv", 0.25)
    s = M.parse(reg.render())
    assert M.value(s, M.GENERATION_TOKENS) == 5 and M.value(s, M.KV_USAGE) == 0.25
    b, total, n = M.histogram(s, M.ITL)
    assert n == 4 and abs(total - 0.245) < 1e-12 and b[-1] == (math.inf, 4.0)
    # rank 0.5 x 4 = 2 falls in (0.01, 0.025] holding counts 1..3: 0.01 + 0.015 x (2 - 1) / (3 - 1)
    assert abs(M.histogram_quantile(0.5, b) - 0.0175) < 1e-12
    assert M.token_buckets(8192) == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def test_delta_subtracts_counters_only():
    a = [M.Sample("x_total", {}, 5.0), M.Sample("g", {}, 3.0)]
    b = [M.Sample("x_total", {}, 7.0), M.Sample("g", {}, 1.0)]
    d = {s.name: s.value for s in M.delta(b, a)}
    assert d == {"x_total": 2.0, "g": 1.0}


def test_report_labels(tmp_path):
    j, md = R.write({"rows": [{"a": 1, "b": 2.5}]}, "SIMULATED", "demo", str(tmp_path))
    assert json.loads(j.read_text())["label"] == "SIMULATED" and "**SIMULATED**" in md.read_text()
    assert "a" in R.table([{"a": 1}]) and R.sparkline([1, 2, 3]) and "#" in R.histogram([1, 10, 100], log=True)
