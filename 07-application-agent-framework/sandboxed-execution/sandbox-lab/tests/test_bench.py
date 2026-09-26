"""Pool sizing reproduces the fact sheet's worked numbers (the ones the core's pool.py pins too), and
latency rows always say where they came from."""
import math

from sandboxlab import bench


def test_littles_law_pool():
    p = bench.littles_law_pool(5, 2, 3)          # lambda 5/s, 2 s runs, 3 s cold start
    assert (p["busy"], p["warming"], p["total_ceil"]) == (10, 15, 25)


def test_erlang_c_table():
    want = {11: (0.682, 1.364), 12: (0.449, 0.449), 13: (0.285, 0.190), 14: (0.174, 0.087), 16: (0.057, 0.019)}
    for c, (pw, wq) in want.items():
        assert round(bench.erlang_c(10, c), 3) == pw
        assert round(bench.mean_wait_s(5, 2, c), 3) == wq
    assert bench.erlang_c(10, 10) == 1.0 and math.isinf(bench.mean_wait_s(5, 2, 10))


def test_scaling_primer_rate():
    lam = round(27.1 * 0.2, 2)                   # scaling primer §3.2: 27.1 tool calls/s at peak, 20 % run_code
    assert lam == 5.42
    assert bench.servers_for(lam, 2, max_mean_wait_s=0.1) == 15
    assert bench.littles_law_pool(lam, 2, 3)["total_ceil"] == 28


def test_cost_per_execution_by_hand():
    # 2.5 s held, a $0.067/h node share split across 3 sandboxes: 2.5 * 0.067 / 3600 / 3
    assert math.isclose(bench.cost_per_execution(2.5, 0.067, 3), 2.5 * 0.067 / 3600 / 3)


def test_percentile_is_linear():
    assert bench.percentile([1, 2, 3, 4], 0.5) == 2.5 and bench.percentile([5], 0.95) == 5


def test_measured_and_sample_rows_are_labelled():
    measured = bench.measure(["fork_exec", "python"], n=3)
    assert [m.level for m in measured] == ["fork_exec", "python"]
    assert all(m.label == "measured on this machine" and m.p95_ms >= m.p50_ms > 0 for m in measured)
    label, rows = bench.samples()
    assert label == "sample output in the documented format (illustrative)" and all(r.source for r in rows)
    ladder = bench.ladder(measured)
    assert ladder[:2] == measured and {r.level for r in ladder} >= {"docker:runc", "docker:runsc", "k8s:warm(exec)"}
