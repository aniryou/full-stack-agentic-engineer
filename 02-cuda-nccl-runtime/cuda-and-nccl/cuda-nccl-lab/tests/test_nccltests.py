"""Parsing nccl-tests output and re-checking its algbw/busbw (fixtures are labelled illustrative)."""
import pytest

from gpurt import nccltests


def test_modern_format(fixture_text):
    res = nccltests.parse(fixture_text("nccl_all_reduce_8gpu_sample.txt"), op="all_reduce")
    assert res.nranks == 8 and len(res.devices) == 8 and res.header["nGpus"] == 8
    assert len(res.rows) == 28 and res.rows[0].redop == "sum" and res.rows[0].root == -1
    assert res.rows[-1].ip_time_us is not None and res.avg_busbw is not None and res.out_of_bounds == 0
    assert nccltests.recheck(res) == []


def test_legacy_format_without_redop_root_or_groups(fixture_text):
    res = nccltests.parse(fixture_text("nccl_all_gather_4gpu_legacy_sample.txt"), op="all_gather")
    assert res.devices == [] and res.nranks == 4  # falls back to nThread x nGpus
    assert len(res.rows) == 18 and res.rows[0].redop is None and res.rows[0].wrong == 0.0
    assert nccltests.recheck(res) == []


def test_summary_recovers_the_model_behind_the_sample(fixture_text):
    s = nccltests.summarize(nccltests.parse(fixture_text("nccl_all_reduce_8gpu_sample.txt"), op="all_reduce"))
    assert s["alpha_us"] == pytest.approx(20, rel=0.1)  # the header states alpha = 20 us
    assert s["busbw_asymptote_gbps"] == pytest.approx(400, rel=0.05)  # and busbw -> 400 GB/s


def test_recheck_catches_a_wrong_factor():
    line = "     1048576        262144     float     sum      -1    100.00   10.49   10.49      0"
    res = nccltests.parse(line, op="all_reduce", nranks=8)  # busbw should be 10.49 x 1.75
    issues = nccltests.recheck(res)
    assert len(issues) == 1 and "busbw" in issues[0]


def test_nccl_info_lines_are_ignored():
    text = "host:1:1 [0] NCCL INFO Channel 00/02 :    0   1\n        4096          1024     float     sum      -1    20.00    0.20    0.20      0\n"
    assert len(nccltests.parse(text, op="all_reduce", nranks=2).rows) == 1


def test_fixtures_are_labelled_as_illustrative(fixture_text):
    for name in ("nccl_all_reduce_8gpu_sample.txt", "nccl_all_gather_4gpu_legacy_sample.txt",
                 "dcgm_exporter_sample.prom", "probe_docker_t4_sample.log", "probe_gke_l4_sample.log"):
        assert "illustrative" in fixture_text(name).splitlines()[0].lower()


def test_current_layout_names_its_own_collective(fixture_text):
    res = nccltests.parse(fixture_text("nccl_all_reduce_8gpu_sample.txt"))  # no op= needed
    assert res.op == "all_reduce" and res.header["nccl_tests_version"] == "2.20.0" and res.header["unalign"] == 0
    assert res.devices[1]["bus_id"] == "0000:20:00" and res.skipped == []


def test_one_log_with_two_runs_is_split(fixture_text):
    modern = fixture_text("nccl_all_reduce_8gpu_sample.txt")
    # an older job script echoed its own marker before a legacy run: markers of mixed kinds
    log = modern + "\n# Collective test starting: all_gather_perf\n" + fixture_text("nccl_all_gather_4gpu_legacy_sample.txt")
    runs = nccltests.parse_many(log)
    assert [(r.op, r.nranks, len(r.rows)) for r in runs] == [("all_reduce", 8, 28), ("all_gather", 4, 18)]
    # two current-format runs, each also preceded by an echoed marker (the duplicate chunks carry no rows)
    doubled = "\n".join("# Collective test starting: all_reduce_perf\n" + modern for _ in range(2))
    assert [len(r.rows) for r in nccltests.parse_many(doubled)] == [28, 28]


@pytest.mark.parametrize("header,row,expect", [
    # v2.9-era all_reduce_perf: redop but no root (12 tokens per row)
    ("#       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error",
     "     1048576        262144   float     sum    100.0   10.49   18.35  0e+00    101.0   10.38   18.17  0e+00",
     ("sum", None, 100.0, 101.0)),
    # current layout with per-iteration columns (-I 1) and a timestamp (-S 1)
    ("#       size         count      type   redop    root     time   algbw   busbw  #wrong    i_min    i_max    i_p99    i_cv%"
     "     time   algbw   busbw  #wrong    i_min    i_max    i_p99    i_cv%            timestamp",
     "     1048576        262144     float     sum      -1   100.00   10.49   18.35       0    98.10   103.20   103.00     1.20"
     "   101.00   10.38   18.17       0    99.00   104.00   104.00     1.10  2026-09-26 10:00:00",
     ("sum", -1, 100.0, 101.0)),
])
def test_other_column_layouts(header, row, expect):
    res = nccltests.parse(header + "\n" + row, op="all_reduce", nranks=8)
    r = res.rows[0]
    assert (r.redop, r.root, r.time_us, r.ip_time_us) == expect and res.skipped == []
    assert nccltests.recheck(res) == []
    no_header = nccltests.parse(row, op="all_reduce", nranks=8) if "timestamp" not in header else None
    assert no_header is None or (no_header.rows[0].redop, no_header.rows[0].root) == expect[:2]


def test_unrecognised_rows_are_reported_not_dropped_silently():
    res = nccltests.parse("     1048576        262144     float     sum      -1    100.00   10.49", op="all_reduce", nranks=8)
    assert res.rows == [] and len(res.skipped) == 1
