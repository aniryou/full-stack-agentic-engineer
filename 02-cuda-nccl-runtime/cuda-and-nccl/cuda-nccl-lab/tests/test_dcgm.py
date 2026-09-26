"""DCGM exporter text -> signals, XID triage and alerts (the fixture is labelled illustrative)."""
import pathlib

import pytest
import yaml

from gpurt import dcgm


@pytest.fixture
def snaps(fixture_text):
    samples, meta = dcgm.parse_prometheus(fixture_text("dcgm_exporter_sample.prom"))
    assert meta["DCGM_FI_DEV_GPU_UTIL"]["type"] == "gauge"
    return {s.host.split("-")[-1]: s for s in dcgm.snapshots(samples)}  # keyed by node-a .. node-d suffix


def test_prometheus_text_parsing_with_escapes():
    samples, _ = dcgm.parse_prometheus('DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="A \\"quoted\\" name"} 55 1700000000\n')
    assert samples[0].labels["modelName"] == 'A "quoted" name' and samples[0].value == 55.0


def test_gpu_util_overstates_sm_activity(snaps):
    sig = dcgm.derive(snaps["b"])
    assert sig.gpu_util == pytest.approx(0.97) and sig.util_gap == pytest.approx(0.88)
    assert dcgm.bottleneck(sig).startswith("busy but under-filled")


def test_readings_for_each_sample_gpu(snaps):
    assert dcgm.bottleneck(dcgm.derive(snaps["a"])).startswith("memory-bandwidth bound")
    assert dcgm.bottleneck(dcgm.derive(snaps["c"])) == "idle: allocated but unused"
    assert dcgm.bottleneck(dcgm.derive(snaps["d"])) == "throttled: sw_thermal_slowdown"


def test_throttle_bits_and_xid_triage():
    assert dcgm.decode_throttle(0x60) == ["sw_thermal_slowdown", "hw_thermal_slowdown"]
    assert dcgm.decode_throttle(4) == ["sw_power_cap"]
    assert dcgm.triage_xid(79)["owner"] == "hardware" and dcgm.triage_xid(79)["severity"] == "critical"
    assert dcgm.triage_xid(63)["owner"] == "node" and dcgm.triage_xid(13)["owner"] == "application"
    unknown = dcgm.triage_xid(154)  # not in the table: the node operator at warning, as gpusim.health does
    assert (unknown["owner"], unknown["severity"], unknown["known"]) == ("node", "warning", False)


def test_unknown_xids_get_their_own_rule_not_the_application_one():
    sig = dcgm.Signals(1.0, None, None, None, None, None, None, None, [], 154, 0.0, 0.0, [])
    fired = [r.name for r in dcgm.RULES if r.check(sig)]
    assert fired == ["GpuXidUnknown"] and next(r for r in dcgm.RULES if r.name == "GpuXidUnknown").severity == "warning"
    for code in (13, 31, 43, 45):
        assert [r.name for r in dcgm.RULES if r.check(sig.__class__(**{**sig.__dict__, "xid": code}))] == ["GpuXidApplication"]


def test_alert_rules_fire_where_expected(snaps):
    fired = {(rule, gpu.split("/")[0].split("-")[-1]) for rule, _, gpu in dcgm.evaluate(list(snaps.values()))}
    assert fired == {("GpuBusyButUnderfilled", "b"), ("GpuXidApplication", "c"), ("GpuIdleWhileAllocated", "c"),
                     ("GpuUncorrectableRemappedRows", "d"), ("GpuThermalThrottling", "d")}


def test_xid_owners_partition_the_catalogue():
    owners = {o for _, o, _ in dcgm.XIDS.values()}
    assert owners == {"application", "node", "hardware"}
    hw, node = set(dcgm.xids_owned_by("hardware")), set(dcgm.xids_owned_by("node"))
    assert {48, 64, 74, 79, 95} == hw and not hw & node


def test_promql_bit_test_matches_the_python_decode():
    for mask in range(0, 512):
        promql_says = (mask // 32) % 4 > 0  # floor(x / 32) % 4 > 0
        assert promql_says == bool(set(dcgm.decode_throttle(mask)) & dcgm.THERMAL)


def test_rules_manifest_is_valid_yaml_and_in_sync_with_deploy(fixture_text):
    doc = yaml.safe_load(dcgm.rules_manifest())
    # cluster-scoped: a namespaced GMP Rules object only sees metrics from its own namespace
    assert doc["apiVersion"] == "monitoring.googleapis.com/v1" and doc["kind"] == "ClusterRules"
    assert "namespace" not in doc["metadata"]
    rules = doc["spec"]["groups"][0]["rules"]
    assert [r["alert"] for r in rules] == [r.name for r in dcgm.RULES]
    assert not any("<pod>" in r["expr"] or "<namespace>" in r["annotations"]["summary"] for r in rules)
    assert 'exported_pod!=""' in next(r["expr"] for r in rules if r["alert"] == "GpuIdleWhileAllocated")
    po = yaml.safe_load(dcgm.rules_manifest(kind="prometheus-operator"))
    assert (po["kind"], po["metadata"]["namespace"]) == ("PrometheusRule", "monitoring")
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (root / "deploy/gke/06-dcgm-alert-rules.yaml").read_text() == dcgm.rules_manifest()


def test_every_rule_is_valid_promql():
    promql = pytest.importorskip("promql_parser")
    for r in dcgm.RULES:
        ast = promql.parse(r.render()[0])
        if r.name.startswith("GpuXid"):  # the code list is parenthesised: `and` binds tighter than `or`
            assert str(ast.op) == "and" and "changes(" in r.expr, r.name


def test_which_exporter_can_fire_which_rules():
    stock = dcgm.EXPORTED_BY["stock dcgm-exporter (etc/default-counters.csv)"]
    gmp = dcgm.EXPORTED_BY["GMP DCGM example (prometheus-engine examples/nvidia-dcgm)"]
    assert dcgm.rules_that_cannot_fire(stock) == ["GpuThermalThrottling", "GpuHardwareSlowdown", "GpuBusyButUnderfilled"]
    assert "GpuXidHardware" in dcgm.rules_that_cannot_fire(gmp) and "GpuBusyButUnderfilled" not in dcgm.rules_that_cannot_fire(gmp)
    assert dcgm.rules_that_cannot_fire(stock | {"DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_PROF_SM_ACTIVE"}) == []


def test_the_lab_counters_csv_serves_every_field_the_module_reads():
    csv = pathlib.Path(__file__).resolve().parents[1] / "deploy/any-gpu/dcgm-counters.csv"
    enabled = {line.split(",")[0].strip() for line in csv.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")}
    assert enabled == set(dcgm.EXPORTED_BY["lab (deploy/any-gpu/dcgm-counters.csv)"])
    assert dcgm.SIGNAL_FIELDS | set().union(*(r.fields for r in dcgm.RULES)) <= enabled
    assert all(len(line.split(",")) >= 3 for line in csv.read_text().splitlines() if line.startswith("DCGM_"))


def test_the_fixture_exports_what_the_lab_csv_does(fixture_text):
    samples, _ = dcgm.parse_prometheus(fixture_text("dcgm_exporter_sample.prom"))
    assert dcgm.rules_that_cannot_fire({s.name for s in samples}) == []
    stock_only = "\n".join(line for line in fixture_text("dcgm_exporter_sample.prom").splitlines()
                           if "SM_ACTIVE" not in line and "CLOCKS_EVENT" not in line)
    report = dcgm.report(stock_only)
    assert report.startswith("not in this scrape: DCGM_FI_DEV_CLOCKS_EVENT_REASONS, DCGM_FI_PROF_SM_ACTIVE")


def test_scraped_workload_labels_are_read_too():
    text = 'DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="n1",namespace="gmp-public",pod="dcgm-x",exported_namespace="serving",exported_pod="vllm-0"} 3\n'
    snap = dcgm.snapshots(dcgm.parse_prometheus(text)[0])[0]
    assert snap.pods == {"serving/vllm-0"}
