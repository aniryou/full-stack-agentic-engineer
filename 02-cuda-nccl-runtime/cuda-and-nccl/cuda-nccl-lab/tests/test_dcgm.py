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
    assert dcgm.triage_xid(79)["category"] == "hardware" and dcgm.triage_xid(13)["category"] == "application"
    assert dcgm.triage_xid(9999)["meaning"] == "unknown XID"


def test_alert_rules_fire_where_expected(snaps):
    fired = {(rule, gpu.split("/")[0].split("-")[-1]) for rule, _, gpu in dcgm.evaluate(list(snaps.values()))}
    assert fired == {("GpuBusyButUnderfilled", "b"), ("GpuXidOther", "c"), ("GpuIdleWhileAllocated", "c"),
                     ("GpuUncorrectableRemappedRows", "d"), ("GpuThermalThrottling", "d")}


def test_promql_bit_test_matches_the_python_decode():
    for mask in range(0, 512):
        promql_says = (mask // 32) % 4 > 0  # floor(x / 32) % 4 > 0
        assert promql_says == bool(set(dcgm.decode_throttle(mask)) & dcgm.THERMAL)


def test_rules_manifest_is_valid_yaml_and_in_sync_with_deploy(fixture_text):
    doc = yaml.safe_load(dcgm.rules_manifest())
    assert doc["apiVersion"] == "monitoring.googleapis.com/v1" and doc["kind"] == "Rules"
    assert [r["alert"] for r in doc["spec"]["groups"][0]["rules"]] == [r.name for r in dcgm.RULES]
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (root / "deploy/gke/06-dcgm-alert-rules.yaml").read_text() == dcgm.rules_manifest()
