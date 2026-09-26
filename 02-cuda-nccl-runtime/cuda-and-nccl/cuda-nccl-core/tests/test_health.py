"""GPU util vs SM active, throttle reasons, XID triage and alert severities."""
import pytest

from gpusim import health as H


def test_gpu_util_measures_time_not_space():
    c = H.util_counters([(0.0, 1.0, 8)], n_sms=132, window=1.0)     # one 8-block kernel, all the time
    assert c["gpu_util"] == 1.0 and c["sm_active"] == pytest.approx(8 / 132)


def test_counters_over_a_timeline():
    c = H.util_counters([(0.0, 0.5, 132), (0.5, 0.75, 66)], n_sms=132, window=1.0)
    assert c == pytest.approx({"gpu_util": 0.75, "sm_active": 0.5 + 0.125})
    overlap = H.util_counters([(0.0, 0.6, 100), (0.4, 1.0, 100)], n_sms=132, window=1.0)
    assert overlap["sm_active"] == pytest.approx((0.8 * 100 + 0.2 * 132) / 132)   # capped at all SMs


def test_throttle_bits():
    assert H.decode_throttle(0x44) == ["sw_power_cap", "hw_thermal_slowdown"]
    assert H.decode_throttle(0) == []


def test_xid_owners():
    assert {H.triage_xid(c).owner for c in (13, 31, 43, 45)} == {"app"}
    assert {H.triage_xid(c).owner for c in (48, 64, 74, 79, 95)} == {"hardware"}
    assert {H.triage_xid(c).owner for c in (63, 92, 94, 119)} == {"node"}
    assert H.triage_xid(12345).meaning == "not in this table"


def test_diagnose_sees_through_gpu_util():
    sample = {"DCGM_FI_DEV_GPU_UTIL": 100, "DCGM_FI_PROF_SM_ACTIVE": 0.12,
              "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": 0.02, "DCGM_FI_PROF_DRAM_ACTIVE": 0.7}
    found = " ".join(H.diagnose(sample))
    assert "busy but mostly empty" in found and "memory-bandwidth-bound" in found
    assert H.diagnose({"DCGM_FI_DEV_GPU_UTIL": 30}) == ["no findings"]


def test_alert_severity_follows_the_owner():
    assert H.alerts({"DCGM_FI_DEV_XID_ERRORS": 79})[0][0] == "page"
    assert H.alerts({"DCGM_FI_DEV_XID_ERRORS": 13})[0][0] == "notify"
    assert H.alerts({"DCGM_FI_DEV_ROW_REMAP_PENDING": 1}) == [("ticket", "row remap pending: reset the GPU when drained")]
    assert H.alerts({"DCGM_FI_DEV_CLOCKS_EVENT_REASONS": 0x4}) == []       # power cap alone is normal
