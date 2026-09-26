"""MIG placement geometry and the time-slicing / MPS / MIG latency model."""
import pytest

from gpusim import sharing as S


def test_mig_geometry_is_self_consistent():
    for gpu, profiles in S.MIG.items():
        assert [p.max_count for p in profiles] == [7, 4, 3, 2, 1, 1], gpu
        for p in profiles:
            assert all(s + p.mem_slices <= 8 and s + p.compute <= 7 for s in p.starts), (gpu, p)


@pytest.mark.parametrize("mix,fits", [
    (["4g.40gb", "2g.20gb", "1g.10gb"], True),
    (["3g.40gb", "2g.20gb", "1g.10gb", "1g.10gb"], True),
    (["2g.20gb"] * 3 + ["1g.10gb"], True),
    (["1g.20gb"] * 4, True),
    (["1g.10gb"] * 7, True),
    (["3g.40gb", "3g.40gb", "1g.10gb"], False),   # 7 compute slices, but no memory slice left
    (["4g.40gb", "4g.40gb"], False),              # 4g may only start at slice 0
    (["1g.10gb"] * 8, False),
])
def test_which_mixes_fit_an_h100(mix, fits):
    assert (S.pack("H100-80GB", mix) is not None) == fits


def test_unplanned_creation_order_fragments_the_gpu():
    order = ["1g.10gb", "1g.10gb", "1g.10gb", "4g.40gb"]
    assert S.first_fit("H100-80GB", order)[-1] == ("4g.40gb", None)
    assert S.pack("H100-80GB", order)[0] == ("4g.40gb", 0)


def test_time_slicing_latency_hand_computed():
    # 10 ms of work in 2 ms quanta = 5 quanta; between two of ours: 3 x (2 + 0.05) + 0.05 = 6.2 ms
    t = S.timeslice_latency(10.0, tenants=4, quantum_ms=2.0, switch_ms=0.05)
    assert t["best"] == pytest.approx(10 + 4 * 6.2) and t["worst"] == pytest.approx(10 + 5 * 6.2)
    assert S.timeslice_latency(10.0, tenants=1)["mean"] == 10.0


def test_small_kernels_share_well_on_mps_badly_with_time_slicing():
    lat = {m: S.shared_latency(10.0, 4, m, util=0.2) for m in ("exclusive", "mps", "mig", "time_slicing")}
    assert lat["exclusive"] == lat["mps"] == 10.0
    assert lat["mig"] == pytest.approx(10 * 0.2 / (1 / 7))
    assert lat["time_slicing"] > 3 * lat["exclusive"]
    assert S.shared_latency(10.0, 4, "mps", util=1.0) == 40.0     # a saturating kernel gains nothing
