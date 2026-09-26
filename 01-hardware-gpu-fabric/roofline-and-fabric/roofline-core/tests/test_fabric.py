"""Fabrics: alpha-beta, collectives, TP cost, topology, staged copies."""
import pytest

from roofline import fabric, llm, specs

L = fabric.LINKS
M70 = llm.PRESETS["llama-3.1-70b"]
GIB = 1024 ** 3


def test_link_ladder_is_per_direction():
    assert L["nvlink4"].gbs == 18 * 25 == specs.get("h100-sxm").scaleup_gbs_per_dir
    assert L["nvlink5"].gbs == 18 * 50
    assert L["pcie5x16"].gbs == pytest.approx(32 * 16 * 128 / 130 / 8, abs=0.05)   # 32 GT/s, 128b/130b
    assert L["ib-ndr"].gbs == 400 / 8


def test_alpha_beta():
    lk = fabric.Link("test", 100, 10, "x")
    assert fabric.transfer_time(0, lk) == pytest.approx(10e-6)
    assert fabric.transfer_time(1e9, lk) == pytest.approx(10e-6 + 0.01)


def test_ring_allreduce_formula():
    lk = fabric.Link("test", 100, 5, "x")
    # 2(p-1) alpha + 2(p-1)/p n/beta, p = 8, n = 1 GB
    assert fabric.ring_allreduce_time(1e9, 8, lk) == pytest.approx(14 * 5e-6 + 2 * 7 / 8 * 1e9 / 100e9)
    assert fabric.ring_allgather_time(1e9, 8, lk) == pytest.approx(7 * 5e-6 + 7 / 8 * 1e9 / 100e9)
    assert fabric.recursive_doubling_allreduce_time(1e9, 8, lk) == pytest.approx(3 * (5e-6 + 0.01))
    assert fabric.ring_allreduce_time(1e9, 1, lk) == 0.0
    assert fabric.allreduce_crossover_bytes(8, L["nvlink4"]) == pytest.approx(7.2e6)   # p alpha beta


def test_busbw_matches_nccl_tests_definition():
    t = fabric.ring_allreduce_time(GIB, 8, L["nvlink4"])
    assert fabric.algbw(GIB, t) / 1e9 == pytest.approx(255.4, abs=0.1)
    assert fabric.busbw(GIB, t, 8) / 1e9 == pytest.approx(447.0, abs=0.1)   # ~ the 450 GB/s link
    assert fabric.BUSBW_FACTOR["allgather"](8) == pytest.approx(7 / 8)
    assert fabric.BUSBW_FACTOR["broadcast"](8) == 1.0


def test_tp_allreduce_cost():
    assert fabric.tp_allreduce_bytes(M70, 1) == 8192 * 2 == 16_384
    assert fabric.tp_allreduces_per_step(M70) == 160
    dec_ring = fabric.tp_comm_time(M70, 1, 8, L["nvlink4"])
    dec_rd = fabric.tp_comm_time(M70, 1, 8, L["nvlink4"], algo="recursive-doubling")
    assert dec_ring * 1e3 == pytest.approx(4.49, abs=0.01)        # latency-bound: 160 x 14 alpha
    assert dec_rd * 1e3 == pytest.approx(0.98, abs=0.01)
    in_node = fabric.tp_comm_time(M70, 4096, 8, L["nvlink4"])
    cross = fabric.tp_comm_time(M70, 4096, 8, L["ib-ndr"])
    assert in_node * 1e3 == pytest.approx(46.2, abs=0.1) and cross * 1e3 == pytest.approx(387.0, abs=0.1)
    per_gpu_compute = llm.prefill(M70, specs.get("h100-sxm"), 4096).t_compute / 8
    assert per_gpu_compute * 1e3 == pytest.approx(73.6, abs=0.1)
    assert cross / per_gpu_compute > 5                                # TP across the NIC: don't


def test_rails_let_every_gpu_use_its_own_nic():
    rails = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"])
    one_nic = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"], nics_per_node=1)
    assert rails * 1e3 == pytest.approx(8.26, abs=0.01)
    assert one_nic * 1e3 == pytest.approx(36.45, abs=0.01)


def test_leaf_spine_oversubscription_and_bisection():
    ls = fabric.leaf_spine(2048, 64)
    assert (ls.leaves, ls.spines, ls.oversubscription, ls.max_endpoints) == (64, 32, 1.0, 2048)
    ls3 = fabric.leaf_spine(3072, 64, down_per_leaf=48)
    assert (ls3.leaves, ls3.spines, ls3.oversubscription) == (64, 16, 3.0)
    assert fabric.bisection_gbs(3072, 400) == 76_800                  # 1536 x 50 GB/s
    assert fabric.bisection_gbs(3072, 400, 3.0) == 25_600
    with pytest.raises(ValueError):
        fabric.leaf_spine(4096, 64)                                   # needs a third tier


def test_rail_optimized_hops():
    assert fabric.switch_hops((0, 3), (0, 5), 32) == 0                # same node: NVLink
    assert fabric.switch_hops((0, 3), (17, 3), 32) == 1               # same rail, same leaf group
    assert fabric.switch_hops((0, 3), (17, 5), 32) == 3               # cross-rail: via spine
    assert fabric.switch_hops((0, 3), (17, 5), 32, pxn=True) == 1     # NVLink hop, then rail
    assert fabric.switch_hops((0, 3), (40, 3), 32) == 3               # other leaf group
    assert fabric.switch_hops((0, 3), (17, 5), 32, rail_optimized=False) == 1


def test_staged_versus_pipelined_copy():
    sf = fabric.staged_transfer_time(GIB, [63, 50, 63])
    assert sf == pytest.approx(GIB / 63e9 + GIB / 50e9 + GIB / 63e9)
    piped = fabric.staged_transfer_time(GIB, [63, 50, 63], chunk_bytes=1024 ** 2)
    assert piped == pytest.approx(GIB / 50e9 + 2 * 1024 ** 2 / 63e9)
    assert piped * 1e3 == pytest.approx(21.51, abs=0.01) and sf * 1e3 == pytest.approx(55.56, abs=0.01)


def test_topology_codes_rank():
    codes = ["SYS", "PIX", "NV4", "NODE", "PHB", "NV18", "PXB"]
    assert sorted(codes, key=fabric.topo_rank) == ["NV18", "NV4", "PIX", "PXB", "PHB", "NODE", "SYS"]
    assert set(fabric.TOPO_LEGEND) == {"NV#", "PIX", "PXB", "PHB", "NODE", "SYS"}
