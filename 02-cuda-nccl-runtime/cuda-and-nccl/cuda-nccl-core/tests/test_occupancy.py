"""Occupancy arithmetic pinned to hand-computed cases (the cuda_occupancy.h rules)."""
import pytest

from gpusim import occupancy as occ


def test_register_limited_on_l4():
    # 64 regs x 32 lanes = 2048 regs/warp; 16384 per sub-partition -> 8 warps x 4 = 32 warps = 4 blocks of 8
    o = occ.occupancy(256, regs_per_thread=64, cc="8.9")
    assert (o.blocks_per_sm, o.active_warps, o.limiter) == (4, 32, "registers")
    assert o.occupancy == pytest.approx(32 / 48)


def test_shared_memory_limited_on_l4():
    # 48 KB + 1 KB reserved = 49 KB per block; 100 KB per SM -> 2 blocks
    o = occ.occupancy(256, regs_per_thread=64, smem_per_block=48 * 1024, cc="8.9")
    assert (o.blocks_per_sm, o.limiter) == (2, "shared_mem")


def test_full_occupancy_when_only_warp_slots_bind():
    o = occ.occupancy(256, regs_per_thread=32, cc="8.9")
    assert (o.blocks_per_sm, o.occupancy, o.limiter) == (6, 1.0, "warps")


def test_registers_are_split_across_four_sub_partitions():
    # 80 regs -> 2560/warp; 16384 // 2560 = 6 warps per sub-partition -> 24 warps, not 65536 // 2560 = 25
    assert occ.occupancy(32, regs_per_thread=80, cc="9.0").blocks_per_sm == 24


def test_impossible_launches_get_zero_blocks():
    assert occ.occupancy(128, regs_per_thread=256, cc="9.0").blocks_per_sm == 0
    assert occ.occupancy(128, smem_per_block=100 * 1024, cc="8.9").blocks_per_sm == 0    # max 99 KB/block
    with pytest.raises(ValueError):
        occ.occupancy(2048, cc="9.0")


def test_turing_has_half_the_warp_slots():
    o = occ.occupancy(256, regs_per_thread=64, cc="7.5")
    assert (o.blocks_per_sm, o.max_warps, o.occupancy) == (4, 32, 1.0)


def test_partial_last_wave():
    w = occ.waves(140, n_sms=132, blocks_per_sm=1)
    assert w["waves"] == 2 and w["efficiency"] == pytest.approx(140 / 264)


def test_littles_law_on_h100():
    assert occ.bytes_in_flight(3.35e12, 600e-9) == pytest.approx(2.01e6)
    # 2.01 MB / 132 SMs / (2048 threads x 4 B) ~ 1.86 independent 4-byte loads per thread
    assert occ.loads_in_flight_per_thread("H100-SXM", 600e-9) == pytest.approx(1.859, abs=1e-3)


def test_every_cc_in_the_compat_table_has_occupancy_limits():
    from gpusim import compat
    for gpu, (cc, *_rest) in compat.GPUS.items():
        assert cc in occ.SMS, f"{gpu} (CC {cc}) has no occupancy row"
        occ.occupancy(256, cc=cc)


def test_blackwell_rows():
    # CC 12.0 (RTX 50, RTX PRO 6000): 48 warps, 32 blocks, 128 KB smem per SM, 99 KB per block
    o = occ.occupancy(256, regs_per_thread=32, cc="12.0")
    assert (o.blocks_per_sm, o.max_warps, o.occupancy, o.limiter) == (6, 48, 1.0, "warps")
    # 40 KB + 1 KB reserved = 41 KB per block: 128 KB per SM -> 3 blocks, where CC 8.9's 100 KB holds 2
    assert occ.occupancy(256, smem_per_block=40 * 1024, cc="12.0").blocks_per_sm == 3
    assert occ.occupancy(256, smem_per_block=40 * 1024, cc="8.9").blocks_per_sm == 2
    assert occ.occupancy(128, smem_per_block=100 * 1024, cc="12.0").blocks_per_sm == 0   # max 99 KB/block
    # CC 10.3 (B300) has CC 10.0's limits: 64 warps, 228 KB per SM
    assert occ.occupancy(256, regs_per_thread=64, cc="10.3").blocks_per_sm == \
        occ.occupancy(256, regs_per_thread=64, cc="10.0").blocks_per_sm == 4


def test_unknown_cc_is_a_clear_error():
    with pytest.raises(KeyError, match="no occupancy limits for compute capability '6.1'; known: 7.0"):
        occ.occupancy(256, cc="6.1")


def test_primer_occupancy_table_matches_sms():
    """PRIMER §2.3's per-CC table is gpusim.occupancy.SMS."""
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[2] / "PRIMER.md").read_text()
    rows = re.findall(r"^\| (\d+\.\d) \([^)]*\) \| ([\d,]+) \| (\d+) \| (\d+) \| ([\d,]+) \| (\d+) KB \| (\d+) KB",
                      text, re.M)
    assert {r[0] for r in rows} >= {"7.5", "8.0", "8.9", "9.0", "10.0", "10.3", "12.0"}
    for cc, threads, warps, blocks, regs, smem, smem_block in rows:
        sm = occ.SMS[cc]
        assert (sm.max_threads, sm.max_warps, sm.max_blocks, sm.regs) == \
            (int(threads.replace(",", "")), int(warps), int(blocks), int(regs.replace(",", "")))
        assert (sm.smem, sm.smem_per_block) == (int(smem) * 1024, int(smem_block) * 1024)
