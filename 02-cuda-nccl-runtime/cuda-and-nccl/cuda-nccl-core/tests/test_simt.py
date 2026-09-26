"""SIMT rules pinned to NVIDIA's documented model (PRIMER sections 2 and 3)."""
import math

import numpy as np
import pytest

from gpusim import simt

W = simt.warp_addresses


@pytest.mark.parametrize("stride,offset,elem,sectors", [
    (1, 0, 4, 4),     # 32 x 4 B, aligned: 128 B = 4 sectors
    (1, 1, 4, 5),     # same, shifted by one float: straddles a fifth sector
    (2, 0, 4, 8),     # every other float: 256 B span
    (32, 0, 4, 32),   # a column of a 32-wide row-major float matrix: one sector per lane
    (1, 0, 8, 8),     # doubles: 256 B
    (1, 0, 16, 16),   # float4: 512 B
    (0, 0, 4, 1),     # every lane reads the same float
])
def test_sectors_per_warp_request(stride, offset, elem, sectors):
    assert simt.coalescing(W(stride, offset, elem), elem).sectors == sectors


def test_efficiency_matches_the_best_practices_guide():
    assert simt.coalescing(W(1)).efficiency == 1.0
    assert simt.coalescing(W(1, 1)).efficiency == pytest.approx(128 / 160)
    assert simt.coalescing(W(2)).efficiency == 0.5
    assert simt.coalescing(W(32)).efficiency == 32 * 4 / (32 * 32)


def test_inactive_lanes_generate_no_traffic():
    in_bounds = np.arange(32) < 8
    assert simt.coalescing(W(1), active=in_bounds).sectors == 1


def test_bank_conflict_degree_is_gcd_of_word_stride_and_32():
    for s in range(1, 65):
        assert simt.bank_conflicts(W(s)).degree == math.gcd(s, 32)


def test_padding_a_32x32_tile_removes_column_conflicts():
    column = lambda pitch: np.arange(32) * pitch * 4       # tile[lane][0] with a row pitch in floats
    assert simt.bank_conflicts(column(32)).wavefronts == 32
    assert simt.bank_conflicts(column(33)).wavefronts == 1


def test_same_word_is_a_broadcast():
    assert simt.bank_conflicts(W(0)) == simt.BankAccess(1, 1, 1)


def test_wide_accesses_are_served_per_half_and_quarter_warp():
    assert simt.bank_conflicts(W(1, elem_bytes=8), 8) == simt.BankAccess(2, 2, 1)
    assert simt.bank_conflicts(W(1, elem_bytes=16), 16) == simt.BankAccess(4, 4, 1)
    assert simt.bank_conflicts(W(2, elem_bytes=8), 8).degree == 2


def test_divergent_branch_pays_for_both_paths():
    tid = np.arange(64)
    d = simt.divergence(tid % 2, {0: 10, 1: 10})
    assert (d.issued, d.lane_ops, d.efficiency) == (40, 640, 0.5)
    assert simt.divergence((tid // 32) % 2, {0: 10, 1: 10}).efficiency == 1.0


def test_loop_runs_until_the_slowest_lane_finishes():
    d = simt.loop_divergence([1] * 31 + [32])
    assert (d.issued, d.lane_ops) == (32, 63)
