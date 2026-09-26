"""Checks for paged_attention_minimal.py. Offline, CPU only:  python3 -m pytest -q test_paged_attention_minimal.py"""
import numpy as np
import pytest

import paged_attention_minimal as pam


def test_blocks_in_use_counts_this_pool_not_the_default():
    mgr = pam.BlockManager(num_blocks=8)            # smaller than NUM_BLOCKS = 64
    assert mgr.blocks_in_use() == 0
    held = [mgr.allocate() for _ in range(3)]
    assert mgr.blocks_in_use() == 3
    for b in held:
        mgr.release(b)
    assert mgr.blocks_in_use() == 0


def test_small_pool_serves_a_sequence_and_frees_it():
    mgr = pam.BlockManager(num_blocks=4)
    seq = pam.Sequence(mgr)
    rng = np.random.default_rng(1)
    for _ in range(3 * pam.BLOCK_SIZE + 1):         # 13 tokens -> 4 blocks
        seq.append_kv(rng.standard_normal(pam.D_HEAD), rng.standard_normal(pam.D_HEAD))
    assert mgr.blocks_in_use() == 4
    q = rng.standard_normal(pam.D_HEAD)
    assert np.allclose(pam.paged_attention_blockwise(q, seq), pam.reference(q, seq))
    with pytest.raises(MemoryError):                # the fifth block does not exist
        for _ in range(pam.BLOCK_SIZE):
            seq.append_kv(np.zeros(pam.D_HEAD), np.zeros(pam.D_HEAD))
    seq.free()
    assert mgr.blocks_in_use() == 0


def test_pool_larger_than_the_backing_tensor_is_refused():
    with pytest.raises(ValueError):
        pam.BlockManager(num_blocks=pam.NUM_BLOCKS + 1)
