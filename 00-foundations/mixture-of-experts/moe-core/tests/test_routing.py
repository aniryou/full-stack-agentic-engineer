"""Load balance: the Switch loss in both normalisations, z-loss, capacity, dropless, bias, expert choice."""
import numpy as np
import pytest

from moecore import routing as R


def balanced(t=64, e=8, k=2):
    idx = np.array([[(i * k + j) % e for j in range(k)] for i in range(t)])   # every expert gets t k / e rows
    return np.full((t, e), 1 / e), idx


def test_switch_loss_is_k_in_hf_convention_and_1_in_megatron():
    probs, idx = balanced()
    assert R.switch_aux_loss(probs, idx, "hf") == pytest.approx(2.0)        # E sum (k/E)(1/E) = k
    assert R.switch_aux_loss(probs, idx, "megatron") == pytest.approx(1.0)


def test_switch_loss_when_collapsed():
    t, e = 32, 8
    probs = np.zeros((t, e)); probs[:, 0] = 1.0
    assert R.switch_aux_loss(probs, np.zeros((t, 1), dtype=int)) == pytest.approx(8.0)   # E: its maximum at k = 1
    probs = np.zeros((t, e)); probs[:, :2] = 0.5
    idx = np.tile([0, 1], (t, 1))
    assert R.switch_aux_loss(probs, idx, "hf") == pytest.approx(8.0)
    assert R.switch_aux_loss(probs, idx, "megatron") == pytest.approx(4.0)


def test_aux_gradient_is_e_f_over_t():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=10)
    idx = np.argmax(probs, axis=1)[:, None]
    g = R.switch_aux_grad(probs, idx)
    eps = 1e-6
    for (t, e) in [(0, 0), (3, 2), (9, 3)]:
        bumped = probs.copy(); bumped[t, e] += eps
        fd = (R.switch_aux_loss(bumped, idx) - R.switch_aux_loss(probs, idx)) / eps
        assert g[t, e] == pytest.approx(fd, rel=1e-4)


def test_sequence_loss_sees_what_the_batch_average_hides():
    t, e = 8, 4                     # 4 sequences of 8 tokens; sequence i sends everything to expert i
    idx = np.repeat(np.arange(4), t)[:, None]
    probs = np.eye(e)[idx[:, 0]]
    assert R.switch_aux_loss(probs, idx, "megatron") == pytest.approx(1.0)   # the batch looks perfect
    assert R.sequence_aux_loss(probs, idx, t) == pytest.approx(4.0)          # each sequence is collapsed


def test_z_loss_hand_computed():
    assert R.z_loss(np.zeros((3, 4))) == pytest.approx(np.log(4) ** 2)      # 1.9218
    assert R.z_loss(np.full((2, 4), 10.0)) == pytest.approx((10 + np.log(4)) ** 2)  # shifts are penalised


def test_capacity_and_dropping():
    assert R.capacity(1024, 8, 2, 1.25) == 320 and R.capacity(1024, 8, 2, 1.0) == 256
    idx = np.array([[0], [0], [0], [1]])
    w = np.array([[0.2], [0.9], [0.5], [1.0]])
    assert R.apply_capacity(idx, w, 2, "position")[:, 0].tolist() == [True, True, False, True]
    assert R.apply_capacity(idx, w, 2, "probs")[:, 0].tolist() == [False, True, True, True]


def test_align_block_size_reproduces_vllms_docstring_example():
    """vLLM moe_align_block_size docstring, with 0-based experts: 12 assignments, 4 experts, block 4."""
    idx = np.array([[1, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1, 2]])
    ids, experts, total = R.align_block_size(idx, 4, 4)
    assert ids.tolist() == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
    assert experts.tolist() == [0, 1, 2, 3] and total == 16


def test_bias_update_is_a_sign_step():
    b = R.update_bias(np.zeros(4), np.array([10, 2, 4, 0]), 0.001)   # mean load 4
    np.testing.assert_allclose(b, [-0.001, 0.001, 0.0, 0.001])


def test_expert_choice_is_balanced_by_construction():
    probs = np.random.default_rng(1).dirichlet(np.ones(8), size=64)
    a = R.expert_choice(probs, 16)                                  # capacity = T k / E with k = 2
    assert (a.sum(axis=0) == 16).all()
    per_token = a.sum(axis=1)
    assert per_token.sum() == 128 and per_token.min() == 0 or per_token.max() > 2   # uneven per token


def test_stats():
    s = R.stats(np.array([40, 20, 20, 0]))
    assert s["max_over_mean"] == 2.0 and s["balancedness"] == 0.5 and s["dead"] == 1
    assert R.stats(np.full(8, 5))["entropy"] == pytest.approx(1.0)
