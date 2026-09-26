"""The toy trainer: hand-written gradients are right; no balancing collapses; aux loss and bias balance."""
import numpy as np
import pytest

from moecore import train as T


@pytest.mark.parametrize("hidden", [0, 6])
def test_hand_written_gradients_match_finite_differences(hidden):
    task = T.make_task(n=64, seed=3)
    rng = np.random.default_rng(0)
    w_r = rng.standard_normal((8, 4)) * 0.3
    w1 = rng.standard_normal((4, 8, hidden or 4)) * 0.3
    w2 = rng.standard_normal((4, hidden, 4)) * 0.3 if hidden else np.zeros((4, 0, 0))
    alpha = 0.5

    def total(wr, a, b):
        task_l, aux, *_ = T.loss_and_grads(task.x, task.y, wr, a, b, 2, alpha)
        return task_l + alpha * aux

    *_, grads = T.loss_and_grads(task.x, task.y, w_r, w1, w2, 2, alpha)
    eps = 1e-6
    for arr_i, arr in enumerate((w_r, w1, w2)):
        if arr.size == 0:
            continue
        for flat in rng.choice(arr.size, 5, replace=False):
            pos = np.unravel_index(flat, arr.shape)
            args = [w_r.copy(), w1.copy(), w2.copy()]
            args[arr_i][pos] += eps
            fd = (total(*args) - total(w_r, w1, w2)) / eps
            assert grads[arr_i][pos] == pytest.approx(fd, rel=1e-3, abs=1e-7)


def test_no_balancing_collapses_and_balancing_fixes_it():
    for seed in range(5):
        task = T.make_task(seed=seed)
        none, aux, bias = (T.train(task, balance=b) for b in ("none", "aux", "bias"))
        assert (none.final_share < 0.05).sum() >= 1, seed               # at least one dead expert
        for h in (aux, bias):
            assert (h.final_share > 0.15).all() and h.final_share.max() < 0.4, seed
            assert h.loss[-1] < none.loss[-1], seed


def test_demo_seed_collapses_from_a_split_start_and_specialises_when_balanced():
    task = T.make_task(seed=6)
    none, aux, bias = (T.train(task, balance=b) for b in ("none", "aux", "bias"))
    assert none.share[0].max() < 0.6 and none.final_share.max() > 0.9    # 55/45 at step 0, then one expert
    for h, floor in ((aux, 0.75), (bias, 0.9)):
        purity = h.placement.max(axis=1) / h.placement.sum(axis=1)
        assert len(set(h.placement.argmax(axis=1))) == 4                  # each cluster has its own expert
        assert purity.mean() > floor


def test_too_small_an_aux_weight_does_not_prevent_collapse():
    task = T.make_task(seed=4)
    assert T.train(task, balance="aux", alpha=0.01).final_share.max() > 0.9
    assert T.train(task, balance="aux", alpha=0.1).final_share.max() < 0.35
