"""Soft targets, temperature and the divergences, pinned to hand-computed values (primer §2)."""
import math

import numpy as np
import pytest

from distillcore import losses as L

Z = np.array([[4.0, 3.0, 1.0, 0.0, -1.0]])        # teacher logits, primer §2
V = np.array([[3.0, 3.5, 0.0, 0.5, -1.0]])        # student logits


def fd(f, v, eps=1e-6):
    g = np.zeros_like(v)
    for idx in np.ndindex(v.shape):
        vp, vm = v.copy(), v.copy()
        vp[idx] += eps
        vm[idx] -= eps
        g[idx] = (f(vp)[0] - f(vm)[0]) / (2 * eps)
    return g


def test_five_token_soft_targets_by_hand():
    p1, q1 = L.softmax(Z)[0], L.softmax(V)[0]
    assert np.allclose(p1, [0.6931, 0.2550, 0.0345, 0.0127, 0.0047], atol=5e-5)
    assert np.allclose(q1, [0.3573, 0.5891, 0.0178, 0.0293, 0.0065], atol=5e-5)
    assert np.allclose(L.softmax(Z, 2)[0], [0.4885, 0.2963, 0.1090, 0.0661, 0.0401], atol=5e-5)
    assert round(float(L.kl(p1, q1)[0]), 5) == 0.25650


def test_temperature_shrinks_the_kl_and_t_squared_restores_it():
    rows = {T: (L.kd(V, Z, T, scale=False)[0], L.kd(V, Z, T)[0]) for T in (1, 2, 4, 10, 100, 1000)}
    assert round(rows[2][0], 5) == 0.06639 and round(rows[2][1], 5) == 0.26558
    assert [round(rows[T][1], 5) for T in (4, 10, 100, 1000)] == [0.25542, 0.24196, 0.23129, 0.23013]
    assert rows[1000][0] < 1e-6                                    # the unscaled soft term vanishes


def test_gradient_is_q_minus_p_over_t():
    T = 2.0
    _, g = L.kd(V, Z, T, scale=False)
    want = (L.softmax(V, T) - L.softmax(Z, T)) / T
    assert np.allclose(g, want) and np.allclose(g, fd(lambda v: L.kd(v, Z, T, scale=False), V), atol=1e-7)
    assert np.allclose(g[0], [-0.073543, 0.071047, -0.016410, 0.015853, 0.003053], atol=5e-7)
    _, g2 = L.kd(V, Z, T)
    assert np.allclose(g2, T * T * g)                               # T²·KL: gradient T·(q − p)


def test_high_temperature_limit_is_centred_logit_matching():
    mse, g = L.logit_mse(V, Z)
    assert abs(mse - 0.23) < 1e-12                                  # (1/2N)·‖c(v) − c(z)‖² = 2.3/10
    assert abs(L.kd(V, Z, 1e4)[0] - mse) < 1e-4
    assert np.allclose(L.kd(V, Z, 1e4)[1], g, atol=1e-4)            # the gradient: ((v − v̄) − (z − z̄))/N


def test_hard_label_is_soft_ce_with_a_one_hot_target_and_hinton_mixes():
    y = np.array([1])
    assert np.allclose(L.hard_ce(V, y)[1], L.soft_ce(V, np.eye(5)[y])[1])
    lh, gh = L.hinton(V, Z, y, T=2.0, alpha=0.3)
    assert math.isclose(lh, 0.3 * L.kd(V, Z, 2.0)[0] + 0.7 * L.hard_ce(V, y)[0])
    assert np.allclose(gh, fd(lambda v: L.hinton(v, Z, y, 2.0, 0.3), V), atol=1e-7)


def test_a_hard_label_adds_one_minus_sum_p_squared_of_noise():
    p = np.array([0.8, 0.1, 0.1])
    assert abs(L.label_noise(p)[0] - 0.34) < 1e-12
    q = np.array([0.5, 0.3, 0.2])
    exp_hard = sum(pi * np.sum((np.eye(3)[i] - q) ** 2) for i, pi in enumerate(p))
    assert math.isclose(exp_hard - np.sum((p - q) ** 2), 0.34)     # E‖onehot − q‖² = ‖p − q‖² + (1 − Σp²)


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_gkd_gradient_matches_finite_differences(beta):
    rng = np.random.default_rng(0)
    v, p = rng.normal(size=(3, 6)), L.softmax(2 * rng.normal(size=(3, 6)))
    assert np.allclose(L.gkd(v, p, beta)[1], fd(lambda x: L.gkd(x, p, beta), v), atol=1e-7)


def test_gkd_endpoints_are_the_exact_kls_as_trl_computes_them():
    p, q = L.softmax(Z), L.softmax(V)
    assert round(L.gkd(V, p, 0.0)[0], 6) == 0.256501 and round(L.gkd(V, p, 1.0)[0], 6) == 0.271424
    vals = {b: round(L.gkd(V, p, b)[0], 6) for b in (0.01, 0.1, 0.5, 0.9, 0.99)}
    assert vals == {0.01: 0.002539, 0.1: 0.023026, 0.5: 0.064431, 0.9: 0.024067, 0.99: 0.002683}
    assert all(v <= math.log(2) for v in vals.values())
    assert abs(vals[0.01] / 0.01 - float(L.kl(p, q)[0])) < 0.005  # small β ≈ β·KL(p ‖ q)
