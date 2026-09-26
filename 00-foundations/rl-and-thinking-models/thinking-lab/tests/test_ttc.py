"""Test-time-compute estimators pinned to hand values and to brute-force enumeration."""
import itertools
import math

from thinklab.thinking import ttc as T


def test_pass_at_k_hand_values():
    assert abs(T.pass_at_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(T.pass_at_k(10, 3, 5) - 0.916667) < 1e-6 and T.pass_at_k(10, 3, 8) == 1.0
    assert abs(T.pass_at_k(16, 4, 4) - 0.728022) < 1e-6 and abs(T.pass_at_k(64, 16, 8) - 0.914746) < 1e-6


def test_pass_at_k_equals_enumeration_and_pass_hat_k():
    n, c = 7, 3
    items = [1] * c + [0] * (n - c)
    for k in range(1, n + 1):
        subsets = list(itertools.combinations(items, k))
        assert abs(T.pass_at_k(n, c, k) - sum(any(s) for s in subsets) / len(subsets)) < 1e-12
        assert abs(T.pass_hat_k(n, c, k) - sum(all(s) for s in subsets) / len(subsets)) < 1e-12
    assert T.pass_hat_k(10, 3, 5) == 0.0 and abs(T.pass_hat_k(16, 4, 4) - 0.000549) < 1e-6


def test_majority_vote_and_maj_at_k():
    assert T.majority_vote(["b", "a", "a", "b"]) == "b" and T.majority_vote([None]) is None
    assert T.maj_at_k(["x", "x", "y"], "x", 3) == 1.0
    assert 0.0 <= T.maj_at_k(["x", "y", "y", "x", "z"], "x", 3) <= 1.0


def test_best_of_n_with_a_perfect_scorer_is_pass_at_k():
    correct = [1, 0, 0, 1, 0, 0, 0, 0]
    got = T.best_of_n_accuracy(correct, [float(c) for c in correct], 3, trials=4000)
    assert abs(got - T.pass_at_k(8, 2, 3)) < 0.03


def test_wilson_cost_and_compute_optimal():
    lo, hi = T.wilson_interval(30, 60)
    assert abs(lo - 0.3773) < 1e-3 and abs(hi - 0.6227) < 1e-3 and T.wilson_interval(0, 0) == (0.0, 1.0)
    assert T.cost_per_correct(1000, 0.5, 2.0) == 0.004 and T.cost_per_correct(1, 0, 1) == math.inf
    opts = [{"tokens": 100, "accuracy": 0.3}, {"tokens": 500, "accuracy": 0.6}, {"tokens": 400, "accuracy": 0.6}]
    assert T.compute_optimal(opts, 450)["tokens"] == 400 and T.compute_optimal(opts, 50) is None
