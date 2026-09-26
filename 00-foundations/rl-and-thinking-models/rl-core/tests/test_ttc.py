"""Test-time compute: estimators pinned to the HumanEval / τ-bench formulas, voting, verifiers, allocation."""
import numpy as np

from rlcore import ttc


def test_unbiased_pass_at_k_hand_values():
    assert np.isclose(ttc.pass_at_k(10, 3, 1), 0.3) and round(ttc.pass_at_k(10, 3, 5), 6) == 0.916667
    assert ttc.pass_at_k(10, 3, 8) == 1.0                                  # n − c < k: some draw must hit
    assert round(ttc.pass_at_k(16, 4, 4), 6) == 0.728022 and round(ttc.pass_at_k(64, 16, 8), 6) == 0.914746


def test_pass_hat_k_is_all_k_succeed():
    assert ttc.pass_hat_k(10, 3, 5) == 0.0 and round(ttc.pass_hat_k(16, 4, 4), 6) == 0.000549
    assert ttc.pass_hat_k(10, 10, 5) == 1.0


def test_plugin_estimator_is_biased_and_the_unbiased_one_is_not():
    truth = 1 - 0.9 ** 5
    assert np.isclose(ttc.expected_estimate(ttc.pass_at_k, 10, 5, 0.1), truth)
    plug = ttc.expected_estimate(ttc.plugin_pass_at_k, 10, 5, 0.1)
    assert round(plug, 4) == 0.3485 and round(truth, 4) == 0.4095


def test_majority_vote_exact():
    assert ttc.majority_vote(["7", "3", "7", "5"]) == "7"
    assert np.isclose(ttc.majority_accuracy(0.4, [0.42, 0.18], 1), 0.4)
    # n = 3, one wrong answer: right iff ≥ 2 of 3 right
    assert np.isclose(ttc.majority_accuracy(0.6, [0.4], 3), 0.6 ** 3 + 3 * 0.6 ** 2 * 0.4)
    # a common misconception out-polls the right answer; spread-out mistakes do not
    assert round(ttc.majority_accuracy(0.4, [0.42, 0.18], 15), 3) == 0.449
    assert round(ttc.majority_accuracy(0.4, [0.15] * 4, 15), 3) == 0.780


def test_best_of_n_with_a_perfect_and_a_noisy_scorer():
    rng = np.random.default_rng(0)
    assert abs(ttc.best_of_n_accuracy(0.3, 4, 0.0, rng) - (1 - 0.7 ** 4)) < 0.01
    assert ttc.best_of_n_accuracy(0.3, 16, 1.0, rng) < ttc.best_of_n_accuracy(0.3, 16, 0.5, rng) < 1 - 0.7 ** 16


def test_thinking_has_diminishing_returns():
    qs = ttc.question_set(viable=None)
    acc = [ttc.accuracy(qs, L) for L in (0, 1000, 2000, 4000, 8000)]
    gains = np.diff(acc)
    assert acc[0] == 0.0 and all(gains > 0) and all(np.diff(gains / [1000, 1000, 2000, 4000]) < 0)


def test_allocation_depends_on_what_makes_questions_hard():
    """Only slow-to-crack questions (a = 1): one long sample wins. Unreliable approaches: with a verifier,
    more samples win as the budget grows; with a vote, thinking longer stays the better use until late."""
    only_slow, mixed = ttc.question_set(viable=None), ttc.question_set()
    assert ttc.allocate(only_slow, 4000)[1][0] == 1
    assert [ttc.allocate(mixed, b)[1][0] for b in (1000, 4000, 16000)] == [2, 8, 16]
    assert [ttc.allocate(mixed, b, method="vote")[1][0] for b in (1000, 4000, 16000)] == [1, 1, 4]
