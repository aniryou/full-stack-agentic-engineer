"""The toy environments and the tabular policy: verifiers, exact spaces, closed-form gradients."""
import math

import numpy as np

from rlcore import ANSWER, THINK, Policy, SeqTask, ThinkTask, Trajectory


def test_random_bracket_success_is_catalan_over_2_to_the_n():
    task = SeqTask("brackets", 8)
    assert task.random_success_rate() == 14 / 256 == math.comb(8, 4) / 5 / 256   # Catalan(4) = 14
    assert SeqTask("brackets", 6).random_success_rate() == 5 / 64                   # Catalan(3) = 5


def test_buggy_verifier_passes_what_the_true_one_rejects():
    task = SeqTask("brackets", 8)
    assert task.verify([0, 1, 0, 0, 1, 1, 0, 1]) == 1.0            # ()(())()
    assert task.verify([1, 0, 0, 1, 0, 1, 0, 1]) == 0.0            # )(()()() — goes negative first
    assert task.buggy_verify([1, 0, 0, 1, 0, 1, 0, 1]) == 1.0      # the early exit counts it as a pass
    seqs = task.all_sequences()
    assert sum(task.buggy_verify(s) for s in seqs) == 256 - math.comb(8, 4) + 14   # ever-negative + balanced


def test_reward_depends_only_on_the_final_state():
    """The 'broken' state makes the verifier a function of the last state, so a per-state policy can
    represent the KL-optimal policy exactly."""
    task = SeqTask("brackets", 8)
    by_state = {}
    for s in task.all_sequences():
        final = task.state(0, s[:-1]) * 2 + s[-1]
        by_state.setdefault(final, set()).add((task.verify(s), task.buggy_verify(s)))
    assert all(len(v) == 1 for v in by_state.values())


def test_think_task_accuracy_and_optimal_length():
    task = ThinkTask(e0=0.8, q=0.1, max_think=32)
    assert task.accuracy(0) == 1 - 0.8 and abs(task.accuracy(10) - (1 - 0.8 * 0.9 ** 10)) < 1e-12
    L = task.optimal_length(cost=0.01)                            # marginal gain e0·(−ln 0.9)·0.9^L = cost
    assert abs(0.8 * -math.log(0.9) * 0.9 ** L - 0.01) < 1e-12 and abs(L - 20.23) < 0.01
    assert task.optimal_length(cost=0.0) == 32


def test_think_task_truncation_and_budget_forcing():
    rng = np.random.default_rng(0)
    cut = ThinkTask(max_think=4)
    r, info = cut.score(0, [THINK] * 4, rng)
    assert info["truncated"] and r == 0.0                         # max_tokens ran out inside the think block
    forced = ThinkTask(e0=0.0, max_think=4, force_answer=True)
    r, info = forced.score(0, [THINK] * 4, rng)
    assert not info["truncated"] and r == 1.0                     # e0 = 0: any answer is right
    _, info = cut.score(0, [THINK, THINK, ANSWER], rng)
    assert info["length"] == 2 and not info["truncated"]


def test_length_distribution_is_exact():
    task = ThinkTask(max_think=6)
    s = np.full(6, 0.3)
    d = task.length_distribution(s)
    assert abs(d.sum() - 1) < 1e-12 and abs(d[0] - 0.3) < 1e-12 and abs(d[-1] - 0.7 ** 6) < 1e-12
    pol = Policy.for_task(task)
    pol.theta[:, ANSWER] = np.log(0.3 / 0.7)
    rng = np.random.default_rng(1)
    lengths = [t.info["length"] + t.info["truncated"] for t in pol.sample(task, rng, 20000)]
    emp = np.bincount(lengths, minlength=7) / 20000              # truncated counted as length 6+1
    assert np.abs(emp[:6] - d[:6]).max() < 0.01 and abs(emp[-1] - d[-1]) < 0.01


def test_grad_logprob_matches_finite_differences():
    task = SeqTask("brackets", 6)
    pol = Policy.for_task(task, np.random.default_rng(2).standard_normal((task.n_states, 2)))
    traj = pol.sample(task, np.random.default_rng(3), 1)[0]
    w = np.linspace(0.5, 2.0, traj.length)
    g = pol.grad_logprob(traj, w)
    eps, num = 1e-6, np.zeros_like(pol.theta)
    for idx in zip(*np.nonzero(np.ones_like(pol.theta))):
        up, dn = pol.copy(), pol.copy()
        up.theta[idx] += eps
        dn.theta[idx] -= eps
        num[idx] = (np.dot(w, up.token_logprobs(traj)) - np.dot(w, dn.token_logprobs(traj))) / (2 * eps)
    assert np.allclose(g, num, atol=1e-7)


def test_sequence_probs_sum_to_one_and_copies_are_frozen():
    task = SeqTask("brackets", 6)
    pol = Policy.for_task(task, np.random.default_rng(4).standard_normal((task.n_states, 2)))
    assert abs(pol.sequence_probs(task).sum() - 1) < 1e-12
    ref = pol.copy()
    pol.step(np.ones_like(pol.theta), 1.0)
    assert not np.shares_memory(ref.theta, pol.theta)
    t = Trajectory(0, [task.state(0, [])], [0])
    assert ref.token_logprobs(t)[0] == np.log(ref.probs()[t.states[0], 0])
