"""REINFORCE, baselines, the KL penalty and its closed form, reward hacking, and RL lengthening thinking."""
import numpy as np

from rlcore import Policy, SeqTask, ThinkTask, pg


def test_advantages_hand_computed():
    r = [1.0, 0.0, 0.0, 1.0]
    assert np.allclose(pg.advantages(r, "none"), r)
    assert np.allclose(pg.advantages(r, "mean"), [0.5, -0.5, -0.5, 0.5])
    assert np.allclose(pg.advantages(r, "loo"), [1 - 1 / 3, -2 / 3, -2 / 3, 1 - 1 / 3])


def test_sft_reference_is_weak_and_hackable(brackets, sft_ref):
    true = pg.expected(sft_ref, brackets, brackets.verify)
    proxy = pg.expected(sft_ref, brackets, brackets.buggy_verify)
    assert round(true, 3) == 0.297 and round(proxy, 3) == 0.727


def test_reinforce_is_unbiased(brackets, sft_ref):
    """The mean of many REINFORCE estimates equals the exact gradient of E[R] (computed by enumeration)."""
    rng = np.random.default_rng(0)
    est = np.mean([pg.reinforce_grad(sft_ref, sft_ref.sample(brackets, rng, 16), "loo") for _ in range(400)], axis=0)
    eps, exact = 1e-5, np.zeros_like(sft_ref.theta)
    for i, j in zip(*np.nonzero(np.abs(est) > 0.02)):              # the parameters that matter
        up, dn = sft_ref.copy(), sft_ref.copy()
        up.theta[i, j] += eps
        dn.theta[i, j] -= eps
        exact[i, j] = (pg.expected(up, brackets, brackets.verify) - pg.expected(dn, brackets, brackets.verify)) / (2 * eps)
    mask = np.abs(est) > 0.02
    assert np.corrcoef(est[mask], exact[mask])[0, 1] > 0.95


def test_a_baseline_cuts_variance_and_ignores_reward_offsets():
    task = ThinkTask(e0=0.8, q=0.15, max_think=16)
    pol = Policy.for_task(task)
    v = {(b, off): pg.grad_variance(pol, task, np.random.default_rng(1), b, batch=8, trials=150, offset=off)
         for b in ("none", "mean") for off in (0.0, 5.0)}
    assert v[("mean", 0.0)] < 0.7 * v[("none", 0.0)]
    assert v[("none", 5.0)] > 20 * v[("none", 0.0)]                  # a constant reward is pure noise without a baseline
    assert np.isclose(v[("mean", 5.0)], v[("mean", 0.0)])             # ...and invisible with one (same seed)


def test_kl_optimal_closed_form(brackets, sft_ref):
    seqs = brackets.all_sequences()
    ref_p = sft_ref.sequence_probs(brackets, seqs)
    R = np.array([brackets.verify(s) for s in seqs])
    pi, er, kl = pg.kl_optimal(ref_p, R, beta=0.5)
    assert abs(kl - np.sum(pi * np.log(pi / ref_p))) < 1e-10          # E[R]/β − log Z is the KL
    assert abs(er - pi @ R) < 1e-12
    for other in (ref_p, 0.5 * pi + 0.5 * ref_p):                    # π* beats other policies on E[R] − β·KL
        assert er - 0.5 * kl >= other @ R - 0.5 * np.sum(other * np.log(other / ref_p)) - 1e-12
    assert np.isclose(pi[R == 1].sum(), ref_p[R == 1].sum() * np.e ** 2 / (ref_p[R == 1].sum() * np.e ** 2 + ref_p[R == 0].sum()))


def test_rl_finds_the_verifier_bug(brackets, sft_ref):
    """Trained on the buggy check, P(pass) → ~1 while true correctness collapses; β = 0.3 holds it near π_ref."""
    buggy = SeqTask("brackets", 8, verifier="buggy")
    hacked, leashed = sft_ref.copy(), sft_ref.copy()
    pg.train_reinforce(hacked, buggy, np.random.default_rng(0), steps=200, batch=16, lr=0.5)
    pg.train_reinforce(leashed, buggy, np.random.default_rng(0), steps=200, batch=16, lr=0.5, ref=sft_ref, beta=0.3)
    assert pg.expected(hacked, brackets, brackets.buggy_verify) > 0.99
    assert pg.expected(hacked, brackets, brackets.verify) < 0.08
    assert pg.expected(leashed, brackets, brackets.verify) > 0.3 and pg.kl_seq(leashed, sft_ref, brackets) < 0.3


def test_rl_lengthens_thinking_and_a_token_cost_stops_it():
    free, costly = ThinkTask(e0=0.8, q=0.15, max_think=16), ThinkTask(e0=0.8, q=0.15, max_think=16, cost=0.02)
    lengths = {}
    for name, task in (("free", free), ("costly", costly)):
        pol = Policy.for_task(task)                                  # starts answering at once half the time
        pg.train_reinforce(pol, task, np.random.default_rng(0), steps=300, batch=16, lr=2.0)
        lengths[name] = task.expected(pol.stop_probs(task))["length"]
    assert Policy.for_task(free) and lengths["free"] > 9 > lengths["costly"] > 5
