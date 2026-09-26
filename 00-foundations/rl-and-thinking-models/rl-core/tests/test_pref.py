"""Bradley–Terry, DPO against its closed form, IPO, GAE, and reward-model over-optimisation."""
import math

import numpy as np

from rlcore import pg, pref


def test_bt_is_shift_invariant_and_fits_a_linear_reward():
    assert np.isclose(pref.bt_prob(2.0, 1.0), pref.bt_prob(12.0, 11.0)) and np.isclose(pref.bt_prob(1, 1), 0.5)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 2))
    y = rng.standard_normal((4000, 2))
    w_true = np.array([1.5, -0.5])
    a_wins = rng.random(4000) < pref.bt_prob(x @ w_true, y @ w_true)
    ch, rj = np.where(a_wins[:, None], x, y), np.where(a_wins[:, None], y, x)
    w = pref.fit_bradley_terry(ch, rj, l2=0.0, steps=1500)
    assert np.allclose(w, w_true, atol=0.15)
    assert pref.bt_nll(w, ch, rj) < math.log(2)


def test_dpo_and_ipo_losses_hand_computed():
    assert round(float(pref.dpo_loss(1.0, -1.0, beta=0.1)), 6) == 0.598139     # −log σ(0.2)
    assert np.isclose(pref.dpo_loss(0.3, 0.3, beta=0.1), math.log(2))           # zero margin: ln 2
    assert np.isclose(pref.ipo_loss(1.0, -1.0, beta=0.1), (2 - 5) ** 2)         # target margin 1/(2β) = 5


def test_dpo_gradient_matches_finite_differences(brackets, sft_ref):
    seqs = brackets.all_sequences()
    pairs = [(seqs[i], seqs[j]) for i, j in [(10, 200), (77, 3), (140, 141)]]
    data = pref.encode_pairs(brackets, pairs)
    pol = sft_ref.copy()
    pol.theta += np.random.default_rng(1).standard_normal(pol.theta.shape) * 0.3

    def loss(p):
        lp = lambda q, k: pref._seq_logps(q.theta, data[k], data["n"])
        return float(np.mean(pref.dpo_loss(lp(p, "chosen") - lp(sft_ref, "chosen"), lp(p, "rejected") - lp(sft_ref, "rejected"), 0.5)))

    before = pol.copy()
    pref.dpo_step(pol, sft_ref, data, beta=0.5, lr=1.0)
    step = pol.theta - before.theta                                  # = −∇loss (lr 1)
    eps, i, j = 1e-6, *np.unravel_index(np.abs(step).argmax(), step.shape)
    up, dn = before.copy(), before.copy()
    up.theta[i, j] += eps
    dn.theta[i, j] -= eps
    assert np.isclose(step[i, j], -(loss(up) - loss(dn)) / (2 * eps), atol=1e-7)


def test_dpo_recovers_the_kl_optimal_policy(brackets, sft_ref):
    """Pairs labelled by Bradley–Terry with r = 3·balanced: DPO lands on π* ∝ π_ref·exp(r/β), and its
    implicit reward β·log(π/π_ref) recovers the reward gap of 3."""
    seqs = brackets.all_sequences()
    ref_p = sft_ref.sequence_probs(brackets, seqs)
    R = 3.0 * np.array([brackets.verify(s) for s in seqs])
    pistar, _, _ = pg.kl_optimal(ref_p, R, beta=1.0)
    rng = np.random.default_rng(0)
    a, b = rng.choice(len(seqs), 4096, p=ref_p), rng.choice(len(seqs), 4096, p=ref_p)
    wins = rng.random(4096) < pref.bt_prob(R[a], R[b])
    data = pref.encode_pairs(brackets, [(seqs[i], seqs[j]) if w else (seqs[j], seqs[i]) for i, j, w in zip(a, b, wins)])
    pol = sft_ref.copy()
    for _ in range(150):
        m = pref.dpo_step(pol, sft_ref, data, beta=1.0, lr=5.0)
    p = pol.sequence_probs(brackets, seqs)
    assert abs(p @ (R > 0) - pistar @ (R > 0)) < 0.02 and np.sum(p * np.log(p / pistar)) < 0.02
    ir = np.array([pref.implicit_reward(pol, sft_ref, brackets, s, 1.0) for s in seqs])
    assert abs((ir[R > 0].mean() - ir[R == 0].mean()) - 3.0) < 0.15
    assert m["rewards/chosen"] < 0 < m["rewards/margins"]            # the chosen log-ratio falls; the margin grows


def test_gae_hand_computed():
    r, v = [0.0, 0.0, 1.0], [0.5, 0.6, 0.8]
    delta = [0 + 0.6 - 0.5, 0 + 0.8 - 0.6, 1 + 0 - 0.8]               # 0.1, 0.2, 0.2
    assert np.allclose(pref.gae(r, v, 1.0, 0.0), delta)
    assert np.allclose(pref.gae(r, v, 1.0, 1.0), [1 - 0.5, 1 - 0.6, 1 - 0.8])   # Monte-Carlo return − V
    assert np.allclose(pref.gae(r, v, 1.0, 0.5), [0.1 + 0.5 * (0.2 + 0.5 * 0.2), 0.2 + 0.5 * 0.2, 0.2])


def test_length_biased_rm_is_over_optimised():
    """The RM learns the annotator's length taste; optimising it harder raises the proxy monotonically while
    true quality peaks and then falls below the reference, and the answers get longer."""
    cat = pref.response_catalogue()
    ref = np.exp(-cat["padding"] / 150)
    ref /= ref.sum()
    ch, rj = pref.length_biased_prefs(cat, 3000, length_weight=0.6, ref_probs=ref)
    X = np.stack([cat["quality"], cat["length"] / 100], axis=1)
    w = pref.fit_bradley_terry(X[ch], X[rj], steps=2000)
    assert abs(w[0] - 1.0) < 0.1 and abs(w[1] - 0.6) < 0.1
    rows = [pg.kl_optimal(ref, X @ w, beta) for beta in (10, 3, 1, 0.5, 0.3, 0.1)]
    proxy = [er for _, er, _ in rows]
    gold = [pi @ cat["quality"] for pi, _, _ in rows]
    length = [pi @ cat["length"] for pi, _, _ in rows]
    assert all(np.diff(proxy) > 0) and all(np.diff(length) > 0)
    assert max(gold) > gold[0] > gold[-1] and gold[-1] < ref @ cat["quality"]
