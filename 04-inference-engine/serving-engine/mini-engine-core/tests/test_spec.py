"""Speculative decoding is exact: a chi-square test that can also catch a wrong implementation."""
import numpy as np

from minengine import SMALL, TinyLM, encode, spec


def test_expected_tokens_and_speedup_hand_computed():
    assert np.isclose(spec.expected_tokens(0.8, 4), (1 - 0.8 ** 5) / 0.2) and np.isclose(spec.expected_tokens(0.8, 4), 3.3616)
    assert spec.expected_tokens(1.0, 4) == 5 and spec.expected_tokens(0.0, 4) == 1
    assert np.isclose(spec.speedup(0.8, 4, 0.1), 3.3616 / 1.4)       # E[tokens] / (k c + 1)
    # k = 4, 5, 6, 7 -> 2.401, 2.460, 2.470, 2.448: the optimum is shallow
    assert spec.best_k(0.8, 0.1) == 6 and spec.best_k(0.95, 0.02) > spec.best_k(0.6, 0.02)


def test_acceptance_rate_is_one_minus_total_variation():
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert np.isclose(spec.acceptance_rate(p, q), 0.7) and np.isclose(0.7, 1 - 0.5 * np.abs(p - q).sum())


P = np.array([[.20, .45, .20, .15], [.60, .10, .10, .20], [.05, .05, .30, .60], [.25, .25, .25, .25]])
Q = np.array([[.40, .20, .30, .10], [.25, .25, .25, .25], [.10, .60, .10, .20], [.70, .10, .10, .10]])


def _joint_chi2(verify_fn, n=12000):
    """Generate 3 tokens after [0] with k=2 drafts from Q, count all 64 outcomes, compare with P's joint."""
    rng, counts = np.random.default_rng(0), np.zeros((4, 4, 4))
    orig, spec.verify = spec.verify, verify_fn
    try:
        for _ in range(n):
            out, _ = spec.speculative_generate(lambda t: P[np.asarray(t)], [0], 3, k=2,
                                               draft_probs=lambda t: Q[np.asarray(t)], seed=rng)
            counts[tuple(out)] += 1
    finally:
        spec.verify = orig
    expected = n * np.einsum("a,ab,bc->abc", P[0], P, P)
    return ((counts - expected) ** 2 / expected).sum()


def test_speculative_sampling_follows_the_target_distribution():
    assert _joint_chi2(spec.verify) < 103.4                          # chi2(63 dof) at p = 0.001


def test_the_statistical_test_catches_a_plausible_bug():
    def resample_from_p(p, q, draft, rng):                          # "on rejection, just sample p": wrong
        out = []
        for i, x in enumerate(draft):
            if rng.random() < min(1.0, p[i, x] / q[i, x]):
                out.append(int(x))
                continue
            return out + [int(rng.choice(p.shape[1], p=p[i]))]
        return out + [int(rng.choice(p.shape[1], p=p[len(draft)]))]
    assert _joint_chi2(resample_from_p) > 103.4


def test_greedy_speculation_reproduces_greedy_decoding_and_measured_acceptance():
    target, draft = TinyLM(), TinyLM(SMALL, seed=1)
    prompt = encode("The engine runs ")
    out, st = spec.speculative_generate(spec.lm_probs(target, 0), prompt, 24, k=3,
                                        draft_probs=spec.lm_probs(draft, 0))
    assert out == target.generate_dense(prompt, 24)
    assert st.passes < 24 and 0 < st.acceptance <= 1                  # fewer target passes than tokens


def test_ngram_prompt_lookup():
    toks = encode("the cat sat. the cat")
    assert spec.ngram_propose(toks, 3, n=3) == encode(" sa")          # continues the earlier "the cat"
    assert spec.ngram_propose(encode("abcdef"), 3) == []
