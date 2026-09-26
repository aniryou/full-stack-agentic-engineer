"""Sampling reshapes the distribution; formulas pinned to hand-computed values."""
import numpy as np

from minengine import ChoiceFSM, Engine, SamplingParams, TinyLM
from minengine.sampler import (apply_penalties, min_p_filter, probs, process_logits, sample,
                               top_k_filter, top_p_filter)

LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))       # a distribution you can do in your head


def kept(x):
    return set(np.flatnonzero(np.isfinite(x)).tolist())


def test_top_k_top_p_min_p_hand_computed():
    assert kept(top_k_filter(LOGITS, 2)) == {0, 1}
    assert kept(top_p_filter(LOGITS, 0.8)) == {0, 1}      # 0.5 + 0.3 reaches 0.8
    assert kept(top_p_filter(LOGITS, 0.81)) == {0, 1, 2}  # ...0.81 needs the third token
    assert kept(top_p_filter(LOGITS, 0.01)) == {0}        # the top token always survives
    assert kept(min_p_filter(LOGITS, 0.2)) == {0, 1, 2}   # threshold 0.2 x 0.5 = 0.1
    np.testing.assert_allclose(probs(top_k_filter(LOGITS, 2))[:2], [0.625, 0.375])   # renormalised


def test_order_of_operations_matches_vllm():
    """temperature -> min-p -> top-k -> top-p (vLLM V1's Sampler). Each case keeps a different set
    under the wrong order, so a reordered pipeline fails here."""
    # T = 2 first: probs ~ sqrt(p) = .379 .294 .208 .120; min-p 0.35 x .379 = .133 keeps {0, 1, 2}
    # (min-p on the raw probs would keep only {0, 1}: 0.15 < 0.35 x 0.5)
    assert kept(process_logits(LOGITS, SamplingParams(temperature=2.0, min_p=0.35))) == {0, 1, 2}
    # top-k 2 first renormalises to .625 / .375; top-p 0.6 then keeps {0} (top-p first would keep {0, 1})
    assert kept(process_logits(LOGITS, SamplingParams(top_k=2, top_p=0.6))) == {0}
    # T = 0.5 first: probs ~ p^2 = .685 .247 .062 .007; top-p 0.6 keeps {0} (on the raw probs {0, 1})
    assert kept(process_logits(LOGITS, SamplingParams(temperature=0.5, top_p=0.6))) == {0}


def test_temperature_and_greedy():
    p = SamplingParams(temperature=2.0)
    np.testing.assert_allclose(probs(process_logits(LOGITS, p)), probs(LOGITS / 2))
    tok, _, _ = sample(LOGITS, SamplingParams(temperature=0), np.random.default_rng(0))
    assert tok == 0


def test_penalty_semantics():
    x = np.array([2.0, -2.0, 1.0, 0.5])
    rep = apply_penalties(x, SamplingParams(repetition_penalty=2.0), prompt_ids=[0], output_ids=[1])
    np.testing.assert_allclose(rep, [1.0, -4.0, 1.0, 0.5])            # divide positives, multiply negatives
    pf = apply_penalties(x, SamplingParams(presence_penalty=0.5, frequency_penalty=0.25), output_ids=[2, 2, 3])
    np.testing.assert_allclose(pf, [2.0, -2.0, 1.0 - 0.5 - 0.5, 0.5 - 0.5 - 0.25])


def test_draws_follow_the_processed_distribution():
    rng, p = np.random.default_rng(0), SamplingParams(top_k=3, temperature=1.0)
    n = 20000
    counts = np.bincount([sample(LOGITS, p, rng)[0] for _ in range(n)], minlength=4)
    expected = n * np.array([0.5, 0.3, 0.15, 0.0]) / 0.95
    assert counts[3] == 0
    chi2 = ((counts[:3] - expected[:3]) ** 2 / expected[:3]).sum()
    assert chi2 < 13.8                                                 # chi2(2 dof), p = 0.001


def test_logprobs_are_raw_not_processed():
    tok, lp, top = sample(LOGITS, SamplingParams(temperature=0.3, logprobs=2), np.random.default_rng(1))
    assert np.isclose(lp, LOGITS[tok]) and set(top) == {0, 1}          # LOGITS are already log-probs


def test_seeded_request_is_reproducible_whatever_it_is_batched_with():
    model, sp = TinyLM(), SamplingParams(max_tokens=16, temperature=0.9, top_p=0.9, seed=7)
    alone = Engine(model, num_blocks=64, block_size=4).generate(["The engine "], sp)[0].text
    mixed = Engine(model, num_blocks=64, block_size=4).generate(
        ["When memory", "The engine ", "Each step"], [SamplingParams(max_tokens=9), sp, SamplingParams(temperature=1.5)])
    assert mixed[1].text == alone


def test_fsm_mask_forces_a_legal_output():
    fsm = ChoiceFSM(['{"answer": "yes"}', '{"answer": "no"}'])
    outs = Engine(TinyLM(), num_blocks=64, block_size=4).generate(
        ["Is it on? "] * 6, [SamplingParams(max_tokens=40, fsm=fsm, seed=s) for s in range(6)])
    assert {o.text for o in outs} <= {'{"answer": "yes"}', '{"answer": "no"}'}
    assert all(o.finish_reason == "finished_stopped" for o in outs)   # the FSM only allows EOS at the end
