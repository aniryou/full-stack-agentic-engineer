"""A distilled draft for speculative decoding: acceptance is the metric (primer §7).

The drafts' α and speedups come from trained toy models: the primer's numbers are one seeded run on one CPU, and
another CPU family's BLAS and SIMD kernels move them (0.988 ran 0.971–0.990 across 15 CPU variants), so they are
compared with a tolerance of about three times the largest deviation seen. The orderings and the cap stay exact."""
import math

import numpy as np
import pytest

from distillcore import ModLang, TinyLM, draft as Dr, losses as L, seqkd, train
from distillcore.tinylm import fit_language

P, Q = np.array([0.5, 0.3, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.4])


def test_acceptance_is_one_minus_tv_and_greedy_drafting_changes_it():
    assert math.isclose(Dr.acceptance_rate(P, Q), 0.6) and math.isclose(1 - 0.5 * np.abs(P - Q).sum(), 0.6)
    assert Dr.greedy_acceptance(P, Q) == 0.05                      # argmax q = token 3, p(3) = 0.05


def test_what_vllms_counters_show_for_alpha_06_k4():
    v = Dr.vllm_view(0.6, 4)
    assert round(v["mean_acceptance_length"], 4) == 2.3056 and round(v["draft_acceptance_rate"], 4) == 0.3264
    assert np.allclose(v["per_position"], [0.6, 0.36, 0.216, 0.1296])
    assert Dr.best_k(0.6, 0.1) == 3 and round(Dr.speedup(0.6, 3, 0.1), 4) == 1.6738


@pytest.fixture(scope="module")
def drafts():
    """A fine-tuned target (its noise all on +1), its text, and three drafts of width 16."""
    base, tuned = ModLang(11, 0.2), ModLang(11, 0.2, skew=1.0)
    C = base.contexts()
    target = fit_language(tuned, 64)
    text = target.sample(C[np.random.default_rng(1).integers(0, 121, 500)], 12, np.random.default_rng(2))
    big = target.sample(C[np.random.default_rng(3).integers(0, 121, 2000)], 12, np.random.default_rng(4))
    PT = target.probs(C)
    kd = TinyLM(11, 16, 8, seed=3)
    train(kd, C, lambda z, i: L.soft_ce(z, PT[i]), 1500)
    sk = TinyLM(11, 16, 8, seed=3)
    seqkd.sft(sk, big, 1500, batch=512)
    return target, text, {"off-the-shelf": fit_language(base, 16, seed=3), "kd": kd, "seqkd": sk}


def test_a_draft_distilled_from_the_target_accepts_more(drafts):
    target, text, d = drafts
    a = {k: Dr.acceptance_on_text(target, m, text) for k, m in d.items()}
    assert a["off-the-shelf"]["alpha"] == pytest.approx(0.890, abs=0.008)
    assert a["kd"]["alpha"] == pytest.approx(0.988, abs=0.06)
    assert a["off-the-shelf"]["alpha"] < a["seqkd"]["alpha"] < a["kd"]["alpha"]
    top1 = Dr.acceptance_on_text(target, target, text)["greedy"]  # a perfect draft: the target's own top-1, ≈ 0.8
    assert top1 == pytest.approx(0.8, abs=0.0025)
    for v in a.values():
        assert abs(v["alpha"] + v["tv"] - 1) < 1e-12 and v["kl"] >= 0
        assert v["greedy"] <= top1 + 1e-12                         # greedy drafting is capped by the target's top-1


def test_speedup_prefers_the_distilled_draft(drafts):
    target, text, d = drafts
    c = Dr.draft_cost(d["kd"].n_params, target.n_params)
    assert round(c, 3) == 0.289                                   # parameter counts: exact
    s = {k: Dr.speedup(Dr.acceptance_on_text(target, m, text)["alpha"], 4, c) for k, m in d.items()}
    assert s["off-the-shelf"] == pytest.approx(1.86, abs=0.025) and s["kd"] == pytest.approx(2.26, abs=0.25)
