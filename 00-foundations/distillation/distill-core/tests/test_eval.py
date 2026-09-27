"""Measuring a student: agreement, accuracy with intervals, the gap by slice (primer §8)."""
import numpy as np

from distillcore import TinyLM, eval as E, losses as L, seqkd, train
from distillcore.tinylm import fit_language


def test_agreement_metrics_by_hand():
    ref = np.array([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    test = np.array([[2.0, 3.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    assert E.argmax_agreement(ref, test) == 0.5
    assert E.topk_overlap(ref, test, 2) == 1.0 and E.topk_overlap(ref, test, 1) == 0.5
    assert abs(E.kl(ref, test) - float(L.kl(L.softmax(ref), L.softmax(test)).mean())) < 1e-12
    assert E.kl(ref, ref) == 0


def test_wilson_intervals():
    assert tuple(round(x, 4) for x in E.wilson_interval(30, 60)) == (0.3773, 0.6227)
    assert tuple(round(x, 4) for x in E.wilson_interval(170, 200)) == (0.7939, 0.8929)
    assert E.wilson_interval(0, 20)[0] == 0 and round(E.wilson_interval(0, 20)[1], 4) == 0.1611
    assert E.wilson_interval(0, 0) == (0.0, 1.0)


def test_compare_counts_the_flips_and_the_gap_sorts_worst_first():
    t, s = np.array([1, 1, 1, 0, 1, 0], bool), np.array([1, 0, 1, 1, 0, 0], bool)
    c = E.compare(t, s)
    assert (c["lost"], c["gained"]) == (2, 1) and abs(c["paired_z"] - (1 - 2) / np.sqrt(3)) < 1e-12
    rows = E.capability_gap({"easy": (t, t), "hard": (t, s)})
    assert [r["slice"] for r in rows] == ["hard", "easy"] and rows[1]["gap"] == 0


def test_the_student_gap_hides_in_rare_inputs_and_long_outputs(lang, teacher, bench):
    C, prompts = lang.contexts(), bench["prompts"]
    common = set(map(tuple, prompts))
    rare = np.array([c for c in C if tuple(c) not in common])
    small = fit_language(lang, 8)
    ok = lambda m, ctx, n: lang.verify(m.sample(ctx, n, None, 0.0))
    rows = {r["slice"]: r for r in E.capability_gap({f"{k} n={n}": (ok(teacher, x, n), ok(small, x, n))
                                                       for k, x in (("common", prompts), ("rare", rare)) for n in (2, 8)})}
    assert rows["common n=2"]["student"] == 1.0 and rows["rare n=8"]["student"] < rows["rare n=2"]["student"] < 1
    assert round(rows["rare n=8"]["student"], 3) == 0.525 and rows["rare n=8"]["n"] == 101


def test_a_student_can_beat_its_teacher_while_agreeing_less(lang):
    """A weak teacher (605 hard labels, H = 64) is right on 95.0% of contexts; a student trained on its samples
    that pass the verifier is right on 97.5%, and agrees with it less than a student of unfiltered samples."""
    C = lang.contexts()
    ctx = C[np.random.default_rng(7).integers(0, 121, 605)]
    y = lang.sample_next(ctx, np.random.default_rng(8))
    weak = TinyLM(11, 64, 8, seed=2)
    train(weak, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
    data = seqkd.teacher_data(weak, C, 20, 1, np.random.default_rng(9))
    filt, raw = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
    seqkd.sft(filt, seqkd.keep_verified(lang, data), 600)
    seqkd.sft(raw, data, 600)
    acc = {k: E.vs_truth(m, lang)["rule_acc"] for k, m in (("teacher", weak), ("filtered", filt), ("raw", raw))}
    assert acc["filtered"] > acc["teacher"] > acc["raw"] and round(acc["teacher"], 3) == 0.950
    zt = weak.logits(C)
    assert E.kl(zt, filt.logits(C)) > 5 * E.kl(zt, raw.logits(C))
    assert E.argmax_agreement(zt, filt.logits(C)) < E.argmax_agreement(zt, raw.logits(C))
