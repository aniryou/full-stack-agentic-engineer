"""The tiny task (pure Python): demonstrations, mixes, the parser and the verifier."""
import random

import pytest

from distillab.tinylm.task import (EOS, END_THINK, THINK, SumTask, data_mix, from_text, full_mix, parse, render,
                                   teacher_mix, verify)


def test_mixes_are_distributions_over_scratchpad_lengths():
    for mix in (teacher_mix(6), data_mix(6), full_mix(6), teacher_mix(4, 0.5, 0.2)):
        assert sum(mix) == pytest.approx(1.0) and min(mix) >= 0
    assert teacher_mix(6)[-1] == 0.7 and data_mix(6)[0] == pytest.approx(0.8)


def test_demonstrations_are_correct_and_parse():
    task = SumTask(6, 5)
    rng = random.Random(0)
    for p, c in task.demos(200, rng, teacher_mix(6)):
        r = parse(c)
        assert verify(p, c) and r["ok"] and r["answer"] == p.answer and len(c) == r["scratch"] + 4
        assert c[1:1 + r["scratch"]] == p.running_sums()[: r["scratch"]]
    with pytest.raises(ValueError):
        task.demos(1, rng, [1.0])


def test_verifier_rejects_broken_formats_and_wrong_answers():
    task = SumTask(6, 5)
    p = task.sample(random.Random(1))
    assert not verify(p, [THINK, END_THINK, (p.answer + 1) % 5, EOS])
    assert not verify(p, [THINK, 3, p.answer, EOS])              # never closed the scratchpad
    assert not verify(p, [END_THINK, p.answer, EOS])
    assert verify(p, task.demo(p, 0) + [14, 14])                 # padding after <eos> is ignored


def test_render_and_from_text_round_trip():
    task = SumTask(6, 5)
    p = task.sample(random.Random(2))
    c = task.demo(p, 6)
    assert from_text(render(p.prompt + c)) == p.prompt + c
    assert from_text("<think> 7 </think>") is None               # 7 is not a base-5 digit
