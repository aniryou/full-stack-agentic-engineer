"""The generated problems: deterministic, verifiable, with canonical scratchpads the verifier accepts, and a
training split kept apart from the eval split."""
import random

import pytest

from distillab import data as D


def test_sets_are_deterministic_and_cycle_through_kinds_and_difficulties():
    a, b = D.make_set(40, seed=3), D.make_set(40, seed=3)
    assert a == b and len({(p.kind, p.difficulty) for p in a}) == 20
    assert D.make_set(40, seed=3, split="train") != a


@pytest.mark.parametrize("p", D.make_set(100, seed=1), ids=lambda p: p.id)
def test_canonical_scratchpad_passes_the_verifier_and_the_question_is_recoverable(p):
    assert D.verify(p, D.scratchpad(p))
    f = D.find("some preamble " + p.prompt)
    assert f is not None and (f.kind, f.difficulty, f.answer) == (p.kind, p.difficulty, p.answer)


def test_verifier_reads_the_last_boxed_answer_of_the_content():
    p = D.make_set(1, seed=0)[0]
    assert D.extract_answer("first \\boxed{1} then \\boxed{ 42 }") == "42"
    assert D.extract_answer("Answer: Friday.") == "friday" and D.extract_answer(None) is None
    assert not D.verify(p, "no answer here") and not D.verify(p, None)


def test_decontaminate_drops_eval_questions_from_training():
    train, ev = D.make_set(500, seed=0, split="train"), D.make_set(100, seed=0)
    kept, dropped = D.decontaminate(train, ev)
    assert dropped and len(kept) + len(dropped) == 500
    assert not {p.question for p in kept} & {p.question for p in ev}


def test_modsum_difficulty_two_is_the_tiny_task():
    rng = random.Random(0)
    p = D.make_problem("modsum", 2, rng)
    digits = [int(x) for x in p.question.split("(")[1].split(")")[0].split(" + ")]
    assert len(digits) == 6 and all(0 <= d < 5 for d in digits) and p.answer == str(sum(digits) % 5)
