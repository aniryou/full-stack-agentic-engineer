"""The generated eval set: deterministic, every answer checkable by the solver, difficulty recoverable."""
from thinklab.thinking import evalset as E


def test_deterministic_and_balanced():
    a, b = E.make_evalset(40, seed=3), E.make_evalset(40, seed=3)
    assert a == b and a != E.make_evalset(40, seed=4)
    assert {p.kind for p in a} == set(E.KINDS) and {p.difficulty for p in a} == {1, 2, 3, 4}


def test_solver_and_difficulty_agree_with_the_generator():
    for p in E.make_evalset(500, seed=11):
        assert E.solve(p.question) == p.answer, p
        assert E.difficulty_of(p.question) == p.difficulty, p


def test_verify_reads_content_only():
    p = E.make_evalset(1)[0]
    assert E.verify(p, f"blah \\boxed{{{p.answer}}}") and not E.verify(p, "no box") and not E.verify(p, None)
    assert p.prompt.endswith(E.SUFFIX) and p.messages()[0]["role"] == "user"


def test_hand_checked_questions():
    assert E.solve("Today is Tuesday. What day of the week will it be in 100 days?") == "thursday"
    assert E.solve("What is (23 * 98 - 875) * 4?") == "5516"
    assert E.solve("Ann is taller than Ben. Cal is shorter than Ben. Who is the shortest?") == "cal"
    assert E.solve('How many times does the letter \'r\' appear in "strawberry"?') == "3"
