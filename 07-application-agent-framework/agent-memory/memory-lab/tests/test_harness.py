"""The planted-facts harness: the generator, the grader, Wilson intervals, and the numbers the notebooks narrate."""
import pytest

from memlab import harness as H


@pytest.fixture(scope="module")
def ds():
    return H.generate(7)


@pytest.fixture(scope="module")
def modes(ds):
    return H.compare_modes(ds, budget_tokens=128)


def test_generator_is_seeded_and_shaped(ds):
    again = H.generate(7)
    assert [q.__dict__ for q in again.questions] == [q.__dict__ for q in ds.questions]
    assert H.generate(8).questions[0].answer != ds.questions[0].answer or H.generate(8).haystacks != ds.haystacks
    assert ds.counts() == {"extraction": 36, "preference": 6, "multi-session": 6, "temporal": 6,
                           "knowledge-update": 12, "abstention": 12, "adversarial": 6}
    assert all(q.id.endswith("_abs") == (q.category == "abstention") for q in ds.questions)
    assert sum(q.paraphrase for q in ds.questions) == 18
    assert all(q.stale for q in ds.questions if q.category == "knowledge-update")
    assert set(H.SHAPES) == set(H.CATEGORIES)


def test_grade():
    q = H.Question("x", "u", "knowledge-update", "Which city?", "Porto", "Lisbon")
    assert [H.grade(q, a) for a in ("Porto.", "Lisbon.", "I don't know.", "Mars.")] == \
        ["correct", "stale", "abstained", "wrong"]
    u = H.Question("y_abs", "u", "abstention", "Allergies?", None)
    assert H.grade(u, "Not mentioned anywhere.") == "correct" and H.grade(u, "Peanuts.") == "hallucinated"


def test_wilson_interval_pins():
    assert tuple(round(x, 4) for x in H.wilson_interval(45, 50)) == (0.7864, 0.9565)
    assert tuple(round(x, 4) for x in H.wilson_interval(0, 20)) == (0.0, 0.1611)
    assert tuple(round(x, 4) for x in H.wilson_interval(20, 20)) == (0.8389, 1.0)
    assert H.wilson_interval(0, 0) == (0.0, 1.0)


def test_mode_comparison_numbers(modes):
    got = {m: r.rate() for m, r in modes.items()}
    assert got == {"none": (18, 84), "full_history": (84, 84), "tools": (66, 84), "implicit": (66, 84),
                   "pinned": (84, 84)}
    imp = modes["implicit"]
    assert imp.rate("correct", "preference") == (0, 6) and imp.rate("correct", paraphrase=True) == (6, 18)
    assert modes["tools"].rate("correct", "preference") == (0, 6) and modes["pinned"].rate("correct", "preference") == (6, 6)
    assert modes["pinned"].mean("model_calls") > modes["implicit"].mean("model_calls") == 1.0
    assert modes["full_history"].mean("input_tokens") > 3 * modes["implicit"].mean("input_tokens")
    assert "95% Wilson" in H.summary(modes) and "paraphrase subset" in imp.table()


def test_recall_vs_budget_and_the_knee(ds):
    curve = H.recall_vs_budget(ds, write="episodes")
    recalls = [p["recall"] for p in curve]
    assert recalls == sorted(recalls) and curve[0]["memory_tokens"] == 0
    assert H.knee(curve)["budget"] == 128 and round(H.knee(curve)["recall"], 3) == 0.636


def test_consolidation_helps_raw_episodes_and_can_resurface_a_stale_value(ds):
    raw = H.run(ds, "implicit", write="episodes", budget_tokens=64)
    con = H.run(ds, "implicit", write="episodes", consolidate=True, budget_tokens=64)
    facts_only = H.run(ds, "implicit", write="episodes", consolidate=True, budget_tokens=64, kinds=("semantic",))
    assert raw.rate() == (57, 84) and con.rate() == (68, 84) and con.rate("stale") == (1, 84)
    assert facts_only.rate("stale") == (0, 84)


def test_smaller_profile_costs_accuracy(ds):
    assert H.run(ds, "pinned", profile_items=2).rate()[0] < H.run(ds, "pinned", profile_items=6).rate()[0]
