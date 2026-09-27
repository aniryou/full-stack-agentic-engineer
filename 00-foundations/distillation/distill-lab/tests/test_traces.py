"""Reasoning traces: the bundled file is what the fake thinking teacher produces, and the filters behave."""
import pytest

from distillab import data as D
from distillab import teacher as TE
from distillab import traces as TR
from distillab.client import Client
from distillab.fakeserver import FakeTeacher


def test_bundled_traces_are_current():
    with FakeTeacher("thinker") as url:
        fresh = TR.collect(Client(url, "Qwen/Qwen3-1.7B"), D.make_set(40, seed=0, split="train"), n=4, seed=0)
    assert [t.to_dict() for t in fresh] == [t.to_dict() for t in TR.load_bundled()], \
        "regenerate distillab/assets/samples/traces_thinker_illustrative.jsonl"


def test_filters_and_the_length_trade():
    t = TR.load_bundled()
    kept = TR.filter_traces(t)
    assert kept and all(x.correct for x in kept) and len(kept) <= sum(x.correct for x in t)
    rows = TR.trade(t, [None, 200, 100, 60])
    shares = [r["share of correct"] for r in rows]
    hard = [r["hardest covered"] for r in rows]
    assert shares == sorted(shares, reverse=True) and hard == sorted(hard, reverse=True)
    assert rows[0]["share of correct"] == 1.0 and hard[2] < rows[2]["problems covered"]   # a cap starves the hard ones first
    st = TR.length_stats(t)
    assert st["p50"] <= st["p90"] <= st["p99"] <= st["max"]


def test_sft_rows_and_the_bill():
    t = TR.filter_traces(TR.load_bundled())[:5]
    rows = TR.to_sft_rows(t)
    assert TE.check_rows(rows) == [] and all("reasoning_content" in r["messages"][1] for r in rows)
    inline = TR.to_sft_rows(t, "inline")
    assert all(r["messages"][1]["content"].startswith("<think>") for r in inline)
    bill = TR.reasoning_bill(t, 9.0)
    assert bill["cost $"] == pytest.approx(bill["output tokens"] * 9 / 1e6, abs=1e-4)


def test_compare_measures_each_served_model_the_same_way():
    with FakeTeacher("thinker") as t_url, FakeTeacher("student") as s_url:
        rows = TR.compare({"teacher": Client(t_url, "t"), "student": Client(s_url, "s")}, D.make_set(20, seed=4), n=2)
    assert [r["model"] for r in rows] == ["teacher", "student"] and all(r["traces"] == 40 for r in rows)
    assert all(0 <= r["accuracy"] <= 1 and r["mean reasoning"] > 0 and "-" in r["95% CI"] for r in rows)
