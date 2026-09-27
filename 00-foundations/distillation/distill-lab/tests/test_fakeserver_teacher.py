"""The fake teacher speaks vLLM's protocol (simulated), and the teacher-data pipeline runs against it."""
import json
import math
import urllib.request

import pytest

from distillab import data as D
from distillab import metrics as M
from distillab import teacher as TE
from distillab.client import Client, render_chatml
from distillab.fakeserver import FakeTeacher, score_text, tokens


@pytest.fixture(scope="module")
def servers():
    t, s, th = FakeTeacher("teacher"), FakeTeacher("student"), FakeTeacher("thinker")
    urls = t.start(), s.start(), th.start()
    yield {"teacher": Client(urls[0], t.model), "student": Client(urls[1], s.model), "thinker": Client(urls[2], th.model),
           "urls": urls}
    for x in (t, s, th):
        x.stop()


def test_version_says_simulated(servers):
    v = json.loads(urllib.request.urlopen(servers["urls"][0] + "/version").read())
    assert v["simulated"] is True


def test_chat_samples_logprobs_and_the_max_logprobs_limit(servers):
    p = D.make_set(4, seed=0)[3]
    c = servers["teacher"].chat(p.messages(), n=3, temperature=0.8, top_logprobs=2, seed=5)
    assert len(c) == 3 and all(x.ok and x.content for x in c)
    assert all(len(x.logprobs) == len(tokens(x.content)) and len(x.top_logprobs[0]) == 2 for x in c)
    bad = servers["teacher"].chat(p.messages(), top_logprobs=21)[0]
    assert "greater than max allowed: 20" in bad.error
    cut = servers["teacher"].chat(p.messages(), max_tokens=3)[0]
    assert cut.finish_reason == "length" and cut.completion_tokens == 3


def _sample(i, correct, finish, content="<answer>3</answer>"):
    return TE.Sample(f"p{i}", "sum", 1, "q", "prompt", content, None, correct, finish, 10, 20, None, i)


def test_a_cut_off_answer_is_not_data_even_when_it_is_right():
    """Rejection sampling drops a truncated sample whatever the verifier said; so does a draft's `keep="all"`."""
    ok, cut, wrong = _sample(0, True, "stop"), _sample(1, True, "length", "<answer>3</answer> and then"), _sample(2, False, "stop", "x")
    assert TE.filter_verified([ok, cut, wrong]) == [ok]
    kept, rows = TE.funnel([ok, cut, wrong])
    assert kept == [ok] and [r["samples"] for r in rows][:3] == [3, 1, 1]
    every, rows_all = TE.funnel([ok, cut, wrong], keep="all")
    assert every == [ok, wrong] and rows_all[1]["samples"] == 2
    with pytest.raises(ValueError):
        TE.funnel([ok], keep="some")


def test_the_fake_teachers_truncated_samples_never_reach_the_data(servers):
    probs = D.make_set(12, seed=2, split="train")
    samples = TE.generate(servers["teacher"], probs, n=2, temperature=1.0, max_tokens=12)
    cut = [s for s in samples if s.finish_reason == "length"]
    assert cut and not any(s in TE.funnel(samples)[0] or s in TE.funnel(samples, keep="all")[0] for s in cut)


def test_temperature_zero_repeats_itself_and_dedup_sees_it(servers):
    probs = D.make_set(10, seed=1, split="train")
    greedy = TE.generate(servers["teacher"], probs, n=4, temperature=0.0)
    assert len(TE.dedup(greedy)) == len(probs)                         # identical samples collapse to one each
    hot = TE.generate(servers["teacher"], probs, n=4, temperature=1.0)
    assert len(TE.dedup(hot)) > len(probs)


def test_pipeline_funnel_bill_and_trl_rows(servers, tmp_path):
    train, _ = D.decontaminate(D.make_set(60, seed=0, split="train"), D.make_set(60, seed=0))
    samples = TE.generate(servers["teacher"], train, n=4, temperature=1.0, max_tokens=256)
    kept, rows = TE.funnel(samples, max_completion_tokens=40)
    counts = [r["samples"] for r in rows]
    assert counts[0] == 4 * len(train) and counts[0] >= counts[1] >= counts[2] >= counts[3]
    assert all(s.correct and s.completion_tokens <= 40 for s in kept)
    assert any(not s.correct for s in samples)                         # the verifier has work to do
    bill = TE.token_bill(samples)
    assert bill["completion_tokens"] == sum(s.completion_tokens for s in samples)
    for fmt in (TE.to_messages(kept), TE.to_prompt_completion(kept)):
        assert TE.check_rows(fmt) == []
        path = TE.write_jsonl(fmt, tmp_path / "rows.jsonl")
        assert TE.read_jsonl(path) == fmt
    assert TE.check_rows([{"messages": [{"role": "user", "content": "q"}]}, {"text": "x"}]) != []


def test_thinking_teacher_fills_reasoning_and_usage(servers):
    p = D.make_set(8, seed=2)[7]
    c = servers["thinker"].chat(p.messages(), temperature=0.6, max_tokens=4096)[0]
    assert c.reasoning and c.content and c.reasoning_tokens == len(tokens(c.reasoning.strip()))
    msg = TE.assistant_message(TE.Sample(p.id, p.kind, p.difficulty, p.question, p.prompt, c.content, c.reasoning,
                                         True, "stop", 0, 0, 0), "inline")
    assert msg["content"].startswith("<think>\n") and "</think>\n\n" in msg["content"]


def test_scoring_a_students_text_gives_the_on_policy_reward(servers):
    """prompt_logprobs of the student's own text: high while it follows the teacher, a sharp drop at the slip."""
    probs = [p for p in D.make_set(60, seed=3) if p.kind == "arith" and p.difficulty >= 3]
    seen_slip = seen_clean = False
    for i, p in enumerate(probs * 3):
        s = servers["student"].chat(p.messages(), temperature=1.0, seed=i, top_logprobs=0)[0]
        scored = servers["teacher"].score(p.messages(), s.content)
        assert [t for t, _ in scored] == tokens(s.content)             # aligned token by token (shared tokenizer)
        assert scored == pytest.approx([(t, v) for t, v in zip(tokens(s.content), score_text(
            {"on_path": 0.9}, p.prompt, s.content))])
        rewards = [lt - ls for (_, lt), (_, ls) in zip(scored, s.logprobs)]
        if D.verify(p, s.content):
            seen_clean = True
            assert min(v for _, v in scored) > math.log(0.1)
        else:
            seen_slip = True
            assert min(rewards) < -4
    assert seen_slip and seen_clean


def test_metrics_count_generated_tokens(servers):
    url = servers["urls"][0]
    before = M.scrape(url)
    servers["teacher"].chat(D.make_set(1)[0].messages(), n=2)
    after = M.scrape(url)
    assert after.value(M.GENERATION_TOKENS) > before.value(M.GENERATION_TOKENS)
    assert after.value(M.REQUEST_SUCCESS) - before.value(M.REQUEST_SUCCESS) == 2


def test_chatml_rendering():
    msgs = [{"role": "user", "content": "hi"}]
    assert render_chatml(msgs).startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
    assert render_chatml(msgs, default_system=None) == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert TE.topk_target([("a", math.log(0.5)), ("b", math.log(0.3))]) == (pytest.approx({"a": 0.625, "b": 0.375}),
                                                                             pytest.approx(0.2))
