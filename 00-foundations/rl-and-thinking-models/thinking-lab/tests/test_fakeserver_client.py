"""The fake vLLM end to end through the client: fields, switches, budgets, the max_tokens trap, streaming,
n > 1, the two-call recipe, cached tokens across turns, errors and vLLM-named metrics."""
import json
import urllib.error
import urllib.request

import pytest

from thinklab import metrics as M
from thinklab.fakeserver import FakeServer
from thinklab.thinking import budget as B
from thinklab.thinking.client import ThinkingClient, parse_response, request_body
from thinklab.thinking.evalset import make_evalset
from thinklab.workload import run_open_loop

PROBS = make_evalset(20, seed=0)
HARD = PROBS[15]                                      # difficulty 4


@pytest.fixture(scope="module")
def server():
    with FakeServer(time_scale=0.001) as url:
        yield url


@pytest.fixture(scope="module")
def client(server):
    return ThinkingClient(server)


def test_version_models_and_health(server):
    assert json.loads(urllib.request.urlopen(server + "/version").read())["simulated"] is True
    assert urllib.request.urlopen(server + "/health").status == 200


def test_thinking_on_and_off(client):
    on = client.chat(HARD.messages(), seed=1)
    off = client.chat(HARD.messages(), thinking=False, seed=1)
    assert on.reasoning and on.reasoning_tokens > 0 and on.completion_tokens > on.reasoning_tokens
    assert off.reasoning is None and off.reasoning_tokens == 0 and off.content
    assert client.chat(HARD.messages(), reasoning_effort="none", recommended=False).reasoning is None
    soft = [{"role": "user", "content": HARD.prompt + " /no_think"}]
    assert client.chat(soft).reasoning is None


def test_max_tokens_trap_and_budgets(client):
    cut = client.chat(HARD.messages(), max_tokens=32, seed=2)
    assert B.classify(cut) == "cut_in_thinking" and cut.content is None and cut.finish_reason == "length"
    nat = B.native(client, HARD.messages(), 32, 3000, seed=2)
    assert B.classify(nat) in ("answered", "no_answer") and nat.reasoning_tokens <= 32 + 25
    two = B.two_call(client, HARD.messages(), 32, 3000, seed=2)
    assert two.content and two.reasoning_tokens <= 32


def test_streaming_matches_non_streaming(client):
    a = client.chat(PROBS[3].messages(), seed=7)
    b = client.chat(PROBS[3].messages(), seed=7, stream=True)
    assert (a.reasoning, a.content, a.finish_reason) == (b.reasoning, b.content, b.finish_reason)
    assert b.reasoning_tokens == a.reasoning_tokens and b.ttft <= b.ttfc and len(b.itl) > 0


def test_reasoning_content_spelling():
    with FakeServer(time_scale=0.001, reasoning_field="reasoning_content") as url:
        r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(
            request_body(PROBS[0].messages(), model="Qwen/Qwen3-0.6B")).encode(), headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(r).read())
    msg = data["choices"][0]["message"]
    assert "reasoning" not in msg and msg["reasoning_content"]
    assert parse_response(data)[0].reasoning == msg["reasoning_content"]


def test_n_samples_and_errors(client, server):
    outs = client.chat(PROBS[0].messages(), n=4, seed=3)
    assert len(outs) == 4 and all(o.content for o in outs)
    bad = client.chat(PROBS[0].messages(), max_tokens=100_000)
    assert "HTTP 400" in bad.error and "max_model_len" in bad.error
    assert "HTTP 400" in client.chat(PROBS[0].messages(), budget=-5).error


def test_prefix_cache_across_turns_stops_at_the_header(client):
    sys_msg = {"role": "system", "content": "You are careful. " * 30}
    h = [sys_msg, {"role": "user", "content": PROBS[5].question}]
    c1 = client.chat(h, seed=4)
    c2 = client.chat(h + [{"role": "assistant", "content": c1.content}, {"role": "user", "content": "Thanks, now double it."}], seed=4)
    assert 0 < c2.cached_tokens < c2.prompt_tokens and c2.cached_tokens % 16 == 0
    assert c2.cached_tokens <= c1.prompt_tokens


def test_metrics_use_vllm_names_and_move(server):
    before = M.scrape(server)
    run = run_open_loop(server, [(p.messages(), {"budget": 64}, "t") for p in PROBS[:6]], rate=1000.0)
    after = M.scrape(server)
    d = M.delta(after, before)
    assert run.summary()["completed"] == 6 and run.simulated
    assert M.value(d, M.REQUEST_SUCCESS) == 6 and M.value(d, M.GENERATION_TOKENS) > 6 * 30
    names = {s.name for s in after}
    for n in ("vllm:num_requests_running", "vllm:kv_cache_usage_perc", "vllm:generation_tokens_total",
              "vllm:inter_token_latency_seconds_bucket", "vllm:request_generation_tokens_bucket",
              "vllm:num_preemptions_total", "vllm:prefix_cache_hits_total"):
        assert n in names, n
    assert not any("reason" in n for n in names)          # v0.30.0 has no reasoning-specific metric
