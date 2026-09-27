"""The fake server counts prefix-cache hits the way vLLM does; the layout bench predicts them exactly (simulated)."""
import json
import math
import urllib.error
import urllib.request

import numpy as np
import pytest

from memlab import cachebench as cb
from memlab.embedders import HashingEmbedder, OpenAIEmbeddings
from memlab.fakeserver import FakeLLMServer, PrefixCache, block_hashes, hit_rate, render_chat, scrape, tokenize
from memlab.llm import ChatClient


@pytest.fixture(scope="module")
def server():
    s = FakeLLMServer(enable_prompt_tokens_details=True)
    s.start()
    yield s
    s.stop()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_expected_cached_tokens_pins():
    assert cb.expected_cached_tokens(list(range(64)), list(range(64)), 16) == 48
    assert cb.expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30, 16) == 32
    assert cb.expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8) == 16


def test_prefix_cache_rules_and_salt_on_the_first_block_only():
    c = PrefixCache()
    toks = list(range(64))
    assert c.lookup(toks) == 0
    c.insert(toks)
    assert c.lookup(toks) == 48 and c.lookup(toks + [99] * 20) == 48      # prev left (64 - 1) // 16 = 3 blocks
    assert c.lookup(toks, salt="tenant-a") == 0
    a, b = block_hashes(toks, "s1"), block_hashes(toks, "s2")
    assert all(x != y for x, y in zip(a, b))                              # the salt changes every block by chaining
    assert c.reset() == 3 and c.lookup(toks) == 0


def test_cached_tokens_only_with_the_flag():
    msgs = [{"role": "system", "content": "S" * 400}, {"role": "user", "content": "hi"}]
    with FakeLLMServer() as url:
        _, first = post(url + "/v1/chat/completions", {"messages": msgs})
        _, again = post(url + "/v1/chat/completions", {"messages": msgs})
        assert "prompt_tokens_details" not in again["usage"]
        m = scrape(url)
        assert m["vllm:prefix_cache_hits_total"] > 0 and m["vllm:prefix_cache_queries_total"] == 2 * first["usage"]["prompt_tokens"]


def test_cache_salt_validation(server):
    msgs = [{"role": "user", "content": "hi"}]
    assert post(server.url + "/v1/chat/completions", {"messages": msgs, "cache_salt": "x" * 129})[0] == 400
    assert post(server.url + "/v1/chat/completions", {"messages": msgs, "cache_salt": "a/b"})[0] == 400
    assert post(server.url + "/v1/chat/completions", {"messages": msgs, "cache_salt": "x" * 128})[0] == 200


def test_tool_calls_round_trip(server):
    tools = [{"name": "remember", "description": "Store a fact.", "parameters": {"type": "object",
              "properties": {"text": {"type": "string"}}, "required": ["text"]}}]
    r = ChatClient(server.url).generate([{"role": "user", "content": "I live in Porto."}], tools)
    assert r.tool_calls[0].name == "remember" and r.tool_calls[0].args["value"] == "Porto"
    assert r.usage["cached_tokens"] is not None


def test_embeddings_endpoint_matches_the_hashing_embedder(server):
    assert OpenAIEmbeddings(server.url).model_name == "memlab/hashing-1024"      # discovered among two models
    e = OpenAIEmbeddings(server.url, "memlab/hashing-1024")
    got = e.encode(["user lives in lisbon", "a dog"])
    assert got.shape == (2, 1024) and np.allclose(got, HashingEmbedder().encode(["user lives in lisbon", "a dog"]))


def test_template_is_stable_and_tokens_are_len_over_4():
    msgs = [{"role": "system", "content": "abc"}, {"role": "user", "content": "hello"}]
    text = render_chat(msgs)
    assert text.endswith("<|im_start|>assistant\n") and render_chat(msgs) == text
    assert len(tokenize(text)) == math.ceil(len(text) / 4)


def test_step_cost_and_cost_pins():
    L4, H100, q, l8 = cb.GPUS["L4"], cb.GPUS["H100-SXM"], cb.LLMS["qwen2.5-1.5b"], cb.LLMS["llama-3.1-8b"]
    pins = [(L4, q, 78.7, 15.1), (L4, l8, 401.0, 65.6), (H100, q, 11.4, 3.2), (H100, l8, 50.8, 7.7)]
    for gpu, llm, cold, warm in pins:
        assert round(cb.step_cost(gpu, llm, [(0, 2000)])["t"] * 1e3, 1) == cold
        assert round(cb.step_cost(gpu, llm, [(1800, 200)])["t"] * 1e3, 1) == warm
    assert cb.step_cost(L4, q, [(0, 2000)])["bound"] == "compute" and cb.step_cost(L4, q, [(1800, 200)])["bound"] == "memory"
    assert math.isclose(cb.turn_cost(cb.PRICES["gemini-3.5-flash"], 5000, 350, 2700), 0.007005)


def test_layouts_predicted_equals_simulated_and_the_ordering_holds(server):
    runs = {l: cb.run_layout(server.url, l, turns=8) for l in cb.LAYOUTS + cb.VARIANTS}
    for r in runs.values():
        assert r.cached_source.startswith("SIMULATED") and r.timing.startswith("SIMULATED (roofline, L4")
        assert all(row.cached_tokens == row.predicted_cached for row in r.rows), r.layout
    rate = {k: round(r.hit_rate, 3) for k, r in runs.items()}
    assert rate == {"before_history": 0.38, "pinned": 0.899, "tail": 0.674, "tail_after": 0.77}, rate
    assert runs["tail"].prefill_ms < runs["before_history"].prefill_ms * 0.6
    # every prompt of the implicit layouts is under the provider's 4,096-token minimum: nothing is billed cached, so
    # on the bill the layouts differ only by their prompt lengths - the saving is the engine's, in prefill time
    assert max(r.prompt_tokens for k in ("before_history", "tail", "tail_after") for r in runs[k].rows) < 4096
    assert runs["tail_after"].cost_usd == runs["tail"].cost_usd
    assert abs(runs["before_history"].cost_usd / runs["tail"].cost_usd - 1) < 0.01
    every_hit = cb.Price(1.5, 9.0, 0.15)                                  # a provider with no minimum
    cost = {k: sum(cb.turn_cost(every_hit, r.prompt_tokens, 60, r.cached_tokens or 0) for r in runs[k].rows)
            for k in ("before_history", "tail", "tail_after")}
    assert cost["tail_after"] < cost["tail"] < cost["before_history"]


def test_a_real_target_labels_each_column_by_what_it_is():
    """At T1 only the cached tokens are measured: the time is a roofline model and the dollars a price table."""
    run = cb.LayoutRun("tail", "MEASURED on http://gpu-box:8000", timing="SIMULATED (roofline, T4 + qwen2.5-1.5b)",
                       dollars="gemini-3.5-flash list prices (verify)")
    assert run.source.startswith("cached tokens MEASURED") and "prefill ms SIMULATED" in run.source
    text = cb.summary({"tail": run})
    assert "hit rate    : cached tokens MEASURED" in text and "prefill ms  : SIMULATED (roofline, T4" in text
    assert "MEASURED" not in text.split("prefill ms  :")[1]


def test_reset_prefix_cache_answers_like_vllm(server):
    """vLLM v0.30.0's dev endpoint clears the whole cache and answers only {"success": bool}: no count, no salt."""
    cb.run_layout(server.url, "pinned", turns=2)
    req = urllib.request.Request(server.url + "/reset_prefix_cache", data=b"", method="POST")
    assert json.loads(urllib.request.urlopen(req).read()) == {"success": True}
    assert len(server.cache.blocks) == 0


def test_metrics_window_equals_per_request_usage(server):
    before = scrape(server.url)
    run = cb.run_layout(server.url, "pinned", turns=3)
    assert math.isclose(hit_rate(before, scrape(server.url)), run.hit_rate)
    text = urllib.request.urlopen(server.url + "/metrics").read().decode()
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total",
                 "vllm:time_to_first_token_seconds_bucket", "vllm:cache_config_info"):
        assert name in text
    assert json.loads(urllib.request.urlopen(server.url + "/version").read())["simulated"] is True
