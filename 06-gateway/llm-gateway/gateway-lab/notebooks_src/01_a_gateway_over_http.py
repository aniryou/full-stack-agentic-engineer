# %% [markdown]
# # 01 · A gateway over HTTP: one front door, one API, one ledger row per request
#
# **Tier:** T0. The gateway and two fake providers start in this process and talk over localhost HTTP. Every
# latency and token count that the fakes produce is **simulated**. T0 + Docker: the same three processes in Docker
# Compose ([`deploy/local`](../deploy/local/README.md)). T1: a real `vllm serve` as one of the providers
# ([`deploy/any-gpu`](../deploy/any-gpu/README.md), `GWLAB_VLLM_URL`).
#
# ## The one-minute version
#
# A gateway is the one place that every model call goes through. Thus it owns the things that no single app must own:
#
# - The **provider keys**. Callers get *virtual* keys. The gateway stores only a hash of each key, and each key has a
#   scope of aliases. An operator can revoke a key.
# - The **decision** for each request: if it runs, and which model, provider or pool serves it. Then the 05 router
#   selects the replica, and the engine makes the batches (05 PRIMER §1.3, §3.3, §7).
# - The **meter**: one ledger row for each request, with a price from the `usage` of the provider.
# - The **trace**: GenAI spans.
#
# The common language of the gateway is chat completions over SSE. An adapter table translates the other dialects. The
# bugs are in the data that does not go into the common form. This data is Anthropic's `input_tokens`, which does not
# include the cache, the fragments of tool-call arguments, and the `reasoning` text (PRIMER §1).
#
# Two details make metering work on streams:
#
# 1. The gateway sets `stream_options.include_usage` on every streamed upstream request. Thus it always gets the count
#    of the provider. It never sets it on a non-streamed request, because vLLM answers 400. The gateway also removes
#    that usage chunk from the response to a client that did not ask for it.
# 2. The gateway parses every chunk. But the 05 router sends the bytes on with no change. The gateway owns the bill, thus it
#    must read it.
#
# After this notebook, you can do these tasks:
#
# - Issue a key.
# - Trace one request through the pipeline.
# - Read what went over the wire in the two dialects. You see usage that arrives in pieces, an error after the first
#   byte, and a body that the other dialect accepts.
# - Find the ledger row and the spans of the request.
#
# The concepts are in PRIMER §1 (One front door, one API), §5 (Metering, tracing and chargeback) and §6 (Keys, tenants
# and isolation) ([`PRIMER.md`](../../PRIMER.md)).

# %%
import hashlib, json, os, random, statistics
from gwlab import client, env, sse, t1
from gwlab.gateway import adapters, otel
from gwlab.stack import LocalStack

tiers = env.describe()
stack = LocalStack().start()          # acme: OpenAI dialect, bolt: Anthropic dialect, both SIMULATED; the gateway
print("gateway:", stack.url, "| providers:", {n: f.url for n, f in stack.fakes.items()})

# %% [markdown]
# ## Worked example: a virtual key, and what the gateway stores
#
# An operator issues keys through the admin API. The admin API returns the secret **once**. The gateway stores the
# SHA-256 of the secret (as LiteLLM's `hash_token` does) and a display prefix. The key sets the tenant. A caller can put
# any value in a header, but the gateway bills and limits the tenant of the key.

# %%
key_a = stack.issue_key("team-a")
key_b = stack.issue_key("team-b", aliases=["chat"], budget_usd=0.50)
rows = stack.gateway.store.query("SELECT key_id, key_hash, display, tenant, aliases, budget_usd FROM keys")
for r in rows:
    print(r)
print("secret stored anywhere?", any(key_a in json.dumps(r) for r in rows),
      "| its hash stored?", any(hashlib.sha256(key_a.encode()).hexdigest() == r["key_hash"] for r in rows))

# %% [markdown]
# ## Worked example: one request, non-streamed and streamed
#
# The client gives an **alias** (`chat`), not a model. The gateway resolves the alias to a chain (`acme/fast`, then
# `bolt/haiku`). It calls the first target with the provider key of *that* target. It answers with headers that tell
# which target served the request. The `model` in the body is the upstream model that really answered (OpenAI does the
# same for an alias).

# %%
r = stack.chat(key_a, "What does a gateway own?")
print(r.status, {k: v for k, v in r.headers.items() if k.startswith("x-gwlab") or k == "x-request-id"})
print("model:", r.json["model"], "| usage:", r.json["usage"])
row = stack.ledger(request_id=r.header("x-request-id"))[0]
print({k: row[k] for k in ("tenant", "alias", "target", "prompt_tokens", "completion_tokens", "usage_source", "cost_usd")})

# %% [markdown]
# The next cell sends the request as a stream, and the client does not ask for usage. On the wire, the client sees a
# role chunk, content chunks, a finish chunk and `data: [DONE]`. It sees **no** usage chunk. But the ledger row has the
# counts from the provider itself (`usage_source: provider`). The gateway added `include_usage` to the upstream request
# and removed the chunk from the response. If the client asks for usage (`stream_options.include_usage`), it gets the
# chunk with `choices: []` immediately before `[DONE]`.

# %%
plain = stack.chat(key_a, "Explain streaming in one line.", stream=True)
events = [e for e in plain.raw.decode().split("\n\n") if e]
for e in events[:2] + ["..."] + events[-2:]:
    print(e[:150])
print("ledger:", {k: stack.ledger(request_id=plain.header("x-request-id"))[0][k] for k in ("completion_tokens", "usage_source")})
asked = stack.chat(key_a, "Explain streaming in one line.", stream=True, stream_options={"include_usage": True})
print("with include_usage, last chunk before [DONE]:", json.dumps(asked.chunks[-1])[:170])

# %% [markdown]
# ## Worked example: the other dialect
#
# `team-eu` can use only providers in the `eu` region (residency). Thus its `chat` goes to `bolt`, an
# Anthropic-dialect provider. The next cell shows what bolt itself sends: named events, a cumulative `output_tokens`,
# and no `[DONE]`. It also shows what the gateway changes this into: the same chat chunks as in the previous worked
# example.

# %%
key_eu = stack.issue_key("team-eu")
direct = client.request("POST", stack.fake_url("bolt") + "/v1/messages",
                        {"model": "claude-haiku-4-5", "max_tokens": 6, "stream": True,
                         "messages": [{"role": "user", "content": "hi"}]},
                        {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"})   # the operator's key, not a caller's
print("bolt speaks:", [e.event for e in sse.SSEParser().feed(direct.raw)])
via = stack.chat(key_eu, "hi", stream=True, max_completion_tokens=6)
print("the client gets:", [list((c.get("choices") or [{}])[0].get("delta", {}).keys()) for c in via.chunks], "| done:", via.done)
print("served by", via.header("x-gwlab-target"))

# %% [markdown]
# ## Exercise 1.1 — Anthropic's usage, from the events on the wire
#
# The core wrote the usage table for non-streamed responses (gateway-core notebook 01, exercise 1.1). On a stream, the
# counts arrive in pieces, and the pieces are the trap.
#
# Anthropic's `message_start` has `message.usage` with the input side and an `output_tokens` of 1. The input side is
# `input_tokens` (which **excludes** the cache), `cache_read_input_tokens` and `cache_creation_input_tokens`. The final
# `message_delta` has `usage` again, **cumulative**. It has the full `output_tokens` and, when the API sends it,
# `output_tokens_details.thinking_tokens`. A later value replaces an earlier value. No count is a sum across events.
#
# Write `usage_from_events(events)`. Its input is a list of `(event_name, data_dict)` pairs that you read from the
# wire. Return the canonical `{"prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"}`. The prompt
# count includes the cache reads and the cache writes. The completion count includes the thinking tokens.
#
# If the stream
# ended before its `message_delta`, return `None`. Then the numbers of `message_start` are not a bill, and the gateway
# must make an estimate.
#
# The check reads real streams from `bolt` over HTTP: a warm stream with cache reads, and a stream that the provider
# cuts in the middle. It compares your result with the translator and the ledger of the gateway.

# %%
def bolt_events(content, max_tokens=24):
    """Stream straight from bolt (as the operator, with its key) and parse the SSE events."""
    raw = client.request("POST", stack.fake_url("bolt") + "/v1/messages",
                         {"model": "claude-haiku-4-5", "max_tokens": max_tokens, "stream": True,
                          "system": "You answer billing questions. " * 12, "messages": [{"role": "user", "content": content}]},
                         {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"}).raw
    return [(e.event, e.json()) for e in sse.SSEParser().feed(raw)]

warm = bolt_events("When is my invoice due?")            # the first request fills bolt's prefix cache ...
warm = bolt_events("When is my invoice due?")            # ... the second reads it back
print([name for name, _ in warm])
print("message_start usage:", warm[0][1]["message"]["usage"])
print("message_delta usage:", next(d for n, d in warm if n == "message_delta")["usage"])

# %% exercise
def usage_from_events(events):
    ### BEGIN SOLUTION
    raw, finished = {}, False
    for name, data in events:
        if name == "message_start":
            raw.update((data.get("message") or {}).get("usage") or {})
        elif name == "message_delta":
            raw.update({k: v for k, v in (data.get("usage") or {}).items() if v is not None})
            finished = True
    if not finished:
        return None
    g = lambda k: raw.get(k) or 0                                                    # noqa: E731
    return {"prompt_tokens": g("input_tokens") + g("cache_read_input_tokens") + g("cache_creation_input_tokens"),
            "completion_tokens": g("output_tokens"), "cached_tokens": g("cache_read_input_tokens"),
            "reasoning_tokens": (raw.get("output_tokens_details") or {}).get("thinking_tokens", 0)}
    ### END SOLUTION

# %% check
fields = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
def translated(events):
    tr = adapters.AnthropicStreamTranslator("claude-haiku-4-5")
    for name, data in events:
        tr.feed(name, data)
    return tr.usage

bundled = [tuple(e) for e in adapters.payload("anthropic_stream")["events"]]
for evs in (warm, bundled):
    mine, ref = usage_from_events(evs), translated(evs)
    assert [mine[f] for f in fields] == [getattr(ref, f) for f in fields], (mine, ref)
assert usage_from_events(warm)["cached_tokens"] > 0                                  # the warm stream read the cache
stack.fault("bolt", "midstream", count=1, after_chunks=3)
cut = bolt_events("When is my invoice due?", max_tokens=40)
assert "error" in [n for n, _ in cut] and usage_from_events(cut) is None             # cut: no bill in the stream
stack.fault("bolt", "midstream", count=1, after_chunks=3)
via = stack.chat(key_eu, "When is my invoice due?", stream=True, max_completion_tokens=40)
row = stack.ledger(request_id=via.header("x-request-id"))[0]
assert via.error and row["usage_source"] == "estimate" and row["completion_tokens"] > 0
print(f"✅ warm: {usage_from_events(warm)} | cut after 3 chunks: None, so the gateway billed "
      f"{row['completion_tokens']} completion tokens from the relayed chunks, marked '{row['usage_source']}'")

# %% [markdown]
# ## Exercise 1.2 — rebuild a streamed answer, and notice an error after the first byte
#
# Write `accumulate(chunks)`. Its input is the `chat.completion.chunk` dicts that a client read from the gateway.
# Return a dict with these keys:
#
# - `content`: the concatenated text.
# - `tool_calls`: in the order of `index`, each `{"id", "name", "arguments"}`.
# - `finish_reason`: the last value that is not null.
# - `usage`: the last value that is not null. It is never a sum: with continuous usage, every chunk has a cumulative
#   total.
# - `error`: the `error` object if the stream had one, else `None`.
#
# The first delta of a tool call has its `id` and `function.name`. Later deltas have only `index` and a fragment of
# `function.arguments`. An error in a 200 stream is how the gateway tells you that the answer is not complete. After the
# first byte, the gateway cannot fall back any more, thus it tells you in the stream.
#
# The check replays three streams:
#
# - a tool-call stream that the gateway translated from the Anthropic events of bolt,
# - a stream made by hand with two interleaved calls,
# - a live stream that `acme` cuts in the middle of the answer.

# %% exercise
def accumulate(chunks: list) -> dict:
    ### BEGIN SOLUTION
    content, calls, finish, usage, error = "", {}, None, None, None
    for ch in chunks:
        if ch.get("error"):
            error = ch["error"]
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            finish = c.get("finish_reason") or finish
    return {"content": content, "tool_calls": [calls[i] for i in sorted(calls)], "finish_reason": finish, "usage": usage,
            "error": error}
    ### END SOLUTION

# %% check
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
tc_stream = stack.chat(key_eu, "What is the weather in Paris today?", stream=True, tools=tools, stream_options={"include_usage": True})
mine, ref = accumulate(tc_stream.chunks), sse.StreamAccumulator()
for c in tc_stream.chunks:
    ref.add(c)
assert mine["tool_calls"][0]["name"] == "get_weather" and json.loads(mine["tool_calls"][0]["arguments"]) == json.loads(ref.calls[0]["function"]["arguments"])
assert mine["finish_reason"] == "tool_calls" and mine["usage"] == ref.usage and mine["error"] is None
hand = [{"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "t2", "arguments": "{\"y\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "t1", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"x\": 1}"}}]}}], "usage": {"completion_tokens": 3}},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": ": 2}"}}]}}], "usage": {"completion_tokens": 5}},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]
out = accumulate(hand)
assert [(c["id"], c["name"], json.loads(c["arguments"])) for c in out["tool_calls"]] == [("a", "t1", {"x": 1}), ("b", "t2", {"y": 2})]
assert out["usage"] == {"completion_tokens": 5}
stack.fault("acme", "midstream", count=1, after_chunks=4)
cut = stack.chat(key_a, "Tell me a long story.", stream=True)
got = accumulate(cut.chunks)
assert cut.status == 200 and got["error"] is not None and got["content"] == cut.text and got["finish_reason"] is None
print(f"✅ tool calls rebuilt by index; a stream acme cut after 4 chunks is still an HTTP 200 with "
      f"{len(got['content'].split())} words and error {got['error'].get('code')!r} — no finish_reason, so not an answer")

# %% [markdown]
# ## Exercise 1.3 — the body the gateway sends to an Anthropic-dialect provider
#
# The OpenAI half of the relay rule is the exercise 1.3 of the core: ask for `include_usage` on streams only. The other
# dialect needs a real translation, and `bolt` refuses a body that has an error in it. Write `to_anthropic(body, model,
# cap)` for a chat-completions `body` with no tools. Return a body with these parts:
#
# - The `system` and `developer` messages, moved out into one `system` string. Join them with a newline. If there are
#   none, do not include the key.
# - The other messages, as `{"role", "content"}`.
# - `model`.
# - `max_tokens`. Anthropic makes it necessary. Use `cap` if the caller gives it, else
#   `adapters.ANTHROPIC_DEFAULT_MAX_TOKENS`.
# - `stream`, as a bool.
# - `temperature` and `top_p`, copied when they are present.
# - `stop`, as a list in `stop_sequences`.
# - No `stream_options`. Anthropic streams always have usage.
#
# The check compares your body with the adapter of the gateway on a grid of bodies. It sends each of your bodies
# directly to `bolt`. Then it shows what bolt says to a body with no `max_tokens`.

# %% exercise
def to_anthropic(body: dict, model: str, cap=None) -> dict:
    ### BEGIN SOLUTION
    system = [m["content"] for m in body["messages"] if m["role"] in ("system", "developer")]
    up = {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in body["messages"]
                                       if m["role"] not in ("system", "developer")],
          "max_tokens": cap or adapters.ANTHROPIC_DEFAULT_MAX_TOKENS, "stream": bool(body.get("stream"))}
    if system:
        up["system"] = "\n".join(system)
    for k in ("temperature", "top_p"):
        if k in body:
            up[k] = body[k]
    if body.get("stop"):
        up["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    return up
    ### END SOLUTION

# %% check
msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"}]
grid = [({"model": "chat", "messages": msgs[1:2]}, None), ({"model": "chat", "messages": msgs, "stream": True}, 64),
        ({"model": "chat", "messages": msgs, "temperature": 0, "stop": "END"}, None),
        ({"model": "chat", "messages": [{"role": "developer", "content": "Rules."}] + msgs, "top_p": 0.9,
          "stop": ["A", "B"], "stream": True, "stream_options": {"include_usage": True}}, 8)]
for b, cap in grid:
    mine, ref = to_anthropic(b, "claude-haiku-4-5", cap), adapters.to_upstream("anthropic", b, "claude-haiku-4-5",
                                                                              stream=bool(b.get("stream")), output_cap=cap)
    assert mine == ref, (mine, ref)
    sent = client.request("POST", stack.fake_url("bolt") + "/v1/messages", mine,
                          {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"})
    assert sent.status == 200, sent.json
bad = client.request("POST", stack.fake_url("bolt") + "/v1/messages", {k: v for k, v in mine.items() if k != "max_tokens"},
                     {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"})
assert bad.status == 400
print(f"✅ four bodies translated as the gateway does and accepted by bolt; without max_tokens: {bad.status} "
      f"{bad.json['error']['message']!r}")

# %% [markdown]
# ## Worked example: spans, written as OTLP/JSON lines and read back
#
# Every request makes one SERVER span (`POST /v1/chat/completions`). It also makes one CLIENT span for each upstream
# target that it tried (`chat {model}`). The spans use the GenAI names that `gwlab.gateway.otel` pins. These names are
# the set that semantic-conventions v1.41.0 and the main branch of the GenAI repo have in common. Make sure that they are
# still correct when you move the pin.
#
# The next cell sends a request whose first target fails one time. The result is two client spans under one server
# span.

# %%
path = os.path.abspath("_spans_01.jsonl")
stack.gateway.tracer.path = path
open(path, "w").close()
stack.fault("acme", "503", count=1)
r = stack.chat(key_a, "hello", stream=True)
spans = otel.read_spans(path)
from gwlab import report
print(report.spans_tree([s for s in spans if s.trace_id == stack.last_decision()["trace_id"]]))

# %% [markdown]
# ## Exercise 1.4 — the gateway's own time
#
# Use the spans of one trace to calculate the milliseconds that the gateway spent *outside* upstream calls. This time
# is the duration of the server span minus the sum of the durations of its client spans. Include the failed attempts.
# The caller waited during these attempts, but the gateway did not process the request during them either. Write
# `gateway_ms(spans)`.
#
# The check calculates the value for ten requests that it reads back from the file. It compares each value with the
# arithmetic of the lab. At T0, the expected value is a few milliseconds. The hop has a low cost. What a gateway costs
# is a failure point and a box that holds every key (PRIMER §1).

# %% exercise
def gateway_ms(spans: list) -> float:
    ### BEGIN SOLUTION
    server = next(s for s in spans if s.kind == otel.KIND["server"])
    upstream = sum(s.duration_s for s in spans if s.kind == otel.KIND["client"])
    return (server.duration_s - upstream) * 1e3
    ### END SOLUTION

# %% check
open(path, "w").close()
ids = []
for i in range(10):
    stack.chat(key_a, f"question {i}", stream=True)
    ids.append(stack.last_decision()["trace_id"])
spans = otel.read_spans(path)
own = []
for t in ids:
    tr = [s for s in spans if s.trace_id == t]
    ref = next(s for s in tr if s.kind == 2).duration_s - sum(s.duration_s for s in tr if s.kind == 3)
    assert abs(gateway_ms(tr) - ref * 1e3) < 1e-6
    own.append(gateway_ms(tr))
assert all(x >= 0 for x in own)
print(f"✅ gateway time per request: median {statistics.median(own):.1f} ms of {statistics.median(s.duration_s for s in spans if s.kind == 2) * 1e3:.0f} ms end to end (simulated upstream)")
os.remove(path)

# %% [markdown]
# ## T0 + Docker: the same stack in containers
#
# With Docker, `deploy/local/up.sh` builds one image. It runs the gateway, `acme` and `bolt`, and `acme` fails 30 % of
# the requests. `smoke.sh` issues a key and sends ten requests. If a gateway runs already and `GWLAB_URL` points to it,
# the next cell talks to it. If not, the cell prints the commands and a bundled sample output in the documented format
# (illustrative).

# %%
if tiers["gateway_url"]:
    url = tiers["gateway_url"].rstrip("/")
    k = client.post_json(url + "/admin/keys", {"tenant": "team-a"}, token=os.environ.get("GWLAB_ADMIN_TOKEN", "dev-admin-token")).json["key"]
    for i in range(5):
        rr = client.chat(url, k, {"model": "chat", "messages": [{"role": "user", "content": f"hello {i}"}]}, stream=True)
        print(rr.status, rr.header("x-gwlab-target"), "attempts", rr.header("x-gwlab-attempts"))
else:
    print("no GWLAB_URL: to run it with Docker -> cd deploy/local && ./up.sh && ./smoke.sh && ./down.sh"
          + ("" if tiers["docker"] else "   (no Docker daemon here)"))
    print(env.sample("compose_smoke.txt")[:900])

# %% [markdown]
# ## T1: a real vLLM as one provider
#
# When you set `GWLAB_VLLM_URL`, a second gateway starts with the `vllm` config. The URL is for deploy/any-gpu, which
# serves `Qwen/Qwen2.5-0.5B-Instruct` as `lab/llm` with `--enable-prompt-tokens-details`. In the `vllm` config, vLLM is
# first in `chat`, and a fake is the fallback. The same request now returns measured usage from a real engine. It gets
# the label `MEASURED` only when vLLM really served it (`gwlab.t1.label`). A fallback to the fake gets the label
# `SIMULATED`, whatever the cell expected.

# %%
if tiers["vllm_url"]:
    for tag, target, ttft, usage in t1.first_requests(tiers["vllm_url"], ["What does a gateway own?",
                                                                          "What does a gateway own? Answer in one word."],
                                                      fallback=stack.fakes["acme"].spec):
        print(tag, target, f"TTFT {ttft * 1e3:.0f} ms", usage)      # MEASURED only when vLLM served it
else:
    print("T1 skipped: start vLLM with deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %%
stack.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Every model call goes through one gateway. Callers hold virtual keys. The gateway keeps only a
# hash of each key, and each key has a scope of aliases and a budget. One call revokes a key. The tenant comes from the
# verified key, never from a header. The provider keys are only in the gateway.
#
# "A request gives an alias. The gateway decides if the request runs (budget, limits) and which provider, model, region
# or pool serves it. In a self-hosted pool, the endpoint picker selects the replica. Everything uses chat completions
# over SSE. Adapters translate the other dialects, also the usage fields that do not align.
#
# "The gateway parses each stream. It asks the provider for usage on every stream, and it removes the usage for clients
# that did not ask for it. It writes one ledger row with a price for each request, and also spans with the GenAI names.
# Its cost is a hop of a few milliseconds, a failure point that we must run with redundancy, and one box that holds
# every key. That is why it is the service that we protect the most."
#
# **Drill 1.** *Why not let each app call the providers with its own keys?* Then every app holds provider keys. Each
# app does retries, limits and fallbacks in a different way, and no one place bills anything. Then a leaked key is a
# provider incident, not a revocation with one call. Also, no one can answer "who spent what".
#
# **Drill 2.** *A client streams without `include_usage`. How does the ledger know the output count?* The gateway sets
# it on the upstream request anyway, and it removes the usage chunk from the response. If the stream stops before that
# chunk, the gateway bills the row from the chunks that it relayed and marks the row as an estimate. The row is never 0.
#
# **Drill 3.** *Anthropic reports 2,300 input tokens and 2,700 cache reads; what goes in `prompt_tokens`?* The answer
# is 5,000. Anthropic's `input_tokens` excludes the cache. Thus the canonical prompt count (and OTel's
# `gen_ai.usage.input_tokens`) is the sum. If you bill 2,300 as input and then give a discount for the cache again, you
# charge less than the correct price.
