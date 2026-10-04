# %% [markdown]
# # 01 · One front door
#
# **Tier:** T0. It uses only the CPU and no network, and it runs in a few seconds. Everything runs in process on a virtual clock. The
# same gateway over real HTTP, in front of fake providers or one vLLM (T1), is `gateway-lab` notebook
# `01_a_gateway_over_http`.
#
# ## The one-minute version
# A gateway is the one service that every app calls. The apps do not call the providers. The gateway speaks one API:
# OpenAI-style chat completions, sent as a stream of server-sent events. It puts the differences of each provider into
# **adapters that are mostly data**:
#
# - which usage fields add up to the prompt tokens and the completion tokens,
# - how the finish reasons map,
# - which header carries the key.
#
# A streamed answer is not one response. It is a sequence of chunks, and you must **accumulate** them. Accumulate the
# content by concatenation, the tool calls by `index`, and the usage from one extra chunk. That chunk comes only if you asked for
# it. Also, a failure can arrive *inside* an HTTP 200. At the end of this notebook, you can do these things:
#
# - normalise the usage of four providers,
# - accumulate a stream with parallel tool calls,
# - relay a stream so that the gateway always meters it,
# - bill a cut stream,
# - tell what the hop costs in availability,
# - read the GenAI spans of a request back from a file.
#
# Primer: §1 *One front door, one API* (`../PRIMER.md`). See §5.6 for the spans and §6.1 for the keys.

# %%
import json
import tempfile
from pathlib import Path

from gwcore import api, keys, otel, routing
from gwcore.gateway import Gateway
from gwcore.providers import ADAPTERS, PAYLOADS, Clock, FakeProvider, normalize_stream, normalize_usage
from gwcore.routing import Router, Target

print(PAYLOADS["_label"][:160], "...")
for dialect in ("openai", "anthropic", "gemini", "vllm"):
    body = PAYLOADS[dialect]["response"]
    raw = body.get("usage") or body.get("usageMetadata")
    print(f"{dialect:9s} raw usage {json.dumps(raw)[:90]}...")
    print(f"{'':9s} canonical {normalize_usage(dialect, raw)}")

# %% [markdown]
# The call is the same, with four spellings and one canonical answer. The answer is 5,000 prompt tokens (2,700 of them
# cached) and 1,550 completion tokens (1,200 of them reasoning).
#
# Look at the two traps in the raw rows. Anthropic's
# `input_tokens` is 2,300. It **excludes** the cached tokens. Gemini's `candidatesTokenCount` is 350. Gemini reports
# the thinking tokens **outside** it. The adapter table holds that knowledge:

# %%
for dialect, a in ADAPTERS.items():
    print(f"{dialect:9s} auth={a['auth'][0]:14s} prompt_tokens <- {' + '.join(a['usage']['prompt_tokens'])}")

# %% [markdown]
# ## Worked example 1 — a stream, frame by frame
# A fake provider on a virtual clock (TTFT 0.3 s, ITL 20 ms) sends a short answer as a stream. We ask for usage. Thus
# one extra chunk with `choices: []` arrives before `data: [DONE]`.

# %%
clock = Clock()
p = FakeProvider("openai", clock, ttft=0.3, itl=0.02, output_tokens=6)
resp = p.chat(api.chat_request("gpt-5.4-mini", [{"role": "user", "content": "hello"}], stream=True, include_usage=True))
wire = "".join(api.sse(ev) for ev in resp.events)
print(wire)
print(f"virtual time after the stream: {clock.now():.2f} s  (= 0.3 + 5 x 0.02)")
acc = api.StreamAccumulator()
for ev in api.parse_sse(wire):
    acc.add(ev)
print("text:", repr(acc.text()), "| finish:", acc.finish_reason, "| usage:", acc.usage["completion_tokens"], "tokens")

# %% [markdown]
# ## Worked example 2 — other dialects stream differently
# Anthropic sends named events. Gemini sends whole function calls and puts a flag on the thought parts.
# `normalize_stream` changes both into canonical chunks. The *usage* survives: the final `message_delta` of Anthropic
# carries the cumulative count, and `output_tokens_details` holds the thinking tokens (anthropic-sdk-python 1.8.0
# `MessageDeltaUsage`). But this is true only if the stream ends.
#
# If you cut the stream before `message_delta`, you hold only the usage of `message_start`. Its `output_tokens` is 1,
# and that is not a bill. Then the normaliser emits no usage chunk at all. The gateway estimates from the deltas that it
# relayed, and it marks the value as an estimate. It never invents the count that it does not have.

# %%
for dialect in ("anthropic", "gemini"):
    acc = api.StreamAccumulator()
    for ev in normalize_stream(dialect, PAYLOADS[dialect]["stream"]):
        acc.add(ev)
    print(f"{dialect:9s} text={acc.text()!r} reasoning={''.join(acc.reasoning)!r}")
    print(f"{'':9s} tool calls={acc.tool_calls()} usage={acc.usage}")
events = PAYLOADS["anthropic"]["stream"]
cut = normalize_stream("anthropic", events[:[e["event"] for e in events].index("message_delta")])
acc = api.StreamAccumulator()
for ev in cut:
    acc.add(ev)
print("anthropic, cut before message_delta: usage", acc.usage, "| estimate from the deltas:", acc.output_estimate(), "tokens")

# %% [markdown]
# ## Worked example 3 — the whole front door
# `Gateway.handle()` runs the path of §1.3:
#
# 1. authenticate a **virtual key** (the tenant comes from the key),
# 2. screen,
# 3. cache,
# 4. reserve tokens,
# 5. walk the chain,
# 6. relay and meter,
# 7. reconcile,
# 8. record.
#
# The primary provider has a scripted outage. Thus the request falls through to the next target. This occurs before
# the first byte, so the client never sees the outage.

# %%
clock = Clock()
providers = {"google": FakeProvider("google", clock, outages=[(0, 60, 503)]),
             "openai": FakeProvider("openai", clock, ttft=0.4)}
chain = {"chat": [Target("google", "gemini-3.5-flash"), Target("openai", "gpt-5.4-mini")]}
ks = keys.KeyStore(seed=7)
key = ks.issue("acme", tpm=100_000)
gw = Gateway(keys=ks, router=Router(chain), providers=providers, clock=clock)
print("virtual key (shown once):", key[:12] + "...", "| stored:", list(ks.by_hash)[0][:16] + "... (a SHA-256)")
r = gw.handle(key, api.chat_request("chat", [{"role": "user", "content": "Summarise our refund policy"}], stream=True))
print("status", r.status, "| attempts", [(t.provider, o) for t, o in r.attempts], "| served by", r.headers)
print("first frame:", r.frames[0].strip()[:110], "...")
print("last frames:", [f.strip() for f in r.frames[-2:]])
print("ledger row:", r.row)

# %% [markdown]
# The client did not ask for usage, and no frame carries it. But the ledger row has exact token counts. The gateway
# asked upstream for `include_usage` for its own use, and it removed the chunk on the way back. Exercise 1.3 uses this
# pattern.
#
# ## Exercise 1.1 — write two adapters by hand
# Write `anthropic_usage(raw)` and `gemini_usage(raw)`. Each returns the canonical dict with these keys:
#
# - `prompt_tokens` (with the cached tokens and the cache-written tokens),
# - `completion_tokens` (with the reasoning tokens),
# - `cached_tokens`,
# - `cache_write_tokens` (the prompt tokens that the provider writes to its cache and bills at their own price),
# - `reasoning_tokens`,
# - `total_tokens`.
#
# The field names are in the table in §1.5 of the primer. If a field is not there, it counts as 0. Gemini reports no cache writes.

# %% exercise
def anthropic_usage(raw):
    ### BEGIN SOLUTION
    g = lambda k: raw.get(k) or 0
    prompt = g("input_tokens") + g("cache_read_input_tokens") + g("cache_creation_input_tokens")
    out = g("output_tokens")
    return {"prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": g("cache_read_input_tokens"),
            "cache_write_tokens": g("cache_creation_input_tokens"),
            "reasoning_tokens": (raw.get("output_tokens_details") or {}).get("thinking_tokens", 0), "total_tokens": prompt + out}
    ### END SOLUTION


def gemini_usage(raw):
    ### BEGIN SOLUTION
    g = lambda k: raw.get(k) or 0
    prompt = g("promptTokenCount") + g("toolUsePromptTokenCount")
    out = g("candidatesTokenCount") + g("thoughtsTokenCount")
    return {"prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": g("cachedContentTokenCount"),
            "cache_write_tokens": 0, "reasoning_tokens": g("thoughtsTokenCount"), "total_tokens": prompt + out}
    ### END SOLUTION

# %% check
import random

assert anthropic_usage(PAYLOADS["anthropic"]["response"]["usage"]) == normalize_usage("anthropic", PAYLOADS["anthropic"]["response"]["usage"])
assert gemini_usage(PAYLOADS["gemini"]["response"]["usageMetadata"])["completion_tokens"] == 1550
rng = random.Random(0)
for _ in range(200):
    a = {k: rng.randint(0, 5000) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")}
    a["output_tokens_details"] = {"thinking_tokens": rng.randint(0, a["output_tokens"])}
    g = {k: rng.randint(0, 5000) for k in ("promptTokenCount", "cachedContentTokenCount", "candidatesTokenCount", "thoughtsTokenCount")}
    assert anthropic_usage(a) == normalize_usage("anthropic", a) and gemini_usage(g) == normalize_usage("gemini", g)
print("✅ two adapters written by hand match the table-driven ones on 200 random usages")

# %% [markdown]
# ## Exercise 1.2 — accumulate parallel tool calls
# The key of each tool-call delta is its `index`. Only the first delta of a call carries `id` and `function.name`. The
# later deltas carry fragments of `function.arguments`. Write `accumulate_tool_calls(chunks)`. It returns a list of
# `(id, name, arguments_string)` in the order of the index. Ignore the chunks with no choices (the usage chunk) and the
# `DONE` marker.

# %% exercise
def accumulate_tool_calls(chunks):
    ### BEGIN SOLUTION
    calls = {}
    for ch in chunks:
        if ch == api.DONE:
            continue
        for choice in ch.get("choices", []):
            for tc in choice.get("delta", {}).get("tool_calls") or []:
                c = calls.setdefault(tc["index"], [None, None, ""])
                c[0] = tc.get("id") or c[0]
                c[1] = (tc.get("function") or {}).get("name") or c[1]
                c[2] += (tc.get("function") or {}).get("arguments") or ""
    return [tuple(calls[i]) for i in sorted(calls)]
    ### END SOLUTION

# %% check
got = accumulate_tool_calls(PAYLOADS["openai"]["stream"])
assert got == [("call_a", "get_weather", '{"city": "Paris"}'), ("call_b", "get_time", '{"tz": "Europe/Paris"}')], got
assert all(json.loads(args) for _, _, args in got)
print("✅ two interleaved calls reassembled by index:", got)

# %% [markdown]
# ## Exercise 1.3 — relay so that every stream is metered
# Write the two halves of the relay rule:
#
# * `upstream_body(request)`: what the gateway sends upstream. This is a copy of the request, plus
#   `stream_options = {"include_usage": True}` **only** when `stream` is true (if not, vLLM answers 400).
# * `client_events(events, client_asked)`: the events that the client receives. These are all the events, except the
#   usage chunk (`choices == []` and a `usage`) when the client did not ask for it.

# %% exercise
def upstream_body(request):
    ### BEGIN SOLUTION
    body = dict(request)
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}
    return body
    ### END SOLUTION


def client_events(events, client_asked):
    ### BEGIN SOLUTION
    return [ev for ev in events if client_asked or ev == api.DONE or not (ev.get("choices") == [] and ev.get("usage"))]
    ### END SOLUTION

# %% check
fake = FakeProvider("x", Clock(), output_tokens=5)
plain = api.chat_request("m", [{"role": "user", "content": "hi"}])
assert fake.chat(upstream_body(plain)).status == 200 and "stream_options" not in upstream_body(plain)
streamed = api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True)
evs = list(fake.chat(upstream_body(streamed)).events)
assert evs[-2]["usage"]["completion_tokens"] == 5                  # the gateway always has usage ...
assert all(e == api.DONE or e["choices"] for e in client_events(evs, False))   # ... the client sees it only if asked
assert client_events(evs, True) == evs and streamed == api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True)
print("✅ usage injected upstream only on streams, and stripped from clients that did not ask")

# %% [markdown]
# ## Exercise 1.4 — bill a stream that was cut
# The provider stops after the 7th chunk, and no usage chunk arrives. The ledger must not bill 0. Calculate
# `cut_estimate`: the completion tokens to bill. Estimate them over all the content that the client received, with the
# method of `api.estimate_tokens` (four characters a token). Use only `cut_events`.

# %%
dying = FakeProvider("y", Clock(), output_tokens=40, fail_after=7)
cut_events = list(dying.chat(api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True, include_usage=True)).events)
print(cut_events[-2:])

# %% exercise
### BEGIN SOLUTION
received = "".join(ev["choices"][0]["delta"].get("content", "") for ev in cut_events if ev != api.DONE and ev.get("choices"))
cut_estimate = api.estimate_tokens(received)
### END SOLUTION

# %% check
acc = api.StreamAccumulator()
for ev in cut_events:
    acc.add(ev)
assert acc.usage is None and acc.error is not None
assert cut_estimate == acc.output_estimate() and cut_estimate > 0
print(f"✅ cut after 7 chunks: bill {cut_estimate} completion tokens, marked estimated (never 0, never the cap)")

# %% [markdown]
# ## Exercise 1.5 — what the hop costs in availability
# The gateway is in series with every provider. Your chain answers when one of its two targets answers: 99.5 % and
# 99.0 %, independent. The gateway itself has 99.95 %. Set `with_gateway` to the availability that a client sees. Use
# `routing.chain_availability`, and think of the gateway as the common-mode term. Set `nines_lost` to how much of the
# availability of the chain the gateway costs, in percentage points.

# %% exercise
### BEGIN SOLUTION
chain_only = routing.chain_availability([0.995, 0.99])
with_gateway = routing.chain_availability([0.995, 0.99], common_mode=0.0005)
nines_lost = 100 * (chain_only - with_gateway)
### END SOLUTION

# %% check
assert abs(with_gateway - 0.9995 * 0.99995) < 1e-12
assert abs(nines_lost - 0.049997) < 1e-5
print(f"✅ chain {routing.chain_availability([0.995, 0.99]):.3%} -> with the gateway {with_gateway:.3%}: the hop costs "
      f"{nines_lost:.3f} points, ten times the chain's own {100 * (1 - routing.chain_availability([0.995, 0.99])):.3f} % downtime")

# %% [markdown]
# ## Exercise 1.6 — read the request back from its spans
# The gateway wrote one SERVER span for each request and one CLIENT span for each upstream target that it tried, as
# OTLP/JSON lines. Use only the file to make `attempts`. This is a list of `(gen_ai.request.model, outcome)` for the
# CLIENT spans (kind 3) of the one trace, in the order in which they started. The outcome is the `error.type` of the
# span, or `"ok"`.

# %%
path = Path(tempfile.mkdtemp()) / "spans.jsonl"
print(gw.tracer.export_jsonl(path), "spans written; first line:", path.read_text().splitlines()[0][:150], "...")

# %% exercise
### BEGIN SOLUTION
spans = otel.read_jsonl(path)
client = sorted((s for s in spans if s["kind"] == 3), key=lambda s: s["start_ns"])
attempts = [(s["attrs"]["gen_ai.request.model"], s["attrs"].get("error.type", "ok")) for s in client]
### END SOLUTION

# %% check
assert attempts == [("gemini-3.5-flash", "503"), ("gpt-5.4-mini", "ok")], attempts
server = next(s for s in otel.read_jsonl(path) if s["kind"] == 2)
assert server["attrs"]["gen_ai.response.model"] == "gpt-5.4-mini" and server["attrs"]["gw.attempts"] == 2
print("✅ from the file alone:", attempts, "| names pinned to", otel.PINNED)

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Every app calls one gateway with a virtual key. The key, never a header, tells which
# tenant it is. The gateway speaks OpenAI chat completions over SSE to all callers. It normalises each provider through
# an adapter table.
#
# "Anthropic's input tokens exclude the cache. The thinking of Gemini is outside its candidates count. Also, vLLM reports a
# failure in the middle of a stream inside an HTTP 200, with an integer error code.
#
# "On streams, we always ask upstream for usage, and we remove it for clients that did not ask. Thus the gateway meters
# every stream. If a stream is cut before its usage chunk, we bill it on the tokens that the gateway relayed. We mark the row as an
# estimate.
#
# "Every request leaves a server span and one client span for each target that the gateway tried, with pinned GenAI
# names. The price is a hop in series with every provider. Here it cost 0.05 points of availability. Thus the gateway
# runs as the most critical service that we have."
#
# **Drill questions**
# 1. *Why does the gateway inject `stream_options.include_usage` itself?* Because the usage chunk is the bill, and it
#    arrives only if the request asks for it. It is possible that the client does not ask. Inject it only on streamed
#    requests (if not, vLLM rejects it). Remove the chunk from the clients that did not ask for it.
# 2. *A stream stops after 200 tokens and no usage arrives. What does the ledger say?* The ledger has an estimated row,
#    with status `cut`, for the tokens that the gateway counted from the relayed deltas. Later, the gateway reconciles
#    the row against the export of the provider. The row is never zero and never the output cap.
# 3. *What does not normalise across providers?* Three things. First, the stream shapes (named events against chunks,
#    whole function calls against argument fragments, no `[DONE]` from Anthropic). Second, the location of the
#    reasoning text. Third, the time when the counts arrive.
#
#    The thinking tokens of Anthropic come only in the final `message_delta`. Thus, if the stream is cut before
#    that `message_delta`, the gateway has no usage that it can trust. The adapter reports the count that is not there.
#    The ledger estimates that count and says so. The ledger does not invent the count.
