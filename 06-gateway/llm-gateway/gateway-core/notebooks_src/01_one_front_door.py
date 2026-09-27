# %% [markdown]
# # 01 · One front door
#
# **Tier:** T0 — CPU only, no network, a few seconds. Everything runs in process on a virtual clock. The same gateway
# over real HTTP, in front of fake providers or one vLLM (T1), is `gateway-lab` notebook `01_a_gateway_over_http`.
#
# ## The one-minute version
# A gateway is the one service every app calls instead of calling providers. It speaks one API — OpenAI-style chat
# completions, streamed as server-sent events — and pushes each provider's differences into **adapters that are mostly
# data**: which usage fields add up to prompt and completion tokens, how finish reasons map, which header carries the
# key. A streamed answer is not a response but a sequence of chunks you must **accumulate** (content by concatenation,
# tool calls by `index`, usage from one extra chunk that only comes if you asked), and a failure can arrive *inside* an
# HTTP 200. By the end you can normalise four providers' usage, accumulate a stream with parallel tool calls, relay a
# stream so it is always metered, bill a cut stream, say what the hop costs in availability, and read a request's
# GenAI spans back from a file.
#
# Primer: §1 *One front door, one API* (`../PRIMER.md`); §5.6 for the spans, §6.1 for the keys.

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
# Same call, four spellings, one canonical answer: 5,000 prompt tokens (2,700 of them cached) and 1,550 completion
# tokens (1,200 of them reasoning). Look at the two traps in the raw rows: Anthropic's `input_tokens` is 2,300 — it
# **excludes** the cached tokens — and Gemini's `candidatesTokenCount` is 350 — thinking is reported **outside** it.
# The adapter table is where that knowledge lives:

# %%
for dialect, a in ADAPTERS.items():
    print(f"{dialect:9s} auth={a['auth'][0]:14s} prompt_tokens <- {' + '.join(a['usage']['prompt_tokens'])}")

# %% [markdown]
# ## Worked example 1 — a stream, frame by frame
# A fake provider on a virtual clock (TTFT 0.3 s, ITL 20 ms) streams a short answer. We ask for usage, so one extra
# chunk with `choices: []` arrives before `data: [DONE]`.

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
# Anthropic sends named events; Gemini sends whole function calls and flags thought parts. `normalize_stream` turns
# both into canonical chunks. The *usage* survives — Anthropic's final `message_delta` carries the cumulative count,
# thinking tokens included in `output_tokens_details` (anthropic-sdk-python 1.8.0 `MessageDeltaUsage`) — but only
# if the stream ends. Cut it before `message_delta` and all you hold is `message_start`'s usage, whose
# `output_tokens` is 1: not a bill. The normaliser then emits no usage chunk at all, and the gateway estimates from
# the deltas it relayed, marked as an estimate — it never invents the missing count.

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
# `Gateway.handle()` runs the §1.3 path: authenticate a **virtual key** (the tenant comes from it), screen, cache,
# reserve tokens, walk the chain, relay and meter, reconcile, record. The primary provider is in a scripted outage,
# so the request falls through to the next target — before the first byte, so the client never notices.

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
# The client did not ask for usage, and no frame carries it — yet the ledger row has exact token counts. The gateway
# asked upstream for `include_usage` on its own behalf and stripped the chunk on the way back. That is the pattern of
# Exercise 1.3.
#
# ## Exercise 1.1 — write two adapters by hand
# Write `anthropic_usage(raw)` and `gemini_usage(raw)`: each returns the canonical dict with keys `prompt_tokens`
# (cached and cache-written included), `completion_tokens` (reasoning included), `cached_tokens`, `cache_write_tokens`
# (prompt tokens written to the provider's cache, which bill at their own price), `reasoning_tokens` and
# `total_tokens`. Field names are in the primer's §1.5 table. Missing fields count as 0; Gemini reports no cache writes.

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
# Tool-call deltas are keyed by `index`. Only the first delta for a call carries `id` and `function.name`; later ones
# carry fragments of `function.arguments`. Write `accumulate_tool_calls(chunks)` returning a list, ordered by index, of
# `(id, name, arguments_string)`. Ignore chunks that have no choices (the usage chunk) and the `DONE` marker.

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
# * `upstream_body(request)` — what the gateway sends upstream: a copy of the request, plus
#   `stream_options = {"include_usage": True}` **only** when `stream` is true (vLLM answers 400 otherwise);
# * `client_events(events, client_asked)` — the events the client receives: every event, except the usage chunk
#   (`choices == []` and a `usage`) when the client did not ask for it.

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
# The provider dies after the 7th chunk: no usage chunk ever arrives. The ledger must not bill 0. Compute
# `cut_estimate` — the completion tokens to bill, estimated the way `api.estimate_tokens` does (four characters a
# token) over all the content the client received. Use only `cut_events`.

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
# The gateway is in series with every provider. Your chain answers when either target does: 99.5 % and 99.0 %,
# independent. The gateway itself is 99.95 %. Set `with_gateway` to the availability a client sees (use
# `routing.chain_availability`, and think of the gateway as the common-mode term), and `nines_lost` to how much of
# the chain's availability the gateway costs, in percentage points.

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
# The gateway wrote one SERVER span per request and one CLIENT span per upstream target it tried, as OTLP/JSON lines.
# From the file alone, build `attempts`: a list of `(gen_ai.request.model, outcome)` for the CLIENT spans (kind 3) of
# the one trace, in the order they started, where outcome is the span's `error.type` or `"ok"`.

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
# **The two-minute version.** "Every app calls one gateway with a virtual key; the key — never a header — says which
# tenant it is. The gateway speaks OpenAI chat completions over SSE to everyone and normalises each provider through
# an adapter table: Anthropic's input tokens exclude the cache, Gemini's thinking sits outside its candidates count,
# vLLM reports a mid-stream failure inside an HTTP 200 with an integer error code. We always ask upstream for usage on
# streams and strip it for clients that did not ask, so every stream is metered, and a stream cut before its usage
# chunk is billed on the tokens it relayed, marked as an estimate. Every request leaves a server span and one client
# span per target tried, with pinned GenAI names. The price is a hop in series with every provider — here it cost
# 0.05 points of availability — so the gateway runs as the most critical service we have."
#
# **Drill questions**
# 1. *Why does the gateway inject `stream_options.include_usage` itself?* — Because the usage chunk is the bill and it
#    only arrives if asked; the client may not ask. Inject it only on streamed requests (vLLM rejects it otherwise)
#    and strip the chunk from clients that did not ask for it.
# 2. *A stream dies after 200 tokens and no usage arrives. What does the ledger say?* — An estimated row for the tokens
#    counted from the relayed deltas, status `cut`, reconciled later against the provider's export — never zero,
#    never the output cap.
# 3. *What does not normalise across providers?* — Stream shapes (named events vs chunks, whole function calls vs
#    argument fragments, no `[DONE]` from Anthropic), where reasoning text lives, and when the counts arrive: Anthropic's
#    thinking tokens come only in the final `message_delta`, so a stream cut before it has no usage to trust. The
#    adapter reports what is missing, and the ledger estimates it and says so, rather than inventing it.
