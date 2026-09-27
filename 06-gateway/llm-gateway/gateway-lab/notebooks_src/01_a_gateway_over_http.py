# %% [markdown]
# # 01 · A gateway over HTTP: one front door, one API, one ledger row per request
#
# **Tier:** T0 — the gateway and two fake providers start in this process and talk over localhost HTTP; every
# latency and token count the fakes produce is **simulated**. T0 + Docker: the same three processes in Docker
# Compose ([`deploy/local`](../deploy/local/README.md)). T1: a real `vllm serve` as one of the providers
# ([`deploy/any-gpu`](../deploy/any-gpu/README.md), `GWLAB_VLLM_URL`).
#
# ## The one-minute version
#
# A gateway is the one place every model call passes through, so it owns what no single app should: the
# **provider keys** (callers get *virtual* keys, hashed at rest, scoped to aliases, revocable), the **decision
# whether a request runs and which model, provider or pool serves it** (the 05 router then picks the replica,
# the engine batches — 05 PRIMER §1.3, §3.3, §7), the **meter** (one ledger row per request, priced from the
# provider's `usage`) and the **trace** (GenAI spans). Its lingua franca is chat completions over SSE; other
# dialects are translated by an adapter table, and what does not normalise — Anthropic's `input_tokens`
# excluding the cache, tool-call argument fragments, `reasoning` text — is where the bugs live (PRIMER §1).
#
# Two details make metering work on streams. The gateway sets `stream_options.include_usage` on every streamed
# upstream request (never on a non-streamed one: vLLM answers 400) so it always gets the provider's count — and
# strips that usage chunk from clients that did not ask for it. And it parses every chunk, where the 05 router
# forwards bytes untouched: it owns the bill, so it must read it.
#
# After this notebook you can issue a key, trace one request through the pipeline, read what went over the wire
# in both dialects, and find its ledger row and spans. Concepts: PRIMER §1 One front door, one API; §5 Metering,
# tracing and chargeback; §6 Keys, tenants and isolation ([`PRIMER.md`](../../PRIMER.md)).

# %%
import hashlib, json, os, random, statistics
from gwlab import client, env, sse
from gwlab.gateway import adapters, otel
from gwlab.stack import LocalStack

tiers = env.describe()
stack = LocalStack().start()          # acme: OpenAI dialect, bolt: Anthropic dialect, both SIMULATED; the gateway
print("gateway:", stack.url, "| providers:", {n: f.url for n, f in stack.fakes.items()})

# %% [markdown]
# ## Worked example: a virtual key, and what the gateway stores
#
# An operator issues keys through the admin API. The secret is returned **once**; the gateway stores its SHA-256
# (as LiteLLM's `hash_token` does) and a display prefix. The tenant is bound to the key: whatever a caller puts
# in a header, the gateway bills and limits the key's tenant.

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
# The client names an **alias** (`chat`), not a model. The gateway resolves it to a chain (`acme/fast`, then
# `bolt/haiku`), calls the first target with *its* provider key, and answers with headers that say who served
# the request. The body's `model` is the upstream model that actually answered (what OpenAI does for an alias).

# %%
r = stack.chat(key_a, "What does a gateway own?")
print(r.status, {k: v for k, v in r.headers.items() if k.startswith("x-gwlab") or k == "x-request-id"})
print("model:", r.json["model"], "| usage:", r.json["usage"])
row = stack.ledger(request_id=r.header("x-request-id"))[0]
print({k: row[k] for k in ("tenant", "alias", "target", "prompt_tokens", "completion_tokens", "usage_source", "cost_usd")})

# %% [markdown]
# Now streamed, without asking for usage. On the wire the client sees a role chunk, content chunks, a finish
# chunk and `data: [DONE]` — and **no** usage chunk. Yet the ledger row carries the provider's own counts
# (`usage_source: provider`): the gateway injected `include_usage` upstream and stripped the chunk on the way
# back. Asking for it (`stream_options.include_usage`) gets the chunk with `choices: []` just before `[DONE]`.

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
# `team-eu` may only use providers in the `eu` region (residency), so its `chat` goes to `bolt`, an
# Anthropic-dialect provider. Below: what bolt itself sends (named events, cumulative `output_tokens`, no
# `[DONE]`) and what the gateway turns it into (the same chat chunks as above).

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
# ## Exercise 1.1 — normalise usage across dialects
#
# Write `normalise_usage(dialect, raw)` returning a dict with `prompt_tokens`, `completion_tokens`,
# `cached_tokens` and `reasoning_tokens`, where prompt tokens **include** cached ones and completion tokens
# **include** reasoning ones (the canonical meaning; a thinking token bills as output, 00.5 PRIMER §5):
#
# | dialect | prompt | cached | completion | reasoning |
# |---|---|---|---|---|
# | `openai` | `prompt_tokens` | `prompt_tokens_details.cached_tokens` | `completion_tokens` | `completion_tokens_details.reasoning_tokens` |
# | `anthropic` | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` | `cache_read_input_tokens` | `output_tokens` | `output_tokens_details.thinking_tokens` |
# | `gemini` | `promptTokenCount + toolUsePromptTokenCount` | `cachedContentTokenCount` | `candidatesTokenCount + thoughtsTokenCount` | `thoughtsTokenCount` |
#
# Missing fields count as 0. The check runs the bundled payloads (sample output in the documented format,
# illustrative) and 300 random usage objects against the lab's adapter table.

# %% exercise
def normalise_usage(dialect: str, raw: dict) -> dict:
    ### BEGIN SOLUTION
    def g(path):
        cur = raw
        for p in path.split("."):
            cur = cur.get(p) if isinstance(cur, dict) else None
        return cur or 0
    if dialect == "openai":
        return {"prompt_tokens": g("prompt_tokens"), "completion_tokens": g("completion_tokens"),
                "cached_tokens": g("prompt_tokens_details.cached_tokens"),
                "reasoning_tokens": g("completion_tokens_details.reasoning_tokens")}
    if dialect == "anthropic":
        return {"prompt_tokens": g("input_tokens") + g("cache_read_input_tokens") + g("cache_creation_input_tokens"),
                "completion_tokens": g("output_tokens"), "cached_tokens": g("cache_read_input_tokens"),
                "reasoning_tokens": g("output_tokens_details.thinking_tokens")}
    return {"prompt_tokens": g("promptTokenCount") + g("toolUsePromptTokenCount"),
            "completion_tokens": g("candidatesTokenCount") + g("thoughtsTokenCount"),
            "cached_tokens": g("cachedContentTokenCount"), "reasoning_tokens": g("thoughtsTokenCount")}
    ### END SOLUTION

# %% check
fields = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
for d, raw in (("openai", adapters.payload("openai_usage")["usage"]), ("anthropic", adapters.payload("anthropic_message")["usage"]),
               ("gemini", adapters.payload("gemini_response")["usageMetadata"])):
    mine, ref = normalise_usage(d, raw), adapters.normalise_usage(d, raw)
    assert [mine[f] for f in fields] == [getattr(ref, f) for f in fields], (d, mine)
rng = random.Random(1)
for _ in range(300):
    n = lambda: rng.choice([0, rng.randint(1, 5000)])   # noqa: E731
    samples = {"openai": {"prompt_tokens": n(), "completion_tokens": n(), "prompt_tokens_details": {"cached_tokens": n()},
                          "completion_tokens_details": {"reasoning_tokens": n()}},
               "anthropic": {"input_tokens": n(), "cache_read_input_tokens": n(), "cache_creation_input_tokens": n(),
                             "output_tokens": n(), "output_tokens_details": {"thinking_tokens": n()}},
               "gemini": {"promptTokenCount": n(), "toolUsePromptTokenCount": n(), "cachedContentTokenCount": n(),
                          "candidatesTokenCount": n(), "thoughtsTokenCount": n()}}
    for d, raw in samples.items():
        ref = adapters.normalise_usage(d, raw)
        assert [normalise_usage(d, raw)[f] for f in fields] == [getattr(ref, f) for f in fields], (d, raw)
print("✅ one usage shape for three dialects: Anthropic's prompt is 2,300 + 2,700 cached = 5,000; Gemini's output is 230 + 120 thoughts")

# %% [markdown]
# ## Exercise 1.2 — rebuild a streamed answer, tool calls included
#
# Write `accumulate(chunks)` for a list of `chat.completion.chunk` dicts. Return a dict with `content` (the
# concatenated text), `tool_calls` (a list ordered by `index`, each `{"id", "name", "arguments"}`),
# `finish_reason` (the last non-null one) and `usage` (the last non-null one — never a sum: with continuous
# usage every chunk carries a running total). A tool call's first delta carries its `id` and `function.name`;
# later deltas carry only `index` and a fragment of `function.arguments`. The check replays a real tool-call
# stream the gateway produced from bolt's Anthropic events, and a hand-made stream with two interleaved calls.

# %% exercise
def accumulate(chunks: list) -> dict:
    ### BEGIN SOLUTION
    content, calls, finish, usage = "", {}, None, None
    for ch in chunks:
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
    return {"content": content, "tool_calls": [calls[i] for i in sorted(calls)], "finish_reason": finish, "usage": usage}
    ### END SOLUTION

# %% check
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
tc_stream = stack.chat(key_eu, "What is the weather in Paris today?", stream=True, tools=tools, stream_options={"include_usage": True})
mine, ref = accumulate(tc_stream.chunks), sse.StreamAccumulator()
for c in tc_stream.chunks:
    ref.add(c)
assert mine["tool_calls"][0]["name"] == "get_weather" and json.loads(mine["tool_calls"][0]["arguments"]) == json.loads(ref.calls[0]["function"]["arguments"])
assert mine["finish_reason"] == "tool_calls" and mine["usage"] == ref.usage
hand = [{"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "t2", "arguments": "{\"y\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "t1", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"x\": 1}"}}]}}], "usage": {"completion_tokens": 3}},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": ": 2}"}}]}}], "usage": {"completion_tokens": 5}},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]
out = accumulate(hand)
assert [(c["id"], c["name"], json.loads(c["arguments"])) for c in out["tool_calls"]] == [("a", "t1", {"x": 1}), ("b", "t2", {"y": 2})]
assert out["usage"] == {"completion_tokens": 5}
print("✅ tool calls rebuilt by index from", len(tc_stream.chunks), "chunks the gateway translated from Anthropic events")

# %% [markdown]
# ## Exercise 1.3 — the body the gateway sends upstream
#
# Write `upstream_body(body, upstream_model)` for an OpenAI-dialect provider. Return `(new_body, strip_usage)`:
# the body with `model` replaced; on a **streamed** request `stream_options` set to `{"include_usage": True}`; on a
# non-streamed request no `stream_options` key at all (vLLM rejects it with a 400); and `strip_usage` True when
# the client streamed without asking for usage (the relay then drops the usage chunk). Do not modify `body`.
# The check compares with the rules on a grid of bodies and then sends every body you produce straight to the
# vLLM-shaped fake (as the operator, with the provider key): each must be accepted.

# %% exercise
def upstream_body(body: dict, upstream_model: str) -> tuple:
    ### BEGIN SOLUTION
    up = {k: v for k, v in body.items() if k != "stream_options"}
    up["model"] = upstream_model
    stream = bool(body.get("stream"))
    if stream:
        up["stream_options"] = {"include_usage": True}
    asked = bool((body.get("stream_options") or {}).get("include_usage"))
    return up, stream and not asked
    ### END SOLUTION

# %% check
msgs = [{"role": "user", "content": "hi"}]
grid = [{"model": "chat", "messages": msgs, **extra} for extra in (
    {}, {"stream": False}, {"stream": True}, {"stream": True, "stream_options": {"include_usage": True}},
    {"stream": True, "stream_options": {"include_usage": False}}, {"stream": False, "stream_options": {"include_usage": True}})]
for b in grid:
    before = json.dumps(b, sort_keys=True)
    up, strip = upstream_body(b, "fast-1")
    assert json.dumps(b, sort_keys=True) == before, "the client's body was modified"
    assert up["model"] == "fast-1" and ("stream_options" in up) == bool(b.get("stream"))
    assert strip == (bool(b.get("stream")) and not (b.get("stream_options") or {}).get("include_usage"))
    sent = client.request("POST", stack.fake_url("acme") + "/v1/chat/completions", up, {"Authorization": "Bearer acme-key-1"})
    assert sent.status == 200, (up, sent.json)
print("✅ usage asked for on every stream, never on a non-stream (vLLM's 400); the client decides only what it sees")

# %% [markdown]
# ## Worked example: spans, written as OTLP/JSON lines and read back
#
# Every request produces one SERVER span (`POST /v1/chat/completions`) and one CLIENT span per upstream target
# it tried (`chat {model}`), with the GenAI names pinned in `gwlab.gateway.otel` (the set common to
# semantic-conventions v1.41.0 and the GenAI repo's main; verify when you move the pin). Below, a request
# whose first target fails once: two client spans under one server span.

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
# From one trace's spans, compute how many milliseconds the gateway spent *outside* upstream calls:
# the server span's duration minus the sum of its client spans' durations (failed attempts included — they
# are time the caller waited, but not the gateway's processing either). Write `gateway_ms(spans)`. The check
# computes it for ten requests read back from the file and compares with the lab's arithmetic; it should be a
# few milliseconds at T0 — the hop is cheap; what a gateway costs is a failure point and a box holding every
# key (PRIMER §1).

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
# With Docker, `deploy/local/up.sh` builds one image and runs the gateway, `acme` (failing 30 % of requests)
# and `bolt`; `smoke.sh` issues a key and sends ten requests. If a gateway is already running and
# `GWLAB_URL` points at it, the next cell talks to it; otherwise it prints the commands and a bundled sample
# output in the documented format (illustrative).

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
# With `GWLAB_VLLM_URL` set (deploy/any-gpu: `Qwen/Qwen2.5-0.5B-Instruct` served as `lab/llm` with
# `--enable-prompt-tokens-details`), a second gateway starts with the `vllm` config: vLLM first in `chat`, a fake
# as the fallback. The same request now returns measured usage from a real engine.

# %%
if tiers["vllm_url"]:
    t1 = LocalStack(config="vllm", upstreams={"local": tiers["vllm_url"]}, fakes={"acme": stack.fakes["acme"].spec}).start()
    k1 = t1.issue_key("team-a")
    for q in ("What does a gateway own?", "What does a gateway own? Answer in one word."):
        rr = t1.chat(k1, q, stream=True, stream_options={"include_usage": True})
        print("MEASURED", rr.header("x-gwlab-target"), f"TTFT {rr.ttft_s * 1e3:.0f} ms", rr.usage)
    t1.stop()
else:
    print("T1 skipped: start vLLM with deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %%
stack.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Every model call goes through one gateway. Callers hold virtual keys — hashed, scoped to
# aliases, budgeted, revocable in one call — and the tenant comes from the verified key, never a header; the
# provider keys live only in the gateway. A request names an alias; the gateway decides whether it runs
# (budget, limits) and which provider, model, region or pool serves it; inside a self-hosted pool the endpoint
# picker chooses the replica. Everything speaks chat completions over SSE; adapters translate the other
# dialects, including the usage fields that do not line up. The gateway parses each stream — it asks the
# provider for usage on every stream and strips it for clients that did not ask — and writes one priced ledger
# row per request, plus spans with the GenAI names. What it costs is a hop of a few milliseconds, a failure
# point to run redundantly, and one box holding every key, which is why it is the most defended service we
# have."
#
# **Drill 1.** *Why not let each app call the providers with its own keys?* — Every app then holds provider
# keys, implements retries, limits and fallbacks differently, and bills nothing in one place; a leaked key is a
# provider incident instead of a one-call revocation, and no one can answer "who spent what".
#
# **Drill 2.** *A client streams without `include_usage`. How does the ledger know the output count?* — The
# gateway sets it upstream anyway and strips the usage chunk from the response; if the stream is cut before
# that chunk, the row is billed from the chunks relayed and marked as an estimate — never 0.
#
# **Drill 3.** *Anthropic reports 2,300 input tokens and 2,700 cache reads; what goes in `prompt_tokens`?* —
# 5,000: Anthropic's `input_tokens` excludes the cache, so the canonical prompt count (and OTel's
# `gen_ai.usage.input_tokens`) is the sum; billing 2,300 as input and discounting the cache again undercharges.
