# The LLM gateway: one front door for many models — routing, fallbacks, caching, metering and tenant isolation

*Layer 06 of the stack (the control plane in front of models). Facts checked 26 September 2026 against upstream
sources: the OpenAI OpenAPI description (`openai/openai-openapi` at `42eccf1`), vLLM at tag v0.30.0, the OpenTelemetry
GenAI conventions (semantic-conventions v1.41.0 and semantic-conventions-genai at `e57c543`), the MCP specification
revision 2026-07-28, the OAuth 2.1 and Client ID Metadata Document drafts, the RFC 9449 source, SPIFFE and SPIRE,
LiteLLM 1.104.0, Envoy AI Gateway v1.1.0, Kong 3.10.0, Portkey gateway 1.15.2, the Anthropic and Google GenAI Python
SDKs, NeMo Guardrails and PurpleLlama. Product facts that could not be checked in a source are marked `(verify)`;
the dated Verify list is at the end. Every formula names the function in [`gateway-core`](gateway-core/) (package
`gwcore`) that computes it, and every latency in it is simulated on a virtual clock. The detailed lab is
[`gateway-lab`](gateway-lab/).*

An LLM gateway is the one service every app, agent and tenant calls instead of calling model providers directly. It
holds the provider keys, decides whether a request may run and which model, provider, region or pool serves it,
meters what it cost, and records what happened. This primer builds on material already in the repo and cites
rather than repeats it: the [scaling primer](../scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
(cost per conversation §3.4, quota strategy and the token bucket §5.1, retries and breakers §5.2, admission and degrade
levels §5.3, streaming §5.7, observability §5.10, tenancy §5.12 — modules 06.1–06.3), the
[identity primer](../identity-security/agentic-identity-gcp-lab/docs/primer.md) (agent identity §3.3, delegation §3.5,
enforcement points §4.1, secrets §5, screening §6.1, MCP as a resource server §7.1, tenancy §8, audit §9 — module
06.6), the [05 orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md) (the three decisions §1.3,
priorities §3.2, who decides what §3.3, model routing §7) and the
[07.2 agent platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) (its MCP
egress gateway, OAuth flow, tracing and breakers). Read those first; this primer assumes them.

---

## The one-minute version

A gateway is worth its hop when many apps, tenants and providers meet: one API (OpenAI-style chat completions over
server-sent events) in front of every provider, with the differences pushed into **adapters that are mostly data**.
It makes the decisions that must be made once, centrally: **whether** a request runs (the key, the tenant's budget
and token limits, a guardrail) and **which** model, provider, region or pool serves it (an alias resolves to an ordered
**fallback chain**). Inside a self-hosted pool the 05 router picks the replica and the engine batches, not the gateway.

Five mechanisms carry the design. **Fallback chains** fall through only on failures another target could fix — 429,
5xx, a timeout, a context too long — never on a bad request or a content-policy refusal, and **never after the
first byte** has reached the client; a breaker per target turns a dead provider into an instant skip. **Caches**
save a whole call (exact or semantic, namespaced by tenant, guarded on numbers and entities, measured for false hits)
or part of one (provider prompt caching, the engine's prefix cache). **Token limits** reserve prompt plus an output
bound at admission, debit tokens as they stream and reconcile at the end, because output length is unknown up front
and heavy-tailed. **Metering** prices the provider's `usage` — thinking tokens bill as output — into a ledger row per
request, and splits a shared pool's bill by GPU time. **Isolation** derives every tenant-scoped thing (limits, cache
namespace, `cache_salt`, ledger, traces) from the verified key, and the provider keys never leave the gateway.

The price is real: a hop on every request, a failure point in series with every provider, and every key in one box.
After this primer you can walk that design in a review and put numbers on each part.

---

## 1. One front door, one API

### 1.1 What the gateway owns

Without a gateway every app holds provider keys, implements its own retries, counts its own tokens and reports its
own spend — which means nobody can answer "what did tenant X cost last week" or "rotate the OpenAI key" in one place.
A gateway owns, for many apps, tenants and providers at once:

| Concern | What the gateway does | Primer section | `gwcore` module |
|---|---|---|---|
| Keys | issues virtual keys; holds provider keys; rotates them | §6 | `keys` |
| Policy | screens input and output; refuses what policy forbids | §7 | `guardrails` |
| Limits | requests and tokens per minute per key, tenant and provider | §4 | `ratelimit` |
| Routing | alias → fallback chain; filters and policies; breakers | §2 | `routing` |
| Caching | exact and semantic response caches, per tenant | §3 | `cache` |
| Metering | a ledger row per request, priced from `usage` | §5 | `metering` |
| Traces | GenAI spans per request and per upstream attempt | §5 | `otel` |
| MCP egress | the client side of MCP authorization for agents | §8 | `mcp_authz` |

### 1.2 The split: gateway, router, engine

Three layers make three different decisions, on three time scales
([05 PRIMER §1.3](../../05-orchestrator/serving-orchestration/PRIMER.md#13-the-three-decisions), §3.3):

```
 app / agent ──► GATEWAY (06)                     ──► ROUTER / EPP (05)            ──► ENGINE (04)
                 whether it runs: key, budget,        which replica of the pool:        which requests share
                 tokens, guardrail                    prefix affinity, queue, KV        the next step: batching,
                 which model / provider / region /    load; flow control and            KV blocks, preemption
                 pool: the fallback chain             priority inside the pool
                 per request, per tenant              per request, per replica          per step (milliseconds)
```

The 05 primer states it from below: "model routing proper — choosing a cheaper model per request — is a gateway
decision" and that layer "routes among replicas of the model already chosen" (§7), with one `InferencePool` per base
model. So a self-hosted target in a gateway chain is a *pool*, not a pod (§2.6), and the gateway's admission control
(scaling primer §5.3, `scalelab.admission`) sits above the router's flow control.

### 1.3 The request path

`gwcore.gateway.Gateway.handle()` is the whole path, in order; each stage can end the request with an OpenAI-shaped
error (`api.error_body()`):

```
 1 authenticate   virtual key → tenant, tier, scopes, budget            401 / 403
 2 screen input   guardrail hook "input" (§7)                             400 content_policy
 3 exact cache    key = SHA-256(tenant, model, messages, params) (§3)    hit → replay, $0
 4 reserve        tenant TPM: prompt + output cap (§4)                    429 + Retry-After
 5 route          chain → filters → policy → breaker per target (§2)     fall through or fail
 6 relay + meter  forward chunks; debit tokens; strip unasked usage      error chunk after 1st byte
 7 reconcile      usage → ledger row → budget; release the reservation (§5)
 8 record         cache store (complete answers only); spans (§5)
```

### 1.4 Chat completions and SSE as the lingua franca

The OpenAI chat-completions request (`model`, `messages`, `tools`, `max_completion_tokens`, `stream`, …) is the API
every gateway, engine and SDK speaks, vLLM's API server included (serving-engine PRIMER §1; the 04 lab's
[`fakeserver.py`](../../04-inference-engine/serving-engine/vllm-serving-lab/servelab/fakeserver.py) emulates it). A
streamed answer is a sequence of `chat.completion.chunk` objects, each an SSE frame `data: {…}` and a blank line,
ended by `data: [DONE]` (`api.sse()`, `api.parse_sse()`). Three details decide whether a gateway meters correctly:

- **Usage arrives only if asked.** With `stream_options.include_usage: true`, one extra chunk with `choices: []` and
  the request's `usage` is sent before `[DONE]`; if the stream is interrupted it may never arrive (the OpenAPI
  description says so). The gateway therefore always asks upstream, and strips that chunk from a client that did not
  ask (`Gateway.handle()`, stage 6) — the same design as LiteLLM's proxy. It must ask only on streamed requests:
  vLLM rejects `stream_options` without `stream: true` with a 400.
- **A stream is accumulated, not read.** Content concatenates; tool calls arrive as deltas keyed by `index`, where only
  the first delta carries the call's `id` and `function.name` and the `arguments` string arrives in fragments that are
  not valid JSON until the last one (`api.StreamAccumulator`). A cut stream leaves half a JSON object: report it,
  never guess it (`tool_calls()` returns `valid_json: False`).
- **Errors can arrive inside a 200.** vLLM v0.30.0 reports a failure mid-stream as a `data:` chunk holding an
  `error` object, then `[DONE]`, all under HTTP 200 — and its error `code` is the HTTP status as an integer where
  OpenAI's is a string (`api.normalize_error()`). A relay that checks only the status meters a failure as a success.

The gateway's token stream is not the agent's own API. The 07.2 lab's
[notebook 07](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/07_agent_api_streaming_tasks.ipynb)
streams *agent events* (`Last-Event-ID` resume, `Idempotency-Key`, a per-tenant 429) to an end user; the gateway
relays *model tokens* to that agent. Resume across a gateway failure is the agent API's job, not the gateway's.

### 1.5 Adapters as data, and what does not normalise

Most of a provider adapter is a table: the path, the auth header, which usage fields add up to canonical ones, how
finish reasons map (`providers.ADAPTERS`). The bundled samples (`providers.PAYLOADS`, sample output in the documented
format, illustrative) are one call — 5,000 prompt tokens of which 2,700 cached, 350 visible output tokens and 1,200
reasoning tokens — spelled four ways, and `providers.normalize_usage()` maps every one of them to 5,000 prompt and
1,550 completion tokens:

| Canonical | OpenAI / vLLM | Anthropic Messages | Gemini |
|---|---|---|---|
| `prompt_tokens` (cached included) | `prompt_tokens` | `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` (2,300 + 2,700 + 0) | `promptTokenCount` (+ `toolUsePromptTokenCount`) |
| `cached_tokens` | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | `cachedContentTokenCount` |
| `cache_write_tokens` | `prompt_tokens_details.cache_write_tokens` | `cache_creation_input_tokens` | — |
| `completion_tokens` (reasoning included) | `completion_tokens` | `output_tokens` | `candidatesTokenCount` + `thoughtsTokenCount` (350 + 1,200) |
| `reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.thinking_tokens` | `thoughtsTokenCount` |
| auth header | `Authorization: Bearer` | `x-api-key` + `anthropic-version` | `x-goog-api-key` |

Two traps sit in that table. Anthropic's `input_tokens` *excludes* cache reads and writes, so feeding it straight
into a price function that expects cached tokens inside the prompt count (`scalelab.capacity.cost_per_call`,
`metering.price_call()`) discounts the cache twice. Gemini reports thinking *outside* `candidatesTokenCount`, so a
gateway that bills candidates alone under-bills a thinking call (§5.2). Cache *writes* are prompt tokens too, but
Anthropic prices them above the input rate (1.25× on claude-haiku-4-5), so they get their own canonical field and
their own price (§5.3). vLLM fills `cached_tokens` only with `--enable-prompt-tokens-details` and
`reasoning_tokens` only with a `--reasoning-parser`.

What does not normalise cleanly is the stream. Anthropic sends named events (`message_start` with the input usage,
`content_block_delta` carrying `text_delta`, `input_json_delta` or `thinking_delta`, `message_delta` with cumulative
output usage, `message_stop` and no `[DONE]`); the Gemini API sends whole function calls, not argument fragments
(`partial_args` is Vertex-only), and flags thought parts `thought: true`; vLLM names the reasoning text `reasoning` (it
renames an incoming `reasoning_content`), and OpenAI's schema has neither. `providers.normalize_stream()` turns the
first two into canonical chunks. The counts do normalise, but only at the end: Anthropic's usage is cumulative, and
the final `message_delta` carries the whole of it — `output_tokens_details.thinking_tokens` included when the API
sends it (optional in the SDK's `MessageDeltaUsage`) — so the §1.5 stream normalises to the same 1,550 completion
and 1,200 reasoning tokens as the non-streamed response. What does not survive is a stream cut before that event:
all it holds is `message_start`'s usage, whose `output_tokens` is 1. That is not a bill, so the normaliser emits no
usage chunk for a cut stream, and the gateway estimates from the deltas it relayed and marks the row estimated
(§5.2) — it reconciles the gap later rather than inventing a count.

### 1.6 What it costs

A gateway is a hop on every request, a failure point in series with every provider, and every key in one box. In
series, availabilities multiply: a gateway at 99.95 % in front of a chain that answers 99.995 % of the time
(§2.5) gives 99.945 % (`routing.chain_availability([0.995, 0.99], common_mode=0.0005)` = 0.999450025). So the
gateway is run like the most critical service you have: several replicas across zones, no state in process (limits
and caches in a shared store, §4.4), a fast path whose added latency you measure and budget (the lab runs the gateway
over HTTP on localhost, where you can time the hop), and keys held in a secret manager with access logged (§6).

---

## 2. Model routing and fallback chains

### 2.1 Aliases and chains

Clients ask for an alias — `chat`, `chat-cheap`, `embed` — never a provider's model id: ids are retired on a clock
and move between tiers, and the scaling primer's rule applies (model ids live in configuration, §5.2). An alias
resolves to an ordered chain of `(provider, model, region)` targets (`routing.Target`), optionally per tenant tier
(`chat@gold`, `Router.candidates(tier=…)`):

```
chat      → [google/gemini-3.5-flash (global), openai/gpt-5.4-mini, anthropic/claude-haiku-4-5]
chat@gold → [openai/gpt-5.4-mini, google/gemini-3.5-flash]
chat-eu   → [google/gemini-3.5-flash (europe-west4), self/lab/llm (europe-west4)]   residency: no global endpoint
```

### 2.2 Filters and policies

`Router.candidates()` first **filters** — the request's needs against each target's capabilities (`routing.needs()`:
tools; reasoning when `reasoning_effort` is above minimal), prompt plus output against the context window, the
region against the tenant's residency — then **orders** by a policy:

| Policy | Orders by | Use when |
|---|---|---|
| `ordered` | the chain as written | a primary and its siblings (the default) |
| `cheapest` | blended $ for this request's prompt and output cap (`routing.blended_cost()`) | quality is equal across the chain |
| `ewma_ttft` | a per-target moving average of TTFT (`routing.ewma()`, α = 0.2; TTFT as serving-engine PRIMER §11 defines it) | latency-sensitive routes; unmeasured targets first so they get measured |
| `canary` | a stable hash of the request id against the targets' weights | a new model on a share of traffic |
| tier | a separate chain per tier (`chat@gold`) | paid tiers get the better or less contended model |
| effort | the reasoning filter above | only thinking models take high `reasoning_effort` |

Routing by effort is the RL primer's point in practice ([00.5 PRIMER §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)):
a thinking token is billed as output, so sending only the hard requests to a thinking model is the cheapest way to
buy accuracy — which needs a classifier, or a cheap first pass, that knows which requests are hard.

### 2.3 What falls through, and what must not

Fall through only when another target could plausibly succeed where this one failed (`routing.falls_through()`):

| Falls through | Why | Must not fall through | Why |
|---|---|---|---|
| 429 | that provider's quota; a sibling has its own pool | 400 bad request | every target gets the same bad request |
| 500, 502, 503, 504 | that provider is sick | 401, 403 | your credentials or scopes, not their health |
| 529 `overloaded_error` (Anthropic) | that provider is overloaded; try elsewhere | 404 | a wrong model id is a configuration bug |
| 408, timeout / no answer | that provider is slow or down | content-policy refusal | falling through launders a refusal into another model's answer |
| 400 `context_length_exceeded` | a longer-context target can take it | | |

Falling through on a content-policy refusal is sometimes chosen deliberately (LiteLLM has `content_policy_fallbacks`);
if you choose it, make it a named policy per route with an owner, not a default.

**Fall back only before the first byte.** Once a chunk has reached the client, a second model's output cannot be
spliced onto the first — the client would read two answers as one. After the first byte the gateway surfaces the
failure as an error chunk and `[DONE]`, and meters what was relayed (§5.2). Before it — a 5xx, a timeout, or an error
chunk that arrives *first* inside a 200 — nothing has been sent, and the next target is tried
(`Gateway.handle()`, stage 6; tested in `tests/test_gateway_otel.py`).

### 2.4 Retries and breakers

Retries with full jitter, `Retry-After` and a deadline are the scaling primer's (§5.2; `scalelab.resilience.backoff`,
`call_with_retries`), and the breaker's full state machine is taught in the 07.2 lab's
[notebook 10](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/10_reliability_retries_breakers.ipynb)
([`agentlab/reliability/breaker.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/reliability/breaker.py)).
The gateway applies one breaker **per target** (`routing.Breaker`), following 07.2's rule: open after `threshold`
consecutive failures, fail fast for `cooldown` seconds, then let exactly one probe decide. (`scalelab`'s breaker trips
on a failure *ratio* in a window instead; either works per target — name the one you run.) A probe that ends in a
status saying nothing about the provider's health — a 400 for this request, our own 401 — decides nothing: the
breaker hands the probe back (`Breaker.release()`) so the next request probes, instead of staying half-open forever. 07.2's `FallbackChain`
labels a result served by a fallback as degraded; the gateway does the same with `Result.attempts` and a header.

A breaker learns only as fast as failures complete. With a 10 s timeout and 50 requests a second, the first
10 × 50 = 500 requests are all in flight before the breaker has seen three failures (502 with the three that trip
it; core notebook 02, exercise 2.4, counts it against `routing.Breaker`). Fast failure signals — connect
errors, a 503, a deadline on the *first byte* shorter than the whole-response timeout — shorten that window; that is
why the gateway times out on TTFT separately from end-to-end.

### 2.5 Chain availability, latency and cost

With independent failures, a chain fails only if every target fails; a common-mode failure (the gateway itself, one
region's egress, two aliases on one provider) takes the whole chain down at once
(`routing.chain_availability(avails, common_mode)` = (1 − c) × (1 − Π(1 − aᵢ))):

```
primary 99.5 %, fallback 99.0 %, independent:        1 − 0.005 × 0.01 = 99.995 %
the same, with a 0.1 % common-mode failure:          0.999 × 0.99995  = 99.895 %  (0.99895005)
```

The common-mode term dominates: a third independent target moves the independent part to 99.99995 % and the chain
stays at 99.9 %. Put fallbacks in *different* failure domains, and size their quota for the traffic they will take.

Expected latency and cost come from walking the chain (`routing.chain_cost(steps)`): each step succeeds with
probability p, and a failure's time is paid by every later outcome. In a bad hour the primary fails 5 % of requests;
the chain is gemini-3.5-flash (TTFT 0.6 s) → gpt-5.4-mini (0.5 s) → claude-haiku-4-5 (0.7 s), each call the §5.3
shape priced by `metering.price_call()`:

| How the primary fails | Mean time to first token | P(all fail) | Mean $ per request |
|---|---|---|---|
| fast (a 503 in 0.15 s) | 0.603 s | 5.0 × 10⁻⁶ | $0.006830 |
| slow (a 10 s timeout) | 1.095 s | 5.0 × 10⁻⁶ | $0.006830 |
| no fallback at all | 0.577 s, but 5 % of requests fail | 5 % | — |

The timeout adds 0.05 × (10 − 0.15) = 0.49 s to the *mean* and puts every failed request's p95 at ten seconds; the
breaker (§2.4) and a first-byte deadline bring it back. The fallbacks here are cheaper per call than the primary, so
the mean cost barely moves; a fallback to a *pricier* model at full traffic is how an outage doubles a bill (drill 1).

### 2.6 A self-hosted target is a pool

A target such as `self/lab/llm` is an OpenAI-compatible endpoint in front of many replicas — on Kubernetes, a Gateway
route to an `InferencePool` whose endpoint picker chooses the pod
([05 PRIMER §9](../../05-orchestrator/serving-orchestration/PRIMER.md#9-the-kubernetes-native-stack-september-2026)).
The gateway treats the pool as one target with one breaker; it does not see replicas, and should not: prefix
affinity and KV load are the router's signals. Two things cross the boundary. Priority: a tenant tier maps to the
pool's priority, and the directions differ — vLLM's `priority` is *lower = earlier* (and non-zero values error unless
the server runs `--scheduling-policy priority`), llm-d's `InferenceObjective.priority` is *higher = first* (§3.2
there), Envoy's `backendRefs.priority` 0 is the primary — so map tiers explicitly. And cost: a pool has no per-token
price; its bill is split by GPU time (§5.4). Hosted provider or own pool is a break-even on $ per million tokens against utilisation — the 01 primer's
[§8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens) turns a GPU-hour into
$/M and §5.3 below compares the two at one call shape; module 06.5 (the scaling lab) works the capacity side. A
Provisioned Throughput commitment is a target too, with its own quota and spill-over (scaling primer §3.5).

---

## 3. Caching at the gateway

### 3.1 Four caches, four keys

| Cache | Keyed on | Saves | Risks | Where |
|---|---|---|---|---|
| Exact response | SHA-256 of tenant + every field that changes the answer (`cache.exact_key()`) | the whole call | stale answers; personal data if the class is wrong | gateway |
| Semantic response | nearest cached query ≥ threshold τ in the tenant's namespace, entities equal (`cache.SemanticCache`) | the whole call, on paraphrases too | a near miss served someone else's answer | gateway |
| Provider prompt cache | the prompt's prefix, provider-side | ~90 % of the cached input's price, some TTFT | none to correctness; placement and TTL are the provider's | provider |
| Engine prefix cache | chained block hashes, first block salted per tenant | prefill compute, TTFT | a timing side channel between tenants without a salt | engine |

The last two are never wrong — they reuse computation, not answers — and they are taught elsewhere: provider prompt
caching is the scaling primer's §3.4 lever (the anchor call costs $0.01065 uncached and $0.007005 with 2,700 of its
5,000 input tokens cached, 34 % less, `metering.price_call()`), and the engine's block-hash prefix cache with its
per-tenant `cache_salt` is [serving-engine PRIMER §5](../../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching)
(module 04.3). The gateway's job for those two is layout and isolation: stable prefixes first, and a `cache_salt` per
tenant on every request to a vLLM pool (§6.4). The first two return a stored *answer*, and are this section's subject.

### 3.2 What is safe to cache

An answer may be reused only if it would be the same answer for this caller now. That excludes personal classes
("what is the status of my order 1234"), time-sensitive classes ("is the service down right now"), anything with
tools (tool calls act on the world), and sampled outputs where variety is the point. A gateway cannot infer the class
from the text — "How do I reset *my* password?" is an FAQ and contains "my" — so the **route declares it**
(`metadata.cache_class`, `cache.cacheable()`), and only allowlisted classes are looked up or stored. The lab's
notebook 03 shows why: its regex reads "How do I cancel my subscription?" as personal and "What is my plan limit?"
as general, so there a regex may only *veto* a declared shared class, never make a request cacheable. The key is
namespaced by the **verified tenant** (and by user for per-user classes); OSS Portkey's exact-cache key has no tenant
in it (§9), and LiteLLM's semantic cache scopes by key by default — check what your product does.

The exact key hashes what changes the answer — model alias, messages, tools, `response_format`, sampling parameters,
`max_completion_tokens`, `reasoning_effort` — and not what does not: `stream` (a cached answer can be re-streamed,
`Gateway._replay()`) or the caller's `user` field. It stores only complete answers (`finish_reason: stop`) and expires
them on a TTL; bump a version in the namespace when a model or prompt template changes.

### 3.3 Semantic caching, measured

Embed the query, find the nearest cached query in the tenant's namespace, serve its answer if the cosine similarity
clears a threshold τ — and guard exactly what an embedding cannot see: numbers, dates, codes and named entities must
match (`cache.entities()`; the [embeddings primer §15](../../07-application-agent-framework/retrieval-rag/embeddings-lab/docs/primer.md#15-retrieval-pipelines-for-rag-and-agents),
"Embeddings elsewhere in agent systems", and the [vector-databases primer §17](../../07-application-agent-framework/retrieval-rag/vector-databases-primer.md)
warn about exactly these false positives). The core's embedder is the 07.4 hashing embedder
([`ragkit.embed.HashingEmbedder`](../../07-application-agent-framework/retrieval-rag/rag-from-scratch/ragkit/embed.py),
reproduced vector for vector in a test): lexical, not semantic, so it is the floor a real embedder must beat.

`cache.sweep_thresholds()` runs the bundled labelled sample (`gwcore/data/traffic.json`: 15 cached questions, 23
paraphrases that should hit, 15 near misses that must not, 6 unrelated, 5 uncacheable) through the cache at each τ.
"Hits" is the share of paraphrases served the right answer; "false hits" is wrong answers served over all 44
cacheable lookups:

| τ | Hits, no guard | False hits, no guard | Hits, entity guard | False hits, entity guard |
|---|---|---|---|---|
| 0.60 | 100.0 % | 34.1 % | 95.7 % | 18.2 % |
| 0.80 | 47.8 % | 29.5 % | 43.5 % | 15.9 % |
| 0.90 | 21.7 % | 4.5 % | 17.4 % | 4.5 % |
| 0.95 | 17.4 % | 0.0 % | 13.0 % | 0.0 % |

Read it as a design review would. On this embedder near misses sit *closer* than paraphrases ("What were the Q3 2024
revenue figures?" scores 0.882 against the cached Q3 2025 question; "Is SSO included in the Pro plan?" scores 0.617
against its own answer), so no threshold gives many hits without false ones. The entity guard halves false hits at
low thresholds — it catches Q3 2024 vs Q3 2025, SCIM vs SSO, 403 vs 429, GBP vs EUR — but cannot see a word-level
difference ("Business" vs "Team" plan, "disable" vs "enable"), and it costs one correct hit: it reads the shouted
"HOW DO I CANCEL MY SUBSCRIPTION" as five codes. At τ = 0.95 only case and punctuation variants hit — work a
*normalised* exact key would do safely. A real embedder (at T1, for example a small model on vLLM's pooling runner or
all-MiniLM-L6-v2 on CPU, verify) raises the paraphrase scores, but near misses that differ in one word stay close in
any embedding. So: cache narrow, declared classes;
measure hit and false-hit rates on your own labelled traffic before choosing τ; add a verifier (an exact slot check,
or a cheap model) where a false hit is expensive.

### 3.4 Invalidation and tenants

Answers expire (`ExactCache(ttl)`, `SemanticCache(ttl)`), and a namespace version is bumped when the model, the prompt
template or the source data changes. Tenants never share a namespace (`tests/test_cache.py` checks that the same
question from another tenant misses). The expected value of a cache is hits × the call's price minus false hits ×
the cost of a wrong answer; at the §5.3 call a correct hit saves $0.007005, and a wrong order status given to another
customer costs more than any number of hits save (drill 2).

---

## 4. Streaming-aware rate limits

### 4.1 Why per-request charges fail

The scaling primer builds the token bucket (§5.1: rate = your share of the tier, capacity = the burst you tolerate, a
six-second burst by default, Redis + Lua once there are two instances) and admission with degrade levels (§5.3,
`scalelab.admission.AdmissionController`); CURRICULUM §3.4 names 06.3 its home. `gwcore.ratelimit.TokenBucket` follows
`scalelab.resilience.TokenBucket`'s rule exactly — a test drives both over the same arrivals — with an explicit clock
and a `try_acquire()` that refuses instead of waiting.

The problem is what the bucket charges. A request's tokens are unknown at admission — most of them are output that
has not been generated — and output length is heavy-tailed. The RL primer's thinking workload
([00.5 PRIMER §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)) is a
lognormal with median 1,500 and σ = 1: mean 2,473, p99 fifteen thousand. A bucket that charges a fixed estimate per
request — sized for yesterday's outputs, median 300 (mean 494.6, `ratelimit.lognormal_mean()`), plus a 1,500-token
prompt — admits the same *requests* per minute whatever they turn out to cost. When outputs grow, it lets through
(prompt + new mean) / (prompt + old mean) = 3,973 / 1,995 = **1.99×** the tokens it was sized for
(`ratelimit.overadmission_ratio()`), and the provider answers with 429s.

### 4.2 Reserve → stream → reconcile

Charge tokens, not requests, and charge them in three moves (`ratelimit.ReserveLimiter`):

```
admit      reserve = prompt + output bound;  admit only if  used(window) + reserved + reserve ≤ limit    admit()
stream     each chunk's tokens move from reserved to used                                               debit()
reconcile  at the end, debit what usage says was not yet debited; release what was reserved and unused finish()
```

If every reservation is a true upper bound — the gateway enforces the output cap it reserved, which is the hard cap —
then `used + reserved` never exceeds the limit (processing moves tokens from one term to the other; expiry and
release only lower it), so a provider counting the same tokens in the same window never sees more than the limit
(`tests/test_ratelimit.py` checks the invariant on random traffic). A stream cut before its usage chunk reconciles on
the tokens counted from its deltas, marked estimated (§5.2). Envoy AI Gateway's token costs (`llmRequestCosts`) and a
naive bucket debit only *after* the response, so concurrent long streams are never reserved. LiteLLM's v3 limiter
(`parallel_request_limiter_v3.py` at `849f303`) reserves the prompt estimate plus `max_tokens`; when the request has
none, a per-key configured estimate or else max(the input estimate, 1,024) — a quarter of its 4,096 default, lowered
to a quarter of the smallest configured TPM limit when that is smaller, in which case it also writes the lowered
bound into the request as `max_tokens` — and reconciles after the call (verify both).

The experiment below also assumes something about the provider: that it counts tokens as they are processed. Hosted
APIs may instead do their own reserve → reconcile, charging the requested output bound at admission — OpenAI's
rate-limit guide counts the larger of `max_tokens` and an estimate against TPM, and Anthropic estimates output tokens
per minute from `max_tokens` when a request starts and corrects it at the end (verify both: from memory, the docs
sites could not be read here). Against such a provider the gateway's reservation must follow the provider's rule:
reserve the `max_tokens` it actually sends upstream, or send upstream the bound it reserved. A gateway that reserves
prompt + 4,096 while forwarding a 16,384 cap is refused at the provider's door. The §4.3 numbers hold for providers
and self-hosted pools that meter processed tokens.

### 4.3 The experiment

`ratelimit.compare_buckets()` offers 12 requests a second for ten minutes to a provider key with a 1,000,000
tokens-per-minute limit (1,500-token prompts, 50 output tokens a second per stream; the provider is modelled as a
sliding 60-second window over processed tokens — simulated; each run continues past t = 600 s until its admitted
streams drain):

| Limiter | Admitted | Peak window ÷ limit | Seconds over the limit (whole run) | Tokens served ÷ limit |
|---|---|---|---|---|
| per-request bucket, yesterday's outputs (median 300) | 5,055 | 1.03 | 116 (between t = 59 and 600 s) | 1.00 |
| per-request bucket, thinking outputs (median 1,500) | 5,055 | 1.97 | 604 (t = 45–648 s, continuously) | 1.87 |
| reserve prompt + 4,096, debit, reconcile | 1,920 | 0.77 | 0 | 0.704 |
| reserve prompt + the 16,384 cap, debit, reconcile | 725 | 0.31 | 0 | 0.265 |

The per-request bucket cannot see the rollout: it admits exactly as many requests and pushes 1.97× the limit at the
peak; from the moment it binds (t = 45 s) the provider's window stays over the limit without a break until the
admitted thinking streams have drained (t = 648 s). Sized right on the mean it still sits over the limit for 116 of 600
seconds, because the tail is not the mean. Reserving the cap is exact and wasteful: a 16K reservation held for a minute-long stream strands most of the
budget (26.5 % served). Reserving an estimate and reconciling serves 70.4 % with no second over the limit here —
910,498 tokens overran their reservations and were debited as they streamed, so admission saw them — but that is
measured, not guaranteed: sweeping the reservation (notebook 04, exercise 4.3), 1,024 serves 98.9 % of the limit
but spends 56 seconds over it, 2,048 serves 89.3 % with 2 seconds over, and 4,096 is the smallest with none. Size the
reservation by simulation on your own output distribution — the cap where a provider 429 is unacceptable — and never
charge a per-request constant.

### 4.4 RPM and TPM, hierarchies and shared state

Providers enforce requests and tokens per minute together, so the gateway does too: a `ReserveLimiter` in tokens and
one in requests (reserve 1). Limits nest — key within tenant within org within provider key — and a request must fit
every level; `ratelimit.admit_all()` checks all levels, then commits all, so a refusal at the provider level leaves
nothing half-reserved at the tenant level. With several gateway replicas that check-and-commit must be one atomic step
on shared state: one Lua script on one Redis node (the keys of one script in one hash slot on a cluster), with the
refund at reconcile a second script (verify: Redis semantics were not read from a source here; LiteLLM's v3 limiter is a
working example with fixed-window counters).

### 4.5 The noisy neighbour

A shared provider key has one TPM. Without per-tenant limits, one tenant's thinking rollout consumes it and every
other tenant gets 429s it did nothing to earn; admission's degrade levels (scaling primer §5.3) then shed the wrong
traffic. Per-tenant token limits under the provider limit, with priority tiers above them (§6.4), keep one tenant's
tail inside its own budget; the lab's `bench.py` scripts tenants against shared fake providers (its notebook 04).

---

## 5. Metering, tracing and chargeback

### 5.1 The ledger row

One row per request, written at reconcile (`metering.LedgerRow`, `row_from_usage()`): request id, tenant, key id (a
hash prefix, never the key), model, provider, prompt / completion / cached / reasoning tokens, cost, an `estimated`
flag, status (`ok` or `cut`), TTFT, duration and the trace id. It lines up with the identity primer's audit event (§9)
and the identity lab's [`AuditEvent`](../identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py): who
(tenant, key), what (model, tokens), under which decision, and which trace to open. The ledger is billing-grade —
every row, retained like financial records — which is what separates it from traces (§5.6).

### 5.2 Usage is authoritative

The provider's `usage` is the bill; the gateway prices it and estimates only where it does not exist. Thinking bills
as output ([00.5 PRIMER §5](../../00-foundations/rl-and-thinking-models/PRIMER.md#5-thinking-models)): the §1.5 call
with 1,200 reasoning tokens costs $0.017805 on gemini-3.5-flash, 2.54× the $0.007005 of the same call billed on its
350 visible tokens alone (`metering.price_call()`) — which is what a gateway that reads Gemini's
`candidatesTokenCount` without `thoughtsTokenCount` would charge. Estimates (`api.estimate_tokens()`, four characters
a token, labelled) cover two cases only: admission (§4.2), and a stream cut before its usage chunk, billed on the
tokens counted from the deltas it relayed, never zero (`StreamAccumulator.output_estimate()`, `estimated=True`).

### 5.3 Dollars per million tokens

The price table in `providers.CATALOGUE` is dated 2026-09-26 (verify): the Gemini rows equal `scalelab.capacity.PRICES`
(checked 5 September, unchanged), and a test proves `metering.price_call()` equals `scalelab.capacity.cost_per_call` on
every one of those rows. Cache reads, cache writes and the rest of the prompt are three prices: on claude-haiku-4-5,
10,000 tokens written to the prompt cache cost $0.0125 at the 1.25 write rate, not the $0.0100 an input-rate ledger
would bill (`price_call(..., cache_write_tokens=)`). For the scaling primer's anchor call — 5,000 input tokens of which 2,700 cached, 350
output — blended over its 5,350 tokens (`metering.cost_per_million()`):

| Model | $ per call | Blended $ per 1M tokens |
|---|---|---|
| gemini-3.5-flash | $0.007005 | $1.3093 |
| claude-haiku-4-5 | $0.00432 | $0.8075 |
| gpt-5.4-mini | $0.0035025 | $0.6547 |
| gemini-3.5-flash-lite | $0.001646 | $0.3077 |

Self-hosted rows come from the 01 primer ([§8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens)):
`metering.self_hosted_per_million()` is `roofline.cost.cost_per_million_tokens()` — $/GPU-hour ÷ (tokens/s × 3,600 ×
utilisation) × 10⁶ — and a test reproduces §8.1's rows from `roofline.llm.decode()` itself: Llama-3.1-8B on an H100
Spot at $3.7/hour, batch 68, 6,846.5 tokens/s → **$0.150** per 1M output tokens at 100 % utilisation ($0.250 at 60 %);
on demand at $11 → $0.446; an L4 at $0.70 → $0.660. Those are *output* tokens at decode rate; prefill tokens are an
order of magnitude cheaper, which is the root of the hosted input/output price asymmetry and why the two rows are not
the same unit as the blended column. Compare hosted and self-hosted at your call shape, not per "token".

### 5.4 Chargeback

Hosted spend is charged back directly: the sum of each tenant's ledger costs. A self-hosted pool has one bill and no
per-token price, so it is split (`metering.chargeback()`). Splitting by tokens treats a prompt token like an output
token; splitting by GPU-seconds charges prompt tokens at the prefill rate (~68,000 tokens/s on the §5.3 H100, 01 PRIMER
§8.1) and output tokens at the decode rate (6,846.5 tokens/s). For a $10,000 month shared by a RAG tenant (90 M prompt,
1 M output tokens) and a thinking tenant (5 M prompt, 10 M output):

| Split by | RAG tenant | Thinking tenant |
|---|---|---|
| tokens | $8,584.91 (85.8 %) | $1,415.09 |
| GPU-seconds | $4,892.57 (48.9 %) | $5,107.43 |

Token-proportional chargeback makes the prompt-heavy tenant subsidise the output-heavy one. Measure GPU time from the
engine (vLLM's `vllm:request_prefill_time_seconds` and `vllm:request_decode_time_seconds` histograms) when it is
available; the rates above are the fallback.

### 5.5 Reconciliation

The ledger and the provider's usage export must agree, and drift has causes you can name: retries the provider billed
but the ledger recorded once (bill every attempt that reached the model), cut streams billed on estimates, cached
tokens priced wrongly, a tokenizer mismatch in estimates. `Ledger.reconcile()` compares per-model totals and flags
any field off by more than a tolerance (1 % by default); run it daily, alert on it beside cost per conversation (§5.10).

### 5.6 Traces are not the ledger: GenAI spans

Traces are sampled evidence for debugging; the ledger is the bill. The gateway emits one **SERVER** span per request
(`POST /v1/chat/completions`) and one **CLIENT** span per upstream target it tried (`chat {model}`), because different
providers are different operations; retries against one target stay inside that target's span (`gwcore.otel.Tracer`).
Attribute names follow the OpenTelemetry GenAI conventions, which are still at *development* stability and moving:
v1.41.0 (April 2026) is the last semantic-conventions release that defines them, and they now live in
`semantic-conventions-genai`, which has renamed `gen_ai.usage.cache_creation.input_tokens` to `…cache_write…` and
replaced the `gen_ai.client.token.usage` histogram with counters, both unreleased. `otel.py` pins the names the two
agree on — the same constants as the 07.2 lab's
[`tracing.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/observability/tracing.py)
(its notebook 09; a test compares them) plus `gen_ai.provider.name` (required; there is no well-known value for a
self-hosted server, so `vllm` is our choice), `gen_ai.response.model` (the model that actually answered — the alias goes in
`gen_ai.request.model` on the server span), `gen_ai.response.time_to_first_chunk` and `gen_ai.request.stream` — and
defaults to the released names where they disagree, keeping the map (`otel.RENAMED_ON_MAIN`). Spans are written as
OTLP/JSON, one export request per line, with hex ids, integer span kinds and nanosecond timestamps as strings, and
read back (`otel.read_jsonl()`). Metric labels stay low-cardinality (scaling primer §5.10): model, tenant, outcome —
never a request or conversation id.

---

## 6. Keys, tenants and isolation

### 6.1 Virtual keys

Apps get **virtual keys**, not provider keys (`keys.KeyStore`): stored only as a SHA-256 hash (as LiteLLM's
`hash_token` does), shown once, **scoped** (which aliases), **budgeted** (dollars — the scaling primer's §5.11 dollar
budget, per key; `authorize()` refuses at `max_budget`), limited (tokens per minute, §4), tiered, and **revocable**. A
leaked virtual key is one tenant's budget until it is revoked; a leaked provider key is every tenant's.

### 6.2 Provider keys only in the gateway

Provider keys live in a secret manager, readable only by the gateway's identity, and are injected on the way out —
the identity primer's **gateway path** (§5): the app never sees them, the same pattern as the sandbox's egress proxy
that injects credentials the sandboxed code never holds
([sandboxed-execution PRIMER §4](../../07-application-agent-framework/sandboxed-execution/PRIMER.md#4-network-and-secrets)).
Rotation is add, overlap, retire (`keys.ProviderKeys.rotate()`): the new key becomes current at once, the old stays
valid for an overlap window so in-flight requests and every gateway replica finish on it, then it expires. Never
expose an engine's port: vLLM's `--api-key` guards only paths under `/v1`, `/v2`, `/inference` and `/cohere`;
`/health`, `/metrics` and the rest stay open.

### 6.3 The tenant comes from the verified key

The tenant, tier and scopes are whatever the verified key maps to (`Gateway.handle()`, stage 1) — never a header or a
body field the caller sets, because anything the caller sets, an attacker sets. The gateway strips route metadata
before forwarding, and sets per-tenant fields itself.

### 6.4 Isolation by layer

| Layer | Isolated by | If it is missing |
|---|---|---|
| Limits | a `ReserveLimiter` per tenant under the provider key's (§4.4) | the noisy neighbour (§4.5) |
| Response caches | namespace = verified tenant (+ user for per-user classes) (§3.2) | one tenant's answer served to another |
| Engine prefix cache | `cache_salt` = base64url(HMAC-SHA256(gateway secret, tenant)), 43 characters (`keys.cache_salt()`) | a timing side channel across tenants (serving-engine PRIMER §5) |
| Ledger | tenant and key id on every row | spend nobody can bill |
| Traces | tenant attribute; content opt-in and redacted | one tenant's prompts on another's dashboard |
| Residency | the key's regions, applied as the chain's region filter (§2.2) | data processed outside the region |
| Priority | tier → pool priority, directions mapped (§2.6; 05 PRIMER §3.2) | the batch job ahead of the interactive agent |

vLLM v0.30.0 validates `cache_salt`: non-empty, at most 128 characters, none of `@ / \` or NUL
(`keys.valid_cache_salt()`); it salts the first block only and the chained hashes carry it forward. Derive it from
the verified tenant with a gateway secret, so it is unguessable and nobody can join another tenant's cache by
sending that tenant's salt.

### 6.5 In brief: the gateway's own identity

The gateway is itself a workload that calls providers, MCP servers and its own stores, so it needs a first-class
identity (identity primer §3.3–3.5): on SPIFFE, an X.509-SVID from the **Workload API** — gRPC over a Unix socket
named by `SPIFFE_ENDPOINT_SOCKET`, every call carrying the `workload.spiffe.io: true` metadata, `FetchX509SVID` a
server stream whose every message holds the full state (an SVID missing from a message was revoked). SPIRE issues
one-hour X.509-SVIDs by default and rotates when the remaining lifetime falls to half-life ± 10 % of the half-life:
between 1,620 and 1,980 seconds before expiry, 27–33 minutes (`keys.svid_rotation_window(3600)`). The agent re-draws
that jittered half-life on every check (`rotationutil.shouldRotateByHalf`), so with a check every few seconds rotation
lands near the top of the window, not uniformly across it: `keys.FakeWorkloadAPI`, which models a check every 5 s
(verify), rotates with a mean of 32.1 minutes left. A gateway pools long-lived TLS connections, and an established session
keeps the certificate of its handshake, so it must rebuild its TLS configuration from the stream and cycle
connections before the old SVID expires (verify: this is operational practice, not in the standards).

---

## 7. Guardrails and what they cost

### 7.1 Hooks and placements

A guardrail is a check at a **hook** — the input, a tool call, a tool result, the streamed output, the final output
(`guardrails.HOOKS`) — at a **placement** that decides its latency and exposure. The core's checker is a regex
screener (`guardrails.RegexScreener`, a labelled stand-in whose patterns are bounded: an unbounded `[\w]+@` is
quadratic on a long prompt, and a check on every request must not be the cheapest DoS). Real checkers are classifiers
such as Llama Prompt Guard 2 (22M and 86M parameters, a 512-token window, input-side) or Llama Guard (1B, 8B, 12B;
prompts and responses), or a managed service such as Model Armor (identity primer §4.1, §6.1: `INSPECT_ONLY` first).

| Placement | How | TTFT added | Exposure |
|---|---|---|---|
| inline input | check, then call the model | the check | none |
| parallel input | call the model at once; hold its first token until the verdict | max(0, check − model TTFT) | none |
| parallel with cancel | stream at once; cancel on a bad verdict | 0 | tokens before the verdict |
| held-back window | release output in windows of W tokens, each checked first | (W − 1) × ITL + check | none, per window |
| final | check the whole answer, then send it | the whole generation + check | none; no streaming |
| shadow | check off the path; log only | 0 | everything (measurement only) |

### 7.2 Latency

`guardrails.added_latency()` puts numbers on the table for a 300-token answer at TTFT 0.4 s and ITL 20 ms (end to end
6.38 s). An input check of 19.3 ms (Prompt Guard 2 22M on an A100 at 512 tokens, verify) adds 19.3 ms inline and
nothing in parallel, because it finishes long before the first token; the 86M model's 92.4 ms against a 50 ms TTFT (a
small self-hosted model, or a cached prefix) adds 42.4 ms if the first token is held, or lets 3 tokens through if the
stream is cancelled instead (`guardrails.exposed_tokens()`). On the output side, with an illustrative 150 ms check:

```
held-back window W = 200 tokens (NeMo Guardrails' default chunk size):  TTFT + 0.02 × 199 + 0.15 = + 4.13 s
held-back window W = 50 tokens:                                          TTFT + 0.02 × 49  + 0.15 = + 1.13 s
final check:                                                             TTFT + 0.02 × 299 + 0.15 = + 6.13 s
```

A held-back window only keeps up if each check finishes within W × ITL (4 s at W = 200), and it puts W tokens of
generation in front of every user's first token. NeMo Guardrails' streaming output rails default to
`stream_first: True` — tokens reach the client *before* the rail sees them — so its default is "parallel with
cancel", not a held-back window (verify when you configure it).

### 7.3 Dollars and false positives

A classifier on a dedicated GPU costs its GPU time: at $3.7 an hour for an A100 (verify), 92.4 ms a check one at a time
is $0.095 per 1,000 checks, and batched 16 at a time $0.0059 — *assuming* a batch of 16 finishes in about the same
92.4 ms, which the model card does not say (its figure is one 512-token classification); measure the batch latency
before you rely on it (`guardrails.cost_per_1k_checks()` divides by the concurrency you give it). False positives
compound: checking the input and the output of the scaling primer's 13 model calls per conversation is 26 checks,
and at a 1 % false-positive rate 23.0 % of conversations hit at least one false block; at 0.1 %, 2.57 %
(`guardrails.false_block_rate()` = 1 − (1 − fpr)^checks). That — not the dollars — is usually the binding cost, and it
is why screens start in inspect-only mode and why the rate is measured per placement.

### 7.4 Risk reduced, risk bounded

Guardrails are probabilistic; they **reduce** risk. What **bounds** it is deterministic and sits outside the model:
scoped keys and tokens, deny-by-default tool policy, egress allowlists, confirmation gates (identity primer §0, idea
4; the four enforcement points of §4.1; §6). A design review should hear both halves: "we screen input and output,
and a fully hijacked model still cannot exceed what its credentials allow". Prompt injection's canonical home is
06.6 (CURRICULUM §3.4); this section only places the check and prices it.

---

## 8. The gateway as an MCP client

### 8.1 Why the gateway is the client

Agents reach MCP servers through the gateway — the 07.2 lab's
[MCP egress gateway](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/mcp/gateway.py)
([notebook 05](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/05_mcp_server_client_gateway.ipynb))
enforces policy, screens arguments and results, forbids token passthrough and audits every call. The MCP server is an
OAuth 2.1 resource server (identity primer §7.1), so whoever calls it runs the **client** side of the MCP
authorization spec (revision 2026-07-28, verify; the 07.2 lab's `docs/MCP_REVISIONS.md` lists what changed per revision).
If the gateway is the client, tokens live in the gateway, per principal, and the agent never holds one. The 07.2
lab's [`agentlab/auth/oauth.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/auth/oauth.py)
([notebook 06](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/06_oauth_identity_propagation.ipynb))
already runs discovery → PKCE → code → token; `gwcore.mcp_authz` reproduces its `challenge_for` and adds only what a
gateway needs: Client ID Metadata Documents, refresh rotation with reuse detection, a token per (principal,
resource) with its scopes, and DPoP nonces.

### 8.2 The flow

```
 gateway ── call tool, no token ──────────────► MCP server
         ◄─ 401  WWW-Authenticate: Bearer resource_metadata="…/.well-known/oauth-protected-resource/mcp", scope="tickets:read"
 GET protected-resource metadata (RFC 9728)  → authorization_servers: [https://auth.example.com/tenant1]
 GET AS metadata (RFC 8414, else OIDC)        → issuer must equal the URL's issuer; refuse without S256; CIMD supported?
 authorize  client_id = https://gateway.example.com/oauth/client.json    (the Client ID Metadata Document)
            code_challenge = S256(verifier), resource = the MCP server's URL, scope = "tickets:read"
         ◄─ code + iss                        → RFC 9207: iss must equal the issuer exactly
 token      code + verifier + resource (+ a DPoP proof)
         ◄─ access token (aud = the MCP server), refresh token (rotated on every use)
 call tool  Authorization: Bearer … (or DPoP … + proof) ───────────────────► 200
```

Each step is a function in `mcp_authz` and a fixed rule in the spec (verify each against revision 2026-07-28):

- **Discovery** (`prm_urls()`): the 401's `resource_metadata` if present, else the path-aware well-known URI, then the
  root one — clients must support both. For the authorization server (`as_metadata_urls()`), an issuer with a path
  such as `https://auth.example.com/tenant1` is tried as RFC 8414 path insertion
  (`/.well-known/oauth-authorization-server/tenant1`), OIDC path insertion (`/.well-known/openid-configuration/tenant1`),
  then OIDC path appending (`/tenant1/.well-known/openid-configuration`); the document's `issuer` must equal the one
  the URL was built from.
- **Registration**: pre-registered, else a **Client ID Metadata Document** when the AS advertises
  `client_id_metadata_document_supported` (SHOULD), else Dynamic Client Registration (MAY, deprecated). The
  `client_id` is an HTTPS URL with a path whose document holds at least `client_id` (equal to the URL),
  `client_name` and `redirect_uris`, and no shared secret (`validate_cimd()`); the AS fetches it with a ~5 KB read cap,
  refuses special-use addresses, and never caches an invalid one.
- **PKCE** is S256 only, and the client **must refuse** when the metadata lacks `code_challenge_methods_supported`.
  RFC 7636 Appendix B pins it: verifier `dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk` → challenge
  `E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM` (`mcp_authz.s256()`, equal to 07.2's `challenge_for` in a test).
- **`resource`** (RFC 8707) goes in **both** the authorization and the token request, so the token's audience is
  that one MCP server and a token stolen from one server is refused at another.
- **`iss`** (RFC 9207): if the AS advertises it, a missing or different `iss` in the response is a mix-up; compare
  as plain strings (`iss_ok()`).

### 8.3 Tokens per principal and resource; refresh rotation

The gateway calls many MCP servers for many users, so its token store is keyed by **(principal, resource)** and
records the scopes each token carries (`MCPClient.tokens`); the reference Python SDK keeps one per server, which is
right for a desktop client and wrong for a gateway. Tokens are short-lived; refresh tokens for public clients
**rotate** — every refresh returns a new one and invalidates the old — and the AS keeps the family: if an old refresh
token is presented again, someone has a copy, so the AS **revokes the whole grant**, the active tokens included
(OAuth 2.1 §4.3.1; `FakeAS.token()`, tested by replaying a rotated token). Refresh requests carry `resource` too.

### 8.4 Step-up

A token that lacks a tool's scope gets `403` with `WWW-Authenticate: Bearer error="insufficient_scope", scope="…"`.
The client re-authorizes with the **union** of the scopes it held and the ones demanded — never only the new one,
which would drop what it already had — and retries a bounded number of times (`MCPClient.call()`, which gives up after
`max_steps`). The first authorization asks for the scope in the 401 challenge if there is one, else the metadata's
`scopes_supported`: least privilege first, step up when needed.

### 8.5 DPoP nonces

A sender-constrained token (identity primer §3.5) is bound to a key: the client sends `Authorization: DPoP <token>`
plus a `DPoP` proof — a JWT with `jti`, `htm`, `htu`, `iat`, and `ath` = base64url(SHA-256(token)) — signed by that
key. RFC 9449 lets servers demand a fresh **nonce** in the proof: an authorization server answers `400
{"error":"use_dpop_nonce"}` with a `DPoP-Nonce` header (§8 of the RFC), a resource server answers `401` with
`WWW-Authenticate: DPoP error="use_dpop_nonce"` and `DPoP-Nonce` (§9); the client retries once with the nonce and
keeps the latest one per server. DPoP comes from RFC 9449, **not** the MCP spec, which does not mention it. The
identity lab's `agentsec.identity.tokens.DPoP.proof()` accepts a `nonce` but its `verify()` never issues or checks
one; `mcp_authz` closes that gap. Its signer is pluggable, and the one shipped is an HMAC stand-in
(`mcp_authz.HMACSigner`), labelled non-conformant: RFC 9449 requires an asymmetric algorithm, and the standard
library has none. The lab signs with `cryptography` when it is installed.

---

## 9. Where to run it, and what to adopt

### 9.1 Tiers

| Tier | What runs | Cost |
|---|---|---|
| T0 in-process | all of `gateway-core`: providers, the chain, caches, limits, ledger, spans and the MCP flow on a virtual clock | free |
| T0 on localhost | [`gateway-lab`](gateway-lab/): an async OpenAI-compatible gateway over HTTP in front of fake providers, built like the 05 lab's [`igwlab` router](../../05-orchestrator/serving-orchestration/inference-gateway-lab/igwlab/router/server.py) | free |
| T0 + Docker | the lab's compose stack: the gateway and two fake providers, one set to fail | free |
| T1 | the gateway in front of one real vLLM (`vllm/vllm-openai:v0.30.0`, Qwen2.5-0.5B-Instruct, `--enable-prompt-tokens-details`) on a free Colab or Kaggle T4, with a fake provider as the fallback | free, or ~$0.3–0.7 an hour rented (verify) |
| T3 | the gateway on Cloud Run or GKE with the 04 lab's [Cloud Run or GKE vLLM](../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/) or the 05 lab's [GKE Inference Gateway](../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/) as upstreams; no new Terraform | pay per use; see [`COMPUTE.md`](../../COMPUTE.md) |

On Google Cloud the gateway is a stateless service (Cloud Run: requests up to 60 minutes, streams not capped, scaling
primer §5.7) with keys in Secret Manager, limits and caches in Memorystore and spans to Cloud Trace over OTLP; Apigee
and Agent Gateway are the managed pieces (scaling primer §9, identity primer §10; verify). Elsewhere the same pieces
map one to one:

| The gateway needs | On Google Cloud | Anywhere else |
|---|---|---|
| a stateless, multi-replica service | Cloud Run, GKE | any container host or Kubernetes cluster; locally the lab's compose stack or kind |
| shared buckets and caches (§4.4) | Memorystore | Redis or Valkey |
| provider keys (§6.2) | Secret Manager | Vault, or the cloud's own secret manager |
| spans (§5.6) | Cloud Trace over OTLP | any OTLP backend: Jaeger, Tempo, an OpenTelemetry Collector |
| a T1 upstream | the 04 lab's Cloud Run or GKE vLLM | a rented GPU (RunPod, Vast, Lambda; prices in `COMPUTE.md`, verify) running the lab's `deploy/any-gpu` |

### 9.2 Build or adopt

Most teams adopt a gateway and keep this primer as the checklist to evaluate it. What the upstream code at the pinned
versions shows (all verify — products move monthly):

| Concern | LiteLLM proxy 1.104.0 | Envoy AI Gateway v1.1.0 | Kong OSS 3.10.0 | Portkey gateway 1.15.2 |
|---|---|---|---|---|
| §1 adapters | many providers; OpenAI API | `AIServiceBackend` schemas (OpenAI, Anthropic, Bedrock, Vertex, …) | `ai-proxy`: `route_type` `llm/v1/chat`, providers incl. openai, anthropic, gemini | 78 provider directories |
| §2 fallbacks | `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, cooldowns | `backendRefs[].priority` + retry policy; an `InferencePool` backend leaves fallback to the EPP | load balancing and fallback in Enterprise `ai-proxy-advanced` | `strategy.mode` `fallback`, `on_status_codes` |
| §3 caching | exact and semantic (Redis, Qdrant) types; semantic cache scoped per key by default | not checked | semantic cache is Enterprise | `cache.mode` simple / semantic; OSS is exact only, key without a tenant |
| §4 token limits | `tpm_limit`/`rpm_limit` per key; v3 limiter reserves an estimate | `llmRequestCosts` feed a global rate limit, debited after the response | token rate limiting is Enterprise | not checked (retries honour `Retry-After`) |
| §5 metering | spend log per request; keys hashed | GenAI metrics (`gen_ai.client.token.usage`) | logs prompt/completion tokens and cost | not checked |
| §6 keys | virtual keys: models, budget, TPM/RPM | `BackendSecurityPolicy` holds upstream credentials | `auth` per plugin | virtual keys |
| §7 guardrails | `pre_call`, `during_call`, `post_call` and MCP-call hooks | not checked | `ai-prompt-guard` | input and output guardrail hooks |
| §8 MCP | MCP-call guardrail hooks; MCP gateway (verify) | `MCPRoute` | not in the OSS plugins read | not checked |

Adopting one does not remove the design decisions: which failures fall through, what is cacheable per route, what
a reservation is, who owns chargeback, where the tenant comes from, where each check sits. Ask each product the
questions of §1–§8 and test its answers with the lab's scripted outages and tenants.

---

## In a design review

**The two-minute walkthrough.** "Every app and agent calls one gateway with a virtual key; the key maps to a tenant,
a tier, a budget and scopes, and nothing the caller sends can change that. The gateway speaks OpenAI chat completions
over SSE to everyone and normalises each provider's usage and finish reasons through adapters that are mostly data —
Anthropic's input tokens exclude the cache, Gemini's thinking sits outside candidates, vLLM puts errors inside a 200.
Clients ask for aliases; an alias is an ordered chain of provider, model and region, filtered by capability, context
and residency, ordered by a policy, with a breaker per target. We fall through on 429, 5xx, timeouts and
context-too-long — never on a bad request or a policy refusal — and only before the first byte; after it we surface
the error. Chain availability is capped by whatever the targets share, so fallbacks go in different failure domains.
Caches: exact and semantic answers only for classes a route declares cacheable, namespaced by tenant, guarded on
numbers and entities, with the threshold chosen on labelled traffic; provider prompt caching and the engine's prefix
cache for everything else, isolated by a per-tenant `cache_salt` we derive. Limits are in tokens: reserve prompt plus
the output bound, debit as it streams, reconcile with usage — a per-request bucket let through twice the provider's
TPM the day thinking shipped. Every request writes a ledger row priced from usage, thinking billed as output, cut
streams on counted deltas; the self-hosted pool is charged back by GPU-seconds; traces carry pinned GenAI names but the
ledger is the bill. Provider keys live only in the gateway and rotate with an overlap. Guardrails are placed by what
they cost — inline input checks are cheap, held-back output windows cost a window of generation in TTFT, false
positives compound per conversation — and policy outside the model bounds what they miss. For MCP the gateway is the
OAuth client: discovery, a metadata-document client id, PKCE with resource, rotating refresh tokens, step-up with the
union of scopes, DPoP nonces — tokens per principal and resource, never in the agent."

**Drill questions**

1. *A provider outage lasted five minutes; our incident lasted forty, and the bill doubled.* — Clients retried without
   a budget, so the retry wave outlasted the outage; streams that failed midway were re-run from the start, paying
   twice for output already generated; and the whole chain fell through to a pricier model at full traffic until
   that model's quota ran out. Retry budgets with jitter, a breaker per target, fallback only before the first byte,
   fallback capacity sized and priced in advance in an independent failure domain, and a cost alert per tenant (§2).

2. *The semantic cache answered one user with another user's order status.* — The query was personal, a class that
   must never be cached, and the key had no tenant or user namespace; a lexical near match ("order 1234" against
   "order 1243") cleared the threshold. Declare cacheable classes per route, namespace by tenant, guard numbers and
   entities exactly, and measure false hits on labelled traffic before lowering τ. The engine's prefix cache is exact
   and isolated with `cache_salt` (§3, §6.4).

3. *After a thinking-model rollout the provider started returning 429s, but our request rate limit never tripped.* —
   The bucket charged per request while outputs grew by an order of magnitude with a heavy tail, so the same request
   rate carried twice the tokens per minute (1.99× by the closed form; in §4.3 the tokens served rose from 1.00× to
1.87× the limit, with the window peaking at 1.97×). Meter tokens: reserve
   at admission, debit as chunks stream, reconcile at the end, cap output; enforce TPM beside RPM per tenant; budget
   thinking rather than truncating it with `max_tokens` (§4).

4. *Why can't the gateway fall back to another model when a stream dies halfway?* — The client has already rendered
   part of an answer; a second model's continuation would read as one answer from two models, with no shared state.
   Surface an error chunk, meter what was relayed as an estimate, and let the agent (which owns the conversation)
   decide whether to retry the turn (§2.3, §5.2).

5. *The app sets an `X-Tenant` header and the gateway uses it for the cache namespace. What is wrong?* — Anything the
   caller sets, an attacker sets: one tenant can read another's cache, spend its budget and poison its answers. The
   tenant comes from the verified key; the cache namespace, limits, ledger and `cache_salt` derive from it (§6.3, §6.4).

6. *An MCP server returns 403 insufficient_scope for `close_ticket`. What does the gateway do, and what must it never
   do?* — Re-authorize that principal for that resource with the union of the scopes it holds and the one demanded
   (with PKCE and `resource`), store the new token under (principal, resource), and retry a bounded number of times.
   It must never pass the agent's own token through, ask for only the new scope (dropping the old ones), or reuse one
   principal's token for another (§8.3, §8.4).

---

## Glossary

| Term | Meaning |
|---|---|
| Alias | The model name a client asks for (`chat`); resolves to a fallback chain. |
| Fallback chain | An ordered list of (provider, model, region) targets tried until one answers. |
| Falls through | A failure another target could fix (429, 5xx, timeout, context length); the next target is tried. |
| Breaker | Per-target state machine: open after consecutive failures, one probe after the cooldown. |
| Common-mode failure | One failure that takes every target down (the gateway, a shared region). |
| Exact / semantic cache | Reuse a stored answer on an identical key / on a query similar above a threshold. |
| Entity guard | Refuse a semantic hit unless numbers, dates and codes match exactly. |
| `cache_salt` | vLLM request field that salts the first KV block so tenants never share prefix-cache entries. |
| Reserve → stream → reconcile | Token metering: hold an upper bound at admission, debit as tokens stream, release the rest. |
| Over-admission | Tokens let through beyond a provider's limit because the charge at admission was too small. |
| Ledger row | The billing record of one request, priced from the provider's `usage`. |
| Chargeback | Splitting a shared cost (a GPU pool) among tenants, by tokens or by GPU time. |
| Virtual key | A gateway-issued key: hashed at rest, scoped, budgeted, revocable; maps to a tenant. |
| SVID | A SPIFFE verifiable identity document (here X.509) issued to a workload through the Workload API. |
| Held-back window | Output released in windows of W tokens, each checked before release. |
| CIMD | Client ID Metadata Document: an HTTPS URL that is the OAuth client id and serves the client's metadata. |
| PKCE S256 | Proof Key for Code Exchange: the token request proves possession of the verifier behind the challenge. |
| Refresh rotation | Each refresh returns a new refresh token; reuse of an old one revokes the grant. |
| DPoP nonce | A server-supplied value a DPoP proof must carry (RFC 9449 §8–§9). |

---

## Sources

- OpenAI OpenAPI description, `openai/openai-openapi` `42eccf1` (2026-09-26): `CreateChatCompletionRequest`,
  `ChatCompletionStreamOptions`, `CreateChatCompletionStreamResponse`, `CompletionUsage`, `ErrorResponse`.
- vLLM v0.30.0 (`ced6857`): `vllm/entrypoints/openai/chat_completion/protocol.py` (`cache_salt`, `priority`,
  `validate_stream_options`), `vllm/entrypoints/generate/base/protocol.py` (`StreamOptions`), `chat_completion/serving.py`
  (errors inside the stream, `[DONE]`), `vllm/entrypoints/launchers/cli_args.py`, `vllm/v1/core/kv_cache_utils.py`.
- OpenTelemetry semantic conventions v1.41.0 (`model/gen-ai/`) and `semantic-conventions-genai` `e57c543`
  (`model/gen-ai/*.yaml`, `changelog.d/`); OTLP/JSON encoding (`opentelemetry-proto` `docs/specification.md`).
- Model Context Protocol specification, revision 2026-07-28, `basic/authorization/` (index, authorization server
  discovery, client registration, security considerations); the MCP Python SDK `src/mcp/client/auth/`.
- OAuth 2.1 (draft-ietf-oauth-v2-1, editor's copy) §4.1.1, §4.3.1, §7.5; draft-ietf-oauth-client-id-metadata-document;
  RFC 7636 (PKCE, Appendix B), RFC 8414, RFC 8707, RFC 9207, RFC 9728; RFC 9449 (DPoP; `danielfett/draft-dpop`).
- SPIFFE `standards/SPIFFE_Workload_API.md`, `SPIFFE_Workload_Endpoint.md`; SPIRE `doc/spire_server.md`,
  `doc/spire_agent.md`, `pkg/common/rotationutil/rotationutil.go`.
- LiteLLM 1.104.0 (`litellm/router.py`, `litellm/proxy/`, `litellm/caching/`, `model_prices_and_context_window.json`
  at `849f303`); Envoy AI Gateway v1.1.0 (`api/v1beta1/`, `site/docs/`); Kong 3.10.0 (`kong/plugins/ai-*`,
  `kong/llm/`); Portkey gateway 1.15.2 (`src/providers/`, `src/middlewares/`).
- Anthropic Python SDK 1.8.0 (`src/anthropic/types/`), Google GenAI Python SDK 2.25.0 (`google/genai/types.py`).
- NeMo Guardrails (`nemoguardrails/rails/llm/config.py`); PurpleLlama model cards (Llama Guard 3 and 4, Prompt Guard 2).
- In this repo: every primer and module linked above, and [`CURRICULUM.md`](../../CURRICULUM.md) §3.4's canonical homes.

---

## Verify list

Dated 26 September 2026. Re-check before relying on any of these.

| Fact | Status here |
|---|---|
| Hosted prices (USD per 1M: input / output / cached): gemini-3.5-flash 1.50 / 9.00 / 0.15; gemini-3.5-flash-lite 0.30 / 2.50 / 0.03; gpt-5.4-mini 0.75 / 4.50 / 0.075; claude-haiku-4-5 1.00 / 5.00 / 0.10 (cache write 1.25) | Gemini from `scalelab.capacity.PRICES` (5 Sep) and LiteLLM's price file @`849f303`, which agree; others from that file |
| Context windows: gpt-5.4-mini 272,000; claude-haiku-4-5 200,000; Gemini rows 1,048,576 | first two from LiteLLM's file; the Gemini value is not from a source here |
| GPU prices: H100 Spot $3.7/GPU-hour, on demand $11; L4 $0.70; A100 ~$3.7/hour | the repo's FACTS and 01 PRIMER §8.1, us-central1; see `COMPUTE.md` |
| `data: [DONE]` for chat completions; `x-ratelimit-*` headers | `[DONE]` only inside the `include_usage` description; `x-ratelimit-*` not in the OpenAPI description (prose docs only) |
| vLLM v0.30.0: `cache_salt` ≤ 128 characters without `@ / \` NUL, first block only; `priority` lower = earlier, needs `--scheduling-policy priority`; `--api-key` guards `/v1`, `/v2`, `/inference` and `/cohere` only (`GUARDED_PREFIX`) | read at the tag |
| OTel GenAI: v1.41.0 last release defining GenAI; `cache_write` and `gen_ai.client.inference.*` unreleased on genai main | read 2026-09-26; names will move |
| MCP authorization 2026-07-28: CIMD SHOULD, DCR MAY and deprecated; S256 refusal rule; `resource` in both requests; RFC 9207 `iss`; no DPoP | spec pages read at `ab3a39c` |
| OAuth 2.1 refresh-rotation reuse detection (§4.3.1) and which draft number is current | editor's copy; draft number (verify) |
| SPIRE: X.509-SVID TTL 1 h, JWT-SVID 5 min, rotation at half-life ± 10 %; `availability_target` ≥ 24 h | SPIRE docs and `rotationutil.go`; long-lived TLS rotation practice not in the standards |
| Prompt Guard 2: 22M 19.3 ms, 86M 92.4 ms per 512-token classification on an A100 | model card |
| NeMo Guardrails streaming output rails: `chunk_size` 200, `context_size` 50, `stream_first` True | `config.py` at `e549dda` |
| Product features in §9.2 (LiteLLM, Envoy AI Gateway, Kong OSS vs Enterprise, Portkey OSS cache) | read in each repo at the pinned commit; managed gateways (Apigee, Agent Gateway) from the repo's primers |
| Cloud Run request timeout up to 60 minutes; Cloud Trace via OTLP (`telemetry.googleapis.com`) | from the scaling primer §5.7 and §9 |
| Redis + Lua atomic check-and-commit; one hash slot per script on a cluster | not read from a Redis source here |
| How hosted providers count tokens against TPM/OTPM: OpenAI the larger of `max_tokens` and an estimate at admission; Anthropic output tokens estimated from `max_tokens` and corrected at the end | from memory; the providers' rate-limit pages could not be read here |
| LiteLLM v3 limiter: prompt + `max_tokens`, else max(input estimate, 1,024 = 4,096 ÷ 4, capped at a quarter of the smallest TPM limit), an implicit `max_tokens` when capped, reconcile after | `parallel_request_limiter_v3.py` @`849f303` |
| SPIRE agent checks rotation on its sync loop (default 5 s, backing off on errors) | `pkg/agent/manager/manager.go`; the fake's 5 s check is a model of it |
