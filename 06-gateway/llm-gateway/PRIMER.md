# The LLM gateway: one front door for many models — routing, fallbacks, caching, metering and tenant isolation

*This primer is in layer 06 of the stack (the control plane in front of models). A check on 26 September 2026
compared its facts with upstream sources. The first sources are the OpenAI OpenAPI description
(`openai/openai-openapi` at `42eccf1`) and vLLM at tag v0.30.0. Then come the OpenTelemetry GenAI conventions
(semantic-conventions v1.41.0 and semantic-conventions-genai at `e57c543`) and the MCP specification revision
2026-07-28. The check also used the OAuth 2.1 and Client ID Metadata Document drafts, the RFC 9449 source, and SPIFFE
and SPIRE.*

*Four sources are gateways: LiteLLM 1.104.0, Envoy AI Gateway v1.1.0, Kong 3.10.0 and Portkey gateway 1.15.2. The last
sources are the Anthropic and Google GenAI Python SDKs, NeMo Guardrails and PurpleLlama. The tag `(verify)` marks each
product fact that the check did not find in a source. The dated Verify list is at the end.*

*Each formula gives the name of the function in [`gateway-core`](gateway-core/) (package `gwcore`) that calculates
it. `gwcore` simulates every latency on a virtual clock. The detailed lab is [`gateway-lab`](gateway-lab/).*

An LLM gateway is the one service that every app, agent and tenant calls. They do not call the model providers
directly. The gateway holds the provider keys. It decides if a request can run, and which model, provider, region or
pool serves the request. It measures what the request cost, and it records what occurred.

This primer uses material that is already in the repo. It gives references to that material and does not repeat it:

- the [scaling primer](../scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md), for cost per
  conversation §3.4, quota strategy and the token bucket §5.1, and retries and breakers §5.2. It also covers admission
  and degrade levels §5.3, streaming §5.7, observability §5.10 and tenancy §5.12 (modules 06.1–06.3).
- the [identity primer](../identity-security/agentic-identity-gcp-lab/docs/primer.md), for agent identity §3.3,
  delegation §3.5, enforcement points §4.1, secrets §5 and screening §6.1. It also covers MCP as a resource server
  §7.1, tenancy §8 and audit §9 (module 06.6).
- the [05 orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md), for the three decisions §1.3,
  priorities §3.2, who decides what §3.3 and model routing §7.
- the [07.2 agent platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/), for
  its MCP egress gateway, OAuth flow, tracing and breakers.

Read those documents first. This primer assumes that you know them.

---

## The one-minute version

A gateway is worth its hop when many apps, tenants and providers meet. It puts one API (OpenAI-style chat
completions over server-sent events) in front of every provider. The differences between the providers go into
**adapters that are mostly data**.

The gateway makes the decisions that must occur one time, in one central place. It decides **if** a request runs:
the key, the tenant's budget and token limits, and a guardrail decide this. It also decides **which** model,
provider, region or pool serves the request: an alias resolves to an ordered **fallback chain**. Inside a self-hosted
pool, the 05 router selects the replica and the engine batches the requests, not the gateway.

The design has five mechanisms.

- **Fallback chains** fall through only on a failure where another target can succeed. These failures are a 429, a
  5xx, a timeout, or a context that is too long. They never fall through on a bad request or on a content-policy
  refusal. They **never** fall through **after the first byte** has reached the client. A breaker per target makes
  the gateway skip a provider that is down, with no wait.
- **Caches** save a whole call or a part of one. An exact or semantic cache saves the whole call. It has a namespace
  per tenant and a guard on numbers and entities, and you measure its false hits. Provider prompt caching and the
  engine's prefix cache save a part of the call.
- **Token limits** reserve the prompt plus an output bound at admission. They debit the tokens while the stream sends
  them, and they reconcile at the end. The reason is that the output length is unknown at the start and
  heavy-tailed.
- **Metering** puts a price on the provider's `usage` and writes one ledger row per request. The bill counts thinking
  tokens as output. Metering also divides the bill of a shared pool by GPU time.
- **Isolation** takes every tenant-scoped thing (limits, cache namespace, `cache_salt`, ledger, traces) from the
  verified key. The provider keys never leave the gateway.

The gateway has a real price: a hop on every request, a failure point in series with every provider, and every key
in one box. After this primer, you can walk through that design in a review and put numbers on each part.

---

## 1. One front door, one API

### 1.1 What the gateway owns

Without a gateway, every app holds provider keys, writes its own retry code, counts its own tokens and reports its own
spend. Thus nobody can answer "what did tenant X cost last week" or "rotate the OpenAI key" in one place. A gateway
owns these concerns for many apps, tenants and providers at the same time:

| Concern | What the gateway does | Primer section | `gwcore` module |
|---|---|---|---|
| Keys | issues virtual keys, holds provider keys and rotates them | §6 | `keys` |
| Policy | screens the input and the output, and refuses what the policy does not permit | §7 | `guardrails` |
| Limits | requests and tokens per minute per key, tenant and provider | §4 | `ratelimit` |
| Routing | resolves an alias to a fallback chain, applies filters and policies, operates breakers | §2 | `routing` |
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

The 05 primer says the same thing from below. It says that "model routing proper — choosing a cheaper model per
request — is a gateway decision". It also says that layer 05 "routes among replicas of the model already chosen"
(§7), with one `InferencePool` per base model. Thus a self-hosted target in a gateway chain is a *pool*, not a pod
(§2.6). Also, the gateway's admission control (scaling primer §5.3, `scalelab.admission`) sits above the router's
flow control.

### 1.3 The request path

`gwcore.gateway.Gateway.handle()` is the whole path, in order. Each stage can end the request with an OpenAI-shaped
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

The OpenAI chat-completions request (`model`, `messages`, `tools`, `max_completion_tokens`, `stream`, …) is the API that
every gateway, engine and SDK speaks. The API server of vLLM also speaks it (serving-engine PRIMER §1), and the 04 lab's
[`fakeserver.py`](../../04-inference-engine/serving-engine/vllm-serving-lab/servelab/fakeserver.py) emulates it. A
streamed answer is a sequence of `chat.completion.chunk` objects. Each object is an SSE frame `data: {…}` and a blank
line, and `data: [DONE]` ends the sequence (`api.sse()`, `api.parse_sse()`). Three details decide if a gateway does its
metering correctly:

- **Usage arrives only if asked.** With `stream_options.include_usage: true`, the server sends one more chunk before
  `[DONE]`. That chunk has `choices: []` and the request's `usage`. If the stream stops before its end, it is
  possible that this chunk never arrives (the OpenAPI description says so). Thus the gateway always asks upstream for
  the usage. It removes that chunk from the stream to a client that did not ask (`Gateway.handle()`, stage 6), with
  the same design as LiteLLM's proxy. It must ask only on streamed requests: vLLM rejects `stream_options` without
  `stream: true` with a 400.
- **The gateway accumulates a stream, and does not read each chunk alone.** It concatenates the content. Tool calls
  arrive as deltas with `index` as their key. Only the first delta carries the call's `id` and `function.name`. The
  `arguments` string arrives in fragments, and the fragments are not valid JSON until the last one arrives
  (`api.StreamAccumulator`). A cut stream leaves half a JSON object: report it, and never guess the rest (`tool_calls()`
  returns `valid_json: False`).
- **Errors can arrive inside a 200.** In vLLM v0.30.0, the server reports a failure in the middle of a stream as a
  `data:` chunk that holds an `error` object. Then it sends `[DONE]`, and all of this is under HTTP 200. Also, its
  error `code` is the HTTP status as an integer, but OpenAI's code is a string (`api.normalize_error()`). A relay that
  examines only the status counts a failure as a success in its metering.

The gateway's token stream is not the agent's own API. The 07.2 lab's [notebook
07](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/07_agent_api_streaming_tasks.ipynb)
sends a stream of *agent events* to an end user (`Last-Event-ID` resume, `Idempotency-Key`, a per-tenant 429). The
gateway relays *model tokens* to that agent. A resume across a gateway failure is the job of the agent API, not of the
gateway.

### 1.5 Adapters as data, and what does not normalise

Most of a provider adapter is a table (`providers.ADAPTERS`). The table gives the path and the auth header. It also
tells which usage fields add up to the canonical ones, and how the finish reasons map. The bundled samples
(`providers.PAYLOADS`, sample output in the documented format, illustrative) show one call, written in four ways. The
call has 5,000 prompt tokens (2,700 of them cached), 350 visible output tokens and 1,200 reasoning tokens.
For the four forms, `providers.normalize_usage()` maps every one of them to 5,000 prompt and 1,550 completion tokens:

| Canonical | OpenAI / vLLM | Anthropic Messages | Gemini |
|---|---|---|---|
| `prompt_tokens` (cached included) | `prompt_tokens` | `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` (2,300 + 2,700 + 0) | `promptTokenCount` (+ `toolUsePromptTokenCount`) |
| `cached_tokens` | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | `cachedContentTokenCount` |
| `cache_write_tokens` | `prompt_tokens_details.cache_write_tokens` | `cache_creation_input_tokens` | — |
| `completion_tokens` (reasoning included) | `completion_tokens` | `output_tokens` | `candidatesTokenCount` + `thoughtsTokenCount` (350 + 1,200) |
| `reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.thinking_tokens` | `thoughtsTokenCount` |
| auth header | `Authorization: Bearer` | `x-api-key` + `anthropic-version` | `x-goog-api-key` |

That table has two traps. Anthropic's `input_tokens` does *not* include cache reads and writes. Some price functions
expect the cached tokens inside the prompt count (`scalelab.capacity.cost_per_call`, `metering.price_call()`). If you
give `input_tokens` directly to such a function, it applies the cache discount two times. Gemini reports thinking
*outside* `candidatesTokenCount`. Thus a gateway that bills only the candidates bills a thinking call too low (§5.2).

Cache *writes* are prompt tokens too. But Anthropic gives them a price above the input rate (1.25× on
claude-haiku-4-5). Thus they get their own canonical field and their own price (§5.3). The vLLM server fills
`cached_tokens` only with `--enable-prompt-tokens-details`, and `reasoning_tokens` only with a `--reasoning-parser`.

The stream does not normalise cleanly. Anthropic sends named events: `message_start` with the input usage,
`content_block_delta` that carries `text_delta`, `input_json_delta` or `thinking_delta`, `message_delta` with
cumulative output usage, and `message_stop`. It sends no `[DONE]`. The Gemini API sends whole function calls, not
argument fragments (`partial_args` is Vertex-only), and it marks thought parts with `thought: true`. In vLLM, the
reasoning text has the name `reasoning` (vLLM renames a `reasoning_content` that it receives), and OpenAI's schema has
neither. `providers.normalize_stream()` changes the first two into canonical chunks.

The counts do normalise, but only at the end. Anthropic's usage is cumulative, and the final `message_delta` carries
all of it. This includes `output_tokens_details.thinking_tokens` when the API sends it (the field is optional in the
SDK's `MessageDeltaUsage`). Thus the §1.5 stream normalises to the same 1,550 completion and 1,200 reasoning tokens as
the non-streamed response.

The counts do not survive a cut stream that ends before that event. All that it holds is the usage of
`message_start`, and its `output_tokens` is 1. That is not a bill. Thus the normaliser emits no usage chunk for a cut
stream. The gateway makes an estimate from the deltas that it relayed, and it marks the row as estimated (§5.2). It
reconciles the difference later, and it does not invent a count.

### 1.6 What it costs

A gateway is a hop on every request, a failure point in series with every provider, and every key in one box. In
series, availabilities multiply. A gateway at 99.95 % in front of a chain that answers 99.995 % of the time (§2.5)
gives 99.945 % (`routing.chain_availability([0.995, 0.99], common_mode=0.0005)` = 0.999450025). Thus, operate the
gateway like the most critical service that you have:

- Run several replicas across zones.
- Keep no state in the process. Put the limits and caches in a shared store (§4.4).
- Keep the path fast. Measure its added latency. Give that latency a budget. The lab runs the gateway over HTTP on
  localhost, where you can measure the time of the hop.
- Keep the keys in a secret manager, which logs each access (§6).

---

## 2. Model routing and fallback chains

### 2.1 Aliases and chains

Clients ask for an alias (`chat`, `chat-cheap`, `embed`), never for a provider's model id. The providers retire model
ids on a schedule, and the ids move between tiers. The scaling primer's rule applies: model ids live in configuration
(§5.2). An alias resolves to an ordered chain of `(provider, model, region)` targets (`routing.Target`). As an
option, the chain can be different for each tenant tier (`chat@gold`, `Router.candidates(tier=…)`):

```
chat      → [google/gemini-3.5-flash (global), openai/gpt-5.4-mini, anthropic/claude-haiku-4-5]
chat@gold → [openai/gpt-5.4-mini, google/gemini-3.5-flash]
chat-eu   → [google/gemini-3.5-flash (europe-west4), self/lab/llm (europe-west4)]   residency: no global endpoint
```

### 2.2 Filters and policies

`Router.candidates()` first **filters** the targets. The filters compare these items:

- the request's needs with each target's capabilities (`routing.needs()`: tools, and reasoning when
  `reasoning_effort` is above minimal),
- the prompt plus the output with the context window,
- the region with the tenant's residency.

Then the router **orders** the targets that pass the filters. A policy sets the order:

| Policy | Orders by | Use when |
|---|---|---|
| `ordered` | the chain as written | a primary and its siblings (the default) |
| `cheapest` | blended $ for this request's prompt and output cap (`routing.blended_cost()`) | quality is equal across the chain |
| `ewma_ttft` | a per-target moving average of TTFT (`routing.ewma()`, $\alpha$ = 0.2, with TTFT as serving-engine PRIMER §11 defines it) | latency-sensitive routes. Unmeasured targets come first, so that they get a measurement. |
| `canary` | a stable hash of the request id against the targets' weights | a new model on a share of traffic |
| tier | a separate chain per tier (`chat@gold`) | paid tiers get the better or less contended model |
| effort | the reasoning filter of §2.2 | only thinking models take high `reasoning_effort` |

Routing by effort puts the RL primer's point into practice ([00.5 PRIMER
§7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)). The bill counts a thinking
token as output. Thus the lowest-cost way to buy accuracy is to send only the hard requests to a thinking model. This
routing needs a classifier, or a low-cost first pass, that knows which requests are hard.

### 2.3 What falls through, and what must not

Fall through only when another target has a real chance to succeed where this one failed
(`routing.falls_through()`):

| Falls through | Why | Must not fall through | Why |
|---|---|---|---|
| 429 | that provider's quota. A sibling has its own pool. | 400 bad request | every target gets the same bad request |
| 500, 502, 503, 504 | that provider is not healthy | 401, 403 | your credentials or scopes, not their health |
| 529 `overloaded_error` (Anthropic) | that provider is overloaded. Try a different target. | 404 | an incorrect model id is a configuration bug |
| 408, timeout / no answer | that provider is slow or down | content-policy refusal | a fall-through turns a refusal into another model's answer and hides the refusal |
| 400 `context_length_exceeded` | a longer-context target can take it | | |

Sometimes a design falls through on a content-policy refusal on purpose (LiteLLM has `content_policy_fallbacks`). If
you select this behaviour, make it a named policy per route with an owner, not a default.

**Fall back only before the first byte.** When a chunk has reached the client, the gateway cannot join the output of
a second model to the first. Such a join makes the client read two answers as one. After the first byte, the gateway
shows the failure to the client as an error chunk and `[DONE]`. It records the usage of the tokens that it relayed
(§5.2).

Before the first byte, the gateway has sent nothing, and it tries the next target. This applies to a 5xx, a timeout,
or an error chunk that arrives *first* inside a 200 (`Gateway.handle()`, stage 6, with a test in
`tests/test_gateway_otel.py`).

### 2.4 Retries and breakers

Retries with full jitter, `Retry-After` and a deadline come from the scaling primer (§5.2,
`scalelab.resilience.backoff`, `call_with_retries`). The 07.2 lab's [notebook
10](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/10_reliability_retries_breakers.ipynb)
teaches the full state machine of the breaker
([`agentlab/reliability/breaker.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/reliability/breaker.py)).

The gateway applies one breaker **per target** (`routing.Breaker`), with the rule of 07.2. The breaker opens after
`threshold` consecutive failures, and it fails fast for `cooldown` seconds. Then it lets exactly one probe decide.
The breaker of `scalelab` opens on a failure *ratio* in a window instead. Each of the two works per target. Name the
one that you run.

Some statuses say nothing about the provider's health, for example a 400 for this request or our own 401. A probe
that ends in such a status decides nothing. The breaker gives the probe back (`Breaker.release()`), so that the next
request does the probe. Otherwise, the breaker stays half-open forever. The `FallbackChain` of 07.2 labels a result
from a fallback as degraded. The gateway does the same with `Result.attempts` and a header.

A breaker learns only as fast as failures complete. Take a 10 s timeout and 50 requests a second. Then the first
10 × 50 = 500 requests are all in flight before the breaker sees three failures (502 with the three that trip it).
[Core notebook 02](gateway-core/notebooks/02_routing_and_fallback_chains.ipynb), exercise 2.4, counts this against
`routing.Breaker`.

Fast failure signals make that window shorter. Examples are connect errors, a 503, and a deadline on the *first
byte* that is shorter than the whole-response timeout. That is why the gateway times out on TTFT separately from the
end-to-end timeout.

### 2.5 Chain availability, latency and cost

With independent failures, a chain fails only if every target fails. A common-mode failure makes the whole chain fail
at the same time. Examples are the gateway itself, the egress of one region, and two aliases on one provider.
`routing.chain_availability(avails, common_mode)` calculates
$(1 - c) \times \left(1 - \prod_i (1 - a_i)\right)$:

```
primary 99.5 %, fallback 99.0 %, independent:        1 − 0.005 × 0.01 = 99.995 %
the same, with a 0.1 % common-mode failure:          0.999 × 0.99995  = 99.895 %  (0.99895005)
```

The common-mode term has the largest effect on the result. A third independent target moves the independent part to
99.99995 %, but the chain stays at 99.9 %. Put fallbacks in *different* failure domains. Give them a quota that is
sufficient for the traffic that they will take.

The expected latency and cost come from a walk along the chain (`routing.chain_cost(steps)`). Each step succeeds with
probability $p$, and every later outcome pays the time of a failure. In a bad hour, the primary fails 5 % of
requests. The chain is gemini-3.5-flash (TTFT 0.6 s), then gpt-5.4-mini (0.5 s), then claude-haiku-4-5 (0.7 s).
Each call has the §5.3 shape, and `metering.price_call()` gives its price:

| How the primary fails | Mean time to first token | P(all fail) | Mean $ per request |
|---|---|---|---|
| fast (a 503 in 0.15 s) | 0.603 s | 5.0 × 10⁻⁶ | $0.006830 |
| slow (a 10 s timeout) | 1.095 s | 5.0 × 10⁻⁶ | $0.006830 |
| no fallback at all | 0.577 s, but 5 % of requests fail | 5 % | — |

The timeout adds 0.05 × (10 − 0.15) = 0.49 s to the *mean*. It also puts the p95 of every failed request at ten
seconds. The breaker (§2.4) and a first-byte deadline bring it back. The fallbacks here have a lower cost per
call than the primary, so the mean cost almost does not change. But a fallback to a *higher-cost* model at full
traffic is how an outage doubles a bill (drill 1).

### 2.6 A self-hosted target is a pool

A target such as `self/lab/llm` is an OpenAI-compatible endpoint in front of many replicas. On Kubernetes, it is a
Gateway route to an `InferencePool`, and the endpoint picker of that pool selects the pod
([05 PRIMER §9](../../05-orchestrator/serving-orchestration/PRIMER.md#9-the-kubernetes-native-stack-september-2026)).
The gateway treats the pool as one target with one breaker. It does not see the replicas, and that is correct: prefix
affinity and KV load are the signals of the router.

Two things cross the boundary. The first is priority. A tenant tier maps to the priority of the pool, but the
directions are different:

- vLLM's `priority` is *lower = earlier*. A non-zero value causes an error unless the server runs
  `--scheduling-policy priority`.
- llm-d's `InferenceObjective.priority` is *higher = first* (§3.2 of the 05 primer).
- Envoy's `backendRefs.priority` 0 is the primary.

Thus, map the tiers explicitly. The second thing is cost. A pool has no per-token price, and chargeback divides its
bill by GPU time (§5.4).

The choice between a hosted provider and your own pool is a break-even on $ per million tokens against utilisation. The
01 primer's [§8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens) changes a
GPU-hour into $/M, and §5.3 compares the two at one call shape. Module 06.5 covers the capacity side, in the scaling
lab's [Mistral scaling primer
§3.5–3.6](../scaling-admission-cost/agentic-scaling-lab/docs/mistral/01-scaling-primer.md#35-self-hosted-the-replica-then-the-fleet)
and its [notebook 05](../scaling-admission-cost/agentic-scaling-lab/notebooks/05_hosted_or_own_gpus.ipynb). A
Provisioned Throughput commitment is a target too, with its own quota and spill-over (scaling primer §3.5).

---

## 3. Caching at the gateway

### 3.1 Four caches, four keys

| Cache | Keyed on | Saves | Risks | Where |
|---|---|---|---|---|
| Exact response | SHA-256 of tenant + every field that changes the answer (`cache.exact_key()`) | the whole call | stale answers, and personal data if the class is incorrect | gateway |
| Semantic response | nearest cached query ≥ threshold $\tau$ in the tenant's namespace, entities equal (`cache.SemanticCache`) | the whole call, on paraphrases too | a near miss that gets someone else's answer | gateway |
| Provider prompt cache | the prompt's prefix, provider-side | ~90 % of the cached input's price, some TTFT | none to correctness. The provider controls placement and TTL. | provider |
| Engine prefix cache | chained block hashes, a salt per tenant on the first block | prefill compute, TTFT | a timing side channel between tenants without a salt | engine |

The last two caches are never incorrect, because they reuse computation, not answers. Other primers teach them. Provider
prompt caching is the §3.4 lever of the scaling primer. The anchor call costs $0.01065 uncached and $0.007005 with 2,700
of its 5,000 input tokens cached, 34 % less (`metering.price_call()`). The engine's block-hash prefix cache with its
per-tenant `cache_salt` is in [serving-engine PRIMER
§5](../../04-inference-engine/serving-engine/PRIMER.md#5-prefix-caching) (module 04.3).

For those two caches, the job of the gateway is layout and isolation. It puts stable prefixes first, and it sends a
`cache_salt` per tenant on every request to a vLLM pool (§6.4). The first two caches return a stored *answer*. They
are the subject of this section.

### 3.2 What is safe to cache

The gateway can reuse an answer only if it is the same answer for this caller now. That rule excludes personal
classes ("what is the status of my order 1234") and time-sensitive classes ("is the service down right now"). It
also excludes anything with tools, because tool calls act on the world. It also excludes sampled outputs where
variety is the point.

A gateway cannot tell the class from the text. For example, "How do I reset *my* password?" is an FAQ, but it
contains "my". Thus the **route declares the class** (`metadata.cache_class`, `cache.cacheable()`). The gateway looks
up or stores only the classes on the allowlist.

The lab's [notebook 03](gateway-lab/notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) shows why. Its regex reads
"How do I cancel my subscription?" as personal and "What is my plan limit?" as general. Thus, in that notebook, a
regex can only *veto* a declared shared class. It never makes a request cacheable.

The namespace of the key contains the **verified tenant**, and for per-user classes it also contains the user. OSS
Portkey's exact-cache key has no tenant in it (§9). LiteLLM's semantic cache uses the virtual key of the caller as its
scope by default. Find out what your product does.

The exact key is a hash of the fields that change the answer: the model alias, messages, tools, `response_format`,
sampling parameters, `max_completion_tokens` and `reasoning_effort`. It does not include the fields that do not change
the answer. These are `stream` (the gateway can send a cached answer again as a new stream, `Gateway._replay()`) and
the caller's `user` field. The cache stores only complete answers (`finish_reason: stop`) and expires them on a TTL.
When a model or a prompt template changes, increase a version in the namespace.

### 3.3 Semantic caching, measured

Embed the query. Find the nearest cached query in the tenant's namespace. If the cosine similarity is at or above a
threshold $\tau$, serve its answer. Also guard exactly what an embedding cannot see: numbers, dates, codes and named
entities must match (`cache.entities()`). The [embeddings primer
§15](../../07-application-agent-framework/retrieval-rag/embeddings-lab/docs/primer.md#15-retrieval-pipelines-for-rag-and-agents),
"Embeddings elsewhere in agent systems", and the [vector-databases primer
§17](../../07-application-agent-framework/retrieval-rag/vector-databases-primer.md) warn about exactly these false
positives.

The core's embedder is the 07.4 hashing embedder
([`ragkit.embed.HashingEmbedder`](../../07-application-agent-framework/retrieval-rag/rag-from-scratch/ragkit/embed.py)),
and a test reproduces it vector for vector. It is lexical, not semantic. Thus it is the floor that a real embedder
must beat.

`cache.sweep_thresholds()` sends the bundled labelled sample (`gwcore/data/traffic.json`) through the cache at each
$\tau$. The sample has 15 cached questions, 23 paraphrases that should hit, 15 near misses that must not, 6 unrelated,
5 uncacheable. "Hits" is the share of paraphrases that get the correct answer. "False hits" is the incorrect answers
served over all 44 cacheable lookups:

| $\tau$ | Hits, no guard | False hits, no guard | Hits, entity guard | False hits, entity guard |
|---|---|---|---|---|
| 0.60 | 100.0 % | 34.1 % | 95.7 % | 18.2 % |
| 0.80 | 47.8 % | 29.5 % | 43.5 % | 15.9 % |
| 0.90 | 21.7 % | 4.5 % | 17.4 % | 4.5 % |
| 0.95 | 17.4 % | 0.0 % | 13.0 % | 0.0 % |

Read the table as a design review reads it. On this embedder, near misses are *nearer* to the cached questions than the
paraphrases are. "What were the Q3 2024 revenue figures?" scores 0.882 against the cached Q3 2025 question. "Is SSO
included in the Pro plan?" scores 0.617 against its own answer. Thus no threshold gives many hits without false hits.

The entity guard halves the false hits at low thresholds. It finds Q3 2024 against Q3 2025, SCIM against SSO, 403
against 429 and GBP against EUR. But it cannot see a word-level difference ("Business" against "Team" plan, "disable"
against "enable"). It also costs one correct hit, because it reads the shouted "HOW DO I CANCEL MY SUBSCRIPTION" as
five codes. At $\tau$ = 0.95, only case and punctuation variants hit. A *normalised* exact key can do that work
safely.

A real embedder increases the paraphrase scores (at T1, for example a small model on vLLM's pooling runner or
all-MiniLM-L6-v2 on CPU, verify). But near misses that differ in one word stay near the cached query in any embedding.
Thus:

- Cache narrow, declared classes.
- Measure the hit and false-hit rates on your own labelled traffic before you select $\tau$.
- Where a false hit has a high cost, add a verifier (an exact slot check, or a low-cost model).

### 3.4 Invalidation and tenants

Answers expire (`ExactCache(ttl)`, `SemanticCache(ttl)`). When the model, the prompt template or the source data
changes, the namespace version increases. Tenants never share a namespace (`tests/test_cache.py` makes sure that the
same question from another tenant misses). The expected value of a cache is

$$
\text{hits} \times \text{the call's price} - \text{false hits} \times \text{the cost of a wrong answer};
$$

at the §5.3 call, a correct hit saves $0.007005. But an incorrect order status that goes to another customer costs
more than any number of hits save (drill 2).

---

## 4. Streaming-aware rate limits

### 4.1 Why per-request charges fail

The scaling primer builds the token bucket (§5.1). There, the rate is your share of the tier, and the capacity is the
burst that you accept. The default is a six-second burst, and the bucket uses Redis + Lua when there are two or more
instances. The scaling primer also builds admission with degrade levels (§5.3,
`scalelab.admission.AdmissionController`). CURRICULUM §3.4 names 06.3 as its home.

`gwcore.ratelimit.TokenBucket` uses exactly the rule of `scalelab.resilience.TokenBucket`, and a test drives both
over the same arrivals. `gwcore.ratelimit.TokenBucket` has an explicit clock and a `try_acquire()` that refuses and
does not wait.

The problem is what the bucket charges. At admission, the tokens of a request are unknown, because most of them are
output that the model has not generated yet. Also, the output length is heavy-tailed. The thinking workload of the RL
primer ([00.5 PRIMER §7](../../00-foundations/rl-and-thinking-models/PRIMER.md#7-what-thinking-does-to-serving)) is a
lognormal with median 1,500 and $\sigma$ = 1. The lognormal gives these values: mean 2,473, p99 fifteen thousand.

Take a bucket that charges a constant estimate per request. The estimate fits yesterday's outputs, median 300 (mean
494.6, `ratelimit.lognormal_mean()`), plus a 1,500-token prompt. This bucket admits the same number of *requests* per
minute, whatever their real cost is. When the outputs grow, it lets through
$(\text{prompt} + \text{new mean}) / (\text{prompt} + \text{old mean})$ = 3,973 / 1,995 = **1.99×** the tokens that you
calculated its size for (`ratelimit.overadmission_ratio()`). Then the provider answers with 429s.

### 4.2 Reserve → stream → reconcile

Charge tokens, not requests. Charge them in three moves (`ratelimit.ReserveLimiter`):

1. **admit**: $\text{reserve} = \text{prompt} + \text{output bound}$. Admit the request only if
   $\text{used}(\text{window}) + \text{reserved} + \text{reserve} \le \text{limit}$ (`admit()`).
2. **stream**: the tokens of each chunk move from reserved to used (`debit()`).
3. **reconcile**: at the end, debit the rest of the usage. Release the reserved tokens that the request did not use
   (`finish()`).

Suppose that every reservation is a true upper bound: the gateway enforces the output cap that it reserved, and that
cap is the hard cap. Then $\text{used} + \text{reserved}$ never goes above the limit. When the model processes tokens,
they move from one term to the other, and expiry and release only decrease the sum. Thus a provider that counts the
same tokens in the same window never sees more than the limit. `tests/test_ratelimit.py` examines this invariant on
random traffic. A stream cut before its usage chunk reconciles on the tokens counted from its deltas, and the gateway
marks it as estimated (§5.2).

Envoy AI Gateway's token costs (`llmRequestCosts`) and a simple bucket debit only *after* the response, so they never
reserve concurrent long streams.

LiteLLM's v3 limiter (`parallel_request_limiter_v3.py` at `849f303`) reserves the prompt estimate plus `max_tokens`.
When the request has no `max_tokens`, it reserves a per-key configured estimate, or else max(the input estimate, 1,024).
That 1,024 is a quarter of its 4,096 default, lowered to a quarter of the smallest configured TPM limit when that is
smaller. In that case, it also writes the lowered bound into the request as `max_tokens`. Then it reconciles after the
call. Both of these behaviours of the LiteLLM limiter need a check (verify both).

The experiment in §4.3 also assumes something about the provider: it counts tokens when the model processes them. It is
possible that a hosted API does its own reserve and reconcile instead, and charges the requested output bound at
admission. OpenAI's rate-limit guide counts the larger of `max_tokens` and an estimate against TPM. Anthropic estimates
output tokens per minute from `max_tokens` when a request starts. It corrects the estimate at the end. Both of these
provider facts are from memory (verify both: the docs sites were not available for this primer).

Against such a provider, the reservation of the gateway must obey the rule of the provider. Reserve the `max_tokens`
that the gateway actually sends upstream, or send upstream the bound that it reserved. Take a gateway that reserves
prompt + 4,096 but forwards a 16,384 cap. The provider refuses its requests at the door. The §4.3 numbers are correct
for providers and self-hosted pools that count the processed tokens.

### 4.3 The experiment

`ratelimit.compare_buckets()` sends 12 requests a second for ten minutes to a provider key with a 1,000,000
tokens-per-minute limit. The prompts have 1,500 tokens, and each stream gives 50 output tokens a second. In this
simulation, the provider is a sliding 60-second window over processed tokens (simulated). Each run continues past t =
600 s until its admitted streams drain:

| Limiter | Admitted | Peak window ÷ limit | Seconds over the limit (whole run) | Tokens served ÷ limit |
|---|---|---|---|---|
| per-request bucket, yesterday's outputs (median 300) | 5,055 | 1.03 | 116 (between t = 59 and 600 s) | 1.00 |
| per-request bucket, thinking outputs (median 1,500) | 5,055 | 1.97 | 604 (t = 45–648 s, continuously) | 1.87 |
| reserve prompt + 4,096, debit, reconcile | 1,920 | 0.77 | 0 | 0.704 |
| reserve prompt + the 16,384 cap, debit, reconcile | 725 | 0.31 | 0 | 0.265 |

The per-request bucket cannot see the rollout. It admits exactly as many requests, and it pushes 1.97× the limit at the
peak. Then, from the moment it binds (t = 45 s), the provider's window stays over the limit with no break. The window
stays over the limit until the admitted thinking streams have drained (t = 648 s). The per-request bucket for
yesterday's outputs has the correct size for the mean, but it still stays over the limit for 116 of 600 seconds. The
reason is that the tail is not the mean.

A reservation of the full cap is exact and wasteful. A 16K reservation held for a minute-long stream leaves most of
the budget unused (26.5 % served).

A reservation of an estimate, with a reconcile at the end, serves 70.4 % with no second over the limit here. In this
run, 910,498 tokens overran their reservations, and the gateway debited them as they arrived, so admission saw them.
But that result comes from a measurement, not a guarantee. Exercise 4.3 of [core notebook
04](gateway-core/notebooks/04_token_limits_metering_and_chargeback.ipynb) does a sweep of the reservation. 1,024
serves 98.9 % of the limit but spends 56 seconds over it, 2,048 serves 89.3 % with 2 seconds over, and 4,096 is the
smallest with none.

Find the size of the reservation by simulation on your own output distribution. Where a provider 429 is not
acceptable, reserve the cap. Never charge a per-request constant.

### 4.4 RPM and TPM, hierarchies and shared state

Providers enforce requests and tokens per minute together, so the gateway does too. It has a `ReserveLimiter` in
tokens and one in requests (reserve 1). Limits nest: the key is inside the tenant, the tenant is inside the org, and
the org is inside the provider key. A request must fit every level. `ratelimit.admit_all()` examines all the levels
and then commits all of them. Thus a refusal at the provider level leaves nothing half-reserved at the tenant level.

With several gateway replicas, that check-and-commit must be one atomic step on shared state. It is one Lua script on
one Redis node, and on a cluster the keys of one script are in one hash slot. The refund at reconcile is a second
script (verify: the fact check of this primer did not read the Redis semantics from a source). LiteLLM's v3 limiter
is an example that works, with fixed-window counters (verify).

### 4.5 The noisy neighbour

A shared provider key has one TPM. Without per-tenant limits, the thinking rollout of one tenant uses all of it. Then
every other tenant gets 429s that it did nothing to cause. The degrade levels of admission (scaling primer §5.3) then
shed the incorrect traffic, the traffic of tenants that did not cause the 429s. Per-tenant token limits under the provider limit, with priority tiers above them (§6.4),
keep the tail of one tenant inside its own budget. The lab's `bench.py` runs scripted tenants against shared fake
providers (its [notebook 04](gateway-lab/notebooks/04_streaming_limits_metering_and_chargeback.ipynb)).

---

## 5. Metering, tracing and chargeback

### 5.1 The ledger row

The gateway writes one row per request at reconcile (`metering.LedgerRow`, `row_from_usage()`). The row has the
request id, the tenant, the key id (a hash prefix, never the key), the model and the provider. It also has the
prompt / completion / cached / reasoning tokens, the cost, an `estimated` flag and the status (`ok` or `cut`). It also
has the TTFT, the duration and the trace id.

The row agrees with the audit event of the identity primer (§9) and with the identity lab's
[`AuditEvent`](../identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py). Like them, the row records
who (tenant, key), what (model, tokens), under which decision, and which trace to open. The ledger is billing-grade:
you keep every row, as you keep financial records. That is the difference between the ledger and traces (§5.6).

### 5.2 Usage is authoritative

The provider's `usage` is the bill. The gateway puts a price on it, and it makes an estimate only where no `usage`
exists. The bill counts thinking as output ([00.5 PRIMER
§5](../../00-foundations/rl-and-thinking-models/PRIMER.md#5-thinking-models)). The §1.5 call with 1,200 reasoning tokens
costs $0.017805 on gemini-3.5-flash, 2.54× the $0.007005 of the same call billed on its 350 visible tokens alone.
`metering.price_call()` gives both prices. A gateway that reads Gemini's `candidatesTokenCount` without
`thoughtsTokenCount` charges the lower price.

Estimates (`api.estimate_tokens()`, four characters a token, labelled) cover only two cases. The first case is
admission (§4.2). The second case is a stream cut before its usage chunk. The gateway bills that stream on the tokens
counted from the deltas that it relayed, never on zero (`StreamAccumulator.output_estimate()`, `estimated=True`).

### 5.3 Dollars per million tokens

The price table in `providers.CATALOGUE` has the date 2026-09-26 (verify). The Gemini rows are equal to
`scalelab.capacity.PRICES` (examined 5 September, unchanged). A test proves that `metering.price_call()` is equal to
`scalelab.capacity.cost_per_call` on every one of those rows. Cache reads, cache writes and the rest of the prompt
have three different prices. On claude-haiku-4-5, 10,000 tokens written to the prompt cache cost $0.0125 at the 1.25
write rate, not the $0.0100 of an input-rate ledger (`price_call(..., cache_write_tokens=)`).

The table shows the scaling primer's anchor call: 5,000 input tokens (2,700 of them cached) and 350 output tokens.
The blended price is over its 5,350 tokens (`metering.cost_per_million()`):

| Model | $ per call | Blended $ per 1M tokens |
|---|---|---|
| gemini-3.5-flash | $0.007005 | $1.3093 |
| claude-haiku-4-5 | $0.00432 | $0.8075 |
| gpt-5.4-mini | $0.0035025 | $0.6547 |
| gemini-3.5-flash-lite | $0.001646 | $0.3077 |

Self-hosted rows come from the 01 primer
([§8.1](../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md#81-from-gpu-hour-to-m-tokens)).
`metering.self_hosted_per_million()` is `roofline.cost.cost_per_million_tokens()`:

$$
\frac{\text{\$/GPU-hour}}{\text{tokens/s} \times 3{,}600 \times \text{utilisation}} \times 10^{6}
$$

Also, a test reproduces the rows of §8.1 from `roofline.llm.decode()` itself:

- Llama-3.1-8B on an H100 Spot at $3.7/hour, batch 68, 6,846.5 tokens/s: **$0.150** per 1M output tokens at 100 %
  utilisation ($0.250 at 60 %).
- On demand at $11 → $0.446.
- An L4 at $0.70 → $0.660.

Those numbers are for *output* tokens at the decode rate. Prefill tokens cost an order of magnitude less. That is the
root of the asymmetry between hosted input and output prices. It is also why the two rows are not the same unit as
the blended column. Compare hosted and self-hosted at your call shape, not per "token".

### 5.4 Chargeback

The chargeback of hosted spend is direct: each tenant pays the sum of its ledger costs. A self-hosted pool has one
bill and no per-token price, so `metering.chargeback()` divides the bill. A division by tokens treats a prompt token
like an output token. A division by GPU-seconds charges prompt tokens at the prefill rate (~68,000 tokens/s on the
§5.3 H100, 01 PRIMER §8.1). It charges output tokens at the decode rate (6,846.5 tokens/s).

Take a $10,000 month that two tenants share. The RAG tenant has 90 M prompt and 1 M output tokens. The thinking
tenant has 5 M prompt and 10 M output tokens:

| Split by | RAG tenant | Thinking tenant |
|---|---|---|
| tokens | $8,584.91 (85.8 %) | $1,415.09 |
| GPU-seconds | $4,892.57 (48.9 %) | $5,107.43 |

Token-proportional chargeback makes the prompt-heavy tenant pay part of the cost of the output-heavy tenant. When the
engine gives GPU time, measure it there (vLLM's `vllm:request_prefill_time_seconds` and
`vllm:request_decode_time_seconds` histograms). Otherwise, the rates in this section are the fallback.

### 5.5 Reconciliation

The ledger and the usage export of the provider must agree. Drift between them has causes that you can name:

- retries that the provider billed but the ledger recorded one time (bill every attempt that reached the model),
- cut streams billed on estimates,
- cached tokens with an incorrect price,
- a tokenizer mismatch in estimates.

`Ledger.reconcile()` compares per-model totals. It marks any field that is off by more than a tolerance (1 % by
default). Run it daily, and alert on it beside cost per conversation (§5.10).

### 5.6 Traces are not the ledger: GenAI spans

Traces are a sample of evidence that helps you find bugs. The ledger is the bill. The gateway emits one **SERVER** span
per request (`POST /v1/chat/completions`). It also emits one **CLIENT** span per upstream target that it tried
(`chat {model}`), because different providers are different operations. Retries against one target stay inside the
span of that target (`gwcore.otel.Tracer`).

Attribute names obey the OpenTelemetry GenAI conventions. These conventions are still at *development* stability, and
they change. The v1.41.0 release (April 2026) is the last semantic-conventions release that defines them. They now live
in `semantic-conventions-genai`. That repository renamed `gen_ai.usage.cache_creation.input_tokens` to `…cache_write…`
and replaced the `gen_ai.client.token.usage` histogram with counters. Neither change has a release yet.

`otel.py` pins the names that the two agree on. These are the same constants as in the 07.2 lab's
[`tracing.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/observability/tracing.py)
(its [notebook
09](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/09_tracing_and_metrics.ipynb)),
and a test compares them. `otel.py` also pins these names:

- `gen_ai.provider.name`. It is necessary, and there is no well-known value for a self-hosted server, so `vllm` is
  our choice.
- `gen_ai.response.model`, the model that actually answered. The alias goes in `gen_ai.request.model` on the server
  span.
- `gen_ai.response.time_to_first_chunk` and `gen_ai.request.stream`.

Where the two disagree, `otel.py` uses the released names by default, and it keeps the map
(`otel.RENAMED_ON_MAIN`).

The tracer writes spans as OTLP/JSON, one export request per line. The spans have hex ids, integer span kinds, and
nanosecond timestamps as strings. `otel.read_jsonl()` reads them back. Metric labels stay low-cardinality (scaling
primer §5.10): model, tenant and outcome, never a request id or a conversation id.

---

## 6. Keys, tenants and isolation

### 6.1 Virtual keys

Apps get **virtual keys**, not provider keys (`keys.KeyStore`). A virtual key has these properties:

- The gateway stores it only as a SHA-256 hash (as LiteLLM's `hash_token` does).
- The gateway shows it one time.
- It has a **scope**: the aliases that it can use.
- It has a **budget** in dollars. This is the scaling primer's §5.11 dollar budget, per key, and `authorize()`
  refuses at `max_budget`.
- It has limits in tokens per minute (§4).
- It has a tier.
- It is **revocable**.

A leaked virtual key puts one tenant's budget at risk until its revocation. A leaked provider key puts the budget of
every tenant at risk.

### 6.2 Provider keys only in the gateway

Provider keys live in a secret manager, and only the gateway's identity can read them. The gateway injects them on the
way out. This is the **gateway path** of the identity primer (§5): the app never sees the keys. It is the same pattern
as the egress proxy of the sandbox, which injects credentials that the sandboxed code never holds ([sandboxed-execution
PRIMER §4](../../07-application-agent-framework/sandboxed-execution/PRIMER.md#4-network-and-secrets)).

Rotation is add, overlap, retire (`keys.ProviderKeys.rotate()`). The new key becomes current at once. The old key
stays valid for an overlap window, so that in-flight requests and every gateway replica finish on it. Then the old
key expires.

Never expose an engine's port. vLLM's `--api-key` guards only paths under `/v1`, `/v2`, `/inference` and `/cohere`.
`/health`, `/metrics` and the rest stay open.

### 6.3 The tenant comes from the verified key

The tenant, the tier and the scopes are what the verified key maps to (`Gateway.handle()`, stage 1). They never come
from a header or a body field that the caller sets, because anything the caller sets, an attacker sets. The gateway
removes the route metadata before it forwards the request. It sets the per-tenant fields itself.

### 6.4 Isolation by layer

| Layer | Isolated by | If it is missing |
|---|---|---|
| Limits | a `ReserveLimiter` per tenant under the provider key's (§4.4) | the noisy neighbour (§4.5) |
| Response caches | namespace = verified tenant (+ user for per-user classes) (§3.2) | one tenant's answer served to another |
| Engine prefix cache | `cache_salt` = base64url(HMAC-SHA256(gateway secret, tenant)), 43 characters (`keys.cache_salt()`) | a timing side channel across tenants (serving-engine PRIMER §5) |
| Ledger | tenant and key id on every row | spend that nobody can bill |
| Traces | tenant attribute, content opt-in and redacted | one tenant's prompts on another's dashboard |
| Residency | the key's regions, applied as the chain's region filter (§2.2) | data processed outside the region |
| Priority | the tier maps to the pool priority, with the directions mapped (§2.6, 05 PRIMER §3.2) | the batch job ahead of the interactive agent |

vLLM v0.30.0 validates `cache_salt`. The value must be non-empty and at most 128 characters, with none of `@ / \` or
NUL (`keys.valid_cache_salt()`). In vLLM, only the first block gets the salt, and the chained hashes carry the salt
forward. Derive the salt from the verified tenant with a gateway secret. Then nobody can guess it, and nobody can join
the cache of another tenant with that tenant's salt.

### 6.5 In brief: the gateway's own identity

The gateway is itself a workload that calls providers, MCP servers and its own stores. Thus it needs a first-class
identity (identity primer §3.3–3.5). On SPIFFE, this identity is an X.509-SVID from the **Workload API**. The Workload
API is gRPC over a Unix socket that `SPIFFE_ENDPOINT_SOCKET` names, and every call carries the
`workload.spiffe.io: true` metadata. `FetchX509SVID` is a server stream, and every message in it holds the full state.
When an SVID is not in a message, that SVID is revoked.

SPIRE issues one-hour X.509-SVIDs by default. It rotates an SVID when the rest of its lifetime decreases to
half-life ± 10 % of the half-life. This gives a window between 1,620 and 1,980 seconds before expiry, 27–33 minutes
(`keys.svid_rotation_window(3600)`). The SPIRE agent draws that jittered half-life again on every check
(`rotationutil.shouldRotateByHalf`). Thus, with a check every few seconds, the rotation occurs near the top of the
window, not uniformly across it. `keys.FakeWorkloadAPI` models a check every 5 s (verify), and it rotates with a mean
of 32.1 minutes left.

A gateway keeps a pool of long-lived TLS connections. An established session keeps the certificate of its handshake.
Thus the gateway must do two things (verify: this is operational practice, not in the standards). It must build its
TLS configuration again from the stream. It must also replace its connections before the old SVID expires.

---

## 7. Guardrails and what they cost

### 7.1 Hooks and placements

A guardrail is a check at a **hook**, at a **placement** that decides its latency and exposure. The hooks are the
input, a tool call, a tool result, the streamed output and the final output (`guardrails.HOOKS`).

The core's checker is a regex screener (`guardrails.RegexScreener`). It is a labelled stand-in, and its patterns have
bounds. An unbounded `[\w]+@` is quadratic on a long prompt, and a check on every request must not become the
lowest-cost DoS. Real checkers are classifiers such as Llama Prompt Guard 2 (22M and 86M parameters, a 512-token window,
input-side). Another such classifier is Llama Guard (1B, 8B and 12B), which examines prompts and responses. A managed
service such as Model Armor is also a real checker (identity primer §4.1, §6.1: `INSPECT_ONLY` first).

| Placement | How | TTFT added | Exposure |
|---|---|---|---|
| inline input | do the check, then call the model | the check | none |
| parallel input | call the model at once, and hold its first token until the verdict | $\max(0, \text{check} - \text{model TTFT})$ | none |
| parallel with cancel | start the stream at once, and cancel on a bad verdict | 0 | tokens before the verdict |
| held-back window | release output in windows of $W$ tokens, each with a check first | $(W - 1) \times \text{ITL} + \text{check}$ | none, per window |
| final | do a check of the whole answer, then send it | the whole generation + check | none, and no streaming |
| shadow | do the check off the path, and log only | 0 | everything (measurement only) |

### 7.2 Latency

`guardrails.added_latency()` puts numbers on the table for a 300-token answer at TTFT 0.4 s and ITL 20 ms (end to end
6.38 s). An input check of 19.3 ms (Prompt Guard 2 22M on an A100 at 512 tokens, verify) adds 19.3 ms inline. It adds
nothing in parallel, because it finishes long before the first token. Take the 86M model's 92.4 ms against a 50 ms
TTFT (a small self-hosted model, or a cached prefix). It adds 42.4 ms if the first token is held, or lets 3 tokens
through if the gateway cancels the stream instead (`guardrails.exposed_tokens()`). On the output side, with an
illustrative 150 ms check:

- held-back window $W$ = 200 tokens (NeMo Guardrails' default chunk size): TTFT + 0.02 × 199 + 0.15 = + 4.13 s
- held-back window $W$ = 50 tokens: TTFT + 0.02 × 49 + 0.15 = + 1.13 s
- final check: TTFT + 0.02 × 299 + 0.15 = + 6.13 s

A held-back window keeps up only if each check finishes within $W \times \text{ITL}$ (4 s at $W$ = 200). It also puts
$W$ tokens of generation in front of the first token of every user. NeMo Guardrails' streaming output rails have the
default `stream_first: True`. With it, tokens reach the client *before* the rail sees them. Thus its default is
"parallel with cancel", not a held-back window (verify when you configure it).

### 7.3 Dollars and false positives

A classifier on a dedicated GPU costs its GPU time. At $3.7 an hour for an A100 (verify), a check takes 92.4 ms when
the checks run one at a time. That is $0.095 per 1,000 checks, and batched 16 at a time $0.0059 — *assuming* a batch
of 16 finishes in about the same 92.4 ms. The model card does not say this (its figure is one 512-token
classification). Measure the batch latency before you rely on it (`guardrails.cost_per_1k_checks()` divides by the
concurrency that you give it).

The chance of a false block increases with each check, because the pass rates of the checks multiply. The scaling primer
has 13 model calls per conversation, and a check of the input and the output of each call gives 26 checks. At a 1 %
false-positive rate, 23.0 % of conversations hit at least one false block, and at 0.1 %, 2.57 %
(`guardrails.false_block_rate()` = $1 - (1 - \text{fpr})^{\text{checks}}$). That cost, not the dollars, is usually the
cost that limits the design. It is why screens start in inspect-only mode, and why you measure the rate per placement.

### 7.4 Risk reduced, risk bounded

Guardrails are probabilistic, and they **reduce** risk. What **bounds** the risk is deterministic and sits outside the
model. The parts are keys and tokens with scopes, deny-by-default tool policy, egress allowlists and confirmation
gates. The identity primer covers them (§0, idea 4, the four enforcement points of §4.1, and §6).

Give a design review both halves: "we screen input and output, and a fully hijacked model still cannot exceed what its
credentials allow". The canonical home of prompt injection is 06.6 (CURRICULUM §3.4). This section only places the
check and gives its price.

---

## 8. The gateway as an MCP client

### 8.1 Why the gateway is the client

Agents reach MCP servers through the gateway. The 07.2 lab's [MCP egress
gateway](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/mcp/gateway.py)
([notebook
05](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/05_mcp_server_client_gateway.ipynb))
enforces policy, screens arguments and results, and does not permit token passthrough. It also writes an audit record
for every call. The MCP server is an OAuth 2.1 resource server ([identity primer
§7.1](../identity-security/agentic-identity-gcp-lab/docs/primer.md#71-mcp-the-server-is-an-oauth-21-resource-server)).
Thus the caller of the server runs the **client** side of the MCP authorization spec (revision 2026-07-28, verify). The
07.2 lab's
[`docs/MCP_REVISIONS.md`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/docs/MCP_REVISIONS.md)
lists what changed in each revision.

If the gateway is the client, tokens live in the gateway, per principal, and the agent never holds one. The 07.2 lab's
[`agentlab/auth/oauth.py`](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/auth/oauth.py)
([notebook
06](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/notebooks/06_oauth_identity_propagation.ipynb))
already runs discovery, then PKCE, then code, then token. `gwcore.mcp_authz` reproduces its `challenge_for` and adds
only what a gateway needs:

- Client ID Metadata Documents,
- refresh rotation with reuse detection,
- a token per (principal, resource) with its scopes,
- DPoP nonces.

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

Each step is a function in `mcp_authz` and a rule that the spec sets (verify each against revision 2026-07-28):

- **Discovery** (`prm_urls()`): use the `resource_metadata` of the 401 if it is present. Otherwise, use the
  path-aware well-known URI, then the root one. Clients must support both. For the authorization server
  (`as_metadata_urls()`), take an issuer with a path such as `https://auth.example.com/tenant1`. The client tries RFC
  8414 path insertion (`/.well-known/oauth-authorization-server/tenant1`), then OIDC path insertion
  (`/.well-known/openid-configuration/tenant1`), then OIDC path appending (`/tenant1/.well-known/openid-configuration`).
  The `issuer` of the document must be equal to the issuer that the client used to build the URL.
- **Registration**: the first choice is pre-registration. If the client is not pre-registered, it uses a **Client ID
  Metadata Document** when the AS advertises `client_id_metadata_document_supported` (SHOULD). Otherwise, it uses
  Dynamic Client Registration (MAY, deprecated). The `client_id` is an HTTPS URL with a path. Its document holds at
  least `client_id` (equal to the URL), `client_name` and `redirect_uris`, and no shared secret (`validate_cimd()`). The
  AS fetches the document with a ~5 KB read cap, refuses special-use addresses, and never caches an invalid document.
- **PKCE** is S256 only. The client **must refuse** when the metadata does not have
  `code_challenge_methods_supported`. RFC 7636 Appendix B pins the method: verifier
  `dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk` → challenge `E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM`
  (`mcp_authz.s256()`, equal to 07.2's `challenge_for` in a test).
- **`resource`** (RFC 8707) goes in **both** the authorization request and the token request. Thus the audience of
  the token is that one MCP server, and another server refuses a token stolen from one server.
- **`iss`** (RFC 9207): if the AS advertises it, an absent or different `iss` in the response is a mix-up. Compare the
  values as plain strings (`iss_ok()`).

### 8.3 Tokens per principal and resource; refresh rotation

The gateway calls many MCP servers for many users. Thus the key of its token store is **(principal, resource)**, and
the store records the scopes that each token carries (`MCPClient.tokens`). The reference Python SDK keeps one token
per server. That is correct for a desktop client and incorrect for a gateway.

Tokens are short-lived. Refresh tokens for public clients **rotate**: every refresh returns a new refresh token and
makes the old one invalid. The AS keeps the family. If a client presents an old refresh token again, someone has a
copy. Thus the AS **revokes the whole grant**, and the active tokens too (OAuth 2.1 §4.3.1, `FakeAS.token()`, with a
test that replays a rotated token). Refresh requests carry `resource` too.

### 8.4 Step-up

A token that does not have the scope of a tool gets `403` with
`WWW-Authenticate: Bearer error="insufficient_scope", scope="…"`. The client authorizes again with the **union** of
the scopes that it held and the scopes that the server demands. It never asks for only the new scope, because that
drops the scopes that it already had. It retries a bounded number of times (`MCPClient.call()`, which stops after
`max_steps`). The first authorization asks for the scope in the 401 challenge, if there is one. Otherwise, it asks for
the `scopes_supported` of the metadata: least privilege first, and a step-up when necessary.

### 8.5 DPoP nonces

A sender-constrained token ([identity primer
§3.5](../identity-security/agentic-identity-gcp-lab/docs/primer.md#35-delegation-mechanics-standards-you-should-be-able-to-draw))
has a binding to a key. The client sends `Authorization: DPoP <token>` and a `DPoP` proof that the client signs with
that key. The proof is a JWT with `jti`, `htm`, `htu`, `iat`, and `ath` = base64url(SHA-256(token)).

RFC 9449 lets servers demand a fresh **nonce** in the proof. An authorization server answers
`400 {"error":"use_dpop_nonce"}` with a `DPoP-Nonce` header (§8 of the RFC). A resource server answers `401` with
`WWW-Authenticate: DPoP error="use_dpop_nonce"` and `DPoP-Nonce` (§9). The client retries one time with the nonce, and
it keeps the latest nonce for each server.

DPoP comes from RFC 9449, **not** from the MCP spec. The MCP spec does not mention it. The identity lab's
`agentsec.identity.tokens.DPoP.proof()` accepts a `nonce`, but its `verify()` never issues a nonce or examines one.
`mcp_authz` closes that gap.

The signer of `mcp_authz` is pluggable, and the signer in the package is an HMAC stand-in (`mcp_authz.HMACSigner`) with
the label non-conformant. RFC 9449 makes an asymmetric algorithm necessary, and the standard library has none. The lab
signs with `cryptography` when that package is available.

---

## 9. Where to run it, and what to adopt

### 9.1 Tiers

| Tier | What runs | Cost |
|---|---|---|
| T0 in-process | all of `gateway-core`: providers, the chain, caches, limits, ledger, spans and the MCP flow on a virtual clock | free |
| T0 on localhost | [`gateway-lab`](gateway-lab/): an async OpenAI-compatible gateway over HTTP in front of fake providers, built like the 05 lab's [`igwlab` router](../../05-orchestrator/serving-orchestration/inference-gateway-lab/igwlab/router/server.py) | free |
| T0 + Docker | the lab's [compose stack](gateway-lab/deploy/local/): the gateway and two fake providers, one of them set to fail | free |
| T1 | the gateway in front of one real vLLM (`vllm/vllm-openai:v0.30.0`, Qwen2.5-0.5B-Instruct, `--enable-prompt-tokens-details`) on a free Colab or Kaggle T4, with a fake provider as the fallback | free, or ~$0.3–0.7 an hour rented (verify) |
| T3 | the gateway on Cloud Run or GKE with the 04 lab's [Cloud Run or GKE vLLM](../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/) or the 05 lab's [GKE Inference Gateway](../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/) as upstreams. It needs no new Terraform. | pay per use. See [`COMPUTE.md`](../../COMPUTE.md). |

On Google Cloud, the gateway is a stateless service (Cloud Run: requests up to 60 minutes, streams not capped,
scaling primer §5.7). Its keys are in Secret Manager, its limits and caches are in Memorystore, and its spans go to
Cloud Trace over OTLP. Apigee and Agent Gateway are the managed pieces (scaling primer §9, identity primer §10,
verify). On other platforms, the same pieces map one to one:

| The gateway needs | On Google Cloud | Anywhere else |
|---|---|---|
| a stateless, multi-replica service | Cloud Run, GKE | any container host or Kubernetes cluster. Locally, the lab's compose stack or kind. |
| shared buckets and caches (§4.4) | Memorystore | Redis or Valkey |
| provider keys (§6.2) | Secret Manager | Vault, or the cloud's own secret manager |
| spans (§5.6) | Cloud Trace over OTLP | any OTLP backend: Jaeger, Tempo, an OpenTelemetry Collector |
| a T1 upstream | the 04 lab's Cloud Run or GKE vLLM | a rented GPU (RunPod, Vast, Lambda, with prices in [`COMPUTE.md`](../../COMPUTE.md), verify) that runs the lab's [`deploy/any-gpu`](gateway-lab/deploy/any-gpu/) |

### 9.2 Build or adopt

Most teams adopt a gateway. They keep this primer as the checklist to examine it. The table gives what the upstream
code shows at the pinned versions (all verify, because products change monthly):

| Concern | LiteLLM proxy 1.104.0 | Envoy AI Gateway v1.1.0 | Kong OSS 3.10.0 | Portkey gateway 1.15.2 |
|---|---|---|---|---|
| §1 adapters | many providers, OpenAI API | `AIServiceBackend` schemas (OpenAI, Anthropic, Bedrock, Vertex, …) | `ai-proxy`: `route_type` `llm/v1/chat`, providers such as openai, anthropic, gemini | 78 provider directories |
| §2 fallbacks | `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, cooldowns | `backendRefs[].priority` + retry policy. An `InferencePool` backend leaves fallback to the EPP. | load balancing and fallback in Enterprise `ai-proxy-advanced` | `strategy.mode` `fallback`, `on_status_codes` |
| §3 caching | exact and semantic (Redis, Qdrant) types. The scope of the semantic cache is per key by default. | not examined | semantic cache is Enterprise | `cache.mode` simple / semantic. OSS is exact only, and its key has no tenant. |
| §4 token limits | `tpm_limit`/`rpm_limit` per key. The v3 limiter reserves an estimate. | `llmRequestCosts` feed a global rate limit, and the debit occurs after the response | token rate limiting is Enterprise | not examined (retries obey `Retry-After`) |
| §5 metering | spend log per request, keys hashed | GenAI metrics (`gen_ai.client.token.usage`) | logs prompt/completion tokens and cost | not examined |
| §6 keys | virtual keys: models, budget, TPM/RPM | `BackendSecurityPolicy` holds upstream credentials | `auth` per plugin | virtual keys |
| §7 guardrails | `pre_call`, `during_call`, `post_call` and MCP-call hooks | not examined | `ai-prompt-guard` | input and output guardrail hooks |
| §8 MCP | MCP-call guardrail hooks, MCP gateway (verify) | `MCPRoute` | not in the OSS plugins read | not examined |

When you adopt a gateway, the design decisions stay. You still decide which failures fall through, what is cacheable
per route, and what a reservation is. You also decide who owns chargeback, where the tenant comes from, and where each
check sits. Ask each product the questions of §1–§8. Then examine its answers with the lab's scripted outages and
tenants.

---

## In a design review

**The two-minute walkthrough.** "Every app and agent calls one gateway with a virtual key. The key maps to a tenant,
a tier, a budget and scopes, and nothing that the caller sends can change that. The gateway speaks OpenAI chat
completions over SSE to every client. It normalises the usage and finish reasons of each provider through adapters
that are mostly data. Anthropic's input tokens do not include the cache, Gemini's thinking is outside the candidates,
and vLLM puts errors inside a 200.

"Clients ask for aliases. An alias is an ordered chain of provider, model and region. We filter the chain by
capability, context and residency, and order it by a policy, with a breaker per target. We fall through on 429, 5xx,
timeouts and context-too-long, but never on a bad request or a policy refusal. We fall through only before the first
byte, and after it we show the error to the client. Whatever the targets share sets a cap on the chain availability,
so fallbacks go in different failure domains.

"For caches, we keep exact and semantic answers only for classes that a route declares cacheable. These caches have a
namespace per tenant and a guard on numbers and entities, and we select the threshold on labelled traffic. For
everything else, we use provider prompt caching and the engine's prefix cache, isolated by a per-tenant `cache_salt`
that we derive. Limits are in tokens: reserve the prompt plus the output bound, debit as the stream sends tokens, and
reconcile with usage. A per-request bucket let through twice the provider's TPM on the day of the thinking release.

"Every request writes a ledger row with a price from usage. The bill counts thinking as output, and cut streams bill
on counted deltas. The chargeback of the self-hosted pool is by GPU-seconds. Traces carry pinned GenAI names, but the
ledger is the bill. Provider keys live only in the gateway and rotate with an overlap.

"We place guardrails by their cost. Inline input checks are low-cost, and held-back output windows cost a window of
generation in TTFT. The chance of a false block increases with each check in a conversation. Policy outside the model
bounds what the guardrails miss.

"For MCP, the gateway is the OAuth client. It does discovery, and it uses a metadata-document client id and PKCE with
resource. It uses refresh tokens that rotate, a step-up with the union of scopes, and DPoP nonces. The tokens are per
principal and resource, never in the agent."

**Drill questions**

1. *A provider outage lasted five minutes. Our incident lasted forty, and the bill doubled.*

    Clients retried without a budget, so the retry wave lasted longer than the outage. Streams that failed in the middle
    ran again from the start, and the bill counted two times the output that the model had already generated. Also, the
    whole chain fell through to a higher-cost model at full traffic until the quota of that model ran out.

    Use retry budgets with jitter and a breaker per target. Fall back only before the first byte. Calculate the size and the cost
    of the fallback capacity in advance. Put that capacity in an independent failure domain. Add a cost alert per tenant
    (§2).

2. *The semantic cache answered one user with another user's order status.*

    The query was personal, and the gateway must never cache that class. The key had no tenant or user namespace. A
    lexical near match ("order 1234" against "order 1243") passed the threshold. Declare the cacheable classes per
    route, give each tenant its own namespace, and guard numbers and entities exactly. Measure false hits on labelled
    traffic before you decrease $\tau$. The engine's prefix cache is exact and isolated with `cache_salt` (§3, §6.4).

3. *After a thinking-model rollout, the provider started to return 429s, but our request rate limit never tripped.*

    The bucket charged per request, while outputs grew by an order of magnitude with a heavy tail. Thus the same
    request rate carried twice the tokens per minute (1.99× by the closed form). In §4.3, the tokens served rose from
    1.00× to 1.87× the limit, with the window peaking at 1.97×. Charge tokens: reserve at admission, debit as chunks
    arrive, reconcile at the end, and cap the output. Enforce TPM beside RPM per tenant. Give thinking a budget, and do
    not truncate it with `max_tokens` (§4).

4. *Why can the gateway not fall back to another model when a stream fails halfway?*

    The client has already rendered part of an answer. If a second model continues it, the client reads one answer
    from two models, with no shared state. Send an error chunk to the client. Bill the relayed tokens as an estimate.
    Let the agent (which owns the conversation) decide if it retries the turn (§2.3, §5.2).

5. *The app sets an `X-Tenant` header and the gateway uses it for the cache namespace. What is incorrect?*

    Anything the caller sets, an attacker sets. Thus one tenant can read the cache of another tenant, spend its budget
    and poison its answers. The tenant comes from the verified key. The cache namespace, the limits, the ledger and
    the `cache_salt` come from the tenant (§6.3, §6.4).

6. *An MCP server returns 403 insufficient_scope for `close_ticket`. What does the gateway do, and what must it never
   do?*

    Authorize that principal again for that resource, with PKCE and `resource`. Ask for the union of the scopes that
    it holds and the scope that the server demands. Store the new token under (principal, resource), and retry a
    bounded number of times. The gateway must never pass the agent's own token through. It must never ask for only the
    new scope, because that drops the old scopes. It must never use the token of one principal for another principal
    (§8.3, §8.4).

---

## Glossary

| Term | Meaning |
|---|---|
| Alias | The model name that a client asks for (`chat`). It resolves to a fallback chain. |
| Fallback chain | An ordered list of (provider, model, region) targets. The gateway tries them until one answers. |
| Falls through | A failure where another target can succeed (429, 5xx, timeout, context length). The gateway tries the next target. |
| Breaker | Per-target state machine: open after consecutive failures, one probe after the cooldown. |
| Common-mode failure | One failure that stops every target (the gateway, a shared region). |
| Exact / semantic cache | Reuse a stored answer on an identical key / on a query whose similarity is above a threshold. |
| Entity guard | Refuse a semantic hit unless numbers, dates and codes match exactly. |
| `cache_salt` | A vLLM request field that adds a salt to the first KV block, so that tenants never share prefix-cache entries. |
| Reserve → stream → reconcile | Token metering: hold an upper bound at admission, debit the tokens as the stream sends them, and release the rest. |
| Over-admission | Tokens let through beyond a provider's limit because the charge at admission was too small. |
| Ledger row | The billing record of one request, priced from the provider's `usage`. |
| Chargeback | The division of a shared cost (a GPU pool) among tenants, by tokens or by GPU time. |
| Virtual key | A gateway-issued key: hashed at rest, scoped, budgeted, revocable. It maps to a tenant. |
| SVID | A SPIFFE verifiable identity document (here X.509) issued to a workload through the Workload API. |
| Held-back window | Output released in windows of $W$ tokens, each with a check before release. |
| CIMD | Client ID Metadata Document: an HTTPS URL that is the OAuth client id and serves the client's metadata. |
| PKCE S256 | Proof Key for Code Exchange: the token request proves possession of the verifier behind the challenge. |
| Refresh rotation | Each refresh returns a new refresh token. Reuse of an old one revokes the grant. |
| DPoP nonce | A server-supplied value that a DPoP proof must carry (RFC 9449 §8–§9). |

---

## Sources

- OpenAI OpenAPI description, `openai/openai-openapi` `42eccf1` (2026-09-26): `CreateChatCompletionRequest`,
  `ChatCompletionStreamOptions`, `CreateChatCompletionStreamResponse`, `CompletionUsage`, `ErrorResponse`.
- vLLM v0.30.0 (`ced6857`): `vllm/entrypoints/openai/chat_completion/protocol.py` (`cache_salt`, `priority`,
  `validate_stream_options`), `vllm/entrypoints/generate/base/protocol.py` (`StreamOptions`),
  `chat_completion/serving.py` (errors inside the stream, `[DONE]`), `vllm/entrypoints/launchers/cli_args.py`,
  `vllm/v1/core/kv_cache_utils.py`.
- OpenTelemetry semantic conventions v1.41.0 (`model/gen-ai/`) and `semantic-conventions-genai` `e57c543`
  (`model/gen-ai/*.yaml`, `changelog.d/`). OTLP/JSON encoding (`opentelemetry-proto` `docs/specification.md`).
- Model Context Protocol specification, revision 2026-07-28, `basic/authorization/` (index, authorization server
  discovery, client registration, security considerations). The MCP Python SDK `src/mcp/client/auth/`.
- OAuth 2.1 (draft-ietf-oauth-v2-1, editor's copy) §4.1.1, §4.3.1, §7.5, and
  draft-ietf-oauth-client-id-metadata-document. RFC 7636 (PKCE, Appendix B), RFC 8414, RFC 8707, RFC 9207 and RFC 9728.
  RFC 9449 (DPoP, `danielfett/draft-dpop`).
- SPIFFE `standards/SPIFFE_Workload_API.md`, `SPIFFE_Workload_Endpoint.md`. SPIRE `doc/spire_server.md`,
  `doc/spire_agent.md`, `pkg/common/rotationutil/rotationutil.go`.
- LiteLLM 1.104.0 (`litellm/router.py`, `litellm/proxy/`, `litellm/caching/`, `model_prices_and_context_window.json`
  at `849f303`). Envoy AI Gateway v1.1.0 (`api/v1beta1/`, `site/docs/`). Kong 3.10.0 (`kong/plugins/ai-*`,
  `kong/llm/`). Portkey gateway 1.15.2 (`src/providers/`, `src/middlewares/`).
- Anthropic Python SDK 1.8.0 (`src/anthropic/types/`), Google GenAI Python SDK 2.25.0 (`google/genai/types.py`).
- NeMo Guardrails (`nemoguardrails/rails/llm/config.py`). PurpleLlama model cards (Llama Guard 3 and 4, Prompt Guard 2).
- In this repo: every primer and module that this primer links, and the canonical homes in §3.4 of
  [`CURRICULUM.md`](../../CURRICULUM.md).

---

## Verify list

This list has the date 26 September 2026. Do the check again before you rely on any of these facts.

| Fact | Status here |
|---|---|
| Hosted prices (USD per 1M: input / output / cached). Gemini: gemini-3.5-flash 1.50 / 9.00 / 0.15, gemini-3.5-flash-lite 0.30 / 2.50 / 0.03. OpenAI and Anthropic: gpt-5.4-mini 0.75 / 4.50 / 0.075, claude-haiku-4-5 1.00 / 5.00 / 0.10 (cache write 1.25) | Gemini from `scalelab.capacity.PRICES` (5 Sep) and LiteLLM's price file @`849f303`, which agree. The others are from that file. |
| Context windows: gpt-5.4-mini 272,000, claude-haiku-4-5 200,000, Gemini rows 1,048,576 | The first two are from LiteLLM's file. The Gemini value is not from a source here. |
| GPU prices: H100 Spot $3.7/GPU-hour, on demand $11, L4 $0.70, A100 ~$3.7/hour | the repo's FACTS and 01 PRIMER §8.1, us-central1. See `COMPUTE.md`. |
| `data: [DONE]` for chat completions, `x-ratelimit-*` headers | `[DONE]` only inside the `include_usage` description. `x-ratelimit-*` is not in the OpenAPI description (prose docs only). |
| vLLM v0.30.0: `cache_salt` ≤ 128 characters without `@ / \` NUL, first block only. `priority` lower = earlier, needs `--scheduling-policy priority`. `--api-key` guards `/v1`, `/v2`, `/inference` and `/cohere` only (`GUARDED_PREFIX`) | read at the tag |
| OTel GenAI: v1.41.0 is the last release that defines GenAI. `cache_write` and `gen_ai.client.inference.*` are on genai main but have no release | read 2026-09-26. The names will change. |
| MCP authorization 2026-07-28: CIMD SHOULD, DCR MAY and deprecated, S256 refusal rule, `resource` in both requests, RFC 9207 `iss`, no DPoP | spec pages read at `ab3a39c` |
| OAuth 2.1 refresh-rotation reuse detection (§4.3.1) and which draft number is current | editor's copy, draft number (verify) |
| SPIRE: X.509-SVID TTL 1 h, JWT-SVID 5 min, rotation at half-life ± 10 %, `availability_target` ≥ 24 h | SPIRE docs and `rotationutil.go`. The long-lived TLS rotation practice is not in the standards. |
| Prompt Guard 2: 22M 19.3 ms, 86M 92.4 ms per 512-token classification on an A100 | model card |
| NeMo Guardrails streaming output rails: `chunk_size` 200, `context_size` 50, `stream_first` True | `config.py` at `e549dda` |
| Product features in §9.2 (LiteLLM, Envoy AI Gateway, Kong OSS against Enterprise, Portkey OSS cache) | read in each repo at the pinned commit. Managed gateways (Apigee, Agent Gateway) are from the repo's primers. |
| Cloud Run request timeout up to 60 minutes, Cloud Trace via OTLP (`telemetry.googleapis.com`) | from the scaling primer §5.7 and §9 |
| Redis + Lua atomic check-and-commit, one hash slot per script on a cluster | not read from a Redis source here |
| How hosted providers count tokens against TPM/OTPM. OpenAI: the larger of `max_tokens` and an estimate at admission. Anthropic: output tokens estimated from `max_tokens` and corrected at the end | from memory. The providers' rate-limit pages were not available here. |
| LiteLLM v3 limiter: prompt + `max_tokens`, else max(input estimate, 1,024 = 4,096 ÷ 4, capped at a quarter of the smallest TPM limit). An implicit `max_tokens` when capped, and a reconcile after the call | `parallel_request_limiter_v3.py` @`849f303` |
| SPIRE agent does a rotation check on its sync loop (default 5 s, with a back-off on errors) | `pkg/agent/manager/manager.go`. The fake's 5 s check is a model of it. |
