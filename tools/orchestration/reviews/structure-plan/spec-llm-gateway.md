# SPEC §6b block — `06-gateway/llm-gateway/` (draft for package `c4-gateway`, 2026-09-26)

How to use this file. Paste the block into `tools/orchestration/SPEC.md` §6b after the `sandboxed-execution` block, add
`| 06 | 06-gateway/llm-gateway | gateway-core (gwcore) | gateway-lab (gwlab) |` to §2's layout table, and make §6b's opening
count this topic. For `build_topic.js`: `specBlock` = the block's heading title, `researchBrief` = "Research brief" below,
`existing` = "Existing material (paths)" below. Facts marked "spot-checked" were read from upstream while drafting (commits in the
brief); every other product fact stays `(verify)` until the research agent's fact sheet says otherwise.

---

### 06 · llm-gateway — "The LLM gateway: one front door for many models — routing, fallbacks, caching, metering and tenant isolation"
Primer sections: 1 One front door, one API (what a gateway owns for many apps, tenants and providers — keys, policy, limits, metering, routing, caching,
traces; the split: the gateway decides whether a request runs and which model, provider, region or pool serves it, the 05 router picks the replica, the
engine batches (05 PRIMER §1.3, §3.3, §7); the request path; chat completions + SSE as the lingua franca (`[DONE]`, `stream_options.include_usage`);
provider adapters as data; what does not normalise (tool-call deltas, `reasoning_content`); its cost — a hop, a failure point, every key in one box)
· 2 Model routing and fallback chains (aliases → ordered (provider, model, region) chains; capability filters; policies — cheapest capable, EWMA TTFT,
canary, tenant tier, effort (00.5 PRIMER §7); what falls through (429, 5xx, timeout, context length) and what must not (400, auth, content policy); fall
back only before the first byte; retries and breakers: scaling primer §5.2, 07.2 notebook 10; chain availability, independent and common-mode
(`routing.chain_availability()`), expected latency and cost (`routing.chain_cost()`); a self-hosted target is a pool — the EPP picks the pod) · 3 Caching at
the gateway (exact, semantic, provider prompt caching (scaling primer §3.4) and the engine's prefix cache (serving-engine PRIMER §5) — what each saves and
risks; what is safe to cache; tenant namespaces; embed → nearest neighbour → threshold → an exact guard on numbers, dates and entities (embeddings primer
§15); hit and false-hit rate vs threshold on the bundled sample (`cache.sweep_thresholds()`); invalidation; a lexical T0 embedder) · 4 Streaming-aware rate
limits (on top of scaling primer §5.1 and §5.3 (06.3); why per-request charges fail — cost is unknown at admission and heavy-tailed (00.5 PRIMER §7);
reserve → stream → reconcile, a hard cap; RPM and TPM together; hierarchical limits, shared state (Redis + Lua); how far a per-request bucket over-admits a
provider's TPM (`ratelimit.compare_buckets()`); the noisy neighbour) · 5 Metering, tracing and chargeback (the ledger row; `usage` is authoritative —
thinking bills as output (00.5 PRIMER §5) — estimates cover admission and cut streams; $ per 1M tokens for a model table, hosted (verify, 2026-09-26) and
self-hosted (01 PRIMER §8.1), blended (`metering.cost_per_million()`); chargeback by tokens or GPU-seconds (`metering.chargeback()`); reconciliation; a
billing-grade ledger vs sampled OpenTelemetry GenAI spans, names still moving (verify)) · 6 Keys, tenants and isolation (virtual keys — hashed, scoped,
budgeted, revocable; provider keys only in the gateway, rotated with overlap (identity primer §5's gateway path); the tenant from the verified key, never a
header; isolation by layer — buckets, cache namespaces, `cache_salt`, ledger, traces, residency; tier → priority below (05 PRIMER §3.2); in brief, the
gateway's own identity — SPIFFE Workload API, SVID rotation (identity primer §3.3–3.5)) · 7 Guardrails and what they cost (vendor-neutral hooks — input,
tool call and result, streamed and final output; placements — inline, parallel with cancel, held-back windows, shadow (identity primer §6.1); latency added
before and after generation (`guardrails.added_latency()`), dollars per 1k checks, false positives; guardrails reduce risk, policy bounds it (identity
primer §0, §4.1, §6)) · 8 The gateway as an MCP client (agents reach MCP servers through it (07.2 notebook 05), so it runs the client side of the MCP
authorization spec (2026-07-28, verify): 401 → RFC 9728 → RFC 8414 or OIDC discovery → a Client ID Metadata Document → PKCE S256 with `resource` → code →
token → refresh rotation with reuse detection → step-up; tokens per principal and resource; DPoP nonces (RFC 9449, not the MCP spec); link identity primer
§3.5, §7.1 and 07.2's `agentlab.auth.oauth`) · 9 Where to run it, and what to adopt (T0 in-process and on localhost; T0 + Docker; T1 one vLLM; T3 the 04 and
05 labs' GCP deploys as upstreams; LiteLLM, Envoy AI Gateway, Kong, Portkey and managed gateways against §1–§8, verify-marked; `COMPUTE.md`).
Core `gwcore` (standard library + numpy, all in-process): `api.py` (chat-completions shapes, SSE, stream accumulation with tool-call deltas and the usage
chunk, a labelled token estimate), `providers.py` (the adapter table with bundled payloads per dialect (illustrative), a model catalogue with prices,
`FakeProvider` on a virtual clock — TTFT/ITL, errors, 429 + `Retry-After`, outages), `routing.py` (chains, filters, fallback classes, a breaker per target),
`cache.py` (exact + semantic; a hashing embedder built like `ragkit.embed.HashingEmbedder`; a labelled traffic sample with paraphrases, near misses and
uncacheable queries), `ratelimit.py` (per-request vs reserve → stream → reconcile), `metering.py` (ledger, prices, chargeback), `keys.py` (virtual and
provider keys, a fake Workload API rotating an SVID), `guardrails.py` (hooks, a regex screener labelled as a stand-in), `mcp_authz.py` (§8 against a fake AS
and MCP server; PKCE pinned to RFC 7636 Appendix B; DPoP from a pluggable signer, a labelled HMAC stand-in — no asymmetric crypto in the stdlib), `otel.py`
(07.2's GenAI names plus `gen_ai.provider.name`, `gen_ai.response.model`; JSON-lines export and read-back), `gateway.py` (the §1 pipeline). Tests pin every
§2–§7 number and reproduce `scalelab.capacity.cost_per_call`, `roofline.cost.cost_per_million_tokens` and `agentlab.auth.oauth.challenge_for`.
Core notebooks: `01_one_front_door`, `02_routing_and_fallback_chains`, `03_exact_and_semantic_caching`, `04_token_limits_metering_and_chargeback`,
`05_guardrails_keys_and_mcp_authorization`.
Lab `gwlab`: `gateway/` (aiohttp, pyyaml, sqlite3 — an async OpenAI-compatible gateway in the style of `igwlab/router/server.py`: `/v1/chat/completions`
streamed or not, `/v1/models`, `/metrics`, `/admin/keys`; a YAML config with the provider table, chains, tenants, limits, cache and guardrails; an SSE relay
that injects `stream_options.include_usage`, strips the usage chunk nobody asked for and meters partial output on disconnect; sqlite3 keys, cache and
ledger; spans to JSON lines, OTLP when installed), `fakes.py` (providers over HTTP modelled on `servelab/fakeserver.py`, with their own RPM/TPM limits,
outages and a second dialect), `mcp/` (§8 over HTTP; DPoP with `cryptography` when installed), `bench.py` (scripted tenants), `report.py`.
Deploy: `deploy/local/` (compose: the gateway + two fake providers, one set to fail; `DRY_RUN=1`; no Docker → printed commands, illustrative output),
`deploy/any-gpu/` (T1: `vllm/vllm-openai:v0.30.0`, Qwen2.5-0.5B-Instruct as `lab/llm`, `--enable-prompt-tokens-details`, a fake fallback; Colab/Kaggle T4
recipe), `deploy/gcp/` (a README, no Terraform: the 04 lab's Cloud Run or GKE vLLM, the 05 lab's GKE Inference Gateway; what to change, cost).
Lab notebooks: `01_a_gateway_over_http` (T0; T0 + Docker the compose stack; T1 vLLM as one provider), `02_outages_fallbacks_and_breakers` (T0; T1 stop vLLM
mid-run), `03_semantic_cache_vs_the_prefix_cache` (T0 emulated; T1 vLLM's `cached_tokens` measured), `04_streaming_limits_metering_and_chargeback` (T0; T1
the ledger reconciled with vLLM's `usage` and `/metrics`), `05_guardrails_and_mcp_authorization_over_http` (T0).
Must cite/reuse: `06-gateway/scaling-admission-cost/agentic-scaling-lab/` (primer §3.4–3.5, §5.1–5.3, §5.7, §5.10–5.12; `scalelab.resilience`, `.admission`,
`.capacity`); `06-gateway/identity-security/agentic-identity-gcp-lab/` (`docs/primer.md` §0, §3.3–3.5, §4.1, §5, §6.1, §7.1, §8, §9;
`agentsec.identity.tokens.DPoP`); `07-.../gcp-agent-platform-lab/` (notebooks 05, 06, 07, 09, 10; `agentlab.auth.oauth`, `.mcp.gateway`, `.observability`,
`.reliability`); `05-orchestrator/serving-orchestration/` (PRIMER §1.3, §3.2–3.3, §7, §9; `igwlab`); `04-inference-engine/serving-engine/` (PRIMER §1, §5,
§11; `servelab/fakeserver.py`); `00-foundations/rl-and-thinking-models/PRIMER.md` §5, §7; `01-hardware-gpu-fabric/roofline-and-fabric/` (PRIMER §8.1;
`roofline.cost`); `07-.../embeddings-lab/docs/primer.md` §15; `07-.../sandboxed-execution/PRIMER.md` §4; CURRICULUM §3.4's canonical homes.

---

## Research brief

For the research agent (`facts-llm-gateway.md`). Clone into `$SP/ref/` with
`flock $SP/ref/.lock git clone --depth 1 [--branch <tag>] https://github.com/<org>/<repo> $SP/ref/<name>` (docs sites are blocked; every
repo below was confirmed to exist with `git ls-remote` on 2026-09-26). Quote exact identifiers; anything without a source stays `(unverified)`.

1. **OpenAI chat-completions contract** (primer §1, §4, §5; `api.py`, `providers.py`, `gwlab/fakes.py`) — clone `openai/openai-openapi`; read the
   chat-completions schemas (`CreateChatCompletionRequest`, `CreateChatCompletionResponse`, `CreateChatCompletionStreamResponse`, `CompletionUsage`,
   the stream-options and error objects). Verify: `stream`, `stream_options.include_usage` (and any sibling fields), `max_completion_tokens` and the
   status of `max_tokens`, `user` and whatever supersedes it (`safety_identifier`, `prompt_cache_key` — which exist), `service_tier`, `tools`,
   `tool_choice`, `parallel_tool_calls`, `response_format`, `seed`; `usage.prompt_tokens_details.cached_tokens`,
   `usage.completion_tokens_details.reasoning_tokens`; `chat.completion.chunk`, `choices[].delta` including `tool_calls[].index` deltas, the
   `finish_reason` values; whether the `data: [DONE]` terminator and the `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` and `retry-after`
   headers are in the spec or only in prose docs (if only docs, the primer keeps them `(verify)`).
2. **vLLM's OpenAI server at the repo pin** (§1, §3, §5, §6; `deploy/any-gpu/`, lab notebooks 01–04) — clone `vllm-project/vllm` with `--branch
   v0.30.0`; paths from main `4be061c` (below) may differ at the tag. Verify at the tag: `ChatCompletionRequest.cache_salt` (validation; that it salts
   the first block only, as serving-engine PRIMER §5 says), `priority` and the scheduling policy it needs, `StreamOptions` (`include_usage`,
   `continuous_usage_stats`), the CLI spellings of `enable_force_include_usage` and `enable_prompt_tokens_details`, `--api-key`, the error body,
   whether v0.30.0 serves an Anthropic-compatible `/v1/messages` (then the adapter table can be exercised against one vLLM at T1), and the embeddings
   endpoint and flags for a small pooling model (an optional T1 embedder for the semantic cache).
3. **OpenTelemetry GenAI conventions** (§5; `otel.py`, the lab's spans) — clone `open-telemetry/semantic-conventions-genai` (read
   `model/gen-ai/{spans,metrics,token-metrics,registry}.yaml`, `docs/gen-ai/gen-ai-spans.md`, `gen-ai-metrics.md`, `gen-ai-token-metrics.md`,
   `openai.md`, `anthropic.md`, `mcp.md`, `changelog.d/`) and `open-telemetry/semantic-conventions` at tag `v1.41.0` (`model/gen-ai/`, `docs/gen-ai/`;
   the last release that still defines them — v1.42.0 to v1.44.0 carry only a "Moved" stub, and the genai repo has no release tag yet). Verify: which
   names the gateway should pin — the set common to v1.41.0 and genai main, plus an explicit choice for token metrics; `gen_ai.provider.name`
   well-known values and what to set for a self-hosted OpenAI-compatible server; span name and kind for the gateway's server span and its client spans
   to providers; `gen_ai.usage.{input_tokens,output_tokens,cache_read.input_tokens,cache_write.input_tokens,reasoning.output_tokens}`,
   `gen_ai.response.model`, `gen_ai.response.time_to_first_chunk`, `gen_ai.request.stream`, `gen_ai.conversation.id`, `server.address`, `error.type`;
   the client metrics and the token counters and histograms that replace `gen_ai.client.token.usage`; an OTLP/JSON file layout the exporter can
   follow. Confirm the 07.2 lab's constants in `agentlab/observability/tracing.py` are still current.
4. **MCP authorization, client side** (§8; `mcp_authz.py`, `gwlab/mcp/`) — clone `modelcontextprotocol/modelcontextprotocol` (read
   `docs/specification/2026-07-28/basic/authorization/{index,authorization-server-discovery,client-registration,security-considerations}.mdx`, the
   same under `draft/`, `seps/1046-*.md`, `seps/991-*.md`, `seps/2468-*.md`, `docs/extensions/auth/enterprise-managed-authorization.mdx`) and
   `modelcontextprotocol/python-sdk` (`src/mcp/client/auth*`, a reference client: token storage, refresh, CIMD vs DCR). Verify: discovery order and
   fallbacks (PRM from `WWW-Authenticate` vs well-known URIs; RFC 8414 and OIDC path insertion and appending), `scope` in challenges and step-up, CIMD
   requirements and registration priority, the PKCE refusal rule, `resource` in both requests, refresh rotation for public clients, the `iss` check;
   the status of SEP-1046 (client credentials: the gateway acting on its own authority) and of the enterprise-managed authorization extension.
5. **OAuth 2.1 and Client ID Metadata Documents** (§8) — clone `oauth-wg/oauth-v2-1` (current draft number; §4.3.1 refresh rotation, §4.1.1 S256,
   §7.5.2 PKCE; what it says about refresh-token reuse detection) and `oauth-wg/draft-ietf-oauth-client-id-metadata-document` (current draft number,
   required metadata, the authorization server's fetch rules: caching, size limits, SSRF).
6. **DPoP nonces** (§8; `mcp_authz.py`) — clone `danielfett/draft-dpop` (`main.md`, the RFC 9449 source). Verify the RFC section numbers for the
   AS-provided and RS-provided nonce, `dpop_signing_alg_values_supported`, the proof claims (`jti`, `htm`, `htu`, `iat`, `ath`, `nonce`), nonce
   lifetime guidance and what a client may cache. The identity lab's `agentsec.identity.tokens.DPoP.proof()` accepts a `nonce`, but `verify()` never
   issues or checks one: that gap is what §8 closes.
7. **SPIFFE Workload API and SVID rotation** (§6; `keys.py`) — clone `spiffe/spiffe` (`standards/SPIFFE_Workload_API.md`,
   `SPIFFE_Workload_Endpoint.md`, `X509-SVID.md`) and `spiffe/spire` (`doc/spire_server.md`, `doc/spire_agent.md`,
   `pkg/common/rotationutil/rotationutil.go`). The spot-checked values are below; still verify the agent's availability-target setting (name and
   default) and how a rotation reaches a workload that keeps long-lived upstream connections.
8. **LiteLLM proxy** (§9 comparison; price source for item 13) — clone `BerriAI/litellm` (read `litellm/router.py`, the proxy's key-management and
   spend-tracking code under `litellm/proxy/`, `litellm/caching/`, `litellm/integrations/opentelemetry.py`, `docs/my-website/docs/proxy/`,
   `model_prices_and_context_window.json`). Verify: virtual-key fields (`models`, `max_budget`, `budget_duration`, `tpm_limit`, `rpm_limit`,
   `metadata`) and whether keys are hashed at rest; router `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `allowed_fails`,
   `cooldown_time`, `num_retries`; cache `type` values including the semantic ones and `similarity_threshold`; the spend-log table and its fields;
   guardrail `mode` values; how streamed usage is counted; the GenAI attributes its OTel integration emits.
9. **Envoy AI Gateway** (§2, §4, §5, §9) — clone `envoyproxy/ai-gateway` (read `api/v1alpha1/`, `site/docs/`, `examples/`). Verify: API group and
   version; `AIGatewayRoute`, `AIServiceBackend`, `BackendSecurityPolicy` (upstream credentials); the token-cost rules (`llmRequestCosts` and its
   types) that feed Envoy Gateway's global rate limit; how usage is read from streamed responses; backend priority and fallback; InferencePool
   support; the GenAI metrics and span attributes it emits.
10. **Kong AI Gateway** (§9) — clone `Kong/kong` (read `kong/plugins/ai-*/schema.lua`, `kong/llm/`). Verify which `ai-*` plugins are in the
    open-source repo and which the repo only references as Enterprise (for example `ai-proxy-advanced`, `ai-rate-limiting-advanced`,
    `ai-semantic-cache`), `ai-proxy`'s `route_type`, `model.provider` and `auth` fields, and the usage fields it logs.
11. **Portkey gateway** (§1, §2, §9) — clone `Portkey-AI/gateway` (read `src/providers/` — one directory per provider, the adapters-as-data comparison
    — `src/handlers/`, `src/middlewares/hooks/`, the config types). Verify `strategy.mode` values, `on_status_codes`, `targets[].weight`, `retry`,
    `cache.mode` (simple, semantic), the guardrail hook names, and the `x-portkey-config` header.
12. **Two non-OpenAI dialects for the adapter table** (§1; `providers.py` payloads, labelled "sample output in the documented format (illustrative)")
    — clone `anthropics/anthropic-sdk-python` (`src/anthropic/types/`: usage fields `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
    `cache_creation_input_tokens`; stop reasons; stream event types and which one carries usage; auth and version headers) and
    `googleapis/python-genai` (`google/genai/types.py`: `usage_metadata` token-count fields including cached and thinking tokens; finish reasons; the
    API-key header; the SSE streaming endpoint). Record each field path exactly; the adapter table is data built from them.
13. **Prices for the §5 model table, dated 2026-09-26** — re-check `scalelab.capacity.PRICES` (dated 2026-09-05 in the repo; the core must reproduce
    `cost_per_call` on those values, so if they changed keep both and say which is which), and take two or three further hosted rows from LiteLLM's
    `model_prices_and_context_window.json` at a pinned commit (`input_cost_per_token`, `output_cost_per_token`, `cache_read_input_token_cost`), all
    `(verify)`. Self-hosted rows come from the repo (`roofline.cost`, 01 PRIMER §8.1).
14. **Guardrail checkers for the §7 table** — clone `NVIDIA/NeMo-Guardrails` (rail types; streaming output-rail settings — chunk size, context size,
    whether tokens reach the client before the check) and `meta-llama/PurpleLlama` (current Llama Guard and Prompt Guard versions and sizes); Model
    Armor's modes come from the identity primer §4.1 and §6.1. For each checker: where it can sit (inline, parallel, stream windows) and whether it
    screens a stream.
15. **GCP facts for `deploy/gcp/README.md`** (no Terraform is written) — no upstream repo holds these, so they stay `(verify)` unless the repo already
    states them: `gcloud run deploy` for a CPU service with `--set-secrets`; ID tokens for a private Cloud Run upstream (audience = the service URL);
    Cloud Run's maximum request timeout (the scaling primer §5.7 says 60 minutes); the Cloud Trace OTLP endpoint (the scaling primer §9 names
    `telemetry.googleapis.com`); the 05 lab's Gateway address and route
    (`05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/gateway.yaml`).

**Spot-checked while drafting (2026-09-26) — confirm, do not rediscover:**
- `open-telemetry/semantic-conventions-genai` main `e57c543` (2026-09-24), all stability "development": the inference client span carries
  `gen_ai.operation.name` (required), `gen_ai.request.model`, `gen_ai.request.stream`, `gen_ai.response.model` ("the exact name of the model
  actually used"), `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.response.time_to_first_chunk`, `gen_ai.usage.input_tokens`,
  `…output_tokens`, `…cache_read.input_tokens`, `…cache_write.input_tokens`, `…reasoning.output_tokens` (included in output_tokens),
  `gen_ai.conversation.id`, `gen_ai.provider.name`, `server.address`, `error.type`. Metrics: `gen_ai.client.operation.duration`,
  `gen_ai.client.operation.time_to_first_chunk`, `gen_ai.client.operation.time_per_output_chunk`, `gen_ai.server.{request.duration,
  time_to_first_token,time_per_output_token}`. Unreleased breaking changes in `changelog.d/`: #374 replaces the `gen_ai.client.token.usage`
  histogram with `gen_ai.client.inference.usage.*` counters and `gen_ai.client.inference.operation.{input,output}_tokens` histograms; #440
  renames `gen_ai.usage.cache_creation.input_tokens` to `…cache_write.input_tokens`. #216: a span covers the operation as the caller sees it,
  retries included; #211: usage reports billed counts; #219: `gen_ai.conversation.id` only from a real conversation id; #220: MCP trace context
  travels as unprefixed `traceparent`/`tracestate`/`baggage` in `params._meta` (SEP-414).
- `open-telemetry/semantic-conventions`: v1.42.0 (2026-06-12), v1.43.0 and v1.44.0 (2026-08-04) hold only a "Moved" stub for GenAI; v1.41.0
  (2026-04-28) is the last release that defines it — metrics `gen_ai.client.operation.duration`, `…time_to_first_chunk`, `…time_per_output_chunk`,
  `gen_ai.client.token.usage`, the three `gen_ai.server.*`; attributes
  `gen_ai.usage.{input_tokens,output_tokens,cache_read.input_tokens,cache_creation.input_tokens,reasoning.output_tokens}`. So
  `gen_ai.client.token.usage` and `cache_creation` are the released names, `gen_ai.client.inference.usage.*` and `cache_write` the unreleased ones.
  Defined in both registries (genai main uses `file_format: definition/2` with `- key:`): every constant in the 07.2 lab's `tracing.py`, plus
  `gen_ai.provider.name`, `gen_ai.response.model`, `gen_ai.conversation.id`, `gen_ai.response.time_to_first_chunk`, `gen_ai.request.stream`;
  `gen_ai.provider.name` well-known values include `openai`, `anthropic`, `gcp.gemini`, `gcp.vertex_ai`, `gcp.gen_ai`.
- `modelcontextprotocol/modelcontextprotocol` main `ab3a39c` (2026-09-24): revision 2026-07-28 splits authorization into four pages; CIMD is SHOULD
  (`client_id_metadata_document_supported`; `client_id` an HTTPS URL with a path; `client_id`, `client_name`, `redirect_uris` required), Dynamic
  Client Registration is MAY and deprecated; clients MUST support both PRM discovery mechanisms and both AS metadata mechanisms, MUST use S256 and
  refuse when `code_challenge_methods_supported` is absent, MUST send `resource` in authorization and token requests; authorization servers MUST
  rotate refresh tokens for public clients (OAuth 2.1 draft-13 §4.3.1); RFC 9207 `iss` is listed. "DPoP" appears nowhere in the 2026-07-28 or
  draft authorization pages or the auth extensions.
- `vllm-project/vllm` main `4be061c` (2026-09-26): `ChatCompletionRequest` (`vllm/entrypoints/openai/chat_completion/protocol.py`) has
  `stream_options`, `max_completion_tokens`, `user`, `service_tier`, `priority` ("lower means earlier"; non-zero errors unless the model uses
  priority scheduling) and `cache_salt`; `StreamOptions(include_usage, continuous_usage_stats)` is in `vllm/entrypoints/generate/base/protocol.py`;
  `UsageInfo.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` in `vllm/entrypoints/serve/engine/protocol.py`;
  servers take `enable_prompt_tokens_details` and `enable_force_include_usage`; an Anthropic-compatible entrypoint exists in
  `vllm/entrypoints/anthropic/`.
- `danielfett/draft-dpop` `d1acf0d` (the RFC 9449 text): an AS answers 400 with `"error": "use_dpop_nonce"` and a `DPoP-Nonce` header; a resource
  server answers 401 with `WWW-Authenticate: DPoP error="use_dpop_nonce"` and `DPoP-Nonce`; a server may supply a fresh nonce on any response; at
  most one `DPoP-Nonce` header.
- `spiffe/spiffe` main: the Workload Endpoint is gRPC over a Unix socket (TCP only with strong network attestation), located by
  `SPIFFE_ENDPOINT_SOCKET` (`unix:///path`); `FetchX509SVID` and `FetchX509Bundles` are server-streaming and every message carries the full state.
  `spiffe/spire` main: `default_x509_svid_ttl` 1h, `default_jwt_svid_ttl` 5m; X.509-SVIDs rotate at half-life with ±10 % jitter
  (`pkg/common/rotationutil/rotationutil.go`) or at an availability target.
- RFC 7636 Appendix B: `code_verifier` `dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk` → S256 `code_challenge`
  `E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM` (recomputed with `hashlib`).

---

## Existing material (paths)

Cite and reuse; duplicating any of this is a major review finding. Paths are as of `main` on 2026-09-26: package `c2-mistral` is folding the
Mistral variants into `agentic-scaling-lab` and `agentic-identity-core`, and `c3-nbdirs` will move the 06 identity notebooks, so re-resolve paths at
build time. Cores never import other labs; a test that reproduces another lab's number loads that module by path and skips when it is absent, as
`00-foundations/rl-and-thinking-models/rl-core/tests/test_workload.py` (`_load()`) does. The lab never imports `gwcore`; its tests may pin both to
the same decisions when `gwcore` is installed (as `sandbox-lab/tests/test_probes.py` does).

- `06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md` — §3.4 cost per conversation (the $0.0070 call), §3.5
  Provisioned Throughput, §5.1 quota strategy and the token bucket (burst = capacity, Redis + Lua once there are two instances), §5.2 retries,
  full jitter, breakers, sibling-model fallback, hedging, §5.3 admission and degrade levels (a per-tenant bucket first), §5.7 streaming and
  connections, §5.10 observability (GenAI attributes, low-cardinality labels, the cost-per-conversation alert), §5.11 budgets, §5.12 tenancy,
  priority, regions, §9 the Google mapping (Apigee, Agent Gateway, Model Armor). Primer §2, §4, §5, §6 build on these and link them.
- `…/agentic-scaling-lab/scalelab/resilience.py` — `TokenBucket` (charges `tokens` up front; a request larger than the burst goes into debt),
  `backoff`, `CircuitBreaker`, `call_with_retries`. `gwcore.ratelimit`'s per-request bucket follows `TokenBucket`'s rule, and a test drives both
  over the same arrivals (the module imports `.clock`, so put `agentic-scaling-lab/` on `sys.path` in that test rather than loading the file alone).
- `…/scalelab/admission.py` — `AdmissionController`, degrade levels 0–3: cited in §4, not reimplemented.
- `…/scalelab/capacity.py` — `PRICES` (Gemini, checked 2026-09-05), `cost_per_call`: the hosted rows of the §5 table; a test asserts
  `cost_per_call("gemini-3.5-flash", 5000, 350, 2700)` is $0.007005, the number the RL primer §5 also reproduces.
- The 06.5 hosted-vs-own-GPUs break-even (`agentic-scaling-lab-mistral/docs/01-scaling-primer.md` §3.5–3.6, `scalelab/serving.py`; moving under
  `c2-mistral`): cite by module number when §2 and §5 compare a hosted provider with a self-hosted pool.
- `06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md` — §0 (idea 4: deterministic controls outrank probabilistic ones), §3.3
  (agent identity, 24-hour certificates, sender-constrained tokens), §3.5 (RFC 8693, DPoP, mTLS-bound tokens), §4.1 (four enforcement points,
  Model Armor as the model-side one), §5 (secrets; the direct path vs the gateway path), §6.1 (screen in and out; `INSPECT_ONLY` first), §7.1 (the
  MCP server as an OAuth 2.1 resource server), §8 (tenancy), §9 (the audit event), §10 (Agent Gateway). Primer §6, §7, §8 link these.
- `…/agentic-identity-gcp-lab/src/agentsec/identity/tokens.py` — `DPoP.proof(..., nonce=)` and `DPoP.verify()` (`jti` replay, `ath`, `cnf.jkt`; no
  server nonce) with real keys via `cryptography`; `…/agentsec/audit/log.py` — `AuditEvent`, whose fields the ledger row and the gateway's audit
  record line up with.
- `07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/auth/oauth.py` (notebook `06_oauth_identity_propagation`) —
  `make_verifier`, `challenge_for`, `AuthorizationServer` (RFC 8414 metadata, PKCE, the RFC 9207 `iss` check), `discover_and_authorize`, `step_up`,
  `parse_www_authenticate`, HS256 tokens with stdlib `hmac`. It already runs discovery → PKCE → code → token: `gwcore.mcp_authz` reproduces
  `challenge_for` on RFC 7636 Appendix B and adds only what a gateway needs — Client ID Metadata Documents, refresh rotation with reuse
  detection, a token cache per (principal, resource, scopes), DPoP nonces.
- `…/gcp-agent-platform-lab/agentlab/mcp/gateway.py` (notebook `05_mcp_server_client_gateway`) — the MCP egress gateway: policy, screening, no token
  passthrough, audit. Primer §8 starts from it. `docs/MCP_REVISIONS.md` lists the authorization changes per MCP revision.
- `…/gcp-agent-platform-lab/agentlab/observability/tracing.py` and `metrics.py` (notebook `09_tracing_and_metrics`) — the `GEN_AI_*` constants
  (`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `…output_tokens`, `…cache_read.input_tokens`,
  `gen_ai.response.finish_reasons`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, opt-in `gen_ai.input.messages`/`output.messages`),
  `RedactingExporter`, `PriceTable`, `ttft_and_tps`. `gwcore.otel` uses the same names, and a test compares them.
- `…/gcp-agent-platform-lab/agentlab/reliability/fallback.py` and `breaker.py` (notebook `10_reliability_retries_breakers`) — `FallbackChain` (labels
  degraded results) and the circuit breaker's canonical home (CURRICULUM §3.4): §2 links them; `routing` applies a breaker per target.
- `…/gcp-agent-platform-lab/notebooks_src/07_agent_api_streaming_tasks.py` — the agent's own API: SSE events, `Last-Event-ID`, `Idempotency-Key`, a
  per-tenant 429. §1 separates it from the gateway's OpenAI-compatible token stream.
- `05-orchestrator/serving-orchestration/PRIMER.md` — §1.3 (the three decisions; admission is 06's), §3.2 (`InferenceObjective` priority), §3.3
  (who decides what), §7 ("model routing proper … is a gateway decision"; one pool per base model; body-based routing; `InferenceModelRewrite`), §9
  (the Kubernetes-native stack). §1, §2 and §6 state the 05/06 split from these.
- `05-orchestrator/serving-orchestration/inference-gateway-lab/` — the lab's structural model: `igwlab/router/server.py` (aiohttp streaming relay,
  error shapes, `/metrics`, `/debug/state`), `igwlab/fakebackend.py`, `igwlab/promtext.py`, `igwlab/configs/*.yaml`, `tests/test_router_e2e.py`
  (real localhost HTTP in tests), `deploy/local/` (compose with `up.sh`/`smoke.sh`/`down.sh`, `DRY_RUN=1`), `deploy/any-gpu/` (the T1 recipe to
  mirror), and `deploy/gcp/terraform/` + `deploy/gke/` (the GKE Inference Gateway that `deploy/gcp/README.md` points at).
- `04-inference-engine/serving-engine/PRIMER.md` — §1 (the API server), §5 (prefix caching, the per-tenant `cache_salt`, prompts laid out for hits),
  §11 (TTFT, ITL, open vs closed loop), §12. `…/vllm-serving-lab/servelab/fakeserver.py` already emulates `stream_options.include_usage` and
  `prompt_tokens_details.cached_tokens` (the model for `gwlab/fakes.py`); `…/vllm-serving-lab/deploy/gcp/cloud-run/terraform/` and
  `deploy/gcp/gke/` are the T3 upstreams. `00-foundations/rl-and-thinking-models/thinking-lab/deploy/gcp/README.md` is the model for a no-Terraform
  pointer README.
- `00-foundations/rl-and-thinking-models/PRIMER.md` — §5 ("a thinking token is billed as an output token"; `reasoning_tokens`), §7 (heavy-tailed
  outputs; "routing by effort at the gateway").
- `01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md` §8.1 and `roofline-core/roofline/cost.py` — `cost_per_million_tokens(price_per_gpu_hour,
  tokens_per_s, utilisation=1.0, n_gpus=1)`; the self-hosted rows of the §5 table reproduce §8.1's H100 Spot bf16 ($0.150/M at 100 %) and L4
  ($0.660/M) rows.
- `07-application-agent-framework/retrieval-rag/embeddings-lab/docs/primer.md` §15 ("Embeddings elsewhere in agent systems": semantic caching's
  false positives, kNN routing) and `…/retrieval-rag/vector-databases-primer.md` §17 — §3 links them, does not restate them.
  `…/retrieval-rag/rag-from-scratch/ragkit/embed.py` `HashingEmbedder` (crc32 buckets, L2-normalised, labelled "not semantic") is the construction
  `gwcore.cache` reuses; a test compares vectors on shared tokens.
- `07-application-agent-framework/sandboxed-execution/PRIMER.md` §4 — the egress proxy that injects credentials the caller never holds: the same
  pattern as provider keys held only by the gateway (§6).
- `CURRICULUM.md` §2 (the 06 row this topic closes), §3.4 (canonical homes: token bucket 06.3, circuit breaker 07.2 notebook 10, prompt injection
  06.6, prefix caching 04.3 — apply them, do not re-teach them), §6 (the backlog line); `COMPUTE.md` §6 and §9 (tiers, prices, the `(verify)`
  convention); `tools/orchestration/reviews/2026-09-26-adversarial-review.md` §9, 06 row (the acceptance list: routing and fallback chains,
  semantic caching, metering and chargeback, OTel GenAI conventions, streaming-aware limits, virtual keys, tenant isolation, guardrail placement
  and cost, the MCP client flow, DPoP nonces, the SPIFFE Workload API and SVID rotation).

---

## Curriculum

**Module number: 06.7** (06.1–06.6 are taken). For `CURRICULUM.md` §4, under "### 06 · Gateway", after the table:

#### 06.7 The LLM gateway — [`llm-gateway`](../../../../06-gateway/llm-gateway/README.md)

"Put one front door in front of every model: route and fall back across providers, cache what is safe, meter tokens as they stream, and keep
tenants apart." Primer: [`PRIMER.md`](../../../../06-gateway/llm-gateway/PRIMER.md). Core: [`gateway-core`](../../../../06-gateway/llm-gateway/gateway-core/) (package
`gwcore`, standard library + numpy: the chat-completions lingua franca and an adapter table, routing and fallback chains, exact and semantic
caches, token buckets that meter streamed output, a metering ledger and chargeback, virtual keys, guardrail placement, the MCP client
authorization flow, GenAI spans). Lab: [`gateway-lab`](../../../../06-gateway/llm-gateway/gateway-lab/) (package `gwlab`: an async OpenAI-compatible gateway
in front of fake providers or a real vLLM; every notebook runs at T0).

| Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---|---:|---|
| **06.7.1 One front door** | say what the gateway decides and what it leaves to the router (05) and the engine (04); normalise three providers to chat completions with an adapter table; relay a stream chunk by chunk and meter it, disconnects included; issue a virtual key and read a request's GenAI spans back from a file | §1 One front door, one API | [`01_one_front_door`](../../../../06-gateway/llm-gateway/gateway-core/notebooks/01_one_front_door.ipynb) | [`01_a_gateway_over_http`](../../../../06-gateway/llm-gateway/gateway-lab/notebooks/01_a_gateway_over_http.ipynb) | 3 | T0 (T1 with one vLLM) |
| **06.7.2 Routing and fallback chains** | build alias chains with capability filters and routing policies; say which failures fall through and why a stream never falls back after its first byte; compute a chain's availability with independent and common-mode failures, and its expected latency and cost; watch breakers and fallbacks through a scripted outage | §2 Model routing and fallback chains | [`02_routing_and_fallback_chains`](../../../../06-gateway/llm-gateway/gateway-core/notebooks/02_routing_and_fallback_chains.ipynb) | [`02_outages_fallbacks_and_breakers`](../../../../06-gateway/llm-gateway/gateway-lab/notebooks/02_outages_fallbacks_and_breakers.ipynb) | 3 | T0 → T1 |
| **06.7.3 Caching at the gateway** | place exact, semantic, provider prompt and engine prefix caches and say what each saves and risks; sweep a similarity threshold for hit rate against false hits, with and without an entity guard; keep tenants apart in the cache and in the engine (`cache_salt`); measure a semantic hit against a prefix-cache hit | §3 Caching at the gateway | [`03_exact_and_semantic_caching`](../../../../06-gateway/llm-gateway/gateway-core/notebooks/03_exact_and_semantic_caching.ipynb) | [`03_semantic_cache_vs_the_prefix_cache`](../../../../06-gateway/llm-gateway/gateway-lab/notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) | 3 | T0 → T1 |
| **06.7.4 Tokens: limits, metering and chargeback** | show how far a per-request bucket over-admits a provider's tokens per minute and fix it with reserve → stream → reconcile; enforce RPM and TPM per tenant; price a request from `usage`, reasoning tokens included; compute $ per 1M tokens across hosted and self-hosted models and a per-tenant chargeback; reconcile the ledger with the provider's counters | §4 Streaming-aware rate limits · §5 Metering, tracing and chargeback | [`04_token_limits_metering_and_chargeback`](../../../../06-gateway/llm-gateway/gateway-core/notebooks/04_token_limits_metering_and_chargeback.ipynb) | [`04_streaming_limits_metering_and_chargeback`](../../../../06-gateway/llm-gateway/gateway-lab/notebooks/04_streaming_limits_metering_and_chargeback.ipynb) | 4 | T0 → T1 |
| **06.7.5 Keys, guardrails and MCP authorization** | hold provider keys only in the gateway and rotate them; scope and revoke virtual keys; place a guardrail and predict the TTFT and E2E it adds; run the MCP client flow — discovery, a Client ID Metadata Document, PKCE, refresh rotation, step-up — and answer a DPoP nonce challenge; explain SVID rotation | §6 Keys, tenants and isolation · §7 Guardrails and what they cost · §8 The gateway as an MCP client · §9 Where to run it, and what to adopt | [`05_guardrails_keys_and_mcp_authorization`](../../../../06-gateway/llm-gateway/gateway-core/notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) | [`05_guardrails_and_mcp_authorization_over_http`](../../../../06-gateway/llm-gateway/gateway-lab/notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) | 3.5 | T0 |

Deploy targets: [`local`](../../../../06-gateway/llm-gateway/gateway-lab/deploy/local/) (docker compose: the gateway and two fake providers),
[`any-gpu`](../../../../06-gateway/llm-gateway/gateway-lab/deploy/any-gpu/) (the gateway in front of one vLLM, a fake provider as the fallback),
[`gcp`](../../../../06-gateway/llm-gateway/gateway-lab/deploy/gcp/) (a README: the 04 lab's Cloud Run or GKE vLLM and the 05 lab's GKE Inference Gateway as
upstreams; no new Terraform).

**Hours: 16.5** (3 + 3 + 3 + 4 + 3.5), estimated like the other modules: an engineer comfortable with Python who does every exercise, T0 throughout;
add about 2 GPU-hours for the T1 paths of lab notebooks 01–04.

Other edits the integrator makes in `CURRICULUM.md`:
- §3.2: a new step 25 after step 24 (06.6) — `| 25 | 06 | 06.7 | llm-gateway PRIMER; gateway-core 01–05; gateway-lab 01–05 | 16.5 | T0 | T0; T1 for
  lab 01–04 |` — and steps 25–28 become 26–29. The total becomes about 285.5 hours (269 + 16.5), and the newer topics about 85 of them. §3.1 gets one sentence:
  the gateway comes after identity (06.6) because its §6 and §8 build on the identity primer's token discipline and OAuth.
- §3.3, "Agent builder" route (optional): add 06.7.1 and 06.7.4 (core 01 and 04, about 4 h) after 06.1–06.3.
- §3.4: add 06.7.4 to the token bucket's "also applied in" (buckets in tokens that meter streamed output), 06.7.2 to the circuit breaker's
  (a breaker per provider target), 06.7.3 to prefix caching's (beside the gateway's semantic cache) and 06.7.5 to prompt injection's (where a
  guardrail sits and what it costs); add a row "Semantic caching → 06.7.3 (llm-gateway PRIMER §3, `gateway-core` 03): the only place that sweeps
  the threshold against false hits on labelled traffic and places it beside prompt and prefix caches; also in 07.4 (embeddings primer §15,
  vector-databases primer §17)".
- §2, 06 row: move the gateway items out of "Not covered yet" and leave what this topic does not do (for example a distributed limiter measured
  under real concurrency, multi-region gateway failover, MCP client credentials and enterprise-managed authorization, a real classifier guardrail
  measured, A2A through the gateway).
- §6: mark the gateway row built ("An LLM gateway, now [`llm-gateway`](../../../../06-gateway/llm-gateway/README.md) (06.7)").
- §5.2, three cross-layer drills:

| # | Prompt | Answer sketch | Modules |
|---:|---|---|---|
| 16 | A provider outage lasted five minutes; our incident lasted forty, and the bill doubled. | Clients retried without a budget, so the retry wave outlasted the outage; streams that failed midway were re-run from the start, paying twice for output already generated; and the whole chain fell through to a pricier model at full traffic until that model's own quota ran out. Retry budgets with jitter, a breaker per target, fallback only before the first byte (after it, surface the error), fallback capacity sized and priced in advance in an independent failure domain, and a cost alert per tenant. | 06.7.2, 06.3, 07.2 |
| 17 | The semantic cache answered one user with another user's order status. | The query was personal, a class that must never be cached, and the key had no tenant or user namespace; a lexical near-match ("order 1234" against "order 1243") cleared the threshold. Classify what may be cached, namespace by tenant, guard numbers and entities exactly, and measure false hits on labelled traffic before lowering the threshold. The engine's prefix cache is exact and is isolated with `cache_salt`. | 06.7.3, 04.3, 06.6 |
| 18 | After a thinking-model rollout the provider started returning 429s, but our request rate limit never tripped. | The bucket charged per request while outputs grew by an order of magnitude with a heavy tail, so the same request rate carried many times the tokens per minute. Meter tokens: reserve at admission, debit as chunks stream, reconcile at the end, cap output; enforce TPM beside RPM per tenant; budget thinking rather than truncating it with `max_tokens`. | 06.7.4, 00.5, 06.3 |
