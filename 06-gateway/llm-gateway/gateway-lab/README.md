# gateway-lab — run an LLM gateway over real HTTP and watch it route, fall back, cache, meter and authorize

After this lab, you can put one OpenAI-compatible front door in front of several providers. You can say what the
gateway decides, and what it leaves to the router and the engine. You can also show these things with requests that
you sent yourself:

- How the gateway falls back before the first byte.
- How it meters streamed tokens, disconnects included.
- How it keeps tenants apart in its limits, caches and ledger.
- How it places a guardrail.
- How it gets MCP tokens for the agents behind it.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m gwlab stack`. It takes less than ten seconds.
   You get the gateway, two fake providers and a demo key on your laptop, with a `curl` line to try.
2. Read the primer's [§1 One front door, one API](../PRIMER.md). It gives the concepts that every notebook uses.
   Then open [`notebooks/01_a_gateway_over_http.ipynb`](notebooks/01_a_gateway_over_http.ipynb).
3. When you have a GPU of any type (a free Colab T4 is sufficient), use [`deploy/any-gpu/`](deploy/any-gpu/README.md).
   It puts a real vLLM behind the same gateway, and notebooks 01–04 measure it.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, free. T1 is one small GPU (Colab/Kaggle T4 or a rented card). T2 is a
multi-GPU box, rented for an hour (not used here). T3 is the Google Cloud deployment, optional.*

Every notebook runs at T0. The gateway is real. Its upstreams are fake providers, and their timings and token counts are **simulated**
and have a label that says so.

Each notebook opens with *the one-minute version* and gives worked examples over HTTP. Then it gives 4–5 exercises
(write the key function, predict a number, select a setting). A check that prints ✅ comes after each exercise. The
notebook closes with *in a design review*. The answers are in [`solutions/`](solutions/). The times are for the lab
notebook alone: about 6 hours for all five.

| # | Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|---|
| 01 | [`a_gateway_over_http`](notebooks/01_a_gateway_over_http.ipynb) | Issue a hashed, scoped virtual key. Trace one request through the pipeline. Read the usage of Anthropic from live stream events, and know when a cut stream has no usage. Rebuild tool calls from deltas, and see an error after the first byte. Translate a body that `bolt` accepts. See why the gateway injects `include_usage` and strips it. Measure the time of the gateway's own hop from its spans. | §1, §5, §6 | ~1.25 h | T0, T0 + Docker, T1 |
| 02 | [`outages_fallbacks_and_breakers`](notebooks/02_outages_fallbacks_and_breakers.ipynb) | Predict what a client sees for seven live faults (fall through, an error in the stream, or a 502). Read the failure rate of each target and the availability of the chain from the decision log. Predict the latency cost of a fallback. Predict how many requests a breaker lets reach a dead target. | §2 | ~1 h | T0, T1 (stops vLLM during the run, opt-in) |
| 03 | [`semantic_cache_vs_the_prefix_cache`](notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) | Make the cache key from the classes that the route declares (per-user classes per user), with a regex only as a veto. Do a sweep of a semantic threshold. Then make sure that the live gateway serves exactly what the sweep predicted. Do the prefix-cache timing attack that `cache_salt` closes. Predict `cached_tokens` from the block rule of vLLM. | §3 | ~1.25 h | T0 emulated, T1 measured |
| 04 | [`streaming_limits_metering_and_chargeback`](notebooks/04_streaming_limits_metering_and_chargeback.ipynb) | Show how far a per-request bucket over-admits the TPM of a provider. Repair this with reserve, then stream, then reconcile. Protect a tenant from a noisy neighbour. Calculate the price of requests from `usage`. Divide a shared GPU by tokens or by GPU-seconds. Reconcile the ledger with the counters of the provider. | §4, §5 | ~1.25 h | T0, T1 (vLLM's `/metrics`) |
| 05 | [`guardrails_and_mcp_authorization_over_http`](notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) | Measure what each guardrail placement adds to TTFT and what it lets leak. Rotate a provider key under load, and fail no request. Run the MCP client flow: discovery, CIMD, PKCE, step-up. Show that a second MCP server refuses a token minted for a different MCP server. Do a refresh rotation with reuse detection. Use DPoP nonces. | §6, §7, §8 | ~1.25 h | T0 |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the gateway and fake providers in one process (`gwlab.stack.LocalStack`), over localhost HTTP | every notebook, all tests |
| **T0 + Docker** | any machine with Docker | the gateway and two fake providers (one fails 30 % of requests) in Compose | [`deploy/local/`](deploy/local/README.md). Notebook 01 talks to it through `GWLAB_URL`. |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm/vllm-openai:v0.30.0`, which serves Qwen2.5-0.5B-Instruct as `lab/llm` behind the gateway, and a fake fallback | [`deploy/any-gpu/`](deploy/any-gpu/README.md). `GWLAB_VLLM_URL` turns on the T1 cells of notebooks 01–04. |
| **T3** | GCP | the 04 lab's Cloud Run or GKE vLLM, or the 05 lab's GKE Inference Gateway, as upstreams (no new Terraform) | [`deploy/gcp/`](deploy/gcp/README.md) |

## Run it

```bash
cd gateway-lab
python3 -m pip install -e ".[dev]"            # aiohttp, pyyaml, numpy; dev: pytest, jupyter. Optional: ".[dpop]" for ES256
python3 -m pytest -q                          # 165 tests + 2 skipped (166 + 1 with the [dpop] extra), ~25 s, offline, real localhost HTTP
python3 -m gwlab stack --port 8080            # fakes + gateway; prints the admin token, a demo key and a curl line
python3 -m gwlab key --tenant team-b          # another virtual key from the admin API
python3 -m jupyterlab notebooks               # the exercises; answers in solutions/
```

Talk to it as you talk to any OpenAI-compatible server. The key is a virtual key, and the model is an alias:

```bash
curl -sN localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"chat","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8080/metrics | grep ^gwlab_
curl -s localhost:8080/debug/state -H 'Authorization: Bearer dev-admin-token'     # breakers, buckets, recent decisions
```

## The library (`gwlab/`, ~5,400 lines)

| Module | Lines | The idea |
|---|---:|---|
| `gateway/server.py` | ~730 | The §1 pipeline in aiohttp. Its steps, in this order: authenticate, input guardrail, cache, admit, route, call, relay, account. Authenticate takes the tenant from the key, never from a header. Admit applies RPM + TPM. Call uses the key of the provider. Relay parses every chunk, translates, injects and strips usage, meters, runs the output guardrail, and falls back only before the first byte. Account reconciles and writes the ledger row, the spans and the metrics. The endpoints are `/v1/models`, `/metrics`, `/admin/keys`, `/debug/state`, `/mcp/{server}`. |
| `gateway/adapters.py` | ~340 | Provider adapters as data: the OpenAI, Anthropic and Gemini rows (paths, auth headers, usage fields, finish reasons). It translates requests to Anthropic Messages, and Anthropic stream events to chat chunks. It normalises errors to string codes. |
| `gateway/config.py` | ~260 | The YAML config: providers, the model catalogue with prices, aliases (ordered chains and policies), tenants, limits, cache, guardrails, breakers. It fills `${VAR:-default}` from the environment. |
| `gateway/routing.py` | ~170 | Fall-through classes, a consecutive-failure breaker per target (07.2's rule), capability/context/residency filters, ordered/cheapest/EWMA-TTFT/canary policies, chain availability and expected latency. |
| `gateway/ratelimit.py` | ~190 | Token buckets with debt, for RPM and TPM per tenant, per key and for the whole gateway. `reserve` (reserve, then stream, then reconcile, a hard cap) against `per_request`. The `x-ratelimit-*` headers. |
| `gateway/cache.py` | ~290 | Exact and semantic caches in sqlite3 + numpy, with the hashing embedder of `ragkit`. The class that the route declares sets cacheability (`metadata.cache_class`, on an allowlist). A regex only vetoes. Namespaces per tenant, alias and system prompt, plus the user for per-user classes. The entity guard, `cache_salt`, the labelled traffic sample and the threshold sweep. |
| `gateway/metering.py` | ~160 | Dated price rows (verify), cost from canonical usage, $/1M blended and self-hosted, ledger rows, totals, chargeback by tokens or GPU-seconds, reconciliation. |
| `gateway/keys.py`, `store.py` | ~200 | Virtual keys with a hash at rest, a scope, a budget and revocation. Provider keys with rotation and overlap. The three sqlite tables. |
| `gateway/guardrails.py` | ~110 | A regex screener (a labelled stand-in) and the latency of each placement: inline, parallel, held-back windows, full, shadow. |
| `gateway/otel.py` | ~210 | Spans with the GenAI names pinned (v1.41.0 ∩ genai main), OTLP/JSON lines and read-back, OTLP/HTTP export when installed. |
| `fakes.py` | ~560 | Fake providers over HTTP in two dialects, with the special behaviours of vLLM. They have their own RPM/TPM windows, a 16-token-block prefix cache with `cache_salt`, heavy-tailed outputs, tool calls, reasoning tokens and switchable faults. Their `/metrics` use the names of vLLM. |
| `mcp/` | ~830 | A fake authorization server and a fake MCP server. The MCP client of the gateway (discovery, CIMD, PKCE S256 with `resource`, per-principal tokens, step-up, rotation with reuse detection). DPoP proofs and nonces (ES256 with `cryptography`, or, without it, a labelled HMAC stand-in). |
| `t1.py` | ~160 | The T1 cells of notebooks 01–04 as functions. A gateway in front of a real vLLM gives a result the label `MEASURED` only when vLLM served it. A function stops vLLM during the run. Assertions on measured rows hold the prefix-cache rule and the salt. The `/metrics` reconciliation skips when vLLM is gone (tested offline against a stand-in). |
| `sse.py`, `tokens.py`, `promtext.py`, `client.py`, `bench.py`, `report.py`, `stack.py`, `env.py`, `__main__.py` | ~1,100 | An SSE parser and stream accumulation, the labelled token estimate, Prometheus text, a client that measures time, scripted tenants (open loop). Tables, the in-process stack, tier detection (T1 only if vLLM answers `/health`), the CLI. |

This lab never imports the core package next to it (`gateway-core`, `gwcore`). Instead, its tests pin its numbers to
the canonical homes of the repo ([`tests/test_repo_numbers.py`](tests/test_repo_numbers.py)):

- `scalelab.capacity.cost_per_call` ($0.007005 for the §3.4 call of the scaling primer).
- `roofline.cost.cost_per_million_tokens` (the $0.150 and $0.660 of 01 PRIMER §8.1).
- The `challenge_for` of 07.2 on RFC 7636 Appendix B, and its GenAI span names.
- The hashing embedder of `ragkit`.
- The rule of `scalelab.resilience.TokenBucket`.

## Where it sits

Read [`../PRIMER.md`](../PRIMER.md) at the same time as this lab. The minimal in-process version of every mechanism
is in [`../gateway-core/`](../gateway-core/). The material before this lab is:

- The token bucket and admission control of [06.3 `agentic-scaling-lab`](../../scaling-admission-cost/agentic-scaling-lab/).
- The token discipline and OAuth of [06.6 identity](../../identity-security/agentic-identity-gcp-lab/docs/primer.md).
- The MCP gateway and breakers of the 07.2 platform lab ([`gcp-agent-platform-lab`](../../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/)).

Below this lab are the replica router of layer 05 and the engine of layer 04. This gateway uses the replica router
([`inference-gateway-lab`](../../../05-orchestrator/serving-orchestration/inference-gateway-lab/)) as one upstream
pool. It depends on the prefix cache and the `usage` of the engine
([`vllm-serving-lab`](../../../04-inference-engine/serving-engine/vllm-serving-lab/)).

## Caveats

- **Simulated vs measured.** The latencies of the fake providers are sleeps, and their token counts come from a word
  tokenizer. Every response carries `x-gwlab-simulated: true`, and every notebook labels such numbers. If
  `GWLAB_VLLM_URL` has a value and vLLM answers `/health`, the T1 cells measure a real vLLM. They label a row
  `MEASURED` only when vLLM served it (`gwlab/t1.py`). A fallback to a fake has the label `SIMULATED`.
- **Compressed time.** Notebook 04 compresses the rate-limit "minute" to 2 s, so that a run takes seconds. The
  arithmetic is the same at 60 s.
- **Stand-ins, labelled.** These parts are stand-ins:
  - The guardrail is a regex with a simulated check time.
  - The embedder of the semantic cache is lexical (a hashing embedder).
  - The second guard of the cache is a regex that can only veto a declared class. The route declares the class
    itself, and no code infers it.
  - The fake AS auto-approves the `login_hint` principal (there is no browser).
  - Tokens between the two fakes are HS256.
  - Without `cryptography`, DPoP proofs are HMAC-signed, and they are not RFC 9449-conformant.
  - The fake AS accepts loopback HTTP `client_id` URLs in development only.
- **One process.** Keys, cache and ledger are sqlite3, and the buckets live in memory. If you run two gateway
  replicas, they need a shared database and Redis-backed buckets (scaling primer §5.1). This lab does not build
  multi-region gateway failover or a distributed limiter under real concurrency. It also does not build MCP client
  credentials and enterprise-managed authorization, or a measurement of a real classifier guardrail.
- **Checked by construction.** The checks of the deploy paths are `bash -n`, `DRY_RUN=1` runs and tests of
  config/compose consistency. This lab does not run the deploy paths on real infrastructure.

## Verify list (facts dated September 2026 that move)

These facts come from upstream sources, read on 2026-09-26 (the fact sheet of the topic names each file). Examine
them again when you move a pin:

- vLLM **v0.30.0**:
  - `stream_options` without `stream` is a 400.
  - vLLM validates `cache_salt` (at most 128 characters, no `@ / \` or NUL). The salt applies to the first block
    only.
  - `--enable-prompt-tokens-details` adds `cached_tokens`.
  - `--enable-force-include-usage` also turns on continuous usage.
  - A failure during the stream is an `error` object in a `data:` event inside the HTTP 200. Then `[DONE]` comes.
  - `ErrorInfo.code` is the HTTP status as an int.
  - `--api-key` protects only `/v1`, `/v2`, `/inference` and `/cohere` (`GUARDED_PREFIX`). `/health`, `/metrics`
    and the rest stay open.
  - The field for reasoning text is `reasoning`.
- OpenAI chat completions (openai-openapi 2.3.0): `include_usage` sends a final chunk with `choices: []`, and the
  other chunks have `usage: null`. `max_tokens` is deprecated, and `max_completion_tokens` replaces it.
  `Retry-After` is in the spec, but the `x-ratelimit-*` headers are only in prose docs.
- Anthropic Messages (anthropic-sdk-python 1.8.0): `input_tokens` does not include cache reads and writes. Also the
  stream event names, and 529 `overloaded_error`. Gemini (python-genai 2.25.0): `thoughtsTokenCount` is outside
  `candidatesTokenCount`.
- OpenTelemetry GenAI: the names have the status "development". `gen_ai.usage.cache_creation.input_tokens` and
  `gen_ai.client.token.usage` are the v1.41.0 names. The main branch of the GenAI repo renamed them (`MAIN_NAMES`).
  `gen_ai.provider.name` has no value for a self-hosted server (this lab uses `vllm`, a choice).
- MCP authorization, revision **2026-07-28**: the discovery order, CIMD as SHOULD, and DCR deprecated. S256 is
  necessary, and a refusal comes when the server does not advertise it. `resource` is in both requests. Also RFC
  9207 `iss`. DPoP
  is RFC 9449, and it is not part of the MCP spec. The CIMD draft rules (5 KB, no special-use addresses) come from
  the IETF editor's copy.
- The prices in `gateway/metering.py` (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gpt-5.4-mini`,
  `claude-haiku-4-5`), dated 2026-09-26. The GPU prices in `configs/vllm.yaml` and in the deploy READMEs.
- `vllm/vllm-openai:v0.30.0` and `Qwen/Qwen2.5-0.5B-Instruct` for the T1 path. The availability of GPUs on Colab
  and Kaggle.

MIT licensed.
