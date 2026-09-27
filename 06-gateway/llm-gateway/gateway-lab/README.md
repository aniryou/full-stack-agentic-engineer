# gateway-lab — run an LLM gateway over real HTTP and watch it route, fall back, cache, meter and authorize

After this lab you can put one OpenAI-compatible front door in front of several providers, say what it decides
and what it leaves to the router and the engine, and show — with requests you sent yourself — how it falls back
before the first byte, meters streamed tokens (disconnects included), keeps tenants apart in its limits, caches
and ledger, places a guardrail, and obtains MCP tokens for the agents behind it.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m gwlab stack` — under ten seconds: the gateway, two fake
   providers and a demo key on your laptop, with a `curl` line to try.
2. Read the primer's [§1 One front door, one API](../PRIMER.md) (the concepts every notebook uses), then open
   [`notebooks/01_a_gateway_over_http.ipynb`](notebooks/01_a_gateway_over_http.ipynb).
3. When you have any GPU (a free Colab T4 is enough): [`deploy/any-gpu/`](deploy/any-gpu/README.md) puts a real
   vLLM behind the same gateway, and notebooks 01–04 measure it.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU
box, rented for an hour (not used here); T3 = the Google Cloud deployment, optional.* Every notebook runs at T0:
the gateway is real, its upstreams are fake providers whose timings and token counts are **simulated** and
labelled so. Each opens with *the one-minute version*, works examples over HTTP, then 4–5 exercises (implement
the key function, predict a number, pick a setting) each followed by a check that prints ✅, and closes with
*in a design review*. Answers are in [`solutions/`](solutions/). Times are for the lab notebook alone, about 6 hours
for all five.

| # | Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|---|
| 01 | [`a_gateway_over_http`](notebooks/01_a_gateway_over_http.ipynb) | issue a hashed, scoped virtual key; trace one request through the pipeline; read Anthropic's usage off live stream events, and know when a cut stream has none; rebuild tool calls from deltas and notice an error after the first byte; translate a body `bolt` accepts; see why the gateway injects `include_usage` and strips it; time the gateway's own hop from its spans | §1, §5, §6 | ~1.25 h | T0; T0 + Docker; T1 |
| 02 | [`outages_fallbacks_and_breakers`](notebooks/02_outages_fallbacks_and_breakers.ipynb) | predict what a client sees for seven live faults (fall through, an error in the stream, or a 502); read per-target failure rates and the chain's availability off the decision log; predict the latency a fallback costs; predict how many requests a breaker lets reach a dead target | §2 | ~1 h | T0; T1 (stops vLLM mid-run, opt-in) |
| 03 | [`semantic_cache_vs_the_prefix_cache`](notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) | key the cache on classes the route declares (per-user ones per user), with a regex only as a veto; sweep a semantic threshold, then check the live gateway serves exactly what the sweep predicted; play the prefix-cache timing attack that `cache_salt` closes; predict `cached_tokens` from vLLM's block rule | §3 | ~1.25 h | T0 emulated; T1 measured |
| 04 | [`streaming_limits_metering_and_chargeback`](notebooks/04_streaming_limits_metering_and_chargeback.ipynb) | show how far a per-request bucket over-admits a provider's TPM and fix it with reserve → stream → reconcile; protect a tenant from a noisy neighbour; price requests from `usage`; split a shared GPU by tokens or GPU-seconds; reconcile the ledger with the provider's counters | §4, §5 | ~1.25 h | T0; T1 (vLLM's `/metrics`) |
| 05 | [`guardrails_and_mcp_authorization_over_http`](notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) | measure what each guardrail placement adds to TTFT and what it lets leak; rotate a provider key under load without failing a request; run the MCP client flow — discovery, CIMD, PKCE, step-up; show a token minted for one MCP server refused at another; refresh rotation with reuse detection; DPoP nonces | §6, §7, §8 | ~1.25 h | T0 |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the gateway + fake providers in one process (`gwlab.stack.LocalStack`), over localhost HTTP | every notebook, all tests |
| **T0 + Docker** | any machine with Docker | the gateway + two fake providers (one failing 30 % of requests) in Compose | [`deploy/local/`](deploy/local/README.md); notebook 01 talks to it via `GWLAB_URL` |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | `vllm/vllm-openai:v0.30.0` serving Qwen2.5-0.5B-Instruct as `lab/llm` behind the gateway, a fake fallback | [`deploy/any-gpu/`](deploy/any-gpu/README.md); `GWLAB_VLLM_URL` turns on the T1 cells of notebooks 01–04 |
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

Talk to it like any OpenAI-compatible server — the key is a virtual key, the model is an alias:

```bash
curl -sN localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"chat","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8080/metrics | grep ^gwlab_
curl -s localhost:8080/debug/state -H 'Authorization: Bearer dev-admin-token'     # breakers, buckets, recent decisions
```

## The library (`gwlab/`, ~5,400 lines)

| Module | Lines | The idea |
|---|---:|---|
| `gateway/server.py` | ~730 | the §1 pipeline in aiohttp: authenticate (tenant from the key, never a header) → input guardrail → cache → admit (RPM + TPM) → route → call with the provider's key → relay (parse every chunk, translate, inject and strip usage, meter, output guardrail, fall back only before the first byte) → account (reconcile, ledger row, spans, metrics); `/v1/models`, `/metrics`, `/admin/keys`, `/debug/state`, `/mcp/{server}` |
| `gateway/adapters.py` | ~340 | provider adapters as data: the OpenAI, Anthropic and Gemini rows (paths, auth headers, usage fields, finish reasons); request translation to Anthropic Messages; Anthropic stream events → chat chunks; errors normalised to string codes |
| `gateway/config.py` | ~260 | the YAML config: providers, the model catalogue with prices, aliases (ordered chains and policies), tenants, limits, cache, guardrails, breakers; `${VAR:-default}` from the environment |
| `gateway/routing.py` | ~170 | fall-through classes, a consecutive-failure breaker per target (07.2's rule), capability/context/residency filters, ordered/cheapest/EWMA-TTFT/canary policies, chain availability and expected latency |
| `gateway/ratelimit.py` | ~190 | token buckets with debt; RPM and TPM per tenant, per key, and gateway-wide; `reserve` (reserve → stream → reconcile, a hard cap) vs `per_request`; `x-ratelimit-*` headers |
| `gateway/cache.py` | ~290 | exact and semantic caches in sqlite3 + numpy; the hashing embedder of `ragkit`; cacheability from the class the route declares (`metadata.cache_class`, allowlisted; a regex only vetoes); tenant/alias/system-prompt namespaces, plus the user for per-user classes; the entity guard; `cache_salt`; the labelled traffic sample and the threshold sweep |
| `gateway/metering.py` | ~160 | dated price rows (verify), cost from canonical usage, $/1M blended and self-hosted, ledger rows, totals, chargeback by tokens or GPU-seconds, reconciliation |
| `gateway/keys.py`, `store.py` | ~200 | virtual keys hashed at rest, scoped, budgeted, revocable; provider keys with rotation and overlap; the three sqlite tables |
| `gateway/guardrails.py` | ~110 | a regex screener (a labelled stand-in) and the latency of each placement: inline, parallel, held-back windows, full, shadow |
| `gateway/otel.py` | ~210 | spans with the GenAI names pinned (v1.41.0 ∩ genai main), OTLP/JSON lines and read-back, OTLP/HTTP export when installed |
| `fakes.py` | ~560 | fake providers over HTTP in two dialects with vLLM's quirks, their own RPM/TPM windows, a 16-token-block prefix cache with `cache_salt`, heavy-tailed outputs, tool calls, reasoning tokens, and switchable faults; vLLM-named `/metrics` |
| `mcp/` | ~830 | a fake authorization server and MCP server; the gateway's MCP client (discovery, CIMD, PKCE S256 with `resource`, per-principal tokens, step-up, rotation with reuse detection); DPoP proofs and nonces (ES256 with `cryptography`, else a labelled HMAC stand-in) |
| `t1.py` | ~160 | the T1 cells of notebooks 01–04 as functions: a gateway in front of a real vLLM, every result labelled `MEASURED` only when vLLM served it; stopping vLLM mid-run; the prefix-cache rule and salt asserted on measured rows; `/metrics` reconciliation that skips when vLLM is gone (tested offline against a stand-in) |
| `sse.py`, `tokens.py`, `promtext.py`, `client.py`, `bench.py`, `report.py`, `stack.py`, `env.py`, `__main__.py` | ~1,100 | SSE parsing and stream accumulation, the labelled token estimate, Prometheus text, a timing client, scripted tenants (open loop), tables, the in-process stack, tier detection (T1 only if vLLM answers `/health`), the CLI |

This lab never imports the core package (`gateway-core`, `gwcore`) next door; its tests pin its numbers to the
repo's canonical homes instead ([`tests/test_repo_numbers.py`](tests/test_repo_numbers.py)): `scalelab.capacity.cost_per_call`
($0.007005 for the scaling primer's §3.4 call), `roofline.cost.cost_per_million_tokens` (01 PRIMER §8.1's $0.150 and
$0.660), 07.2's `challenge_for` on RFC 7636 Appendix B and its GenAI span names, `ragkit`'s hashing embedder, and
`scalelab.resilience.TokenBucket`'s rule.

## Where it sits

Read [`../PRIMER.md`](../PRIMER.md) alongside; the minimal in-process version of every mechanism is
[`../gateway-core/`](../gateway-core/). Before this lab: the token bucket and admission control of
[06.3 `agentic-scaling-lab`](../../scaling-admission-cost/agentic-scaling-lab/), the token discipline and OAuth of
[06.6 identity](../../identity-security/agentic-identity-gcp-lab/docs/primer.md), and the 07.2 platform lab's MCP
gateway and breakers ([`gcp-agent-platform-lab`](../../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/)).
Below it: the replica router of layer 05 ([`inference-gateway-lab`](../../../05-orchestrator/serving-orchestration/inference-gateway-lab/)),
which this gateway uses as one upstream pool, and the engine of layer 04 ([`vllm-serving-lab`](../../../04-inference-engine/serving-engine/vllm-serving-lab/)),
whose prefix cache and `usage` it depends on.

## Caveats

- **Simulated vs measured.** The fake providers' latencies are sleeps and their token counts come from a word
  tokenizer; every response carries `x-gwlab-simulated: true` and every notebook labels such numbers. With
  `GWLAB_VLLM_URL` set and vLLM answering `/health`, the T1 cells measure a real vLLM, and label a row `MEASURED`
  only when vLLM served it (`gwlab/t1.py`; a fallback to a fake is labelled `SIMULATED`).
- **Compressed time.** Notebook 04 compresses the rate-limit "minute" to 2 s so a run takes seconds; the
  arithmetic is the same at 60 s.
- **Stand-ins, labelled.** The guardrail is a regex with a simulated check time, the semantic cache's embedder is
  lexical (a hashing embedder), the cache's second guard is a regex that can only veto a declared class (the class
  itself is declared by the route, never inferred), the fake AS auto-approves the `login_hint`
  principal (there is no browser), tokens are HS256 between the two fakes, and without `cryptography` DPoP proofs
  are HMAC-signed — not RFC 9449-conformant. Loopback HTTP `client_id` URLs are accepted by the fake AS in
  development only.
- **One process.** Keys, cache and ledger are sqlite3 and the buckets live in memory: two gateway replicas would
  need a shared database and Redis-backed buckets (scaling primer §5.1). Not built here: multi-region gateway
  failover, a distributed limiter under real concurrency, MCP client credentials and enterprise-managed
  authorization, a real classifier guardrail measured.
- **Checked by construction.** The deploy paths are checked with `bash -n`, `DRY_RUN=1` runs and config/compose
  consistency tests, not run on real infrastructure here.

## Verify list (facts dated September 2026 that move)

Read from upstream sources on 2026-09-26 (the topic's fact sheet names each file); re-check when you move a pin:

- vLLM **v0.30.0**: `stream_options` without `stream` is a 400; `cache_salt` is validated (at most 128 characters,
  no `@ / \` or NUL) and salts the first block only; `--enable-prompt-tokens-details` adds `cached_tokens`;
  `--enable-force-include-usage` also turns on continuous usage; a mid-stream failure is an `error` object in a
  `data:` event inside the HTTP 200, then `[DONE]`; `ErrorInfo.code` is the HTTP status as an int; `--api-key`
  guards only `/v1`, `/v2`, `/inference` and `/cohere` (`GUARDED_PREFIX`; `/health`, `/metrics` and the rest stay
  open); reasoning text is `reasoning`.
- OpenAI chat completions (openai-openapi 2.3.0): `include_usage` sends a final chunk with `choices: []` and
  `usage: null` on other chunks; `max_tokens` is deprecated for `max_completion_tokens`; `Retry-After` is in the
  spec, the `x-ratelimit-*` headers only in prose docs.
- Anthropic Messages (anthropic-sdk-python 1.8.0): `input_tokens` excludes cache reads and writes; stream event
  names; 529 `overloaded_error`. Gemini (python-genai 2.25.0): `thoughtsTokenCount` sits outside
  `candidatesTokenCount`.
- OpenTelemetry GenAI: names are "development"; `gen_ai.usage.cache_creation.input_tokens` and
  `gen_ai.client.token.usage` are the v1.41.0 names, renamed on the GenAI repo's main (`MAIN_NAMES`);
  `gen_ai.provider.name` has no value for a self-hosted server (this lab uses `vllm`, a choice).
- MCP authorization, revision **2026-07-28**: discovery order, CIMD as SHOULD and DCR deprecated, S256 required
  and a refusal when it is not advertised, `resource` in both requests, RFC 9207 `iss`; DPoP is RFC 9449, not
  part of the MCP spec. CIMD draft rules (5 KB, no special-use addresses) from the IETF editor's copy.
- Prices in `gateway/metering.py` (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gpt-5.4-mini`,
  `claude-haiku-4-5`), dated 2026-09-26; GPU prices in `configs/vllm.yaml` and the deploy READMEs.
- `vllm/vllm-openai:v0.30.0` and `Qwen/Qwen2.5-0.5B-Instruct` for the T1 path; Colab and Kaggle GPU availability.

MIT licensed.
