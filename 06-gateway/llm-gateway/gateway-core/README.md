# gateway-core — build an LLM gateway's decisions in one sitting, with standard library and numpy

After this, you can explain every decision that a gateway makes for one request:

- Which tenant the request comes from.
- If it can run.
- Which target serves it, and what occurs when that target fails.
- If a cached answer is safe.
- How many tokens it can reserve.
- What it cost, and who pays.

You can explain them because you wrote the code that makes each decision. The code is in `gwcore`, a gateway that runs
in process on a virtual clock. It is sufficiently small that you can read all of it in two sessions.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1 (one front door, one API). It takes 20 minutes.
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. It runs 109 tests and 1 skip in
   approximately 16 s. The tests include "the ledger prices a call exactly as `scalelab.capacity.cost_per_call` does,
   on every price row".
3. Open [`notebooks/01_one_front_door.ipynb`](notebooks/01_one_front_door.ipynb). Send a request through the front
   door while its primary provider is down.

## What you get

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network. The code simulates
every latency on a virtual clock, and labels it as simulated.*

Each notebook starts with "The one-minute version". Then it gives worked examples against the code. It sets
exercises, each with a check cell that prints ✅. It ends with "In a design review". The finished versions are in
[`solutions/`](solutions/).

The topic takes approximately 16.5 hours with the primer and the lab (module 06.7 of the
curriculum of the repository). The notebooks of the core take approximately 8 hours.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_one_front_door`](notebooks/01_one_front_door.ipynb) | Normalise the usage of four providers with an adapter table, with cache writes included. See which usage a stream carries, and when a cut stream has none. Accumulate a stream with parallel tool calls. Relay so that the gateway meters every stream. Bill a cut stream. Give the cost of the hop in availability. Read a request back from its GenAI spans. | §1, §5.6, §6.1 | ~1.5 h | T0 |
| [`02_routing_and_fallback_chains`](notebooks/02_routing_and_fallback_chains.ipynb) | Filter and order a chain. Say which failures go to the next target (408 and 529 included). Calculate the availability, the expected latency and the expected cost of a chain. Predict how many concurrent requests get to a dead target before its breaker opens. Apply the first-byte rule. Select a first-byte deadline from a budget. Show why a timeout for the whole response cannot be that deadline. | §2 | ~1.5 h | T0 |
| [`03_exact_and_semantic_caching`](notebooks/03_exact_and_semantic_caching.ipynb) | Build a safe exact key. Extract what an entity guard compares. Select a threshold under a false-hit budget on labelled traffic. Give the cost of a false hit. Decide what a namespace must hold for shared classes and for per-user classes. Compare with the prompt cache of the provider. | §3, §6.4 | ~1.5 h | T0 |
| [`04_token_limits_metering_and_chargeback`](notebooks/04_token_limits_metering_and_chargeback.ipynb) | Predict how much a per-request bucket over-admits. Build the sequence reserve, then stream, then reconcile. Calculate the size of a reservation by simulation. Bill a thinking call from raw usage. Charge back a pool by GPU-seconds. Find ledger drift, and predict it from the cut streams. | §4, §5 | ~2 h | T0 |
| [`05_guardrails_keys_and_mcp_authorization`](notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) | Calculate the `cache_salt` of a tenant. Put a held-back window under a TTFT budget. Set a false-positive budget. Build the MCP discovery URLs. Calculate PKCE. Find refresh-token reuse. | §6, §7, §8 (you only read §9) | ~1.5 h | T0 |

## Run it

```bash
cd 06-gateway/llm-gateway/gateway-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 109 tests (+1 skipped), ~16 s
python3 -m jupyterlab notebooks                # do the exercises
```

The library needs only numpy:

```python
from gwcore import Clock, FakeProvider, Gateway, Router, Target, api, keys

clock = Clock()
providers = {"google": FakeProvider("google", clock, outages=[(0, 60, 503)]), "openai": FakeProvider("openai", clock)}
ks = keys.KeyStore(seed=0)
key = ks.issue("acme", tpm=100_000)
gw = Gateway(keys=ks, router=Router({"chat": [Target("google", "gemini-3.5-flash"), Target("openai", "gpt-5.4-mini")]}),
             providers=providers, clock=clock)
r = gw.handle(key, api.chat_request("chat", [{"role": "user", "content": "hello"}], stream=True))
print([(t.provider, outcome) for t, outcome in r.attempts])   # [('google', 503), ('openai', 'ok')]
print(r.row.tenant, r.row.model, r.row.completion_tokens, f"${r.row.cost:.6f}")   # acme gpt-5.4-mini 40 $0.000181
```

## The whole library

Read the modules in this order. Each module starts with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`gwcore/api.py`](gwcore/api.py) | ~125 | Chat completions and SSE as the common language. A stream accumulator that concatenates content, puts tool calls together again by `index`, keeps the usage chunk and catches errors inside a 200. A labelled token estimate. |
| [`gwcore/providers.py`](gwcore/providers.py) | ~260 | Adapters as data: the usage fields of four dialects, with cache writes included, and the finish reasons. An adapter never guesses that an unknown finish reason is `stop`. Stream normalisation for Anthropic and Gemini events. A cut stream gives no usage. The dated model catalogue with prices. A fake provider on a virtual clock, with TTFT and ITL. Its `max_completion_tokens` is one limit for the reasoning tokens and the visible tokens together. It also has its own RPM/TPM with `Retry-After`, outages and a failure mid-stream. |
| [`gwcore/routing.py`](gwcore/routing.py) | ~145 | Aliases that resolve to chains. Capability, context and region filters. Policies. Which failures go to the next target (408, 429, 5xx, 529, a timeout, context length). A consecutive-failure breaker for each target (the rule of 07.2). When its probe says nothing about health, the breaker gives the probe back. Chain availability, and the expected latency and cost of a chain. |
| [`gwcore/cache.py`](gwcore/cache.py) | ~120 | What a route can cache. A tenant-namespaced exact key. The 07.4 hashing embedder. A semantic cache with an entity guard. The threshold sweep over the bundled labelled traffic. |
| [`gwcore/ratelimit.py`](gwcore/ratelimit.py) | ~210 | The token-bucket rule of `scalelab` on an explicit clock. A sliding window meter. The sequence reserve, then stream, then reconcile, with refunds that decrease only the debits of the same request. Hierarchical all-or-nothing admission. The closed form of the over-admission. The §4 simulation. |
| [`gwcore/metering.py`](gwcore/metering.py) | ~110 | Usage priced into ledger rows. Cache reads, cache writes and reasoning output each have their own price. Blended and self-hosted $ per 1M tokens. Reconciliation against the counts of the provider. Chargeback by tokens or by GPU-seconds. |
| [`gwcore/otel.py`](gwcore/otel.py) | ~95 | One server span, and one client span for each target. GenAI attribute names pinned to v1.41.0, with the map to the names that continue to change. OTLP/JSON lines, written out and read back. |
| [`gwcore/keys.py`](gwcore/keys.py) | ~140 | Virtual keys (hashed, scoped, budgeted, revocable). Provider keys rotated with an overlap. The per-tenant `cache_salt`. SVID rotation at half-life ± 10 %, with a new random value on every check (as SPIRE does). A fake Workload API stream. |
| [`gwcore/guardrails.py`](gwcore/guardrails.py) | ~80 | Hooks and placements. The TTFT and the end-to-end latency that each placement adds. Exposure. Dollars per 1,000 checks. False blocks that compound over a conversation. A labelled regex stand-in. |
| [`gwcore/mcp_authz.py`](gwcore/mcp_authz.py) | ~335 | The MCP client flow against a fake authorization server and MCP server. It covers discovery URLs, CIMD validation, PKCE S256 and `resource` (another server refuses a token for one server). It also covers `iss`, refresh rotation with reuse detection and step-up. It also covers DPoP proofs, examined for `htm`, `htu`, `iat` and `jti` replay, and nonces. All of it goes through a pluggable signer (an HMAC stand-in). |
| [`gwcore/gateway.py`](gwcore/gateway.py) | ~185 | The §1 pipeline that connects them: key, then screen, then cache, then reserve, then chain, then relay and meter, then reconcile, then record. Only the health of a provider trips its breaker. A 401 from a provider is our 502. Residency comes from the key. |

The data files are [`gwcore/data/payloads.json`](gwcore/data/payloads.json) and
[`gwcore/data/traffic.json`](gwcore/data/traffic.json). The first has one call in four dialects, as sample output in
the documented format (illustrative). The second is the labelled cache sample: 15 cached questions, 23 paraphrases,
15 near misses, 6 unrelated and 5 uncacheable probes.

## What the tests prove

`tests/` has one focused test for each concept. There are 102 of them, plus 8 checks on the notebook tools. One of these
checks skips for percent-source labs. The tests run offline in ~16 s. The tests that carry the claims are these:

- **Numbers from other labs of the repo, reproduced** (`test_repo_numbers.py`). The test loads the other labs by path, and skips when
  a lab is absent:
  - `metering.price_call` equals `scalelab.capacity.cost_per_call` on every price row and four call shapes. For the
    call of the scaling primer, the result is $0.007005.
  - `ratelimit.TokenBucket` waits exactly as `scalelab.resilience.TokenBucket` does on 200 arrivals. Some arrivals are
    larger than the burst.
  - The self-hosted rows are `roofline.cost.cost_per_million_tokens` on the throughput of `roofline.llm.decode()`
    ($0.150, $0.446, $0.660 per 1M).
  - `mcp_authz.s256` equals `agentlab.auth.oauth.challenge_for` of 07.2, and the GenAI names equal the `tracing.py`
    constants of 07.2. The test reads both from source with `ast`, because `agentlab` needs pydantic.
  - The embedder equals `ragkit.embed.HashingEmbedder` vector for vector.
- **Every number the primer quotes**: the test calculates it again, and the number must appear verbatim in the primer
  (`test_primer_numbers.py`).
- **The reserve invariant**: `used in window + reserved` never exceeds the limit on random traffic. Thus a provider
  that counts the same tokens never sees more (`test_ratelimit.py`).
- **The first-byte rule**: an error before the first chunk goes to the next target. After the first chunk, the client
  gets an error chunk, and the gateway calls no second model. The ledger bills the relayed deltas as an estimate
  (`test_gateway_otel.py`).
- **MCP**: the tests cover these items (`test_mcp_authz.py`):
  - RFC 7636 Appendix B.
  - The discovery order.
  - A refusal without S256.
  - The scope of the 401 first, then step-up to the union.
  - Rotation on refresh.
  - A replayed refresh token revokes the grant and the live access token.
  - Tokens for each principal.
  - Another MCP server refuses a token for one MCP server. A server also refuses a code exchanged for another
    resource.
  - DPoP nonces from the authorization server (400) and the resource server (401). The servers refuse proofs for
    another method or URL, a stale `iat` or a replayed `jti`.
- **The breaker cannot become stuck**: when a 400 answers a half-open probe, the breaker gives the probe back. When the
  provider is healthy again, the gateway sends it a probe and serves from it again (`test_gateway_otel.py`). A refund
  never lets the window read below the quantity already processed (`test_ratelimit.py`).
- **Formulas pinned to hand-computed values**:
  - chain availability (1 − 0.005 × 0.01),
  - the expected chain latency, term by term,
  - the waits and the debt of the token bucket,
  - the over-admission ratio 1.99,
  - held-back window latency,
  - false-block rates,
  - the SVID rotation window (1,620–1,980 s),
  - the 43-character `cache_salt`,
  - the chargeback splits.

## Caveats: what is faithful, and what is simplified

These parts agree with the sources as read on 2026-09-26 (the Sources and Verify list of the primer):

- The chat-completions and usage field names.
- The `stream_options` rule, the error shape, the `cache_salt` validation and the priority direction of vLLM v0.30.0.
- The OTel GenAI names and the OTLP/JSON encoding.
- The MCP 2026-07-28 client rules.
- RFC 7636, 9207, 9449 nonce responses.
- The rotation rule of SPIRE.

The code simplifies these parts:

- The providers, the authorization server, the MCP server and the Workload API are in-process fakes.
- Where no usage exists, the code estimates tokens at four characters each.
- The gateway handles one request at a time on a virtual clock. Concurrency appears in `ratelimit.simulate`, on a
  1-second grid.
- The DPoP signer is an HMAC **stand-in** that RFC 9449 forbids. When you install `cryptography`, the lab signs with
  real keys.
- The regex screener is a stand-in for a classifier.
- Prices and context windows have a date and the `(verify)` tag.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three and the tests. On Colab, the first cell of each notebook clones the repository and
installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../gateway-lab/`](../gateway-lab/) for the same gateway over real HTTP (aiohttp, sqlite3 keys, cache and
ledger). The lab also has fake providers with their own limits and outages, and a compose stack. It also has one real
vLLM as a provider on a T4 at no cost. Its GCP deploy points to the Cloud Run or GKE vLLM of the 04 lab, and to the GKE
Inference Gateway of the 05 lab. [`COMPUTE.md`](../../../COMPUTE.md) tells where each tier runs and what it costs. The
`gateway-core` package is MIT licensed.
