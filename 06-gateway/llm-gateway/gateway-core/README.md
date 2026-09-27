# gateway-core — build an LLM gateway's decisions in one sitting, with standard library and numpy

After this you can explain every decision a gateway makes for one request — which tenant it is, whether it may run,
which target serves it and what happens when that target fails, whether a cached answer is safe, how many tokens it
may reserve, what it cost and who pays — because you will have written the code that makes each one, in `gwcore`, a
gateway small enough to read in two sittings that runs in process on a virtual clock.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1 (one front door, one API), 20 minutes.
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 109 tests and 1 skip in about 16 s,
   including "the ledger prices a call exactly as `scalelab.capacity.cost_per_call` does, on every price row".
3. Open [`notebooks/01_one_front_door.ipynb`](notebooks/01_one_front_door.ipynb) and send a request through the
   front door while its primary provider is down.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network; every latency is simulated on
a virtual clock and labelled so.* Each notebook opens with "The one-minute version", works examples against the code,
sets exercises with a check cell that prints ✅, and ends with "In a design review". Finished versions are in
[`solutions/`](solutions/). About 16.5 hours with the primer and the lab (the repo's curriculum, module 06.7); the core's notebooks take about 8.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_one_front_door`](notebooks/01_one_front_door.ipynb) | normalise four providers' usage with an adapter table, cache writes included; see which usage a stream carries and when a cut stream has none; accumulate a stream with parallel tool calls; relay so every stream is metered; bill a cut stream; price the hop in availability; read a request back from its GenAI spans | §1, §5.6, §6.1 | ~1.5 h | T0 |
| [`02_routing_and_fallback_chains`](notebooks/02_routing_and_fallback_chains.ipynb) | filter and order a chain; say what falls through (408 and 529 included); compute a chain's availability and expected latency and cost; predict how many overlapping requests reach a dead target before its breaker opens; apply the first-byte rule; pick a first-byte deadline from a budget and show why a whole-response timeout cannot be it | §2 | ~1.5 h | T0 |
| [`03_exact_and_semantic_caching`](notebooks/03_exact_and_semantic_caching.ipynb) | build a safe exact key; extract what an entity guard compares; choose a threshold under a false-hit budget on labelled traffic; price a false hit; decide what a namespace must hold for shared and per-user classes; compare with the provider's prompt cache | §3, §6.4 | ~1.5 h | T0 |
| [`04_token_limits_metering_and_chargeback`](notebooks/04_token_limits_metering_and_chargeback.ipynb) | predict how far a per-request bucket over-admits; build reserve → stream → reconcile; size a reservation by simulation; bill a thinking call from raw usage; charge back a pool by GPU-seconds; find ledger drift and predict it from the cut streams | §4, §5 | ~2 h | T0 |
| [`05_guardrails_keys_and_mcp_authorization`](notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) | derive a tenant's `cache_salt`; place a held-back window under a TTFT budget; set a false-positive budget; build the MCP discovery URLs; compute PKCE; detect refresh-token reuse | §6, §7, §8 (§9 is a reading) | ~1.5 h | T0 |

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

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`gwcore/api.py`](gwcore/api.py) | ~125 | chat completions and SSE as the lingua franca; a stream accumulator that concatenates content, reassembles tool calls by `index`, keeps the usage chunk and catches errors inside a 200; a labelled token estimate |
| [`gwcore/providers.py`](gwcore/providers.py) | ~260 | adapters as data (four dialects' usage fields, cache writes included, and finish reasons — an unknown one is never guessed to be `stop`), stream normalisation for Anthropic and Gemini events (a cut stream yields no usage), the dated model catalogue with prices, and a fake provider on a virtual clock: TTFT, ITL, `max_completion_tokens` bounding reasoning and visible tokens together, its own RPM/TPM with `Retry-After`, outages, a failure mid-stream |
| [`gwcore/routing.py`](gwcore/routing.py) | ~145 | aliases → chains; capability, context and region filters; policies; which failures fall through (408, 429, 5xx, 529, a timeout, context length); a consecutive-failure breaker per target (07.2's rule) whose probe is handed back when it says nothing about health; chain availability and expected latency and cost |
| [`gwcore/cache.py`](gwcore/cache.py) | ~120 | what a route may cache; a tenant-namespaced exact key; the 07.4 hashing embedder; a semantic cache with an entity guard; the threshold sweep over the bundled labelled traffic |
| [`gwcore/ratelimit.py`](gwcore/ratelimit.py) | ~210 | `scalelab`'s token-bucket rule on an explicit clock; a sliding window meter; reserve → stream → reconcile, with refunds that shrink the request's own debits; hierarchical all-or-nothing admission; the over-admission closed form; the §4 simulation |
| [`gwcore/metering.py`](gwcore/metering.py) | ~110 | usage priced into ledger rows (cache reads, cache writes and reasoning output each at their own price), blended and self-hosted $ per 1M tokens, reconciliation against the provider's counts, chargeback by tokens or GPU-seconds |
| [`gwcore/otel.py`](gwcore/otel.py) | ~95 | one server span and one client span per target, GenAI attribute names pinned to v1.41.0 with the map to the moving names, OTLP/JSON lines out and back |
| [`gwcore/keys.py`](gwcore/keys.py) | ~140 | virtual keys (hashed, scoped, budgeted, revocable), provider keys rotated with an overlap, the per-tenant `cache_salt`, SVID rotation at half-life ± 10 % re-drawn on every check (as SPIRE does) and a fake Workload API stream |
| [`gwcore/guardrails.py`](gwcore/guardrails.py) | ~80 | hooks and placements; the TTFT and end-to-end latency each placement adds; exposure; dollars per 1,000 checks; false blocks compounding over a conversation; a labelled regex stand-in |
| [`gwcore/mcp_authz.py`](gwcore/mcp_authz.py) | ~335 | the MCP client flow against a fake authorization server and MCP server: discovery URLs, CIMD validation, PKCE S256, `resource` (a token for one server is refused at another), `iss`, refresh rotation with reuse detection, step-up, DPoP proofs checked for `htm`, `htu`, `iat` and `jti` replay, and nonces, through a pluggable (HMAC stand-in) signer |
| [`gwcore/gateway.py`](gwcore/gateway.py) | ~185 | the §1 pipeline that wires them together: key → screen → cache → reserve → chain → relay and meter → reconcile → record; only a provider's health trips its breaker, a provider's 401 is our 502, residency comes from the key |

Data: [`gwcore/data/payloads.json`](gwcore/data/payloads.json) (one call in four dialects, sample output in the
documented format, illustrative) and [`gwcore/data/traffic.json`](gwcore/data/traffic.json) (the labelled cache
sample: 15 cached questions, 23 paraphrases, 15 near misses, 6 unrelated and 5 uncacheable probes).

## What the tests prove

`tests/` has one focused test per concept (102, plus 8 notebook-tooling checks, one of which skips for percent-source labs; offline, ~16 s). The ones that carry the claims:

- **Existing repo numbers, reproduced** (`test_repo_numbers.py`, loading the other labs by path and skipping when
  absent): `metering.price_call` equals `scalelab.capacity.cost_per_call` on every price row and four call shapes
  ($0.007005 for the scaling primer's call); `ratelimit.TokenBucket` waits exactly as `scalelab.resilience.TokenBucket`
  on 200 arrivals, some larger than the burst; the self-hosted rows are `roofline.cost.cost_per_million_tokens` on
  `roofline.llm.decode()`'s throughput ($0.150, $0.446, $0.660 per 1M); `mcp_authz.s256` equals 07.2's
  `agentlab.auth.oauth.challenge_for` and the GenAI names equal 07.2's `tracing.py` constants (both read from source
  with `ast`, since `agentlab` needs pydantic); the embedder equals `ragkit.embed.HashingEmbedder` vector for vector.
- **Every number the primer quotes** is recomputed and must appear verbatim (`test_primer_numbers.py`).
- **The reserve invariant**: `used in window + reserved` never exceeds the limit on random traffic, so a provider
  counting the same tokens never sees more (`test_ratelimit.py`).
- **The first-byte rule**: an error before the first chunk falls through; after it, the client gets an error chunk,
  no second model is called, and the ledger bills the relayed deltas as an estimate (`test_gateway_otel.py`).
- **MCP**: RFC 7636 Appendix B; discovery order; refusal without S256; the 401's scope first, then step-up to the
  union; rotation on refresh; a replayed refresh token revokes the grant and the live access token; tokens per
  principal; a token for one MCP server refused at another, and a code exchanged for another resource refused;
  DPoP nonces from the authorization server (400) and the resource server (401), and proofs refused for another
  method or URL, a stale `iat` or a replayed `jti` (`test_mcp_authz.py`).
- **The breaker cannot wedge**: a half-open probe answered with a 400 is handed back, and the healed provider is
  probed and served again (`test_gateway_otel.py`); a refund never lets the window read below what was processed
  (`test_ratelimit.py`).
- **Formulas pinned to hand-computed values**: chain availability (1 − 0.005 × 0.01), expected chain latency term by
  term, the token bucket's waits and debt, the over-admission ratio 1.99, held-back window latency, false-block
  rates, the SVID rotation window (1,620–1,980 s), the 43-character `cache_salt`, chargeback splits.

## Caveats: what is faithful, and what is simplified

Faithful to the sources read on 2026-09-26 (the primer's Sources and Verify list): the chat-completions and usage field
names, vLLM v0.30.0's `stream_options` rule, error shape, `cache_salt` validation and priority direction; the OTel
GenAI names and OTLP/JSON encoding; the MCP 2026-07-28 client rules; RFC 7636, 9207, 9449 nonce responses; SPIRE's
rotation rule. Simplified: providers, the authorization server, the MCP server and the Workload API are in-process
fakes; tokens are estimated at four characters each where no usage exists; the gateway handles one request at a time
on a virtual clock (concurrency appears in `ratelimit.simulate`, a 1-second grid); the DPoP signer is an HMAC
**stand-in** that RFC 9449 forbids (the lab signs with real keys when `cryptography` is installed); the regex screener
stands in for a classifier; prices and context windows are dated `(verify)`.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs this
package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../gateway-lab/`](../gateway-lab/) for the same gateway over real HTTP (aiohttp, sqlite3 keys, cache and
ledger), fake providers with their own limits and outages, a compose stack, and one real vLLM as a provider on a free
T4; its GCP deploy points at the 04 lab's Cloud Run or GKE vLLM and the 05 lab's GKE Inference Gateway. Where each
tier runs and what it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
