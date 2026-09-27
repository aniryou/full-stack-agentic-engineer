# llm-gateway — one front door for many models: routing, fallbacks, caching, metering and tenant isolation

After this topic you can design the service every app and agent calls instead of calling model providers, and defend
each decision in it with a number: which failures fall back and when they must not, what may be cached and at what
false-hit rate, how token limits survive a thinking-model rollout, what a request cost and who pays, where the tenant
comes from, what a guardrail costs in latency, and how the gateway authorizes itself to MCP servers.

## Start here

1. Read [PRIMER.md](PRIMER.md) §1–§2 (45 min): one front door, one API; routing and fallback chains.
2. `cd gateway-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 109 tests in about 16 s,
   free, on any laptop; then open [`gateway-core/notebooks/01_one_front_door.ipynb`](gateway-core/notebooks/01_one_front_door.ipynb)
   and watch a request fall through a provider outage before its first byte.
3. When you want real HTTP (still free), or one real vLLM on a free Colab or Kaggle T4, continue in
   [`gateway-lab/`](gateway-lab/).

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T0 + Docker = the same laptop with Docker, free; T1 = one small GPU (a free
Colab or Kaggle T4, or a rented 24 GB card); T3 = the Google Cloud deployment, optional.* The topic is the repo's
module 06.7, about 16.5 hours with every exercise; the T1 paths add about 2 GPU-hours.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [PRIMER.md](PRIMER.md) | explain the gateway's decisions and their costs, §1–§9, with a design-review walkthrough and six drills | ~2.5 h | reading |
| [`gateway-core/`](gateway-core/) (package `gwcore`, standard library + numpy) | build each decision in process on a virtual clock: adapters, chains and breakers, exact and semantic caches, reserve → stream → reconcile, the ledger and chargeback, virtual keys and `cache_salt`, guardrail placement, the MCP client flow, GenAI spans | ~8 h, 5 notebooks | T0 |
| [`gateway-lab/`](gateway-lab/) (package `gwlab`) | run the same gateway over HTTP in front of fake providers or a real vLLM: what a client sees for each fault, a semantic cache on declared classes checked live against its sweep, the prefix-cache timing attack `cache_salt` closes, a ledger reconciled with vLLM's `usage` and `/metrics`, key rotation under load, MCP authorization over HTTP | ~6 h, 5 notebooks | T0, T0 + Docker, T1 |

The modules, one per pair of notebooks:

| Module | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---:|---|
| 06.7.1 One front door | §1 | [`01_one_front_door`](gateway-core/notebooks/01_one_front_door.ipynb) | [`01_a_gateway_over_http`](gateway-lab/notebooks/01_a_gateway_over_http.ipynb) | 3 | T0 (T1 with one vLLM) |
| 06.7.2 Routing and fallback chains | §2 | [`02_routing_and_fallback_chains`](gateway-core/notebooks/02_routing_and_fallback_chains.ipynb) | [`02_outages_fallbacks_and_breakers`](gateway-lab/notebooks/02_outages_fallbacks_and_breakers.ipynb) | 3 | T0 → T1 |
| 06.7.3 Caching at the gateway | §3 | [`03_exact_and_semantic_caching`](gateway-core/notebooks/03_exact_and_semantic_caching.ipynb) | [`03_semantic_cache_vs_the_prefix_cache`](gateway-lab/notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) | 3 | T0 → T1 |
| 06.7.4 Tokens: limits, metering and chargeback | §4, §5 | [`04_token_limits_metering_and_chargeback`](gateway-core/notebooks/04_token_limits_metering_and_chargeback.ipynb) | [`04_streaming_limits_metering_and_chargeback`](gateway-lab/notebooks/04_streaming_limits_metering_and_chargeback.ipynb) | 4 | T0 → T1 |
| 06.7.5 Keys, guardrails and MCP authorization | §6–§9 | [`05_guardrails_keys_and_mcp_authorization`](gateway-core/notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) | [`05_guardrails_and_mcp_authorization_over_http`](gateway-lab/notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) | 3.5 | T0 |

## Run it

```bash
cd 06-gateway/llm-gateway/gateway-core
python3 -m pip install -r requirements.txt          # numpy + pytest + Jupyter
python3 -m pytest -q                                 # 109 tests (+1 skipped), ~16 s
python3 tools/run_notebooks.py solutions             # the five finished notebooks, ~10 s
python3 -m jupyterlab notebooks                      # do the exercises
```

The lab's install, tests and deploy targets ([`deploy/local`](gateway-lab/deploy/local/) compose,
[`deploy/any-gpu`](gateway-lab/deploy/any-gpu/) with vLLM, [`deploy/gcp`](gateway-lab/deploy/gcp/) — a README pointing at
existing deploys, no new Terraform) are in [`gateway-lab/`](gateway-lab/).

## How it fits

Read first: the [scaling primer](../scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) (06.1–06.3:
cost per conversation, the token bucket, retries and breakers, admission) and the
[identity primer](../identity-security/agentic-identity-gcp-lab/docs/primer.md) (06.6: agent identity, delegation,
secrets, screening, the MCP server as a resource server); this topic applies them at the one service that holds every
key. Below it, the [05 orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md) picks the replica
inside a self-hosted pool and the [serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) §5 is
the prefix cache the gateway isolates per tenant with `cache_salt`. Beside it, the 07.2
[agent platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) is the agent
side of the same calls: its MCP egress gateway, OAuth flow, GenAI tracing and breakers. The
[RL and thinking-models primer](../../00-foundations/rl-and-thinking-models/PRIMER.md) §5 and §7 explain the output
tails that break per-request limits. The learning path is in [`CURRICULUM.md`](../../CURRICULUM.md).

## Caveats

- Everything in `gateway-core` is simulated on a virtual clock and labelled so; providers, the authorization server,
  the MCP server and the Workload API are in-process fakes, and the provider payloads are sample output in the
  documented format (illustrative). The lab measures the same things over real HTTP, and on a real vLLM at T1.
- The semantic-cache numbers come from a lexical hashing embedder on a small labelled sample: they show the method
  and the failure modes, not what a real embedder scores on your traffic.
- The core's DPoP signer is an HMAC stand-in that RFC 9449 forbids (the standard library has no asymmetric keys); it
  is labelled, and the lab signs with real keys when `cryptography` is installed.
- Prices, context windows, product features and spec revisions are dated 2026-09-26 and marked `(verify)` in the
  primer's Verify list; GPU prices and obtainability are in [`COMPUTE.md`](../../COMPUTE.md). The T1 paths are free on
  Colab or Kaggle, or about $0.3–0.7 an hour on a rented 24 GB GPU (verify); the T3 path reuses the 04 and 05 labs'
  GCP deploys and bills what they bill.
