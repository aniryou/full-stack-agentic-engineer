# llm-gateway — one front door for many models: routing, fallbacks, caching, metering and tenant isolation

After this topic, you can design the service that every app and agent calls in place of the model providers. You can
also defend each decision in it with a number:

- Which failures go to a fallback, and when a failure must not go to one.
- What the gateway can cache, and at what false-hit rate.
- How token limits continue to work during a thinking-model rollout.
- What a request cost, and who pays.
- Where the tenant comes from.
- What a guardrail costs in latency.
- How the gateway authorizes itself to MCP servers.

## Start here

1. Read [PRIMER.md](PRIMER.md) §1–§2 (45 min). They are about one front door and one API, and about routing and
   fallback chains.
2. Run `cd gateway-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 109 tests run in
   approximately 16 s, at no cost, on any laptop. Then open
   [`gateway-core/notebooks/01_one_front_door.ipynb`](gateway-core/notebooks/01_one_front_door.ipynb). Watch how a
   request goes to the next target during a provider outage, before its first byte.
3. When you want real HTTP, continue in [`gateway-lab/`](gateway-lab/). It is still at no cost. Also continue there
   for one real vLLM on a Colab or Kaggle T4 at no cost.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T0 + Docker is the same laptop with Docker, at no cost. T1 is one
small GPU (a Colab or Kaggle T4 at no cost, or a rented 24 GB card). T3 is the Google Cloud deployment, and it is
optional.*

The topic is module 06.7 of the repository. It takes approximately 16.5 hours with every exercise. The T1
paths add approximately 2 GPU-hours.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [PRIMER.md](PRIMER.md) | Explain the decisions of the gateway and their costs, §1–§9. The primer has a walkthrough for a design review and six drills. | ~2.5 h | reading |
| [`gateway-core/`](gateway-core/) (package `gwcore`, standard library + numpy) | Build each decision in process on a virtual clock. You build adapters, chains and breakers, exact and semantic caches, and the sequence reserve, then stream, then reconcile. You also build the ledger and chargeback, virtual keys and `cache_salt`, guardrail placement, the MCP client flow and GenAI spans. | ~8 h, 5 notebooks | T0 |
| [`gateway-lab/`](gateway-lab/) (package `gwlab`) | Run the same gateway over HTTP in front of fake providers or a real vLLM. See what a client sees for each fault. Put a semantic cache on declared classes, and compare it live with its sweep. See the prefix-cache timing attack that `cache_salt` closes. Reconcile a ledger with the `usage` and `/metrics` of vLLM. The lab also shows key rotation under load and MCP authorization over HTTP. | ~6 h, 5 notebooks | T0, T0 + Docker, T1 |

The modules, one for each pair of notebooks:

| Module | Primer | Core notebook | Lab notebook | Hours | Tier |
|---|---|---|---|---:|---|
| 06.7.1 One front door | §1 | [`01_one_front_door`](gateway-core/notebooks/01_one_front_door.ipynb) | [`01_a_gateway_over_http`](gateway-lab/notebooks/01_a_gateway_over_http.ipynb) | 3 | T0 (T1 with one vLLM) |
| 06.7.2 Routing and fallback chains | §2 | [`02_routing_and_fallback_chains`](gateway-core/notebooks/02_routing_and_fallback_chains.ipynb) | [`02_outages_fallbacks_and_breakers`](gateway-lab/notebooks/02_outages_fallbacks_and_breakers.ipynb) | 3 | T0 to T1 |
| 06.7.3 Caching at the gateway | §3 | [`03_exact_and_semantic_caching`](gateway-core/notebooks/03_exact_and_semantic_caching.ipynb) | [`03_semantic_cache_vs_the_prefix_cache`](gateway-lab/notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) | 3 | T0 to T1 |
| 06.7.4 Tokens: limits, metering and chargeback | §4, §5 | [`04_token_limits_metering_and_chargeback`](gateway-core/notebooks/04_token_limits_metering_and_chargeback.ipynb) | [`04_streaming_limits_metering_and_chargeback`](gateway-lab/notebooks/04_streaming_limits_metering_and_chargeback.ipynb) | 4 | T0 to T1 |
| 06.7.5 Keys, guardrails and MCP authorization | §6–§9 | [`05_guardrails_keys_and_mcp_authorization`](gateway-core/notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) | [`05_guardrails_and_mcp_authorization_over_http`](gateway-lab/notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) | 3.5 | T0 |

## Run it

```bash
cd 06-gateway/llm-gateway/gateway-core
python3 -m pip install -r requirements.txt          # numpy + pytest + Jupyter
python3 -m pytest -q                                 # 109 tests (+1 skipped), ~16 s
python3 tools/run_notebooks.py solutions             # the five finished notebooks, ~10 s
python3 -m jupyterlab notebooks                      # do the exercises
```

The install, the tests and the deploy targets of the lab are in [`gateway-lab/`](gateway-lab/). The deploy targets
are:

- [`deploy/local`](gateway-lab/deploy/local/), with compose.
- [`deploy/any-gpu`](gateway-lab/deploy/any-gpu/), with vLLM.
- [`deploy/gcp`](gateway-lab/deploy/gcp/), a README that points to deploys that exist already. It has no new Terraform.

## How it fits

Read two primers first. The [scaling primer](../scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
(06.1–06.3) covers the cost per conversation, the token bucket, retries and breakers, and admission. The
[identity primer](../identity-security/agentic-identity-gcp-lab/docs/primer.md) (06.6) covers agent identity,
delegation, secrets, screening, and the MCP server as a resource server. This topic applies them at the one service
that holds every key.

Below this topic, the [05 orchestration primer](../../05-orchestrator/serving-orchestration/PRIMER.md) explains how
the router selects the replica inside a self-hosted pool. The
[serving-engine primer](../../04-inference-engine/serving-engine/PRIMER.md) §5 explains the prefix cache. The gateway
isolates that prefix cache for each tenant with `cache_salt`. Beside this topic, the 07.2
[agent platform lab](../../07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/) is the agent
side of the same calls. It has its MCP egress gateway, its OAuth flow, GenAI tracing and breakers.

The [RL and thinking-models primer](../../00-foundations/rl-and-thinking-models/PRIMER.md) §5 and §7 explain the
output tails that break per-request limits. The learning path is in [`CURRICULUM.md`](../../CURRICULUM.md).

## Caveats

- `gateway-core` simulates everything on a virtual clock, and labels it as simulated. The providers, the authorization
  server, the MCP server and the Workload API are in-process fakes. The provider payloads are sample output in the
  documented format (illustrative). The lab measures the same things over real HTTP, and on a real vLLM at T1.
- The semantic-cache numbers come from a lexical hashing embedder on a small labelled sample. They show the method and
  the failure modes. They do not show what a real embedder scores on your traffic.
- The DPoP signer of the core is an HMAC stand-in that RFC 9449 forbids. The core uses it because the standard library
  has no asymmetric keys. The core labels it. When you install `cryptography`, the lab signs with real keys.
- Prices, context windows, product features and spec revisions have the date 2026-09-26. The Verify list of the primer
  marks them `(verify)`. [`COMPUTE.md`](../../COMPUTE.md) gives GPU prices and the availability of GPUs. The T1 paths
  cost nothing on Colab or Kaggle, or approximately $0.3–0.7 an hour on a rented 24 GB GPU (verify). The T3 path uses the GCP
  deploys of the 04 and 05 labs again, and it bills what they bill.
