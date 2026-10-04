# 06 · Gateway

Decide, outside the model, who can run what, and how much of it. After this layer, you can do these things:

- Give each agent its own identity.
- Let an agent act for a user through an RFC 8693 token exchange. The exchange does not make the grant of the user
  wider.
- Enforce a deny-by-default tool policy with an audit trail.
- Change conversations per day into tokens per minute and dollars.
- Find the provisioned-throughput (or own-GPU) break-even.
- Keep a service available during a 429 storm with rate limits, breakers and admission control.
- Put one front door in front of every model. The front door routes requests across providers, and its fallback
  chain also crosses providers. The front door also caches what is safe, does token metering during streaming and
  keeps tenants apart.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, memory, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, model routing, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the control plane in front of the models. It decides if a request runs at all, before layer 05 decides
where it runs.

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T1 is one small GPU (a Colab/Kaggle T4 or a rented card). T2 is a
multi-GPU box that you rent for an hour. T3 is the Google Cloud deployment, and it is optional.* "T0 + Docker" is a
laptop with Docker, also at no cost.

Every lab in this layer runs at T0 with no key and no GPU. The times are
approximate. They come from the curriculum of the repo ([`CURRICULUM.md`](../CURRICULUM.md), modules 06.1–06.7).

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`identity-security/`](identity-security/README.md) | give an agent its own principal. Keep the authority of the agent separate from the authority that a user delegates to it. Enforce policy before each tool call. Make an MCP server verify the audience and the scope itself. Screen untrusted content. Record each decision in the audit trail with the two identities. The topic has a one-file [core](identity-security/agentic-identity-core/README.md), the full [Google Cloud lab](identity-security/agentic-identity-gcp-lab/README.md) with its [primer](identity-security/agentic-identity-gcp-lab/docs/primer.md), and the core on Mistral's platform. The core on Mistral's platform is beside the one-file core, as an optional provider path. | ~12 h (core + lab), +1 h for the Mistral path | T0 (T3 optional) |
| [`scaling-admission-cost/`](scaling-admission-cost/README.md) | calculate the size of an agent service from conversations per day (tokens per minute, turns in flight, cost per conversation). Find the provisioned-throughput break-even. Set a limit on a turn, and continue after a crash in the middle of a turn. Make traffic smooth, retry, open circuit breakers and shed load with `Retry-After`. Decide between a hosted API and your own vLLM fleet. The topic has the [lab](scaling-admission-cost/agentic-scaling-lab/README.md) with its [primer](scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) and its Mistral provider path: a [primer](scaling-admission-cost/agentic-scaling-lab/docs/mistral/01-scaling-primer.md), a vLLM fleet backend and notebook 05. | ~6 h, +2 h for the hosted-vs-own-GPUs path | T0 |
| [`llm-gateway/`](llm-gateway/README.md) | put one front door in front of many models, and support each decision in it with a number. Find which failures go to the fallback chain, and why a fallback never occurs after the first byte. Find what the gateway can cache, and at what false-hit rate. Set token limits that continue to work during a thinking-model rollout (reserve, then stream, then reconcile). Calculate the prices in a ledger from `usage`, and calculate a chargeback for a shared GPU pool. Get the tenant from the verified key, never from a header. Measure what a guardrail adds to TTFT. Use the gateway as the OAuth client of MCP servers. The topic has a [PRIMER](llm-gateway/PRIMER.md), [`gateway-core`](llm-gateway/gateway-core/README.md) (standard library + numpy, 5 notebooks) and [`gateway-lab`](llm-gateway/gateway-lab/README.md) (5 notebooks). The lab runs the same gateway over HTTP in front of fake providers or one vLLM. It also has a compose stack and MCP authorization over HTTP. | ~16.5 h: ~2.5 h primer, ~8 h core, ~6 h lab | T0, T0 + Docker, T1 (T3 reuses the 04 and 05 deploys) |

## Start here

1. Read the [scaling primer](scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1–3. These
   sections show why agents scale differently, and they give the arithmetic.
2. Run `cd scaling-admission-cost/agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m scalelab.capacity`.
   It takes a second, and it prints the capacity plan for 100,000 conversations a day. Then open
   [`01_scaling_math`](scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb).
3. For identity, run `python3 agentsec_core.py` in
   [`identity-security/agentic-identity-core/`](identity-security/agentic-identity-core/README.md). It runs the five
   moves end to end and shows the audit timeline. Then do the steps in the topic [README](identity-security/README.md).
4. For the gateway in front of the models, read the llm-gateway [PRIMER](llm-gateway/PRIMER.md) §1–§2. Then run the
   tests of [`gateway-core`](llm-gateway/gateway-core/README.md) and its first notebook. The topic
   [README](llm-gateway/README.md) gives the order.

## Run it

Each lab has its own environment (one venv for each lab).

```bash
cd identity-security/agentic-identity-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q   # 32 tests (3 skip without the Mistral client or a key)
cd ../agentic-identity-gcp-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q                      # 41 tests, ~6 s
cd ../../scaling-admission-cost/agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q  # 37 tests, ~3 s
cd ../../llm-gateway/gateway-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q           # 109 tests (+1 skipped), ~16 s
cd ../gateway-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q                                   # 165 tests + 2 skipped (166 + 1 with the [dpop] extra), ~25 s
python3 -m gwlab stack --port 8080                                                                               # the gateway, two fake providers and a demo key
```

Then open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`), or use the links in "Run in Colab". The
identity cores keep their notebooks beside the code. The checks of the scaling notebooks print `not attempted yet`
until you fill in an exercise. This is the intended result, not a failure.

## How it fits

**Builds on** capacity planning in [`00-foundations`](../00-foundations/README.md) (tokens, TTFT and TPOT, Little's
law). That is the one prerequisite for the scaling topic. The identity topic needs nothing from the layers below. The
scaling lab has a hosted-vs-own-GPUs path (its Mistral primer and notebook 05). That path is easier after layer 04's
[`serving-engine`](../04-inference-engine/serving-engine/README.md) (the batch and the step time of a replica).

The [`llm-gateway`](llm-gateway/README.md) topic comes last in the layer. It applies the token bucket and the
breakers of the scaling topic to tokens and providers. It also applies the token discipline of the identity topic to
provider keys and MCP servers. It also goes down to the prefix cache of layer 04 (the per-tenant `cache_salt` that it
sends) and to the router of layer 05. A self-hosted target is a pool, and the endpoint picker of that pool selects the
pod.

In the [curriculum's spiral](../CURRICULUM.md#31-why-this-order), this layer comes after the orchestrator (05) and
before the agents (07). If you already build agents, the curriculum suggests a fast read of 06.1 first, for
motivation.

**Leads to** layer 07 ([`07-application-agent-framework`](../07-application-agent-framework/README.md)). Layer 07
has the agent loop, and this layer sets the limits of its turns. Layer 07 also has
[sandboxed execution](../07-application-agent-framework/sandboxed-execution/README.md), which builds on §6.2 of the
identity primer. Layer 05 ([`05-orchestrator`](../05-orchestrator/README.md)) is below this layer. Admission and
model routing are in this layer, and replica routing and autoscaling are in layer 05.

## Caveats

- The latencies and 429s of the scaling lab come from simulations on a virtual clock. The simulations use a fake
  model with a shared tokens-per-minute pool, or a simulated vLLM fleet. Prices, quotas and model ids are snapshots
  with a date. Each primer marks them with the (verify) tag.
- The identity plane runs on local fakes at T0. The Terraform of the GCP lab (T3, optional) connects the same code to
  Google Cloud products. These are the agent identity, the credential broker and the products that screen content.
  These product facts are from September 2026 (verify).
- A model API key (Gemini or Mistral) is optional everywhere. It replaces a scripted model with a real one.
- The providers, the authorization server and the MCP server of the LLM gateway are fakes at T0. The lab simulates their
  latencies and token counts, and labels them. One real vLLM (T1) changes TTFT, `cached_tokens` and the reconciliation of
  the ledger into measurements. The semantic cache of the gateway uses a lexical hashing embedder. A labelled HMAC
  stand-in signs the DPoP proofs of the gateway, unless you install `cryptography` (the `dpop` extra of the lab).
- This layer does not cover these items yet (see [`CURRICULUM.md`](../CURRICULUM.md) §2):
  - A distributed limiter measured under real concurrency.
  - Multi-region gateway failover.
  - MCP client credentials and enterprise-managed authorization.
  - A real classifier guardrail measured in the request path.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab. Every lab keeps what you open (exercise blanks and lessons) in `notebooks/`, and the worked answer to a blank in `solutions/` under the same file name: those are the *answers*, so try the exercise first.

- **`identity-security/agentic-identity-core/`** — [core_mistral_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/notebooks/core_mistral_practice.ipynb) · [core_mistral_walkthrough](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/notebooks/core_mistral_walkthrough.ipynb) · [core_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/notebooks/core_practice.ipynb) · [core_walkthrough](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/notebooks/core_walkthrough.ipynb) — *answers:* [core_mistral_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/solutions/core_mistral_practice.ipynb) · [core_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/solutions/core_practice.ipynb)
- **`identity-security/agentic-identity-gcp-lab/`** — [01_agent_identity_and_principals](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/01_agent_identity_and_principals.ipynb) · [01_agent_identity_and_principals_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/01_agent_identity_and_principals_practice.ipynb) · [02_delegation_and_token_exchange](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/02_delegation_and_token_exchange.ipynb) · [02_delegation_and_token_exchange_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/02_delegation_and_token_exchange_practice.ipynb) · [03_auth_manager_broker](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/03_auth_manager_broker.ipynb) · [03_auth_manager_broker_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/03_auth_manager_broker_practice.ipynb) · [04_policy_enforcement_point](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/04_policy_enforcement_point.ipynb) · [04_policy_enforcement_point_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/04_policy_enforcement_point_practice.ipynb) · [05_prompt_injection_and_guardrails](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/05_prompt_injection_and_guardrails.ipynb) · [05_prompt_injection_and_guardrails_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/05_prompt_injection_and_guardrails_practice.ipynb) · [06_mcp_resource_server](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/06_mcp_resource_server.ipynb) · [06_mcp_resource_server_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/06_mcp_resource_server_practice.ipynb) · [07_a2a_agent_cards](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/07_a2a_agent_cards.ipynb) · [07_a2a_agent_cards_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/07_a2a_agent_cards_practice.ipynb) · [08_audit_and_governance](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/08_audit_and_governance.ipynb) · [08_audit_and_governance_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/08_audit_and_governance_practice.ipynb) · [09_code_evaluation_drills](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/09_code_evaluation_drills.ipynb) · [09_code_evaluation_drills_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/09_code_evaluation_drills_practice.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/01_agent_identity_and_principals_practice.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/02_delegation_and_token_exchange_practice.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/03_auth_manager_broker_practice.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/04_policy_enforcement_point_practice.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/05_prompt_injection_and_guardrails_practice.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/06_mcp_resource_server_practice.ipynb) · [07](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/07_a2a_agent_cards_practice.ipynb) · [08](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/08_audit_and_governance_practice.ipynb) · [09](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/solutions/09_code_evaluation_drills_practice.ipynb)
- **`llm-gateway/gateway-core/`** — [01_one_front_door](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/notebooks/01_one_front_door.ipynb) · [02_routing_and_fallback_chains](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/notebooks/02_routing_and_fallback_chains.ipynb) · [03_exact_and_semantic_caching](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/notebooks/03_exact_and_semantic_caching.ipynb) · [04_token_limits_metering_and_chargeback](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/notebooks/04_token_limits_metering_and_chargeback.ipynb) · [05_guardrails_keys_and_mcp_authorization](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/notebooks/05_guardrails_keys_and_mcp_authorization.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/solutions/01_one_front_door.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/solutions/02_routing_and_fallback_chains.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/solutions/03_exact_and_semantic_caching.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/solutions/04_token_limits_metering_and_chargeback.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-core/solutions/05_guardrails_keys_and_mcp_authorization.ipynb)
- **`llm-gateway/gateway-lab/`** — [01_a_gateway_over_http](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/notebooks/01_a_gateway_over_http.ipynb) · [02_outages_fallbacks_and_breakers](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/notebooks/02_outages_fallbacks_and_breakers.ipynb) · [03_semantic_cache_vs_the_prefix_cache](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/notebooks/03_semantic_cache_vs_the_prefix_cache.ipynb) · [04_streaming_limits_metering_and_chargeback](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/notebooks/04_streaming_limits_metering_and_chargeback.ipynb) · [05_guardrails_and_mcp_authorization_over_http](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/notebooks/05_guardrails_and_mcp_authorization_over_http.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/solutions/01_a_gateway_over_http.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/solutions/02_outages_fallbacks_and_breakers.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/solutions/03_semantic_cache_vs_the_prefix_cache.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/solutions/04_streaming_limits_metering_and_chargeback.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/llm-gateway/gateway-lab/solutions/05_guardrails_and_mcp_authorization_over_http.ipynb)
- **`scaling-admission-cost/agentic-scaling-lab/`** — [01_scaling_math](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb) · [02_turn_loop_and_durability](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/02_turn_loop_and_durability.ipynb) · [03_rate_limits_and_admission](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/03_rate_limits_and_admission.ipynb) · [04_load_to_settings](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/04_load_to_settings.ipynb) · [05_hosted_or_own_gpus](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/05_hosted_or_own_gpus.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/solutions/01_scaling_math.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/solutions/02_turn_loop_and_durability.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/solutions/03_rate_limits_and_admission.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/solutions/04_load_to_settings.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/solutions/05_hosted_or_own_gpus.ipynb)
<!-- colab-links:end -->
