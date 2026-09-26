# 06 · Gateway

Decide, outside the model, who may run what and how much of it: after this layer you can give each agent its own
identity, let it act for a user through an RFC 8693 token exchange without widening the user's grant, enforce
deny-by-default tool policy with an audit trail, turn conversations per day into tokens per minute and dollars, find
the provisioned-throughput (or own-GPU) break-even, and keep a service up under a 429 storm with rate limits,
breakers and admission control.

## Where this layer sits

```
   07 Agents and applications         the agent: loop, tools, sandboxes, state, durable execution, retrieval
   06 Gateway                         who may run what: identity, policy, rate limits, admission, cost
   05 Orchestrator                    many engine replicas as one service: routing, autoscaling, P/D split
   04 Inference engine                one model on its GPUs: the step loop, the KV cache, batching, kernels
   03 Kubernetes and GPU scheduling   GPUs made schedulable: device plugin, scheduler, gangs, quotas
   02 CUDA, NCCL and runtime          container to GPU: driver, CUDA, kernels, NCCL, GPU sharing, health
   01 Hardware and fabric             GPUs, memory, NVLink, NICs, storage: the roofline, the cost of a token
   00 Foundations                     the model itself, beneath the stack: shapes, capacity math, MoE, RL
```

This layer is the control plane in front of the models: it decides whether a request runs at all, before layer 05
decides where.

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU box,
rented for an hour; T3 = the Google Cloud deployment, optional.* Every lab in this layer runs at T0 with no key and
no GPU. Times are rough and come from the repo's curriculum ([`CURRICULUM.md`](../CURRICULUM.md), modules 06.1–06.6).

| Topic | You will be able to… | Time | Tier |
|---|---|---|---|
| [`identity-security/`](identity-security/README.md) | give an agent its own principal; separate its own authority from authority delegated by a user; enforce policy before every tool call; make an MCP server verify audience and scope itself; screen untrusted content; audit every decision with both identities — a one-file [core](identity-security/agentic-identity-core/README.md), the full [Google Cloud lab](identity-security/agentic-identity-gcp-lab/README.md) with its [primer](identity-security/agentic-identity-gcp-lab/docs/primer.md), and a [Mistral variant](identity-security/agentic-identity-core-mistral/README.md) of the core | ~12 h (core + lab); +1–2 h for the Mistral core | T0 (T3 optional) |
| [`scaling-admission-cost/`](scaling-admission-cost/README.md) | size an agent service from conversations per day (tokens per minute, turns in flight, cost per conversation); find the provisioned-throughput break-even; bound a turn and survive a crash mid-turn; smooth traffic, retry, break circuits and shed load with `Retry-After`; decide between a hosted API and your own vLLM fleet — the [GCP lab](scaling-admission-cost/agentic-scaling-lab/README.md) with its [primer](scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) and the [Mistral variant](scaling-admission-cost/agentic-scaling-lab-mistral/README.md) | ~6 h; +2 h for the Mistral variant | T0 |

## Start here

1. Read the [scaling primer](scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1–3 (why agents
   scale differently, the arithmetic).
2. `cd scaling-admission-cost/agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m scalelab.capacity`
   — a second, and it prints the capacity plan for 100,000 conversations a day; then open
   [`01_scaling_math`](scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb).
3. For identity, run `python3 agentsec_core.py` in
   [`identity-security/agentic-identity-core/`](identity-security/agentic-identity-core/README.md): the five moves
   end to end with the audit timeline. Then follow the topic [README](identity-security/README.md).

## Run it

Each lab has its own environment. The two scaling labs both install a package named `scalelab`: use a venv each.

```bash
cd identity-security/agentic-identity-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q   # 13 tests
cd ../agentic-identity-gcp-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q                      # 41 tests, ~6 s
cd ../../scaling-admission-cost/agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q  # 14 tests, ~4 s
```

Then open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`; the identity cores keep theirs beside
the code), or use the Colab links below. The scaling notebooks' checks print `not attempted yet` until you fill an exercise in; that
is by design, not a failure.

## How it fits

**Builds on** capacity planning in [`00-foundations`](../00-foundations/README.md) (tokens, TTFT and TPOT, Little's
law) — the one prerequisite for the scaling topic; the identity topic needs nothing from the layers below. The
Mistral scaling variant's hosted-vs-own-GPUs chapter is easier after layer 04's
[`serving-engine`](../04-inference-engine/serving-engine/README.md) (a replica's batch and step time). In the
[curriculum's spiral](../CURRICULUM.md#31-why-this-order) this layer comes after the orchestrator (05) and before
the agents (07); if you already build agents, the curriculum suggests skimming 06.1 first for motivation.
**Leads to** layer 07 ([`07-application-agent-framework`](../07-application-agent-framework/README.md)): the agent
loop whose turns this layer bounds, and [sandboxed execution](../07-application-agent-framework/sandboxed-execution/README.md),
which builds on the identity primer's §6.2. Layer 05 ([`05-orchestrator`](../05-orchestrator/README.md)) sits below:
admission here, routing and replica autoscaling there.

## Caveats

- The scaling labs' latencies and 429s come from simulations on a virtual clock (a fake model with a shared
  tokens-per-minute pool); prices, quotas and model ids are dated snapshots marked (verify) in each primer.
- The identity plane runs on local fakes at T0; the GCP lab's Terraform (T3, optional) binds the same code to
  Google Cloud's agent identity, credential broker and screening products, dated September 2026 (verify).
- A model API key (Gemini or Mistral) is optional everywhere: it swaps a scripted model for a real one.
- Not covered yet: the LLM gateway itself — model routing and fallback chains, semantic caching, token metering and
  chargeback; see [`CURRICULUM.md`](../CURRICULUM.md) §2 and §6.

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). One line per lab: each link opens that notebook in Colab, exercises first. *Answers* are the worked answer keys (in a `solutions/` or `worked/` folder, named `*_solution` or `*_solved`, or a `*_worked` notebook beside its `*_practice` twin when the folder has no `solutions/` of its own): try the exercise first. Any other `*_worked` notebook is a walkthrough lesson.

- **`identity-security/agentic-identity-core/`** — [core_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_practice.ipynb) · [core_walkthrough](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_walkthrough.ipynb) — *answers:* [core_solution](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_solution.ipynb)
- **`identity-security/agentic-identity-core-mistral/`** — [core_mistral_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_practice.ipynb) · [core_mistral_walkthrough](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_walkthrough.ipynb) — *answers:* [core_mistral_solution](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_solution.ipynb)
- **`identity-security/agentic-identity-gcp-lab/`** — [01_agent_identity_and_principals](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/01_agent_identity_and_principals.ipynb) · [02_delegation_and_token_exchange](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/02_delegation_and_token_exchange.ipynb) · [03_auth_manager_broker](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/03_auth_manager_broker.ipynb) · [04_policy_enforcement_point](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/04_policy_enforcement_point.ipynb) · [05_prompt_injection_and_guardrails](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/05_prompt_injection_and_guardrails.ipynb) · [06_mcp_resource_server](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/06_mcp_resource_server.ipynb) · [07_a2a_agent_cards](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/07_a2a_agent_cards.ipynb) · [08_audit_and_governance](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/08_audit_and_governance.ipynb) · [09_code_evaluation_drills](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/09_code_evaluation_drills.ipynb) · [practice/01_agent_identity_and_principals_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/01_agent_identity_and_principals_practice.ipynb) · [practice/02_delegation_and_token_exchange_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/02_delegation_and_token_exchange_practice.ipynb) · [practice/03_auth_manager_broker_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/03_auth_manager_broker_practice.ipynb) · [practice/04_policy_enforcement_point_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/04_policy_enforcement_point_practice.ipynb) · [practice/05_prompt_injection_and_guardrails_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/05_prompt_injection_and_guardrails_practice.ipynb) · [practice/06_mcp_resource_server_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/06_mcp_resource_server_practice.ipynb) · [practice/07_a2a_agent_cards_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/07_a2a_agent_cards_practice.ipynb) · [practice/08_audit_and_governance_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/08_audit_and_governance_practice.ipynb) · [practice/09_code_evaluation_drills_practice](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/09_code_evaluation_drills_practice.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/01_agent_identity_and_principals_solution.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/02_delegation_and_token_exchange_solution.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/03_auth_manager_broker_solution.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/04_policy_enforcement_point_solution.ipynb) · [05](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/05_prompt_injection_and_guardrails_solution.ipynb) · [06](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/06_mcp_resource_server_solution.ipynb) · [07](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/07_a2a_agent_cards_solution.ipynb) · [08](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/08_audit_and_governance_solution.ipynb) · [09](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/09_code_evaluation_drills_solution.ipynb)
- **`scaling-admission-cost/agentic-scaling-lab/`** — [01_scaling_math](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb) · [02_turn_loop_and_durability](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/02_turn_loop_and_durability.ipynb) · [03_rate_limits_and_admission](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/03_rate_limits_and_admission.ipynb) · [04_load_to_settings](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/04_load_to_settings.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/01_scaling_math_solutions.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/02_turn_loop_and_durability_solutions.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/03_rate_limits_and_admission_solutions.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/04_load_to_settings_solutions.ipynb)
- **`scaling-admission-cost/agentic-scaling-lab-mistral/`** — [01_scaling_math](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/01_scaling_math.ipynb) · [02_turn_loop_and_durability](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/02_turn_loop_and_durability.ipynb) · [03_rate_limits_and_admission](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/03_rate_limits_and_admission.ipynb) · [04_load_to_settings](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/04_load_to_settings.ipynb) — *answers:* [01](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/01_scaling_math_solutions.ipynb) · [02](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/02_turn_loop_and_durability_solutions.ipynb) · [03](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/03_rate_limits_and_admission_solutions.ipynb) · [04](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/04_load_to_settings_solutions.ipynb)
<!-- colab-links:end -->
