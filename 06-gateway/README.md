# 06 · Gateway — auth, rate limits, routing, quotas, observability, cost

The control plane in front of the models: who may call, how much, where the call goes, and what
it cost. Policy and economics enforced in code, outside the model.

**Covers:** authn/authz & agent identity, delegation/token-exchange, policy enforcement,
rate limiting & admission control, request routing, quotas & provisioned-throughput budgeting,
cost accounting, observability, guardrails / prompt-injection defense.

**Signal keywords:** auth, OAuth, OIDC, SPIFFE, token exchange, RFC 8693, policy, guardrail,
rate limit, admission control, token bucket, quota, provisioned throughput, cost per conversation,
circuit breaker, audit, MCP authorization, gateway.

## Current contents

### `identity-security/`
- **`agentic-identity-core/`** — identity & security for an agent loop in one ~250-line file
  (`agentsec_core.py`): identity, own-vs-delegated authority, policy, resource-server verification, audit.
- **`agentic-identity-gcp-lab/`** — the full lab: SPIFFE principals, cert-bound tokens, RFC 8693
  token exchange, DPoP, policy enforcement, prompt-injection guardrails, MCP resource server,
  A2A agent cards, audit & governance (ADK 2 + MCP + Terraform).
- **`agentic-identity-core-mistral/`** — Mistral-flavoured variant of the identity core
  (`agentsec_core_mistral.py` + walkthrough/practice/solution notebooks).

### `scaling-admission-cost/`
- **`agentic-scaling-lab/`** — scaling by *bounding tokens*: capacity math, provisioned-throughput
  break-even, token-bucket/backoff/circuit-breaker resilience, admission control & load-shedding,
  a Cloud Run + Gemini reference architecture. Cross-refs **05-orchestrator**.
- **`agentic-scaling-lab-mistral/`** — Mistral-platform variant (adds `scalelab/serving.py`;
  platform-mapping docs in place of the GCP mapping).

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

**`identity-security/agentic-identity-core-mistral/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_practice.ipynb) `core_mistral_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_solution.ipynb) `core_mistral_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core-mistral/core_mistral_walkthrough.ipynb) `core_mistral_walkthrough.ipynb`

**`identity-security/agentic-identity-core/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_practice.ipynb) `core_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_solution.ipynb) `core_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-core/core_walkthrough.ipynb) `core_walkthrough.ipynb`

**`identity-security/agentic-identity-gcp-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/01_agent_identity_and_principals.ipynb) `01_agent_identity_and_principals.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/02_delegation_and_token_exchange.ipynb) `02_delegation_and_token_exchange.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/03_auth_manager_broker.ipynb) `03_auth_manager_broker.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/04_policy_enforcement_point.ipynb) `04_policy_enforcement_point.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/05_prompt_injection_and_guardrails.ipynb) `05_prompt_injection_and_guardrails.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/06_mcp_resource_server.ipynb) `06_mcp_resource_server.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/07_a2a_agent_cards.ipynb) `07_a2a_agent_cards.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/08_audit_and_governance.ipynb) `08_audit_and_governance.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/09_code_evaluation_drills.ipynb) `09_code_evaluation_drills.ipynb`

**`identity-security/agentic-identity-gcp-lab/notebooks/practice/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/01_agent_identity_and_principals_practice.ipynb) `01_agent_identity_and_principals_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/02_delegation_and_token_exchange_practice.ipynb) `02_delegation_and_token_exchange_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/03_auth_manager_broker_practice.ipynb) `03_auth_manager_broker_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/04_policy_enforcement_point_practice.ipynb) `04_policy_enforcement_point_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/05_prompt_injection_and_guardrails_practice.ipynb) `05_prompt_injection_and_guardrails_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/06_mcp_resource_server_practice.ipynb) `06_mcp_resource_server_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/07_a2a_agent_cards_practice.ipynb) `07_a2a_agent_cards_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/08_audit_and_governance_practice.ipynb) `08_audit_and_governance_practice.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/practice/09_code_evaluation_drills_practice.ipynb) `09_code_evaluation_drills_practice.ipynb`

**`identity-security/agentic-identity-gcp-lab/notebooks/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/01_agent_identity_and_principals_solution.ipynb) `01_agent_identity_and_principals_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/02_delegation_and_token_exchange_solution.ipynb) `02_delegation_and_token_exchange_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/03_auth_manager_broker_solution.ipynb) `03_auth_manager_broker_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/04_policy_enforcement_point_solution.ipynb) `04_policy_enforcement_point_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/05_prompt_injection_and_guardrails_solution.ipynb) `05_prompt_injection_and_guardrails_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/06_mcp_resource_server_solution.ipynb) `06_mcp_resource_server_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/07_a2a_agent_cards_solution.ipynb) `07_a2a_agent_cards_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/08_audit_and_governance_solution.ipynb) `08_audit_and_governance_solution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/identity-security/agentic-identity-gcp-lab/notebooks/solutions/09_code_evaluation_drills_solution.ipynb) `09_code_evaluation_drills_solution.ipynb`

**`scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/01_scaling_math.ipynb) `01_scaling_math.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/02_turn_loop_and_durability.ipynb) `02_turn_loop_and_durability.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/03_rate_limits_and_admission.ipynb) `03_rate_limits_and_admission.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/04_load_to_settings.ipynb) `04_load_to_settings.ipynb`

**`scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/01_scaling_math_solutions.ipynb) `01_scaling_math_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/02_turn_loop_and_durability_solutions.ipynb) `02_turn_loop_and_durability_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/03_rate_limits_and_admission_solutions.ipynb) `03_rate_limits_and_admission_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/notebooks/solutions/04_load_to_settings_solutions.ipynb) `04_load_to_settings_solutions.ipynb`

**`scaling-admission-cost/agentic-scaling-lab/notebooks/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/01_scaling_math.ipynb) `01_scaling_math.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/02_turn_loop_and_durability.ipynb) `02_turn_loop_and_durability.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/03_rate_limits_and_admission.ipynb) `03_rate_limits_and_admission.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/04_load_to_settings.ipynb) `04_load_to_settings.ipynb`

**`scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/`**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/01_scaling_math_solutions.ipynb) `01_scaling_math_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/02_turn_loop_and_durability_solutions.ipynb) `02_turn_loop_and_durability_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/03_rate_limits_and_admission_solutions.ipynb) `03_rate_limits_and_admission_solutions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniryou/full-stack-agentic-engineer/blob/main/06-gateway/scaling-admission-cost/agentic-scaling-lab/notebooks/solutions/04_load_to_settings_solutions.ipynb) `04_load_to_settings_solutions.ipynb`
<!-- colab-links:end -->
