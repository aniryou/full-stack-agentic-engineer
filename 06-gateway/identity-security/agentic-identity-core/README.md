# agentic-identity-core — identity and security for an agent loop, in one Python file

After this lab you can show, in code that runs on a laptop, the five moves every agent-security design is an
elaboration of — who the agent is, whose authority it acts under, where policy is enforced, how a tool server checks
a token, and what the audit records — and say where each one lives on Google Cloud and on Mistral's platform.

## Start here

1. `python3 -m pip install -r requirements.txt && python3 agentsec_core.py` — a second: the story end to end, with
   the audit timeline.
2. Read [`agentsec_core.py`](agentsec_core.py) top to bottom (about 30 min; ~400 lines, one dependency).
3. Open [`core_walkthrough.ipynb`](core_walkthrough.ipynb), then do [`core_practice.ipynb`](core_practice.ipynb).

## What you get

*T0 = a laptop or Colab CPU, free: no GPU, no key, no cloud account.* Times are rough.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentsec_core.py`](agentsec_core.py) + `core_walkthrough` / `core_practice` / `core_solution` | build the five moves (below) and break each one on purpose; 19 practice blanks with self-checking asserts | 1–2 h | T0 |
| [`agentsec_core_mistral.py`](agentsec_core_mistral.py) + `core_mistral_walkthrough` / `core_mistral_practice` / `core_mistral_solution` | put the same five moves (imported from `agentsec_core.py`, not copied) behind a real model doing function calling (with an offline scripted twin), Mistral's moderation classifier as the screener, and a per-agent key; say where each control lives when the platform gives you the model and connectors but not the identity plane; 20 practice blanks | +1 h | T0 (a key adds the live model) |

**The provider path.** The optional [`requirements-mistral.txt`](requirements-mistral.txt) adds only the `mistralai`
client, whose request and response types the Mistral notebooks and two tests use offline; the core, the Mistral
file's demo and every other test run without it, and without a `MISTRAL_API_KEY` the live path stops with a labelled
message instead of failing.

## Run it

```bash
python3 -m pip install -r requirements.txt           # PyJWT[crypto], pytest
python3 agentsec_core.py                             # the story end to end, with the audit timeline
python3 agentsec_core_mistral.py                     # the same story with a scripted model and a local screener
python3 -m pytest -q                                 # 32 tests, ~2 s (two skip without the Mistral client, one without a key)
python3 -m pip install jupyterlab && python3 -m jupyterlab core_walkthrough.ipynb

python3 -m pip install -r requirements-mistral.txt   # optional: the Mistral client, for the core_mistral_* notebooks
MISTRAL_API_KEY=... python3 agentsec_core_mistral.py # live: mistral-medium-latest + mistral-moderation-2603 (billed)
```

## The five moves

| Move | What the code does | The class | On Google Cloud | On Mistral |
|---|---|---|---|---|
| 1. Identity | every agent is its own principal (SPIFFE-style ID) | `AgentIdentity` | Agent Identity (SPIFFE, certificate-bound tokens) | a **service account** in a Studio workspace with its own workspace-scoped key (connector scope *shared connectors only*); self-hosted, your platform's workload identity |
| 2. Authority | act under the agent's OWN authority or one DELEGATED by a user: a token naming both (`sub` = user, `act` = agent), for ONE audience, narrow scope, 5-minute life; an agent can never widen a user's grant. The STS checks the user token's `aud`, authenticates the agent by its own token (`actor_token`), and honours the user's `may_act` (RFC 8693 §4.4) | `Issuer`, `Authority` | Auth Manager / STS (RFC 8693 token exchange) | **your STS** for your own tool servers; for Studio connectors Mistral brokers credentials per `consumer_scope` (`user` / `workspace` / `organization`), end users authorize OAuth connectors via `connectors.get_auth_url`, and the agent never sees the token |
| 3. Policy | enforced outside the model before every tool call: deny by default, tiers, scopes, human confirmation that shows the real tool + args | `Policy`, `Rule` | ADK `before_tool_callback` (`SecurityPlugin` in the full lab) | the model only *proposes* `tool_calls`; you decide. Studio's `tool_configuration.include / exclude / requires_confirmation` and `Confirmation` allow/deny are the same idea for one connector's tools |
| 4. Resource | the tool server verifies audience + scope itself and authorizes by the *verified* subject; never accepts or forwards someone else's token | `ToolServer` | MCP authorization spec (RFC 9728 / 8707); Agent Gateway | a registered MCP connector with an auth method (`bearer`, `none`, `oauth2` authorization_code / client_credentials) |
| 5. Audit | one event per decision, both identities | `AuditLog` | Cloud Audit Logs + Agent Observability | Studio **Observability** (traces, spans, logs) + **AI Registry**; Le Chat Enterprise audit logs for the chat product |
| Screening | block a prompt before the model; tag tool output as data with its provenance | `screen()`, `fence()`; `MistralModeration` / `LocalScreener` | Model Armor templates + floor settings | `mistral-moderation-2603` (categories incl. `jailbreaking` and `pii`) as a pre-check, or inline with `guardrails=[{"moderation_llm_v2": {...}}]` on `chat.complete`, agents and conversations |

## What the demo shows

```
list_tickets      → allowed (read, delegated by Ana, scope tickets:read)
run_sql           → denied  (not in the policy: default deny — even when the model has its schema)
refund 35 USD     → allowed (destructive, but inside the pre-approved envelope ≤ 50)
refund 60 USD     → human confirmation, then allowed; the approval is in the audit log
refund Ben's T-3  → the SERVER refuses: Ana's token says who she is, not the request body
replay the token at another API → rejected: wrong audience
another agent asks to act for Ana → rejected: her token's may_act names only the support agent
"Ignore previous instructions…"  → blocked before the model runs (the Mistral file: category `jailbreaking`)
```

## Where the Mistral version sits in a deployment

- **Studio (managed)** — the agent runs your code (or the Agents/Conversations API) with a service-account key from a
  dedicated workspace; connectors carry the per-user or per-workspace credentials for SaaS tools; moderation runs
  inline as guardrails; Observability holds the traces. Your policy layer still sits between `tool_calls` and execution.
- **Cloud marketplaces** (Azure AI Foundry, AWS Bedrock, Google Cloud Vertex AI and others) — the model endpoint is
  the cloud's; the identity plane is that cloud's IAM; everything in these files runs unchanged next to it.
- **Self-hosted** (vLLM, TensorRT-LLM, TGI …) — you own the whole stack: the model endpoint is another resource server
  behind your STS; moderation is an open moderation model or your own classifier; the five moves are the architecture.

The live path was checked against the client's own request/response models (`mistralai` 2.10): the transcript
`agentsec_core_mistral.py` builds is what `client.chat.complete` accepts, and the moderation call is
`classifiers.moderate_chat`.

## When you want the step-up

[`agentic-identity-gcp-lab`](../agentic-identity-gcp-lab/README.md) implements each move in production shape:
certificate-bound tokens and DPoP, a credential broker with the consent round-trip, the policy as a plugin on a real
agent runner, an MCP server on the `mcp` SDK, signed A2A agent cards, and Terraform for Google Cloud. Its
[primer](../agentic-identity-gcp-lab/docs/primer.md) is the concept document for both labs. Module 06.6 in
[`CURRICULUM.md`](../../../CURRICULUM.md).

## Caveats

- Everything is local: a token issuer, an in-memory tool server and a regex screener at T0. The semantics
  match the production pieces; the cryptography is real (RS256 via PyJWT).
- Mistral's platform moved quickly in 2026. Re-check the connector credential scopes and `get_auth_url` flow
  (Connectors were in public preview in May 2026), the `guardrails` request field and `moderation_llm_v2` config,
  service-account roles in the Admin API, and the current model names (`mistral-medium-latest` was Mistral Medium 3.5
  when this was written) — all checked September 2026 (verify). Sources:
  [Studio connectors](https://docs.mistral.ai/studio/connectors),
  [managing connectors](https://docs.mistral.ai/studio/connectors/management),
  [moderation & guardrailing](https://docs.mistral.ai/studio/safety-moderation),
  [Agents API basics](https://docs.mistral.ai/agents/agents_basics),
  [API keys](https://docs.mistral.ai/admin/identity-access/api-keys),
  [organizations and workspaces](https://docs.mistral.ai/admin/security-access/organization),
  [deployment options](https://docs.mistral.ai/models/deployment).
