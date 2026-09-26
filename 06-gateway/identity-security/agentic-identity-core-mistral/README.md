# agentic-identity-core-mistral

**Identity & security for an agent loop, on Mistral — the core concept in one Python file.**

This is `agentic-identity-core` moved onto Mistral's platform. Same five moves, same ~300 lines;
the model is a real Mistral model doing function calling (with an offline scripted twin), the
screener is Mistral's moderation classifier, and the mapping says where each control lives on
Mistral instead of Google Cloud.

The important difference from the Google version is *what the platform gives you*. Mistral
gives you the model, the tool plumbing (Studio connectors = registered MCP servers with
per-user / per-workspace / per-organization credentials) and the moderation classifier. The
identity plane (who the agent is, who it acts for) and the policy layer (what it may do) are
yours — or your cloud's, when you self-host.

| Move | In this file | On Mistral | On Google Cloud (for contrast) |
|---|---|---|---|
| 1. Identity | `AgentIdentity` — one agent, one principal; the key is read at use, never stored | a **service account** in a Studio **workspace** with its own key (keys are workspace-scoped; connector scope *shared connectors only*); self-hosted, your platform's workload identity (SPIFFE / cloud IAM) | Agent Identity (SPIFFE, certificate-bound tokens) |
| 2. Authority | `Issuer.exchange` — user token + the agent's own token (`actor_token`) → one token with `sub`=user, `act`=agent, one `aud`, narrowed scope, 5-minute life; the STS checks the user token's `aud` and `may_act` and verifies the agent's token first | **your STS** for your own tool servers. For Studio connectors, Mistral brokers credentials with `consumer_scope` `user` / `workspace` / `organization`; end users authorize OAuth connectors via `connectors.get_auth_url`; the agent never sees the token | Auth Manager (3LO/2LO/API key providers) |
| 3. Policy | `Policy.evaluate` before every tool call: deny by default, tiers, scopes, argument envelope, human confirmation showing the real tool + args | the model only *proposes* `tool_calls`; you decide. Studio's `tool_configuration.include / exclude / requires_confirmation` on a connector and `Confirmation` allow/deny are the same idea for connector tools | ADK `before_tool_callback`, Agent Gateway |
| 4. Resource | `ToolServer.call` verifies `aud` + scope, then authorizes by the **verified subject** | a registered MCP connector with an auth method (`bearer`, `none`, `oauth2` authorization_code / client_credentials); the connector calls your server with the brokered credential | MCP server on Cloud Run behind Agent Gateway / IAP |
| 5. Audit | `AuditLog` — one event per decision, both identities | Studio **Observability** (traces, spans, logs) + **AI Registry**; Le Chat Enterprise audit logs for the chat product | Cloud Audit Logs, Agent Observability |
| Screening | `MistralModeration` (live) / `LocalScreener` (offline) | `mistral-moderation-2603` — categories incl. `jailbreaking` (prompt injection) and `pii`; as a pre-check, or inline with `guardrails=[{"moderation_llm_v2": {"custom_category_thresholds": {...}, "action": "block"}}]` on `chat.complete`, agents and conversations | Model Armor templates + floor settings |

## Run it

```bash
pip install -r requirements.txt
python agentsec_core_mistral.py                       # offline: scripted model, local screener
MISTRAL_API_KEY=... python agentsec_core_mistral.py   # live: mistral-medium-latest + moderation-2603
pytest -q                                             # 16 offline tests (+1 live test when the key is set)
jupyter lab core_mistral_walkthrough.ipynb
```

`core_mistral_walkthrough.ipynb` is the worked version; `core_mistral_practice.ipynb` has 20
blanks with self-checking asserts; `core_mistral_solution.ipynb` is the filled-in practice.

The live path was validated against the SDK's own request/response models (`mistralai` 2.10):
the transcript this file builds — `tools=[{"type": "function", ...}]`, the assistant turn with
`tool_calls`, one `{"role": "tool", "tool_call_id": ...}` message per call — is exactly what
`client.chat.complete` accepts, and the moderation call uses `classifiers.moderate_chat`.

## What the demo shows

```
list_tickets      → allowed (read, delegated by Ana, scope tickets:read)
run_sql           → denied  (not in the policy: default deny — even though the model has the schema)
refund 35         → allowed (destructive, inside the pre-approved envelope ≤ 50)
refund 60         → human confirmation, then allowed; the approval is in the audit log
refund Ben's T-3  → the SERVER refuses: Ana's token says who she is, not the request body
replay the token at another API → rejected: wrong audience
another agent asks to act for Ana → rejected: her token's may_act names only the support agent
"Ignore previous instructions…"  → blocked by moderation (`jailbreaking`) before the model runs
```

## Where this sits in a Mistral deployment

- **Studio (SaaS)** — the agent runs your code (or the Agents/Conversations API) with a
  service-account key from a dedicated workspace; connectors carry the per-user or per-workspace
  credentials for SaaS tools; moderation runs inline as guardrails; Observability holds the
  traces. Your policy layer still sits between `tool_calls` and execution.
- **Cloud marketplaces** (Azure AI Foundry, AWS Bedrock, Google Cloud Vertex AI, and others) —
  the model endpoint is the cloud's; the identity plane is that cloud's IAM (workload identity,
  managed identities, service accounts); everything in this file runs unchanged next to it.
- **Self-hosted / Mistral Compute** (vLLM, TensorRT-LLM, TGI …) — you own the whole stack: the
  model endpoint is just another resource server behind your STS; moderation is the open
  moderation model or your own classifier; the five moves are the architecture.

## Verify before relying on it

Mistral's platform moved quickly in 2026. Re-check: the connector credential scopes and
`get_auth_url` flow (Connectors were in public preview in May 2026), the `guardrails` request
field and `moderation_llm_v2` config, service-account roles in the Admin API, and the current
model names (`mistral-medium-latest` was Mistral Medium 3.5 when this was written).

Sources: [Studio connectors](https://docs.mistral.ai/studio/connectors),
[managing connectors](https://docs.mistral.ai/studio/connectors/management),
[connectors announcement](https://mistral.ai/news/connectors/),
[moderation & guardrailing](https://docs.mistral.ai/studio/safety-moderation),
[Agents API basics](https://docs.mistral.ai/agents/agents_basics),
[Mistral AI Studio](https://mistral.ai/news/ai-studio/),
[API keys](https://docs.mistral.ai/admin/identity-access/api-keys),
[organizations and workspaces](https://docs.mistral.ai/admin/security-access/organization),
[deployment options](https://docs.mistral.ai/models/deployment).
