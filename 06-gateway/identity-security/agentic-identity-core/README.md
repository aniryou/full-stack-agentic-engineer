# agentic-identity-core — identity and security for an agent loop, in one Python file

After this lab, you can show five moves in code that runs on a laptop. Every agent-security design is a more
detailed form of these five moves:

- Who the agent is.
- Whose authority the agent acts under.
- Where the system enforces the policy.
- How a tool server examines a token.
- What the audit records.

You can also say where each move lives on Google Cloud and on Mistral's platform.

## Start here

1. Run `python3 -m pip install -r requirements.txt && python3 agentsec_core.py`. It takes a second. It shows the
   story from end to end, with the audit timeline.
2. Read [`agentsec_core.py`](agentsec_core.py) from top to bottom (about 30 min). It has ~400 lines and one
   dependency.
3. Open [`notebooks/core_walkthrough.ipynb`](notebooks/core_walkthrough.ipynb). Then do
   [`notebooks/core_practice.ipynb`](notebooks/core_practice.ipynb). Its answers are in
   [`solutions/core_practice.ipynb`](solutions/core_practice.ipynb).

## What you get

*T0 is a laptop or a Colab CPU, at no cost: no GPU, no key, no cloud account.* The times are approximate.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentsec_core.py`](agentsec_core.py) + `notebooks/core_walkthrough`, `notebooks/core_practice` (answers: `solutions/core_practice`) | Build the five moves (the table in "The five moves") and break each one on purpose. There are 19 practice blanks, with asserts that do a check of your answers. | 1–2 h | T0 |
| [`agentsec_core_mistral.py`](agentsec_core_mistral.py) + `notebooks/core_mistral_walkthrough`, `notebooks/core_mistral_practice` (answers: `solutions/core_mistral_practice`) | Put the same five moves behind a real model that does function calling (with an offline scripted twin). The file imports the moves from `agentsec_core.py` and does not copy them. Use Mistral's moderation classifier as the screener, and use a per-agent key. Say where each control lives when the platform gives you the model and connectors but not the identity plane. There are 20 practice blanks. | +1 h | T0 (a key adds the live model) |

**The provider path.** The optional [`requirements-mistral.txt`](requirements-mistral.txt) adds only the `mistralai`
client. The Mistral notebooks and two tests use its request and response types offline. The core, the demo of the
Mistral file and every other test run without it. Without a `MISTRAL_API_KEY`, the live path stops with a labelled
message and does not fail.

## Run it

```bash
python3 -m pip install -r requirements.txt           # PyJWT[crypto], pytest
python3 agentsec_core.py                             # the story end to end, with the audit timeline
python3 agentsec_core_mistral.py                     # the same story with a scripted model and a local screener
python3 -m pytest -q                                 # 32 tests, ~2 s (two skip without the Mistral client, one without a key)
python3 -m pip install jupyterlab && python3 -m jupyterlab notebooks/core_walkthrough.ipynb

python3 -m pip install -r requirements-mistral.txt   # optional: the Mistral client, for the notebooks/core_mistral_* notebooks
MISTRAL_API_KEY=... python3 agentsec_core_mistral.py # live: mistral-medium-latest + mistral-moderation-2603 (billed)
```

## The five moves

| Move | What the code does | The class | On Google Cloud | On Mistral |
|---|---|---|---|---|
| 1. Identity | Every agent is its own principal (SPIFFE-style ID). | `AgentIdentity` | Agent Identity (SPIFFE, certificate-bound tokens) | A **service account** in a Studio workspace, with its own workspace-scoped key (connector scope *shared connectors only*). For a self-hosted deployment, the identity is the workload identity of your platform. |
| 2. Authority | Act under the agent's OWN authority, or under an authority that a user DELEGATED. A delegated token names both (`sub` = user, `act` = agent). It is for ONE audience, with a narrow scope and a 5-minute life. An agent can never make a user's grant wider. The STS examines the `aud` of the user token and authenticates the agent by its own token (`actor_token`). It obeys the user's `may_act` (RFC 8693 §4.4). | `Issuer`, `Authority` | Auth Manager / STS (RFC 8693 token exchange) | Use **your STS** for your own tool servers. For Studio connectors, Mistral is the credential broker. The `consumer_scope` (`user` / `workspace` / `organization`) sets the scope of each credential. End users authorize OAuth connectors through `connectors.get_auth_url`. The agent never sees the token. |
| 3. Policy | The code enforces the policy outside the model, before every tool call. The policy has deny by default, tiers, scopes, and human confirmation that shows the real tool and its arguments. | `Policy`, `Rule` | ADK `before_tool_callback` (`SecurityPlugin` in the full lab) | The model only *proposes* `tool_calls`, and you decide. Studio's `tool_configuration.include / exclude / requires_confirmation` and `Confirmation` allow/deny are the same idea for the tools of one connector. |
| 4. Resource | The tool server itself verifies the audience and the scope, and authorizes by the *verified* subject. It never accepts or forwards the token of a different principal. | `ToolServer` | MCP authorization spec (RFC 9728 / 8707), Agent Gateway | A registered MCP connector with an auth method (`bearer`, `none`, `oauth2` authorization_code / client_credentials). |
| 5. Audit | One event for each decision, with both identities. | `AuditLog` | Cloud Audit Logs and Agent Observability | Studio **Observability** (traces, spans, logs) and **AI Registry**. Le Chat Enterprise audit logs for the chat product. |
| Screening | Block a prompt before the model. Tag tool output as data, with its provenance. | `screen()`, `fence()`, `MistralModeration` / `LocalScreener` | Model Armor templates and floor settings | `mistral-moderation-2603` (its categories include `jailbreaking` and `pii`) as a check before the model. Or use it inline with `guardrails=[{"moderation_llm_v2": {...}}]` on `chat.complete`, agents and conversations. |

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

- **Studio (managed)**: the agent runs your code (or the Agents/Conversations API) with a service-account key from a
  dedicated workspace. Connectors carry the per-user or per-workspace credentials for SaaS tools. Moderation runs
  inline as guardrails. Observability holds the traces. Your policy layer still sits between `tool_calls` and
  execution.
- **Cloud marketplaces** (Azure AI Foundry, AWS Bedrock, Google Cloud Vertex AI and others): the model endpoint
  belongs to the cloud. The identity plane is the IAM of that cloud. Everything in these files runs unchanged next to
  it.
- **Self-hosted** (vLLM, TensorRT-LLM, TGI …): you own the full stack. The model endpoint is one more resource server
  behind your STS. Moderation is an open moderation model or your own classifier. The five moves are the
  architecture.

A check compared the live path with the request and response models of the client itself (`mistralai` 2.10). The
transcript that `agentsec_core_mistral.py` builds is what `client.chat.complete` accepts. The moderation call is
`classifiers.moderate_chat`.

## When you want the step-up

[`agentic-identity-gcp-lab`](../agentic-identity-gcp-lab/README.md) implements each move in production shape. It has
certificate-bound tokens and DPoP, a credential broker with the consent round-trip, and the policy as a plugin on a
real agent runner. It also has an MCP server on the `mcp` SDK, signed A2A agent cards, and Terraform for Google
Cloud. Its [primer](../agentic-identity-gcp-lab/docs/primer.md) is the concept document for both labs. The two labs
are module 06.6 in [`CURRICULUM.md`](../../../CURRICULUM.md).

## Caveats

- At T0, everything is local: a token issuer, an in-memory tool server and a regex screener. The semantics match
  the production pieces. The cryptography is real (RS256 through PyJWT).
- Mistral's platform changed fast in 2026. Examine these items again:

    - The connector credential scopes and the `get_auth_url` flow. Connectors were in public preview in May 2026.
    - The `guardrails` request field and the `moderation_llm_v2` config.
    - The service-account roles in the Admin API.
    - The current model names. `mistral-medium-latest` was Mistral Medium 3.5 at the date of this text.

    A check of all these items occurred in September 2026 (verify). Sources:
    [Studio connectors](https://docs.mistral.ai/studio/connectors),
    [managing connectors](https://docs.mistral.ai/studio/connectors/management),
    [moderation & guardrailing](https://docs.mistral.ai/studio/safety-moderation),
    [Agents API basics](https://docs.mistral.ai/agents/agents_basics),
    [API keys](https://docs.mistral.ai/admin/identity-access/api-keys),
    [organizations and workspaces](https://docs.mistral.ai/admin/security-access/organization),
    [deployment options](https://docs.mistral.ai/models/deployment).
