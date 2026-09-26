# agentic-identity-core

**Identity & security for an agent loop — the core concept, in one Python file.**

This is the small version of `agentic-identity-gcp-lab`: no ADK, no MCP SDK, no Terraform. One
file (`agentsec_core.py`, ~300 lines of code), one dependency (`PyJWT[crypto]`), and the five
moves that everything else is an elaboration of.

| Move | What the code does | The class | On Google Cloud |
|---|---|---|---|
| 1. Identity | every agent is its own principal (SPIFFE-style ID) | `AgentIdentity` | Agent Identity |
| 2. Authority | act under the agent's OWN authority or one DELEGATED by a user: a token naming both (`sub` = user, `act` = agent), for ONE audience, narrow scope, 5-minute life; an agent can never widen a user's grant. The STS checks the user token's `aud`, authenticates the agent by its own token (`actor_token`), and honours the user's `may_act` (RFC 8693 §4.4) | `Issuer`, `Authority` | Auth Manager / STS (RFC 8693 token exchange) |
| 3. Policy | enforced outside the model before every tool call: deny by default, tiers, scopes, human confirmation that shows the real tool + args | `Policy`, `Rule` | ADK `before_tool_callback` (`SecurityPlugin` in the full lab) |
| 4. Resource | the tool server verifies audience + scope itself and authorizes by the *verified* subject; never accepts or forwards someone else's token | `ToolServer` | MCP authorization spec (RFC 9728 / 8707); Agent Gateway |
| 5. Audit | one event per decision, both identities | `AuditLog` | Cloud Audit Logs + Agent Observability |

Plus two tiny helpers for the untrusted-content boundary: `screen()` (block a prompt before the
model, Model Armor's job) and `fence()` (tag tool output as data with its provenance).

## Run it

```bash
pip install "PyJWT[crypto]" pytest jupyter
python agentsec_core.py          # the story end to end, with the audit timeline
pytest -q                        # 13 one-sentence-each tests
jupyter lab core_walkthrough.ipynb
```

`core_walkthrough.ipynb` is the worked version; `core_practice.ipynb` has 19 blanks with
self-checking asserts; `core_solution.ipynb` is the filled-in practice notebook.

## What the demo shows

```
list_tickets      → allowed (read, delegated by Ana, scope tickets:read)
run_sql           → denied  (not in the policy: default deny)
refund 35 USD     → allowed (destructive, but inside the pre-approved envelope ≤ 50)
refund 60 USD     → human confirmation, then allowed; the approval is in the audit log
refund Ben's T-3  → the SERVER refuses: Ana's token says who she is, not the request body
replay the token at another API → rejected: wrong audience
another agent asks to act for Ana → rejected: her token's may_act names only the support agent
"Ignore previous instructions…"  → blocked before the model runs
```

## When you want the step-up

`agentic-identity-gcp-lab` implements each move in production shape: certificate-bound tokens
and DPoP, an Auth-Manager-shaped broker with the ADK consent round-trip, the policy as an ADK
plugin on the real Runner, an MCP server on the `mcp` SDK, signed A2A agent cards, and Terraform
for Agent Engine / Cloud Run / Model Armor / VPC-SC. The primer (`docs/primer.md` there) is the
concept document for both.
