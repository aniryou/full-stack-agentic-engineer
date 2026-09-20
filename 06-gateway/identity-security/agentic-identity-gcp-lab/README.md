# agentic-identity-gcp-lab

**Identity & security for agentic systems — a primer and a reference implementation on Google Cloud.**

Written September 2026 against Google Cloud's agent platform. Everything runs **offline** by
default (local fakes for the identity plane), and the same code binds to Google Cloud
(**Agent Identity**, **Auth Manager**, **Agent Engine**, **Model Armor**, **Secret Manager**) when
you set `AGENTSEC_PROFILE=gcp` and apply the Terraform.

```
docs/primer.md        ← start here: the primer (13 numbered sections, drillable)
notebooks/            ← worked examples + fill-in-the-blank practice + solutions, one per primer topic
src/agentsec/         ← the reference implementation (Python, ADK 2.8, MCP SDK, A2A SDK)
policies/             ← deny-by-default tool policy (YAML)
infra/terraform/      ← Google Cloud infrastructure (validated with provider 8.1); infra/scripts/ for gcloud-only steps
tests/                ← 37 tests exercising every flow end to end through the real ADK Runner
```

## What the reference implementation demonstrates

| Primer idea | Where |
|---|---|
| Agent as a first-class principal (SPIFFE ID, `principal://`, `principalSet://`), runtime-attested certificate, **certificate-bound tokens** that fail on replay | `identity/principals.py`, `identity/certs.py`, `identity/tokens.py` |
| **Own vs delegated authority**; RFC 8693 token exchange with `act` chains; DPoP (RFC 9449); Credential Access Boundaries | `identity/delegation.py`, `identity/tokens.py`, `identity/downscope.py` |
| **Credential broker** with Auth Manager semantics (3LO consent, 2LO, API key, IAM on providers, dual-identity access log) and the ADK adapter so tools use `GcpAuthProviderScheme` unchanged | `identity/auth_manager.py`, `agents/root_agent.py::make_crm_lookup_tool` |
| **Runtime policy enforcement point**: deny-by-default tool policy with tiers, principals, scopes, constraints, egress allowlist, budgets, human confirmation — as an ADK plugin on every tool call | `policy/*.py`, `policies/support-agent.yaml` |
| **Model-side screening** (Model Armor shape) and the **untrusted-content boundary** (provenance fencing, sanitisation, SSRF-safe egress) | `guardrails/*.py` |
| **MCP server as an OAuth 2.1 resource server**: RFC 9728 metadata, audience validation, scope→tool, annotations, no token passthrough, optional DPoP | `mcp/server.py`, `mcp/client.py` |
| **A2A**: Agent Card security schemes, detached-JWS signing, per-hop re-authorization | `a2a/*.py` |
| **Audit by construction**: one structured event per decision with user + agent + authority + reasons + approver | `audit/log.py` |
| Google Cloud wiring: Agent Engine with `identity_type = AGENT_IDENTITY`, Auth Manager provider, Cloud Run MCP server registered in Agent Registry, Agent Gateway, Model Armor template + floor settings, deny policy, audit sink, opt-in VPC-SC and org policy | `infra/terraform/*.tf`, `infra/scripts/*`, `docs/deploy.md` |

## Quick start (offline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # 37 tests, ~10s
agentsec demo                 # policy, confirmation round-trip and audit trail in your terminal
agentsec token-demo           # mint / exchange / inspect a delegated, certificate-bound token
agentsec policy-check --agent "spiffe://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/987654321098/locations/us-central1/reasoningEngines/support-agent" \
    --tool issue_refund --args '{"order_id":"O-5001","amount":120,"currency":"USD","reason":"x"}' \
    --user ana@customer.example --scopes payments:refund
jupyter lab notebooks/        # worked examples; then notebooks/practice with notebooks/solutions
```

Run the tickets MCP server on its own (`agentsec mcp-serve --port 8765`) and probe it with
`curl -i http://127.0.0.1:8765/mcp` (expect `401` + `WWW-Authenticate … resource_metadata=`), then
`curl http://127.0.0.1:8765/.well-known/oauth-protected-resource/mcp`.

## Deploying to Google Cloud

See `docs/deploy.md`. In short: `terraform apply` in `infra/terraform` (APIs, identities, secret,
Auth Manager provider, Cloud Run MCP server, Agent Registry, Model Armor, audit sink; Agent Gateway,
VPC-SC and org policy are opt-in), then `infra/scripts/deploy_agent_engine.py` deploys the ADK
agent with `identity_type=AGENT_IDENTITY`, and `infra/scripts/grant_agent_iam.sh` binds the
resulting `principal://…` to exactly the roles the policy needs. Items marked `# VERIFY:` in the
Terraform are product details to re-check against current docs before relying on them.

## How to work through it

1. Read `docs/primer.md` once end to end; each section ends with a 30-second one-sentence answer.
2. Work `notebooks/01…09` in order, then do the practice notebooks without looking at solutions.
3. Drill §11: whiteboard the five system-design prompts, and time yourself on the code-evaluation snippets in `notebooks/09_code_evaluation_drills.ipynb`.
4. Re-check the Verify list (§13) before relying on it — several products here went GA in 2026 and details move.

## Layout of the identity plane (local profile)

```
user ──(IdP id token)──▶ front-end ──seed session (user, scopes | delegated token)──▶ ADK Runner
                                                                                        │ SecurityPlugin (PEP)
        LocalRuntimeCA ──issues cert (SPIFFE SAN)──▶ AgentIdentity                       │
        TokenIssuer (STS): mint · exchange(RFC 8693) · verify(aud, scope, cnf)          ▼
        LocalAuthManager (broker): providers · IAM · consent · retrieve/finalize   tools ─▶ MCP server (resource server)
        LocalScreener (Model Armor shape) · EgressPolicy · wrap_untrusted                │      └─▶ upstream (separate token)
        AuditLog: one event per decision, user + agent + authority                         └─▶ peer agent (A2A, signed card)
```

License: Apache-2.0. Product facts verified on 5 September 2026 — see `docs/sources.md`.
