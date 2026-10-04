# agentic-identity-gcp-lab

**Identity and security for agentic systems: a primer and a reference implementation on Google Cloud.**

The text and the code of this lab are from September 2026. They are for Google Cloud's agent platform at that
date. By default, everything runs **offline**, with local fakes for the identity plane. The same code connects to
Google Cloud (**Agent Identity**, **Auth Manager**, **Agent Engine**, **Model Armor**, **Secret Manager**) when you
set `AGENTSEC_PROFILE=gcp` and apply the Terraform.

```
docs/primer.md        ← start here: the primer (13 numbered sections, drillable)
notebooks/            ← worked examples + fill-in-the-blank practice, one pair per primer topic
solutions/            ← the practice notebooks completed, under the same file names
src/agentsec/         ← the reference implementation (Python, ADK 2.8, MCP SDK, A2A SDK)
policies/             ← deny-by-default tool policy (YAML)
infra/terraform/      ← Google Cloud infrastructure (validated with provider 8.1); infra/scripts/ for gcloud-only steps
tests/                ← 41 tests exercising every flow end to end through the real ADK Runner
```

**Time and tier:** ~10 h at T0 (approximate), or ~12 h with the core. The lab is in module 06.6 of
[`CURRICULUM.md`](../../../CURRICULUM.md). T0 is a laptop or a Colab CPU, at no cost. At T0, everything runs offline
on local fakes, with no key. A Google Cloud project adds the optional T3 path (`terraform apply` and the agent deploy
in `docs/deploy.md`). You pay for the T3 path per use.

## What the reference implementation demonstrates

| Primer idea | Where |
|---|---|
| An agent as a first-class principal (SPIFFE ID, `principal://`, `principalSet://`), a runtime-attested certificate, and **certificate-bound tokens** that fail on replay. | `identity/principals.py`, `identity/certs.py`, `identity/tokens.py` |
| **Own authority against delegated authority**, RFC 8693 token exchange with `act` chains, DPoP (RFC 9449), Credential Access Boundaries. | `identity/delegation.py`, `identity/tokens.py`, `identity/downscope.py` |
| A **credential broker** with Auth Manager semantics (3LO consent, 2LO, API key, IAM on providers, dual-identity access log). The ADK adapter lets tools use `GcpAuthProviderScheme` unchanged. | `identity/auth_manager.py`, `agents/root_agent.py::make_crm_lookup_tool` |
| **Runtime policy enforcement point**: a deny-by-default tool policy with tiers, principals, scopes, constraints, an egress allowlist, budgets and human confirmation. It is an ADK plugin on every tool call. | `policy/*.py`, `policies/support-agent.yaml` |
| **Model-side screening** (Model Armor shape) and the **untrusted-content boundary** (provenance fencing, sanitisation, SSRF-safe egress). | `guardrails/*.py` |
| **MCP server as an OAuth 2.1 resource server**: RFC 9728 metadata, audience validation, a map from scope to tool, annotations, no token passthrough, optional DPoP. | `mcp/server.py`, `mcp/client.py` |
| **A2A**: Agent Card security schemes, detached-JWS signatures, per-hop re-authorization. | `a2a/*.py` |
| **Audit by construction**: one structured event for each decision, with the user, the agent, the authority, the reasons and the approver. | `audit/log.py` |
| The Google Cloud configuration has Agent Engine with `identity_type = AGENT_IDENTITY`, an Auth Manager provider, and a Cloud Run MCP server registered in Agent Registry. The configuration also has Agent Gateway, a Model Armor template and floor settings, a deny policy, an audit sink, and opt-in VPC-SC and org policy. | `infra/terraform/*.tf`, `infra/scripts/*`, `docs/deploy.md` |

## Quick start (offline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # 41 tests, ~6 s
agentsec demo                 # policy, confirmation round-trip and audit trail in your terminal
agentsec token-demo           # mint / exchange / inspect a delegated, certificate-bound token
agentsec policy-check --agent "spiffe://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/987654321098/locations/us-central1/reasoningEngines/support-agent" \
    --tool issue_refund --args '{"order_id":"O-5001","amount":120,"currency":"USD","reason":"x"}' \
    --user ana@customer.example --scopes payments:refund
jupyter lab notebooks/        # worked examples, then the *_practice blanks; answers in solutions/
```

Run the tickets MCP server alone (`agentsec mcp-serve --port 8765`). Examine it with
`curl -i http://127.0.0.1:8765/mcp`. Expect `401` and `WWW-Authenticate … resource_metadata=`. Then run
`curl http://127.0.0.1:8765/.well-known/oauth-protected-resource/mcp`.

## Deploying to Google Cloud

Read `docs/deploy.md` for the full procedure. In short, do these steps:

1. Run `terraform apply` in `infra/terraform`. The apply covers the APIs, identities, secret and Auth Manager provider.
   It also covers the Cloud Run MCP server, Agent Registry, Model Armor and audit sink. Agent Gateway, VPC-SC and org policy are opt-in.
2. Run `infra/scripts/deploy_agent_engine.py`. It deploys the ADK agent with `identity_type=AGENT_IDENTITY`.
3. Run `infra/scripts/grant_agent_iam.sh`. The deploy makes a `principal://…`. The script gives this
   principal exactly the roles that the policy needs.

The `# VERIFY:` items in the Terraform are product details. Examine them again against the current docs before you
rely on them.

## How to work through it

1. Read `docs/primer.md` one time, from end to end. Each section ends with a 30-second answer in one sentence.
2. Do `notebooks/01…09` in sequence. Then do the practice notebooks, and do not look at the solutions.
3. Do the drills in §11. Answer the five system-design prompts on a whiteboard. Measure your time on the
   code-evaluation snippets in `notebooks/09_code_evaluation_drills.ipynb`.
4. Examine the Verify list (§13) again before you rely on it, because several products here went GA in 2026.
   Their details change.

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

License: Apache-2.0. A check of the product facts occurred on 5 September 2026. See `docs/sources.md`.
