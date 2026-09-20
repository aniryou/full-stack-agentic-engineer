# infra/terraform — the control plane for the agentsec lab

Terraform for everything around the agent that the primer describes: the agent's own
identity, the broker for its delegated authority, the MCP server it calls, the model-side
guardrails, the audit trail, and the opt-in governance layers (gateway, perimeter, org
policy, PAB). Each file maps to a row of the table in `docs/primer.md` §10.

| File | Primer concept | What it creates |
|---|---|---|
| `apis.tf` | — | Service enablement (opt-in APIs only when the flag is on) |
| `identity.tf` | WIF, no SA keys (§3.4, §5) | Keyless CI deployer + GitHub WIF pool/provider, MCP fallback SA, image repo, staging bucket |
| `reasoning_engine.tf` | Agent as first-class principal (§3.3) | `google_vertex_ai_reasoning_engine` with `identity_type = "AGENT_IDENTITY"` (path B only) |
| `secrets.tf` | Secrets (§5) | One demo secret, write-only payload, accessor for the agent principal only |
| `auth_provider.tf` | Delegated authority (§3.2, §3.5) | Auth Manager 3LO provider with PKCE + allowed scopes |
| `cloudrun_mcp.tf` | MCP as resource server (§7.1) | Cloud Run service, internal ingress, `run.invoker` for the agent only |
| `agent_registry.tf` | Inventory / gateway allow-list (§9) | Registry entry for the MCP server + IAP `roles/iap.egressor` for the agent |
| `agent_gateway.tf` | Network PEP (§4.1, §8) — opt-in | Egress gateway + IAP and Model Armor authorization extensions |
| `model_armor.tf` | Model-side screening (§6) | Strict template (INSPECT_AND_BLOCK) + project floor (INSPECT_ONLY first) |
| `iam.tf` | Envelope (§4.5) | Agent's project roles, deny policy for the agents principalSet, opt-in PAB |
| `logging.tf` | Audit (§9) | Data-access audit logs, BigQuery sink for agent principals + `agentsec-audit` |
| `vpc_sc.tf` | Perimeter (§8) — opt-in | Regular perimeter + egress rule for the agents principalSet |
| `org_policy.tf` | Control-plane guardrails (§4.5) — opt-in | Custom constraints (PKCE on auth providers; Agent Identity on engines — illustrative) |
| `locals.tf` / `outputs.tf` | Principal identifiers (§3.3) | `principal://…`, `principalSet://…`, URLs and names other tooling needs |

## Prerequisites

* Terraform **>= 1.11** (write-only arguments keep the OAuth client secret and the demo
  secret out of state), google provider >= 8.0.
* A project with billing, and `gcloud` authenticated as someone with Owner or the equivalent
  set of admin roles (Service Usage Admin, IAM Admin, Cloud Run Admin, Vertex AI Admin,
  Secret Manager Admin, Model Armor Admin, Logging/BigQuery Admin, Agent Identity Admin).
* For opt-ins: org-level roles — Access Context Manager Admin (`enable_vpc_sc`), Org Policy
  Admin (`enable_org_policy`), PAB Admin (`enable_pab`).
* Application Default Credentials: `gcloud auth application-default login`.

## Order of operations

Terraform resolves most ordering itself, but the agent's identity only exists once the agent
is deployed, and Terraform cannot express every binding. The runbook in `docs/deploy.md`
walks through this in detail; the short version:

1. `terraform init && terraform apply` — APIs, identities, secret, auth provider, Model
   Armor, Cloud Run service (fallback SA), registry entry, audit sink. Agent-specific IAM is
   skipped until an engine ID is known.
2. Build and push the MCP image, then run `../scripts/deploy_mcp_cloud_run.sh` to switch the
   service to `--functional-type=mcp-server --identity-type=agent-identity`.
3. Deploy the agent:
   * **Path A (default)** — `python ../scripts/deploy_agent_engine.py` (SDK, mirrors the
     Google docs and `adk deploy`). Put the returned engine ID in `agent_engine_id` and
     `terraform apply` again: the agent's IAM bindings are now created.
   * **Path B** — `../scripts/build_agent_source.sh`, set
     `deploy_agent_with_terraform = true`, `terraform apply`. One plan, one graph.
4. `../scripts/grant_agent_iam.sh "$(terraform output -raw agent_principal)"` for the binding
   Terraform has no resource for (`roles/agentidentity.user` on the auth provider), plus an
   idempotent re-grant of everything else.
5. Opt-ins, one at a time: `enable_agent_gateway`, then `enable_vpc_sc` (with
   `access_policy_id`), then `enable_org_policy` / `enable_pab`. Each is independent.

## Why the opt-ins are opt-in

* **Agent Gateway** needs a network attachment for private destinations, publicly trusted
  certificates on every destination, and every destination registered; it is not covered by
  VPC-SC itself. Turn it on once the direct path works and you want mTLS + DPoP + IAP per
  SPIFFE ID + Model Armor on egress (all configured in `agent_gateway.tf`).
* **VPC Service Controls** requires an organisation access policy and can lock *you* out of
  the project if the perimeter is wrong. Start with the gcloud dry-run YAML in
  `../scripts/vpc_sc_mcp_rule.yaml` for the MCP-attribute rule.
* **Org policy custom constraints / PAB** are organisation-scoped; one of the constraints
  (`aiplatform.googleapis.com/ReasoningEngine`) is illustrative and marked `# VERIFY:`.
* **Terraform-deployed engine** (`deploy_agent_with_terraform`) requires a source archive and
  a hand-declared `class_methods`; the SDK path is what the product docs describe.

## Things marked `# VERIFY:`

Search the tree for `VERIFY:`. They are facts that were not confirmable from the provider
schema or the docs snapshot in `docs/sources.md` (5 Sep 2026): the deny-policy principalSet
form for agents, whether the Reasoning Engine service agent still needs secret access under
Agent Identity, the PAB binding target for agent identities, the TOOL_SPEC JSON shape, the
registry URI version segment, and the illustrative custom constraint.

## Cost and cleanup

Idle cost is small: the Cloud Run service scales to zero, the engine has `min_instances = 0`,
BigQuery is pay-per-query with 90-day table expiry. Data-access audit logs for `allServices`
are the main variable (`enable_data_access_audit_logs = false` to switch off). The gateway and
a min-instance engine are the two things that bill while idle.

`terraform destroy` removes everything it created (buckets and datasets are `force_destroy` /
`delete_contents_on_destroy`). Two things need manual cleanup: the auto-registered Agent
Registry entry created by `deploy_mcp_cloud_run.sh`, and an engine deployed via path A
(`gcloud`/SDK `delete`, see `docs/deploy.md`). Do not run `destroy` with `enable_vpc_sc = true`
before removing the perimeter, or the perimeter may block the deletions.
