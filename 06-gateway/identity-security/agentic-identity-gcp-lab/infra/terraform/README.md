# infra/terraform — the control plane for the agentsec lab

This folder holds the Terraform for everything around the agent that the primer describes:

* the agent's own identity,
* the broker for its delegated authority,
* the MCP server that the agent calls,
* the model-side guardrails,
* the audit trail,
* the opt-in governance layers (gateway, perimeter, org policy, PAB).

Each file has one row in the table in `docs/primer.md` §10.

| File | Primer concept | What it creates |
|---|---|---|
| `apis.tf` | — | Service enablement (opt-in APIs only when the flag is on) |
| `identity.tf` | WIF, no SA keys (§3.4, §5) | Keyless CI deployer, GitHub WIF pool and provider, MCP fallback SA, image repo, staging bucket |
| `reasoning_engine.tf` | Agent as first-class principal (§3.3) | `google_vertex_ai_reasoning_engine` with `identity_type = "AGENT_IDENTITY"` (path B only) |
| `secrets.tf` | Secrets (§5) | One demo secret, write-only payload, accessor for the agent principal only |
| `auth_provider.tf` | Delegated authority (§3.2, §3.5) | Auth Manager 3LO provider with PKCE and permitted scopes |
| `cloudrun_mcp.tf` | MCP as resource server (§7.1) | Cloud Run service, internal ingress, `run.invoker` for the agent only |
| `agent_registry.tf` | Inventory / gateway allow-list (§9) | Registry entry for the MCP server, and IAP `roles/iap.egressor` for the agent |
| `agent_gateway.tf` | Network PEP (§4.1, §8), opt-in | Egress gateway, with IAP and Model Armor authorization extensions |
| `model_armor.tf` | Model-side screening (§6) | Strict template (INSPECT_AND_BLOCK) and project floor (INSPECT_ONLY first) |
| `iam.tf` | Envelope (§4.5) | Agent's project roles, deny policy for the agents principalSet, opt-in PAB |
| `logging.tf` | Audit (§9) | Data-access audit logs, BigQuery sink for agent principals and `agentsec-audit` |
| `vpc_sc.tf` | Perimeter (§8), opt-in | Regular perimeter, and an egress rule for the agents principalSet |
| `org_policy.tf` | Control-plane guardrails (§4.5), opt-in | Custom constraints: PKCE on auth providers, and Agent Identity on engines (illustrative) |
| `locals.tf` / `outputs.tf` | Principal identifiers (§3.3) | `principal://…`, `principalSet://…`, URLs and names that other tools use |

## Prerequisites

* Terraform **>= 1.11** (write-only arguments keep the OAuth client secret and the demo
  secret out of state), google provider >= 8.0.
* A project with billing, and `gcloud` authenticated as a user with Owner or with the equivalent
  set of admin roles. This set is Service Usage Admin, IAM Admin, Cloud Run Admin, Vertex AI Admin
  and Secret Manager Admin. It also contains Model Armor Admin, Logging/BigQuery Admin and Agent
  Identity Admin.
* For the opt-ins, these org-level roles are necessary: Access Context Manager Admin
  (`enable_vpc_sc`), Org Policy Admin (`enable_org_policy`), PAB Admin (`enable_pab`).
* Application Default Credentials: `gcloud auth application-default login`.

## Order of operations

Terraform finds the correct order for most resources itself. But the agent's identity exists only
after you deploy the agent, and Terraform cannot express every binding. The runbook in
`docs/deploy.md` gives all the steps in detail. This is the short version:

1. Run `terraform init && terraform apply`. This step applies the APIs, the identities, the secret
   and the auth provider. It also applies Model Armor, the Cloud Run service (fallback SA), the
   registry entry and the audit sink. Terraform does not apply the agent-specific IAM until it
   knows an engine ID.
2. Build and push the MCP image. Then run `../scripts/deploy_mcp_cloud_run.sh`. This script changes
   the service to `--functional-type=mcp-server --identity-type=agent-identity`.
3. Deploy the agent:
   * **Path A (default):** run `python ../scripts/deploy_agent_engine.py`. This path uses the SDK,
     and it does the same steps as the Google docs and `adk deploy`. Put the returned engine ID in
     `agent_engine_id`. Then run `terraform apply` again. Terraform now creates the agent's IAM bindings.
   * **Path B:** run `../scripts/build_agent_source.sh`. Set `deploy_agent_with_terraform = true`.
     Then run `terraform apply`. You get one plan and one graph.
4. Run `../scripts/grant_agent_iam.sh "$(terraform output -raw agent_principal)"`. This script
   gives the binding that Terraform has no resource for (`roles/agentidentity.user` on the auth
   provider). It also gives everything else again, and this second grant is idempotent.
5. Turn on the opt-ins one at a time. First `enable_agent_gateway`, then `enable_vpc_sc` (with
   `access_policy_id`), then `enable_org_policy` / `enable_pab`. Each opt-in is independent.

## Why the opt-ins are opt-in

* **Agent Gateway** must have a network attachment for private destinations. It must also have
  publicly trusted certificates on every destination, and you must register every destination.
  VPC-SC itself does not cover the gateway. Turn it on when the direct path works and you want
  mTLS, DPoP, IAP per SPIFFE ID and Model Armor on egress. `agent_gateway.tf` configures all of
  these.
* **VPC Service Controls** must have an organisation access policy. If the perimeter is incorrect,
  it can lock *you* out of the project. Start with the gcloud dry-run YAML in
  `../scripts/vpc_sc_mcp_rule.yaml` for the MCP-attribute rule.
* **Org policy custom constraints / PAB** are organisation-scoped. One of the constraints
  (`aiplatform.googleapis.com/ReasoningEngine`) is illustrative, and it has the mark `# VERIFY:`.
* **Terraform-deployed engine** (`deploy_agent_with_terraform`) must have a source archive and a
  `class_methods` that you declare by hand. The product docs describe the SDK path.

## Things marked `# VERIFY:`

Search the tree for `VERIFY:`. Each mark is on a fact that the provider schema and the docs
snapshot in `docs/sources.md` (5 Sep 2026) did not confirm. These facts are:

* the deny-policy principalSet form for agents,
* if the Reasoning Engine service agent must still have secret access under Agent Identity,
* the PAB binding target for agent identities,
* the TOOL_SPEC JSON shape,
* the registry URI version segment,
* the illustrative custom constraint.

## Cost and cleanup

The idle cost is small. The Cloud Run service scales to zero, and the engine has `min_instances = 0`.
BigQuery is pay-per-query, with a 90-day table expiry. The data-access audit logs for `allServices`
are the main variable. To switch them off, set `enable_data_access_audit_logs = false`. The gateway
and a min-instance engine are the two things that bill while idle.

`terraform destroy` removes everything that it created (buckets and datasets are `force_destroy` /
`delete_contents_on_destroy`). You must remove two things by hand:

* the auto-registered Agent Registry entry that `deploy_mcp_cloud_run.sh` created,
* an engine that you deployed through path A (`gcloud`/SDK `delete`, see `docs/deploy.md`).

Do not run `destroy` with `enable_vpc_sc = true` before you remove the perimeter. If you do, it is
possible that the perimeter blocks the deletions.
