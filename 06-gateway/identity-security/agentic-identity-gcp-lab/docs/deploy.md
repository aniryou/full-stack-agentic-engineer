# Deploy runbook — the reference implementation on Google Cloud

This runbook takes the lab from an empty project to a support agent that runs as its **own
identity**, calls an MCP server that only it may invoke, borrows **user-delegated** credentials
through Auth Manager, is screened by **Model Armor**, and leaves a **dual-identity audit
trail** in BigQuery — then adds the governance opt-ins (Agent Gateway, VPC-SC, org policy).
Concept references point at `docs/primer.md`; product facts come from `docs/sources.md`.

Everything runs offline by default (`AGENTSEC_PROFILE=local`). Nothing here is needed for the
notebooks; this is the `gcp` profile.

---

## 0. Prerequisites

| Need | Why |
|---|---|
| A project with billing, ideally inside an organisation (`org_id`) | Org-scoped trust domain `agents.global.org-<ORG>.system.id.goog`; org-level opt-ins |
| `gcloud` (recent, with the `beta` component) and ADC (`gcloud auth application-default login`) | Beta Cloud Run flags, Agent Identity / Registry / IAP commands |
| Terraform >= 1.11, google provider >= 8.0 | Write-only secret arguments |
| Python 3.11 with `pip install -e ".[gcp]"` and `google-cloud-aiplatform[agent_engines,adk]` | SDK deployment path |
| Roles: Owner on the project (or the admin roles listed in `infra/terraform/README.md`); for opt-ins the org-level Access Context Manager / Org Policy / PAB admin roles | |
| Model Armor availability in your region; the Vertex floor-setting integration is limited to a few regions (see `docs/sources.md`) | `model_armor_location` |

```bash
export PROJECT_ID=my-agentsec-lab
export REGION=us-central1
export PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export ORG_ID=$(gcloud projects get-ancestors "$PROJECT_ID" --format='value(id)' | tail -1)   # blank if no org
gcloud config set project "$PROJECT_ID"
```

---

## 1. APIs, identities, secret, auth provider, MCP service, guardrails, audit (Terraform)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars      # fill project_id, project_number, org_id, github_repo
export TF_VAR_crm_oauth_client_secret='...'       # never in the tfvars file
export TF_VAR_demo_secret_value='demo-secret-rotate-me'
terraform init
terraform plan
terraform apply
```

What this creates, in dependency order: APIs → CI deployer + WIF + fallback SA + image repo +
staging bucket → demo secret (write-only) → Auth Manager 3LO provider (PKCE, allowed scopes)
→ Model Armor template + project floor (INSPECT_ONLY) → Cloud Run MCP service (fallback SA,
internal ingress) → Agent Registry entry → audit sink. Agent-specific IAM is skipped until an
engine ID is known (step 4).

Register the provider's callback URL at the CRM identity provider:

```bash
terraform output auth_provider_redirect_url
```

## 2. Build and deploy the MCP server with its own Agent Identity

```bash
cd ../..   # repo root
PROJECT_ID=$PROJECT_ID REGION=$REGION BUILD=1 infra/scripts/deploy_mcp_cloud_run.sh
```

The script builds `infra/scripts/Dockerfile.mcp` with Cloud Build and runs the documented
command shape:

```
gcloud beta run deploy agentsec-mcp-tickets --image=... \
  --functional-type=mcp-server --identity-type=agent-identity \
  --no-allow-unauthenticated --ingress=internal --region=$REGION
```

Consequences (primer §3.3, §7.1): the service now has principal
`principal://<trust domain>/resources/run/projects/$PROJECT_NUMBER/locations/$REGION/services/agentsec-mcp-tickets`
(Terraform output `mcp_server_principal`), it is auto-registered under `/mcpServers`, and the
fallback SA's permissions do **not** carry over — that is the point.

Verify the identity on the revision:

```bash
REV=$(gcloud run services describe agentsec-mcp-tickets --region=$REGION --format='value(status.latestReadyRevisionName)')
gcloud beta run revisions describe "$REV" --region=$REGION --format=yaml | grep -iE 'identity|spiffe'
```

## 3. Deploy the agent (pick one path)

### Path A — Python SDK (default, what the product docs describe)

```bash
python infra/scripts/deploy_agent_engine.py \
  --project "$PROJECT_ID" --location "$REGION" \
  --staging-bucket "$(terraform -chdir=infra/terraform output -raw agent_staging_bucket | sed 's#gs://##')" \
  --env-from-terraform infra/terraform
```

The script calls
`vertexai.Client(project, location, http_options=dict(api_version="v1beta1")).agent_engines.create(agent=AdkApp(...), config={"identity_type": types.IdentityType.AGENT_IDENTITY, ...})`
and prints the engine ID and `effective_identity`. The ADK app (`infra/deploy/agent_engine_app.py`)
registers `GcpAuthProvider` with ADK's `CredentialManager` and attaches the `SecurityPlugin`.

### Path B — Terraform source-based deployment

```bash
infra/scripts/build_agent_source.sh
cd infra/terraform && terraform apply -var deploy_agent_with_terraform=true
```

Terraform inlines the archive (`filebase64`), sets `spec.identity_type = "AGENT_IDENTITY"`,
declares the ADK `class_methods`, injects the demo secret via `secret_env`, and — when the
gateway is on — `agent_gateway_config.agent_to_anywhere_config`.

## 4. IAM for the agent principal

Path A: tell Terraform the engine ID so it can compute the principal and create the bindings.

```bash
cd infra/terraform
terraform apply -var agent_engine_id=<ID printed by the deploy script>
terraform output agent_principal      # principal://agents.global.org-.../reasoningEngines/<ID>
```

Path B: the bindings were created in step 3.

Then the one binding Terraform cannot express (no IAM resource for auth providers in the
provider), plus an idempotent re-grant of the rest:

```bash
PROJECT_ID=$PROJECT_ID REGION=$REGION \
MCP_SERVER_ID=$(terraform output -raw mcp_server_registry_id) \
../scripts/grant_agent_iam.sh "$(terraform output -raw agent_principal)"
```

What the agent ends up with (primer §4.5): project-level `aiplatform.expressUser`,
`serviceusage.serviceUsageConsumer`, `browser`, `logging.logWriter`; `secretmanager.secretAccessor`
on **one** secret; `run.invoker` on **one** service; `agentidentity.user` on **one** auth
provider; `iap.egressor` on **one** registry entry; and a deny policy on the whole agents
principalSet that forbids `storage.objects.delete` and `iam.serviceAccountKeys.create`.

## 5. Smoke test

```bash
# 5a. the agent answers, and the audit trail shows both identities
python - <<'EOF'
import os, vertexai
client = vertexai.Client(project=os.environ["PROJECT_ID"], location=os.environ["REGION"], http_options=dict(api_version="v1beta1"))
engine = next(e for e in client.agent_engines.list() if e.api_resource.display_name == "agentsec-support-agent")
print("identity:", engine.api_resource.spec.effective_identity)
session = engine.create_session(user_id="ana@customer.example", state={"agentsec:user": {"subject": "ana", "email": "ana@customer.example", "tenant": "acme"}, "agentsec:scopes": ["customers:read", "orders:read"]})
for ev in engine.stream_query(user_id="ana@customer.example", session_id=session["id"], message="Look up my order O-5001"):
    print(ev)
EOF

# 5b. audit rows: platform audit logs by the agent principal + agentsec events
bq query --use_legacy_sql=false "
SELECT timestamp, protoPayload.authenticationInfo.principalSubject, protoPayload.methodName
FROM \`$PROJECT_ID.agentsec_audit.cloudaudit_googleapis_com_data_access\`
WHERE protoPayload.authenticationInfo.principalSubject LIKE 'principal://agents.global%'
ORDER BY timestamp DESC LIMIT 20"

# 5c. the MCP server refuses anonymous calls (401 + RFC 9728 pointer) and unknown audiences
MCP=$(terraform -chdir=infra/terraform output -raw mcp_server_url)
curl -si "$MCP" -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | head -5
# (with mcp_ingress = INGRESS_TRAFFIC_INTERNAL_ONLY this must be run from inside the VPC / gateway)
```

A denied tool call (e.g. `run_sql`, not in `policies/support-agent.yaml`) appears in the
`agentsec-audit` log with `decision=deny`; a prompt that trips Model Armor appears in the
Model Armor sanitize logs (`log_sanitize_operations = true`).

---

## 6. Opt-ins (independent, enable one at a time)

### 6a. Agent Gateway (network PEP: mTLS + DPoP, IAP per SPIFFE ID, Model Armor on egress)

```bash
terraform apply -var enable_agent_gateway=true
# Path A agents: re-deploy with --agent-gateway "$(terraform output -raw agent_gateway_name)"
```

Creates the gateway (`AGENT_TO_ANYWHERE`, registry-scoped), an IAP `REQUEST_AUTHZ` authorization
extension + policy (set `agent_gateway_iap_dry_run = true` to audit first), and a Model Armor
`CONTENT_AUTHZ` extension + policy using the strict template. Check
`terraform output agent_gateway_mtls_endpoint`. Remember the limits: registered destinations
only (≤ 5,000), publicly trusted certificates, no VPC-SC for the gateway itself.

### 6b. VPC Service Controls

```bash
terraform apply -var enable_vpc_sc=true -var access_policy_id=<org access policy id>
# MCP-attribute rule (read-only tools only), dry-run first:
gcloud access-context-manager perimeters dry-run update agentsec_$PROJECT_NUMBER \
  --policy=<id> --set-egress-policies=infra/scripts/vpc_sc_mcp_rule.yaml
```

The perimeter restricts `aiplatform`, `secretmanager`, `run`, both Agent Identity APIs,
`storage` and `bigquery`; the egress rule uses the agents principalSet. Use the restricted
VIP (`restricted.googleapis.com`) for Agent Identity traffic (docs/sources.md).

### 6c. Org policy custom constraints / PAB

```bash
terraform apply -var enable_org_policy=true      # PKCE required on auth providers (+ illustrative Agent Identity constraint, see VERIFY)
terraform apply -var enable_pab=true             # principals of this project confined to this project
```

---

## 7. Verification checklist

| Check | Command |
|---|---|
| Engine runs as Agent Identity | `gcloud ai reasoning-engines describe <ID> --region=$REGION` → `spec.effectiveIdentity` (or the Deployments page, *Identity* column) |
| MCP revision identity | `gcloud beta run revisions describe <REV> --region=$REGION` |
| Agent's effective roles | `gcloud projects get-iam-policy $PROJECT_ID --flatten=bindings[].members --filter="bindings.members:agents.global"` |
| Deny policy | `gcloud iam policies list --attachment-point=cloudresourcemanager.googleapis.com/projects/$PROJECT_ID --kind=denypolicies` |
| Auth provider binding | `gcloud agent-identity auth-providers get-iam-policy tickets-3lo --location=$REGION` |
| Registry + IAP | `gcloud iap web get-iam-policy --resource-type=agent-registry --mcp-server=<ID> --region=$REGION` |
| Model Armor floor | `gcloud model-armor floorsettings describe --full-uri=projects/$PROJECT_ID/locations/global/floorSetting` |
| Token binding | Replay a captured agent token from another host — it is rejected (Context-Aware Access default policy; keep `GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES` unset) |

---

## 8. Teardown

```bash
# 1. opt-ins first (a live perimeter can block the deletions that follow)
cd infra/terraform
terraform apply -var enable_vpc_sc=false -var enable_agent_gateway=false -var enable_org_policy=false -var enable_pab=false
# 2. the SDK-deployed engine (path A) is not in state
python ../scripts/deploy_agent_engine.py --project $PROJECT_ID --location $REGION \
  --delete projects/$PROJECT_ID/locations/$REGION/reasoningEngines/<ID>
# 3. everything else
terraform destroy
# 4. leftovers created outside Terraform
gcloud run services delete agentsec-mcp-tickets --region=$REGION --quiet   # if re-created by the script after destroy
gcloud agent-registry mcp-servers list --location=$REGION                     # delete the auto-registered entry
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/agentsec  # images are removed with the repo
```

The Model Armor floor setting is a singleton: `destroy` resets it rather than deleting it.
Data-access audit-log configuration is removed with the project IAM audit config resource.
