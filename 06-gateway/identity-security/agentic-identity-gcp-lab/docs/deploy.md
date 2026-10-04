# Deploy runbook — the reference implementation on Google Cloud

This runbook takes the lab from an empty project to a support agent. The agent has these properties:

- It runs as its **own identity**.
- It calls an MCP server that only it can call.
- It uses **user-delegated** credentials that it gets through Auth Manager.
- **Model Armor** screens it.
- It leaves a **dual-identity audit trail** in BigQuery.

Then the runbook adds the governance opt-ins (Agent Gateway, VPC-SC, org policy).
The concept references point at `docs/primer.md`. The product facts come from `docs/sources.md`.

By default, everything runs offline (`AGENTSEC_PROFILE=local`). The notebooks do not need
anything in this runbook. This runbook is the `gcp` profile.

---

## 0. Prerequisites

| Need | Why |
|---|---|
| A project with billing. The best place for the project is inside an organisation (`org_id`). | The org-scoped trust domain `agents.global.org-<ORG>.system.id.goog`, and the org-level opt-ins |
| A recent `gcloud` with the `beta` component, and ADC (`gcloud auth application-default login`) | The beta Cloud Run flags, and the commands for Agent Identity, Agent Registry and IAP |
| Terraform >= 1.11, google provider >= 8.0 | The write-only secret arguments |
| Python 3.11 with `pip install -e ".[gcp]"` and `google-cloud-aiplatform[agent_engines,adk]` | The SDK deployment path |
| Roles: Owner on the project, or the admin roles that `infra/terraform/README.md` lists. For the opt-ins, the org-level admin roles for Access Context Manager, Org Policy and PAB. | |
| Model Armor that is available in your region. The Vertex floor-setting integration is available in a few regions only (see `docs/sources.md`). | `model_armor_location` |

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

Terraform creates these resources, in dependency order:

1. The APIs.
2. The CI deployer, WIF, the fallback SA, the image repo and the staging bucket.
3. The demo secret (write-only).
4. The Auth Manager 3LO provider (PKCE, allowed scopes).
5. The Model Armor template and the project floor (INSPECT_ONLY).
6. The Cloud Run MCP service (fallback SA, internal ingress).
7. The Agent Registry entry.
8. The audit sink.

Terraform does not create the agent-specific IAM until it knows an engine ID (step 4).

Register the callback URL of the provider at the CRM identity provider:

```bash
terraform output auth_provider_redirect_url
```

## 2. Build and deploy the MCP server with its own Agent Identity

```bash
cd ../..   # repo root
PROJECT_ID=$PROJECT_ID REGION=$REGION BUILD=1 infra/scripts/deploy_mcp_cloud_run.sh
```

The script builds `infra/scripts/Dockerfile.mcp` with Cloud Build. Then it runs the command in
the shape that the documentation gives:

```
gcloud beta run deploy agentsec-mcp-tickets --image=... \
  --functional-type=mcp-server --identity-type=agent-identity \
  --no-allow-unauthenticated --ingress=internal --region=$REGION
```

The results (primer §3.3, §7.1):

- The service now has the principal
  `principal://<trust domain>/resources/run/projects/$PROJECT_NUMBER/locations/$REGION/services/agentsec-mcp-tickets`
  (Terraform output `mcp_server_principal`).
- The platform registers the service automatically under `/mcpServers`.
- The new principal does **not** get the permissions of the fallback SA. This is the purpose of this step.

Examine the identity on the revision:

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
`vertexai.Client(project, location, http_options=dict(api_version="v1beta1")).agent_engines.create(agent=AdkApp(...), config={"identity_type": types.IdentityType.AGENT_IDENTITY, ...})`.
Then it prints the engine ID and `effective_identity`. The ADK app (`infra/deploy/agent_engine_app.py`)
registers `GcpAuthProvider` with the `CredentialManager` of ADK. It also attaches the `SecurityPlugin`.

### Path B — Terraform source-based deployment

```bash
infra/scripts/build_agent_source.sh
cd infra/terraform && terraform apply -var deploy_agent_with_terraform=true
```

Terraform does these steps:

- It puts the archive inline (`filebase64`).
- It sets `spec.identity_type = "AGENT_IDENTITY"`.
- It declares the ADK `class_methods`.
- It adds the demo secret through `secret_env`.
- When the gateway is on, it also sets `agent_gateway_config.agent_to_anywhere_config`.

## 4. IAM for the agent principal

Path A: give the engine ID to Terraform. Terraform then calculates the principal and creates the bindings.

```bash
cd infra/terraform
terraform apply -var agent_engine_id=<ID printed by the deploy script>
terraform output agent_principal      # principal://agents.global.org-.../reasoningEngines/<ID>
```

Path B: Terraform created the bindings in step 3.

Then run this script. It adds the one binding that Terraform cannot express, because the
provider has no IAM resource for auth providers. It also grants the other bindings again (idempotent):

```bash
PROJECT_ID=$PROJECT_ID REGION=$REGION \
MCP_SERVER_ID=$(terraform output -raw mcp_server_registry_id) \
../scripts/grant_agent_iam.sh "$(terraform output -raw agent_principal)"
```

At the end, the agent has these roles and this policy (primer §4.5):

- At the project level: `aiplatform.expressUser`, `serviceusage.serviceUsageConsumer`, `browser`, `logging.logWriter`.
- `secretmanager.secretAccessor` on **one** secret.
- `run.invoker` on **one** service.
- `agentidentity.user` on **one** auth provider.
- `iap.egressor` on **one** registry entry.
- A deny policy on the complete agents principalSet. It denies `storage.objects.delete` and `iam.serviceAccountKeys.create`.

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

The `agentsec-audit` log records a denied tool call with `decision=deny`. An example of a denied tool call is
`run_sql`, which is not in `policies/support-agent.yaml`. The Model Armor sanitize logs record a
prompt that matches a Model Armor filter (`log_sanitize_operations = true`).

---

## 6. Opt-ins (independent, enable one at a time)

### 6a. Agent Gateway (network PEP: mTLS + DPoP, IAP per SPIFFE ID, Model Armor on egress)

```bash
terraform apply -var enable_agent_gateway=true
# Path A agents: re-deploy with --agent-gateway "$(terraform output -raw agent_gateway_name)"
```

This command creates these resources:

- The gateway (`AGENT_TO_ANYWHERE`, registry-scoped).
- An IAP `REQUEST_AUTHZ` authorization extension and its policy. To audit first, set `agent_gateway_iap_dry_run = true`.
- A Model Armor `CONTENT_AUTHZ` extension and its policy, which use the strict template.

Examine `terraform output agent_gateway_mtls_endpoint`. Remember the limits of the gateway:

- Registered destinations only (≤ 5,000).
- Publicly trusted certificates.
- No VPC-SC for the gateway itself.

### 6b. VPC Service Controls

```bash
terraform apply -var enable_vpc_sc=true -var access_policy_id=<org access policy id>
# MCP-attribute rule (read-only tools only), dry-run first:
gcloud access-context-manager perimeters dry-run update agentsec_$PROJECT_NUMBER \
  --policy=<id> --set-egress-policies=infra/scripts/vpc_sc_mcp_rule.yaml
```

The perimeter restricts `aiplatform`, `secretmanager`, `run`, the two Agent Identity APIs,
`storage` and `bigquery`. The egress rule uses the agents principalSet. Use the restricted
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
| Engine runs as Agent Identity | Run `gcloud ai reasoning-engines describe <ID> --region=$REGION`, then read `spec.effectiveIdentity`. You can also read the *Identity* column of the Deployments page. |
| MCP revision identity | `gcloud beta run revisions describe <REV> --region=$REGION` |
| Agent's effective roles | `gcloud projects get-iam-policy $PROJECT_ID --flatten=bindings[].members --filter="bindings.members:agents.global"` |
| Deny policy | `gcloud iam policies list --attachment-point=cloudresourcemanager.googleapis.com/projects/$PROJECT_ID --kind=denypolicies` |
| Auth provider binding | `gcloud agent-identity auth-providers get-iam-policy tickets-3lo --location=$REGION` |
| Registry + IAP | `gcloud iap web get-iam-policy --resource-type=agent-registry --mcp-server=<ID> --region=$REGION` |
| Model Armor floor | `gcloud model-armor floorsettings describe --full-uri=projects/$PROJECT_ID/locations/global/floorSetting` |
| Token binding | Replay a captured agent token from a different host. The platform rejects it (the default Context-Aware Access policy). Keep `GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES` unset. |

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

The Model Armor floor setting is a singleton. Thus `destroy` resets it and does not delete it.
Terraform removes the data-access audit-log configuration together with the project IAM audit config resource.
