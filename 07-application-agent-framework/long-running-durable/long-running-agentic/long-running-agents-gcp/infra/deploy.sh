#!/usr/bin/env bash
# Deploy the two Cloud Run services (library service + ADK agent) from source.
# Prereqs: gcloud auth, PROJECT/REGION exported, terraform applied once.
set -euo pipefail
PROJECT=${GOOGLE_CLOUD_PROJECT:?}
REGION=${GOOGLE_CLOUD_LOCATION:-asia-southeast1}
RUN_SA="sa-agent-run@${PROJECT}.iam.gserviceaccount.com"
INVOKER_SA="sa-tasks-invoker@${PROJECT}.iam.gserviceaccount.com"

cd "$(dirname "$0")/.."

# 1. library service (reference architecture A)
gcloud run deploy lragents --source . --region "$REGION" --project "$PROJECT" \
  --service-account "$RUN_SA" --no-allow-unauthenticated --timeout 300 --concurrency 20 \
  --set-env-vars "LRAGENTS_BACKEND=gcp,GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${REGION},TASKS_QUEUE=agent-steps,TASKS_INVOKER_SA=${INVOKER_SA},PUBSUB_INVOKER_SA=${INVOKER_SA},GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash}"
SERVICE_URL=$(gcloud run services describe lragents --region "$REGION" --project "$PROJECT" --format 'value(status.url)')
# the service needs its own URL for OIDC audiences + task targets
gcloud run services update lragents --region "$REGION" --project "$PROJECT" --update-env-vars "SERVICE_URL=${SERVICE_URL}"
gcloud run services add-iam-policy-binding lragents --region "$REGION" --project "$PROJECT" \
  --member "serviceAccount:${INVOKER_SA}" --role roles/run.invoker
echo "service: ${SERVICE_URL}  (re-run terraform apply -var service_url=${SERVICE_URL} to wire push subscriptions)"

# 2. ADK agent service (reference architecture C) — needs a session DB
: "${SESSION_SERVICE_URI:=sqlite+aiosqlite:////tmp/sessions.db}"   # replace with postgresql+asyncpg://...?host=/cloudsql/...
gcloud run deploy lragents-adk --source . --region "$REGION" --project "$PROJECT" \
  --service-account "$RUN_SA" --no-allow-unauthenticated --timeout 600 \
  --command python --args "-m,lragents.adk.main" \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${REGION},SESSION_SERVICE_URI=${SESSION_SERVICE_URI},ARTIFACT_SERVICE_URI=${ARTIFACT_SERVICE_URI:-},GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash}"

# 3. Workflows (also managed by terraform; this is the manual path)
gcloud workflows deploy hitl-approval --source infra/workflows/hitl_approval.yaml --location "$REGION" --project "$PROJECT" \
  --service-account "sa-workflows@${PROJECT}.iam.gserviceaccount.com"
gcloud workflows deploy fan-out-fan-in --source infra/workflows/fan_out_fan_in.yaml --location "$REGION" --project "$PROJECT" \
  --service-account "sa-workflows@${PROJECT}.iam.gserviceaccount.com"
