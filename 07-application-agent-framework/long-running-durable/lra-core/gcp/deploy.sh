#!/usr/bin/env bash
# One-service deployment: Firestore + Cloud Tasks + Cloud Run + Cloud Scheduler. ~2 minutes.
set -euo pipefail
PROJECT=${PROJECT:?export PROJECT=your-project}; LOCATION=${LOCATION:-asia-southeast1}
SERVICE=lra-core; QUEUE=agent-steps
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
URL="https://${SERVICE}-${NUMBER}.${LOCATION}.run.app"          # Cloud Run's deterministic URL
SA="lra-tasks@${PROJECT}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com cloudtasks.googleapis.com firestore.googleapis.com cloudscheduler.googleapis.com cloudbuild.googleapis.com
gcloud firestore databases describe --database='(default)' >/dev/null 2>&1 || gcloud firestore databases create --location="$LOCATION" --type=firestore-native
gcloud tasks queues describe "$QUEUE" --location="$LOCATION" >/dev/null 2>&1 || gcloud tasks queues create "$QUEUE" --location="$LOCATION"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || gcloud iam service-accounts create lra-tasks --display-name="Cloud Tasks/Scheduler -> Cloud Run"

cp ../core.py ../workflow.py .                                  # the service is core.py + workflow.py + main.py
gcloud run deploy "$SERVICE" --source . --region "$LOCATION" --no-allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT,LOCATION=$LOCATION,QUEUE=$QUEUE,SERVICE_URL=$URL,TASKS_SA=$SA"

# The service's own identity needs Firestore + enqueue rights; the tasks SA may invoke the service.
RUNTIME_SA=$(gcloud run services describe "$SERVICE" --region "$LOCATION" --format='value(spec.template.spec.serviceAccountName)')
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$RUNTIME_SA" --role=roles/datastore.user -q >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$RUNTIME_SA" --role=roles/cloudtasks.enqueuer -q >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$SA" --member="serviceAccount:$RUNTIME_SA" --role=roles/iam.serviceAccountUser -q >/dev/null
gcloud run services add-iam-policy-binding "$SERVICE" --region "$LOCATION" --member="serviceAccount:$SA" --role=roles/run.invoker -q >/dev/null

gcloud scheduler jobs describe lra-reap --location "$LOCATION" >/dev/null 2>&1 || \
  gcloud scheduler jobs create http lra-reap --location "$LOCATION" --schedule="*/2 * * * *" \
    --uri="$URL/reap" --http-method=POST --oidc-service-account-email="$SA"

echo "deployed: $URL"
echo 'try:  TOKEN=$(gcloud auth print-identity-token)'
echo "      curl -X POST $URL/runs -H \"Authorization: Bearer \$TOKEN\" -H 'Content-Type: application/json' -d '{\"run_id\":\"run-1\",\"topic\":\"durable agents\"}'"
echo "      curl -X POST $URL/runs/run-1/events -H \"Authorization: Bearer \$TOKEN\" -H 'Content-Type: application/json' -d '{\"key\":\"approval:run-1\",\"payload\":{\"decision\":\"approve\"}}'"
