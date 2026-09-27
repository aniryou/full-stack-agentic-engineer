#!/usr/bin/env bash
# Build + push the image with Cloud Build, then apply Terraform. Requires gcloud + terraform.
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-southeast1}"
REPO="${REPO:-lra}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/lra:$(git rev-parse --short HEAD)"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com >/dev/null
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"

cat > /tmp/cloudbuild.yaml <<YAML
steps:
- name: gcr.io/cloud-builders/docker
  args: ['build', '-f', 'services/Dockerfile', '-t', '$IMAGE', '.']
images: ['$IMAGE']
YAML
gcloud builds submit --config /tmp/cloudbuild.yaml .

cd infra/terraform
terraform init -input=false
terraform apply -input=false -auto-approve -var "project_id=$PROJECT_ID" -var "region=$REGION" -var "image=$IMAGE"
terraform output
