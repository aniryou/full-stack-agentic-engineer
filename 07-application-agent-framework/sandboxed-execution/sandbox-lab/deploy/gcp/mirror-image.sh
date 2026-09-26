#!/usr/bin/env bash
# Copy python:3.12-slim into the lab's Artifact Registry repository. The cluster's nodes are private with
# no Cloud NAT, so they can pull only from Google's registries (Private Google Access) - not Docker Hub.
#   DRY_RUN=1 deploy/gcp/mirror-image.sh
#   deploy/gcp/mirror-image.sh                 # REPO from `terraform output -raw repository_url`
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"
need docker "Docker pulls, tags and pushes the image (or use Cloud Shell, which has it)."
need gcloud "Install the Google Cloud CLI."
if [[ -z "${REPO:-}" ]]; then
  if [[ "$DRY_RUN" == "1" ]] || ! command -v terraform >/dev/null 2>&1; then REPO="us-central1-docker.pkg.dev/PROJECT_ID/sandbox"
  else REPO="$(terraform -chdir="$HERE/terraform" output -raw repository_url)"; fi
fi
HOST="${REPO%%/*}"
log "1/3 docker credential helper for $HOST"
run gcloud auth configure-docker "$HOST" --quiet
log "2/3 pull and tag $SANDBOX_IMAGE"
run docker pull "$SANDBOX_IMAGE"
run docker tag "$SANDBOX_IMAGE" "$REPO/$SANDBOX_IMAGE"
log "3/3 push"
run docker push "$REPO/$SANDBOX_IMAGE"
