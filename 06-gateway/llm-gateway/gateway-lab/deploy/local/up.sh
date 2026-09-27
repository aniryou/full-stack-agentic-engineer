#!/usr/bin/env bash
# Build the lab image and start the gateway + two fake providers with Docker Compose; wait until it answers.
#   DRY_RUN=1 ./up.sh   prints every command without running it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

step "1/3 checking prerequisites"
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v docker >/dev/null || { echo "docker not found; rerun with DRY_RUN=1 to see the steps, or use 'python -m gwlab stack'"; exit 1; }
  docker info >/dev/null 2>&1 || { echo "the docker CLI is installed but no daemon answers (start Docker Desktop or dockerd)"; exit 1; }
  docker compose version >/dev/null || { echo "the 'docker compose' plugin is required"; exit 1; }
fi

step "2/3 building gwlab:local and starting gateway, acme (30% failures) and bolt"
run docker compose -f "${HERE}/docker-compose.yaml" up -d --build

step "3/3 waiting for ${GATEWAY_URL}/health"
if [[ "${DRY_RUN}" != "1" ]]; then
  for _ in $(seq 1 60); do
    if curl -fsS "${GATEWAY_URL}/health" >/dev/null 2>&1; then echo "gateway is up"; break; fi
    sleep 1
  done
  curl -fsS "${GATEWAY_URL}/health" >/dev/null || { echo "gateway did not become healthy: docker compose -f ${HERE}/docker-compose.yaml logs gateway"; exit 1; }
else
  echo "+ curl -fsS ${GATEWAY_URL}/health   (retried for up to a minute)"
fi

echo
echo "next: ${HERE}/smoke.sh   |   metrics: ${GATEWAY_URL}/metrics   |   stop: ${HERE}/down.sh"
