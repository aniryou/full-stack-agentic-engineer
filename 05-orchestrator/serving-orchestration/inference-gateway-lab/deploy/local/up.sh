#!/usr/bin/env bash
# Start the local stack with Docker Compose and wait until the router answers.
#   PROFILE=sim|fake (default sim)   DRY_RUN=1 prints the commands without running them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-sim}"
DRY_RUN="${DRY_RUN:-0}"
ROUTER_URL="${ROUTER_URL:-http://localhost:9000}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

case "${PROFILE}" in sim|fake) ;; *) echo "PROFILE must be sim or fake"; exit 2 ;; esac

step "1/3 checking prerequisites"
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v docker >/dev/null || { echo "docker not found; rerun with DRY_RUN=1 to see the steps"; exit 1; }
  docker compose version >/dev/null || { echo "the 'docker compose' plugin is required"; exit 1; }
fi

step "2/3 starting profile '${PROFILE}' (3 backends + router + Prometheus)"
run docker compose -f "${HERE}/docker-compose.yaml" --profile "${PROFILE}" up -d --build

step "3/3 waiting for ${ROUTER_URL}/health"
if [[ "${DRY_RUN}" != "1" ]]; then
  for _ in $(seq 1 60); do
    if curl -fsS "${ROUTER_URL}/health" >/dev/null 2>&1; then echo "router is up"; break; fi
    sleep 2
  done
  curl -fsS "${ROUTER_URL}/health" >/dev/null || { echo "router did not become healthy"; exit 1; }
else
  echo "+ curl -fsS ${ROUTER_URL}/health   (retried for up to 2 minutes)"
fi

echo
echo "next: ${HERE}/smoke.sh   |   Prometheus: http://localhost:9090   |   stop: ${HERE}/down.sh"
