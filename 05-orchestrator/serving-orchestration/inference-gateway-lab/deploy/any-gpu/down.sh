#!/usr/bin/env bash
# Stop the vLLM replicas serve.sh started (or the compose stack). DRY_RUN=1 prints the commands.
# A rented GPU keeps billing until you terminate the machine itself -- do that too.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-${HERE}/.igw-gpu}"
DRY_RUN="${DRY_RUN:-0}"
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
echo "==> stopping the replicas started by serve.sh"
if [[ -f "${STATE_DIR}/pids" ]]; then
  while read -r pid; do run kill "${pid}" || true; done < "${STATE_DIR}/pids"
  run rm -rf "${STATE_DIR}"
else
  echo "(no ${STATE_DIR}/pids: nothing started by serve.sh)"
fi
echo "==> stopping the compose stack, if any"
if command -v docker >/dev/null 2>&1 && [[ "${DRY_RUN}" != "1" ]]; then
  docker compose -f "${HERE}/docker-compose.yaml" down || true
else
  echo "+ docker compose -f ${HERE}/docker-compose.yaml down"
fi
