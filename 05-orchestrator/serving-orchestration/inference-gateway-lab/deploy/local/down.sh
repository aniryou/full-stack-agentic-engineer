#!/usr/bin/env bash
# Stop and remove the local stack (both profiles). DRY_RUN=1 prints the command.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
echo "==> removing containers, networks and the locally built image"
run docker compose -f "${HERE}/docker-compose.yaml" --profile sim --profile fake down --rmi local
