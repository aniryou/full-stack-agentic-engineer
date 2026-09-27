#!/usr/bin/env bash
# Stop and remove the stack, its volume (keys, cache, ledger, spans) and the locally built image.
#   DRY_RUN=1 prints the commands.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
echo "==> removing containers, network and the gwdata volume"
run docker compose -f "${HERE}/docker-compose.yaml" down --volumes
echo "==> removing the locally built image gwlab:local"
run docker image rm gwlab:local || echo "(gwlab:local not present)"
