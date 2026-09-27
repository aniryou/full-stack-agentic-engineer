#!/usr/bin/env bash
# Stop the local stack and delete its volumes — the memory database, the audit log and Postgres's data.
# That is the local "forget everything"; DRY_RUN=1 prints the command only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> stop the stack and remove its volumes"
if [[ "${DRY_RUN}" != "1" ]] && ! docker info >/dev/null 2>&1; then
  echo "   no reachable Docker daemon: nothing to stop"
  DRY_RUN=1
fi
run docker compose -f "${HERE}/compose.yaml" --profile pgvector down --volumes --remove-orphans
