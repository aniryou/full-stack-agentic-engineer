#!/usr/bin/env bash
# Delete the lab's kind cluster (everything in it goes with it; nothing is left running).
#   deploy/kind/down.sh
#   DRY_RUN=1 deploy/kind/down.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

CLUSTER_NAME="${CLUSTER_NAME:-gpu-lab}"
need kind "https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
log "deleting kind cluster '$CLUSTER_NAME'"
run kind delete cluster --name "$CLUSTER_NAME"
