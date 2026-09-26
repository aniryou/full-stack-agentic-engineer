#!/usr/bin/env bash
# Delete the kind cluster (and everything in it). DRY_RUN=1 prints the command.
set -euo pipefail
CLUSTER="${CLUSTER:-igw-lab}"
DRY_RUN="${DRY_RUN:-0}"
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
echo "==> deleting kind cluster ${CLUSTER}"
run kind delete cluster --name "${CLUSTER}"
