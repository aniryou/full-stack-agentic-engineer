#!/usr/bin/env bash
# Remove everything install.sh created (the cluster itself is removed by `terraform destroy`).
# The GPU node pool scales back to 0 once the vLLM pods are gone. DRY_RUN=1 prints the commands.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"
RELEASE="${RELEASE:-vllm-qwen}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

echo "==> deleting HPA, Gateway/HTTPRoute, EPP release, vLLM"
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/hpa.yaml"
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/gateway.yaml"
run helm uninstall "${RELEASE}" -n "${NAMESPACE}" --ignore-not-found
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/vllm.yaml" -f "${HERE}/podmonitoring-vllm.yaml"
echo "the regional load balancer is deleted with the Gateway; run 'terraform destroy' to remove the cluster and VPC"
