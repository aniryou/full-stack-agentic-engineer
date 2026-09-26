#!/usr/bin/env bash
# Remove everything install.sh created (the cluster itself is removed by `terraform destroy`):
# the HPA, Gateway/HTTPRoute, the EPP release, vLLM, the Custom Metrics Stackdriver Adapter and the
# project-level IAM binding install.sh granted it. That binding lives on the *project*, so it would
# survive `terraform destroy`; it is removed only when PROJECT_ID is set.
# The GPU node pool scales back to 0 once the vLLM pods are gone. DRY_RUN=1 prints the commands.
#   PROJECT_ID=... ./uninstall.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-}"
NAMESPACE="${NAMESPACE:-default}"
RELEASE="${RELEASE:-vllm-qwen}"
ADAPTER_SHA="${ADAPTER_SHA:-9500033f8a7d21b99843c2a8bd641a8a6a5685b4}"     # same pin as install.sh
ADAPTER_URL="${ADAPTER_URL:-https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/${ADAPTER_SHA}/custom-metrics-stackdriver-adapter/deploy/production/adapter_new_resource_model.yaml}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

echo "==> deleting HPA, Gateway/HTTPRoute, EPP release, vLLM"
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/hpa.yaml"
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/gateway.yaml"
run helm uninstall "${RELEASE}" -n "${NAMESPACE}" || echo "(release ${RELEASE} not installed)"
run kubectl delete -n "${NAMESPACE}" --ignore-not-found -f "${HERE}/vllm.yaml" -f "${HERE}/podmonitoring-vllm.yaml"

echo "==> deleting the Custom Metrics Stackdriver Adapter"
run kubectl delete --ignore-not-found -f "${ADAPTER_URL}"

echo "==> removing the adapter's project-level IAM binding (roles/monitoring.viewer)"
if [[ -n "${PROJECT_ID}" ]]; then
  PROJECT_NUMBER="$( [[ "${DRY_RUN}" == "1" ]] && echo PROJECT_NUMBER || gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
  run gcloud projects remove-iam-policy-binding "projects/${PROJECT_ID}" --role roles/monitoring.viewer --condition=None \
    --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/custom-metrics/sa/custom-metrics-stackdriver-adapter" \
    || echo "(binding not present)"
else
  echo "PROJECT_ID not set: skipped. It survives terraform destroy; rerun with PROJECT_ID=<id> to remove it."
fi
echo "the regional load balancer is deleted with the Gateway; run 'terraform destroy' to remove the cluster and VPC"
