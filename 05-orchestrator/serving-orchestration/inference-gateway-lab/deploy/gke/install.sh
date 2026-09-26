#!/usr/bin/env bash
# GKE Inference Gateway, end to end, on the cluster from ../gcp/terraform:
#   vLLM on an L4 Spot node -> InferencePool + llm-d EPP (Helm) -> Gateway + HTTPRoute
#   -> HPA on vLLM's waiting and running requests.
# Usage: PROJECT_ID=... ZONE=us-central1-a [CLUSTER=igw-lab] ./install.sh
# DRY_RUN=1 prints every command without running it. Costs money while the GPU node runs (see README).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER="${CLUSTER:-igw-lab}"
NAMESPACE="${NAMESPACE:-default}"
RELEASE="${RELEASE:-vllm-qwen}"                          # = InferencePool name used by gateway.yaml
GAIE_VERSION="${GAIE_VERSION:-v1.6.2}"                   # (verify) only used if GKE does not manage the CRD
ROUTER_VERSION="${ROUTER_VERSION:-v0.10.0}"              # (verify)
ROUTER_CHART="${ROUTER_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-gateway}"
ROUTER_CHART_VERSION="${ROUTER_CHART_VERSION:-${ROUTER_VERSION}}"
# Custom Metrics Stackdriver Adapter, pinned to a k8s-stackdriver commit (master on 2026-09-26; image
# custom-metrics-stackdriver-adapter:v0.16.11-gke.0). VERIFY against current GKE docs before relying on it.
ADAPTER_SHA="${ADAPTER_SHA:-9500033f8a7d21b99843c2a8bd641a8a6a5685b4}"
ADAPTER_URL="${ADAPTER_URL:-https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/${ADAPTER_SHA}/custom-metrics-stackdriver-adapter/deploy/production/adapter_new_resource_model.yaml}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

step "1/7 kubectl credentials for ${CLUSTER} (${ZONE})"
run gcloud container clusters get-credentials "${CLUSTER}" --zone "${ZONE}" --project "${PROJECT_ID}"

step "2/7 CRDs: InferencePool (GKE-managed on >= 1.34.0-gke.1626000, else GIE ${GAIE_VERSION}) and InferenceObjective"
if [[ "${DRY_RUN}" == "1" ]] || ! kubectl get crd inferencepools.inference.networking.k8s.io >/dev/null 2>&1; then
  run kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"
fi
run kubectl apply -f "https://github.com/llm-d/llm-d-router/releases/download/${ROUTER_VERSION}/manifests.yaml"

step "3/7 vLLM Deployment + PodMonitoring (a GPU node is created on demand; first start takes minutes)"
run kubectl apply -n "${NAMESPACE}" -f "${HERE}/vllm.yaml" -f "${HERE}/podmonitoring-vllm.yaml"
run kubectl rollout status -n "${NAMESPACE}" deployment/vllm-qwen --timeout=20m

step "4/7 llm-d EPP + InferencePool + InferenceObjectives (Helm, gateway chart, provider gke)"
run helm upgrade --install "${RELEASE}" "${ROUTER_CHART}" --version "${ROUTER_CHART_VERSION}" \
  -n "${NAMESPACE}" -f "${HERE}/epp-values.yaml" --wait --timeout 10m

step "5/7 Gateway (regional external ALB) + HTTPRoute -> InferencePool"
run kubectl apply -n "${NAMESPACE}" -f "${HERE}/gateway.yaml"
run kubectl wait -n "${NAMESPACE}" --for=condition=Programmed gateway/inference-gateway --timeout=15m

step "6/7 Custom Metrics Stackdriver Adapter (serves GMP metrics to the HPA) + HPA"
run kubectl apply -f "${ADAPTER_URL}"
PROJECT_NUMBER="$( [[ "${DRY_RUN}" == "1" ]] && echo PROJECT_NUMBER || gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
run gcloud projects add-iam-policy-binding "projects/${PROJECT_ID}" --role roles/monitoring.viewer --condition=None \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/custom-metrics/sa/custom-metrics-stackdriver-adapter"
run kubectl apply -n "${NAMESPACE}" -f "${HERE}/hpa.yaml"

step "7/7 try it"
IP="$( [[ "${DRY_RUN}" == "1" ]] && echo GATEWAY_IP || kubectl get gateway/inference-gateway -n "${NAMESPACE}" -o jsonpath='{.status.addresses[0].value}')"
echo "curl -sS http://${IP}/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "  -H 'x-llm-d-inference-objective: premium' \\"
echo "  -d '{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32}'"
echo "watch the HPA:  kubectl get hpa vllm-qwen -n ${NAMESPACE} -w"
echo "tear down:      PROJECT_ID=${PROJECT_ID} ${HERE}/uninstall.sh && (cd ${HERE}/../gcp/terraform && terraform destroy)"
