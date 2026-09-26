#!/usr/bin/env bash
# kind + llm-d Router in standalone mode (EPP + Envoy sidecar) + 3 llm-d-inference-sim replicas.
# Runs on a laptop with Docker (CPU only). DRY_RUN=1 prints every command without running it.
# Versions are pinned by default and marked (verify) in the README; override with env vars.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-igw-lab}"
NAMESPACE="${NAMESPACE:-default}"
RELEASE="${RELEASE:-igw}"                                    # names the InferencePool and the EPP Service (igw-epp)
GAIE_VERSION="${GAIE_VERSION:-v1.6.2}"                       # (verify) Gateway API Inference Extension: InferencePool v1 CRD
ROUTER_VERSION="${ROUTER_VERSION:-v0.10.0}"                  # (verify) llm-d-router: InferenceObjective CRD + EPP image
ROUTER_CHART="${ROUTER_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}"
ROUTER_CHART_VERSION="${ROUTER_CHART_VERSION:-${ROUTER_VERSION}}"   # (verify) llm-d guides use the floating channel "v0"
LOCAL_PORT="${LOCAL_PORT:-8081}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

step "1/6 checking prerequisites (docker, kind, kubectl, helm)"
if [[ "${DRY_RUN}" != "1" ]]; then
  for bin in docker kind kubectl helm; do
    command -v "${bin}" >/dev/null || { echo "${bin} not found; rerun with DRY_RUN=1 to see the steps"; exit 1; }
  done
fi

step "2/6 creating kind cluster '${CLUSTER}' (skipped if it exists)"
if [[ "${DRY_RUN}" != "1" ]] && kind get clusters 2>/dev/null | grep -qx "${CLUSTER}"; then
  echo "cluster ${CLUSTER} already exists"
else
  run kind create cluster --name "${CLUSTER}" --config "${HERE}/kind-config.yaml"
fi
run kubectl config use-context "kind-${CLUSTER}"

step "3/6 installing CRDs: InferencePool (GIE ${GAIE_VERSION}) and InferenceObjective (llm-d-router ${ROUTER_VERSION})"
run kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"
run kubectl apply -f "https://github.com/llm-d/llm-d-router/releases/download/${ROUTER_VERSION}/manifests.yaml"

step "4/6 deploying 3 simulator replicas"
run kubectl apply -n "${NAMESPACE}" -f "${HERE}/sim-deployment.yaml"
run kubectl rollout status -n "${NAMESPACE}" deployment/vllm-sim --timeout=180s

step "5/6 installing the llm-d Router (standalone chart ${ROUTER_CHART_VERSION}) with the lab's EndpointPickerConfig"
run helm upgrade --install "${RELEASE}" "${ROUTER_CHART}" --version "${ROUTER_CHART_VERSION}" \
  -n "${NAMESPACE}" -f "${HERE}/router-values.yaml" --wait --timeout 5m

step "6/6 done"
run kubectl get inferencepools,inferenceobjectives,pods -n "${NAMESPACE}"
echo
echo "Reach the router:  kubectl port-forward -n ${NAMESPACE} svc/${RELEASE}-epp ${LOCAL_PORT}:8081"
echo "Then:              ROUTER_URL=http://localhost:${LOCAL_PORT} ${HERE}/../local/smoke.sh"
echo "Benchmark:         see notebooks/04_local_stack_with_llm_d.ipynb (it detects the port-forward)"
echo "Tear down:         ${HERE}/down.sh"
