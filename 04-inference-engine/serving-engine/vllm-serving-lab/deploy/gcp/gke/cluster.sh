#!/usr/bin/env bash
# A small zonal GKE Standard cluster for one vLLM replica on an L4 Spot node, plus the manifests.
#
#   PROJECT_ID=my-project ./cluster.sh            # create cluster + L4 Spot pool (0..2), deploy vLLM
#   DRY_RUN=1 PROJECT_ID=my-project ./cluster.sh  # print the commands only
#   PROJECT_ID=my-project ./cluster.sh delete     # delete the cluster (and everything on it)
#
# Env: ZONE [us-central1-a], CLUSTER [vllm-lab], MACHINE [g2-standard-8], MAX_NODES [2], HF_TOKEN [unset].
# A Terraform version of the same cluster (with more options) lives in layer 03's lab and layer 05's lab.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER="${CLUSTER:-vllm-lab}"
MACHINE="${MACHINE:-g2-standard-8}"
MAX_NODES="${MAX_NODES:-2}"
DRY_RUN="${DRY_RUN:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

if [[ "${1:-}" == "delete" ]]; then
  echo ">> deleting cluster ${CLUSTER}"
  run gcloud container clusters delete "${CLUSTER}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet
  exit 0
fi

echo ">> 1/5 enable the GKE API"
run gcloud services enable container.googleapis.com --project="${PROJECT_ID}"

echo ">> 2/5 zonal Standard cluster: one small CPU node for system pods, managed Prometheus on"
run gcloud container clusters create "${CLUSTER}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 --num-nodes=1 \
  --enable-managed-prometheus \
  --labels=app=vllm-serving-lab

echo ">> 3/5 L4 Spot node pool that scales from zero; GKE installs the default NVIDIA driver"
run gcloud container node-pools create l4-spot \
  --project="${PROJECT_ID}" --zone="${ZONE}" --cluster="${CLUSTER}" \
  --machine-type="${MACHINE}" \
  --accelerator="type=nvidia-l4,count=1,gpu-driver-version=default" \
  --spot \
  --enable-autoscaling --num-nodes=0 --min-nodes=0 --max-nodes="${MAX_NODES}"

echo ">> 4/5 credentials, optional HF token, the Deployment/Service and the PodMonitoring"
run gcloud container clusters get-credentials "${CLUSTER}" --project="${PROJECT_ID}" --zone="${ZONE}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "+ kubectl create secret generic hf-token --from-literal=token=***"
  if [[ "${DRY_RUN}" != "1" ]]; then
    kubectl create secret generic hf-token --from-literal=token="${HF_TOKEN}" --dry-run=client -o yaml | kubectl apply -f -
  fi
fi
run kubectl apply -f "${HERE}/vllm.yaml" -f "${HERE}/podmonitoring.yaml"

echo ">> 5/5 wait for the pod (a Spot L4 node must be created first: a few minutes, then model load)"
run kubectl rollout status deployment/vllm --timeout=30m

cat <<EOF

Measure it from your laptop:
  kubectl port-forward svc/vllm 8000:8000 &
  python -m servelab bench --url http://127.0.0.1:8000 --rate 4 -n 100 --slo-ttft-ms 1000 --slo-tpot-ms 60
  python -m servelab metrics --url http://127.0.0.1:8000 --window 30
Clean up (the L4 pool scales to zero on its own when the Deployment is deleted; the cluster does not):
  kubectl delete -f ${HERE}/vllm.yaml -f ${HERE}/podmonitoring.yaml
  PROJECT_ID=${PROJECT_ID} $0 delete
EOF
