#!/usr/bin/env bash
# Deploy a quantized model with the SERVING LAB's GCP paths (no new infrastructure code here):
#   TARGET=cloud-run: 04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/deploy.sh (one L4, scale to zero)
#   TARGET=gke:       that lab's deploy/gcp/gke/vllm.yaml, applied, then its container args patched for the scheme
#
#   PROJECT_ID=my-project SCHEME=w4a16 ./deploy-quantized.sh
#   PROJECT_ID=my-project SCHEME=fp8-online KV_CACHE_DTYPE=fp8 ./deploy-quantized.sh
#   TARGET=gke SCHEME=fp8-online ./deploy-quantized.sh          # needs a cluster: the serving lab's cluster.sh
#   DRY_RUN=1 PROJECT_ID=p SCHEME=w4a16 ./deploy-quantized.sh   # print every command
#   PROJECT_ID=my-project ./deploy-quantized.sh delete
#
# Env: TARGET [cloud-run|gke], SCHEME [w4a16: bf16|fp8-online|w4a16], MODEL [per scheme], KV_CACHE_DTYPE [auto],
# MAX_MODEL_LEN [8192], GPU_MEM_UTIL [0.92], PROJECT_ID (cloud-run), DRY_RUN [0]. Both targets run L4s,
# so every scheme below runs natively there (python -m quantlab plan --gpu L4).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING="${SERVING_LAB:-$(cd "${HERE}/../../../../serving-engine/vllm-serving-lab" 2>/dev/null && pwd || echo "${HERE}/../../../../serving-engine/vllm-serving-lab")}"
TARGET="${TARGET:-cloud-run}"
SCHEME="${SCHEME:-w4a16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
DRY_RUN="${DRY_RUN:-0}"
export DRY_RUN

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

QUANT=()
case "${SCHEME}" in
  bf16)       DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct" ;;
  fp8-online) DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"; QUANT=(--quantization fp8_per_tensor) ;;
  w4a16)      DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct-AWQ" ;;
  *) echo "SCHEME=${SCHEME}: this script covers bf16|fp8-online|w4a16 (published or online); for your own"
     echo "checkpoint upload it to GCS and use cloud-run.quantized.tfvars.example (option C)"; exit 2 ;;
esac
MODEL="${MODEL:-${DEFAULT_MODEL}}"
if [[ "${KV_CACHE_DTYPE}" != "auto" ]]; then QUANT+=(--kv-cache-dtype "${KV_CACHE_DTYPE}"); fi

if [[ ! -d "${SERVING}/deploy/gcp" ]]; then
  echo "serving lab not found at ${SERVING} (set SERVING_LAB)"; exit 1
fi

if [[ "${TARGET}" == "cloud-run" ]]; then
  : "${PROJECT_ID:?set PROJECT_ID}"
  # The serving lab's script prints each step and honours DRY_RUN itself (exported above).
  if [[ "${1:-}" == "delete" ]]; then
    echo "+ ${SERVING}/deploy/gcp/cloud-run/deploy.sh delete"
    PROJECT_ID="${PROJECT_ID}" "${SERVING}/deploy/gcp/cloud-run/deploy.sh" delete
    exit 0
  fi
  echo ">> Cloud Run, one L4: ${SCHEME} (${MODEL})"
  EXTRA="$(IFS=,; echo "${QUANT[*]:-}")"
  echo "+ MODEL=${MODEL} EXTRA_ARGS=${EXTRA} ${SERVING}/deploy/gcp/cloud-run/deploy.sh"
  PROJECT_ID="${PROJECT_ID}" MODEL="${MODEL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    EXTRA_ARGS="${EXTRA}" "${SERVING}/deploy/gcp/cloud-run/deploy.sh"
elif [[ "${TARGET}" == "gke" ]]; then
  if [[ "${1:-}" == "delete" ]]; then
    run kubectl delete -f "${SERVING}/deploy/gcp/gke/vllm.yaml" -f "${SERVING}/deploy/gcp/gke/podmonitoring.yaml"
    exit 0
  fi
  echo ">> GKE (a cluster from the serving lab's cluster.sh): apply its manifests, then set the args for ${SCHEME}"
  ARGS="\"${MODEL}\",\"--port=8000\",\"--max-model-len=${MAX_MODEL_LEN}\",\"--gpu-memory-utilization=${GPU_MEM_UTIL}\""
  for ((i = 0; i < ${#QUANT[@]}; i += 2)); do ARGS="${ARGS},\"${QUANT[i]}=${QUANT[i + 1]}\""; done
  run kubectl apply -f "${SERVING}/deploy/gcp/gke/vllm.yaml" -f "${SERVING}/deploy/gcp/gke/podmonitoring.yaml"
  run kubectl patch deployment vllm --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":[${ARGS}]}]"
  run kubectl rollout status deployment/vllm --timeout=20m
else
  echo "TARGET must be cloud-run or gke"; exit 2
fi
