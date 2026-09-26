#!/usr/bin/env bash
# vLLM on Cloud Run with one NVIDIA L4 — the `gcloud run deploy` equivalent of ./terraform.
#
#   PROJECT_ID=my-project ./deploy.sh              # deploy, then smoke-test /v1/models
#   DRY_RUN=1 PROJECT_ID=my-project ./deploy.sh    # print every command, run nothing
#   PROJECT_ID=my-project ./deploy.sh delete       # clean up
#
# Env (defaults in brackets): REGION [us-central1], SERVICE [vllm-l4], IMAGE [vllm/vllm-openai:v0.30.0],
# MODEL [Qwen/Qwen2.5-1.5B-Instruct], MAX_MODEL_LEN [8192], GPU_MEM_UTIL [0.92], CONCURRENCY [32],
# MAX_INSTANCES [1], HF_SECRET [unset: name of a Secret Manager secret holding an HF token],
# EXTRA_ARGS [unset: more vllm flags, comma-separated, e.g. "--max-num-seqs,64"].
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-vllm-l4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
HF_SECRET="${HF_SECRET:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

if [[ "${1:-}" == "delete" ]]; then
  echo ">> deleting ${SERVICE} in ${REGION}"
  run gcloud run services delete "${SERVICE}" --project="${PROJECT_ID}" --region="${REGION}" --quiet
  exit 0
fi

echo ">> 1/3 enable the Cloud Run and Secret Manager APIs"
run gcloud services enable run.googleapis.com secretmanager.googleapis.com --project="${PROJECT_ID}"

# `vllm serve` arguments (the image's ENTRYPOINT is `vllm serve`), comma-separated for --args.
VLLM_ARGS="${MODEL},--served-model-name,${MODEL},--port,8000,--max-model-len,${MAX_MODEL_LEN},--gpu-memory-utilization,${GPU_MEM_UTIL}"
if [[ -n "${EXTRA_ARGS}" ]]; then VLLM_ARGS="${VLLM_ARGS},${EXTRA_ARGS}"; fi

SECRET_FLAGS=()
if [[ -n "${HF_SECRET}" ]]; then SECRET_FLAGS=(--set-secrets="HF_TOKEN=${HF_SECRET}:latest"); fi

echo ">> 2/3 deploy ${SERVICE}: ${IMAGE} serving ${MODEL} on one L4, scale 0..${MAX_INSTANCES}"
# GPU flags are GA since June 2025 (verify the flag names against `gcloud run deploy --help`):
# one L4, no zonal redundancy (cheaper, needs less quota), CPU always allocated (--no-cpu-throttling).
run gcloud run deploy "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --args="${VLLM_ARGS}" \
  --port=8000 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu=8 \
  --memory=32Gi \
  --no-cpu-throttling \
  --concurrency="${CONCURRENCY}" \
  --min-instances=0 \
  --max-instances="${MAX_INSTANCES}" \
  --timeout=600 \
  --startup-probe="httpGet.path=/health,httpGet.port=8000,periodSeconds=10,timeoutSeconds=5,failureThreshold=60" \
  --no-allow-unauthenticated \
  --labels="app=vllm-serving-lab" \
  ${SECRET_FLAGS[@]+"${SECRET_FLAGS[@]}"}

echo ">> 3/3 smoke test (the first request after a scale-to-zero pays the cold start)"
if [[ "${DRY_RUN}" == "1" ]]; then
  URL="https://${SERVICE}-<project-number>.${REGION}.run.app"
else
  URL="$(gcloud run services describe "${SERVICE}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
  curl -sS --max-time 900 -H "Authorization: Bearer $(gcloud auth print-identity-token)" "${URL}/v1/models"
  echo
fi

cat <<EOF

Service URL: ${URL}
Benchmark it with the lab's load generator:
  SERVELAB_BEARER=\$(gcloud auth print-identity-token) python -m servelab bench --url ${URL} \\
      --rate 2 -n 60 --input-len 512 --output-len 128 --slo-ttft-ms 1000 --slo-tpot-ms 60
Engine metrics (the same parser as locally):
  SERVELAB_BEARER=\$(gcloud auth print-identity-token) python -m servelab metrics --url ${URL}
Clean up:
  PROJECT_ID=${PROJECT_ID} REGION=${REGION} $0 delete
EOF
