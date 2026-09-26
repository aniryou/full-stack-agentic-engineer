#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server on any machine with an NVIDIA GPU (T1).
#
#   ./serve.sh                                   # docker if a daemon is reachable, else `vllm serve` from pip
#   MODE=pip MODEL=Qwen/Qwen2.5-1.5B-Instruct ./serve.sh
#   EXTRA_ARGS="--max-num-seqs 64 --kv-cache-dtype fp8" ./serve.sh
#   DRY_RUN=1 ./serve.sh                         # print the command only
#
# Env (defaults in brackets): MODEL [Qwen/Qwen2.5-0.5B-Instruct], PORT [8000], MODE [auto|docker|pip],
# IMAGE [vllm/vllm-openai:v0.30.0], DTYPE [auto; becomes "half" on a T4, which has no bfloat16],
# MAX_MODEL_LEN [4096], GPU_MEM_UTIL [0.90], API_KEY [unset; SET IT on any machine reachable from the
# internet], HF_TOKEN [unset; needed for gated models], EXTRA_ARGS [unset].
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
MODE="${MODE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/3 what is here"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  if [[ "${DTYPE}" == "auto" ]] && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "T4"; then
    DTYPE="half"
    echo "   T4 detected (compute capability 7.5, no bfloat16): using --dtype half"
  fi
else
  echo "   no nvidia-smi: this machine has no NVIDIA GPU visible. Use the fake server instead:"
  echo "   python -m servelab fake --port ${PORT}"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi

FLAGS=(--port "${PORT}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}")
if [[ -n "${API_KEY}" ]]; then FLAGS+=(--api-key "${API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then FLAGS+=(${EXTRA_ARGS}); fi

echo ">> 2/3 predicted capacity (edit the model/GPU names to match; offline arithmetic)"
echo "   python -m servelab size --model <bundled-config> --gpu <T4|L4|RTX4090|...> --max-model-len ${MAX_MODEL_LEN}"

echo ">> 3/3 start vLLM (${MODE}); ready when GET /health returns 200"
if [[ "${MODE}" == "docker" ]]; then
  ENV_ARGS=()
  if [[ -n "${HF_TOKEN:-}" ]]; then ENV_ARGS=(-e HF_TOKEN); fi
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "${IMAGE}" "${MODEL}" "${FLAGS[@]}"
else
  if ! command -v vllm >/dev/null 2>&1; then
    echo "   vllm is not installed: pip install \"vllm>=0.30\"  (pulls a CUDA build of torch, several GB)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
  run vllm serve "${MODEL}" "${FLAGS[@]}"
fi
