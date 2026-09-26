#!/usr/bin/env bash
# Serve a quantized model with vLLM on any NVIDIA GPU (T1), checking the scheme against the GPU first.
#
#   SCHEME=w4a16 ./serve.sh                       # INT4 weight-only: every GPU from a T4 (the T4 speed lever)
#   SCHEME=fp8-online ./serve.sh                  # a bf16 checkpoint quantized to FP8 as it loads
#   SCHEME=fp8 MODEL=./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC ./serve.sh   # a checkpoint from ./compress.sh
#   SCHEME=fp8 KV_CACHE_DTYPE=fp8 ./serve.sh      # + FP8 KV cache (Ampere/Ada via FlashInfer, Hopper via FA3; never a T4)
#   DRY_RUN=1 GPU_CC=10.0 SCHEME=nvfp4 ./serve.sh # print the command for a GPU you do not have
#
# Env (defaults in brackets): SCHEME [w4a16: bf16|fp8|fp8-online|w4a16|w8a8-int8|nvfp4], MODEL [per scheme,
# see below], PORT [8000], MODE [auto|docker|pip], IMAGE [vllm/vllm-openai:v0.30.0], MAX_MODEL_LEN [4096],
# GPU_MEM_UTIL [0.92], KV_CACHE_DTYPE [auto], GPU_CC [from nvidia-smi], API_KEY [unset: set it on any
# public port], EXTRA_ARGS [unset], DRY_RUN [0]. The same rules, with the reasons, are
# `python -m quantlab plan --gpu <T4|L4|A100-80GB|H100-80GB|B200> --scheme <scheme>`.
set -euo pipefail

SCHEME="${SCHEME:-w4a16}"
PORT="${PORT:-8000}"
MODE="${MODE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

# Model and flags per scheme (quantlab.serve.SCHEMES; published ids verify, ./ paths come from ./compress.sh)
QUANT_FLAGS=()
case "${SCHEME}" in
  bf16)       DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct" ;;
  fp8)        DEFAULT_MODEL="./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC" ;;
  fp8-online) DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"; QUANT_FLAGS=(--quantization fp8_per_tensor) ;;
  w4a16)      DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct-AWQ" ;;
  w8a8-int8)  DEFAULT_MODEL="./Qwen2.5-1.5B-Instruct-W8A8-smoothquant-gptq" ;;
  nvfp4)      DEFAULT_MODEL="./Qwen2.5-1.5B-Instruct-NVFP4" ;;
  *) echo "unknown SCHEME=${SCHEME} (bf16|fp8|fp8-online|w4a16|w8a8-int8|nvfp4)"; exit 2 ;;
esac
MODEL="${MODEL:-${DEFAULT_MODEL}}"

echo ">> 1/3 which GPU is this"
CC="${GPU_CC:-}"
if [[ -z "${CC}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
fi
if [[ -z "${CC}" ]]; then
  echo "   no NVIDIA GPU visible. Offline, the fake server plays the part: python -m quantlab fake --scheme ${SCHEME}"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  CC="8.9"
  echo "   DRY_RUN: assuming compute capability ${CC} (an L4); set GPU_CC to plan for another GPU"
fi
SM="${CC//./}"

echo ">> 2/3 does ${SCHEME} run natively on sm_${SM}?"
FLAGS=(--port "${PORT}" --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}")
if (( SM < 75 )); then
  echo "   sm_${SM} is below vLLM v0.30.0's sm_75 floor (Kaggle P100 = 6.0, V100 = 7.0): use a T4 or newer"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi
if (( SM < 80 )); then
  FLAGS+=(--dtype half)
  echo "   no bfloat16 below sm_80 (T4): --dtype half"
fi
case "${SCHEME}" in
  fp8|fp8-online)
    if (( SM >= 89 )); then echo "   FP8 W8A8 on FP8 tensor cores (prefill and decode both faster)"
    else echo "   no FP8 tensor cores: vLLM runs FP8 weights through Marlin as W8A16 (memory win only)"; fi ;;
  w4a16)
    if (( SM == 90 )); then echo "   INT4 W4A16 via Machete (Hopper); decode faster, prefill not"
    else echo "   INT4 W4A16 via Marlin; decode faster, prefill not"; fi ;;
  w8a8-int8)
    if (( SM >= 100 )); then
      echo "   INT8 W8A8 is not supported on compute capability >= 10.0 (vLLM docs): use fp8 or nvfp4"
      if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
    else echo "   INT8 x INT8 tensor cores via CUTLASS"; fi ;;
  nvfp4)
    if (( SM >= 100 )); then echo "   NVFP4 W4A4 on FP4 tensor cores (needs CUDA >= 12.8)"
    else echo "   below sm_100 NVFP4 runs weight-only (Marlin): memory win only"; fi ;;
  bf16) echo "   the baseline" ;;
esac
if [[ "${KV_CACHE_DTYPE}" != "auto" ]]; then
  if (( SM < 80 )); then
    echo "   --kv-cache-dtype ${KV_CACHE_DTYPE}: no attention backend on sm_${SM} reads an FP8 KV cache (Triton needs sm_89, FlashInfer/FA sm_80)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  elif (( SM < 90 )); then
    echo "   --kv-cache-dtype ${KV_CACHE_DTYPE}: FlashAttention 2 has no FP8 KV, vLLM switches to FlashInfer"
  fi
  FLAGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${API_KEY}" ]]; then FLAGS+=(--api-key "${API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then FLAGS+=(${EXTRA_ARGS}); fi

echo ">> 3/3 start vLLM; ready when GET /health returns 200. Check the log for the kernel:"
echo "   'Using MarlinLinearKernel for ...' / 'Selected CutlassFP8ScaledMMLinearKernel for ...' / 'Using FLASHINFER attention backend ...'"
if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi
if [[ "${MODE}" == "docker" ]]; then
  VOLS=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
  MODEL_ARG="${MODEL}"
  if [[ "${MODEL}" == ./* || "${MODEL}" == /* ]]; then       # a local checkpoint: mount it
    VOLS+=(-v "$(cd "$(dirname "${MODEL}")" 2>/dev/null && pwd || echo "$(dirname "${MODEL}")"):/ckpt")
    MODEL_ARG="/ckpt/$(basename "${MODEL}")"
  fi
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" "${VOLS[@]}" \
    "${IMAGE}" "${MODEL_ARG}" ${QUANT_FLAGS[@]+"${QUANT_FLAGS[@]}"} "${FLAGS[@]}"
else
  if ! command -v vllm >/dev/null 2>&1; then
    echo "   vllm is not installed: pip install \"vllm==0.30.0\"  (its own environment; pulls a CUDA build of torch)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
  run vllm serve "${MODEL}" ${QUANT_FLAGS[@]+"${QUANT_FLAGS[@]}"} "${FLAGS[@]}"
fi
