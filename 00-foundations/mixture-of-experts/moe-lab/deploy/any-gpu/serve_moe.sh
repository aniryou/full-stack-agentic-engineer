#!/usr/bin/env bash
# Serve a small MoE with vLLM on whatever NVIDIA GPUs this machine has (T1 one GPU, T2 two GPUs).
#
#   ./serve_moe.sh                                          # OLMoE-1B-7B on one GPU (offloads experts on a 16 GB card)
#   LAYOUT=tp_ep ./serve_moe.sh                             # 2 GPUs: --tensor-parallel-size 2 --enable-expert-parallel
#   LAYOUT=dp_ep ./serve_moe.sh                             # 2 GPUs: --data-parallel-size 2 --enable-expert-parallel
#   MODEL=ibm-granite/granite-3.0-3b-a800m-instruct ./serve_moe.sh
#   ROUTED=1 ./serve_moe.sh                                 # return per-token expert ids (notebook 02)
#   DRY_RUN=1 LAYOUT=tp ./serve_moe.sh                      # print the command only
#
# Env (defaults in brackets): MODEL [allenai/OLMoE-1B-7B-0924-Instruct], PORT [8000], MODE [auto|docker|pip],
# IMAGE [vllm/vllm-openai:v0.30.0], LAYOUT [single|tp|tp_ep|dp_ep], DTYPE [auto; "half" below compute
# capability 8.0 (T4: no bf16)], MAX_MODEL_LEN [4096], GPU_MEM_UTIL [0.92, v0.30.0's default],
# OFFLOAD_GB [auto: 3 on a single GPU with < 20 GB when MODEL is the default OLMoE, else 0],
# ROUTED [0], EPLB [0], API_KEY [unset; SET IT on any machine reachable from the internet],
# HF_TOKEN [unset], EXTRA_ARGS [unset].
set -euo pipefail

MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
PORT="${PORT:-8000}"
MODE="${MODE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
LAYOUT="${LAYOUT:-single}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
OFFLOAD_GB="${OFFLOAD_GB:-auto}"
ROUTED="${ROUTED:-0}"
EPLB="${EPLB:-0}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

launch() { # print, then replace this shell with the server (so its PID is the server's: easy to stop)
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then exec "$@"; fi
}

fail() { # stop, unless this is a dry run
  echo "   $*"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
}

echo ">> 1/4 what is here"
N_GPUS=0
MEM_MIB=0
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
  N_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | wc -l | tr -d ' ')"
  MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null | head -n1 | tr -d ' ')"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n1 | tr -d ' ')"
  CC_NUM="${CC//./}"
  if [[ -n "${CC_NUM}" && "${CC_NUM}" =~ ^[0-9]+$ ]]; then
    if ((CC_NUM < 75)); then
      fail "compute capability ${CC}: below vLLM v0.30.0's sm_75 minimum (Kaggle: pick 'GPU T4 x2', not P100)"
    fi
    if [[ "${DTYPE}" == "auto" ]] && ((CC_NUM < 80)); then
      DTYPE="half"
      echo "   compute capability ${CC} (T4): no bfloat16 -> --dtype half; MXFP4 models (gpt-oss) will not run here"
    fi
  fi
else
  fail "no nvidia-smi: no NVIDIA GPU visible. Use the notebooks' T0 paths instead (python -m moelab fit ...)"
fi

echo ">> 2/4 layout ${LAYOUT}"
FLAGS=(--port "${PORT}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}")
case "${LAYOUT}" in
  single) NEED=1 ;;
  tp)
    NEED=2
    FLAGS+=(--tensor-parallel-size 2)
    ;;
  tp_ep)
    NEED=2
    FLAGS+=(--tensor-parallel-size 2 --enable-expert-parallel)
    ;;
  dp_ep)
    NEED=2
    FLAGS+=(--data-parallel-size 2 --enable-expert-parallel)
    ;;
  *)
    echo "   LAYOUT must be single, tp, tp_ep or dp_ep"
    exit 2
    ;;
esac
if ((N_GPUS < NEED)); then
  fail "layout ${LAYOUT} needs ${NEED} GPUs, this machine shows ${N_GPUS}"
fi
echo "   EP size = TP x DP when --enable-expert-parallel is set; otherwise the experts are tensor-parallel"

if [[ "${OFFLOAD_GB}" == "auto" ]]; then
  OFFLOAD_GB=0
  if [[ "${LAYOUT}" == "single" && "${MODEL}" == "allenai/OLMoE-1B-7B-0924-Instruct" && "${MEM_MIB:-0}" =~ ^[0-9]+$ ]] &&
    ((MEM_MIB > 0 && MEM_MIB < 20000)); then
    # 13.8 GB of 16-bit weights leave no KV room on 16 GB; 3 GiB is the smallest offload that holds
    # 4 x 4096 tokens of KV (python -m moelab fit --gpu T4). Each extra GiB adds ~89 ms per step at
    # ~12 GB/s (PCIe Gen3, verify) and ~8K tokens of KV: raise it only for more sequences.
    OFFLOAD_GB=3
  fi
fi
if [[ "${OFFLOAD_GB}" != "0" ]]; then
  echo "   offloading ${OFFLOAD_GB} GiB of expert weights to pinned CPU memory: read over PCIe in EVERY step"
  FLAGS+=(--cpu-offload-gb "${OFFLOAD_GB}" --cpu-offload-params experts)
fi
if [[ "${ROUTED}" == "1" ]]; then FLAGS+=(--enable-return-routed-experts); fi
if [[ "${EPLB}" == "1" ]]; then FLAGS+=(--enable-eplb); fi
if [[ -n "${API_KEY}" ]]; then FLAGS+=(--api-key "${API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then FLAGS+=(${EXTRA_ARGS}); fi

echo ">> 3/4 predicted fit (offline arithmetic, simulated)"
echo "   python -m moelab fit --model olmoe-1b-7b --gpu <T4|L4|RTX4090>"

if [[ "${MODE}" == "auto" ]]; then
  if docker info > /dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi
echo ">> 4/4 start vLLM (${MODE}); ready when GET /health returns 200. Look for 'Using default MoE config' in the log"
if [[ "${MODE}" == "docker" ]]; then
  ENV_ARGS=()
  if [[ -n "${HF_TOKEN:-}" ]]; then ENV_ARGS=(-e HF_TOKEN); fi
  launch docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "${IMAGE}" "${MODEL}" "${FLAGS[@]}"
else
  if ! command -v vllm > /dev/null 2>&1; then
    fail "vllm is not installed: pip install \"vllm==0.30.0\" (pulls a CUDA build of torch, several GB)"
  fi
  launch vllm serve "${MODEL}" "${FLAGS[@]}"
fi
