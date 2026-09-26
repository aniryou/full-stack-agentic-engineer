#!/usr/bin/env bash
# T1/T2: start N real vLLM replicas on this machine's GPU(s) for the lab router to route across.
#
#   ./serve.sh                                     # 2 x Qwen/Qwen2.5-0.5B-Instruct on :8001 and :8002
#   REPLICAS=3 MODEL=Qwen/Qwen2.5-1.5B-Instruct ./serve.sh
#   DRY_RUN=1 ./serve.sh                           # print every command, start nothing
#
# Replica i runs on GPU (i mod #GPUs); replicas that share a GPU split --gpu-memory-utilization
# (0.85 in total per GPU), so one T4 holds two 0.5B replicas and Kaggle's 2 x T4 holds one per GPU.
# Every replica serves the model under the name `lab/llm` (what the lab's bench sends) with the flags
# the lab's measurements need: --enable-prompt-tokens-details (per-request cached tokens) and an
# explicit --max-num-seqs (the batch slots the HPA targets are derived from).
# Env: REPLICAS [2], MODEL [Qwen/Qwen2.5-0.5B-Instruct], BASE_PORT [8001], MAX_MODEL_LEN [4096],
# MAX_NUM_SEQS [16], DTYPE [auto; half on a T4], GPU_MEM_TOTAL [0.85], STATE_DIR [./.igw-gpu],
# EXTRA_ARGS [unset]. Needs `vllm` on PATH: pip install "vllm==0.30.0" (verify; pulls CUDA torch).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLICAS="${REPLICAS:-2}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_PORT="${BASE_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
DTYPE="${DTYPE:-auto}"
GPU_MEM_TOTAL="${GPU_MEM_TOTAL:-0.85}"
STATE_DIR="${STATE_DIR:-${HERE}/.igw-gpu}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }

step "1/4 GPUs on this machine"
GPUS=1
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  GPUS="$(nvidia-smi -L | grep -c '^GPU' || true)"
  if [[ "${DTYPE}" == "auto" ]] && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "T4"; then
    DTYPE="half"; echo "T4 (compute capability 7.5, no bfloat16): --dtype half"
  fi
elif [[ "${DRY_RUN}" != "1" ]]; then
  echo "no nvidia-smi: no NVIDIA GPU visible. The T0 path needs none: notebooks 01-04 run on fake backends."
  exit 1
fi
[[ "${GPUS}" -ge 1 ]] || GPUS=1
PER_GPU=$(( (REPLICAS + GPUS - 1) / GPUS ))
MEM="$(awk -v t="${GPU_MEM_TOTAL}" -v n="${PER_GPU}" 'BEGIN { printf "%.2f", t / n }')"
echo "${REPLICAS} replica(s) on ${GPUS} GPU(s): up to ${PER_GPU} per GPU, --gpu-memory-utilization ${MEM} each"

step "2/4 vllm on PATH"
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found: pip install \"vllm==0.30.0\"  (verify; several GB). Docker instead: see docker-compose.yaml"
  [[ "${DRY_RUN}" == "1" ]] || exit 1
fi

step "3/4 starting the replicas one at a time (each must be healthy before the next claims GPU memory)"
run mkdir -p "${STATE_DIR}"
BACKENDS=()
for (( i = 0; i < REPLICAS; i++ )); do
  port=$(( BASE_PORT + i ))
  gpu=$(( i % GPUS ))
  flags=(--served-model-name lab/llm --port "${port}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}"
         --max-num-seqs "${MAX_NUM_SEQS}" --gpu-memory-utilization "${MEM}" --enable-prompt-tokens-details)
  # shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
  if [[ -n "${EXTRA_ARGS}" ]]; then flags+=(${EXTRA_ARGS}); fi
  echo "+ CUDA_VISIBLE_DEVICES=${gpu} vllm serve ${MODEL} ${flags[*]} > ${STATE_DIR}/r${i}.log 2>&1 &"
  if [[ "${DRY_RUN}" != "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" nohup vllm serve "${MODEL}" "${flags[@]}" > "${STATE_DIR}/r${i}.log" 2>&1 &
    echo $! >> "${STATE_DIR}/pids"
    for _ in $(seq 1 180); do                       # first start downloads the weights: allow 15 minutes
      if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then echo "r${i} healthy on :${port}"; break; fi
      sleep 5
    done
    curl -fsS "http://127.0.0.1:${port}/health" >/dev/null || { echo "r${i} did not start; see ${STATE_DIR}/r${i}.log"; exit 1; }
  else
    echo "+ curl -fsS http://127.0.0.1:${port}/health   (retried for up to 15 minutes)"
  fi
  BACKENDS+=("r${i}=http://127.0.0.1:${port}")
done

step "4/4 route across them"
LIST="$(IFS=,; echo "${BACKENDS[*]}")"
echo "export IGW_BACKENDS=${LIST}        # notebook 04 then runs its T1 cells against these replicas"
echo "python3 -m igwlab.router --config default-weighted --port 9000 $(printf -- '--backend %s ' "${BACKENDS[@]}")"
echo "stop: ${HERE}/down.sh"
