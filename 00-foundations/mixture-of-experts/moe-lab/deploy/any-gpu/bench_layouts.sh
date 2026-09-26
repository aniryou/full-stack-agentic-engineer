#!/usr/bin/env bash
# Benchmark one MoE in each two-GPU layout (TP, TP+EP, DP+EP) with `vllm bench serve` (T2).
# Kaggle "GPU T4 x2", a rented pair, or a node of the l4x2 GKE pool. Each layout takes both GPUs,
# so the servers run one after another; results land in ${OUT}/bench_<layout>_c<N>.txt and
# `python -m moelab bench-parse FILE` (or notebook 04) reads them.
#
#   ./bench_layouts.sh
#   LAYOUTS="tp tp_ep" CONCURRENCY="1 8 32" ./bench_layouts.sh
#   DRY_RUN=1 ./bench_layouts.sh                            # print every command, run nothing
#
# Env (defaults in brackets): MODEL [allenai/OLMoE-1B-7B-0924-Instruct], LAYOUTS [tp tp_ep dp_ep],
# CONCURRENCY [1 16], INPUT_LEN [256], OUTPUT_LEN [128], PORT [8000], OUT [./results],
# MAX_MODEL_LEN [4096], TIMEOUT [1200 s for a server to become healthy].
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
LAYOUTS="${LAYOUTS:-tp tp_ep dp_ep}"
CONCURRENCY="${CONCURRENCY:-1 16}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
PORT="${PORT:-8000}"
OUT="${OUT:-$PWD/results}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TIMEOUT="${TIMEOUT:-1200}"
DRY_RUN="${DRY_RUN:-0}"
URL="http://127.0.0.1:${PORT}"

wait_healthy() { # poll /health until it answers or the server process dies
  echo "+ wait for ${URL}/health (timeout ${TIMEOUT}s)"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local deadline=$((SECONDS + TIMEOUT))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$1" 2> /dev/null; then
      echo "== the server exited; see ${OUT}/vllm-${2}.log"
      return 1
    fi
    if curl -sf "${URL}/health" > /dev/null; then return 0; fi
    sleep 5
  done
  echo "== not healthy after ${TIMEOUT}s"
  return 1
}

if [[ "${DRY_RUN}" != "1" ]]; then mkdir -p "${OUT}"; fi
for layout in ${LAYOUTS}; do
  echo ">> layout ${layout}"
  log="${OUT}/vllm-${layout}.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_RUN=1 LAYOUT="${layout}" MODEL="${MODEL}" PORT="${PORT}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" MODE=pip \
      "${HERE}/serve_moe.sh" | grep '^+' || true
    pid=0
  else
    LAYOUT="${layout}" MODEL="${MODEL}" PORT="${PORT}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" MODE=pip \
      "${HERE}/serve_moe.sh" > "${log}" 2>&1 &
    pid=$!
  fi
  if ! wait_healthy "${pid}" "${layout}"; then
    kill "${pid}" 2> /dev/null || true
    continue
  fi
  for c in ${CONCURRENCY}; do
    res="${OUT}/bench_${layout}_c${c}.txt"
    echo "+ vllm bench serve ... --max-concurrency ${c} > ${res}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      vllm bench serve --base-url "${URL}" --model "${MODEL}" --dataset-name random \
        --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "$((c > 1 ? 8 * c : 16))" --max-concurrency "${c}" --ignore-eos | tee "${res}"
    fi
  done
  if [[ "${DRY_RUN}" != "1" ]]; then
    kill "${pid}" 2> /dev/null || true
    wait "${pid}" 2> /dev/null || true
    sleep 10 # let the GPUs' memory free before the next layout
  fi
done
echo ">> done: python -m moelab bench-parse ${OUT}/bench_tp_ep_c16.txt  (or notebook 04, Exercise 4.5)"
