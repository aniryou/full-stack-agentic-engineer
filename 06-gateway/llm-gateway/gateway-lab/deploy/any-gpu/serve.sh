#!/usr/bin/env bash
# T1: one real vLLM behind the lab gateway, with a fake provider as the fallback -- all on this GPU machine.
#
#   ./serve.sh                     # vllm serve Qwen/Qwen2.5-0.5B-Instruct as `lab/llm` on :8000, fake acme on :8101,
#                                  # the gateway (config vllm) on :8080
#   DRY_RUN=1 ./serve.sh           # print every command, start nothing
#
# Env: MODEL [Qwen/Qwen2.5-0.5B-Instruct], VLLM_PORT [8000], GATEWAY_PORT [8080], FAKE_PORT [8101],
# MAX_MODEL_LEN [4096], DTYPE [auto; half on a T4], VLLM_API_KEY [unset], STATE_DIR [./.gwlab-gpu], EXTRA_ARGS [unset].
# Needs `vllm` on PATH: pip install "vllm==0.30.0" (verify; pulls a CUDA build of torch, several GB).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(cd "${HERE}/../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
FAKE_PORT="${FAKE_PORT:-8101}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
STATE_DIR="${STATE_DIR:-${HERE}/.gwlab-gpu}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo; echo "==> $*"; }
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
wait_for() {   # url, seconds
  if [[ "${DRY_RUN}" == "1" ]]; then echo "+ curl -fsS $1   (retried for up to $2 s)"; return; fi
  for _ in $(seq 1 "$2"); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "$1 did not answer; see ${STATE_DIR}/*.log"; exit 1
}

step "1/4 the GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  if [[ "${DTYPE}" == "auto" ]] && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "T4"; then
    DTYPE="half"; echo "T4 (compute capability 7.5, no bfloat16): --dtype half"
  fi
elif [[ "${DRY_RUN}" != "1" ]]; then
  echo "no nvidia-smi: no NVIDIA GPU here. The T0 path needs none: every notebook runs on the fake providers."
  exit 1
fi

step "2/4 vllm serve ${MODEL} as lab/llm on :${VLLM_PORT}"
command -v vllm >/dev/null 2>&1 || { echo "vllm not found: pip install \"vllm==0.30.0\" (verify), or use docker-compose.yaml"; [[ "${DRY_RUN}" == "1" ]] || exit 1; }
run mkdir -p "${STATE_DIR}"
flags=(--served-model-name lab/llm --port "${VLLM_PORT}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}"
       --enable-prompt-tokens-details)
if [[ -n "${VLLM_API_KEY:-}" ]]; then flags+=(--api-key "${VLLM_API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then flags+=(${EXTRA_ARGS}); fi
echo "+ vllm serve ${MODEL} ${flags[*]} > ${STATE_DIR}/vllm.log 2>&1 &"
if [[ "${DRY_RUN}" != "1" ]]; then
  nohup vllm serve "${MODEL}" "${flags[@]}" > "${STATE_DIR}/vllm.log" 2>&1 &
  echo $! >> "${STATE_DIR}/pids"
fi
wait_for "http://127.0.0.1:${VLLM_PORT}/health" 900          # first start downloads ~1 GB of weights

step "3/4 the fallback: a fake OpenAI-dialect provider on :${FAKE_PORT} (simulated)"
echo "+ (cd ${LAB} && python3 -m gwlab fake --name acme --port ${FAKE_PORT} > ${STATE_DIR}/acme.log 2>&1 &)"
if [[ "${DRY_RUN}" != "1" ]]; then
  (cd "${LAB}" && nohup python3 -m gwlab fake --name acme --port "${FAKE_PORT}" > "${STATE_DIR}/acme.log" 2>&1 & echo $! >> "${STATE_DIR}/pids")
fi
wait_for "http://127.0.0.1:${FAKE_PORT}/health" 30

step "4/4 the gateway (config vllm) on :${GATEWAY_PORT}"
echo "+ GWLAB_VLLM_URL=http://127.0.0.1:${VLLM_PORT} ACME_URL=http://127.0.0.1:${FAKE_PORT} python3 -m gwlab gateway --config vllm --port ${GATEWAY_PORT} &"
if [[ "${DRY_RUN}" != "1" ]]; then
  (cd "${LAB}" && GWLAB_VLLM_URL="http://127.0.0.1:${VLLM_PORT}" ACME_URL="http://127.0.0.1:${FAKE_PORT}" \
     nohup python3 -m gwlab gateway --config vllm --port "${GATEWAY_PORT}" > "${STATE_DIR}/gateway.log" 2>&1 & echo $! >> "${STATE_DIR}/pids")
fi
wait_for "http://127.0.0.1:${GATEWAY_PORT}/health" 30

echo
echo "export GWLAB_VLLM_URL=http://127.0.0.1:${VLLM_PORT}     # notebooks 01-04 then run their T1 cells"
echo "python3 -m gwlab key --url http://127.0.0.1:${GATEWAY_PORT} --tenant team-a   # a virtual key for curl"
echo "stop: ${HERE}/down.sh"
