#!/usr/bin/env bash
# Serve a teacher with vLLM's OpenAI-compatible server, set up for distillation (T1).
#
#   ./serve_teacher.sh                                              # Qwen/Qwen2.5-1.5B-Instruct on :8000
#   MODEL=Qwen/Qwen3-1.7B ./serve_teacher.sh                        # a thinking teacher (parser qwen3, picked automatically)
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ./serve_teacher.sh   # parser deepseek_r1
#   MODEL=Qwen/Qwen3-4B MAX_MODEL_LEN=16384 ./serve_teacher.sh      # a 24 GB card (L4, RTX 4090)
#   DRY_RUN=1 ./serve_teacher.sh                                    # print the commands only
#
# What makes it a *teacher* server: --max-logprobs (how many top log-probabilities a request may ask for; vLLM
# v0.30.0 defaults to 20 and rejects more), and a reasoning parser for thinking teachers so traces arrive in
# message.reasoning. Env (defaults in brackets): MODEL [Qwen/Qwen2.5-1.5B-Instruct], PORT [8000],
# MODE [auto|docker|pip], IMAGE [vllm/vllm-openai:v0.30.0], DTYPE [auto; "half" below compute capability 8.0],
# MAX_MODEL_LEN [4096], GPU_MEM_UTIL [0.90], MAX_LOGPROBS [20], PARSER [auto], API_KEY [unset; SET IT on any
# machine reachable from the internet], EXTRA_ARGS [unset].
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
MODE="${MODE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_LOGPROBS="${MAX_LOGPROBS:-20}"
PARSER="${PARSER:-auto}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'      # shell-quoted: the printed line can be pasted as is
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/4 what is here"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  CC_NUM="${CC//./}"
  if [[ -n "${CC_NUM}" && "${CC_NUM}" =~ ^[0-9]+$ ]]; then
    if (( CC_NUM < 75 )); then
      echo "   compute capability ${CC}: below vLLM v0.30.0's sm_75 minimum (Kaggle: pick 'GPU T4 x2', not P100)"
      if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
    fi
    if [[ "${DTYPE}" == "auto" ]] && (( CC_NUM < 80 )); then
      DTYPE="half"
      echo "   compute capability ${CC} (T4): no bfloat16, using --dtype half (check the teacher's outputs are sane in fp16)"
    fi
  fi
else
  echo "   no nvidia-smi: no NVIDIA GPU visible. The T0 stand-in is the fake teacher:"
  echo "   python -m distillab fake --port ${PORT}"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

echo ">> 2/4 reasoning parser"
if [[ "${PARSER}" == "auto" ]]; then
  case "${MODEL}" in
    *DeepSeek-R1*|*Thinking-2507*) PARSER="deepseek_r1" ;;
    *Qwen3*) PARSER="qwen3" ;;
    *) PARSER="none" ;;
  esac
fi
echo "   parser: ${PARSER}"

FLAGS=(--port "${PORT}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}"
       --gpu-memory-utilization "${GPU_MEM_UTIL}" --max-logprobs "${MAX_LOGPROBS}")
if [[ "${PARSER}" != "none" ]]; then FLAGS+=(--reasoning-parser "${PARSER}"); fi
if [[ -n "${API_KEY}" ]]; then FLAGS+=(--api-key "${API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then FLAGS+=(${EXTRA_ARGS}); fi

if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi

echo ">> 3/4 when GET /health returns 200, from the lab directory in another shell:"
echo "   export DISTILLAB_URL=http://127.0.0.1:${PORT}"
echo "   python -m distillab teacher-data --problems 2000 -n 4 --out _run_outputs/teacher"

echo ">> 4/4 start vLLM (${MODE})"
if [[ "${MODE}" == "docker" ]]; then
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" "${IMAGE}" "${MODEL}" "${FLAGS[@]}"
else
  if ! command -v vllm >/dev/null 2>&1; then
    echo "   vllm is not installed: pip install \"vllm==0.30.0\"  (pulls a CUDA build of torch, several GB)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
  run vllm serve "${MODEL}" "${FLAGS[@]}"
fi
