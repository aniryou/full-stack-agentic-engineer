#!/usr/bin/env bash
# Serve a thinking model with vLLM's OpenAI-compatible server and a reasoning parser (T1).
#
#   ./serve.sh                                         # Qwen/Qwen3-0.6B on :8000, parser qwen3
#   MODEL=Qwen/Qwen3-1.7B ./serve.sh
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ./serve.sh     # parser deepseek_r1 (picked automatically)
#   MODEL=Qwen/Qwen3-4B MAX_MODEL_LEN=16384 ./serve.sh              # a 24 GB card (L4, RTX 4090)
#   THINKING_DEFAULT=off ./serve.sh                    # Qwen3 hybrid: thinking off unless a request turns it on
#   DRY_RUN=1 ./serve.sh                               # print the commands only
#
# Env (defaults in brackets): MODEL [Qwen/Qwen3-0.6B], PARSER [auto: deepseek_r1 for R1 distills and
# Qwen3 *-Thinking-2507, qwen3 for Qwen3], PORT [8000], MODE [auto|docker|pip], IMAGE [vllm/vllm-openai:v0.30.0],
# DTYPE [auto; "half" below compute capability 8.0 (T4 has no bfloat16)], MAX_MODEL_LEN [8192],
# GPU_MEM_UTIL [0.92, vLLM v0.30.0's default; 0.85 on Colab], BUDGET_END_STR [the phrase vLLM forces before
# </think> when a request's thinking_token_budget runs out], THINKING_DEFAULT [on|off], API_KEY [unset;
# SET IT on any machine reachable from the internet], HF_TOKEN [unset], EXTRA_ARGS [unset].
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PARSER="${PARSER:-auto}"
PORT="${PORT:-8000}"
MODE="${MODE:-auto}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
BUDGET_END_STR="${BUDGET_END_STR:-I have to give the solution based on the reasoning directly now.</think>}"
THINKING_DEFAULT="${THINKING_DEFAULT:-on}"
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
      echo "   compute capability ${CC}: below the sm_75 minimum of vLLM v0.30.0's CUDA 13 builds (Kaggle: pick 'GPU T4 x2', not P100)"
      if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
    fi
    if [[ "${DTYPE}" == "auto" ]] && (( CC_NUM < 80 )); then
      DTYPE="half"
      echo "   compute capability ${CC} (e.g. T4): no bfloat16, using --dtype half (check Qwen3 outputs are sane in fp16)"
    fi
  fi
else
  echo "   no nvidia-smi: no NVIDIA GPU visible. The T0 stand-in is the fake server:"
  echo "   python -m thinklab fake --port ${PORT}"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

echo ">> 2/4 reasoning parser"
if [[ "${PARSER}" == "auto" ]]; then
  case "${MODEL}" in
    *DeepSeek-R1*|*QwQ*|*Thinking-2507*) PARSER="deepseek_r1" ;;   # the template opens <think> in the prompt
    *Qwen3*) PARSER="qwen3" ;;
    *gpt-oss*) PARSER="openai_gptoss" ;;
    *) echo "   cannot guess a parser for ${MODEL}; set PARSER (vllm/docs/features/reasoning_outputs.md lists them)"; exit 1 ;;
  esac
fi
echo "   --reasoning-parser ${PARSER}  (names use underscores in vLLM; SGLang spells them with hyphens)"

FLAGS=(--port "${PORT}" --dtype "${DTYPE}" --max-model-len "${MAX_MODEL_LEN}"
       --gpu-memory-utilization "${GPU_MEM_UTIL}" --reasoning-parser "${PARSER}" --enable-prompt-tokens-details)
if [[ "${PARSER}" == "qwen3" ]]; then
  # What vLLM forces when a request's thinking_token_budget runs out. Other parsers: vLLM derives the
  # boundary tokens from the parser (verify budgets on templates that open <think> in the prompt).
  FLAGS+=(--reasoning-config "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"${BUDGET_END_STR}\"}")
fi
if [[ "${THINKING_DEFAULT}" == "off" ]]; then FLAGS+=(--default-chat-template-kwargs '{"enable_thinking": false}'); fi
if [[ -n "${API_KEY}" ]]; then FLAGS+=(--api-key "${API_KEY}"); fi
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split into flags
if [[ -n "${EXTRA_ARGS}" ]]; then FLAGS+=(${EXTRA_ARGS}); fi

if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi

echo ">> 3/4 when GET /health returns 200, from the lab directory in another shell:"
echo "   THINKLAB_URL=http://127.0.0.1:${PORT} python -m thinklab ask 'What is 17 * 23 - 100?'"
echo "   THINKLAB_URL=http://127.0.0.1:${PORT} jupyter lab notebooks"

echo ">> 4/4 start vLLM (${MODE})"
if [[ "${MODE}" == "docker" ]]; then
  ENV_ARGS=()
  if [[ -n "${HF_TOKEN:-}" ]]; then ENV_ARGS=(-e HF_TOKEN); fi
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "${IMAGE}" "${MODEL}" "${FLAGS[@]}"
else
  if ! command -v vllm >/dev/null 2>&1; then
    echo "   vllm is not installed: pip install \"vllm==0.30.0\"  (pulls a CUDA build of torch, several GB)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
  run vllm serve "${MODEL}" "${FLAGS[@]}"
fi

