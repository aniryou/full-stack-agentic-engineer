#!/usr/bin/env bash
# Serve a target with a draft model for speculative decoding (T1, a 24 GB card) — notebook 04's GPU path.
#
#   ./serve_with_draft.sh                                    # Qwen/Qwen3-4B with the off-the-shelf Qwen/Qwen3-0.6B, k = 4
#   DRAFT=_run_outputs/draft-distilled ./serve_with_draft.sh # a draft SFT'd on the target's own outputs
#   DRAFT=none ./serve_with_draft.sh                         # the baseline, no speculation
#   DRY_RUN=1 ./serve_with_draft.sh                          # print the commands only
#
# The draft must have the target's vocab_size (vLLM's draft_model check). The --speculative-config fields are
# vLLM v0.30.0's (method, model, num_speculative_tokens, draft_sample_method; verify when you move versions).
# Env (defaults in brackets): TARGET [Qwen/Qwen3-4B], DRAFT [Qwen/Qwen3-0.6B], K [4], SAMPLE [probabilistic|greedy;
# vLLM's default is greedy], PORT [8000], MAX_MODEL_LEN [8192], GPU_MEM_UTIL [0.90], IMAGE [vllm/vllm-openai:v0.30.0], MODE [auto].
set -euo pipefail

TARGET="${TARGET:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-Qwen/Qwen3-0.6B}"
K="${K:-4}"
SAMPLE="${SAMPLE:-probabilistic}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
MODE="${MODE:-auto}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/3 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
else
  echo "   no NVIDIA GPU visible: run notebook 04 at T0 (acceptance from the tiny models)"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

FLAGS=(--port "${PORT}" --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}")
if [[ "${DRAFT}" != "none" ]]; then
  SPEC="{\"method\": \"draft_model\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${K}, \"draft_sample_method\": \"${SAMPLE}\"}"
  FLAGS+=(--speculative-config "${SPEC}")
  echo ">> 2/3 speculative config: ${SPEC}"
else
  echo ">> 2/3 no draft: the baseline"
fi

if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi
echo ">> 3/3 start vLLM (${MODE}); then read the counters before and after a load (notebook 04, Exercise 4.1):"
echo "   curl -s localhost:${PORT}/metrics | grep spec_decode"
if [[ "${MODE}" == "docker" ]]; then
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" -v "$(pwd)/_run_outputs:/workspace/_run_outputs" \
    "${IMAGE}" "${TARGET}" "${FLAGS[@]}"
else
  run vllm serve "${TARGET}" "${FLAGS[@]}"
fi
