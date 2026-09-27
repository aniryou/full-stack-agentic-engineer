#!/usr/bin/env bash
# Serve a target with a draft model for speculative decoding (T1, a 24 GB card) — notebook 04's GPU path.
#
#   ./serve_with_draft.sh                                    # Qwen/Qwen3-4B with the off-the-shelf Qwen/Qwen3-0.6B, k = 4
#   DRAFT=_run_outputs/draft-distilled ./serve_with_draft.sh # a local draft: Qwen3-0.6B SFT'd on the target's outputs
#                                                            # (notebook 04: teacher-data --keep all, then hf.sft)
#   DRAFT=none ./serve_with_draft.sh                         # the baseline, no speculation
#   DRY_RUN=1 ./serve_with_draft.sh                          # print the commands only
#
# A DRAFT that is a local directory is found from where you run the script, else from the lab directory, and
# passed by absolute path; in docker mode it is mounted read-only at /drafts/<name> and passed by that path (the
# image's working directory is /vllm-workspace, so a relative path would not resolve there). Anything else is a
# Hugging Face repo id. The draft must have the target's vocab_size (vLLM's draft_model check). The
# --speculative-config fields are vLLM v0.30.0's (method, model, num_speculative_tokens, draft_sample_method;
# verify when you move versions). Env (defaults in brackets): TARGET [Qwen/Qwen3-4B], DRAFT [Qwen/Qwen3-0.6B], K [4],
# SAMPLE [probabilistic|greedy; vLLM's default is greedy], PORT [8000], MAX_MODEL_LEN [8192], GPU_MEM_UTIL [0.90],
# IMAGE [vllm/vllm-openai:v0.30.0], MODE [auto].
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

CALLER_DIR="$(pwd)"
cd "$(dirname "$0")/../.."                                 # the lab directory

echo ">> 1/3 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
else
  echo "   no NVIDIA GPU visible: run notebook 04 at T0 (acceptance from the tiny models)"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

if [[ "${MODE}" == "auto" ]]; then
  if docker info >/dev/null 2>&1; then MODE="docker"; else MODE="pip"; fi
fi

LOCAL_DRAFT=""
if [[ "${DRAFT}" != "none" ]]; then
  for base in "${CALLER_DIR}" "$(pwd)"; do
    cand="${DRAFT}"
    [[ "${cand}" == /* ]] || cand="${base}/${DRAFT}"
    if [[ -d "${cand}" ]]; then LOCAL_DRAFT="$(cd "${cand}" && pwd)"; break; fi
  done
  if [[ -z "${LOCAL_DRAFT}" && ( "${DRAFT}" == /* || "${DRAFT}" == ./* || "${DRAFT}" == ../* || "${DRAFT}" == _run_outputs/* ) ]]; then
    echo "   no local draft at ${DRAFT} (vLLM would look it up as a Hugging Face repo id and fail); train it first:"
    echo "   python -m distillab.hf.sft --model Qwen/Qwen3-0.6B --data _run_outputs/target_pc.jsonl --out ${DRAFT}"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
fi

FLAGS=(--port "${PORT}" --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}")
MOUNTS=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
if [[ "${DRAFT}" != "none" ]]; then
  MODEL_ARG="${DRAFT}"
  if [[ -n "${LOCAL_DRAFT}" ]]; then
    MODEL_ARG="${LOCAL_DRAFT}"
    if [[ "${MODE}" == "docker" ]]; then
      MODEL_ARG="/drafts/$(basename "${LOCAL_DRAFT}")"
      MOUNTS+=(-v "${LOCAL_DRAFT}:${MODEL_ARG}:ro")
    fi
  fi
  SPEC="{\"method\": \"draft_model\", \"model\": \"${MODEL_ARG}\", \"num_speculative_tokens\": ${K}, \"draft_sample_method\": \"${SAMPLE}\"}"
  FLAGS+=(--speculative-config "${SPEC}")
  echo ">> 2/3 speculative config: ${SPEC}"
else
  echo ">> 2/3 no draft: the baseline"
fi

echo ">> 3/3 start vLLM (${MODE}); then read the counters before and after a load (notebook 04, Exercise 4.1):"
echo "   curl -s localhost:${PORT}/metrics | grep spec_decode"
if [[ "${MODE}" == "docker" ]]; then
  run docker run --rm --gpus all --ipc=host -p "${PORT}:${PORT}" "${MOUNTS[@]}" "${IMAGE}" "${TARGET}" "${FLAGS[@]}"
else
  run vllm serve "${TARGET}" "${FLAGS[@]}"
fi
