#!/usr/bin/env bash
# One GRPO step with vLLM generating the rollouts (T1; notebook 05's GPU path).
#
#   ./rl_step.sh                         # Qwen/Qwen2.5-0.5B-Instruct, 8 prompts x 8 rollouts
#   INSTALL=1 ./rl_step.sh               # pip install vllm==0.30.0 and transformers first
#   MODEL=Qwen/Qwen3-0.6B G=4 ./rl_step.sh
#   DRY_RUN=1 ./rl_step.sh               # print the commands only
#
# Memory on a 15 GB T4 (verify on yours): vLLM takes gpu_memory_utilization=0.3 of the card; the trainer
# copy is float32 with plain SGD (no optimizer state) and gradient checkpointing.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROMPTS="${PROMPTS:-8}"
G="${G:-8}"
INSTALL="${INSTALL:-0}"
DRY_RUN="${DRY_RUN:-0}"
PY="${PY:-python3}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'      # shell-quoted: the printed line can be pasted as is
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/3 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
else
  echo "   no NVIDIA GPU visible: run notebook 05 at T0 instead (the bookkeeping on the tiny transformer)"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi

echo ">> 2/3 packages"
if [[ "${INSTALL}" == "1" ]]; then
  run "${PY}" -m pip install -q "vllm==0.30.0" "transformers>=4.56.2"
fi
echo "   (optional, for TRL's GRPOTrainer: pip install trl==1.14.0 — its [vllm] extra pins vllm>=0.20.0,<=0.30.0)"

echo ">> 3/3 one GRPO step: rollouts (vLLM) -> verifier -> advantages -> trainer log-probs -> one SGD step"
cd "$(dirname "$0")/../.."
run "${PY}" -m thinklab rl-step --model "${MODEL}" --prompts "${PROMPTS}" -g "${G}"
