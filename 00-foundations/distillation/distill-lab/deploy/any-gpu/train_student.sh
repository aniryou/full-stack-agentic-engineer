#!/usr/bin/env bash
# Train a student on one GPU with the lab's T1 trainers (distillab.hf) — T4-safe defaults.
#
#   METHOD=sft ./train_student.sh           # SeqKD: SFT of the student on the teacher's kept completions (TRL)
#   METHOD=kd  ./train_student.sh           # logit KD: the teacher in-process, chunked KL at T, alpha-mixed with CE
#   METHOD=gkd ./train_student.sh           # on-policy: TRL's GKDTrainer (lmbda 0.5, beta 0.5)
#   LORA_R=16 METHOD=kd ./train_student.sh  # LoRA adapters when memory is short
#   DRY_RUN=1 ./train_student.sh            # print the commands only
#
# Step 1 needs a teacher served at DISTILLAB_URL (./serve_teacher.sh) and writes _run_outputs/teacher_{pc,msgs}.jsonl;
# stop that server before step 3 on a single 16 GB card. Env (defaults in brackets): METHOD [sft], TEACHER
# [Qwen/Qwen2.5-1.5B-Instruct], STUDENT [Qwen/Qwen2.5-0.5B-Instruct], PROBLEMS [2000], SAMPLES [4], GPU [T4],
# LORA_R [0], INSTALL [0: set 1 to pip install the T1 stack first], PY [python3].
set -euo pipefail

METHOD="${METHOD:-sft}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-1.5B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
PROBLEMS="${PROBLEMS:-2000}"
SAMPLES="${SAMPLES:-4}"
GPU="${GPU:-T4}"
LORA_R="${LORA_R:-0}"
INSTALL="${INSTALL:-0}"
PY="${PY:-python3}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

cd "$(dirname "$0")/../.."

echo ">> 1/4 packages"
if [[ "${INSTALL}" == "1" ]]; then
  run "${PY}" -m pip install -q "trl==1.14.0" "transformers>=4.56.2" peft datasets
fi
echo "   (T1 stack: trl 1.14.0, transformers >= 4.56.2, peft, datasets; vLLM runs in its own process: ./serve_teacher.sh)"

echo ">> 2/4 memory plan (predicted; verify on the card)"
if [[ "${METHOD}" == "sft" ]]; then
  run "${PY}" -m distillab memory --student "$(basename "${STUDENT}" | tr '[:upper:]' '[:lower:]')" --gpu "${GPU}" \
    --regime "$([[ "${LORA_R}" != "0" ]] && echo lora || echo full)" || echo "   (no bundled config for ${STUDENT}: plan skipped)"
else
  run "${PY}" -m distillab memory --student "$(basename "${STUDENT}" | tr '[:upper:]' '[:lower:]')" \
    --teacher "$(basename "${TEACHER}" | tr '[:upper:]' '[:lower:]')" --gpu "${GPU}" \
    --regime "$([[ "${LORA_R}" != "0" ]] && echo lora || echo full)" --chunk 256 || echo "   (no bundled config for this pair: plan skipped)"
fi

echo ">> 3/4 teacher data (needs DISTILLAB_URL, else the fake teacher answers: fine for a dry run, not for training)"
if [[ -z "${DISTILLAB_URL:-}" ]]; then
  echo "   DISTILLAB_URL is not set: start ./serve_teacher.sh and export DISTILLAB_URL=http://127.0.0.1:8000"
  if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
fi
run "${PY}" -m distillab teacher-data --problems "${PROBLEMS}" -n "${SAMPLES}" --out _run_outputs/teacher

echo ">> 4/4 train (${METHOD})"
case "${METHOD}" in
  sft) run "${PY}" -m distillab.hf.sft --model "${STUDENT}" --data _run_outputs/teacher_pc.jsonl \
         --out _run_outputs/student-sft --gpu "${GPU}" --lora-r "${LORA_R}" ;;
  kd)  run "${PY}" -m distillab.hf.kd --teacher "${TEACHER}" --student "${STUDENT}" --data _run_outputs/teacher_msgs.jsonl \
         --out _run_outputs/student-kd --gpu "${GPU}" --lora-r "${LORA_R}" --chunk 256 ;;
  gkd) run "${PY}" -m distillab.hf.gkd --teacher "${TEACHER}" --student "${STUDENT}" --data _run_outputs/teacher_msgs.jsonl \
         --out _run_outputs/student-gkd --gpu "${GPU}" ;;
  *) echo "METHOD must be sft, kd or gkd"; exit 2 ;;
esac
