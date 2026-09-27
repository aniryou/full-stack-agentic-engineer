#!/usr/bin/env bash
# Train a student on one GPU with the lab's T1 trainers (distillab.hf) — T4-safe defaults.
#
#   METHOD=sft ./train_student.sh           # SeqKD: SFT of the student on the teacher's kept completions (TRL)
#   METHOD=kd  ./train_student.sh           # logit KD: the teacher in-process, chunked KL at T, alpha-mixed with CE
#   METHOD=gkd ./train_student.sh           # on-policy: TRL's GKDTrainer (lmbda 0.5, beta 0.5)
#   LORA_R=16 METHOD=kd ./train_student.sh  # LoRA adapters when memory is short
#   DRY_RUN=1 ./train_student.sh            # print the commands only
#
# Steps: 1/4 packages, 2/4 the memory plan for the exact batch and length step 4 uses, 3/4 teacher data through
# DISTILLAB_URL (./serve_teacher.sh; writes _run_outputs/teacher_{pc,msgs}.jsonl), 4/4 training. On one 16 GB card
# the teacher server holds ~90% of the memory, so it must be stopped between steps 3 and 4:
#   STAGE=data ./train_student.sh    # steps 1-3, with the teacher running
#   (stop vLLM)
#   STAGE=train ./train_student.sh   # steps 1, 2 and 4, on the data step 3 wrote
# STAGE=auto (the default) runs everything, but stops after step 3 with that instruction when the teacher is served
# on this machine and it has a single GPU. STAGE=all runs everything regardless (a teacher elsewhere, or a 2nd GPU).
# Env (defaults in brackets): METHOD [sft], STAGE [auto|data|train|all], TEACHER [Qwen/Qwen2.5-1.5B-Instruct],
# STUDENT [Qwen/Qwen2.5-0.5B-Instruct], PROBLEMS [2000], SAMPLES [4], GPU [T4], LORA_R [0],
# BATCH and MAX_LEN [the trainer's defaults: sft 4 x 1024, kd and gkd 2 x 512; the memory plan uses the same],
# INSTALL [0: set 1 to pip install the T1 stack first], PY [python3].
set -euo pipefail

METHOD="${METHOD:-sft}"
STAGE="${STAGE:-auto}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-1.5B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
PROBLEMS="${PROBLEMS:-2000}"
SAMPLES="${SAMPLES:-4}"
GPU="${GPU:-T4}"
LORA_R="${LORA_R:-0}"
INSTALL="${INSTALL:-0}"
PY="${PY:-python3}"
DRY_RUN="${DRY_RUN:-0}"
case "${METHOD}" in
  sft) BATCH="${BATCH:-4}"; MAX_LEN="${MAX_LEN:-1024}" ;;
  kd|gkd) BATCH="${BATCH:-2}"; MAX_LEN="${MAX_LEN:-512}" ;;
  *) echo "METHOD must be sft, kd or gkd"; exit 2 ;;
esac
case "${STAGE}" in
  auto|data|train|all) ;;
  *) echo "STAGE must be auto, data, train or all"; exit 2 ;;
esac

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}
lower() { basename "$1" | tr '[:upper:]' '[:lower:]'; }

cd "$(dirname "$0")/../.."

if [[ "${STAGE}" == "auto" ]]; then
  STAGE="all"
  NGPU="$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)"
  if [[ "${DISTILLAB_URL:-}" =~ ^https?://(127\.0\.0\.1|localhost)(:|/|$) && "${NGPU}" == "1" ]]; then
    STAGE="data-then-stop"
  fi
fi

echo ">> 1/4 packages"
if [[ "${INSTALL}" == "1" ]]; then
  run "${PY}" -m pip install -q "trl==1.14.0" "transformers>=4.56.2" peft datasets
fi
echo "   (T1 stack: trl 1.14.0, transformers >= 4.56.2, peft, datasets; vLLM runs in its own process: ./serve_teacher.sh)"

echo ">> 2/4 memory plan for ${METHOD} at batch ${BATCH} x ${MAX_LEN} tokens (predicted; verify on the card)"
REGIME="$([[ "${LORA_R}" != "0" ]] && echo lora || echo full)"
case "${METHOD}" in
  sft) run "${PY}" -m distillab memory --student "$(lower "${STUDENT}")" --gpu "${GPU}" --regime "${REGIME}" \
         --batch "${BATCH}" --seq "${MAX_LEN}" || echo "   (no bundled config for ${STUDENT}: plan skipped)" ;;
  kd)  run "${PY}" -m distillab memory --student "$(lower "${STUDENT}")" --teacher "$(lower "${TEACHER}")" --gpu "${GPU}" \
         --regime "${REGIME}" --batch "${BATCH}" --seq "${MAX_LEN}" --chunk 256 || echo "   (no bundled config for this pair: plan skipped)" ;;
  gkd) run "${PY}" -m distillab memory --student "$(lower "${STUDENT}")" --teacher "$(lower "${TEACHER}")" --gpu "${GPU}" \
         --regime "${REGIME}" --batch "${BATCH}" --seq "${MAX_LEN}" || echo "   (no bundled config for this pair: plan skipped)" ;;
esac

if [[ "${STAGE}" != "train" ]]; then
  echo ">> 3/4 teacher data (needs DISTILLAB_URL, else the fake teacher answers: fine for a dry run, not for training)"
  if [[ -z "${DISTILLAB_URL:-}" ]]; then
    echo "   DISTILLAB_URL is not set: start ./serve_teacher.sh and export DISTILLAB_URL=http://127.0.0.1:8000"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
  run "${PY}" -m distillab teacher-data --problems "${PROBLEMS}" -n "${SAMPLES}" --out _run_outputs/teacher
else
  echo ">> 3/4 teacher data: skipped (STAGE=train uses _run_outputs/teacher_{pc,msgs}.jsonl)"
  if [[ ! -s _run_outputs/teacher_pc.jsonl || ! -s _run_outputs/teacher_msgs.jsonl ]]; then
    echo "   no teacher data in _run_outputs/: run STAGE=data first (with the teacher served)"
    if [[ "${DRY_RUN}" != "1" ]]; then exit 1; fi
  fi
fi

if [[ "${STAGE}" == "data" || "${STAGE}" == "data-then-stop" ]]; then
  echo ">> 4/4 train: not yet. The teacher server holds most of this GPU; stop it (Ctrl-C, or docker stop), then:"
  echo "   STAGE=train METHOD=${METHOD} $0"
  exit 0
fi

echo ">> 4/4 train (${METHOD})"
case "${METHOD}" in
  sft) run "${PY}" -m distillab.hf.sft --model "${STUDENT}" --data _run_outputs/teacher_pc.jsonl \
         --out _run_outputs/student-sft --gpu "${GPU}" --lora-r "${LORA_R}" --batch-size "${BATCH}" --max-length "${MAX_LEN}" ;;
  kd)  run "${PY}" -m distillab.hf.kd --teacher "${TEACHER}" --student "${STUDENT}" --data _run_outputs/teacher_msgs.jsonl \
         --out _run_outputs/student-kd --gpu "${GPU}" --lora-r "${LORA_R}" --chunk 256 \
         --batch-size "${BATCH}" --max-length "${MAX_LEN}" ;;
  gkd) run "${PY}" -m distillab.hf.gkd --teacher "${TEACHER}" --student "${STUDENT}" --data _run_outputs/teacher_msgs.jsonl \
         --out _run_outputs/student-gkd --gpu "${GPU}" --batch-size "${BATCH}" --max-length "${MAX_LEN}" ;;
esac
