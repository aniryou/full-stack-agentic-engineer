#!/usr/bin/env bash
# Produce a quantized checkpoint with llm-compressor on a GPU box, in its own virtualenv (T1).
# The recipe is rendered by quantlab (the same one notebook 01 prints), then run.
#
#   SCHEME=FP8_DYNAMIC ./compress.sh                   # no calibration data; minutes on a T4/L4
#   SCHEME=W4A16 ALGO=gptq ./compress.sh               # GPTQ, 256 x 1,024-token calibration samples
#   SCHEME=W8A8 ALGO=smoothquant,gptq ./compress.sh    # INT8 W8A8 (not for Blackwell)
#   SCHEME=NVFP4 ./compress.sh                         # 20 calibration samples (activation global scales)
#   DRY_RUN=1 ./compress.sh                            # print the steps and the recipe only
#
# Env: MODEL [Qwen/Qwen2.5-1.5B-Instruct], SCHEME [FP8_DYNAMIC], ALGO [empty = round to nearest],
# VENV [.venv-llmcompressor], LLMC_VERSION [0.14.0], DRY_RUN [0]. llm-compressor and vLLM pin different
# torch builds: keep them in separate environments (vLLM docs). Gated models need HF_TOKEN in the env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(cd "${HERE}/../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SCHEME="${SCHEME:-FP8_DYNAMIC}"
ALGO="${ALGO:-}"
VENV="${VENV:-.venv-llmcompressor}"
LLMC_VERSION="${LLMC_VERSION:-0.14.0}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/3 a separate environment for llm-compressor ${LLMC_VERSION}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "   no GPU visible: FP8_DYNAMIC can run on a CPU for a 0.5-1.5B model but slowly (verify); GPTQ/AWQ want a GPU"
fi
run python3 -m venv "${VENV}"
run "${VENV}/bin/pip" install -q "llmcompressor==${LLMC_VERSION}" -e "${LAB}"

echo ">> 2/3 the recipe (quantlab.compress.llmcompressor_script)"
RECIPE="recipe_${SCHEME}${ALGO:+_${ALGO//,/_}}.py"
if [[ "${DRY_RUN}" == "1" ]]; then
  python3 -m quantlab compress --model "${MODEL}" --scheme "${SCHEME}" --algo "${ALGO}" --print 2>/dev/null \
    || echo "   (install quantlab to print the recipe: pip install -e ${LAB})"
else
  "${VENV}/bin/python" -m quantlab compress --model "${MODEL}" --scheme "${SCHEME}" --algo "${ALGO}" --print > "${RECIPE}"
  cat "${RECIPE}"
fi

echo ">> 3/3 run it; the checkpoint lands in the current directory, ready for: SCHEME=... MODEL=./<dir> ./serve.sh"
run "${VENV}/bin/python" "${RECIPE}"
