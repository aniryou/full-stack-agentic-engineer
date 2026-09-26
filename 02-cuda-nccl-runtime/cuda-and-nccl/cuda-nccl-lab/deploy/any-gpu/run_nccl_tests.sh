#!/usr/bin/env bash
# Build NVIDIA/nccl-tests against the NCCL you already have, then sweep collectives on the local GPUs.
# Works on a VM or a container with the CUDA toolkit (nvcc): RunPod/Vast/Lambda, Kaggle, Colab, GCP.
#
#   bash run_nccl_tests.sh                                  # every local GPU, 8 B .. 256 MB
#   NGPUS=2 MAX_BYTES=1G OPS="all_reduce all_gather" bash run_nccl_tests.sh
#   DRY_RUN=1 bash run_nccl_tests.sh                        # print the steps, run nothing
#
# NCCL is taken from $NCCL_HOME, else the system (libnccl-dev: /usr/include/nccl.h), else the pip wheel
# that PyTorch installs (nvidia-nccl-cu12) — so the test uses the same NCCL your framework does.
# Logs land in $OUT (default ./out, gitignored); parse them with:  python -m gpurt.nccltests out/all_reduce_2gpu.log
# The nccl-tests checkout lives outside the repository ($SRC, default ~/.cache/nccl-tests-<ref>): it is a git
# clone of its own and must not end up inside this mono-repo.
set -euo pipefail

OUT="${OUT:-$PWD/out}"
# VERIFY: v2.20.0 is the latest nccl-tests tag as of 2026-09-26; gpurt.nccltests and gpurt.dist.busbw.plan
# mirror its output layout and buffer sizing. Override with NCCL_TESTS_REF=<tag>.
REF="${NCCL_TESTS_REF:-v2.20.0}"
SRC="${SRC:-${XDG_CACHE_HOME:-$HOME/.cache}/nccl-tests-${REF}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MIN_BYTES="${MIN_BYTES:-8}"
MAX_BYTES="${MAX_BYTES:-256M}"
OPS="${OPS:-all_reduce all_gather}"

step() {
  echo "+ $*" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  "$@"
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: no NVIDIA driver visible here" >&2; exit 1; }
  [[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "nvcc not found under ${CUDA_HOME}: install the CUDA toolkit or use Dockerfile.nccl-tests" >&2; exit 1; }
fi

NGPUS="${NGPUS:-$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)}"
NGPUS="${NGPUS:-1}"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d '.' || true)"
GENCODE="${NVCC_GENCODE:-${CC:+-gencode=arch=compute_${CC},code=sm_${CC}}}" # build for this GPU only: faster
echo "== ${NGPUS} GPU(s), compute capability ${CC:-unknown}; logs -> ${OUT}" >&2

# -- find NCCL -----------------------------------------------------------------------------------
if [[ -z "${NCCL_HOME:-}" && ! -f /usr/include/nccl.h ]]; then
  WHEEL="$(python3 -c 'import nvidia.nccl as m; print(list(m.__path__)[0])' 2> /dev/null || true)"
  if [[ -n "${WHEEL}" && -f "${WHEEL}/include/nccl.h" ]]; then
    NCCL_HOME="${SRC}-nccl" # the wheel ships libnccl.so.2 only; give the linker a libnccl.so (beside, not in, the clone)
    step mkdir -p "${NCCL_HOME}/lib"
    step ln -sfn "${WHEEL}/include" "${NCCL_HOME}/include"
    step ln -sf "${WHEEL}/lib/libnccl.so.2" "${NCCL_HOME}/lib/libnccl.so"
    step ln -sf "${WHEEL}/lib/libnccl.so.2" "${NCCL_HOME}/lib/libnccl.so.2"
    echo "== using NCCL from the pip wheel at ${WHEEL}" >&2
  else
    echo "no NCCL found: set NCCL_HOME, install libnccl-dev, or pip install nvidia-nccl-cu12" >&2
    [[ "${DRY_RUN:-0}" == "1" ]] || exit 1
  fi
fi
if [[ -n "${NCCL_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

# -- build ----------------------------------------------------------------------------------------
if [[ ! -d "${SRC}/.git" ]]; then
  step mkdir -p "$(dirname "${SRC}")"
  step git clone --depth 1 --branch "${REF}" https://github.com/NVIDIA/nccl-tests.git "${SRC}"
fi
step make -C "${SRC}" -j"$(nproc)" MPI=0 CUDA_HOME="${CUDA_HOME}" ${NCCL_HOME:+NCCL_HOME="${NCCL_HOME}"} \
  ${GENCODE:+NVCC_GENCODE="${GENCODE}"}

# -- run ------------------------------------------------------------------------------------------
step mkdir -p "${OUT}"
step bash -c "nvidia-smi topo -m | tee '${OUT}/topo.txt'"
for op in ${OPS}; do
  log="${OUT}/${op}_${NGPUS}gpu.log"
  step bash -c "NCCL_DEBUG=\${NCCL_DEBUG:-WARN} '${SRC}/build/${op}_perf' -b ${MIN_BYTES} -e ${MAX_BYTES} -f 2 -g ${NGPUS} | tee '${log}'"
done
echo "== done; next: python -m gpurt.nccltests ${OUT}/all_reduce_${NGPUS}gpu.log" >&2
