#!/usr/bin/env bash
# Build the gpubench image and run the suite on this host's GPUs; the report lands in ./results.
#
#   deploy/any-gpu/run.sh                        # the default (quick) suite
#   deploy/any-gpu/run.sh --full                 # extra arguments go to `gpubench run`
#   DRY_RUN=1 deploy/any-gpu/run.sh              # print the commands, run nothing
#   BASE=pytorch/pytorch:<tag> deploy/any-gpu/run.sh   # a different CUDA/PyTorch base image
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-gpubench:local}"
BASE="${BASE:-pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime}"
RESULTS="${RESULTS:-$LAB/results}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo "==> $*"; }
run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

step "1/4 check Docker and the NVIDIA runtime"
if [ "$DRY_RUN" != "1" ]; then
  command -v docker >/dev/null 2>&1 || {
    echo "docker not found — use the plain-pip path in deploy/any-gpu/README.md instead" >&2
    exit 1
  }
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "warning: Docker reports no NVIDIA runtime; --gpus needs the NVIDIA Container Toolkit" >&2
  fi
fi

step "2/4 build $IMAGE on $BASE"
run docker build --build-arg "BASE=$BASE" -f "$LAB/deploy/any-gpu/Dockerfile" -t "$IMAGE" "$LAB"

step "3/4 smoke test: does the container see a GPU?"
run docker run --rm --gpus all "$IMAGE" info

step "4/4 run the suite (results in $RESULTS)"
run mkdir -p "$RESULTS"
# --ipc=host and an unlimited memlock are what NVIDIA's container docs recommend for PyTorch
# (shared memory for workers; page-locked host buffers for pinned copies).
run docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v "$RESULTS:/results" "$IMAGE" \
  run --backend torch --out /results "$@"
echo "done: $RESULTS"
