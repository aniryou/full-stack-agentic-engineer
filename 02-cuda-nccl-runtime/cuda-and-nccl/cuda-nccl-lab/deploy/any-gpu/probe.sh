#!/usr/bin/env bash
# What can this container see of its GPU? Prints sections that `python -m gpurt.container --log`
# parses. Read-only; needs only bash, coreutils and grep (nvidia-smi if the driver injected it).
#
#   docker run --rm --gpus all -v "$PWD":/w nvidia/cuda:12.8.1-base-ubuntu24.04 bash /w/probe.sh > probe.log
#   kubectl logs job/gpu-smoke > probe.log            # deploy/gke/01-gpu-smoke.yaml runs this script
#   python -m gpurt.container --log probe.log
#
# DRY_RUN=1 prints the commands (to stderr) without running them.
set -euo pipefail

step() {
  echo "+ $*" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  "$@"
}
section() { echo "=== $1 ==="; }

SMI="$(command -v nvidia-smi || echo /usr/local/nvidia/bin/nvidia-smi)" # GKE mounts it under /usr/local/nvidia

section "gpurt-probe v1"
step date -u +%Y-%m-%dT%H:%M:%SZ

section env
step bash -c 'env | grep -E "^(NVIDIA_|CUDA_|LD_LIBRARY_PATH=)" | sort || true'

section dev
step bash -c 'ls -l /dev/nvidia* /dev/nvidia-caps/* 2>/dev/null || echo "(no /dev/nvidia* device nodes)"'

section mountinfo
step bash -c 'grep -E "nvidia|libcuda" /proc/self/mountinfo || echo "(no NVIDIA mounts)"'

section proc-driver-version
step bash -c 'cat /proc/driver/nvidia/version 2>/dev/null || echo "(no /proc/driver/nvidia/version)"'

section nvidia-smi
if [[ -x "${SMI}" ]]; then
  step bash -c "\"${SMI}\" | head -n 12" || echo "(nvidia-smi failed: exit $?)"
else
  echo "(nvidia-smi unavailable)"
fi

section gpus
if [[ -x "${SMI}" ]]; then
  step "${SMI}" --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader || echo "(query failed: exit $?)"
else
  echo "(nvidia-smi unavailable)"
fi

section end
