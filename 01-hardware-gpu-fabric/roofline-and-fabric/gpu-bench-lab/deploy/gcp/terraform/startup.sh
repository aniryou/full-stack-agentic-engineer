#!/usr/bin/env bash
# gpubench VM startup script: wait for the GPU driver, install the lab, run the suite on the GPU,
# upload the report to GCS, power off. Compute Engine runs it as root on every boot; settings come
# from instance metadata set by compute.tf (gpubench-*).
#
# Try it anywhere without side effects:  DRY_RUN=1 bash startup.sh   (prints each step's commands)
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes

step() { echo "[gpubench] $(date -u +%H:%M:%S) $*"; }
run() {
  step "+ $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}
capture() { # capture FILE CMD...: save a command's output as evidence next to the report
  local file="$1"
  shift
  step "+ $* > $file"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@" >"$file" 2>&1 || true
}
meta() { # meta KEY DEFAULT: one instance attribute from the metadata server
  if [ "$DRY_RUN" = "1" ]; then
    echo "$2"
    return 0
  fi
  curl -fsS -H "Metadata-Flavor: Google" "$MD/$1" 2>/dev/null || echo "$2"
}

BUCKET="$(meta gpubench-bucket example-bucket)"
REPO="$(meta gpubench-repo https://github.com/aniryou/full-stack-agentic-engineer.git)"
REF="$(meta gpubench-ref main)"
LAB_PATH="$(meta gpubench-lab-path 01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab)"
ARGS="$(meta gpubench-args "")"
POWEROFF="$(meta gpubench-poweroff true)"
TORCH_INDEX="$(meta gpubench-torch-index https://download.pytorch.org/whl/cu128)"
WORK=/opt/gpubench # in a dry run nothing is created: every write goes through run/capture
RUN_ID="$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$WORK/results/$RUN_ID"
PY="$WORK/venv/bin/python"

finish() { # upload whatever exists, then power off (unless told not to)
  run gcloud storage cp -r "$OUT" "gs://$BUCKET/results/"
  if [ "$POWEROFF" = "true" ]; then run shutdown -h now; fi
}

if [ -f "$WORK/.done" ] && [ "$DRY_RUN" != "1" ]; then
  step "already ran on this disk ($WORK/.done); delete it to run again"
  exit 0
fi
run mkdir -p "$OUT"

step "1/6 wait for the NVIDIA driver (Deep Learning VM images install it on first boot)"
ready=0
for _ in $(seq 1 60); do
  if [ "$DRY_RUN" = "1" ] || nvidia-smi >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  step "no working nvidia-smi after 10 minutes: see /var/log/syslog; uploading what exists"
  echo "driver not ready: nvidia-smi failed for 10 minutes" >"$OUT/FAILED.txt"
  finish
  exit 1
fi

step "2/6 record the machine"
capture "$OUT/nvidia-smi.txt" nvidia-smi
capture "$OUT/topo.txt" nvidia-smi topo -m
capture "$OUT/inventory.csv" nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,compute_cap,memory.total,memory.used,power.limit,power.default_limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,temperature.gpu,ecc.mode.current,persistence_mode --format=csv
capture "$OUT/lscpu.txt" lscpu
capture "$OUT/disks.txt" lsblk -o NAME,SIZE,TYPE,MODEL,ROTA

step "3/6 fetch the lab ($REPO @ $REF)"
run rm -rf "$WORK/repo"
run git clone --depth 1 --branch "$REF" "$REPO" "$WORK/repo"

step "4/6 python environment (reuse the image's PyTorch if it has one)"
if ! run python3 -m venv --system-site-packages "$WORK/venv"; then
  run apt-get update -qq # Ubuntu ships venv separately
  run apt-get install -y -qq python3-venv
  run python3 -m venv --system-site-packages "$WORK/venv"
fi
if [ "$DRY_RUN" = "1" ] || ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  run "$PY" -m pip install -q --upgrade pip
  if [ -n "$TORCH_INDEX" ]; then
    run "$PY" -m pip install -q torch --index-url "$TORCH_INDEX" # a CUDA build the driver supports
  else
    run "$PY" -m pip install -q torch
  fi
fi
run "$PY" -m pip install -q -e "$WORK/repo/$LAB_PATH"

step "5/6 run the suite on the GPU"
# ARGS is split on purpose: it holds extra CLI flags such as "--full" or "--suite gemm,stream".
# shellcheck disable=SC2086
if ! run "$PY" -m gpubench run --backend torch --out "$OUT" --workdir "$WORK" $ARGS; then
  step "the suite failed; uploading the partial output"
  echo "gpubench run failed; see the serial console log" >"$OUT/FAILED.txt"
fi

step "6/6 upload to gs://$BUCKET/results/$RUN_ID and power off"
run touch "$WORK/.done"
finish
