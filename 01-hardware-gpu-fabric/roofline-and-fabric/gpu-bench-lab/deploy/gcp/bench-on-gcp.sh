#!/usr/bin/env bash
# One command on GCP: create the Spot GPU VM with Terraform, wait for its report in GCS, download
# it, destroy everything.
#
#   PROJECT=my-gpu-lab deploy/gcp/bench-on-gcp.sh
#   PROJECT=my-gpu-lab TF_VARS="-var machine_type=a2-highgpu-2g" deploy/gcp/bench-on-gcp.sh
#   DRY_RUN=1 PROJECT=x deploy/gcp/bench-on-gcp.sh     # print every step, change nothing
#   KEEP=1 ...                                         # skip the final destroy
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$HERE/terraform"
PROJECT="${PROJECT:?set PROJECT=<your GCP project id>}"
OUT="${OUT:-$HERE/../../results/gcp}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"
TF_VARS="${TF_VARS:-}"
DRY_RUN="${DRY_RUN:-0}"
KEEP="${KEEP:-0}"

step() { echo "==> $*"; }
run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}
tf() { run terraform -chdir="$TF_DIR" "$@"; }

step "1/5 tools"
for tool in terraform gcloud; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    if [ "$DRY_RUN" = "1" ]; then echo "   ($tool not installed — fine for a dry run)"; else
      echo "need $tool on PATH" >&2
      exit 1
    fi
  fi
done

step "2/5 create the VM (it starts benchmarking on boot)"
tf init -input=false
# TF_VARS is split on purpose: it holds extra "-var name=value" flags.
# shellcheck disable=SC2086
tf apply -input=false -auto-approve -var "project_id=$PROJECT" $TF_VARS
if [ "$DRY_RUN" = "1" ]; then
  RESULTS="gs://<bucket>/results/"
else
  RESULTS="$(terraform -chdir="$TF_DIR" output -raw results_uri)"
  trap 'echo "!! stopped early: resources may still exist — run: terraform -chdir=$TF_DIR destroy -var project_id=$PROJECT"' ERR
fi

step "3/5 wait up to $TIMEOUT_MIN min for a report in $RESULTS (follow along: terraform -chdir=$TF_DIR output -raw watch_progress)"
if [ "$DRY_RUN" != "1" ]; then
  deadline=$(($(date +%s) + TIMEOUT_MIN * 60))
  until gcloud storage ls --recursive "$RESULTS" 2>/dev/null | grep -Eq '\.json$|FAILED\.txt$'; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "no report after $TIMEOUT_MIN min (Spot preempted? quota? see watch_progress)" >&2
      exit 1
    fi
    sleep 30
  done
  sleep 20 # let the upload of the remaining files finish
fi

step "4/5 download to $OUT"
run mkdir -p "$OUT"
run gcloud storage cp -r "$RESULTS*" "$OUT/"

step "5/5 clean up"
if [ "$KEEP" = "1" ]; then
  echo "KEEP=1: leaving the VM and bucket; delete later with: terraform -chdir=$TF_DIR destroy -var project_id=$PROJECT"
else
  # shellcheck disable=SC2086
  tf destroy -input=false -auto-approve -var "project_id=$PROJECT" $TF_VARS
fi
echo "done: reports in $OUT"
