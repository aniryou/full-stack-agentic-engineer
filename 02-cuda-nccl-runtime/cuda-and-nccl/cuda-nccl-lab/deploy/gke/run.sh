#!/usr/bin/env bash
# Run one of the lab's GKE workloads, wait for it, save its log under ./out.
# Assumes kubectl points at the cluster (terraform output get_credentials prints the command).
#
#   ./run.sh status | smoke | vectoradd | nccl | timeshare | mig | rules | clean
#   DRY_RUN=1 ./run.sh nccl          # print every command, run nothing
#   TIMEOUT=2400 ./run.sh nccl       # seconds; scale-from-zero plus a devel-image pull can take 10+ minutes
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=gpu-lab
OUT="${OUT:-$PWD/out}"
TIMEOUT="${TIMEOUT:-1800}"
TIMEOUT="${TIMEOUT%s}" # accept 1800 or 1800s

run() {
  echo "+ $*" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  "$@"
}

namespace() { run kubectl apply -f "${HERE}/00-namespace.yaml"; }

save_logs() { # $1 = job name
  echo "+ kubectl -n ${NS} logs job/$1 > ${OUT}/$1.log" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  mkdir -p "${OUT}"
  kubectl -n "${NS}" logs "job/$1" | tee "${OUT}/$1.log"
}

wait_job() { # $1 = job name: poll until Complete or Failed, so a failed Job does not block until TIMEOUT
  echo "+ wait for job/$1: Complete or Failed (timeout ${TIMEOUT}s)" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
  local deadline=$((SECONDS + TIMEOUT)) conds
  while ((SECONDS < deadline)); do
    conds="$(kubectl -n "${NS}" get job "$1" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2> /dev/null || true)"
    case " ${conds} " in
      *" Complete "*) return 0 ;;
      *" Failed "*)
        echo "== job/$1 failed" >&2
        return 1
        ;;
    esac
    sleep 10
  done
  echo "== job/$1 not finished after ${TIMEOUT}s (kubectl -n ${NS} describe job/$1)" >&2
  return 1
}

job() { # $1 = manifest, $2 = job name: recreate, wait, save logs (also when the Job failed)
  run kubectl -n "${NS}" delete job "$2" --ignore-not-found
  run kubectl apply -f "${HERE}/$1"
  echo "== waiting for $2 (a GPU node may be scaling up from zero)" >&2
  local rc=0
  wait_job "$2" || rc=$?
  save_logs "$2" || true
  return "${rc}"
}

case "${1:-}" in
  status)
    run kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-gpu-sharing-strategy,cloud.google.com/gke-gpu-partition-size,cloud.google.com/gke-spot
    run kubectl get nodes -o "custom-columns=NODE:.metadata.name,GPU_ALLOCATABLE:.status.allocatable.nvidia\.com/gpu"
    ;;
  smoke)
    namespace
    echo "+ kubectl -n ${NS} create configmap gpurt-probe --from-file=probe.sh=../any-gpu/probe.sh --dry-run=client -o yaml | kubectl apply -f -" >&2
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      kubectl -n "${NS}" create configmap gpurt-probe --from-file=probe.sh="${HERE}/../any-gpu/probe.sh" \
        --dry-run=client -o yaml | kubectl apply -f -
    fi
    job 01-gpu-smoke.yaml gpu-smoke
    echo "== next: python -m gpurt.container --log ${OUT}/gpu-smoke.log" >&2
    ;;
  vectoradd)
    namespace
    job 02-cuda-vectoradd.yaml cuda-vectoradd
    ;;
  nccl)
    namespace
    job 03-nccl-tests-2gpu.yaml nccl-tests-2gpu
    echo "== next: python -m gpurt.nccltests ${OUT}/nccl-tests-2gpu.log" >&2
    ;;
  timeshare)
    namespace
    run kubectl apply -f "${HERE}/04-time-sharing.yaml"
    run kubectl -n "${NS}" rollout status deployment/timeshare-demo --timeout="${TIMEOUT}s"
    run kubectl -n "${NS}" get pods -l app=timeshare-demo -o wide
    run kubectl -n "${NS}" logs -l app=timeshare-demo --prefix
    ;;
  mig)
    namespace
    job 05-mig.yaml mig-1g5gb
    ;;
  rules)
    run kubectl apply -f "${HERE}/06-dcgm-alert-rules.yaml" # ClusterRules: cluster-scoped, no namespace
    echo "== rules only evaluate; firing alerts go to GMP's managed Alertmanager. Give it receivers:" >&2
    echo "   edit alertmanager/alertmanager.example.yaml, then: kubectl -n gmp-public create secret generic alertmanager \\" >&2
    echo "     --from-file=alertmanager.yaml=${HERE}/alertmanager/alertmanager.example.yaml --dry-run=client -o yaml | kubectl apply -f -" >&2
    ;;
  clean)
    run kubectl delete namespace "${NS}" --ignore-not-found
    ;;
  *)
    echo "usage: $0 status|smoke|vectoradd|nccl|timeshare|mig|rules|clean" >&2
    exit 2
    ;;
esac
