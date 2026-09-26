#!/usr/bin/env bash
# Serve an MoE with expert parallelism on layer 02's GKE cluster (the 2 x L4 `l4x2` pool) and benchmark it.
# Assumes kubectl points at that cluster:
#   $(terraform -chdir=../../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform output -raw get_credentials)
#
#   ./run.sh status                  nodes of the l4x2 pool, allocatable GPUs, the vLLM pod
#   ./run.sh up                      namespace + Deployment (TP=2 + EP) + Service; waits for the rollout
#   ./run.sh layout tp|tp_ep|dp_ep   switch the parallel layout (restarts the pod on the same node)
#   ./run.sh bench [CONCURRENCY]     vllm bench serve from a CPU Job -> out/bench_<layout>_c<N>.txt
#   ./run.sh logs                    the server's start-up log (look for 'Using default MoE config')
#   ./run.sh clean                   delete the namespace (the GPU node scales back to zero)
#   DRY_RUN=1 ./run.sh up            print every command, run nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=moe-lab
OUT="${OUT:-$PWD/out}"
TIMEOUT="${TIMEOUT:-2400}"   # seconds: node scale-up from zero + image pull + 28.6 GB of weights
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  echo "+ $*" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  "$@"
}

layout_args() { # the vLLM arguments of each layout, as JSON for kubectl patch
  local common="\"${MODEL}\",\"--port=8000\",\"--max-model-len=4096\",\"--gpu-memory-utilization=0.92\""
  case "$1" in
    tp) echo "[${common},\"--tensor-parallel-size=2\",\"--enable-return-routed-experts\"]" ;;
    tp_ep) echo "[${common},\"--tensor-parallel-size=2\",\"--enable-expert-parallel\",\"--enable-return-routed-experts\"]" ;;
    dp_ep) echo "[${common},\"--data-parallel-size=2\",\"--enable-expert-parallel\",\"--enable-return-routed-experts\"]" ;;
    *)
      echo "layout must be tp, tp_ep or dp_ep" >&2
      return 2
      ;;
  esac
}

current_layout() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "tp_ep"
    return 0
  fi
  local a
  a="$(kubectl -n "${NS}" get deploy vllm-moe -o jsonpath='{.spec.template.spec.containers[0].args}')"
  if [[ "${a}" == *data-parallel-size* ]]; then
    echo dp_ep
  elif [[ "${a}" == *enable-expert-parallel* ]]; then
    echo tp_ep
  else
    echo tp
  fi
}

case "${1:-}" in
  status)
    run kubectl get nodes -l cloud.google.com/gke-nodepool=l4x2 \
      -o custom-columns='NODE:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type'
    run kubectl -n "${NS}" get pods -o wide
    ;;
  up)
    run kubectl apply -f "${HERE}/00-namespace.yaml"
    run kubectl apply -f "${HERE}/01-vllm-moe-ep2.yaml"
    echo "== waiting for the pod (a 2 x L4 Spot node may be scaling up from zero)" >&2
    run kubectl -n "${NS}" rollout status deploy/vllm-moe --timeout="${TIMEOUT}s"
    ;;
  layout)
    args="$(layout_args "${2:-}")"
    run kubectl -n "${NS}" patch deploy vllm-moe --type=json \
      -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":${args}}]"
    run kubectl -n "${NS}" rollout status deploy/vllm-moe --timeout="${TIMEOUT}s"
    ;;
  bench)
    c="${2:-16}"
    lay="$(current_layout)"
    run kubectl -n "${NS}" delete job moe-bench --ignore-not-found
    echo "+ sed 's/--max-concurrency=16/--max-concurrency=${c}/' 02-bench-job.yaml | kubectl apply -f -" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      sed -e "s/--max-concurrency=16/--max-concurrency=${c}/" -e "s/--num-prompts=128/--num-prompts=$((c > 1 ? 8 * c : 16))/" \
        "${HERE}/02-bench-job.yaml" | kubectl apply -f -
    fi
    run kubectl -n "${NS}" wait --for=condition=complete job/moe-bench --timeout=1800s
    echo "+ kubectl -n ${NS} logs job/moe-bench > ${OUT}/bench_${lay}_c${c}.txt" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${OUT}"
      kubectl -n "${NS}" logs job/moe-bench | tee "${OUT}/bench_${lay}_c${c}.txt"
    fi
    ;;
  logs)
    run kubectl -n "${NS}" logs deploy/vllm-moe --tail=400
    ;;
  clean)
    run kubectl delete namespace "${NS}" --ignore-not-found
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
