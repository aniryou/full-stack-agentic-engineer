#!/usr/bin/env bash
# Turn kind workers into fake GPU nodes, from topology.txt:
#   * GKE-style labels: cloud.google.com/gke-nodepool, cloud.google.com/gke-accelerator and the GCE
#     placement labels cloud.google.com/gce-topology-{block,subblock,host} that Kueue TAS reads;
#   * the taint GKE puts on GPU nodes: nvidia.com/gpu=present:NoSchedule;
#   * fake capacity: nvidia.com/gpu is an *extended resource* - just a number in node status. We
#     PATCH it into status.capacity/allocatable; the scheduler counts it like real GPUs, and the
#     kubelet admits pods that request it because no device plugin claims that name. No device is
#     attached: /dev/nvidia* does not exist in the pods. That is the whole trick.
#
# Idempotent. Re-run it after restarting Docker: a kubelet that re-registers its node zeroes
# extended resources it does not manage.
#   deploy/kind/fake-gpus.sh
#   CLUSTER_NAME=other DRY_RUN=1 deploy/kind/fake-gpus.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

CLUSTER_NAME="${CLUSTER_NAME:-gpu-lab}"
CTX="${KUBE_CONTEXT:-kind-$CLUSTER_NAME}"
need kubectl "https://kubernetes.io/docs/tasks/tools/ (kubectl >= 1.27 for patch --subresource)"

k() { run kubectl --context "$CTX" "$@"; }

gpu_nodes=()
while read -r suffix pool gpus accel block sub host; do
  node="$CLUSTER_NAME-$suffix"
  if [[ "$gpus" -gt 0 ]]; then log "$node: pool=$pool, $gpus fake GPUs, $host"; else log "$node: pool=$pool (no GPUs)"; fi
  k label node "$node" --overwrite "cloud.google.com/gke-nodepool=$pool"
  if [[ "$gpus" -gt 0 ]]; then
    k label node "$node" --overwrite \
      "cloud.google.com/gke-accelerator=$accel" \
      "cloud.google.com/gce-topology-block=$block" \
      "cloud.google.com/gce-topology-subblock=$sub" \
      "cloud.google.com/gce-topology-host=$host"
    k taint node "$node" "nvidia.com/gpu=present:NoSchedule" --overwrite
    # "~1" is how a JSON pointer spells "/": the key is nvidia.com/gpu.
    k patch node "$node" --subresource=status --type=json -p \
      "[{\"op\":\"add\",\"path\":\"/status/capacity/nvidia.com~1gpu\",\"value\":\"$gpus\"},{\"op\":\"add\",\"path\":\"/status/allocatable/nvidia.com~1gpu\",\"value\":\"$gpus\"}]"
    gpu_nodes+=("$node=$gpus")
  fi
done < <(grep -Ev '^[[:space:]]*(#|$)' "$HERE/topology.txt")

if [[ "$DRY_RUN" != "1" ]]; then
  log "waiting until each node reports its fake GPUs as allocatable"
  for entry in ${gpu_nodes[@]+"${gpu_nodes[@]}"}; do
    node="${entry%%=*}" want="${entry##*=}"
    for _ in $(seq 1 30); do
      got="$(kubectl --context "$CTX" get node "$node" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
      [[ "$got" == "$want" ]] && break
      sleep 2
    done
    [[ "$got" == "$want" ]] || die "$node reports '$got' GPUs, expected $want"
  done
fi

k get nodes -o custom-columns='NAME:.metadata.name,POOL:.metadata.labels.cloud\.google\.com/gke-nodepool,GPUS:.status.allocatable.nvidia\.com/gpu,SUBBLOCK:.metadata.labels.cloud\.google\.com/gce-topology-subblock,HOST:.metadata.labels.cloud\.google\.com/gce-topology-host,TAINTS:.spec.taints[*].key'
