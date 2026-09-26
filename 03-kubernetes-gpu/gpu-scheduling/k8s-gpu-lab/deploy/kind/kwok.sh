#!/usr/bin/env bash
# Optional: add a fleet of fake 8-GPU H100 nodes with KWOK (Kubernetes WithOut Kubelet) to the kind
# cluster, to watch Kueue Topology-Aware Scheduling place big gangs (scenario k1). KWOK nodes are
# Node objects whose heartbeats and pod status a controller fakes: no containers ever run, so
# hundreds of nodes cost almost nothing. The real scheduler and Kueue still make every decision.
#   deploy/kind/kwok.sh                          # 2 blocks x 4 subblocks x 4 hosts = 32 nodes, 256 GPUs
#   KWOK_BLOCKS=4 KWOK_HOSTS=8 deploy/kind/kwok.sh
#   deploy/kind/kwok.sh --delete                 # remove the fake nodes and the fleet queue
#   DRY_RUN=1 deploy/kind/kwok.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

CLUSTER_NAME="${CLUSTER_NAME:-gpu-lab}"
CTX="${KUBE_CONTEXT:-kind-$CLUSTER_NAME}"
BLOCKS="${KWOK_BLOCKS:-2}" SUBBLOCKS="${KWOK_SUBBLOCKS:-4}" HOSTS="${KWOK_HOSTS:-4}" GPUS="${KWOK_GPUS:-8}"
need kubectl "https://kubernetes.io/docs/tasks/tools/"
k() { run kubectl --context "$CTX" "$@"; }

if [[ "${1:-}" == "--delete" ]]; then
  log "removing the KWOK fleet"
  k delete -f "$HERE/manifests/kwok-fleet-queues.yaml" --ignore-not-found
  k delete nodes -l type=kwok --ignore-not-found
  exit 0
fi

log "KWOK $KWOK_VERSION controller (kube-system) and the 'fast' simulation stages"
k apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/kwok.yaml"
k apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/stage-fast.yaml"
# stage-fast marks Job pods Succeeded right after they start; drop that stage so fake pods stay
# Running and the placement stays visible (kubectl delete the Jobs to free the capacity).
k delete stage pod-complete --ignore-not-found
k -n kube-system wait --for=condition=Available deployment/kwok-controller --timeout=180s

nodes_file="$(mktemp "${TMPDIR:-/tmp}/kwok-nodes.XXXXXX")"
trap 'rm -f "$nodes_file"' EXIT
for ((b = 1; b <= BLOCKS; b++)); do
  for ((s = 1; s <= SUBBLOCKS; s++)); do
    for ((h = 1; h <= HOSTS; h++)); do
      name="kwok-b$b-s$s-h$h"
      cat >>"$nodes_file" <<EOF
---
apiVersion: v1
kind: Node
metadata:
  name: $name
  annotations:
    node.alpha.kubernetes.io/ttl: "0"
    kwok.x-k8s.io/node: fake
  labels:
    kubernetes.io/hostname: $name
    kubernetes.io/os: linux
    kubernetes.io/arch: amd64
    type: kwok
    cloud.google.com/gke-nodepool: kwok-h100
    cloud.google.com/gke-accelerator: nvidia-h100-80gb
    cloud.google.com/gce-topology-block: kwok-b$b
    cloud.google.com/gce-topology-subblock: kwok-b$b-s$s
    cloud.google.com/gce-topology-host: $name
spec:
  taints:
    - {key: kwok.x-k8s.io/node, value: fake, effect: NoSchedule}
    - {key: nvidia.com/gpu, value: present, effect: NoSchedule}
status:
  allocatable: {cpu: "208", memory: 1872Gi, pods: "110", nvidia.com/gpu: "$GPUS"}
  capacity: {cpu: "208", memory: 1872Gi, pods: "110", nvidia.com/gpu: "$GPUS"}
  nodeInfo: {architecture: amd64, operatingSystem: linux, kubeletVersion: fake, kubeProxyVersion: fake,
             bootID: "", containerRuntimeVersion: "", kernelVersion: "", machineID: "", osImage: "", systemUUID: ""}
EOF
    done
  done
done
log "$((BLOCKS * SUBBLOCKS * HOSTS)) fake nodes x $GPUS GPUs (first one shown)"
sed -n '2,/^---/p' "$nodes_file" | sed '$d'
k apply -f "$nodes_file"

log "fleet queue (flavor gpu-h100-kwok tolerates the kwok taint)"
retry 10 6 kubectl --context "$CTX" apply --server-side -f "$HERE/manifests/kwok-fleet-queues.yaml"
k wait --for=condition=Active clusterqueue/fleet-cq --timeout=120s
k get nodes -l type=kwok -o custom-columns='NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,SUBBLOCK:.metadata.labels.cloud\.google\.com/gce-topology-subblock'
echo
echo "Next: python -m k8sgpu kind predict k1 && python -m k8sgpu kind run k1"
