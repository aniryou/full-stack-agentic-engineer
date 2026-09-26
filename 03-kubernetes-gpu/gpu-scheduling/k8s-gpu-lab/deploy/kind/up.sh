#!/usr/bin/env bash
# The whole kind lab in one go - a laptop with Docker, no GPU, $0:
#   1. kind cluster (Kubernetes 1.34): control plane + system worker + 4 workers
#   2. fake-gpus.sh: 4 fake nvidia.com/gpu per worker, GKE-style pool/accelerator/topology labels, GPU taint
#   3. install-addons.sh: JobSet, LeaderWorkerSet, Kueue (pinned in deploy/versions.env)
#   4. the lab's namespaces, Topology, ResourceFlavor, ClusterQueues, LocalQueues and priorities
# Idempotent: re-running skips the cluster if it exists and re-applies the rest.
#   deploy/kind/up.sh
#   DRY_RUN=1 deploy/kind/up.sh        # print every command, run nothing (no Docker needed)
#   PRELOAD_IMAGES=0 deploy/kind/up.sh # let each node pull busybox itself
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

export CLUSTER_NAME="${CLUSTER_NAME:-gpu-lab}"
CTX="kind-$CLUSTER_NAME"
need docker "The kind nodes are containers: install Docker Desktop or Docker Engine (4 GB of memory for Docker is plenty)."
need kind "Install kind $KIND_VERSION: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
need kubectl "Install kubectl >= 1.27: https://kubernetes.io/docs/tasks/tools/"

log "1/4 kind cluster '$CLUSTER_NAME' from ${KIND_NODE_IMAGE%%@*}"
if [[ "$DRY_RUN" != "1" ]] && kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "cluster '$CLUSTER_NAME' exists - skipping create"
else
  run kind create cluster --name "$CLUSTER_NAME" --image "$KIND_NODE_IMAGE" --config "$HERE/kind-config.yaml" --wait 180s
fi

if [[ "${PRELOAD_IMAGES:-1}" == "1" ]]; then
  log "preloading $LAB_IMAGE into the nodes (one registry pull instead of six)"
  if run docker pull "$LAB_IMAGE"; then
    run kind load docker-image "$LAB_IMAGE" --name "$CLUSTER_NAME" || warn "kind load failed; nodes will pull on demand"
  else
    warn "docker pull failed; nodes will pull on demand"
  fi
fi

log "2/4 fake GPUs, labels and taints"
"$HERE/fake-gpus.sh"

log "3/4 JobSet, LeaderWorkerSet, Kueue"
KUBE_CONTEXT="$CTX" "$HERE/install-addons.sh"

log "4/4 lab queues (retried: Kueue's webhook can lag its Deployment by a few seconds)"
retry 10 6 kubectl --context "$CTX" apply --server-side \
  -f "$HERE/manifests/00-namespaces.yaml" \
  -f "$HERE/manifests/10-kueue-topology-flavor.yaml" \
  -f "$HERE/manifests/20-kueue-queues.yaml" \
  -f "$HERE/manifests/30-kueue-priorities.yaml"
run kubectl --context "$CTX" wait --for=condition=Active clusterqueue/team-a-cq clusterqueue/team-b-cq --timeout=120s
run kubectl --context "$CTX" get clusterqueues,localqueues -A

cat <<EOF

Ready. Try the scenarios (from the lab directory):
  python -m k8sgpu kind predict s1     # what should happen
  python -m k8sgpu kind run s1         # make it happen and compare (s1..s6)
  deploy/kind/kwok.sh                  # optional: a 32-node fake H100 fleet (scenario k1)
  deploy/kind/down.sh                  # delete the cluster
EOF
