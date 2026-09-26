#!/usr/bin/env bash
# Install the pinned JobSet, LeaderWorkerSet and Kueue releases and wait for their controllers.
# Used by up.sh (kind) and deploy/gke/install-addons.sh (any other context).
#   deploy/kind/install-addons.sh
#   KUBE_CONTEXT=gke_my-project_us-central1-a_gpu-lab deploy/kind/install-addons.sh
#   SKIP_LWS=1 deploy/kind/install-addons.sh      # batch only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

CTX="${KUBE_CONTEXT:-kind-${CLUSTER_NAME:-gpu-lab}}"
need kubectl "https://kubernetes.io/docs/tasks/tools/"
k() { run kubectl --context "$CTX" "$@"; }

# JobSet and LWS first: Kueue picks up their CRDs for its integrations.
log "JobSet $JOBSET_VERSION (namespace jobset-system)"
k apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

if [[ "${SKIP_LWS:-0}" != "1" ]]; then
  log "LeaderWorkerSet $LWS_VERSION (namespace lws-system)"
  k apply --server-side -f "https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml"
fi

log "Kueue $KUEUE_VERSION (namespace kueue-system; TopologyAwareScheduling is beta and on by default)"
k apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml"

log "waiting for the controllers (their admission webhooks must answer before any queue or job is created)"
k -n jobset-system wait --for=condition=Available deployment/jobset-controller-manager --timeout=300s
if [[ "${SKIP_LWS:-0}" != "1" ]]; then
  k -n lws-system wait --for=condition=Available deployment/lws-controller-manager --timeout=300s
fi
k -n kueue-system wait --for=condition=Available deployment/kueue-controller-manager --timeout=300s
k get crd clusterqueues.kueue.x-k8s.io jobsets.jobset.x-k8s.io
