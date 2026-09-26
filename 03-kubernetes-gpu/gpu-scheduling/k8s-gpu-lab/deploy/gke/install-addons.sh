#!/usr/bin/env bash
# Install JobSet + Kueue (same pinned releases as the kind lab) on the GKE cluster that
# deploy/gcp/terraform created, then the GKE Kueue objects (Spot flavor + DWS flex-start flavor
# behind a ProvisioningRequest admission check).
#   gcloud container clusters get-credentials gpu-lab --location us-central1-a
#   deploy/gke/install-addons.sh
#   KUBE_CONTEXT=gke_my-project_us-central1-a_gpu-lab DRY_RUN=1 deploy/gke/install-addons.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

need kubectl "https://kubernetes.io/docs/tasks/tools/"
CTX="${KUBE_CONTEXT:-$(kubectl config current-context 2>/dev/null || true)}"
if [[ -z "$CTX" && "$DRY_RUN" == "1" ]]; then
  CTX="gke_PROJECT_ZONE_gpu-lab"
fi
[[ "$CTX" == gke_* ]] || die "context '$CTX' does not look like GKE (gke_*). Run gcloud container clusters get-credentials, or set KUBE_CONTEXT."

log "JobSet + Kueue on $CTX (LeaderWorkerSet skipped; SKIP_LWS=0 to add it)"
KUBE_CONTEXT="$CTX" SKIP_LWS="${SKIP_LWS:-1}" "$HERE/../kind/install-addons.sh"

log "GKE Kueue objects: flavors l4-spot / l4-flex, ProvisioningRequestConfig, AdmissionCheck, ClusterQueue, LocalQueue"
retry 10 6 kubectl --context "$CTX" apply --server-side -f "$HERE/10-kueue-gke.yaml"
run kubectl --context "$CTX" wait --for=condition=Active clusterqueue/gke-gpu --timeout=120s || \
  warn "gke-gpu is not Active yet: check 'kubectl describe clusterqueue gke-gpu' (the AdmissionCheck needs GKE's ProvisioningRequest API)"
run kubectl --context "$CTX" get resourceflavors,clusterqueues,admissionchecks
