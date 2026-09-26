#!/usr/bin/env bash
# Apply one GKE example (each is a file in this directory):
#   deploy/gke/apply-examples.sh smoke          # 00: nvidia-smi on the Spot L4 pool (scale from zero)
#   deploy/gke/apply-examples.sh dws            # 20: a 2-node gang through Kueue + DWS flex-start (needs install-addons.sh)
#   deploy/gke/apply-examples.sh computeclass   # 30: the Spot -> on-demand -> flex-start ComputeClass
#   WEIGHTS_BUCKET=my-bucket deploy/gke/apply-examples.sh serving   # 40: vLLM with GCS FUSE weights (applies 30 first)
#   deploy/gke/apply-examples.sh delete         # remove every example object (the nodes scale back to zero)
# Cost: a Spot g2-standard-4 L4 node runs only while a pod needs it; DRY_RUN=1 prints the commands.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

need kubectl "https://kubernetes.io/docs/tasks/tools/"
CTX="${KUBE_CONTEXT:-$(kubectl config current-context 2>/dev/null || true)}"
if [[ -z "$CTX" && "$DRY_RUN" == "1" ]]; then
  CTX="gke_PROJECT_ZONE_gpu-lab"
fi
[[ "$CTX" == gke_* ]] || die "context '$CTX' does not look like GKE (gke_*); set KUBE_CONTEXT."
k() { run kubectl --context "$CTX" "$@"; }

case "${1:-}" in
  smoke)
    k apply -f "$HERE/00-smoke-l4.yaml"
    echo "watch: kubectl get pods -w  (Pending -> TriggeredScaleUp -> Running takes a few minutes: VM boot + driver)"
    ;;
  dws)
    k apply -f "$HERE/20-dws-sample-job.yaml"
    echo "watch: kubectl get workloads,provisioningrequests -n ml -w"
    ;;
  computeclass)
    k apply -f "$HERE/30-computeclass-l4.yaml"
    k get computeclass l4-spot-first -o yaml
    ;;
  serving)
    [[ -n "${WEIGHTS_BUCKET:-}" ]] || die "set WEIGHTS_BUCKET (terraform -chdir=deploy/gcp/terraform output -raw weights_bucket)"
    k apply -f "$HERE/30-computeclass-l4.yaml"
    rendered="$(mktemp "${TMPDIR:-/tmp}/serving.XXXXXX")"
    trap 'rm -f "$rendered"' EXIT
    sed "s|\${WEIGHTS_BUCKET}|$WEIGHTS_BUCKET|g" "$HERE/40-serving-vllm-gcsfuse.yaml" >"$rendered"
    k apply -f "$rendered"
    echo "watch: kubectl -n serving get pods -w ; then kubectl -n serving port-forward svc/vllm-l4 8000:8000"
    ;;
  delete)
    for f in 40-serving-vllm-gcsfuse.yaml 30-computeclass-l4.yaml 20-dws-sample-job.yaml 00-smoke-l4.yaml; do
      k delete -f "$HERE/$f" --ignore-not-found
    done
    ;;
  *)
    die "usage: $0 smoke|dws|computeclass|serving|delete"
    ;;
esac
