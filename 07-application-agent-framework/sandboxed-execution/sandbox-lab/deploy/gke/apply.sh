#!/usr/bin/env bash
# Put the sandbox platform on the GKE Sandbox cluster from deploy/gcp/terraform: credentials, RuntimeClass
# check, images from Artifact Registry, restricted namespaces, NetworkPolicies (Dataplane V2), the egress
# proxy + stand-in upstream, quota, and the admission policies; then one example execution under gVisor.
#   DRY_RUN=1 deploy/gke/apply.sh                       # read the procedure (no gcloud/kubectl needed)
#   deploy/gke/apply.sh                                 # values from `terraform output`
#   WITH_AGENT_SANDBOX=1 deploy/gke/apply.sh            # also the agent-sandbox warm pool (optional/)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"
TF="$HERE/../gcp/terraform"
need gcloud "Install the Google Cloud CLI."
need kubectl "gcloud components install kubectl"

tf_out() {  # tf_out NAME DEFAULT
  if [[ "$DRY_RUN" == "1" ]] || ! command -v terraform >/dev/null 2>&1; then echo "$2"; return; fi
  terraform -chdir="$TF" output -raw "$1" 2>/dev/null || echo "$2"
}
PROJECT_ID="${PROJECT_ID:-$(tf_out project_id PROJECT_ID)}"
ZONE="${ZONE:-$(tf_out zone us-central1-a)}"
CLUSTER="${CLUSTER:-$(tf_out cluster_name sandbox-lab)}"
REPO="${REPO:-$(tf_out repository_url us-central1-docker.pkg.dev/PROJECT_ID/sandbox)}"
K=(kubectl --context "gke_${PROJECT_ID}_${ZONE}_${CLUSTER}")

log "1/7 cluster credentials"
run gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT_ID"

log "2/7 RuntimeClass gvisor (GKE creates it with the sandbox node pool; applied here only if missing)"
if [[ "$DRY_RUN" == "1" ]] || "${K[@]}" get runtimeclass gvisor >/dev/null 2>&1; then
  run "${K[@]}" get runtimeclass gvisor -o yaml
else
  warn "no RuntimeClass gvisor: is the sandbox node pool created? applying the reference object"
  run "${K[@]}" apply -f "$HERE/reference/runtimeclass-gvisor.yaml"
fi

log "3/7 manifests with images from $REPO (private nodes have no route to Docker Hub)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
for f in "$HERE"/*.yaml "$HERE"/workloads/*.yaml "$HERE"/optional/*.yaml; do
  rel="${f#"$HERE"/}"
  mkdir -p "$OUT/$(dirname "$rel")"
  sed "s#LOCATION-docker.pkg.dev/PROJECT_ID/sandbox#$REPO#g" "$f" >"$OUT/$rel"
done
echo "rendered into $OUT (image: $REPO/python:3.12-slim; mirror it first: deploy/gcp/mirror-image.sh)"

log "4/7 namespaces, identities, quota, NetworkPolicies, wrapper - before any sandbox pod"
run "${K[@]}" apply -f "$OUT/00-namespaces.yaml" -f "$OUT/10-rbac.yaml" -f "$OUT/20-quota.yaml" \
  -f "$OUT/30-network-policy.yaml" -f "$OUT/60-wrapper.yaml"

log "5/7 credential (random, never in git) + egress proxy + stand-in upstream"
random_token "$OUT/token"
for ns_secret in sandbox-egress/egress-credentials sandbox-upstream/api-stub-token; do
  ns="${ns_secret%%/*}" secret="${ns_secret##*/}"
  if [[ "$DRY_RUN" != "1" ]] && "${K[@]}" -n "$ns" get secret "$secret" >/dev/null 2>&1; then
    echo "secret $ns/$secret exists - keeping it"
  else
    run "${K[@]}" -n "$ns" create secret generic "$secret" --from-file=token="$OUT/token"
  fi
done
run "${K[@]}" apply -f "$OUT/40-egress-proxy.yaml"
run "${K[@]}" -n sandbox-egress rollout status deploy/egress-proxy --timeout=300s

log "6/7 admission policies"
retry 5 3 "${K[@]}" apply -f "$OUT/50-admission-policy.yaml"

log "7/7 one execution under gVisor (the first one scales the sandbox pool up from zero: minutes, not seconds)"
run "${K[@]}" apply -f "$OUT/workloads/10-run-code-job.yaml"
run "${K[@]}" -n sandbox wait --for=condition=complete job/run-example-0001 --timeout=600s
run "${K[@]}" -n sandbox logs job/run-example-0001

if [[ "${WITH_AGENT_SANDBOX:-0}" == "1" ]]; then
  log "optional: agent-sandbox $AGENT_SANDBOX_VERSION controller + warm pool"
  run "${K[@]}" apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/$AGENT_SANDBOX_VERSION/sandbox-with-extensions.yaml"
  retry 10 6 "${K[@]}" apply -f "$OUT/optional/agent-sandbox.yaml"
fi
