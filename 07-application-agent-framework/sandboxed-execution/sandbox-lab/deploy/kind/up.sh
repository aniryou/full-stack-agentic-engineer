#!/usr/bin/env bash
# The sandbox platform on a laptop: kind (Kubernetes 1.34) with a tainted sandbox worker, restricted Pod
# Security, default-deny NetworkPolicies, the egress proxy and a stand-in upstream, a ResourceQuota and
# LimitRange, and the ValidatingAdmissionPolicies. No gVisor: kind's nodes run runc only (see README).
# Idempotent: re-running skips the cluster if it exists and re-applies the rest.
#   DRY_RUN=1 deploy/kind/up.sh      # print every command, run nothing (no Docker needed)
#   deploy/kind/up.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

export CLUSTER_NAME="${CLUSTER_NAME:-sandbox-lab}"
CTX="kind-$CLUSTER_NAME"
K=(kubectl --context "$CTX")
M="$HERE/manifests"
need docker "The kind nodes are containers: install Docker Desktop or Docker Engine (2 GB of memory for Docker is enough)."
need kind "Install kind $KIND_VERSION: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
need kubectl "Install kubectl >= 1.30 (ValidatingAdmissionPolicy is v1 since 1.30)."

log "1/6 kind cluster '$CLUSTER_NAME' from ${KIND_NODE_IMAGE%%@*} (podPidsLimit set in kind-config.yaml)"
if [[ "$DRY_RUN" != "1" ]] && kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "cluster '$CLUSTER_NAME' exists - skipping create"
else
  run kind create cluster --name "$CLUSTER_NAME" --image "$KIND_NODE_IMAGE" --config "$HERE/kind-config.yaml" --wait 180s
fi

log "2/6 taint the sandbox worker (the RuntimeClass sandbox-runc tolerates it; nothing else does)"
run "${K[@]}" taint node "$CLUSTER_NAME-worker2" sandboxlab/pool=sandbox:NoSchedule --overwrite

log "3/6 preload $SANDBOX_IMAGE and $PROBE_IMAGE into the nodes"
for img in "$SANDBOX_IMAGE" "$PROBE_IMAGE"; do
  if run docker pull "$img"; then
    run kind load docker-image "$img" --name "$CLUSTER_NAME" || warn "kind load $img failed; nodes will pull on demand"
  fi
done

log "4/6 namespaces, RuntimeClass, identities, quota, NetworkPolicies, wrapper - policies BEFORE any sandbox pod"
run "${K[@]}" apply -f "$M/00-namespaces.yaml" -f "$M/05-runtimeclass.yaml" -f "$M/10-rbac.yaml" \
  -f "$M/20-quota.yaml" -f "$M/30-network-policy.yaml" -f "$M/60-wrapper.yaml"

log "5/6 the credential (one random value, never in git) and the egress proxy + stand-in upstream"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
random_token "$TMP/token"
for ns_secret in sandbox-egress/egress-credentials sandbox-upstream/api-stub-token; do
  ns="${ns_secret%%/*}" secret="${ns_secret##*/}"
  if [[ "$DRY_RUN" == "1" ]]; then
    run "${K[@]}" -n "$ns" create secret generic "$secret" --from-file=token="$TMP/token"
  elif ! "${K[@]}" -n "$ns" get secret "$secret" >/dev/null 2>&1; then
    run "${K[@]}" -n "$ns" create secret generic "$secret" --from-file=token="$TMP/token"
  else
    echo "secret $ns/$secret exists - keeping it (delete both secrets to rotate)"
  fi
done
run "${K[@]}" apply -f "$M/40-egress-proxy.yaml"
run "${K[@]}" -n sandbox-egress rollout status deploy/egress-proxy --timeout=180s
run "${K[@]}" -n sandbox-upstream rollout status deploy/api-stub --timeout=180s

log "6/6 admission policies (retried: a new policy takes a moment to be enforced)"
retry 5 3 "${K[@]}" apply -f "$M/50-admission-policy.yaml"
[[ "$DRY_RUN" == "1" ]] || sleep 3
run "${K[@]}" get validatingadmissionpolicies,validatingadmissionpolicybindings

cat <<EON

Ready. From the lab directory:
  deploy/kind/run-examples.sh                  # a Job per execution, the warm pool, and three things that must fail
  SANDBOXLAB_KUBE_CONTEXT=$CTX jupyter lab     # notebook 02 drives this cluster instead of the simulator
  deploy/kind/down.sh                          # delete the cluster
EON
