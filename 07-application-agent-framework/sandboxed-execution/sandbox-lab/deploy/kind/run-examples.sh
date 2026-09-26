#!/usr/bin/env bash
# Exercise the kind sandbox platform: one execution as a Job (reaches the upstream only through the proxy),
# the warm pool, and four things that must fail - a naive pod and an unbounded Job (rejected at admission),
# a gVisor pod (admitted, then Failed: kind has no runsc handler), and direct connections from a sandbox
# pod that bypass the proxy (NetworkPolicy; kindnetd fails open, so this checks enforcement on YOUR cluster).
#   DRY_RUN=1 deploy/kind/run-examples.sh
#   deploy/kind/run-examples.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
CLUSTER_NAME="${CLUSTER_NAME:-sandbox-lab}"
K=(kubectl --context "${KUBE_CONTEXT:-kind-$CLUSTER_NAME}")
W="$HERE/workloads"
need kubectl "Install kubectl."

log "1/6 one execution = one Job"
run "${K[@]}" apply -f "$W/10-run-code-job.yaml"
run "${K[@]}" -n sandbox wait --for=condition=complete job/run-example-0001 --timeout=120s
run "${K[@]}" -n sandbox logs job/run-example-0001

log "2/6 the warm pool: exec into an idle pod, then delete it (the Deployment replaces it)"
run "${K[@]}" apply -f "$W/20-warm-pool.yaml"
run "${K[@]}" -n sandbox rollout status deploy/sandbox-warm --timeout=120s
POD="$([[ "$DRY_RUN" == "1" ]] && echo sandbox-warm-POD || "${K[@]}" -n sandbox get pod -l app=sandbox-warm,sandboxlab/state=idle -o jsonpath='{.items[0].metadata.name}')"
run "${K[@]}" -n sandbox label pod "$POD" sandboxlab/state=claimed --overwrite
if [[ "$DRY_RUN" == "1" ]]; then
  run "${K[@]}" -n sandbox exec -i "$POD" -c run -- python3 -I /opt/sandbox/wrapper.py --stdin --workspace /work/ex1
else
  echo 'print("hello from a warm pod")' | run "${K[@]}" -n sandbox exec -i "$POD" -c run -- python3 -I /opt/sandbox/wrapper.py --stdin --workspace /work/ex1
fi
run "${K[@]}" -n sandbox delete pod "$POD" --wait=false

log "3/6 MUST FAIL: a pod with no hardening (Pod Security + the policy)"
if run "${K[@]}" apply -f "$W/90-rejected-pod.yaml"; then [[ "$DRY_RUN" == "1" ]] || die "the naive pod was admitted"; fi

log "4/6 MUST FAIL: a Job with no deadline, default retries and no TTL"
if run "${K[@]}" apply -f "$W/91-rejected-job.yaml"; then [[ "$DRY_RUN" == "1" ]] || die "the unbounded Job was admitted"; fi

log "5/6 MUST FAIL at run time: RuntimeClass gvisor (handler runsc) on kind"
run "${K[@]}" apply -f "$W/92-gvisor-in-kind.yaml"
[[ "$DRY_RUN" == "1" ]] || sleep 10
run "${K[@]}" -n sandbox-demo get pod needs-gvisor -o wide
run "${K[@]}" -n sandbox-demo get events --field-selector involvedObject.name=needs-gvisor

log "6/6 MUST FAIL: a sandbox pod connecting directly to the api-stub and to kube-dns (only the proxy is allowed)"
run "${K[@]}" apply -f "$W/93-egress-must-fail.yaml"
run "${K[@]}" -n sandbox wait --for=condition=complete job/run-egress-check-0001 --timeout=120s
if [[ "$DRY_RUN" != "1" ]]; then
  OUT="$("${K[@]}" -n sandbox logs job/run-egress-check-0001)"
  echo "$OUT"
  if grep -q "CONNECTED" <<<"$OUT"; then
    die "NetworkPolicy is NOT enforced on this cluster: a sandbox pod bypassed the proxy. kindnetd's policy dataplane fails open (or this kind predates it). Recreate the cluster with networking.disableDefaultCNI: true and install Calico - deploy/kind/README.md"
  fi
  grep -q "via-proxy: reachable" <<<"$OUT" || warn "the proxy route did not answer either: pod networking may be broken, so 'blocked' proves less"
  echo "enforced here: sandbox egress default-deny (direct to kube-dns dropped) and api-stub ingress/egress (direct to the stub dropped); the proxy route works"
fi
