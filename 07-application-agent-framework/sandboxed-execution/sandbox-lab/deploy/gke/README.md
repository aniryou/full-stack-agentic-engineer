# deploy/gke — the sandbox platform on GKE Sandbox, and one execution under gVisor

**What it does.** `apply.sh` points kubectl at the cluster from [`../gcp/terraform`](../gcp/terraform/),
checks that RuntimeClass `gvisor` exists (GKE creates it with the sandbox pool), rewrites the image paths to
your Artifact Registry repository, applies the same generated objects as kind — restricted namespaces,
identities, quota, default-deny NetworkPolicies, the egress proxy and stand-in upstream, the admission policies,
the wrapper — with `runtimeClassName: gvisor` in every sandbox pod, creates the credential Secrets from a random
value, and runs one execution as a Job. With `WITH_AGENT_SANDBOX=1` it also installs the agent-sandbox
controller (v1.0.2, verify) and a `SandboxWarmPool` (`optional/`).

**Cost.** The first execution scales the gVisor pool from zero: one Spot `e2-standard-2` node, billed while it
runs (verify price); the autoscaler removes it after it has been empty for a while (about ten minutes by
default, verify). Everything else is the cluster in [`../gcp/README.md`](../gcp/README.md).

**Clean up.** `kubectl delete ns sandbox sandbox-egress sandbox-upstream sandbox-control` removes the
platform; `terraform -chdir=deploy/gcp/terraform destroy` removes the cluster.

## Run it

```bash
DRY_RUN=1 deploy/gke/apply.sh          # read the procedure
deploy/gke/apply.sh                    # values from `terraform output`; or PROJECT_ID=... ZONE=... CLUSTER=... REPO=...
kubectl -n sandbox logs job/run-example-0001
kubectl -n sandbox-egress logs deploy/egress-proxy     # the egress audit trail, one JSON line per request
kubectl get runtimeclass gvisor -o yaml                 # what GKE created (compare reference/)
```

| Path | What it is |
|---|---|
| `00`-`60-*.yaml` | generated from `sandboxlab/k8s/policy.py` (GKE target), applied in order |
| `workloads/` | the example Job, the warm pool, and the two must-fail objects |
| `reference/runtimeclass-gvisor.yaml` | what GKE's RuntimeClass is expected to contain (verify); applied only if it is missing |
| `optional/agent-sandbox.yaml` | SandboxTemplate + SandboxWarmPool + SandboxClaim (kubernetes-sigs/agent-sandbox `v1beta1`) |

`kubectl apply -f deploy/gke/` (not recursive) applies only the numbered files. The images read
`LOCATION-docker.pkg.dev/PROJECT_ID/sandbox/python:3.12-slim`; `apply.sh` substitutes the repository.
