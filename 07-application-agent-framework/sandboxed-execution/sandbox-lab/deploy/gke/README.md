# deploy/gke — the sandbox platform on GKE Sandbox, and one execution under gVisor

**What it does.** `apply.sh` does these steps:

1. It points kubectl at the cluster from [`../gcp/terraform`](../gcp/terraform/).
2. It makes sure that RuntimeClass `gvisor` exists (GKE creates it with the sandbox pool).
3. It changes the image paths to your Artifact Registry repository.
4. It applies the same generated objects as kind, with `runtimeClassName: gvisor` in each sandbox pod.
   These objects are the restricted namespaces, the identities, the quota and the default-deny
   NetworkPolicies. They are also the egress proxy and stand-in upstream, the admission policies and
   the wrapper.
5. It creates the credential Secrets from a random value.
6. It runs one execution as a Job.

With `WITH_AGENT_SANDBOX=1`, it also installs the agent-sandbox controller (v1.0.2, verify) and a
`SandboxWarmPool` (`optional/`).

**Cost.** The first execution scales the gVisor pool from zero to one Spot `e2-standard-2` node. You pay
for that node while it runs (verify price). The autoscaler removes the node after it stays empty for
some time (about ten minutes by default, verify). The other costs are those of the cluster in
[`../gcp/README.md`](../gcp/README.md).

**Clean up.** `kubectl delete ns sandbox sandbox-egress sandbox-upstream sandbox-control` removes the
platform. `terraform -chdir=deploy/gcp/terraform destroy` removes the cluster.

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
| `workloads/` | the example Job, the warm pool, and the two objects that must fail |
| `reference/runtimeclass-gvisor.yaml` | the expected content of the RuntimeClass of GKE (verify). `apply.sh` applies it only if it does not exist |
| `optional/agent-sandbox.yaml` | SandboxTemplate + SandboxWarmPool + SandboxClaim (kubernetes-sigs/agent-sandbox `v1beta1`) |

`kubectl apply -f deploy/gke/` (not recursive) applies only the numbered files. The image fields contain
`LOCATION-docker.pkg.dev/PROJECT_ID/sandbox/python:3.12-slim`. `apply.sh` puts your repository in its
place.
