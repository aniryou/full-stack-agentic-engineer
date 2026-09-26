# deploy/kind — pod-per-execution on a laptop Kubernetes, with every policy that makes it safe

**What it does.** Creates a 3-node [kind](https://kind.sigs.k8s.io/) cluster (Kubernetes 1.34: control plane,
a system worker, a tainted sandbox worker, `podPidsLimit: 128` on every kubelet) and installs the lab's sandbox
platform: four namespaces with Pod Security `restricted`, a RuntimeClass `sandbox-runc` that pins sandbox pods to
the tainted worker, default-deny NetworkPolicies with one way out (sandbox -> egress proxy:8080), the egress
proxy and a stand-in upstream whose credential lives only in the proxy's Secret, a ResourceQuota and LimitRange,
and two ValidatingAdmissionPolicies. `run-examples.sh` then runs one execution as a Job, one through the warm
pool (`kubectl exec`), and four things that must fail — the last one checks, on your cluster, that the
NetworkPolicies are actually enforced.

**Cost.** $0: everything runs in Docker on your machine, about 2 GB of Docker memory.

**Clean up.** `deploy/kind/down.sh` deletes the cluster and everything in it.

**Needs.** Docker, kind v0.33.0, kubectl >= 1.30, network access to Docker Hub (`python:3.12-slim`,
`busybox:1.38.0`).

## Run it

```bash
# from the lab root (07-application-agent-framework/sandboxed-execution/sandbox-lab)
DRY_RUN=1 deploy/kind/up.sh          # read the whole procedure; nothing is executed
deploy/kind/up.sh                    # ~2 minutes with images cached
deploy/kind/run-examples.sh          # a Job, a warm pod, and the four must-fail checks (the last: NetworkPolicy)
SANDBOXLAB_KUBE_CONTEXT=kind-sandbox-lab python3 -m jupyterlab notebooks   # notebook 02 drives this cluster
deploy/kind/down.sh
```

| File | What it is |
|---|---|
| `kind-config.yaml` | generated: nodes, labels, `serviceSubnet` 10.96.0.0/16, the `podPidsLimit` kubelet patch |
| `manifests/00`-`60` | generated, applied by `up.sh` in order: namespaces, RuntimeClass, RBAC, quota, NetworkPolicies, proxy, admission policies, the wrapper ConfigMap |
| `workloads/10-run-code-job.yaml` | one execution as a Job: deadline 120 s, no retries, TTL 300 s, the code in `SANDBOX_CODE` |
| `workloads/20-warm-pool.yaml` | two idle sandbox pods for `kubectl exec`; the runner deletes each after one use |
| `workloads/90`, `91` | MUST FAIL at admission: a naive pod (Pod Security + policy) and a Job without budgets (policy) |
| `workloads/92-gvisor-in-kind.yaml` | MUST FAIL at run time: RuntimeClass `gvisor` (handler `runsc`) is accepted and the pod ends `Failed` |
| `workloads/93-egress-must-fail.yaml` | MUST FAIL on the network: a conforming sandbox Job that connects straight to the api-stub (`10.96.0.201:8081`) and kube-dns (`10.96.0.10:53`); `run-examples.sh` stops with an error if either prints `CONNECTED` |

## What kind reproduces, and what it does not

| Real on kind | Not real on kind |
|---|---|
| Pod Security Admission, the ValidatingAdmissionPolicies, RuntimeClass scheduling merge | **gVisor or any VM isolation**: every sandbox pod runs under runc on your host kernel |
| NetworkPolicy — kindnetd enforces it through kube-network-policies (its `main.go` at kind v0.33.0 builds that controller; v0.23.0's did not); that your node image bundles that kindnetd is `(verify)`, so step 6 of `run-examples.sh` proves it on your cluster | NetworkPolicy *as a security boundary*: kindnetd's policy dataplane fails **open** — if its controller cannot start it logs and carries on, and every policy here silently stops applying |
| Jobs, deadlines, TTLs, quotas, `podPidsLimit`, emptyDir `sizeLimit` eviction | node pools, autoscaling from zero, Spot, the GKE metadata server |
| the egress proxy, hostAliases instead of DNS, credential injection | private nodes and the absence of a route to the internet |

kind is a place to learn the objects and watch the admission chain work, not a sandbox for untrusted code.
**Do not assume the NetworkPolicies are enforced until step 6 has passed on your cluster**: an older kind,
another CNI without policy support, or a kindnetd whose policy controller failed to start enforces **none**
of them — not the default-deny egress, not egress-only-to-the-proxy, not the api-stub's ingress rule — and
nothing else in the cluster tells you so.

**Which policies step 6 shows enforced.** A pass means: the sandbox namespace's default-deny egress (the
kube-dns target has no ingress policy of its own, so only the sandbox's egress policy can drop it) and the
api-stub's ingress-only-from-the-proxy plus the sandbox's egress-only-to-the-proxy (the direct stub target).
It does not exercise the proxy namespace's own egress policy. **If it fails** (a `CONNECTED` line), policies
are not enforced on this cluster: run `down.sh`; copy `kind-config.yaml` (keep the generated file as it is —
a test checks it) and add `networking: {disableDefaultCNI: true}` to the copy; create the cluster yourself
with `kind create cluster --name sandbox-lab --image <KIND_NODE_IMAGE from ../versions.env> --config <copy>`;
install Calico by its quickstart (pin the version you verified); then run `up.sh` (it skips the create step
for an existing cluster) and `run-examples.sh` again.

## Troubleshooting

* **`no runtime for "runsc" is configured`** on the `needs-gvisor` pod: expected — that is the lesson of `92`.
* **A sandbox pod stays Pending**: `kubectl -n sandbox describe pod` — the sandbox worker is tainted and only
  RuntimeClass `sandbox-runc` adds the toleration; a pod without it has nowhere to go.
* **`failed calling webhook` or the policy seems ignored for a few seconds** after `up.sh`: a new
  ValidatingAdmissionPolicy takes a moment to be enforced; re-apply the must-fail files.
* **The Job pod cannot reach the proxy**: the NetworkPolicies were applied, but kindnetd's enforcement may lag a
  new pod by a moment ("may be started unprotected" in the other direction); retry, and check
  `kubectl -n sandbox-egress logs deploy/egress-proxy` — every allowed and denied request is a JSON line there.
