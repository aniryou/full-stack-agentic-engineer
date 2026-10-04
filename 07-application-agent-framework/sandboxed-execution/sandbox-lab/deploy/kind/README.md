# deploy/kind — pod-per-execution on a laptop Kubernetes, with every policy that makes it safe

**What it does.** These scripts create a 3-node [kind](https://kind.sigs.k8s.io/) cluster (Kubernetes 1.34: control
plane, a system worker, a tainted sandbox worker, `podPidsLimit: 128` on every kubelet). Then they install the
sandbox platform of the lab. The platform has these parts:

- four namespaces with Pod Security `restricted`
- a RuntimeClass `sandbox-runc` that keeps sandbox pods on the tainted worker
- default-deny NetworkPolicies with one way out (from the sandbox to the egress proxy on port 8080)
- the egress proxy, and a substitute upstream service whose credential is only in the Secret of the proxy
- a ResourceQuota and LimitRange
- two ValidatingAdmissionPolicies.

After that, `run-examples.sh` runs one execution as a Job and one execution through the warm pool (`kubectl
exec`). It also runs four things that must fail. The last of the four examines your cluster and makes sure that the
cluster enforces the NetworkPolicies in practice.

**Cost.** $0. Everything runs in Docker on your machine and uses about 2 GB of Docker memory.

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
| `manifests/00`-`60` | Generated files. `up.sh` applies them in order: namespaces, RuntimeClass, RBAC, quota, NetworkPolicies, proxy, admission policies, the wrapper ConfigMap. |
| `workloads/10-run-code-job.yaml` | one execution as a Job: deadline 120 s, no retries, TTL 300 s, the code in `SANDBOX_CODE` |
| `workloads/20-warm-pool.yaml` | Two idle sandbox pods for `kubectl exec`. The runner deletes each pod after one use. |
| `workloads/90`, `91` | MUST FAIL at admission: a simple pod (Pod Security and policy) and a Job without budgets (policy) |
| `workloads/92-gvisor-in-kind.yaml` | MUST FAIL at run time: Kubernetes accepts RuntimeClass `gvisor` (handler `runsc`), and the pod ends `Failed`. |
| `workloads/93-egress-must-fail.yaml` | MUST FAIL on the network: a sandbox Job that obeys the policies and connects directly to the api-stub (`10.96.0.201:8081`) and kube-dns (`10.96.0.10:53`). If one of the two prints `CONNECTED`, `run-examples.sh` stops with an error. |

## What kind reproduces, and what it does not

| Real on kind | Not real on kind |
|---|---|
| Pod Security Admission, the ValidatingAdmissionPolicies, RuntimeClass scheduling merge | **gVisor or any VM isolation**: every sandbox pod runs under runc on your host kernel |
| NetworkPolicy. kindnetd enforces it through kube-network-policies. Its `main.go` at kind v0.33.0 builds that controller. The `main.go` of v0.23.0 did not. That your node image contains this kindnetd is `(verify)`. Thus step 6 of `run-examples.sh` proves it on your cluster. | NetworkPolicy *as a security boundary*: the policy dataplane of kindnetd fails **open**. If its controller cannot start, kindnetd logs the failure and continues. Then every policy here has no effect, and nothing tells you. |
| Jobs, deadlines, TTLs, quotas, `podPidsLimit`, emptyDir `sizeLimit` eviction | node pools, autoscaling from zero, Spot, the GKE metadata server |
| the egress proxy, hostAliases instead of DNS, credential injection | private nodes and the absence of a route to the internet |

kind is a place where you learn the objects and see the admission chain operate. It is not a sandbox for untrusted
code. **Do not assume that the cluster enforces the NetworkPolicies until step 6 passes on your cluster.** An older
kind, another CNI without policy support, or a kindnetd whose policy controller failed to start enforces **none** of
them. Such a cluster does not enforce the default-deny egress, egress-only-to-the-proxy, or the ingress rule of the
api-stub. Nothing else in the cluster tells you this.

**Which policies step 6 shows enforced.** A pass shows two things. First, the cluster enforces the default-deny
egress of the sandbox namespace. The kube-dns target has no ingress policy of its own, so only the egress policy of
the sandbox can drop that connection. Second, the cluster enforces the ingress-only-from-the-proxy rule of the
api-stub, together with the egress-only-to-the-proxy rule of the sandbox (the direct stub target). Step 6 does not
examine the egress policy of the proxy namespace itself.

**If it fails** (a `CONNECTED` line), this cluster does not enforce the policies. Do these steps:

1. Run `down.sh`.
2. Make a copy of `kind-config.yaml`. Keep the generated file as it is, because a test examines it.
3. Add `networking: {disableDefaultCNI: true}` to the copy.
4. Create the cluster yourself with `kind create cluster --name sandbox-lab --image <KIND_NODE_IMAGE from ../versions.env> --config <copy>`.
5. Install Calico as its quickstart tells you. Specify the exact version that you made sure is correct.
6. Run `up.sh` again. It does not do the create step when the cluster exists already.
7. Run `run-examples.sh` again.

## Troubleshooting

* **`no runtime for "runsc" is configured`** on the `needs-gvisor` pod: this is the expected result. It is the
  lesson of `92`.
* **A sandbox pod stays Pending**: run `kubectl -n sandbox describe pod`. The sandbox worker has a taint, and only
  RuntimeClass `sandbox-runc` adds the toleration. A pod without that toleration has no node where it can go.
* **`failed calling webhook` or the policy seems ignored for a few seconds** after `up.sh`: Kubernetes needs a
  short time before it enforces a new ValidatingAdmissionPolicy. Apply the must-fail files again.
* **The Job pod cannot reach the proxy**: the NetworkPolicies are in place. But it is possible that kindnetd
  enforces them on a new pod a short time late. The same delay in the other direction is the case "may be started
  unprotected". Try again. Then examine `kubectl -n sandbox-egress logs deploy/egress-proxy`. Each request that the
  proxy permits or denies is one JSON line there.
