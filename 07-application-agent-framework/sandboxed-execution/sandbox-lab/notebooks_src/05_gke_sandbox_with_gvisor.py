# %% [markdown]
# # 05 · GKE Sandbox with gVisor: the same pods, on a real kernel boundary
#
# **Tier:** T3 to run (a GKE cluster from `deploy/gcp/terraform`), but everything here is **T0**: the
# Terraform is read as text, the manifests are validated against Kubernetes 1.34, the design-review
# checklist is answered offline, and the cost and pool sizing are computed. Deploying it is
# `deploy/gcp/terraform` + `deploy/gke/apply.sh`.
#
# ## The one-minute version
#
# On GKE the top of the isolation ladder is a *node-pool setting*, not code (PRIMER §5, §9). A node
# pool with `sandbox_config { type = "GVISOR" }` runs every pod that names `runtimeClassName: gvisor`
# under gVisor's user-space kernel, on nodes that nothing else lands on. The design decisions:
#
# * **A second, dedicated pool** — GKE Sandbox needs it (the first pool can't be sandboxed); GKE
#   taints and labels the sandbox nodes itself and creates the `gvisor` RuntimeClass.
# * **No route to the internet** — private nodes and no Cloud NAT; the only destinations are Google
#   APIs over Private Google Access, so images come from Artifact Registry.
# * **The credential defence is not NetworkPolicy** — a NetworkPolicy cannot block the node-local
#   metadata server; `GKE_METADATA` mode plus `automountServiceAccountToken: false` and a KSA with
#   no IAM binding is what keeps cloud credentials away from the sandbox.
# * **Spot, autoscaling from zero** — the pool costs nothing idle; the first execution scales it up
#   (minutes, not seconds).
#
# The same manifests you validated on kind (notebook 02) run here, with `runtimeClassName: gvisor`.

# %%
import json
from sandboxlab import bench, gke
from sandboxlab.k8s import manifests as m, policy as P, render

for c in gke.review():
    print(f"{'OK ' if c.ok else 'MISS'}  {c.name:<40} {c.evidence}")

# %% [markdown]
# ## Worked example: the GKE sandbox pod differs from the kind one in one field
#
# The policy renders the same objects for both targets; the sandbox pod's `runtimeClassName` is
# `gvisor` on GKE and `sandbox-runc` on kind, and the image comes from Artifact Registry. Everything
# else — the securityContext, no token, DNS off, sized volumes — is identical.

# %%
gke_pol, kind_pol = P.SandboxPolicy.gke(), P.SandboxPolicy.kind()
g = P.run_code_job(gke_pol, "print(1)", "x")["spec"]["template"]["spec"]
k = P.run_code_job(kind_pol, "print(1)", "x")["spec"]["template"]["spec"]
print("GKE  runtimeClassName:", g["runtimeClassName"], "| image:", g["containers"][0]["image"])
print("kind runtimeClassName:", k["runtimeClassName"], "| image:", k["containers"][0]["image"])
same = {kk: g[kk] for kk in ("automountServiceAccountToken", "dnsPolicy", "securityContext")}
print("identical on both:", json.dumps(same, default=str))

# %% [markdown]
# ## Worked example: cold start by isolation level
#
# gVisor adds to a container's start time; a Kubernetes cold start adds the API, scheduling and node
# scale-up on top. A warm pool trades that away for idle capacity. The rows this machine can measure
# say "measured"; the rest are sample output in the documented format (illustrative).

# %%
for row in bench.ladder(bench.measure(["fork_exec", "process"], n=5)):
    print(row.row())

# %% [markdown]
# ## Exercise 5.1 — the Terraform keeps the sandbox contract
#
# `gke.review()` reads `deploy/gcp/terraform/*.tf` and answers the review questions. Return the set
# of check names that pass, and assert the load-bearing ones are among them: gVisor, no NAT by
# default, Dataplane V2 (NetworkPolicy enforced), and the GKE metadata server.

# %% exercise
def passing_checks() -> set:
    ### BEGIN SOLUTION
    return {c.name for c in gke.review() if c.ok}
    ### END SOLUTION

# %% check
ok = passing_checks()
for needed in ("sandbox pool runs gVisor", "no Cloud NAT by default", "NetworkPolicy enforced",
               "Workload Identity / GKE metadata server", "sandbox pool scales from zero"):
    assert needed in ok, needed
print("✅ the Terraform passes every load-bearing check:")
for n in sorted(ok):
    print("   -", n)

# %% [markdown]
# ## Exercise 5.2 — `type = "GVISOR"` is case-sensitive (a pitfall)
#
# The provider validates `node_config.sandbox_config.type` against exactly `"GVISOR"`; the lowercase
# `"gvisor"` (which is what `gcloud`'s `--sandbox type=gvisor` uses) fails `terraform validate`.
# Read `node_pools.tf` and confirm the upper-case value is present and the lowercase one is not.

# %% exercise
def gvisor_value_is_upper_case() -> bool:
    tf = (render.LAB_ROOT / "deploy/gcp/terraform/node_pools.tf").read_text()
    ### BEGIN SOLUTION
    return 'type = "GVISOR"' in tf and 'type = "gvisor"' not in tf
    ### END SOLUTION

# %% check
assert gvisor_value_is_upper_case()
print('✅ node_config.sandbox_config.type = "GVISOR" (the provider is case-sensitive; gcloud uses lowercase)')

# %% [markdown]
# ## Exercise 5.3 — the metadata server is not a NetworkPolicy problem
#
# A common mistake is to "block the metadata server with a NetworkPolicy". You cannot: traffic to the
# node is always allowed. The defence is the GKE metadata server (`GKE_METADATA`), no mounted token,
# and a service account with no roles. Confirm the sandbox pod sets `automountServiceAccountToken:
# false` and uses the `sandbox-exec` service account, and that the node pools set `GKE_METADATA`.

# %% exercise
def metadata_defences() -> dict:
    tf = (render.LAB_ROOT / "deploy/gcp/terraform/node_pools.tf").read_text()
    ### BEGIN SOLUTION
    return {"no_token": g["automountServiceAccountToken"] is False,
            "unprivileged_sa": g["serviceAccountName"] == "sandbox-exec",
            "gke_metadata": 'mode = "GKE_METADATA"' in tf}
    ### END SOLUTION

# %% check
d = metadata_defences()
assert d == {"no_token": True, "unprivileged_sa": True, "gke_metadata": True}
# and RBAC gives sandbox-exec no permissions at all:
rbac = m.load_all(render.LAB_ROOT / "deploy/gke/10-rbac.yaml")
sa = [o for o in rbac if o["kind"] == "ServiceAccount" and o["metadata"]["name"] == "sandbox-exec"][0]
assert sa["automountServiceAccountToken"] is False
assert not any(o["kind"] == "RoleBinding" and any(s["name"] == "sandbox-exec" for s in o["subjects"]) for o in rbac)
print("✅ no token, an unbound service account, and GKE_METADATA mode — not a NetworkPolicy")

# %% [markdown]
# ## Exercise 5.4 — size the pool and the cost per execution
#
# With a peak of 5 `run_code`/s, 2 s executions and a 45 s gVisor-pod cold start, the lab's warm pool
# is **replace-after-use**: the runner deletes each pod after one execution and the Deployment warms a
# replacement, so every execution holds a slot for 2 s of work *and* 45 s of warm-up. Return the
# Little's-law mean occupancy (the floor), the Erlang C slot count that keeps at most 20% of requests
# waiting for a warm pod (`bench.replace_after_use_slots`), and the cost of one execution: a sandbox
# pod's share of an `e2-standard-2` held for the run plus its replacement's warm-up, at a Spot price
# you look up (mark it verify). See `COMPUTE.md` for prices.

# %% exercise
def pool_and_cost(rate: float, exec_s: float, cold_s: float, node_usd_per_hour: float) -> tuple:
    """Return (mean_occupancy, slots, cost_per_execution_usd) for pods sharing an e2-standard-2 (2 vCPU, 8 GiB)."""
    ### BEGIN SOLUTION
    mean = bench.littles_law_pool(rate, exec_s, cold_s)["total_ceil"]
    slots = bench.replace_after_use_slots(rate, exec_s, cold_s, 0.2)
    fits = gke.pods_per_node(2, 8, 0.5, 0.3125)      # 500m CPU, 320Mi per sandbox pod, after reservations
    cost = gke.cost_per_execution_usd(exec_s + cold_s, node_usd_per_hour, fits)   # run + the replacement's warm-up
    return mean, slots, cost
    ### END SOLUTION

# %% check
mean, slots, cost = pool_and_cost(5, 2, 45, node_usd_per_hour=0.02)   # ~$0.02/h for a Spot e2-standard-2 (VERIFY)
assert mean == bench.littles_law_pool(5, 2, 45)["total_ceil"] == 235   # 10 busy + 225 warming, on average
assert slots == bench.servers_for(5, 47, max_p_wait=0.2) and slots > mean
fits = gke.pods_per_node(2, 8, 0.5, 0.3125)
assert abs(cost - gke.cost_per_execution_usd(47, 0.02, fits)) < 1e-12, "a one-shot pod pays its warm-up too"
reuse = bench.servers_for(5, 2, max_p_wait=0.2)
print(f"✅ replace-after-use: {mean} slots on average, {slots} for P(wait) <= 0.2 — a 45 s cold start dominates")
print(f"   cost per execution ~ ${cost:.6f} at $0.02/h per node (VERIFY the price; see COMPUTE.md), 47 of it warm-up")
print(f"   a reuse pool ({reuse} slots) or a snapshot restore cuts that — reuse carries state between executions,")
print("   and one snapshot must never be restored into two tenants (PRIMER §2, §6)")

# %% [markdown]
# ## Deploying it (T3)
#
# ```bash
# cd deploy/gcp/terraform && cp terraform.tfvars.example terraform.tfvars   # set project_id
# terraform init && terraform apply                 # ~10 min: 2 pools, GVISOR, no NAT
# cd ../../.. && deploy/gcp/mirror-image.sh          # python:3.12-slim -> Artifact Registry
# deploy/gke/apply.sh                                # the platform + one execution under gVisor
# python3 -m sandboxlab gke-review                   # this checklist, on your .tf files
# ```
#
# Clean up with `terraform -chdir=deploy/gcp/terraform destroy`. Cost and cleanup:
# [`deploy/gcp/README.md`](../../../07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gcp/README.md).

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "On GKE the strongest isolation is a node-pool setting: a second pool with
# `sandbox_config { type = \"GVISOR\" }`, and every pod with `runtimeClassName: gvisor` runs on it
# under a user-space kernel, on nodes nothing else touches. GKE taints and labels those nodes and
# creates the RuntimeClass, so a pod needs only the class name. I make the cluster private with no
# Cloud NAT, so nothing has a route to the internet and images come from Artifact Registry over
# Private Google Access — which is still a path to Google APIs, so the sandbox's default-deny
# NetworkPolicy closes it too. The egress proxy reaches only in-cluster upstreams unless I deliberately
# turn NAT on, and even then NetworkPolicy on Dataplane V2 keeps the sandboxes off the internet —
# gVisor is a kernel boundary, not a network one. The one thing a NetworkPolicy cannot do is block the node-local metadata server, so the
# cloud-credential defence is `GKE_METADATA` mode, no mounted token, and a KSA with no IAM binding.
# The pool is Spot and scales from zero, so it costs nothing idle; the trade is a 40-to-50-second
# cold start on the first execution, which is why interactive traffic gets a warm pool. The manifests
# are the ones I validated on kind, with one field changed."
#
# **Drill 1.** *`terraform validate` fails on the sandbox pool — the value looks right.* — The
# provider validates `sandbox_config.type` against `"GVISOR"` exactly; `"gvisor"` (what `gcloud`
# takes) fails. It is the most common typo in a GKE Sandbox module.
#
# **Drill 2.** *We added a NetworkPolicy to block 169.254.169.254 and the pod still reached it.* —
# NetworkPolicy never blocks traffic to the pod's own node, and the metadata server is node-local.
# Use Workload Identity: `GKE_METADATA` on the pool serves pods their KSA's identity, and if that KSA
# has no IAM roles there is nothing to steal. Belt and braces: `automountServiceAccountToken: false`
# so there is no Kubernetes token either.
#
# **Drill 3.** *The first execution took 45 seconds; is gVisor that slow?* — No: gVisor adds tens to
# hundreds of milliseconds. The 45 seconds is Kubernetes scaling the Spot pool from zero — VM boot,
# image pull, node registration. Keep a warm pool (or snapshot/restore) for interactive work, and let
# the pool scale to zero only for bursty or batch traffic where a cold start is acceptable.
