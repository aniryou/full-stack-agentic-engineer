# %% [markdown]
# # 02 · Pod per execution on Kubernetes: what makes a sandbox pod safe, and who enforces it
#
# **Tier:** T0. Here, the notebook builds every manifest, validates it against Kubernetes 1.34 and runs it
# through an offline admission predictor. A simulated cluster really runs the code (in the process sandbox)
# and measures the times of the lifecycle. With Docker + kind (T0 + Docker), the same objects run on a real
# cluster. Run `deploy/kind/up.sh`, then set `SANDBOXLAB_KUBE_CONTEXT`.
#
# ## The one-minute version
#
# On Kubernetes, a sandbox is a pod. The controls that make it safe are real only if the API server
# *refuses* pods without them (PRIMER §5). There are three admission steps, in this order:
#
# 1. **RuntimeClass**: `runtimeClassName: gvisor` (or `sandbox-runc` on kind) selects the runtime. It also
#    merges the node selector and tolerations of the class. Thus the pod goes to the sandbox pool.
# 2. **Pod Security Admission `restricted`**: the upstream baseline. Its rules are non-root, drop `ALL`,
#    `allowPrivilegeEscalation: false`, a seccomp profile and no host namespaces. *It rejects Pods, not
#    Jobs.* The API server accepts a bad Job and rejects its Pods later.
# 3. **A ValidatingAdmissionPolicy** for the rules about sandboxes that PSS does not know. It asks for the
#    runtime class, `automountServiceAccountToken: false`, a read-only root and sized volumes only. It
#    permits no Secrets in the environment. On Jobs, it also asks for a deadline, `backoffLimit: 0` and a
#    TTL.
#
# One execution is one Job (a new pod and a cold start) or one `kubectl exec` into a warm pod. This notebook
# builds the objects from a policy, predicts admission and validates the YAML. Then it runs the lifecycle on
# a simulated cluster. It also shows the one thing that kind cannot do: gVisor.

# %%
from sandboxlab.k8s import admission as A, manifests as m, policy as P, render
from sandboxlab.k8s.runner import JobRunner, SimulatedBackend, WarmPoolRunner

pol = P.SandboxPolicy.kind()
print("target:", pol.target, "| runtime class:", pol.runtime_class, "(handler", pol.runtime_handler + ")")
print("budgets:", pol.budgets)
print("Job deadline", pol.job_deadline_s, "s | backoffLimit 0 | TTL", pol.job_ttl_s, "s")

# %% [markdown]
# ## Worked example: one execution as a Job, and its pod's securityContext
#
# `run_code_job` builds the Job that the runner creates. Read the pod. It has these properties:
#
# - no service-account token,
# - DNS off (only `hostAliases` resolve, thus the sandbox reaches the proxy by name and nothing else),
# - non-root by number,
# - a read-only root,
# - `ALL` capabilities dropped,
# - sized `emptyDir` volumes.

# %%
job = P.run_code_job(pol, "print(sum(range(10)))", "demo-0001", "turn1:step1:call0:9f2c")
spec = job["spec"]["template"]["spec"]
c = spec["containers"][0]
print("Job:", job["metadata"]["name"], "| deadline", job["spec"]["activeDeadlineSeconds"],
      "| backoffLimit", job["spec"]["backoffLimit"], "| TTL", job["spec"]["ttlSecondsAfterFinished"])
print("automountServiceAccountToken:", spec["automountServiceAccountToken"], "| dnsPolicy:", spec["dnsPolicy"])
print("pod securityContext:", spec["securityContext"])
print("container securityContext:", c["securityContext"])
print("volumes:", [(v["name"], v.get("emptyDir", {}).get("sizeLimit", "configMap")) for v in spec["volumes"]])

# %% [markdown]
# ## Worked example: the admission chain, predicted offline
#
# `admission.admit()` runs the same three steps as the API server and returns every violation. The predictor
# admits the well-formed Job. It rejects the two "must fail" examples, at different steps.

# %%
rcs = {pol.runtime_class: P.runtime_class(pol)}
wl = {name: objs for name, (_, objs) in P.workloads(pol).items()}
for name in ("10-run-code-job.yaml", "90-rejected-pod.yaml", "91-rejected-job.yaml"):
    r = A.admit(wl[name][0], runtime_classes=rcs, allowed_runtime_classes=pol.allowed_runtime_classes,
                node_handlers={pol.runtime_handler})
    print(f"\n{name}: {'ADMITTED' if r.admitted else 'REJECTED'}")
    for v in r.violations[:6]:
        print("   ", v)

# %% [markdown]
# ## Exercise 2.1 — Pod Security `restricted`, by hand
#
# Write the checks that restricted PSS does on the `securityContext` of a container. Return the list of
# reasons for which it rejects the pod. If the pod passes, the list is empty. For each container, examine
# these four items:
#
# - `runAsNonRoot` is true (the pod-level value counts),
# - `allowPrivilegeEscalation` is false,
# - `capabilities.drop` contains `ALL`,
# - `seccompProfile.type` is `RuntimeDefault` or `Localhost` (the pod-level value counts).
#
# Compare your result with `A.pss_restricted` of the library.

# %% exercise
def restricted_violations(spec: dict) -> int:
    """Return the number of restricted-PSS violations in a pod spec."""
    ### BEGIN SOLUTION
    psc = spec.get("securityContext") or {}
    pod_nonroot = psc.get("runAsNonRoot") is True
    pod_seccomp = (psc.get("seccompProfile") or {}).get("type")
    n = 0
    for c in spec.get("containers", []):
        sc = c.get("securityContext") or {}
        if not (sc.get("runAsNonRoot") is True or pod_nonroot):
            n += 1
        if sc.get("allowPrivilegeEscalation") is not False:
            n += 1
        if "ALL" not in ((sc.get("capabilities") or {}).get("drop") or []):
            n += 1
        if ((sc.get("seccompProfile") or {}).get("type") or pod_seccomp) not in ("RuntimeDefault", "Localhost"):
            n += 1
    return n
    ### END SOLUTION

# %% check
good = P.run_code_job(pol, "print(1)", "x")["spec"]["template"]["spec"]
bad = P.rejected_pod(pol)["spec"]
assert restricted_violations(good) == 0 == len(A.pss_restricted(good))
assert restricted_violations(bad) == 4 and len(A.pss_restricted(bad)) >= 4
print("✅ the hardened pod passes restricted PSS; the naive one fails four container checks")

# %% [markdown]
# ## Exercise 2.2 — why `backoffLimit` matters for code that is not idempotent
#
# The default `backoffLimit` of a Job is **6**. Thus, when a pod fails, the Job controller retries it up
# to 6 times. That is 7 runs of code that the model wrote, and this code can have side effects. The sandbox
# policy accepts only `backoffLimit: 0`. You get a Job spec as the API server sees it, that is, with the
# defaults filled in. Return the maximum number of times that the code can run.

# %% exercise
def max_runs(job_spec: dict) -> int:
    """Worst-case number of pod runs for a Job spec (backoffLimit defaults to 6 when unset)."""
    ### BEGIN SOLUTION
    return job_spec.get("backoffLimit", 6) + 1
    ### END SOLUTION

# %% check
assert max_runs({"backoffLimit": 0}) == 1 and max_runs({}) == 7 and max_runs({"backoffLimit": 2}) == 3
# the policy's job-no-retries rule rejects anything but backoffLimit 0:
bad_job = P.rejected_job(pol)
bad_job["spec"]["backoffLimit"] = 6
viol = {v.rule for v in A.evaluate(bad_job, pol.allowed_runtime_classes)}
assert "job-no-retries" in viol
print("✅ default retries mean up to 7 runs of non-idempotent code; the policy requires backoffLimit: 0")

# %% [markdown]
# ## Exercise 2.3 — the manifests validate against Kubernetes 1.34
#
# Every core-kind object that the lab renders must pass strict schema validation for 1.34. In CI,
# `kubernetes-validate --strict -k 1.34.0` does this check. Count the core-kind objects that validate for
# the kind target. The validation treats CRD pod templates as Pods. The test suite examines the
# agent-sandbox CRDs against their pinned schemas.

# %% exercise
def count_valid_core_objects() -> int:
    import kubernetes_validate as kv
    ### BEGIN SOLUTION
    n = 0
    for f, o in render.objects("kind"):
        if o["kind"] in m.CORE_KINDS:
            kv.validate(o, "1.34.0", strict=True)
            n += 1
    return n
    ### END SOLUTION

# %% check
try:
    import kubernetes_validate  # noqa: F401
    n = count_valid_core_objects()
    assert n >= 25
    print(f"✅ {n} core-kind objects validate strictly for Kubernetes 1.34")
except ModuleNotFoundError:
    print("kubernetes-validate not installed; `pip install kubernetes-validate` to run this check")

# %% [markdown]
# ## Exercise 2.4 — one way out: read the default-deny NetworkPolicies
#
# The sandbox namespace has a default-deny policy (both directions) and one egress rule. This rule permits
# traffic to the egress-proxy pods on port 8080, and to nothing else. It does not permit DNS (the pod uses
# `hostAliases`). You get the rendered NetworkPolicies. Return the set of `(namespace, port)` that a sandbox
# pod can reach.

# %% exercise
def sandbox_egress_targets() -> set:
    pols = [o for f, o in render.objects("kind") if o["kind"] == "NetworkPolicy"]
    ### BEGIN SOLUTION
    out = set()
    for p in pols:
        if p["metadata"]["namespace"] != P.SANDBOX_NS:
            continue
        for rule in p["spec"].get("egress", []):
            for to in rule["to"]:
                ns = to.get("namespaceSelector", {}).get("matchLabels", {}).get(m.NAME_LABEL)
                for port in rule["ports"]:
                    out.add((ns, port["port"]))
    return out
    ### END SOLUTION

# %% check
assert sandbox_egress_targets() == {(P.EGRESS_NS, 8080)}
deny = [o for f, o in render.objects("kind") if o["kind"] == "NetworkPolicy"
        and o["metadata"]["name"] == "default-deny" and o["metadata"]["namespace"] == P.SANDBOX_NS][0]
assert deny["spec"]["policyTypes"] == ["Ingress", "Egress"] and "egress" not in deny["spec"]
print("✅ a sandbox pod's only egress is the proxy on 8080; the default-deny policy allows nothing else,")
print("   not even DNS — so the pod cannot resolve, let alone reach, an attacker's host")

# %% [markdown]
# ## Exercise 2.5 — one execution, on a simulated cluster
#
# The `SimulatedBackend` admits the Job (with the real predictor) and runs the code (in the real process
# sandbox). It also simulates the times of the Kubernetes lifecycle. Run one execution. Then make sure of
# these three things:
#
# - the result has the label simulated,
# - the result has a cold-start cost,
# - a second delivery of the request with the same idempotency key does **not** run the code two times.

# %% exercise
def run_two_with_same_key():
    """Run the same idempotency key twice through fresh JobRunners on one backend; return (r1, r2, create_count)."""
    backend = SimulatedBackend(pol, seed=0)
    ### BEGIN SOLUTION
    r1 = JobRunner(pol, backend).run("print(6 * 7)", idempotency_key="turn1:step1:call0:k")
    r2 = JobRunner(pol, backend).run("print(6 * 7)", idempotency_key="turn1:step1:call0:k")
    creates = sum(c.startswith("kubectl create") for c in backend.commands)
    return r1, r2, creates
    ### END SOLUTION

# %% check
r1, r2, creates = run_two_with_same_key()
assert r1.exit_reason == "ok" and r1.stdout.strip() == "42" and r1.simulated
assert r1.startup_s > 0.3 and r1.total_s > r1.wall_s      # a Job pays a cold start the code's own time does not include
assert r2.stdout.strip() == "42" and "not run again" in r2.notes[-1] and creates == 2
print(f"✅ simulated Job: {r1.total_s:.2f} s round trip ({r1.startup_s:.2f} s cold start + {r1.wall_s:.2f} s run);")
print("   the second delivery of the same key replayed the result instead of running the code again")

# %% [markdown]
# ## The one thing kind cannot do: gVisor
#
# The nodes of kind run containerd with only the `runc` handler. The API server *accepts* a RuntimeClass
# whose handler is `runsc`. But every pod that uses it ends in phase `Failed`, because no node can create
# its sandbox. On kind, the lab uses `sandbox-runc` (a real RuntimeClass, handler `runc`), thus the pods
# run. The lab keeps a `gvisor` example for one purpose: to show the failure. On GKE (notebook 05), the same
# pods really run under gVisor.

# %%
rc, gpod = P.gvisor_in_kind()[1:]
verdict = A.admit(gpod, runtime_classes={"gvisor": rc, **rcs},
                  allowed_runtime_classes=["gvisor"], node_handlers={"runc"})
print("admitted:", verdict.admitted, "| will fail at run time:", verdict.will_fail)
print("\nkind is a place to learn the objects and watch admission work — not a boundary for untrusted code:")
print("its NetworkPolicy dataplane (kindnetd) fails open, and there is no VM or user-space kernel under the pods.")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "On Kubernetes, a sandbox is a pod. I make its controls mandatory, and I do not only
# hope for them.
# The namespace is Pod Security `restricted`, which gives me non-root, drop-all, no privilege escalation and
# a seccomp profile. But PSS enforces on Pods, thus the API server accepts a bad *Job* and rejects only its
# Pods. You see that in the events of the Job, not at `kubectl apply`.
#
# "Thus I add a ValidatingAdmissionPolicy for the rules that are specific to sandboxes. It asks for a
# RuntimeClass, `automountServiceAccountToken: false`, a read-only root and only sized `emptyDir` volumes.
# It permits no Secrets in the environment. On Jobs, it asks for a deadline, `backoffLimit: 0` and a TTL.
#
# "One execution is one Job: a new pod and a cold start, and Kubernetes deletes it after its TTL. When
# latency is important, I keep a warm pool and use `kubectl exec` into an idle pod. I delete that pod after
# one use, thus no state goes to the next execution. Egress is a default-deny NetworkPolicy with a single
# rule to the proxy. It does not even permit DNS. The credential is in the Secret of the proxy, never in the sandbox.
#
# "I generate every object from one policy. Thus the securityContext, the NetworkPolicy and the admission
# policy cannot become different over time. I also validate the YAML against the target Kubernetes version
# in CI."
#
# **Drill 1.** *We applied a restricted-PSS label, and a Job with a privileged pod template went through. Is
# this a bug?* No. `enforce` applies to Pods. The API server creates the Job object. It rejects the Pods
# when the controller tries to make them, and the Job never makes progress.
#
# Also use `warn`/`audit` (they evaluate templates). Then monitor the conditions of the Job. Or add a VAP
# that matches Jobs.
#
# **Drill 2.** *Default-deny egress broke the sandbox. It cannot resolve DNS.* That is the purpose. A
# default-deny egress policy also blocks DNS. If you permit DNS, you open a DNS-exfiltration channel again.
#
# The sandbox reaches the proxy through a `hostAliases` entry (a ClusterIP that does not change). Thus it
# needs no DNS at all. The proxy resolves the upstream names.
#
# **Drill 3.** *Can we run this on kind in production? It costs less.* kind teaches the objects, but it is
# not a boundary. It has no gVisor (its `runsc` example ends `Failed`), and its NetworkPolicy dataplane
# fails open. Production needs a real runtime class (gVisor or a microVM) on dedicated, tainted nodes, with
# the same manifests that you validated on kind.
