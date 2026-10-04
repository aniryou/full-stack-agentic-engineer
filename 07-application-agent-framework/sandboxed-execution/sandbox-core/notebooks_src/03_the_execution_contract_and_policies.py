# %% [markdown]
# # 03 · The execution contract and policy as data
#
# **Tier:** T0: a laptop, a Colab CPU or CI, free, seconds. The lab applies the rendered manifests for real
# (`../sandbox-lab`, notebooks 02 and 05). Here, the notebook builds and validates them offline.
#
# ## The one-minute version
# A sandbox is an **API with a contract**, not a place. The request carries the code, its inputs and a **budget for
# every resource the code can exhaust**. The result carries truncated output and an **exit reason** that the caller
# can act on. It also carries what the run used, and hashes that make the call auditable and safely repeatable.
#
# The decision "can this run, and under what limits?" is **policy as data**: tool tier, egress allowlist, filesystem
# rule, budget ceiling. The system enforces the policy **twice**:
#
# - by the executor at run time,
# - by the cluster (Pod Security *restricted*, a default-deny NetworkPolicy, a Job that does not
#   retry, a ValidatingAdmissionPolicy).
#
# The same policy object renders both. One honest limit: the egress hosts that a request lists are the *model's own
# claim*. Thus a check of them is a review aid, not enforcement. The NetworkPolicy is the enforcement.
#
# Side effects mean at-least-once delivery. Thus an **idempotency key** (turn, step, call index, args hash: the recipe
# of the scaling primer) lets a redelivered step return the stored result. The step does not run two times.
#
# Primer: `../PRIMER.md` §3 (the execution contract), §5 (sandboxes on Kubernetes). This notebook uses two things
# again: the idempotency recipe of the scaling primer
# (`../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md` §5.4) and the tool tiers
# of the identity primer (§4.2).

# %%
from sandboxcore import (Budgets, Effect, ExecutionRequest, ProcessSandbox, ResultStore,
                         SandboxPolicy, idempotency_key)

# %% [markdown]
# ## Worked example 1 — a request, a result, a tool message
# The result converts to the tool-result shape of agent-core: `{"ok": True, "data": ...}` or, on failure,
# `{"ok": False, "error": <exit reason>, "hint": ...}`. This is the exact shape that a `run_code` tool returns to the
# loop, so the model gets something that it can act on.

# %%
sb = ProcessSandbox()
good = sb.run(ExecutionRequest(code="print(6*7)", budgets=Budgets(cpu_s=1, wall_s=3)))
bad = sb.run(ExecutionRequest(code="while True: pass", budgets=Budgets(cpu_s=1, wall_s=10)))
print("ok run  ->", good.as_tool_result())
print("bad run ->", {k: bad.as_tool_result()[k] for k in ("ok", "error", "hint")})

# %% [markdown]
# ## Worked example 2 — the budget ceiling and clamping
# A policy carries a **maximum** budget. A request can ask for less, never more. `evaluate` denies a request over the
# ceiling, and `clamp` lowers each field that exceeds it. The model cannot widen its own envelope.

# %%
policy = SandboxPolicy(name="demo", max_budgets=Budgets(cpu_s=2, wall_s=5, memory_mb=256))
greedy = ExecutionRequest(code="pass", budgets=Budgets(cpu_s=99, wall_s=99, memory_mb=8000))
print("evaluate greedy:", policy.evaluate(greedy).effect.value, "->", policy.evaluate(greedy).reasons)
clamped = policy.clamp(ExecutionRequest(code="pass", budgets=Budgets(cpu_s=99, memory_mb=8000)))
print("clamped budgets: cpu_s", clamped.budgets.cpu_s, "memory_mb", clamped.budgets.memory_mb)

# %% [markdown]
# ## Worked example 3 — the same policy, rendered to Kubernetes
# `render_k8s()` turns the policy into the objects that enforce it in a cluster. Each object is a plain dict. It
# serialises to the YAML that you apply, and it validates against the Kubernetes 1.34 schemas. Read three details:
#
# - The deadline of the Job is the **startup allowance plus the wall budget**. A Kubernetes deadline also counts the
#   time to schedule the pod and the image pull. `timeout` enforces the wall budget itself inside the pod.
# - Every resource **request is at most its limit**. If not, the API server rejects the pod, and a schema check does
#   not see the problem.
# - The pod has **no DNS**. It finds the proxy through `hostAliases` and the pinned ClusterIP of the proxy Service.

# %%
objs = policy.render_k8s()
for o in objs:
    print(f"  {o['kind']:32} {o['metadata']['name']}")
job = next(o for o in objs if o["kind"] == "Job")
pod = job["spec"]["template"]["spec"]
print("\nthe Job's pod, the load-bearing fields:")
print("  runtimeClassName:", pod["runtimeClassName"], "| automountServiceAccountToken:",
      pod["automountServiceAccountToken"], "| dnsPolicy:", pod["dnsPolicy"], "| hostAliases:", pod["hostAliases"])
print("  Job activeDeadlineSeconds:", job["spec"]["activeDeadlineSeconds"],
      f"(= startup allowance {policy.startup_allowance_s} s + wall budget {int(policy.max_budgets.wall_s)} s)")
print("  command:", pod["containers"][0]["command"][:5], "...")
print("  resources:", pod["containers"][0]["resources"])
print("  container securityContext:", pod["containers"][0]["securityContext"])

# %% [markdown]
# ## Worked example 4 — idempotency: a redelivery does not run the code again
# Delivery is at-least-once, so a redelivered step must not run the code two times. The key is turn + step + call
# index + a hash of the arguments. A `ResultStore` claims the key *before* the run. It returns the stored result on a
# later delivery. It refuses (`InFlight`) a delivery that arrives while the first is still in progress.
#
# Its docstring states its limits:
#
# - It is one process, in memory. A crash forgets it, and a real store writes the claim durably.
# - The side effects of the code itself need the key sent downstream too.

# %%
store = ResultStore()
key = idempotency_key("turn-42", step=3, call_index=0, args={"code": "print('side effect')"})
runs = {"n": 0}


def do_run():
    runs["n"] += 1
    return sb.run(ExecutionRequest(code="print('side effect')", budgets=Budgets(cpu_s=1, wall_s=3)))


r1, replayed1 = store.run_once(key, do_run)
r2, replayed2 = store.run_once(key, do_run)      # the redelivery
print(f"key = {key}")
print(f"first delivery ran: {not replayed1}; second delivery ran: {not replayed2}; total executions: {runs['n']}")

# %% [markdown]
# ## Exercise 3.1 — build the idempotency key
# Implement `make_key(turn_id, step, call_index, args)` to the recipe of the scaling primer. Join the four parts with
# `:`, and use a stable hash for the args (use `sandboxcore.digest`). Two calls with the same arguments must collide.
# Different arguments must not.

# %%
from sandboxcore import digest

# %% exercise
def make_key(turn_id, step, call_index, args):
    ### BEGIN SOLUTION
    return f"{turn_id}:{step}:{call_index}:{digest(args)}"
    ### END SOLUTION

# %% check
assert make_key("t", 1, 0, {"a": 1}) == make_key("t", 1, 0, {"a": 1})
assert make_key("t", 1, 0, {"a": 1}) != make_key("t", 1, 0, {"a": 2})
assert make_key("t", 1, 0, {"a": 1}) == idempotency_key("t", 1, 0, {"a": 1})
print("✅ same arguments -> same key (safe replay); different arguments -> different key")

# %% [markdown]
# ## Exercise 3.2 — a deny-by-default egress decision, from scratch
# Write `egress_decision(allowlist, needs_confirm, hosts)` **without a call to the library**. Return:
#
# - `Effect.DENY` if a requested host is not *exactly* on the allowlist,
# - `Effect.CONFIRM` if the request lists hosts, all of them are on the allowlist, and the policy wants a human to
#   confirm egress,
# - else `Effect.ALLOW`. This is also the result for a request with no hosts.
#
# The check compares your function with `SandboxPolicy.evaluate` on many cases. Then answer in a comment: a hijacked
# model that does not list its destination gets which effect? And what actually stops its socket?

# %% exercise
def egress_decision(allowlist, needs_confirm, hosts):
    ### BEGIN SOLUTION
    if any(h not in allowlist for h in hosts):
        return Effect.DENY
    if hosts and needs_confirm:
        return Effect.CONFIRM
    return Effect.ALLOW
    # An undeclared destination gets ALLOW: the declared list is the model's claim. Only the network layer
    # (default-deny NetworkPolicy, --network none, an empty netns) stops the socket.
    ### END SOLUTION

# %% check
import itertools
allow = ("api.github.com", "pypi.org")
cases = [(), ("api.github.com",), ("evil.example",), ("api.github.com", "evil.example"),
         ("api.github.com.evil.example",), ("pypi.org", "api.github.com")]
for needs, hosts in itertools.product((False, True), cases):
    want = SandboxPolicy(egress_allowlist=allow, egress_needs_confirm=needs).evaluate(
        ExecutionRequest(code="pass", egress=hosts)).effect
    assert egress_decision(allow, needs, hosts) is want, (needs, hosts, want)
print("✅ unknown host -> deny; allowed host -> allow or confirm, per policy — and nothing declared -> allow")

# %% [markdown]
# ## Exercise 3.3 — assert the manifest is safe
# A reviewer must be able to examine a rendered Job in seconds. Write `job_is_hardened(job)`. It returns True only
# when all these conditions are true:
#
# - The pod sets `runtimeClassName`.
# - The pod does not automount the service-account token.
# - The pod has `restartPolicy: Never`, `backoffLimit: 0` and `dnsPolicy: None`.
# - A container has `allowPrivilegeEscalation: False` and `capabilities.drop == ["ALL"]`.
# - The Job has an `activeDeadlineSeconds` of at least 60 s. A tighter value kills a cold pod before its code starts.

# %% exercise
def job_is_hardened(job):
    ### BEGIN SOLUTION
    spec = job["spec"]
    pod = spec["template"]["spec"]
    ctr = pod["containers"][0]["securityContext"]
    return (bool(pod.get("runtimeClassName"))
            and pod.get("automountServiceAccountToken") is False
            and pod.get("restartPolicy") == "Never"
            and pod.get("dnsPolicy") == "None"
            and spec.get("backoffLimit") == 0
            and spec.get("activeDeadlineSeconds", 0) >= 60
            and ctr.get("allowPrivilegeEscalation") is False
            and ctr.get("capabilities", {}).get("drop") == ["ALL"])
    ### END SOLUTION

# %% check
job = next(o for o in SandboxPolicy(egress_allowlist=("h",)).render_k8s() if o["kind"] == "Job")
assert job_is_hardened(job)
# a tampered copy fails the check
import copy
for field, value in ((("spec", "backoffLimit"), 6), (("spec", "activeDeadlineSeconds"), 15),
                     (("spec", "template", "spec", "dnsPolicy"), "ClusterFirst"),
                     (("spec", "template", "spec", "automountServiceAccountToken"), True)):
    bad = copy.deepcopy(job)
    node = bad
    for k in field[:-1]:
        node = node[k]
    node[field[-1]] = value
    assert not job_is_hardened(bad), f"missed {'.'.join(field)} = {value!r}"
print("✅ the linter catches retries, a deadline shorter than a cold start, cluster DNS and a mounted token")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "I treat the sandbox as an API with a contract. The request is code, plus inputs, plus a
# budget for every resource that the code can exhaust: CPU, wall time, memory, processes, disk, output. It also
# carries the policy that applies.
#
# "The result contains truncated output and an exit reason that the model can act on. It also gives the source of
# that reason, because a program can forge its own. The result also has what the run used, and a result hash.
#
# "The policy is data, not prose: tool tier, egress allowlist, filesystem rule, budget ceiling. I enforce it twice.
# The executor clamps and checks at run time. The cluster enforces the same intent with these controls:
#
# - Pod Security restricted,
# - a default-deny NetworkPolicy with no DNS,
# - a Job that does not retry, whose deadline permits a cold start,
# - a ValidatingAdmissionPolicy that rejects each pod without the controls.
#
# "I render both from one object, so they cannot drift apart. The hosts that a request declares are only the claim of
# the model. The network policy enforces egress.
#
# "Delivery is at-least-once. Thus every execution carries an idempotency key of turn, step, call index and an
# argument hash. The executor claims the key before the run. Thus a redelivered step returns the stored result and
# does not run a second time."
#
# **Drill questions**
# 1. *Why enforce policy in the cluster when the executor already checks it?* Defence in depth. If the runtime code is
#    incorrect or something bypasses it, the cluster still rejects a pod that does not conform. Pod Security and
#    the admission policy do this. Deterministic controls outrank a single check.
# 2. *Why does a code tool need an idempotency key?* Its executions can have side effects, and delivery is
#    at-least-once. Without a key, a redelivery runs the code (and its side effect) two times.
# 3. *A Job has `backoffLimit: 6`. What breaks?* Non-idempotent code runs again up to six times on failure. Set
#    `backoffLimit: 0` and `restartPolicy: Never`. Then put a limit on the full Job with `activeDeadlineSeconds`.
# 4. *The Job's `activeDeadlineSeconds` equals the code's 5-second wall budget. What occurs on a busy cluster?* The
#    deadline counts from the start of the Job. Thus the time to schedule the pod, the scale-up and the image pull
#    use it up. Kubernetes kills a cold pod (`DeadlineExceeded`) before the code runs. Give the Job a startup
#    allowance, and enforce the wall budget inside the pod.
