# sandbox-core — run untrusted code without ambient authority, in standard-library Python

After this you can explain (and have built) every part of a safe `run_code` tool: the attack probes that
show what leaks, a process sandbox with resource limits, a wall-clock kill and a UID of its own per
execution, the execution contract with budgets and exit reasons, an allowlisting egress proxy that injects
a credential the code never holds, the policy that renders to Kubernetes, and the warm-pool arithmetic that
sizes and prices it. Pure standard library (PyYAML and kubernetes-validate only for the manifest tests),
about 2,000 lines you can read in an afternoon, plus five fill-in notebooks.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (why a code tool is the most dangerous tool; the isolation ladder).
2. `python3 -m pip install -e ".[dev]" && python3 -m pytest -q` — 81 tests, ~30 s, including "with its own
   UID every probe is contained except egress", "an undeclared exfiltration leaks through a process
   sandbox" and "the rendered manifests validate against Kubernetes 1.34".
3. Open [`notebooks/01_the_threat_model.ipynb`](notebooks/01_the_threat_model.ipynb) and watch a secret leak
   from unsandboxed code, then get contained by the process sandbox.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU, no Docker and no network.* Each
notebook opens with "The one-minute version", works examples against the code, sets exercises with a check
that prints ✅, and ends with "In a design review". Finished versions are in [`solutions/`](solutions/).
About 4 hours with the primer (module 07.5).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_threat_model`](notebooks/01_the_threat_model.ipynb) | run the attack probes through the unsandboxed executor and see what leaks; map each risk to the control that bounds it; give the honest verdict for egress | §1 | 45 min | T0 |
| [`02_a_process_sandbox`](notebooks/02_a_process_sandbox.ipynb) | set rlimits in the child, hold a wall-clock deadline, stream and cap output; map a signal to an exit reason and know which reasons a program can forge; see why `HOME` is not a boundary and a `setsid()` escapee needs a per-execution UID | §2, §3 | 1 h | T0 |
| [`03_the_execution_contract_and_policies`](notebooks/03_the_execution_contract_and_policies.ipynb) | build the request/result contract; write the idempotency key; write the egress decision and see why a declared host is not a control; render and lint the Pod/Job/NetworkPolicy/RuntimeClass/admission YAML | §3, §5 | 1 h | T0 |
| [`04_egress_and_secrets`](notebooks/04_egress_and_secrets.ipynb) | write the allowlist check and the outbound headers; watch the proxy refuse a redirect and a CONNECT against loopback servers; watch an undeclared raw socket leak through a process sandbox; keep DNS closed | §4 | 45 min | T0 |
| [`05_pools_latency_and_cost`](notebooks/05_pools_latency_and_cost.ipynb) | measure the process-level cold start; see why Little's law is the floor; size a replace-after-use pool with Erlang C and check it by simulation; price a code tool, cold start included | §6 | 45 min | T0 |

## Run it

```bash
cd sandbox-core
python3 -m pip install -e ".[dev]"     # the library is stdlib only; dev adds pytest, jupyter, pyyaml, kubernetes-validate
python3 -m pytest -q                    # 81 tests, ~30 s
python3 tools/build_notebooks.py        # rebuild notebooks/ and solutions/
python3 -m jupyterlab notebooks         # do the exercises
```

The library imports with nothing installed:

```python
from sandboxcore import ProcessSandbox, ExecutionRequest, Budgets
sb = ProcessSandbox()
r = sb.run(ExecutionRequest(code="print(6*7)", budgets=Budgets(cpu_s=1, wall_s=3)))
print(r.exit_reason, r.stdout.strip())      # ok 42
print(sb.isolation_report()["network_blocked"])   # False — a process sandbox does not block egress
```

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`threats.py`](sandboxcore/threats.py) | ~320 | the scripted adversary: nine harmless probes (read a secret, read a key by absolute path, phone home, fork bomb, disk fill, CPU spin, hang, flood, outlive the call), each with an expected verdict; a loopback trap and `run_probe` that judges containment by the real effect, not the exit code |
| [`executor.py`](sandboxcore/executor.py) | ~550 | `UnsafeExecutor` (for contrast) and `ProcessSandbox`: clean env, a 0700 workspace, a UID per execution when root, rlimits lowered by the child itself (no `preexec_fn`), a wall-clock deadline, output read as it streams and capped, a process-group kill plus a sweep by UID, exit reasons with their source, and an honest `isolation_report()` |
| [`contract.py`](sandboxcore/contract.py) | ~170 | `ExecutionRequest`/`ExecutionResult`/`Budgets`, the exit-reason vocabulary and where each reason came from, the agent-core tool-result shape, the idempotency key and a `ResultStore` that claims a key before running (and says what it does not guarantee) |
| [`policy.py`](sandboxcore/policy.py) | ~330 | policy as data (tiers, egress allowlist, budget ceiling) with deny-by-default `evaluate` and `clamp`; `render_k8s()` → Namespace, RuntimeClass, NetworkPolicy×2 (proxy only, no DNS), the proxy Service, ResourceQuota, LimitRange, Pod, Job, ValidatingAdmissionPolicy + binding, validated against K8s 1.34 |
| [`proxy.py`](sandboxcore/proxy.py) | ~175 | the allowlisting egress proxy that injects a credential the sandbox never holds; no redirect following, caller credentials dropped, echoes redacted; refuses CONNECT and says why; logs the header name, never its value |
| [`pool.py`](sandboxcore/pool.py) | ~180 | Little's law (the mean occupancy), Erlang C (the slot count), a three-mode pool simulator, and cost per action — the rate pinned to the scaling primer |
| [`audit.py`](sandboxcore/audit.py) | ~90 | one structured event per execution (the identity lab's field names + `budgets_used`/`exit_reason`/`exit_reason_source`/`policy_decision`); an exit-reason histogram that can drop forgeable reasons |
| [`agent.py`](sandboxcore/agent.py) | ~180 | a scripted-LLM agent loop whose `run_code` goes through policy and the sandbox, audited; injection scenarios that show which defence holds — and that an undeclared socket leaks through a process sandbox |

## What the tests prove

`tests/` has one focused test per concept (81, offline, ~30 s):

- **What the process sandbox contains depends on its UID, and egress is never contained.**
  `test_threats.py` runs every probe: with a per-execution UID (as root) all are contained except
  `egress_connect`; with `drop_to_uid=None` (a laptop) the key read and the session escape leak; unsandboxed,
  the secret, key, egress and escape probes all leak. Harmlessness tests check every probe's target comes
  from the harness (loopback only), every loop is bounded, and the key probe never opens a real home
  directory even when the executor passes `HOME` through.
- **Limits, exit reasons and streaming are real.** `test_executor.py` shows a clean environment hides a
  planted secret, a wall-clock kill stops a sleeper and its grandchild, a `setsid()` escapee cannot hold the
  call open and is swept by UID, a flood ends as `output_limit` within wall_s + 1 s without growing the
  parent's memory, a key opens by absolute path unless the UID differs, the workspace is 0700, a forged
  `MemoryError` is labelled `reason_source: code`, and CPU and peak memory are measured per execution (with
  `wait4`, so a child that exits while the parent is still reading does not lose them).
- **The rendered manifests validate against Kubernetes 1.34 and would be admitted** (`test_manifests.py`):
  restricted Pod Security labels, the RuntimeClass, a non-retrying Job whose deadline allows for a cold start
  while `timeout` enforces the wall budget in the pod, every request ≤ its limit, `automountServiceAccountToken:
  false`, dropped capabilities, sized emptyDirs, egress to the proxy only with no DNS (hostAliases to a pinned
  ClusterIP), and an admission policy with `validationActions: [Deny]`.
- **The pool arithmetic** (`test_pool.py`): Little's law gives 25 slots on average, which Erlang C shows is
  unstable; 31 slots for P(wait) ≤ 0.2 (the §6 table), 34 at the scaling primer's 5.42/s, a simulator that
  agrees in all three modes (the primer's simulated figures are pinned), Erlang C stable at 160+ Erlangs, and
  cost per execution that charges the cold start (5 sandbox-seconds; 6.2 with the fleet's headroom).
- **The proxy against real loopback upstreams** (`test_proxy.py`): injection for allowed hosts, 403 before
  any request for others, a 302 to an off-allowlist host not followed (the credential never reaches it),
  caller credentials and hop-by-hop headers stripped, an echoed secret redacted, CONNECT refused with 405, and
  environment proxies ignored. The contract (`test_contract.py`) claims an idempotency key before running and
  refuses a concurrent redelivery. The agent (`test_audit_agent.py`): reading a secret finds nothing, a
  *declared* exfiltration is denied before running, and an *undeclared* one leaks through a process sandbox
  and is audited as `allow` — the honest result.

## Caveats: what is real at T0, and what is not

The process sandbox, the policy engine, the egress proxy, the pool model and the rendered manifests all run
and are tested here. **Latencies for the process sandbox are measured on the build host** (under shared
load, and notebook 05 re-measures them on yours); container/gVisor/microVM/GKE numbers are `(verify)` inputs
or from upstream specs, labelled so. Simulator output is labelled SIMULATED. A process sandbox is **not** a
boundary against a kernel exploit or the network — real isolation is a container, then gVisor, then a
microVM, which is the lab ([`../sandbox-lab`](../sandbox-lab/)). And three of its controls need root to
switch each execution to its own UID: `RLIMIT_NPROC` does nothing as root and is shared with your own
processes as a user, your files stay readable by absolute path, and a `setsid()` escapee cannot be found.
The core reports which of these hold (`isolation_report()`) rather than pretending.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN
SOLUTION` blocks). Edit the sources, then `python3 tools/build_notebooks.py`; `make check` runs the tests and
both notebook passes. On Colab, each notebook's first cell clones the repo and installs this package (see
[`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../sandbox-lab/`](../sandbox-lab/) for the same controls on real Docker (hardened `docker run`, gVisor
when present), a pod-per-execution runner on kind, the egress proxy behind a NetworkPolicy, an agent whose
tools fail closed, and a GKE Sandbox (gVisor) deploy. Where each tier runs and what it costs:
[`../../../COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
