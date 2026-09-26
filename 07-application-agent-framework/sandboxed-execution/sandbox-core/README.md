# sandbox-core — run untrusted code without ambient authority, in standard-library Python

After this you can explain (and have built) every part of a safe `run_code` tool: the attack probes that
show what leaks, a process sandbox with resource limits and a wall-clock kill, the execution contract with
budgets and exit reasons, an allowlisting egress proxy that injects a credential the code never holds, the
policy that renders to Kubernetes, and the warm-pool arithmetic that sizes and prices it. Pure standard
library (PyYAML and kubernetes-validate only for the manifest tests), about 1,000 lines you can read in a
sitting, plus five fill-in notebooks.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (why a code tool is the most dangerous tool; the isolation ladder).
2. `python3 -m pytest -q` — 49 tests, ~7 s, including "every probe is contained except egress" and "the
   rendered manifests validate against Kubernetes 1.34".
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
| [`02_a_process_sandbox`](notebooks/02_a_process_sandbox.ipynb) | set rlimits in the child, kill the process group on a wall-clock timeout, truncate output; map a signal to an exit reason; know the root/NPROC and grandchild caveats | §2, §3 | 1 h | T0 |
| [`03_the_execution_contract_and_policies`](notebooks/03_the_execution_contract_and_policies.ipynb) | build the request/result contract; write the idempotency key; render and lint the Pod/Job/NetworkPolicy/admission YAML | §3, §5 | 1 h | T0 |
| [`04_egress_and_secrets`](notebooks/04_egress_and_secrets.ipynb) | write the allowlist check and the credential injection; see why the network, not `HTTP_PROXY`, enforces egress; avoid the DNS trap | §4 | 45 min | T0 |
| [`05_pools_latency_and_cost`](notebooks/05_pools_latency_and_cost.ipynb) | size a warm pool (Little's law), pick a slot count (Erlang C), choose an isolation level for a latency budget, price a code tool | §6 | 45 min | T0 |

## Run it

```bash
cd sandbox-core
python3 -m pip install -e .            # standard library only; pyyaml + kubernetes-validate for the manifest tests
python3 -m pytest -q                    # 49 tests, ~7 s
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
| [`threats.py`](sandboxcore/threats.py) | ~260 | the scripted adversary: eight harmless probes (read a secret, phone home, fork bomb, disk fill, CPU spin, hang, flood), each with an expected verdict; a loopback trap and `run_probe` that judges containment by the real effect, not the exit code |
| [`executor.py`](sandboxcore/executor.py) | ~300 | `UnsafeExecutor` (for contrast) and `ProcessSandbox`: clean env, ephemeral home, rlimits set in the child, a process-group wall-clock kill, output truncation, exit-reason mapping, and an honest `isolation_report()` |
| [`contract.py`](sandboxcore/contract.py) | ~130 | `ExecutionRequest`/`ExecutionResult`/`Budgets`, the exit-reason vocabulary, the agent-core tool-result shape, the idempotency key and a run-once `ResultStore` |
| [`policy.py`](sandboxcore/policy.py) | ~275 | policy as data (tiers, egress allowlist, budget ceiling) with deny-by-default `evaluate` and `clamp`; `render_k8s()` → Namespace, NetworkPolicy×2, ResourceQuota, LimitRange, Job, ValidatingAdmissionPolicy, validated against K8s 1.34 |
| [`proxy.py`](sandboxcore/proxy.py) | ~125 | the allowlisting egress proxy that injects a credential the sandbox never holds; refuses CONNECT and says why; logs the header name, never its value |
| [`pool.py`](sandboxcore/pool.py) | ~135 | Little's law, Erlang C, a discrete-event warm-pool simulator, and cost per action — the numbers pinned to the scaling primer |
| [`audit.py`](sandboxcore/audit.py) | ~85 | one structured event per execution (the identity lab's field names + `budgets_used`/`exit_reason`/`policy_decision`); an exit-reason histogram for abuse detection |
| [`agent.py`](sandboxcore/agent.py) | ~150 | a scripted-LLM agent loop whose `run_code` routes through policy → sandbox → proxy, audited; injection scenarios that must fail closed |

## What the tests prove

`tests/` has one focused test per concept (49, offline, ~7 s):

- **The process sandbox contains seven of eight probes; egress is the exception.** `test_threats.py` runs
  every probe and asserts the resource abuses and secret reads are contained, the leak probes leak
  unsandboxed, and `egress_connect` is *not* contained by a process sandbox (rlimits do not touch sockets).
- **Exit reasons and limits are real.** `test_executor.py` shows a clean environment hides a planted secret,
  a wall-clock kill stops a sleeper and its grandchild, the CPU/file limits fire, output truncates while the
  program still finishes, and the isolation report is honest about the network and the root/NPROC caveat.
- **The rendered manifests validate against Kubernetes 1.34** and carry the load-bearing fields
  (`test_manifests.py`): restricted Pod Security labels, a non-retrying Job, `automountServiceAccountToken:
  false`, dropped capabilities, sized emptyDirs, default-deny egress opening only the proxy and DNS, and an
  admission policy with `validationActions: [Deny]`.
- **The pool arithmetic matches the scaling primer** (`test_pool.py`): 10 busy + 15 warming = 25 sandboxes,
  the Erlang C table (c=12 → P(wait) 0.449, c=16 → 0.057), and cost per action = actions/turn × cost/action.
- **The proxy injects for allowed hosts and denies others**, and never records a secret value
  (`test_proxy.py`); the contract's idempotency key runs code at most once (`test_contract.py`); and the
  agent's `run_code` fails closed under injection — reading a secret finds nothing, exfiltration is denied
  before running, the honest task works (`test_audit_agent.py`).

## Caveats: what is real at T0, and what is not

The process sandbox, the policy engine, the egress proxy, the pool model and the rendered manifests all run
and are tested here. **Latencies for the process sandbox are measured on the build host** (under shared
load); container/gVisor/microVM/GKE numbers are `(verify)` inputs or from upstream specs, labelled so.
Simulator output is labelled SIMULATED. A process sandbox is **not** a boundary against a kernel exploit or
the network — real isolation is a container, then gVisor, then a microVM, which is the lab
([`../sandbox-lab`](../sandbox-lab/)). And `RLIMIT_NPROC` does nothing as root, so fork-bomb protection needs
a UID drop; the core reports this rather than pretending.

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
