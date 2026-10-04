# sandbox-core — run untrusted code without ambient authority, in standard-library Python

After this core, you can explain every part of a safe `run_code` tool, and you built each part:

- The attack probes that show what leaks.
- A process sandbox with resource limits, a wall-clock kill and a UID of its own for each execution.
- The execution contract with budgets and exit reasons.
- An egress proxy with an allowlist, which injects a credential that the code never holds.
- The policy that renders to Kubernetes.
- The warm-pool arithmetic that calculates its size and its price.

The core uses only the standard library. Only the manifest tests also use PyYAML and kubernetes-validate. The core
has about 2,000 lines that you can read in an afternoon. It also has five notebooks with blanks to fill in.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2. These sections tell why a code tool is the most dangerous tool.
   They also describe the isolation ladder.
2. Run `python3 -m pip install -e ".[dev]" && python3 -m pytest -q`. It runs 100 tests in ~60 s. The tests
   include these three:
   - "with its own UID every probe is contained except egress"
   - "an undeclared exfiltration leaks through a process sandbox"
   - "the rendered manifests validate against Kubernetes 1.34"
3. Open [`notebooks/01_the_threat_model.ipynb`](notebooks/01_the_threat_model.ipynb). See a secret leak from code
   with no sandbox. Then see the process sandbox contain it.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU, no Docker and no network.*

Each notebook starts with "The one-minute version". It then shows worked examples against the code, and gives
exercises, each with a check that prints ✅. It ends with "In a design review". The finished versions are in
[`solutions/`](solutions/). The core and the primer take about 4 hours (module 07.5).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_threat_model`](notebooks/01_the_threat_model.ipynb) | Run the attack probes through the executor with no sandbox, and see what leaks. Map each risk to the control that bounds it. Give the honest verdict for egress. | §1 | 45 min | T0 |
| [`02_a_process_sandbox`](notebooks/02_a_process_sandbox.ipynb) | Set rlimits in the child, hold a wall-clock deadline, and stream and cap the output. Map a signal to an exit reason. Know which reasons a program can forge. See why `HOME` is not a boundary. See why a `setsid()` escapee needs a per-execution UID. | §2, §3 | 1 h | T0 |
| [`03_the_execution_contract_and_policies`](notebooks/03_the_execution_contract_and_policies.ipynb) | Build the request/result contract. Write the idempotency key. Write the egress decision, and see why a declared host is not a control. Render and lint the Pod/Job/NetworkPolicy/RuntimeClass/admission YAML. | §3, §5 | 1 h | T0 |
| [`04_egress_and_secrets`](notebooks/04_egress_and_secrets.ipynb) | Write the allowlist check and the outbound headers. See the proxy refuse a redirect and a CONNECT against loopback servers. See an undeclared raw socket leak through a process sandbox. Keep DNS closed. | §4 | 45 min | T0 |
| [`05_pools_latency_and_cost`](notebooks/05_pools_latency_and_cost.ipynb) | Measure the process-level cold start. See why Little's law is the floor. Calculate the size of a replace-after-use pool with Erlang C, and do a check of it with a simulation. Calculate the price of a code tool, with the cold start included. | §6 | 45 min | T0 |

## Run it

```bash
cd sandbox-core
python3 -m pip install -e ".[dev]"     # the library is stdlib only; dev adds pytest, jupyter, pyyaml, kubernetes-validate
python3 -m pytest -q                    # 100 tests, ~60 s
python3 tools/build_notebooks.py        # rebuild notebooks/ and solutions/
python3 -m jupyterlab notebooks         # do the exercises
```

You can import the library with nothing installed:

```python
from sandboxcore import ProcessSandbox, ExecutionRequest, Budgets
sb = ProcessSandbox()
r = sb.run(ExecutionRequest(code="print(6*7)", budgets=Budgets(cpu_s=1, wall_s=3)))
print(r.exit_reason, r.stdout.strip())      # ok 42
print(sb.isolation_report()["network_blocked"])   # False — a process sandbox does not block egress
```

## The whole library

Read the modules in this order. Each module starts with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`threats.py`](sandboxcore/threats.py) | ~320 | The scripted adversary: nine harmless probes, each with an expected verdict. The probes: read a secret, read a key by absolute path, phone home, fork bomb, disk fill, CPU spin, hang, flood, outlive the call. A loopback trap. `run_probe`, which decides containment from the real effect, not from the exit code. |
| [`executor.py`](sandboxcore/executor.py) | ~550 | `UnsafeExecutor` (for contrast) and `ProcessSandbox`. The sandbox has a clean env, a 0700 workspace, and a UID per execution when it runs as root. The child itself lowers its rlimits (no `preexec_fn`). It has a wall-clock deadline. It reads the output as it streams, and puts a cap on it. It has a process-group kill and a sweep by UID, exit reasons with their source, and an honest `isolation_report()`. |
| [`contract.py`](sandboxcore/contract.py) | ~170 | `ExecutionRequest`/`ExecutionResult`/`Budgets`. The exit-reason vocabulary and the source of each reason. The agent-core tool-result shape. The idempotency key. A `ResultStore` that claims a key before the run, and says what it does not guarantee. |
| [`policy.py`](sandboxcore/policy.py) | ~330 | Policy as data (tiers, egress allowlist, budget ceiling) with deny-by-default `evaluate` and `clamp`. `render_k8s()` makes these objects: Namespace, RuntimeClass, NetworkPolicy×2 (proxy only, no DNS), the proxy Service, ResourceQuota, LimitRange, Pod, Job, ValidatingAdmissionPolicy and its binding. They validate against K8s 1.34. |
| [`proxy.py`](sandboxcore/proxy.py) | ~175 | The egress proxy with an allowlist, which injects a credential that the sandbox never holds. It sends no request to the target of a redirect, it drops caller credentials, and it redacts echoes. It refuses CONNECT and says why. It logs the header name, never its value. |
| [`pool.py`](sandboxcore/pool.py) | ~180 | Little's law (the mean occupancy), Erlang C (the slot count), a three-mode pool simulator, and the cost per action. The tests pin its rate to the scaling primer. |
| [`audit.py`](sandboxcore/audit.py) | ~90 | One structured event per execution. It has the field names of the identity lab, and also `budgets_used`/`exit_reason`/`exit_reason_source`/`policy_decision`. An exit-reason histogram that can drop forgeable reasons. |
| [`agent.py`](sandboxcore/agent.py) | ~180 | A scripted-LLM agent loop whose `run_code` goes through policy and the sandbox, with an audit record. Injection scenarios that show which defence holds. They also show that an undeclared socket leaks through a process sandbox. |

## What the tests prove

`tests/` has one focused test per concept: 92 tests, and also 8 notebook-tooling checks. They run offline, in
~60 s in all.

- **What the process sandbox contains depends on its UID, and egress is never contained.**
  `test_threats.py` runs every probe, with three results:
  - With a per-execution UID (as root), the sandbox contains all the probes except `egress_connect`.
  - With `drop_to_uid=None` (a laptop), the key read and the session escape leak.
  - With no sandbox, the secret, key, egress and escape probes all leak.

  The harmlessness tests make sure of three things. The target of every probe comes from the harness (loopback
  only). Every loop has a bound. The key probe never opens a real home directory, even when the executor passes
  `HOME` through.
- **Limits, exit reasons and streaming are real.** `test_executor.py` shows these results:
  - A clean environment hides a planted secret.
  - A wall-clock kill stops a sleeper and its grandchild.
  - A `setsid()` escapee cannot hold the call open, and the sandbox sweeps it by UID.
  - A flood ends as `output_limit` within wall_s + 1 s, and the memory of the parent does not grow.
  - A key opens by absolute path unless the UID is different.
  - The workspace is 0700.
  - A forged `MemoryError` gets the label `reason_source: code`.
  - The sandbox measures CPU and peak memory per execution. It uses `wait4`, so it does not lose them when a
    child exits while the parent still reads.

  Without a UID switch, the `pids` budget is the run's own process tree. The other threads of a busy user do not
  count.
  A fork burst between two 50 ms counts runs on to the `RLIMIT_NPROC` backstop (32 tasks past the budget, plus any others).
  The others are the tasks that the other processes of the user free during the run. The sandbox still reports
  the burst as `pids`. When more than one limit holds, `wall_timeout` beats `pids`, and `pids` beats
  `output_limit`.
- **The rendered manifests validate against Kubernetes 1.34 and pass admission** (`test_manifests.py`). The tests
  examine these properties:
  - Restricted Pod Security labels.
  - The RuntimeClass.
  - A Job with no retries. Its deadline gives time for a cold start, while `timeout` enforces the wall budget in
    the pod.
  - Every request ≤ its limit.
  - `automountServiceAccountToken: false`.
  - Dropped capabilities.
  - Sized emptyDirs.
  - Egress to the proxy only, with no DNS (hostAliases to a pinned ClusterIP).
  - An admission policy with `validationActions: [Deny]`.
- **The pool arithmetic** (`test_pool.py`). Little's law gives 25 slots on average, and Erlang C shows that this
  pool is unstable. The tests also show these results:
  - 31 slots for P(wait) ≤ 0.2 (the §6 table).
  - 34 at the scaling primer's 5.42/s.
  - A simulator that agrees in all three modes. The tests pin the simulated figures of the primer.
  - Erlang C stays stable at 160+ Erlangs.
  - A cost per execution that charges the cold start: 5 sandbox-seconds, or 6.2 with the headroom of the fleet.
- **The proxy against real loopback upstreams** (`test_proxy.py`). The tests show these results:
  - Injection for hosts on the allowlist.
  - 403 before any request for other hosts.
  - The proxy does not go to the target of a 302 to an off-allowlist host. Thus the credential never reaches
    that host.
  - The proxy strips caller credentials and hop-by-hop headers.
  - The proxy redacts an echoed secret.
  - The proxy refuses CONNECT with 405.
  - The proxy ignores environment proxies.

  The contract (`test_contract.py`) claims an idempotency key before the run, and refuses a concurrent
  redelivery. The agent (`test_audit_agent.py`) gives three results. A read of a secret finds nothing. The policy
  denies a *declared* exfiltration before the run. An *undeclared* exfiltration leaks through a process sandbox,
  and the audit records it as `allow`. This is the honest result.

## Caveats: what is real at T0, and what is not

The process sandbox, the policy engine, the egress proxy, the pool model and the rendered manifests all run here.
The tests also examine them here. **The latencies for the process sandbox are measurements on the build host**,
under shared load. Notebook 05 measures them again on your host. The container/gVisor/microVM/GKE numbers are
`(verify)` inputs or come from upstream specs, and their labels say so. Simulator output has the label SIMULATED.

A process sandbox is **not** a boundary against a kernel exploit or the network. Real isolation is a container,
then gVisor, then a microVM, and these are in the lab ([`../sandbox-lab`](../sandbox-lab/)).

Also, three of its controls need root to switch each execution to its own UID. The three controls and their
limits are these:

- `RLIMIT_NPROC` does nothing as root, and as a user your own processes share it.
  Thus the parent counts the run's own process tree every 50 ms instead.
  A fork burst between two counts runs on to the `RLIMIT_NPROC` backstop: 32 tasks past the budget plus any others.
  The others are the tasks that your other processes free while it runs.
- Your files stay readable by absolute path.
- The sandbox cannot find a `setsid()` escapee.

The core reports which of these controls hold (`isolation_report()`), and it does not pretend.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources. Then run `python3 tools/build_notebooks.py`. `make check` runs the tests and both
notebook passes. On Colab, the first cell of each notebook clones the repository and installs this package (see
[`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../sandbox-lab/`](../sandbox-lab/). It has these parts:

- The same controls on real Docker (hardened `docker run`, gVisor when present).
- A pod-per-execution runner on kind.
- The egress proxy behind a NetworkPolicy.
- An agent whose tools fail closed.
- A GKE Sandbox (gVisor) deploy.

[`../../../COMPUTE.md`](../../../COMPUTE.md) tells where each tier runs and what it costs. The license is MIT.
