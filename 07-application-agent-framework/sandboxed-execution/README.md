# sandboxed-execution — run the model's code without handing it your keys

After this topic, you can take the `run_code` tool of an agent, or any tool that runs untrusted programs that a
model wrote. You can then tell exactly these four things about it:

- What it can reach when an attacker takes control of the model.
- Which control bounds each risk.
- How much isolation you need.
- What a pool of sandboxes costs.

You can also render the Kubernetes objects that enforce these controls, and do a check of them.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §1–§2 (30 min). These sections tell why a code tool is the most dangerous tool.
   They also describe the isolation ladder from process to microVM.
2. Run `cd sandbox-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q`. It runs 100 tests in ~60 s.
   Then open
   [`sandbox-core/notebooks/01_the_threat_model.ipynb`](sandbox-core/notebooks/01_the_threat_model.ipynb).
   See a secret leak from code with no sandbox. Then see the sandbox contain the same secret.
3. When you have Docker or a cluster, go to [`sandbox-lab/`](sandbox-lab/). The lab hardens a real container.
   It runs a pod-per-execution on kind, and it deploys a GKE Sandbox (gVisor) node pool.

## What you get

*Tiers: **T0** = laptop / Colab CPU / CI, free. All the concepts are at T0. **T0 + Docker** = the container rungs
and kind on your own machine, also free. **T3** = the Google Cloud deployment, optional. No part uses a GPU. Thus
the T1/T2 tiers of the repository (one or more GPUs) do not apply.*

Each notebook starts with "The one-minute version". It then shows worked examples against the code,
and gives exercises, each with a check that prints ✅. It ends with "In a design review". The finished versions are in
[`sandbox-core/solutions/`](sandbox-core/solutions/).

| Path | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`sandbox-core`](sandbox-core/) `01_the_threat_model` | Name the blast radius of a code tool. Map each risk (secret, egress, fork bomb, disk, CPU, output) to the control that bounds it. See what leaks with no sandbox. | §1 | 45 min | T0 |
| `02_a_process_sandbox` | Build the T0 boundary: a clean env, a 0700 workspace, a UID per execution, rlimits, a wall-clock kill, and streamed and capped output. Know what it cannot stop. It cannot stop the network. Without the UID, it also cannot protect your files or stop `setsid()` escapes. | §2, §3 | 1 h | T0 |
| `03_the_execution_contract_and_policies` | Design the request/result contract with budgets and exit reasons. Write policy as data. Render Pod Security / NetworkPolicy / Job / admission YAML. Give each execution a key for safe replay. | §3, §5 | 1 h | T0 |
| `04_egress_and_secrets` | Put an egress proxy with an allowlist in front of the sandbox. The proxy injects a credential that the code never holds. It never sends a request to the target of a redirect. See why the network enforces egress, and not `HTTP_PROXY` or a declared host. | §4 | 45 min | T0 |
| `05_pools_latency_and_cost` | See why Little's law is the floor. Calculate the size of a replace-after-use pool with Erlang C. Select an isolation level for a latency budget. Put a number on the cost per action. | §6 | 45 min | T0 |
| [`sandbox-lab`](sandbox-lab/) | Use the same controls on a process with a network namespace, on real Docker, on kind and on GKE Sandbox (gVisor). The lab has an agent with `run_code`/`fetch_url` tools. If the host gives a network namespace, these tools fail closed under injection. | §2–§9 | ~8 h | T0, T0 + Docker, T3 |

## Run it

```bash
cd sandbox-core
python3 -m pip install -e ".[dev]"     # the library is stdlib only; dev adds pytest, jupyter, pyyaml, kubernetes-validate
python3 -m pytest -q                    # 100 tests, ~60 s
python3 tools/build_notebooks.py        # (re)build notebooks/ and solutions/
python3 -m jupyterlab notebooks         # do the exercises
```

To import the library, you do not need to install anything:

```python
from sandboxcore import ProcessSandbox, ExecutionRequest, Budgets
sb = ProcessSandbox()
r = sb.run(ExecutionRequest(code="print(6*7)", budgets=Budgets(cpu_s=1, wall_s=3)))
print(r.exit_reason, r.stdout)          # ok 42
print(sb.isolation_report()["network_blocked"])   # False — a process sandbox does not block egress
```

## How it fits

First, read the [agent-core loop and tool contract](../agent-fundamentals/agent-core/) (07.1). This topic makes
its `run_code` tool safe. The topic uses two neighbour topics directly:

- The [identity and security primer](../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md):
  §2 threat model, §4.2 tool tiers, §5 secrets, §6.2 code execution, §9 audit. This topic cites these sections
  and does not repeat them.
- The [scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md):
  §1 cost per action, §5.4 idempotency.

The Kubernetes controls use the same conventions as the [GPU scheduling lab](../../03-kubernetes-gpu/gpu-scheduling/):
typed manifest builders, a kind deploy script, and offline manifest validation.
[`COMPUTE.md`](../../COMPUTE.md) tells where each tier runs and what it costs. [`CURRICULUM.md`](../../CURRICULUM.md)
gives the learning path.

## Going further / caveats

- **What is real at T0, and what is not.** The process sandbox, the policy engine, the egress proxy, the pool
  model and the rendered manifests all run on a laptop. The tests also examine them there. The latencies for the
  process sandbox are **measurements on the build host**. The container/gVisor/VM/GKE numbers are `(verify)`
  inputs or come from upstream specs, and their labels say so. Simulator output has the label SIMULATED.
- **A process sandbox is not a security boundary against a determined attacker.** It contains resource abuse and
  removes ambient authority. But it does not stop kernel exploits or the network. Real isolation is a container,
  then gVisor (`runsc`), then a microVM (Firecracker/Kata). The lab shows these rungs.
- **Root changes the story, both ways.** Three controls need the sandbox to run each execution as its own UID,
  and that needs root:
  - `RLIMIT_NPROC` limits the number of processes. It has no effect for uid 0. When the sandbox is not root, your own
    processes share the limit.
  - The sandbox keeps your files unreadable. `HOME` redirection is not a boundary.
  - The sandbox finds a process that left the group.

  Colab and many CI containers run as root and get all three. A laptop user gets none. The core reports which
  controls hold, and it does not pretend. See §2 and the Verify list of the primer.
- **kind has no gVisor**, and Kata/Firecracker need `/dev/kvm`. Colab and most laptops do not have `/dev/kvm`.
  The lab says this clearly and uses GKE Sandbox for the gVisor path. The dated product facts are in the Verify
  list of the primer (2026-09-26).
