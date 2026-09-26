# sandboxed-execution — run the model's code without handing it your keys

After this topic you can take an agent's `run_code` (or any tool that runs untrusted, model-written
programs) and say exactly what it can reach when the model is hijacked, which control bounds each risk,
how much isolation you need, and what a pool of sandboxes costs — and you can render and check the
Kubernetes that enforces it.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §1–§2 (30 min): why a code tool is the most dangerous tool, and the
   isolation ladder from process to microVM.
2. `cd sandbox-core && python3 -m pip install -e ".[dev]" && python3 -m pytest -q` — 75 tests, ~30 s; then open
   [`sandbox-core/notebooks/01_the_threat_model.ipynb`](sandbox-core/notebooks/01_the_threat_model.ipynb)
   and watch a secret leak from unsandboxed code, then get contained.
3. When you have Docker or a cluster: [`sandbox-lab/`](sandbox-lab/) hardens a real container, runs a
   pod-per-execution on kind, and deploys a GKE Sandbox (gVisor) node pool.

## What you get

*Tiers: **T0** = laptop / Colab CPU / CI, free — everything conceptual is here; **T0 + Docker** = the
container rungs and kind on your own machine, still free; **T3** = the Google Cloud deployment, optional. No
GPU anywhere, so the repo's T1/T2 (one or more GPUs) do not apply.* Each notebook opens with "The one-minute version",
works examples against the code, sets exercises with a check that prints ✅, and closes with "In a design
review". Finished versions are in [`sandbox-core/solutions/`](sandbox-core/solutions/).

| Path | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`sandbox-core`](sandbox-core/) `01_the_threat_model` | name a code tool's blast radius; map each risk (secret, egress, fork bomb, disk, CPU, output) to the control that bounds it; see what leaks unsandboxed | §1 | 45 min | T0 |
| `02_a_process_sandbox` | build the T0 boundary: clean env, a 0700 workspace, a UID per execution, rlimits, a wall-clock kill, streamed and capped output; know what it cannot stop (the network; without the UID, your files and `setsid()` escapes) | §2, §3 | 1 h | T0 |
| `03_the_execution_contract_and_policies` | design the request/result contract with budgets and exit reasons; write policy as data; render Pod Security / NetworkPolicy / Job / admission YAML; key executions for safe replay | §3, §5 | 1 h | T0 |
| `04_egress_and_secrets` | put an allowlisting egress proxy in front of the sandbox that injects a credential the code never holds and never follows a redirect; see why the network — not `HTTP_PROXY`, not a declared host — enforces egress | §4 | 45 min | T0 |
| `05_pools_latency_and_cost` | see why Little's law is the floor and size a replace-after-use pool with Erlang C; pick an isolation level for a latency budget; put a number on cost per action | §6 | 45 min | T0 |
| [`sandbox-lab`](sandbox-lab/) | the same controls on a process with a network namespace, real Docker, kind and GKE Sandbox (gVisor), with an agent whose `run_code`/`fetch_url` tools fail closed under injection where the host provides a network namespace | §2–§9 | ~8 h | T0, T0 + Docker, T3 |

## Run it

```bash
cd sandbox-core
python3 -m pip install -e ".[dev]"     # the library is stdlib only; dev adds pytest, jupyter, pyyaml, kubernetes-validate
python3 -m pytest -q                    # 75 tests, ~30 s
python3 tools/build_notebooks.py        # (re)build notebooks/ and solutions/
python3 -m jupyterlab notebooks         # do the exercises
```

The library needs nothing installed to import:

```python
from sandboxcore import ProcessSandbox, ExecutionRequest, Budgets
sb = ProcessSandbox()
r = sb.run(ExecutionRequest(code="print(6*7)", budgets=Budgets(cpu_s=1, wall_s=3)))
print(r.exit_reason, r.stdout)          # ok 42
print(sb.isolation_report()["network_blocked"])   # False — a process sandbox does not block egress
```

## How it fits

Read the [agent-core loop and tool contract](../agent-fundamentals/agent-core/) (07.1) first — this topic is
what makes its `run_code` tool safe. It builds directly on two neighbours: the
[identity & security primer](../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
(§2 threat model, §4.2 tool tiers, §5 secrets, §6.2 code execution, §9 audit — cited, not restated) and the
[scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
(§1 cost per action, §5.4 idempotency). The Kubernetes controls reuse the conventions of the
[GPU scheduling lab](../../03-kubernetes-gpu/gpu-scheduling/) (typed manifest builders, a kind deploy script,
offline manifest validation). Where each tier runs and what it costs: [`COMPUTE.md`](../../COMPUTE.md); the
learning path: [`CURRICULUM.md`](../../CURRICULUM.md).

## Going further / caveats

- **What is real at T0, and what is not.** The process sandbox, the policy engine, the egress proxy, the
  pool model and the rendered manifests all run and are tested on a laptop. Latencies for the process
  sandbox are **measured on the build host**; container/gVisor/VM/GKE numbers are `(verify)` inputs or come
  from upstream specs, and are labelled so. Simulator output is labelled SIMULATED.
- **A process sandbox is not a security boundary against a determined attacker.** It contains resource abuse
  and removes ambient authority, but does not stop kernel exploits or the network. Real isolation is a
  container, then gVisor (`runsc`), then a microVM (Firecracker/Kata) — the lab.
- **Root changes the story, both ways.** Three controls need the sandbox to run each execution as its own
  UID, which needs root: `RLIMIT_NPROC` (ignored for uid 0, shared with your processes otherwise), keeping
  your files unreadable (`HOME` redirection is not a boundary), and finding a process that left the group.
  Colab and many CI containers run as root and get all three; a laptop user gets none, and the core reports
  which rather than pretending. See the primer's §2 and Verify list.
- **kind has no gVisor**, and Kata/Firecracker need `/dev/kvm` (not on Colab or most laptops). The lab says
  so plainly and uses GKE Sandbox for the gVisor path. Dated product facts are in the primer's Verify list
  (2026-09-26).
