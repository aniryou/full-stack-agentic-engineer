# sandbox-lab — run model-generated code without ambient authority, and prove it

After this lab you can run an agent's `run_code` and `fetch_url` tools so that a fully hijacked model
gets nothing outside the model: no credentials, no network by default, no durable filesystem — first
with a process sandbox and a hardened container on a laptop, then as pod-per-execution on Kubernetes,
then on a GKE Sandbox (gVisor) node pool. Every claim is a measured verdict, not an assertion.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (30 min): the threat model and the isolation ladder.
2. `pip install -e ".[dev]" && python3 -m sandboxlab probes --level process+netns` — under a minute:
   the attack probes through a process sandbox, each verdict `CONTAINED` or `LEAKED`, measured here.
3. Open [`notebooks/01_hardened_containers.ipynb`](notebooks/01_hardened_containers.ipynb) (T0): run the
   same probes through every rung of the ladder and read what each control stops.

The fastest single command that says something true about your machine:

```bash
python3 -m sandboxlab env      # which isolation levels this machine can actually run
```

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T0 + Docker = the container rungs on your machine; T3 = the
Google Cloud deployment, optional. No GPU is needed anywhere.* Each notebook opens with *the one-minute
version*, works examples against the library, then 3–6 exercises (implement the key check, predict a
number, read a manifest) each followed by a check that prints ✅, and closes with *in a design review*.
Answers are in [`solutions/`](solutions/). About 8 hours in all.

| # | Notebook | Tier | You will be able to… | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`hardened_containers`](notebooks/01_hardened_containers.ipynb) | T0 (+Docker) | climb the isolation ladder — process sandbox, dedicated UID, network namespace, hardened container, gVisor — and read the probe verdict for each; say why `RLIMIT_NPROC` is a no-op for root and why the wall clock and CPU seconds are both needed | §1, §2 | ~2 h |
| 02 | [`pod_per_execution_on_kind`](notebooks/02_pod_per_execution_on_kind.ipynb) | T0 (+kind) | build a sandbox pod, predict the API server's admission chain offline (RuntimeClass → Pod Security → ValidatingAdmissionPolicy), validate it for K8s 1.34, run one execution on a simulated cluster, and see why kind cannot do gVisor | §5 | ~2 h |
| 03 | [`egress_proxy_and_secret_brokering`](notebooks/03_egress_proxy_and_secret_brokering.ipynb) | T0 | route a network-less sandbox through an allowlisting proxy that injects a credential the sandbox never sees, refuses `CONNECT` and SSRF, redacts reflected secrets, and audits every decision | §4 | ~1.5 h |
| 04 | [`an_agent_with_a_sandbox_tool`](notebooks/04_an_agent_with_a_sandbox_tool.ipynb) | T0 | run the 07.1 loop with `run_code`/`fetch_url` behind the sandbox and proxy, feed it a poisoned document, and watch tiers, turn budgets and the boundaries fail closed while the model obeys | §3, §8 | ~1.5 h |
| 05 | [`gke_sandbox_with_gvisor`](notebooks/05_gke_sandbox_with_gvisor.ipynb) | T3 (plannable at T0) | read the GKE Sandbox Terraform as a design-review checklist, see the one field that changes from kind, and size the warm pool and cost per execution | §5, §6, §9 | ~1 h |

Everything runs on a laptop first. The process sandbox, the egress proxy and the agent are standard
library only, so the same files run inside `python:3.12-slim` containers and from ConfigMaps on
Kubernetes. Rungs this machine cannot run — Docker, gVisor, kind, GKE — print the exact commands and
read bundled verdicts labelled *sample output in the documented format (illustrative)*; a simulated
timing is labelled *simulated*; a verdict you measured says *measured on this machine*.

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the process sandbox, the proxy, the agent, the admission predictor, the simulated cluster | every notebook, all tests |
| **T0 + Docker** | a laptop with Docker | the hardened container, gVisor if installed, kind | `deploy/docker/`, `deploy/kind/` |
| **T3** | GCP | a GKE Sandbox (gVisor) node pool via Terraform | `deploy/gcp/terraform/`, `deploy/gke/` |

The probe suite is a superset of the core's: it adds five the core lacks (`proc_environ` and
`write_outside`, which separate a container from a process; `metadata`, a stand-in cloud metadata endpoint;
`memory_hog`; `disk_fill_many`) and names four differently —
`infinite_loop` is the core's `cpu_spin`, `huge_output` its `output_flood`, `egress` its `egress_connect`,
`env_secret`/`ssh_key` its `read_env_secret`/`read_ssh_key` (`sandboxlab.probes.CORE_PROBE_NAMES`). Exit
reasons are one vocabulary in both packages (PRIMER §3: `cpu_time`, `wall_timeout`, `memory`, `pids`,
`output_limit`, …), plus `harness_timeout` for the deliberately unbudgeted contrast executor.

## Run it

```bash
cd sandbox-lab
python3 -m pip install -e ".[dev]"                 # PyYAML + kubernetes-validate; the sandbox itself is stdlib
python3 -m pytest -q                               # 120 tests, ~60 s, offline, no GPU
python3 -m sandboxlab env                          # which isolation levels are measurable here
python3 -m sandboxlab probes --level process+netns # the attack probes through a process sandbox
python3 -m sandboxlab probes --level docker:runc    # measured with Docker; else the command + sample verdicts
python3 -m sandboxlab pool --rate 5.42 --exec 2 --cold 3   # size a warm pool (Little's law + Erlang C)
python3 -m sandboxlab gke-review                   # the GKE Sandbox Terraform checklist, read offline
python3 -m jupyterlab notebooks                    # the exercises; answers in solutions/
```

Optional: `pip install --no-deps cel-python lark jmespath pendulum google-re2` lets the tests evaluate
the ValidatingAdmissionPolicy's CEL and confirm it agrees with the offline predictor (skipped without it).

## How it fits

Read the [identity primer](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
§6.2 (code execution) and §5 (the gateway path for credentials) first — this topic builds out its
"Sandboxed execution" control. It reuses the [07.1 agent](../../agent-fundamentals/agent-core/) tool
contract (the `run_code` tool returns the same result shape), the
[scaling primer](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
§1/§5 (admission, cost per action, the idempotency-key recipe), and layer 03's
[gpu-scheduling](../../../03-kubernetes-gpu/gpu-scheduling/) manifest and kind conventions. The core next
door, [`../sandbox-core/`](../sandbox-core/), is the minimal from-scratch version; this lab never imports
it. Prices and where to get compute: [`COMPUTE.md`](../../../COMPUTE.md).

## The library (`sandboxlab/`, ~4,800 lines, standard library + PyYAML)

| Module | Lines | The idea |
|---|---:|---|
| `wrapper.py` | ~320 | the execution contract enforced *inside* any sandbox: CPU/wall/memory/pids/file/output budgets, a process-group kill, one exit reason, one result line — the same file as a process, a `docker run` entrypoint, or a pod command |
| `process.py` | ~280 | `Unsandboxed` (the contrast) and `ProcessSandbox` (clean env, throwaway workspace, rlimits, dedicated UID, network namespace); `ExecResult.to_tool_result()` is the 07.1 shape |
| `probes.py` | ~420 | the attacks a hijacked model would run, against stand-ins and canaries, each harmless by construction, with a verdict `CONTAINED`/`LEAKED`/`N/A`; the verdict table |
| `seccomp.py` | ~115 | Docker's default seccomp profile minus `memfd_create`/`execveat`/`socketcall` and the extra socket families — derived, with a diff |
| `docker.py` | ~235 | the hardened `docker run` builder (flag by flag, with what each stops) and the gVisor install steps |
| `proxy/` | ~530 | `server.py` the allowlisting egress proxy with credential injection, SSRF guards, reflection redaction and an audit line per decision (TCP or Unix socket); `stub.py` a stand-in upstream; `client.py` the sandbox-side snippet |
| `agent/` | ~350 | `loop.py` the 07.1 loop with tiers, turn budgets, idempotency keys and audit; `tools.py` `run_code`/`fetch_url` through the sandbox and proxy |
| `k8s/` | ~1,670 | `manifests.py` typed builders; `admission.py` the offline admission predictor (PSS + VAP + RuntimeClass) with CEL that matches; `policy.py` one policy rendered into every enforcement point; `runner.py` Job and warm-pool runners over a real or simulated cluster; `render.py` writes the deploy YAML |
| `audit.py`, `bench.py`, `gke.py`, `report.py` | ~480 | the identity-lab audit event and abuse detection; start-up latency and pool sizing (Little's law, Erlang C); the GKE Terraform review; JSON/Markdown reports |

## Deploy

[`deploy/`](deploy/) — [`docker/`](deploy/docker/) (the hardened run script, the proxy-over-socket
variant, the gVisor install, the generated seccomp profile), [`kind/`](deploy/kind/) (a laptop cluster
with restricted Pod Security, default-deny NetworkPolicy, the proxy, quota and the admission policy —
no gVisor), [`gcp/terraform/`](deploy/gcp/terraform/) + [`gke/`](deploy/gke/) (a GKE Sandbox gVisor node
pool, private nodes with no NAT, Artifact Registry, managed Prometheus). Each has a README with cost and
cleanup. The YAML and the seccomp profile are generated from `sandboxlab/k8s/policy.py` and
`sandboxlab/seccomp.py`: `python3 tools/render_manifests.py` (a test fails if a file drifts).

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
make check                                              # all of the above + tests + render check + bash -n
```

## Caveats

- **Simulated vs measured vs illustrative.** A probe verdict says *measured on this machine*; the
  Kubernetes lifecycle timing and the Docker/gVisor verdicts are *simulated* or *sample output in the
  documented format (illustrative)* and say so. Only a real Docker daemon or cluster gives measurements.
- **A sandbox relocates risk, it does not remove it.** The credential moves from the sandbox into the
  proxy; the blast radius moves from the host into one pod. gVisor does not stop code from using what the
  sandbox holds, and does not enforce cgroup limits inside the sandbox (the host sets them).
- **kind is not a boundary.** It teaches the objects and the admission chain; its NetworkPolicy dataplane
  fails open and it has no gVisor. Production wants a real runtime class on dedicated, tainted nodes.
- **Checked by construction.** The deploy paths are checked with `bash -n`, `DRY_RUN=1`, Terraform
  `validate` and `kubernetes-validate --strict -k 1.34.0`, not run on real infrastructure here.

## Verify list (facts dated 2026-09-26 that move)

- gVisor `runsc`: apt "release" channel; Docker runtime name `runsc`; K8s RuntimeClass handler `runsc`
  self-managed, `gvisor` on GKE. Install now ships a `gvisor-bin/` directory beside `runsc` (verify).
- GKE Sandbox: Terraform `node_config.sandbox_config.type = "GVISOR"` (case-sensitive); GKE creates the
  `gvisor` RuntimeClass and taints sandbox nodes `sandbox.gke.io/runtime=gvisor` (verify the handler and
  scheduling with `kubectl get runtimeclass gvisor -o yaml`); no E2 shared-core machine types.
- `python:3.12-slim` — pin a digest before production. kind v0.33.0, node image K8s 1.34.11.
  kubernetes-sigs/agent-sandbox `v1.0.2` for the optional warm-pool CRDs.
- Prices in `bench` and `gke` cost helpers are inputs you supply; every $ figure is `(verify)` against
  [`COMPUTE.md`](../../../COMPUTE.md), which has no CPU-only prices today.

MIT licensed.
