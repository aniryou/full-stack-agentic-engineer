# sandbox-lab — run model-generated code without ambient authority, and prove it

After this lab, you can run the `run_code` and `fetch_url` tools of an agent so that a fully hijacked
model gets nothing outside the model. It gets no credentials, no network by default and no durable
filesystem. First, you do this with a process sandbox and a hardened container on a laptop. Then you do
it as pod-per-execution on Kubernetes. Then you do it on a GKE Sandbox (gVisor) node pool. Each claim is
a measured verdict, not an assertion.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (30 min): the threat model and the isolation ladder.
2. Run `pip install -e ".[dev]" && python3 -m sandboxlab probes --level process+netns`. It takes less
   than a minute. It sends the attack probes through a process sandbox. This machine measures each
   verdict, `CONTAINED` or `LEAKED`.
3. Open [`notebooks/01_hardened_containers.ipynb`](notebooks/01_hardened_containers.ipynb) (T0). Run the
   same probes through each rung of the ladder. Then read what each control stops.

This is the fastest single command that tells you something true about your machine:

```bash
python3 -m sandboxlab env      # which isolation levels this machine can actually run
```

## What you get

*Tiers: T0 = laptop or Colab CPU, free. T0 + Docker = the container rungs on your machine. T3 = the
Google Cloud deployment, optional. No part of the lab needs a GPU.*

Each notebook starts with *the one-minute version*. Then it shows worked examples that use the library. Then it
gives 3–6 exercises (implement the key check, predict a number, read a manifest). After each exercise,
a check prints ✅. The notebook ends with *in a design review*.

The answers are in [`solutions/`](solutions/). The lab takes about 8 hours in all.

| # | Notebook | Tier | You will be able to… | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`hardened_containers`](notebooks/01_hardened_containers.ipynb) | T0 (+Docker) | climb the isolation ladder (process sandbox, dedicated UID, network namespace, hardened container, gVisor) and read the probe verdict for each rung. Say why `RLIMIT_NPROC` has no effect for root, and why you need both the wall clock and CPU seconds | §1, §2 | ~2 h |
| 02 | [`pod_per_execution_on_kind`](notebooks/02_pod_per_execution_on_kind.ipynb) | T0 (+kind) | build a sandbox pod and predict the admission chain of the API server offline (RuntimeClass, then Pod Security, then ValidatingAdmissionPolicy). Validate the pod for K8s 1.34, run one execution on a simulated cluster, and see why kind cannot run gVisor | §5 | ~2 h |
| 03 | [`egress_proxy_and_secret_brokering`](notebooks/03_egress_proxy_and_secret_brokering.ipynb) | T0 | route a sandbox with no network through a proxy with an allowlist. The proxy injects a credential that the sandbox never sees. The proxy also refuses `CONNECT` and SSRF, redacts reflected secrets, and writes an audit line for each decision | §4 | ~1.5 h |
| 04 | [`an_agent_with_a_sandbox_tool`](notebooks/04_an_agent_with_a_sandbox_tool.ipynb) | T0 | run the 07.1 loop with `run_code`/`fetch_url` behind the sandbox and the proxy. Give it a poisoned document. Then see the tiers, the turn budgets and the boundaries fail closed while the model obeys the instructions in the document | §3, §8 | ~1.5 h |
| 05 | [`gke_sandbox_with_gvisor`](notebooks/05_gke_sandbox_with_gvisor.ipynb) | T3 (plannable at T0) | read the GKE Sandbox Terraform as a design-review checklist. Find the one field that changes from kind. Calculate the size of the warm pool and the cost per execution | §5, §6, §9 | ~1 h |

Everything runs on a laptop first. The process sandbox, the egress proxy and the agent use only the
standard library. Thus the same files run inside `python:3.12-slim` containers and from ConfigMaps on
Kubernetes.

If this machine cannot run a rung (Docker, gVisor, kind or GKE), the lab prints the exact commands
for that rung. It also reads bundled verdicts with the label *sample output in the documented format
(illustrative)*. A simulated duration has the label *simulated*. A verdict that you measured says
*measured on this machine*.

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the process sandbox, the proxy, the agent, the admission predictor, the simulated cluster | every notebook, all tests |
| **T0 + Docker** | a laptop with Docker | the hardened container, gVisor if installed, kind | `deploy/docker/`, `deploy/kind/` |
| **T3** | GCP | a GKE Sandbox (gVisor) node pool through Terraform | `deploy/gcp/terraform/`, `deploy/gke/` |

The probe suite contains all the probes of the core, and more. It adds five probes that the core does
not have:

- `proc_environ` and `write_outside`, which separate a container from a process.
- `metadata`, a stand-in cloud metadata endpoint.
- `memory_hog`.
- `disk_fill_many`.

It also gives four probes different names (`sandboxlab.probes.CORE_PROBE_NAMES`):

- `infinite_loop` is the `cpu_spin` of the core.
- `huge_output` is its `output_flood`.
- `egress` is its `egress_connect`.
- `env_secret`/`ssh_key` are its `read_env_secret`/`read_ssh_key`.

Both packages use the same names for exit reasons (PRIMER §3: `cpu_time`, `wall_timeout`, `memory`,
`pids`, `output_limit`, …). The lab adds `harness_timeout` for the contrast executor, which has no
budget by design.

## Run it

```bash
cd sandbox-lab
python3 -m pip install -e ".[dev]"                 # PyYAML + kubernetes-validate; the sandbox itself is stdlib
python3 -m pytest -q                               # 122 tests, ~60 s, offline, no GPU
python3 -m sandboxlab env                          # which isolation levels are measurable here
python3 -m sandboxlab probes --level process+netns # the attack probes through a process sandbox
python3 -m sandboxlab probes --level docker:runc    # measured with Docker; else the command + sample verdicts
python3 -m sandboxlab pool --rate 5.42 --exec 2 --cold 3   # size a warm pool (Little's law + Erlang C)
python3 -m sandboxlab gke-review                   # the GKE Sandbox Terraform checklist, read offline
python3 -m jupyterlab notebooks                    # the exercises; answers in solutions/
```

Optional: `pip install --no-deps cel-python lark jmespath pendulum google-re2` lets the tests evaluate
the CEL of the ValidatingAdmissionPolicy. The tests then make sure that the CEL agrees with the offline
predictor. Without these packages, the tests skip this check.

## How it fits

Read the [identity primer](../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
§6.2 (code execution) and §5 (the gateway path for credentials) first. This topic expands the
"Sandboxed execution" control of that primer. It uses these parts of other topics again:

- the tool contract of the [07.1 agent](../../agent-fundamentals/agent-core/) (the `run_code` tool
  returns the same result shape),
- the [scaling primer](../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
  §1/§5 (admission, cost per action, the idempotency-key recipe),
- the manifest and kind conventions of layer 03's [gpu-scheduling](../../../03-kubernetes-gpu/gpu-scheduling/).

The core beside this lab, [`../sandbox-core/`](../sandbox-core/), is the minimal version, written from
nothing. This lab never imports it. For prices and for where to get compute, see
[`COMPUTE.md`](../../../COMPUTE.md).

## The library (`sandboxlab/`, ~4,800 lines, standard library + PyYAML)

| Module | Lines | The idea |
|---|---:|---|
| `wrapper.py` | ~320 | the execution contract, which the wrapper enforces *inside* any sandbox: CPU/wall/memory/pids/file/output budgets, a process-group kill, one exit reason, one result line. The same file runs as a process, a `docker run` entrypoint or a pod command |
| `process.py` | ~280 | `Unsandboxed` (the contrast) and `ProcessSandbox` (clean env, temporary workspace, rlimits, dedicated UID, network namespace). `ExecResult.to_tool_result()` returns the 07.1 shape |
| `probes.py` | ~420 | the attacks that a hijacked model can run, against stand-ins and canaries. Each attack is harmless by construction and has a verdict `CONTAINED`/`LEAKED`/`N/A`. The module also has the verdict table |
| `seccomp.py` | ~115 | the default seccomp profile of Docker minus `memfd_create`/`execveat`/`socketcall` and the extra socket families. The module derives the profile and shows a diff |
| `docker.py` | ~235 | the hardened `docker run` builder (flag by flag, with what each flag stops) and the gVisor install steps |
| `proxy/` | ~530 | `server.py` is the egress proxy with an allowlist, credential injection, SSRF guards, reflection redaction and an audit line for each decision (TCP or Unix socket). `stub.py` is a stand-in upstream. `client.py` is the snippet for the sandbox side |
| `agent/` | ~350 | `loop.py` is the 07.1 loop with tiers, turn budgets, idempotency keys and audit. `tools.py` has `run_code`/`fetch_url`, which go through the sandbox and the proxy |
| `k8s/` | ~1,670 | `manifests.py` has typed builders. `admission.py` is the offline admission predictor (PSS + VAP + RuntimeClass), with CEL that agrees with the predictor. `policy.py` renders one policy into each enforcement point. `runner.py` has Job and warm-pool runners over a real or simulated cluster. `render.py` writes the deploy YAML |
| `audit.py`, `bench.py`, `gke.py`, `report.py` | ~480 | The identity-lab audit event and abuse detection. Start-up latency and the calculation of pool size (Little's law, Erlang C). The GKE Terraform review. JSON/Markdown reports. |

## Deploy

[`deploy/`](deploy/) contains:

- [`docker/`](deploy/docker/): the hardened run script, the proxy-over-socket variant, the gVisor
  install and the generated seccomp profile.
- [`kind/`](deploy/kind/): a laptop cluster with restricted Pod Security, default-deny NetworkPolicy,
  the proxy, quota and the admission policy. It has no gVisor.
- [`gcp/terraform/`](deploy/gcp/terraform/) + [`gke/`](deploy/gke/): a GKE Sandbox (gVisor) node pool,
  private nodes with no NAT, Artifact Registry and managed Prometheus.

Each target has a README with its cost and its clean-up steps. `python3 tools/render_manifests.py`
generates the YAML and the seccomp profile from `sandboxlab/k8s/policy.py` and `sandboxlab/seccomp.py`.
A test fails if a file no longer agrees with its source.

## Regenerating notebooks

`tools/build_notebooks.py` generates `notebooks/` (exercises) and `solutions/` from `notebooks_src/*.py`
(percent format with `### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
make check                                              # all of the above + tests + render check + bash -n
```

## Caveats

- **Simulated, measured and illustrative.** A probe verdict says *measured on this machine*. The
  duration of the Kubernetes lifecycle and the Docker/gVisor verdicts are *simulated* or *sample output in the
  documented format (illustrative)*, and they say so. Only a real Docker daemon or cluster gives
  measurements.
- **A sandbox moves risk. It does not remove it.** The credential moves from the sandbox into the
  proxy. The blast radius moves from the host into one pod. With gVisor, code can still use what the
  sandbox holds. gVisor also does not enforce cgroup limits inside the sandbox (the host sets them).
- **kind is not a boundary.** It teaches the objects and the admission chain. Its NetworkPolicy
  dataplane fails open, and it has no gVisor. A production system needs a real runtime class on
  dedicated, tainted nodes.
- **Checked by construction.** The lab does a check of the deploy paths with `bash -n`, `DRY_RUN=1`, Terraform
  `validate` and `kubernetes-validate --strict -k 1.34.0`. It does not run them on real infrastructure
  here.

## Verify list (facts dated 2026-09-26 that move)

- gVisor `runsc`: the apt "release" channel and the Docker runtime name `runsc`. The K8s RuntimeClass
  handler is `runsc` when you manage gVisor yourself, and `gvisor` on GKE. The install now supplies a
  `gvisor-bin/` directory beside `runsc` (verify).
- GKE Sandbox: the Terraform field is `node_config.sandbox_config.type = "GVISOR"` (case-sensitive).
  GKE creates the `gvisor` RuntimeClass and puts the taint `sandbox.gke.io/runtime=gvisor` on sandbox
  nodes (verify the handler and scheduling with `kubectl get runtimeclass gvisor -o yaml`). GKE Sandbox does
  not support E2 shared-core machine types.
- `python:3.12-slim`: pin a digest before production. The lab uses kind v0.33.0 with the node image
  K8s 1.34.11. The optional warm-pool CRDs come from kubernetes-sigs/agent-sandbox `v1.0.2`.
- The prices in the `bench` and `gke` cost helpers are inputs that you supply. Every $ figure is
  `(verify)` against [`COMPUTE.md`](../../../COMPUTE.md), which has no CPU-only prices today.

MIT licensed.
