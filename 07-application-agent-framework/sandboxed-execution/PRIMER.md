# Sandboxed execution: running model-generated code and tool calls without ambient authority

*Layer 07 of the stack (the agent itself). Facts checked 26 September 2026 against upstream sources —
gVisor, Firecracker, Kata, Kubernetes 1.34 (Pod Security, NetworkPolicy, RuntimeClass,
ValidatingAdmissionPolicy, Job), kind, the CPython 3.11 `resource`/`subprocess` docs, E2B and
kubernetes-sigs/agent-sandbox. Product facts that could not be checked against a vendor's live docs are
marked `(verify)`; the dated Verify list is at the end. Every formula names the function in
[`sandbox-core`](sandbox-core) (package `sandboxcore`) that computes it; latencies are inputs, labelled
"measured on the build host" or `(verify)`, never invented. The detailed lab is
[`sandbox-lab`](sandbox-lab).*

This primer is about one tool: the one that runs code the model wrote. Everything an agent does that turns
untrusted text into a privileged action (the identity primer's one-line definition of an agent) is sharpest
here — a `run_code` tool, a code interpreter, an "execute this shell command" action, a tool that evaluates
a model-authored query. It builds on material already in the repo and cites rather than repeats it: the
[identity & security primer](../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md)
(the threat model in its §2, tool tiers in §4.2, secrets in §5, code execution in §6.2, audit in §9), the
[scaling primer](../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md)
(cost per action in §1, idempotency in §5.4), the [GPU scheduling primer](../../03-kubernetes-gpu/gpu-scheduling/PRIMER.md)
(what Kubernetes sees, §1; the scheduling cycle, §3; sharing and learning locally, §9–§10), and the
[agent-core loop and tool contract](../agent-fundamentals/agent-core/) (07.1). Read those first; this
primer assumes them.

---

## The one-minute version

A code tool takes an untrusted, model-written program and runs it. If it runs with the agent's environment,
home directory, credentials and network, then a single prompt injection is a shell on your infrastructure —
the OWASP-for-agents risk **ASI05 Unexpected Code Execution**, and the identity primer's rule that "an
'execute code' tool is DESTRUCTIVE-tier by definition". The invariant a sandbox must restore is **no ambient
authority**: no credentials, no network by default, no persistent filesystem, only what the caller
deliberately hands it for one execution.

You get there by climbing an **isolation ladder** and by writing the decision as **policy**, enforced twice.
In-process restrictions are not a boundary; a separate **process** with a clean environment, OS resource
limits and — where it can — its own UID is the first real one; a **container** adds namespaces, cgroups, dropped capabilities and a
read-only, non-root, no-new-privileges root filesystem; a **user-space kernel** (gVisor) puts a second
kernel between the code and the host; a **microVM** (Firecracker, Kata) puts a hardware boundary there. The
execution is an **API with a contract** — code plus inputs plus a budget for every exhaustible resource, in;
truncated output plus an exit reason plus usage, out — and every call carries an idempotency key so an
at-least-once redelivery does not run a side effect twice. The network is a **separate control**: the
sandbox reaches only an allowlisting egress proxy that injects credentials it never holds. On Kubernetes the
same policy renders to Pod Security *restricted*, a default-deny NetworkPolicy, a non-retrying Job, and an
admission policy that rejects any pod missing the controls. And because a sandbox has a cold start, you size
a **warm pool** with the same queueing arithmetic as any service. After this primer you can walk that whole
design in a review and put numbers on it.

---

## 1. Why a sandbox, and the threat model

An agent reads text it did not write — user messages, tool results, retrieved documents, web pages, other
agents' output — and it cannot cryptographically tell instructions from data. The identity primer makes this
its first principle; here it has a specific consequence. The moment the agent can **run code**, the attacker's
target is no longer "make the model say something wrong"; it is "make the model emit a program, and run it
with whatever the runner holds." That is a qualitative jump in blast radius.

**The blast radius of a naive code tool.** Suppose `run_code` is implemented as `subprocess.run([python, "-c",
code])` in the agent's own process. Then model-written code inherits, by default:

- the process **environment** — `AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, API keys, and on a cloud VM the
  reach of the **metadata server** (`169.254.169.254`), which hands out tokens;
- the **home directory** — `~/.ssh`, `~/.config/gcloud`, `~/.kube/config`, `~/.aws/credentials`;
- the **filesystem** the agent can write, and any mounted data;
- the **network**, in and out, with the agent's own identity;
- **unbounded** CPU, memory, processes, disk and output.

`sandboxcore.threats` makes each of these concrete as a **probe**: a short, harmless program that stands in
for what a hijacked model might emit, paired with what it achieves unsandboxed and the exit reason a correct
sandbox produces. `read_env_secret` reads a planted `CLOUD_API_TOKEN`; `read_ssh_key` reads a stand-in key
from a 0700 victim home, by `~` and by its absolute path; `egress_connect` opens a socket to a loopback
listener the harness owns and sends bytes (the exfiltration leg); `fork_bomb`, `disk_fill`, `cpu_spin`,
`sleep_forever` and `output_flood` are the resource abuses; `escape_session` leaves its process group with
`setsid()` and leaves a file behind, so something outlives the call. Run through
`sandboxcore.UnsafeExecutor`, the secret, key, egress and escape probes **leak** — you watch the token come
back, the key print, the connection succeed, the escapee write after the call returned. That demonstration
(notebook 01) is the whole motivation. The lab's suite (`sandboxlab.probes`) adds the probes that separate a
container from a process — reading other processes' `/proc/<pid>/environ`, writing outside the workspace,
reaching a (stand-in) metadata endpoint, a memory hog — and names a few differently;
`sandboxlab.probes.CORE_PROBE_NAMES` maps one set to the other.

**Map each risk to the control that bounds it.** The design question, for every tool, is the identity
primer's: *assume the model is fully hijacked — what is the worst this code can do, and which control outside
the model stops it?* For a code tool the answers form a table you carry into a review:

| Risk (probe) | OWASP-for-agents | Bounded by |
|---|---|---|
| read a credential from the environment | ASI03 Identity & Privilege Abuse | a **clean environment** (nothing inherited) |
| read a private key from home | ASI03 | a **different UID** (or a filesystem that does not contain it: a mount namespace, a container). Pointing `HOME` elsewhere is not a boundary — the absolute path still opens |
| exfiltrate over the network | ASI01 Goal Hijack (exfiltration leg) | the **network layer**: deny-by-default egress + an allowlisting proxy |
| fork bomb | ASI05 Unexpected Code Execution | a **process limit** (`RLIMIT_NPROC` under a dedicated UID; a pids cgroup) |
| fill the disk | ASI05 | a **file-size / disk limit** |
| spin the CPU | ASI05 | a **CPU-time limit** |
| hang forever (no CPU) | ASI08 Cascading Failures | a **wall-clock timeout** (separate from CPU time) |
| flood the output | ASI08 | output **read as it streams**: truncated, and the run killed past a cap |
| outlive the call, leave files behind | ASI05 | a **per-execution UID** swept after the run (or a cgroup / PID namespace that dies with the sandbox) and an ephemeral workspace |

The identity primer's §2 has the full OWASP Top 10 for Agentic Applications (ASI01–ASI10) and the controls
for the rest; this topic builds out the ASI05 row it names ("Sandboxed execution … with no ambient
credentials"). The single most important line of the table is the third: **the network is not bounded by the
process sandbox.** `sandboxcore.ProcessSandbox.isolation_report()` reports `network_blocked: False` for
exactly this reason, and the egress probe is the one probe a process sandbox never contains — §4 is where
egress is actually controlled. The second most important is the key row: three of the controls (the key, the
fork bomb, the escape) exist only when the sandbox can run the code as **its own UID**, which needs root; the
report says whether it could (`uid_dropped`, `filesystem_isolated`, `escapes_swept`).

**The invariant: no ambient authority.** State the target as three "no"s and a "only":

1. **No credentials.** The process starts from a clean environment; secrets are brokered per call (§4), not
   inherited.
2. **No network by default.** Egress is denied unless an allowlist and a proxy permit a specific host.
3. **No persistent filesystem.** The workspace is ephemeral and thrown away; the code runs as a UID that
   cannot open your files, and whatever that UID leaves elsewhere (in `/tmp`) is swept after the run — with
   a per-execution UID; without one, a file written to `/tmp` stays (a container's filesystem avoids this).
4. **Only** what the caller deliberately passes for this one execution (the inputs and budget of §3).

A code tool that meets all four still lets a hijacked model *compute* — but computation with no authority is
a contained blast radius. That is the goal; the rest of this primer is how far up the isolation ladder you
go to trust it, and how you enforce, broker, size and observe it.

---

## 2. The isolation ladder

There is no single "sandbox." There is a ladder of boundaries, each defending against more, each costing
more. Choose the rung by what you are defending against and what you can afford per execution.

**Rung 0 — in-process restrictions are not a boundary.** Removing `__builtins__`, patching `import`, or an
AST allowlist all share the interpreter with the untrusted code, and Python is far too dynamic to fence
in-process: attribute chains, `gc`, C-level tricks and countless CPython escapes have broken every such
attempt. Treat in-process "sandboxes" as ergonomics, never security. The first real boundary is a separate
process.

**Rung 1 — a subprocess with resource limits (`sandboxcore.ProcessSandbox`).** A child process started with
a **clean environment**, an **ephemeral 0700 workspace**, POSIX **resource limits** the child applies before
it runs the code, a **wall-clock deadline**, output **read as it streams** (kept up to a budget, the run killed
past a cap), and a kill of the whole **process group** at the end. When it runs as root it also gives each
execution **its own unprivileged UID** — and that one control carries three more: `RLIMIT_NPROC` becomes real
(it counts per real UID and is ignored for root), your files become unreadable (a different UID cannot open a
0700 home; setting `HOME` elsewhere hides nothing), and after the run the sandbox can kill every process of
that UID — including one that called `setsid()` to leave the group — and remove what it left in `/tmp`.
With a per-execution UID it contains every probe except egress; as a non-root user it cannot switch UID, and
the key read and the escape get through (and the process limit is shared with your own processes). It runs
anywhere, including Colab and CI. What it never does: it shares the host kernel (a kernel exploit escapes it)
and it does not touch the network. The sharp edges are in §2's box below and §3.

**Rung 1b — the same process, plus namespaces and a syscall filter (no container).** Linux can give a plain
process most of what a container has, one piece at a time: an **empty network namespace** (`unshare -n`, or
`unshare -Urn` as an unprivileged user) leaves it only a loopback of its own, so the egress probe fails
closed; a **user namespace** lets an unprivileged user do that and map itself to root inside; a **PID
namespace** hides other processes (and their `/proc/<pid>/environ`); a **mount namespace** can hide the home
directory entirely; **`no_new_privs`** stops setuid binaries escalating; and a **seccomp** filter trims the
syscall surface. Tools such as bubblewrap (`bwrap`) and nsjail package these. The lab's `ProcessSandbox`
climbs this rung (`--level process+netns`: a dedicated UID and an empty network namespace), and its
`seccomp.py` derives a profile. Where it is unavailable: macOS (no namespaces), inside Docker's default
seccomp profile (it blocks `unshare`/user namespaces), and on Ubuntu hosts whose AppArmor policy restricts
unprivileged user namespaces — the lab's `env.netns_mode()` detects each and says so.

**Rung 2 — a hardened container.** Linux namespaces (PID, mount, network, user, IPC, UTS) give the process
its own view of the world; cgroups enforce CPU and memory from outside; and you add the deterministic
hardening flags: `--network none` (no network at all), `--read-only` root filesystem with a small
`--tmpfs` workspace, `--cap-drop ALL`, `--security-opt no-new-privileges`, the default seccomp profile
(which already blocks ~44 dangerous syscalls), a non-root `--user`, `--pids-limit`, `--memory` and `--cpus`.
This is the lab's `docker.py`. It still shares the host kernel — a kernel bug is still an escape — and Docker
itself dominates the start time.

**Rung 3 — a user-space kernel (gVisor).** gVisor's **Sentry** is "an application kernel … written in a
memory-safe language (Go)" that intercepts the workload's syscalls and re-implements Linux (syscalls, memory
management, filesystems, a userspace network stack) in userspace; "**gVisor never passes through any system
call to the host**", and its own seccomp filter blocks the Sentry from making dangerous host calls. A
companion **Gofer** process owns the container filesystem. Escaping requires exploiting *both* the Sentry and
the host kernel, "which do not share any code". The default platform is `systrap` (seccomp-based
interception; works inside VMs); `kvm` is best on bare metal. It runs as the OCI runtime `runsc`
(`docker run --runtime=runsc`, or a Kubernetes `RuntimeClass` whose handler is `runsc`). It does **not**
defend against Spectre-style side channels, attacks in higher layers, or exploits *within* what the sandbox
is configured to reach — so different tenants still go in different sandboxes. It costs on syscall-heavy
workloads (the Sentry is on the path); pure compute runs natively.

**Rung 4 — a microVM (Firecracker, Kata).** A hardware virtualization boundary: the code runs in a
lightweight VM with its **own guest kernel**, so a guest kernel exploit does not reach the host kernel.
**Firecracker** starts a VMM in ~**12 ms** typical (8 CPU ms; wall 6–60 ms) and reaches the guest
`/sbin/init` within **≤ 125 ms**, with a VMM memory overhead **≤ 5 MiB** (its `SPECIFICATION.md`); it needs
**KVM** (`/dev/kvm`), so it runs on bare metal or nested-virt hosts, not on Colab or a typical laptop's
Docker. **Kata Containers** run each pod as its own lightweight VM with a `kata-agent` inside; its 4.x
runtime is `runtime-rs` (Rust) and its Kubernetes RuntimeClasses carry a `-runtime-rs` suffix
(`kata-qemu-runtime-rs`); it also needs `/dev/kvm`. Kata's `RuntimeClass` sets an `overhead.podFixed`
(the QEMU shim reserves 320Mi/250m by default) so the scheduler and quota account for the VMM.

**Snapshot/restore — and its one rule.** Instead of booting, you can boot once, warm the interpreter and
libraries, **snapshot** the VM, and **restore** a copy per execution. Firecracker maps the snapshot's memory
file `MAP_PRIVATE`, so pages load on demand ("very fast snapshot loading"), the memory file must outlive
every VM restored from it, and network connections from the snapshot may drop (its
`docs/snapshotting/snapshot-support.md`). gVisor has the same idea at the sandbox level (`runsc checkpoint`
/ `runsc restore`). How much of a cold start a restore saves depends on the image and the warm-up —
measure it; this primer puts no number on it `(verify)`. The rule: **restoring the same snapshot more than
once is insecure** by Firecracker's own account — every clone starts with the same RNG state, identifiers
and any tokens the snapshot held — unless the guest re-seeds (VMGenID, Linux ≥ 5.18) and regenerates
secrets after restore. Never snapshot a sandbox after it held a tenant's data, and never restore one
snapshot into two tenants.

**Rung 5 — a full VM or a separate machine.** The strongest and slowest: seconds to boot, a whole OS to
manage. Used for the highest-risk or noisiest workloads, or where a compliance boundary demands it.

A compact way to hold the ladder:

| Boundary | Isolates by | Defends against | Does NOT stop | Start cost |
|---|---|---|---|---|
| in-process | nothing real | (nothing) | everything | ~0 |
| process + rlimits (+ its own UID) | a separate process, OS limits; a per-execution UID when root | ambient creds, resource abuse; with the UID: file reads, fork bombs, escapes | kernel exploits, the network; without the UID: your files, escapes | ~1.5 ms fork/exec; ~35 ms Python start, ~50 ms `ProcessSandbox` round trip (measured on the build host, under shared load) |
| process + namespaces, seccomp (`unshare`, `bwrap`, nsjail) | the above + an empty netns, PID/mount namespaces, a syscall filter | the above + egress, other processes' environ, the host filesystem view | kernel exploits (fewer syscalls reach it); unavailable on macOS / inside Docker's default seccomp | + a few ms `(verify)` |
| hardened container | namespaces, cgroups, seccomp, caps | the above, plus most syscalls & the host FS/net view | host-kernel exploits | ~100–500 ms `(verify)`, Docker-dominated |
| gVisor (`runsc`) | a userspace kernel (Sentry) | host-kernel syscalls (a second kernel to break) | side channels, in-sandbox reach | container + tens–hundreds ms `(verify)` |
| microVM (Firecracker/Kata) | hardware virtualization, own guest kernel | guest-kernel escape to host | side channels; needs `/dev/kvm`; a snapshot restored twice shares its RNG and secrets | VMM ~12 ms + ≤125 ms to init (Firecracker spec); Kata ~0.5–2 s `(verify)`; restore from snapshot faster `(verify)` |
| full VM | a whole machine | almost everything at this layer | cost, minutes | tens of seconds `(verify)` |

> **The process sandbox's sharp edges (measured, FACTS §10/§15).** `RLIMIT_CPU` counts **CPU seconds**, not
> wall time, so a sleeper or a blocked I/O call never trips it — always add a wall-clock kill. Soft `<` hard
> makes the soft limit raise a catchable `SIGXCPU` first, then the hard limit `SIGKILL`s; soft `==` hard is
> an immediate `SIGKILL`. `RLIMIT_NPROC` is **per real UID, system-wide, and ignored for uid 0** — so a fork
> bomb is only stopped after switching to an unprivileged UID (which itself needs privilege), and every
> execution sharing that UID shares one budget: with a single `nobody` UID, one run's fork bomb starves the
> others (and any daemon running as `nobody`). Hence a **UID per execution**. Colab and many CI containers
> run as root; a laptop user cannot switch UID at all, and `ProcessSandbox.isolation_report()` says which.
> `RLIMIT_AS` limits **virtual** address space; set it too low and `import numpy` fails (OpenBLAS arenas),
> so keep it ≥ 256 MiB and set `OPENBLAS_NUM_THREADS=1`. A naive `subprocess.run(timeout=)` kills only the
> direct child, leaving a backgrounded **grandchild** re-parented to init — so start a new session and kill
> the whole **process group** (`sandboxcore.executor._kill_group`). The group is advisory, though: code can
> call `setsid()` and leave it, so only the UID (or a cgroup / PID namespace) finds that process
> afterwards. `subprocess`'s `preexec_fn` can deadlock a child forked from a threaded parent, so the sandbox
> switches UID and session inside `Popen` (in C) and lets the child's own interpreter lower its limits
> (`executor._LAUNCHER`). Reading output with `communicate()` buffers all of it in the caller — read it as it
> streams. None of these touch the **network**.

---

## 3. The execution contract

Treat the sandbox as an **API with a contract**, not a place you run things. The contract is what makes the
sandbox composable (any backend — process, container, microVM — honours the same shape) and auditable.

**The request (`sandboxcore.ExecutionRequest`, `Budgets`).** Code, plus inputs, plus a **budget for every
resource the code could exhaust**, plus the policy that applies. The budget is the load-bearing part:

| Budget field | Enforced by | Defaults (`Budgets`) |
|---|---|---|
| `cpu_s` | `RLIMIT_CPU` | 2 |
| `wall_s` | an external wall-clock kill | 5 |
| `memory_mb` | `RLIMIT_AS` (virtual) + the container `--memory`/limit | 256 |
| `pids` | `RLIMIT_NPROC` (off root) / kubelet `podPidsLimit` / `--pids-limit` | 16 |
| `file_mb` | `RLIMIT_FSIZE` (largest single file, not a disk quota) | 8 |
| `disk_mb` | measured after the run; `emptyDir.sizeLimit` in K8s | 16 |
| `output_bytes` | the parent reads as it streams and keeps this much per stream | 16,384 |
| `output_kill_bytes` | the parent stops the run once both streams pass it (`output_limit`) | 1 MiB |
| `open_files` | `RLIMIT_NOFILE` | 64 |

Two things a budget is not: `file_mb` is a *single-file* cap, not a disk quota (that is `disk_mb`, checked
after the run and enforced by `emptyDir.sizeLimit` in the cluster); and `memory_mb` via `RLIMIT_AS` is
*virtual* memory, which is why it has a floor.

**The result (`ExecutionResult`).** Truncated stdout/stderr (with a `truncated` flag and the full byte
counts in `usage`), an **exit reason** from a fixed vocabulary (`ok`, `error`, `wall_timeout`, `cpu_time`,
`memory`, `file_too_large`, `pids`, `open_files`, `output_limit`, `disk_limit`, `killed`, `denied`,
`sandbox_error` — `sandboxcore.contract.EXIT_REASONS`; the lab's wrapper returns the same names, plus
`harness_timeout` for its deliberately unbudgeted contrast executor), **where that reason came from**
(`reason_source`: the parent enforced or measured it, the kernel delivered a signal, or the untrusted
program's own exit status and stderr said so), the `usage` (CPU and peak memory of the child via `wait4`,
wall, disk, output bytes), the `artifacts` (workspace files by digest), any `over_budget` fields, what
`isolation` actually ran it, and what was `swept` afterwards. `ExecutionResult.as_tool_result()` converts this to agent-core's tool-result shape —
`{"ok": True, "data": …}` or `{"ok": False, "error": <exit reason>, "message": …, "hint": …}` — so the model
gets something it can act on ("`cpu_time`: use a cheaper algorithm") rather than a stack trace, exactly as
the scaling primer's §5.6 wants tool failures returned as structured data.

**Exit reasons are for the model, not just the log.** The mapping from a signal or a stderr pattern to an
exit reason lives in `sandboxcore.executor._classify` (a signal: `SIGXCPU` → `cpu_time`, `SIGXFSZ` →
`file_too_large`; stderr: a Python child's `OSError: File too large` → `file_too_large`,
`MemoryError`/OpenBLAS → `memory`, `EAGAIN` → `pids`). A Python child that hits `RLIMIT_FSIZE` exits non-zero
with `EFBIG` rather than dying of `SIGXFSZ` (a shell child dies of the signal) — the executor maps both to
`file_too_large`. **Only some of these can be trusted.** A reason read from stderr is the untrusted program's
own say-so — code can print `MemoryError` and exit 1 — so it is labelled `reason_source: code`: fine as a
hint to the model, not as evidence. `wall_timeout` and `output_limit` (the parent enforced them) and a
`cpu_time` the parent's `wait4` measurement confirms are `parent`; a bare signal is `signal` (the code could
have sent it to itself).

**Idempotency and safe re-execution.** Side effects mean at-least-once delivery (the scaling primer's §1.5:
"at-least-once means idempotency"), so a code tool needs the same discipline as any write. `idempotency_key(
turn_id, step, call_index, args)` follows the scaling primer's §5.4 recipe exactly — turn, step, call index,
and a hash of the arguments — and `ResultStore.run_once(key, run)` **claims the key before running**, returns
the stored result on a later delivery, and refuses (`InFlight`) one that arrives while the first is still
running. Be precise about what that guarantees: within one process, a redelivery does not run the code
again. It is not durable (a crash between the run and the store forgets it — a real store writes the claim
and the result with a conditional put), it does not dedupe across replicas, and it does not make the code's
*own* side effects idempotent: end to end that needs the key forwarded downstream (the egress proxy adding an
`Idempotency-Key` header to the upstream call), as the scaling primer's recipe does. Worked in notebook 03:
two deliveries of the same keyed request, one execution.

**Sessions vs one-shot, timeouts, cancellation.** A one-shot execution is the safe default: fresh sandbox,
run, tear down. A **session** (a warm sandbox you `exec` into repeatedly) is faster (§6) but keeps state
between calls, so it needs its own lifecycle and a hard TTL. Every execution has a wall-clock deadline;
cancellation is the same process-group kill as a timeout.

---

## 4. Network and secrets

The process sandbox does not stop the network (§1, §2), so egress is its **own** control, and it is where
exfiltration and SSRF are actually contained.

**Deny by default; allowlist by host.** The sandbox has no network of its own. When code legitimately needs
an outside host, it goes through an **egress proxy** inside the trust boundary. `sandboxcore.EgressProxy`
checks the URL's host against an allowlist (`ProxyPolicy.allows`); an off-allowlist host gets 403 before any
request leaves, and the attempt is logged. This is the SSRF/exfiltration guard: "any tool that takes a URL"
needs an egress allowlist (identity primer §6.1). The check must hold on **every hop**: an allowed host that
answers `302 Location: http://elsewhere/` would otherwise have the request replayed to a host nobody checked —
and Python's default `urlopen` copies the headers, injected credential included, onto the redirected request.
So the proxy **does not follow redirects**: it hands the 3xx back, and asking for the new URL is a new
request, checked again (`tests/test_proxy.py` runs this against two loopback upstreams).

**The proxy injects the credential the sandbox never holds.** For an allowed host, the proxy adds the
credential header **outbound** (`EgressProxy.fetch` → `ProxyPolicy.inject`), so the secret lives only in the
proxy and the sandboxed code never sets or reads it. It first drops the caller's own `Authorization`,
`Proxy-Authorization`, `Cookie` and hop-by-hop headers (the sandbox can neither forward a credential it found
nor forge one), and it redacts the injected value from any response that echoes it back. This is the identity primer's **gateway path** (§5) —
"end-user credentials are … decrypted only at Agent Gateway, which injects them into the egress request; the
agent code never sees the raw credential" — realised one layer down, and the same host-side header-injection
pattern managed services like E2B use. The audit log records the header **name** injected, never its value
(notebook 04's check asserts the secret never appears in an event).

**Enforcement is the network, not the environment variable — or the tool call.** `HTTP_PROXY` only asks a
well-behaved client to use the proxy; malicious code opens a raw socket and ignores it. Likewise a tool call
that *declares* the hosts its code needs (`ExecutionRequest.egress`) is the model's own claim: checking it
refuses a hijacked model polite enough to name `attacker.example`, and nothing else. `sandboxcore.agent`
shows both: the declared exfiltration is denied before running; the undeclared one — a raw socket — leaks
through a process sandbox and is audited as an ordinary `allow`. What *forces* traffic through the proxy is
the network layer: an empty network namespace or `--network none` with the proxy on a Unix socket (the
lab's rung 1b and Docker path), or a **default-deny egress** NetworkPolicy that opens only the proxy. The
rendered policy in §5 does exactly this.

**Two honest edges.** First, **DNS**: a default-deny egress policy "also blocks DNS traffic," and the
tempting fix — re-open port 53 — reintroduces a **DNS-exfiltration** channel (data smuggled in query names,
resolved by the cluster's DNS on the sandbox's behalf). The safe shape keeps DNS closed: the sandbox reaches
only the proxy (by `hostAliases` pointing at the proxy Service's pinned ClusterIP, with `dnsPolicy: None`),
and the **proxy** resolves names and reaches the allowlist. Second, **HTTPS**: a credential cannot be injected into
an opaque `CONNECT` TLS tunnel without terminating TLS. The teaching proxy therefore brokers **plain HTTP**
(so it can read and rewrite headers) and refuses `CONNECT`; a production proxy either terminates TLS with a
CA the sandbox trusts, or is itself the TLS client while the sandbox speaks plain HTTP to it over loopback
or a cluster address.

**Package installation and secrets in prompts.** Installing packages inside a sandbox is egress by another
name — point `pip` at an internal mirror through the same proxy, or bake dependencies into the image and run
with no egress. And the identity primer's §5 rule stands: **no secrets in prompts, tool descriptions or
session state** — the context window is readable by the model and therefore by an injection. Secrets belong
in the proxy (or a broker), minted per call, short-lived, audience-bound — the identity primer's §3.5 token
exchange for the delegated case.

---

## 5. Sandboxes on Kubernetes

At scale a sandbox is a pod, and the cluster enforces the policy so the runtime code does not have to be the
only line of defence. `sandboxcore.SandboxPolicy.render_k8s()` turns one policy object into the whole set —
Namespace, RuntimeClass, two NetworkPolicies, the proxy's Service, ResourceQuota, LimitRange, a Pod and a Job
per execution, and the ValidatingAdmissionPolicy with its binding — each a plain dict that validates against
the Kubernetes 1.34 schemas (`kubernetes-validate --strict -k 1.34.0`, pinned in `tests/test_manifests.py`).
A schema check is not an admission check, so the same test also asserts what the API server would reject
and the schema cannot see: every resource request is at most its limit. The builders follow the conventions
of the GPU scheduling lab's `k8sgpu.manifests`. This section is the design; the lab applies it on kind and GKE.

**Pod-per-execution vs a warm pool.** Two shapes. **Pod-per-execution**: a Kubernetes **Job** per call,
which is auditable and clean but pays a full pod cold start every time (§6) — fine for minutes-tolerant batch
work, far too slow for interactive use on a busy cluster. **Warm pool**: keep sandbox pods ready and
`kubectl exec` (or the agent-sandbox CRD's adoption) into one, which is sub-second; delete each pod after one
use and let the pool replace it (**replace-after-use**), or it carries state between executions and needs
its own lifecycle. §6 sizes it. The GPU scheduling primer's §8 startup-latency chain is the same
idea for GPU replicas (a cold GPU replica is 380 s there); a sandbox pod is lighter but the shape is
identical.

**The Job, bounded and non-retrying.** `SandboxPolicy.job()` renders a Job with `restartPolicy: Never`,
`backoffLimit: 0` (so non-idempotent code does not silently re-run — Kubernetes' default is **6**, and Cloud
Run jobs default `max_retries` to **3**, the §1.5 idempotency trap at the infrastructure layer),
`activeDeadlineSeconds` bounding the whole Job (it **takes precedence over `backoffLimit`**), and
`ttlSecondsAfterFinished: 300` to clean up — remembering that the TTL deletes the Pod **and its logs**, so
results are collected before it fires. It sets `automountServiceAccountToken: false` so no cloud credential
rides in on the service-account token. **Deadlines count the cold start.** The Job's `activeDeadlineSeconds`
runs from the Job's start — scheduling, node scale-up and the image pull included — and a Pod's own
`activeDeadlineSeconds` runs from the kubelet admitting it, before the pull. Set either to the code's 5-second
wall budget and every cold execution on a busy cluster (§6: ~42–50 s p50) ends `DeadlineExceeded` before its
code starts. So the rendered Job's deadline is a **startup allowance plus the wall budget** (120 + 5 s by
default, `SandboxPolicy.startup_allowance_s`), and the wall budget itself is enforced **inside** the pod: the
command runs under `timeout -s KILL <wall_s>`. (The lab's pods run its wrapper, which enforces every budget
the same way.)

**`securityContext` and Pod Security *restricted*.** `SandboxPolicy.security_context()` renders the fields
the *restricted* Pod Security Standard requires (FACTS §3): pod-level `runAsNonRoot: true`, numeric
`runAsUser: 65534` (numeric so the kubelet can verify non-root without resolving an image's user name),
`seccompProfile.type: RuntimeDefault`; container-level `allowPrivilegeEscalation: false` (which defaults to
**true** if unset — pitfall 10), `readOnlyRootFilesystem: true`, `capabilities.drop: [ALL]`, and the same
seccomp profile. The Namespace carries the `pod-security.kubernetes.io/enforce: restricted` (and
audit/warn) labels at version `v1.34`. One gotcha the lab honours: PSA `enforce` applies to **Pods, not
workload objects**, so a non-conforming Job is *accepted* while its Pods are rejected — watch Job conditions,
not just `kubectl apply`.

**RuntimeClass — the real isolation.** The pod sets `runtimeClassName: gvisor`, and `render_k8s()` renders
that `node.k8s.io/v1` RuntimeClass with handler `runsc` for a self-managed gVisor node
(`SandboxPolicy.runtime_class_obj()`); on GKE, the platform creates a `gvisor` RuntimeClass with the first
GKE Sandbox node pool (its handler name is `gvisor` rather than `runsc` `(verify)`), so the renderer leaves
it out (`render_runtime_class=False`). An unknown RuntimeClass or an unrunnable handler sends the pod to phase
`Failed`. Kata's RuntimeClasses carry `overhead.podFixed` so the
scheduler and ResourceQuota account for the VMM. **kind cannot run gVisor** (§10 of the scheduling primer's
"learning locally" spirit) — kind is a teaching cluster, not a security boundary; the lab says so plainly and
uses GKE Sandbox for the gVisor path.

**NetworkPolicy: default-deny egress, then open only the proxy.** Two policies (§4): a default-deny egress
policy (`policyTypes: [Egress]`, no rules) and one that opens egress to the proxy pods on the proxy port —
and nothing else, not DNS. The pod finds the proxy without a resolver: `dnsPolicy: None`, and `hostAliases`
maps the proxy's name to its Service's pinned ClusterIP (the rendered Service), so there is no DNS channel to
exfiltrate through. The traps: a NetworkPolicy needs a plugin that enforces it (GKE Dataplane V2, Calico,
Cilium; kind's kindnetd enforces standard policy through kube-network-policies — its source at the lab's
pinned kind v0.33.0 builds that controller, v0.23.0's did not — but with **`FailOpen: true`**, and if the
controller cannot start it logs and carries on without policies, so the lab's `run-examples.sh` has a
must-fail step that checks enforcement on your cluster), it never blocks traffic to the pod's own **node** (so deny cloud
metadata with `automountServiceAccountToken: false` and a Workload-Identity KSA that has no IAM, not with
NetworkPolicy), and a pod created before the policy is handled "may be started unprotected" — apply policies
first, gate the first runner on a readiness check.

**ResourceQuota, LimitRange, emptyDir.** A ResourceQuota caps the namespace's pods and cpu/memory; because a
quota on cpu/memory **rejects pods without requests/limits**, ship a LimitRange with defaults. The workspace
is an `emptyDir` with a `sizeLimit` (an unsized `emptyDir` can fill the node; `medium: Memory` counts
against the container's memory limit) — pitfall 21. There is **no per-pod PID field**: set the kubelet's
`podPidsLimit` (a kind `KubeletConfiguration` patch, or GKE's `node_config.kubelet_config.pod_pids_limit`).

**ValidatingAdmissionPolicy: reject non-conforming sandbox pods.** Stable since Kubernetes 1.30, a
`ValidatingAdmissionPolicy` + `ValidatingAdmissionPolicyBinding` (CEL) rejects any pod in the sandbox
namespace that omits the controls — `runtimeClassName == 'gvisor'`, `runAsNonRoot`,
`allowPrivilegeEscalation: false`, `capabilities.drop` containing `ALL`. The binding must set
`validationActions: [Deny]` (Deny + Warn together is invalid), and `resources: ["pods"]` does not cover
`pods/exec` or `pods/ephemeralcontainers` — list them if you mean to police an exec-into-warm-pool path. This
is the deterministic backstop: even if the runtime code is wrong, the cluster refuses an unsafe pod.

**GKE Sandbox and Autopilot.** A GKE **Sandbox** node pool runs pods with `runtimeClassName: gvisor` on
gVisor; Autopilot also accepts `runtimeClassName: gvisor`. In Terraform the field is
`node_config.sandbox_config { type = "GVISOR" }` — **case-sensitive**, `"GVISOR"` not `"gvisor"` (the
provider's validator rejects the lowercase form; a common trap). The lab's Terraform builds a zonal Standard
cluster with a tainted, Spot, autoscale-from-zero sandbox pool and no Cloud NAT: no *internet* egress by
default — but the subnet's Private Google Access still reaches Google APIs (an attacker's bucket is a Google
API), so the NetworkPolicy (or VPC Service Controls) is what closes that path. gVisor does not: its default
`--network=sandbox` gives the workload a full userspace network stack.

---

## 6. Latency, throughput and cost per action

A sandbox per execution has a **cold start**, so the number you keep ready is a queueing problem, and the
per-turn bill is arithmetic.

**Cold start by isolation level.** The lever that dominates the design. Labelled by provenance
(`sandboxcore.pool` documents the ladder; the lab's `bench.py` measures it live):

| Level | Cold start | Source |
|---|---|---|
| fork/exec (`/bin/true`) | ~1.5 ms | measured on the build host |
| Python interpreter per execution | ~35 ms (p50, under shared load) | measured on the build host |
| `ProcessSandbox` round trip (Python, limits, per-execution UID, sweep) | ~50 ms (p50, under shared load) | measured on the build host; notebook 05 measures yours |
| container (runc) | 100–500 ms | `(verify)`; Docker itself dominates |
| gVisor (`runsc`) | runc + tens–hundreds ms | `(verify)` |
| Firecracker microVM | VMM ~12 ms + ≤125 ms to `/sbin/init` | Firecracker `SPECIFICATION.md` |
| Kata pod | ~0.5–2 s | `(unverified)` |
| full VM (GCE) | tens of seconds | `(verify)` |
| microVM restored from a snapshot | faster than a boot | `(verify)` — measure it; mind §2's one-restore-per-tenant rule |
| GKE pod on a busy cluster, burst | ~42–50 s p50 (p90 74–93 s) | agent-sandbox `docs/performance-tuning.md`: a self-run GKE cluster (k8s 1.36.2, 20× e2-standard-16), 75-claim burst |
| warm-pool adoption / `kubectl exec` | sub-second | agent-sandbox, same file |

The jump from ~35 ms (a process) to ~45 s (a fresh pod on a busy cluster) is why interactive code tools use
**warm pools** and exec-into-a-running-pod, not a pod per call.

**Three pools.** A **replace-after-use** warm pool (the safe default) hands each execution a ready sandbox,
destroys it afterwards and warms a replacement in the background: the request never waits on a cold start
if a warm one is ready, but every execution holds a slot for its run **and** its replacement's warm-up. A
**reuse** pool (one warm sandbox serves execution after execution) holds a slot for the run only — faster
and cheaper, but state carries between executions, the thing §3's one-shot default exists to avoid.
**Cold-on-demand** (no pool: a sandbox per request, started when the request arrives) puts the whole cold
start on every request's path.

**Little's law gives the mean, not the size.** Busy sandboxes = arrival rate × execution time
(`pool.busy_sandboxes`); warming replacements = rate × cold start (`pool.warming_sandboxes`). Worked:
λ = 5 executions/s, t_exec = 2 s, t_cold = 3 s → **10 busy + 15 warming = 25 slots occupied on average**
(`pool.mean_occupancy`). That is the offered load *a* = λ(t_exec + t_cold) in Erlangs — a floor. A pool of
exactly 25 has nothing spare when arrivals bunch up: `pool.erlang_c(25, 25)` is 1.0, the queue never clears.

**Choosing a slot count for a wait target (Erlang C).** `pool.erlang_c(a, c)` gives the fraction of
requests that find no warm sandbox, `pool.expected_wait_s` the mean wait. For the replace-after-use pool,
a = 25:

| slots c | P(wait) | E[Wq] |
|---|---|---|
| 26 | 0.782 | 3.912 s |
| 28 | 0.457 | 0.762 s |
| 30 | 0.250 | 0.250 s |
| 31 | 0.180 | 0.150 s |
| 33 | 0.088 | 0.055 s |
| 35 | 0.040 | 0.020 s |

`pool.slots_for_wait_target(5, 2, 3, 0.2)` returns **31** (35 for 5%). A reuse pool at the same rate has
a = λ·t_exec = 10 and needs 14 (P(wait) 0.174; c = 12 gives 0.449) — the FACTS §12 table. Erlang C assumes
exponential hold times; a fixed cold start varies less, so the real P(wait) comes out a little lower. The
discrete-event `pool.simulate()` checks this (SIMULATED; seed 1, 30,000 arrivals — notebook 05's worked
example 4, pinned in `tests/test_pool.py`): replace-after-use at 25 slots — 91% wait, 6.8 s on average; at
31 — 14% wait, 0.06 s; cold-on-demand at 31 — every request waits at least the 3 s cold start.

**Where the arrival rate comes from.** The workload, via the scaling primer's §3.2 arithmetic: **27.1 tool
calls/s at peak**; if a fifth are `run_code`, λ ≈ **5.4/s** (27.1 × 0.2 = 5.42) — a mean occupancy of 27.1
and **34** replace-after-use slots for P(wait) ≤ 0.2.

**Cost per action.** `pool.cost_per_execution(t_exec, t_cold, node_cost_per_s)` = sandbox-seconds held ×
the node's per-second share (+ any fixed per-execution fee). Every one-shot sandbox — a pod per call, a
replace-after-use pool, a pay-per-use service — spends a cold start someone pays for: **t_exec + t_cold =
5 sandbox-seconds** here. A warm pool moves the cold start off the latency path, not off the bill. Only a
reuse pool amortises it (`reuse=True`: 2 s), at the price of state. The fleet view adds the idle headroom the
wait target buys: `pool.pool_cost_per_execution(λ, c, $/s)` = c × $/s ÷ λ = 31/5 = **6.2 sandbox-seconds** per
execution. Then `pool.actions_cost(actions_per_turn, cost_per_action)` closes the scaling primer's §1 loop:
**actions/turn × cost/action** is the per-turn code-execution bill (1.3 tool calls/turn there). Any dollar figure is `(verify)` against
[`COMPUTE.md`](../../COMPUTE.md), which has no CPU-only price today — so the primer states the method and
leaves the price marked.

---

## 7. Browser, computer-use and GPU sandboxes in brief

Code execution is the common case; the same design covers three neighbours.

**Headless browsers.** A browser tool (navigate, click, screenshot, read) is a code sandbox with a browser
inside and the same three "no"s: an ephemeral **profile** (no saved cookies or logins carried between
sessions or tenants), **downloads** confined to the workspace, and **egress** through the same allowlisting
proxy — a browser that can reach arbitrary URLs is an SSRF and exfiltration engine. Screenshot-to-action
loops (the model sees a screenshot and emits a click) add an untrusted-content surface: the *page* is
attacker-controlled, so its text and any on-page instructions are data, screened like any tool result.

**Computer-use.** The same, one level up: a whole desktop the model drives. It needs a full VM or a strong
container (the desktop process tree is large), an ephemeral disk, egress control, and a human-visible action
log — you must be able to see what it actually clicked, not what it said it would.

**GPU sandboxes.** An agent that launches a GPU job (fine-tuning, a heavy eval) runs the same contract plus
the GPU-scheduling layer: the pod requests `nvidia.com/gpu` (an integer, requests == limits), tolerates the
GPU taint and selects the accelerator, and — for isolation — gVisor's `--nvproxy` forwards NVIDIA `ioctl`s
to the host driver with negligible overhead (supported on T4/L4/A100/H100). Everything in the GPU scheduling
primer (§1 what Kubernetes sees, §3 the scheduling cycle, §9 sharing) applies unchanged; the sandbox adds
the security context, the network policy and the budget.

---

## 8. Observability, audit and abuse detection

Every execution must leave a record, the way every tool call does.

**One structured event per execution.** `sandboxcore.audit.AuditEvent` mirrors the identity primer's audit
event (§9 "minimum viable audit … trace ID, invocation ID, user, agent identity, authority mode, tool,
argument hash, policy decision and reasons, approver, result hash, latency, provenance") — the same field
names as the identity lab's `agentsec.audit.AuditEvent`, a standalone copy so the core stays
dependency-free, with `args_digest` computed the same way (sha256 of canonical JSON, first 16 hex). It adds
the fields a sandbox needs: **`budgets_used`** (CPU, wall, peak memory, disk, output), **`exit_reason`**
with its **`exit_reason_source`**, and **`policy_decision`**. The event is
emitted from the executor layer, so it exists even when the code crashes or is killed — the deny path and
the kill path both leave evidence (`SandboxAgent._audit`).

**Metrics.** p50/p95 **startup** and **run** time by isolation level; the **exit-reason histogram**
(`AuditLog.counts()`) — a rising `cpu_time`/`wall_timeout`/`pids`/`output_limit` rate is the abuse signal;
sustained `memory`/`cpu_time` is a miner or a runaway loop; a spike in `pids` is a fork bomb. Count only
reasons the sandbox observed (`counts(trusted_only=True)` drops the ones read from the program's own stderr,
which a hostile program can forge to hide or to frame). Egress attempts need care: a spike of `denied`
*declared* egress is an injection that announced itself, but a competent one does not declare anything — its
attempts show up only where the network is enforced, as proxy 403s and NetworkPolicy drops, so collect those
logs too.

**Detection and response.** The controls of §1–§5 *bound* abuse; the audit trail *detects* it and drives
response. Because every execution is attributable to a principal and carries an argument hash, an
incident is traceable to the content that caused it (provenance tags), and response is the identity primer's
kill switch: a policy change (deny the principal, tighten the egress allowlist) that takes effect without a
redeploy. The scaling primer's §5.3 admission control belongs here too — shed `run_code` first under load,
since it is the most expensive action.

---

## 9. Where to run it

GCP is one target, never a prerequisite; every concept is learnable at T0.

**Laptop (T0).** `sandboxcore.ProcessSandbox` runs the whole conceptual model — clean env, rlimits,
wall-clock kill, streamed output, the policy engine, the egress proxy, the pool arithmetic, and the rendered
manifests (validated offline). As a non-root user it cannot switch UID, and says so; Colab (root) gets the
per-execution UID. The lab's `docker.py` adds a hardened `docker run` (and `--runtime=runsc` if
gVisor is installed) where Docker is available. This is where you learn everything.

**kind (T0 + Docker).** A kind cluster runs the real scheduler, Pod Security, NetworkPolicy (kindnetd at
kind v0.33.0 enforces standard policy but **fails open** — a broken policy controller means no policies,
silently — so the lab's must-fail egress step checks it on your cluster; if it fails, create the cluster
with `networking.disableDefaultCNI: true` and install Calico) and the ValidatingAdmissionPolicy;
the lab's `deploy/kind` creates the restricted namespace, the default-deny NetworkPolicy, the egress proxy,
the quota and the runner Job. **kind cannot run gVisor**, so the runtime-class path is inspected, not
executed, here.

**One GPU/CPU box (T1).** A rented VM or box with Docker runs the hardened container and, with `runsc`
installed, real gVisor; a nested-virt or bare-metal host runs Firecracker/Kata (they need `/dev/kvm`, absent
on Colab and most laptops' Docker).

**GCP (T3).** The lab's Terraform builds a zonal GKE Standard cluster with a **GKE Sandbox (gVisor)** node
pool (`sandbox_config { type = "GVISOR" }`, tainted, Spot, autoscale-from-zero), no Cloud NAT (no *internet*
egress by default — Google APIs stay reachable over Private Google Access until the NetworkPolicy closes
them, §5), Artifact Registry for the sandbox image, and managed Prometheus; `deploy/gke` has the RuntimeClass,
namespace, NetworkPolicy, runner Job and admission policy. Cloud Run **jobs** are the serverless option:
the first-generation execution environment is gVisor-based and the second a microVM-based full Linux
`(verify)`; set `max_retries: 0` for non-idempotent code (the default is 3); and mind that **Cloud Run egress
to the internet is on by default** `(verify)` — closing it needs Direct VPC egress with `ALL_TRAFFIC` plus
firewall rules, so a Cloud Run job is not "no network" until you do.

**Managed sandbox services** (verify-marked; state the default per provider, since "sandbox" ≠ "no network"):

| Service | Isolation | Egress default | Notes |
|---|---|---|---|
| E2B | Firecracker microVMs `(unverified)` | **open** unless `allow_internet_access=False` | host-side header injection like this topic's proxy; `E2B_API_KEY` |
| Modal | gVisor-based `(verify)` | `(verify)` | serverless Python GPUs |
| Daytona | `(verify)` | `(verify)` | — |
| Vertex AI Agent Engine code execution / Agent Sandbox | `(verify)` | `(verify)` | managed; no latency figure checked here |
| Cloud Run jobs | gen1 gVisor / gen2 microVM `(verify)` | **open** `(verify)`; close with Direct VPC egress `ALL_TRAFFIC` + firewall | `max_retries` defaults to 3 |
| kubernetes-sigs/agent-sandbox (self-hosted on your GKE/K8s) | whatever `RuntimeClass` you give it (gVisor, Kata) | its `SandboxTemplate` network policy: empty rules ⇒ default deny | measured on a self-run cluster (k8s 1.36.2, 20× e2-standard-16, 75-claim burst): pod cold start ~42–50 s p50; warm adoption sub-second (`docs/performance-tuning.md`) |

Prices and obtainability: [`COMPUTE.md`](../../COMPUTE.md).

---

## In a design review

**The two-minute walkthrough.** "A `run_code` tool takes untrusted, model-written programs and runs them, so
it is the highest-risk tool we have — DESTRUCTIVE tier by definition. The design goal is *no ambient
authority*: no credentials, no network by default, no persistent filesystem. I get there on an isolation
ladder — in-process restrictions are not a boundary, a separate process with a clean environment and OS
resource limits is the first real one, then a hardened container, then gVisor's user-space kernel, then a
microVM with its own guest kernel — and I pick the rung by what I'm defending against and the cold-start
budget. The execution is an API with a contract: code plus inputs plus a budget for every exhaustible
resource in, truncated output plus an exit reason the model can act on out, and an idempotency key so an
at-least-once redelivery doesn't run a side effect twice. Even the process rung runs each execution as its
own UID, because that is what keeps my files unreadable and finds a process that left the group. The network
is a separate control: deny-by-default egress, and an allowlisting proxy that injects credentials the sandbox
never holds and never follows a redirect — the hosts a tool call declares are the model's claim, not a
control. On Kubernetes I render the same policy to Pod Security restricted, a default-deny NetworkPolicy that
opens only the proxy (no DNS), a non-retrying Job whose deadline allows for the cold start, and a
ValidatingAdmissionPolicy that refuses any pod missing the controls — deterministic backstops for when the
runtime code is wrong. Sandboxes have a cold start, so I size a replace-after-use warm pool with Erlang C on
the workload's arrival rate times run-plus-warm-up — Little's law only gives the floor — and cost per action
is sandbox-seconds, cold start included, times the node price. Every execution leaves one audit event with the principal, the policy decision, budgets used and
the exit reason, and I detect abuse from the exit-reason histogram and shed code execution first under load."

**Drill questions**

1. *Why is a code-execution tool the most dangerous tool an agent has, and what tier is it?* — It turns
   arbitrary model output into arbitrary computation with whatever authority the runner holds, so one
   injection becomes code execution (ASI05). It is DESTRUCTIVE tier by definition (identity primer §6.2).

2. *A colleague says "we sandbox with CPU, memory and PID limits, so we're safe." What's missing?* — The
   network (rlimits never touch sockets — egress needs deny-by-default + a proxy or a netns) and the kernel
   boundary (a process sandbox shares the host kernel; a kernel exploit escapes it). Resource limits bound
   abuse, not exfiltration or escape.

3. *You set a 2-second CPU limit and code still hangs for a minute. Why, and what do you add?* — It is
   sleeping or blocked on I/O and uses no CPU, so `RLIMIT_CPU` never fires. Add a wall-clock timeout that
   kills the whole process group (a naive `subprocess` timeout leaves grandchildren alive).

4. *Where does the credential live when sandboxed code calls an allowed API, and how does the code get it?* —
   Only in the egress proxy. The code makes a plain request to the proxy; the proxy checks the allowlist and
   injects the credential outbound. The code never sets or reads it — the identity primer's gateway path.

5. *Your interactive code tool has a 2-second p95 budget but pods cold-start in ~45 seconds on a busy
   cluster. What do you change?* — Do not start a pod per call. Keep a replace-after-use warm pool and hand
   out ready sandboxes (sub-second), sized with Erlang C on λ(t_exec + t_cold) — busy + warming is only the
   mean. Or use a lighter boundary (gVisor, a microVM restored from a snapshot), remembering that a snapshot
   restored into two tenants shares its RNG state and secrets: one restore per tenant, re-seed after restore.

6. *You run agents as root in CI, with `RLIMIT_NPROC` set. Are fork bombs contained?* — No. `RLIMIT_NPROC` is
   ignored for uid 0 and counts per real UID system-wide, so it does nothing as root. Switch each execution to
   its own unprivileged UID first (which needs privilege to do — a shared `nobody` makes concurrent runs share
   one budget), or enforce the process limit at the container/cgroup layer.

7. *We kill the process group on timeout and set `HOME` to a temp dir. What can still escape?* — A process
   that calls `setsid()` leaves the group and outlives the call, and `HOME` only moves `~`: the real key opens
   by its absolute path. Both need a different UID (killed and swept by UID after the run) or a
   container/PID/mount namespace.

---

## Glossary

| Term | Meaning |
|---|---|
| Ambient authority | Credentials, network and files a process holds by default (inherited env, home, metadata server). A sandbox removes it. |
| Isolation ladder | in-process → process+rlimits → container → gVisor → microVM → VM, increasing defence and cost. |
| rlimit | A POSIX per-process resource limit (`setrlimit`): CPU seconds, address space, processes, file size, open files. |
| `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_NPROC` / `RLIMIT_FSIZE` | CPU seconds / virtual memory / processes per real UID / largest single file. |
| Process group / `setsid` | A set of processes killed together; the sandbox puts the child in its own session so a timeout kills grandchildren too. |
| Exit reason | The fixed vocabulary a sandbox returns (`ok`, `error`, `wall_timeout`, `cpu_time`, `memory`, `file_too_large`, `pids`, `open_files`, `output_limit`, `disk_limit`, `killed`, `denied`, `sandbox_error`) so the model can recover; its source (`parent`/`signal`/`code`) says how far to trust it. |
| Idempotency key | turn + step + call index + args hash; lets an at-least-once redelivery return a stored result (scaling primer §5.4). |
| Egress proxy | An allowlisting forward proxy inside the boundary that injects credentials outbound; the sandbox holds no keys. |
| gVisor / Sentry / Gofer | A user-space kernel (Sentry, Go) that intercepts syscalls and never passes them to the host; a Gofer owns the filesystem. Runs as `runsc`. |
| Firecracker / Kata / microVM | Lightweight VMs with their own guest kernel; a hardware isolation boundary; need `/dev/kvm`. |
| RuntimeClass | A Kubernetes object naming a CRI handler (`runsc`, a Kata shim); pods select it with `runtimeClassName`. |
| Pod Security *restricted* | The strictest built-in Pod Security Standard: non-root, no privilege escalation, drop ALL caps, seccomp RuntimeDefault, limited volumes. |
| NetworkPolicy (default-deny egress) | A policy that drops all egress; you re-open only the proxy — not DNS, which would be an exfiltration channel. Needs an enforcing CNI. |
| ValidatingAdmissionPolicy | CEL-based admission control (stable 1.30) that rejects non-conforming pods; the deterministic backstop. |
| Little's law | in-flight = arrival rate × time held; gives a pool's *mean* occupancy (busy + warming), the floor below any workable size. |
| Erlang C | The M/M/c formula for the fraction of requests that wait; turns a wait target into a slot count. |
| Replace-after-use / reuse pool | A warm pool whose sandboxes run one execution and are replaced (no state carried; each execution pays a warm-up) vs one whose sandboxes serve many (cheaper, stateful). |
| Per-execution UID | A fresh unprivileged UID for each run: makes `RLIMIT_NPROC` real, keeps your files unreadable, and lets the sandbox find and kill every process the run left behind. |
| Cold start | Time from "need a sandbox" to "code running": ms for a process, ~125 ms for a microVM, tens of seconds for a fresh pod. |

---

## Sources

- **gVisor** (`gvisor.dev`, repo `google/gvisor` g3doc, snapshot 2026-09-25): `README.md`,
  `architecture_guide/intro_to_gvisor.md`, `security.md`, `user_guide/{platforms,networking,filesystem,gpu,
  install}.md`, `user_guide/containerd/quick_start.md`, `tutorials/kubernetes.md`.
- **Firecracker** (`firecracker-microvm/firecracker`, 2026-09-25): `SPECIFICATION.md`, `README.md`,
  `docs/{seccomp,jailer,getting-started,design}.md`, `docs/snapshotting/snapshot-support.md`.
- **Kata Containers** (`kata-containers/kata-containers`, 2026-09-25): `docs/{quick-start-guide,hypervisors,
  installation}.md`, `releases/4.2.0.md`, kata-deploy `runtimeclasses.yaml`.
- **Kubernetes** (`kubernetes/website`, 2026-09-26, K8s 1.34): Pod Security Standards & Admission; RuntimeClass
  & Pod Overhead; ValidatingAdmissionPolicy; NetworkPolicy; Job; securityContext; ResourceQuota; LimitRange;
  PID limiting; volumes; configure-service-account; RBAC.
- **kind** (`kubernetes-sigs/kind`), **KWOK**, Run:ai `fake-gpu-operator` (learning locally).
- **kubernetes-sigs/agent-sandbox** (SIG Apps, 2026-09-25): CRDs `Sandbox`/`SandboxTemplate`/`SandboxClaim`/
  `SandboxWarmPool`; `docs/performance-tuning.md` (cold-start measurements).
- **E2B** (`e2b-dev/e2b`), **Docker** (docker/cli, docker/docs; moby default seccomp profile).
- **CPython 3.11** docs: `resource`, `subprocess` (the `preexec_fn`/threads and grandchild caveats).
- **Terraform google provider 8.4.0**: `google_container_node_pool.node_config.sandbox_config`,
  `google_cloud_run_v2_job` execution environment and retries.
- **Repo material cited, not restated**: identity primer (`06-gateway/identity-security/
  agentic-identity-gcp-lab/docs/primer.md` §2, §3.5, §4.2, §5, §6.2, §9; `agentsec_core.py`), scaling primer
  (`06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md` §1, §3.2, §5.3, §5.4,
  §5.6), GPU scheduling primer (`03-kubernetes-gpu/gpu-scheduling/PRIMER.md` §1, §3, §8, §9, §10) and its
  lab, agent-core (`07-application-agent-framework/agent-fundamentals/agent-core` tool/loop contracts).

---

## Verify list

Dated 26 September 2026. Re-check before relying on any of these.

| Fact | Status here |
|---|---|
| OWASP Top 10 for Agentic Applications (ASI01–ASI10), published 9 Dec 2025 | from the identity primer; keep "OWASP wording" verify |
| gVisor install now ships a `gvisor-bin/` dir beside `runsc` (tarball); the old single-binary snippet is stale; auto-download of sidecars dropped end of Sep 2026 | gVisor docs; verify current release |
| gVisor `systrap` default platform; `kvm` for bare metal; `--nvproxy` GPUs T4/L4/A100/H100 | gVisor docs; verify GPU/driver window |
| Container cold start 100–500 ms; gVisor adds tens–hundreds ms | `(verify)` — docs show a graph, no number |
| Firecracker: VMM ~12 ms typical, ≤125 ms to `/sbin/init`, ≤5 MiB VMM overhead, needs `/dev/kvm` | `SPECIFICATION.md`; Kata boot ~0.5–2 s is `(unverified)` |
| GKE pod cold start ~42–50 s p50 (burst, a self-run 20-node e2-standard-16 cluster, k8s 1.36.2); warm adoption sub-second | agent-sandbox `docs/performance-tuning.md` — not a managed service's figure |
| Terraform `sandbox_config.type = "GVISOR"` (case-sensitive; `"gvisor"` fails validate) | provider 8.4.0 source `tf-node_config.go` |
| Kubernetes: PSA `enforce` applies to Pods not workloads; `RLIMIT_NPROC` root/UID behaviour; emptyDir sizeLimit; no per-pod PID field | K8s docs / CPython docs / measured (FACTS §10) |
| kindnetd enforces NetworkPolicy (kube-network-policies) with `FailOpen: true`; kind cannot run gVisor | kindnetd `main.go` at tags v0.33.0 and v0.24.0 builds the policy controller, v0.23.0 does not (read 2026-09-26); that the pinned node image bundles that kindnetd is `(verify)` — the lab's must-fail egress step checks it on your cluster (fallback: `disableDefaultCNI` + Calico) |
| GKE creates RuntimeClass `gvisor` (handler `gvisor`, scheduling + toleration for sandbox nodes) | `(verify)` — `kubectl get runtimeclass gvisor -o yaml` |
| Cloud Run egress open by default; closing it needs Direct VPC egress `ALL_TRAFFIC` + firewall | `(verify)`, docs blocked |
| Snapshot/restore start-up saving (Firecracker, `runsc checkpoint/restore`) | `(verify)`: no number here; the one-restore-per-tenant rule is from Firecracker's snapshot docs |
| E2B egress open by default (`allow_internet_access=False` to deny); Firecracker-based; ~150 ms starts | E2B repo; isolation/latency `(unverified)` |
| Cloud Run jobs `max_retries` default 3; gen1 gVisor / gen2 microVM | provider schema; gen split `(unverified)`, docs blocked |
| Node per-second price for cost-per-execution | `(verify)` — `COMPUTE.md` has no CPU-only price yet |
| kind v0.33.0, node image `kindest/node:v1.34.11@sha256:44e2…`, agent-sandbox v1.0.2, Kata 4.2.0 | FACTS §14; verify latest |
