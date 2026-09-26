# FACTS — 07 · sandboxed-execution (researched 2026-09-26)

Source of truth for the `sandbox-core` / `sandbox-lab` builders and reviewers. Extends `$SP/FACTS.md` (not repeated here).
`$REF` = `$SP/ref`. Every bullet names its source. **(unverified)** = not found in a source; best knowledge + why.
Upstream snapshots: gvisor `849f391` (2026-09-25), k8s-website `af194d7` (2026-09-26), firecracker `edb6061` (2026-09-25),
kata `8a2c6ac` (2026-09-25), kind `dd00511` (2026-09-24), e2b `ccaf9fc` (2026-09-18), kubernetes-sigs/agent-sandbox `68db683`
(2026-09-25, cloned this session to `$REF/agent-sandbox`). Extra files fetched this session: `$REF/docker/*` (docker/docs + docker/cli
reference), `$REF/tf-node_config.go` (terraform-provider-google), `$REF/k8s-volumes.md`, `$REF/kindnetd-main.go`,
`$REF/kata-values.yaml`, `$REF/kata-runtimeclasses.yaml`, `$REF/py-resource.rst`, `$REF/py-subprocess.rst` (CPython 3.11 docs).

## 1. Repo material the new topic must cite, not restate
Paths are relative to the repo root; from `07-application-agent-framework/sandboxed-execution/` prefix `../../` (from a core/lab dir `../../../`).
- **Identity primer** `06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md`:
  - `## 2. Threat model` — the OWASP table ASI01–ASI10 (§13 below). ASI05 row: "Unexpected Code Execution | Model-generated code runs
    with the agent's credentials | Sandboxed execution (Agent Sandbox) with no ambient credentials; separate identity for executors".
  - `### 6.2 Code execution and sandboxes` (full text, quote-worthy): "Model-generated code runs with *no ambient credentials*. Google's
    Agent Sandbox (and Workspaces) exist for this; if you must run code elsewhere, run it under a separate, unprivileged identity, without
    the agent's metadata-server access, with network egress off by default. An "execute code" tool is DESTRUCTIVE-tier by definition."
  - `### 3.5 Delegation mechanics (standards you should be able to draw)` — RFC 8693 token exchange (`sub` + `act`), DPoP (RFC 9449),
    mTLS-bound (RFC 8705), Credential Access Boundaries; lab's `TokenIssuer.exchange()`. Link it for "proxy injects a credential".
  - `## 5. Secrets and credential handling` — "Direct path" vs "Gateway path" (Agent Gateway decrypts and injects at the edge; the agent
    never sees the raw credential). The sandbox egress proxy is the same "gateway path" pattern one layer down.
  - `### 4.2 Tool tiers and deny-by-default` — tiers READ / WRITE / DESTRUCTIVE / EXTERNAL (`agentsec.policy.model.Tier`: `READ="read"`,
    `WRITE="write"`, `DESTRUCTIVE="destructive"`, `EXTERNAL="external"`). `fetch_url` is DESTRUCTIVE/EXTERNAL tier, needs egress allowlist.
  - `## 9. Observability, audit, and governance` — "Minimum viable audit … trace ID, invocation ID, user (if delegated), agent SPIFFE ID,
    authority mode, tool, argument hash (or redacted arguments), policy decision and reasons, approver, result hash, latency, provenance tags".
  - Code: `agentsec/audit/log.py` `@dataclass AuditEvent(event_type, agent, authority, user=None, tool=None, decision=None, reasons=[],
    args_hash=None, args_redacted=None, result_hash=None, approver=None, provenance=[], trace_id=None, invocation_id=None,
    session_id=None, latency_ms=None, ts, id, extra={})`; `event_type` values used: `tool.decision | tool.result | model.screen |
    credential.retrieve | egress | confirmation`; `args_digest(args)` = `sha256(json.dumps(args, sort_keys=True, default=str,
    separators=(",", ":")))[:16]`. `sandboxcore.audit` should mirror these field names (standalone copy, no import) and add
    `budgets_used`, `exit_reason`, `policy_decision`. `agentsec/policy/engine.py`: `ToolCallRequest(tool, args, authority, invocation_id,
    counters, confirmed_by)`, `Decision(effect: Effect.ALLOW|DENY|CONFIRM, reasons, tool_policy, hint)`.
- **Scaling primer** `06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md` (§1 `## 1. What is different
  about scaling agents`; §5 `## 5. The scaling mechanisms`): numbers to reuse, not recompute — 1.3 tool calls/turn, 20.8 turns/s peak,
  **27.1 tool calls/s peak** (§3.2); Little's law 6.9 turns/s × 6 s ≈ 42 in flight (§3.3); §1.5 "at-least-once means idempotency";
  §5.4 "keys every write by turn, step, call index and a hash of the arguments" (use this as the idempotency-key recipe); Cloud Run gives
  "ten seconds after SIGTERM" (§1.5); §5.3 admission control and degrade levels; §5.6 tools: bulkhead, breaker, timeout, one retry for
  idempotent reads only, failures returned as structured data.
- **03 PRIMER** `03-kubernetes-gpu/gpu-scheduling/PRIMER.md`: `## 1. What Kubernetes sees` (1.1 extended resources … 1.4 Taints),
  `## 3. The scheduling cycle`, `## 8. Startup latency` (`gpusched.autoscaler.startup_latency()`; 380 s cold vs 70 s warm GPU replica),
  `## 9. Sharing GPUs at the cluster level`, `### 7.3 Queued provisioning, ComputeClass, Autopilot`, `## 10. Learning locally`.
- **03 lab conventions** `03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/`: `k8sgpu/manifests.py` — `API_VERSIONS` dict, `obj(kind, name,
  namespace=None, *, labels, annotations, **body)`, `_meta()`, `pod_spec()`, `pod_template()`, `job(name, namespace, template, *,
  parallelism=1, completions=None, queue=None, priority_class=None, backoff_limit=0, indexed=False, labels=None, annotations=None,
  active_deadline_s=None, ttl_after_finished_s=None)`, `namespace(name, labels)`, `to_yaml(*objs, header=None)` (a `_Dumper`
  `yaml.SafeDumper` that keeps insertion order), `loads_all()`, `load_all()`, `iter_pod_templates()`; `k8sgpu/render.py` —
  `GENERATED_DIRS`, `rendered_files()`, `write()`, `orphans()`, `stale()`, CLI `python3 -m k8sgpu render --check`; `k8sgpu/lint.py` —
  `@rule` functions over a `Context`, `parse_quantity()`. Deploy: `deploy/lib.sh` (`log`, `warn`, `die`, `quote_cmd`, `run` honouring
  `DRY_RUN=1`, `retry ATTEMPTS SLEEP CMD`, `need CMD HINT`), `deploy/versions.env` (**`KIND_VERSION=v0.33.0`,
  `KIND_NODE_IMAGE=kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d`**,
  `LAB_IMAGE=busybox:1.38.0`), `deploy/kind/up.sh` (sources lib.sh + versions.env, `CLUSTER_NAME`, idempotent, numbered `log "1/4 …"`
  steps), `down.sh`. Terraform `deploy/gcp/terraform/{versions,variables,network,cluster,node_pools,storage,apis,outputs}.tf` +
  `terraform.tfvars.example`; `default_labels = var.labels`; taint effect in TF is **`"NO_SCHEDULE"`** (enum), not `NoSchedule`.
- **agent-core (07.1)** `07-application-agent-framework/agent-fundamentals/agent-core/agentcore/` (package `agentcore`, not `agentlab`):
  `tools.py` — `@tool` / `@tool(confirm=True)` builds `Tool(name, description, schema, fn, required, confirm)`; schema
  `{"name", "description", "parameters": {"type": "object", "properties", "required"}}`; `Tool.run(args)` returns
  `{"ok": True, "data": …}` or `{"ok": False, "error": <kind>, "message": …, "hint"?: …}` (kinds seen: `invalid_arguments`,
  `tool_failure`, `unknown_tool`, `declined`); `ToolError(message, kind="tool_error", hint=None)`; `tool_message(tc, result)` →
  `{"role": "tool", "name", "tool_call_id", "content": json}`. `agent.py` — `Agent(llm, tools, instruction, max_steps=6).run(user_message,
  history=None, on_confirm=None) -> Result(text, messages, steps, done)`. `fake_llm.py` — `FakeLLM(responses=None, policy=None)`,
  helpers `call(name, **args)`, `calls(...)`, `text(t)`, `ToolCall(name, args, id)`, `Response(text, tool_calls)`. The sandbox
  `run_code` tool must return this same result shape (exit reason in `error`, truncated output in `data`).

## 2. gVisor (`$REF/gvisor/g3doc`)
- What it is: "an application kernel that implements a Linux-like interface … written in a memory-safe language (Go) and runs in
  userspace"; OCI runtime **`runsc`** (`g3doc/README.md`). "not a syscall filter … also not a VM" (same).
- **Sentry**: intercepts syscalls and page faults, re-implements Linux (syscalls, mm, filesystems, network stack, processes, signals) in
  Go; "**gVisor never passes through any system call to the host**"; its own seccomp filter prohibits e.g. `exec(2)`, `connect(2)` on
  the host; runs in an empty mount namespace, isolated user namespace. **Gofer**: "a slightly-more-trusted companion process" that owns
  the container filesystem and donates FDs (`architecture_guide/intro_to_gvisor.md`, `security.md` §"sandbox's own interactions").
  Escaping requires exploiting "the gVisor Sentry kernel *and* the host Linux kernel, which do not share any code" (intro).
- **Directfs** on by default (`--directfs=false` disables): Sentry opens files via FDs donated by the gofer; `O_NOFOLLOW` enforced via
  seccomp (`user_guide/filesystem.md` §Directfs).
- **Platforms**: `systrap` (**default** since mid-2023; seccomp `SECCOMP_RET_TRAP` for interception; no virtualization needed; good inside
  VMs), `kvm` (best on bare metal; nested virt slower, "not recommended for production"), `ptrace` ("no longer supported … expected to
  eventually be removed"). Flag `--platform=kvm` (`user_guide/platforms.md`, `architecture_guide/platforms.md`).
- **Does NOT protect against** (intro §"What does gVisor *not* protect against?"): attacks in higher layers (e.g. containerd exploit that
  starts a container without gVisor); Spectre-style side channels; exploits *within* the workload ("the attacker will still have access to
  whatever the sandbox is configured to have access to" → run different tenants in different sandboxes).
- Requirements: "x86_64 and ARM64, and requires Linux 5.6+" (`user_guide/install.md`). Install: apt repo
  `https://storage.googleapis.com/gvisor/releases release main`, `sudo apt-get install -y runsc` ("If you have Docker installed, it will be
  automatically configured"); or tarball `…/releases/release/latest/${ARCH}/gvisor.tar.zstd` containing `runsc`,
  `containerd-shim-runsc-v1` and a **`gvisor-bin/` directory that must stay next to `runsc`** (new since 2026-07; old single-binary
  download instructions are stale; runsc's auto-download of missing sidecars "will be dropped at the end of September 2026").
- Docker: `sudo runsc install` (adds runtime "runsc" to `/etc/docker/daemon.json`), `sudo systemctl restart docker`, then
  `docker run --runtime=runsc --rm hello-world`. Extra flags per runtime: `sudo runsc install --runtime runsc-debug -- --debug …`
  (`user_guide/quick_start/docker.md`). Verify with `dmesg` ("Starting gVisor…") but docs warn an attacker can fake this.
- One-off without Docker: `sudo runsc do echo Hello world`; rootless: `runsc --rootless --network=none do …` (intro). Rootless
  "mainly only suitable for `runsc do`" (`user_guide/rootless.md`).
- **Network** (`user_guide/networking.md`): userspace stack "netstack" in the Sentry; `--network=host` = passthrough ("decreases the
  isolation"); `--network=none` = only a netstack loopback; default `--network=sandbox`. Egress shaping `--qdisc=tbf
  --qdisc-tbf-rate=<bytes/s> --qdisc-tbf-burst=<bytes>` (plain integers; both required), per-pod via annotations
  `dev.gvisor.flag.qdisc*` (annotations can only lower limits).
- **Rootfs overlay** (`user_guide/filesystem.md`): `--overlay2={mount}:{medium}[,size=]`, media `memory`, `self`, `dir=/path`;
  default `--overlay2=root:self` (writes land in a file-backed tmpfs, not on the host rootfs); `--overlay2=none` disables.
- **Resource limits caveat** (`user_guide/compatibility.md`): "in-sandbox cgroups … exist … resource *limits* are not enforced within the
  sandbox"; limit the sandbox via its host cgroup (what Docker/Kubernetes do). Also: `io_uring` disabled by default; no KVM inside;
  `iptables` partial; block-device filesystems not mountable.
- **Performance** (`architecture_guide/performance.md`): no CPU cost for pure compute ("code is executing natively"); raw memory access no
  cost; syscall-heavy workloads pay (redis small ops "large overhead"); fixed memory overhead "small, mostly fixed"; start-up graph only
  (no number in text) and "most of the time overhead above is associated [with] Docker itself" → gVisor cold-start ms is **(verify)**.
- **Checkpoint/restore** (`user_guide/checkpoint_restore.md`): `runsc checkpoint --image-path=<dir> <id>` (`--leave-running` optional;
  default stops the container), restore into a new container: `runsc create <id>` then `runsc restore --image-path=<dir> <id>`;
  `--compression=none` (default) | `flate-best-speed`; `--exclude-committed-zero-pages`; `--direct`. Raw `runsc` only (not via Docker CLI).
- **GPU** (`user_guide/gpu.md`): `runsc --nvproxy`; `nvproxy` forwards NVIDIA `ioctl`s to the host driver (no KMD emulation; "negligible"
  overhead). Supported GPUs: T4, A100, A10G, L4, H100 (others on same arch "will likely work"). Driver versions: strict match;
  `runsc nvproxy list-supported-drivers`; `--nvproxy-allow-unsupported-driver`; supported window "aligns with those available within
  GKE". Docker: `docker run --runtime=runsc --gpus=all …`; works with NVIDIA k8s-device-plugin (incl. CDI) and GKE's device plugin.
- **Kubernetes**: containerd `runtime_type = "io.containerd.runsc.v1"` under `[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runsc]`
  (containerd ≥ 1.3.9/1.4.3; `version = 3` header for containerd 2.x); RuntimeClass `apiVersion: node.k8s.io/v1, kind: RuntimeClass,
  metadata.name: gvisor, handler: runsc` (`user_guide/containerd/quick_start.md`, `crio.md`). Minikube has a gvisor addon
  (`quick_start/kubernetes.md`). GKE: pods set `runtimeClassName: gvisor`; node pool `gcloud container node-pools create gvisor
  --cluster=… --sandbox type=gvisor --machine-type=e2-standard-2` (`tutorials/kubernetes.md`). Autopilot: `runtimeClassName: gvisor` works
  on `gcloud container clusters create-auto` (`tutorials/docker-in-gke-sandbox.md`).
- gVisor is used by Google Cloud Run and DigitalOcean App Platform (`user_guide/compatibility.md`).
- Experimental SDKs (`user_guide/sdk/*.md`, "not meant for production use"): Python `pip install gvisor`; `from gvisor import Sandbox;
  with Sandbox() as sb: stdout, stderr = sb.exec("uname", "-a")`; Go `sandbox.New(ctx, sandbox.WithNetworking(false))` (networking
  needs root).

## 3. Kubernetes (`$REF/k8s-website/content/en/docs`, K8s 1.34 validation via `kubernetes-validate --strict -k 1.34.0`)
**Pod Security Standards** (`concepts/security/pod-security-standards.md`). *Restricted* = everything in Baseline plus:
- Volume types: every `spec.volumes[*]` must be one of `configMap`, `csi`, `downwardAPI`, `emptyDir`, `ephemeral`,
  `persistentVolumeClaim`, `projected`, `secret`.
- `allowPrivilegeEscalation` (containers, initContainers, ephemeralContainers): must be **`false`** (Linux only).
- `runAsNonRoot`: **`true`** (container fields may be nil if pod-level `spec.securityContext.runAsNonRoot: true`).
- `runAsUser` (pod + containers): any non-zero value or undefined.
- `seccompProfile.type`: **`RuntimeDefault` or `Localhost`**; absence and `Unconfined` prohibited (pod-level may cover containers).
- `capabilities.drop` must include **`ALL`**; `capabilities.add` only `NET_BIND_SERVICE` or nil.
- Baseline (also enforced by restricted): `hostNetwork/hostPID/hostIPC` nil/false; `privileged` nil/false; `capabilities.add` limited to
  AUDIT_WRITE, CHOWN, DAC_OVERRIDE, FOWNER, FSETID, KILL, MKNOD, NET_BIND_SERVICE, SETFCAP, SETGID, SETPCAP, SETUID, SYS_CHROOT;
  no `hostPath`; `hostPort` nil/0; probe/lifecycle `host` field nil/"" (v1.34+); AppArmor `appArmorProfile.type` nil|`RuntimeDefault`|
  `Localhost`; SELinux type limited, user/role forbidden; `procMount` nil|`Default`; seccomp not `Unconfined`; sysctls only the safe set.
- **Pod Security Admission** labels (`concepts/security/pod-security-admission.md`): `pod-security.kubernetes.io/<MODE>: <LEVEL>` and
  `pod-security.kubernetes.io/<MODE>-version: <VERSION>`; MODE ∈ `enforce|audit|warn`; LEVEL ∈ `privileged|baseline|restricted`;
  VERSION a minor like `v1.34` or `latest`. **`enforce` applies to Pods only, not to workload objects** (a violating Job is created; its
  Pods are rejected); `audit`/`warn` also evaluate workload templates.
**RuntimeClass** (`concepts/containers/runtime-class.md`, `concepts/scheduling-eviction/pod-overhead.md`):
- `apiVersion: node.k8s.io/v1`, cluster-scoped, fields `handler` (DNS label naming the CRI handler), `scheduling.nodeSelector`
  (merged with the pod's; conflict → rejected), `scheduling.tolerations` (union with the pod's), `overhead.podFixed` (counted by
  scheduler, ResourceQuota and pod cgroup). Docs example: `kata-fc` with `podFixed: {memory: "120Mi", cpu: "250m"}` (illustrative).
- Unknown RuntimeClass or unrunnable handler → pod goes to phase **`Failed`** (event explains). No `runtimeClassName` → default handler.
**ValidatingAdmissionPolicy** (`reference/access-authn-authz/validating-admission-policy.md`, examples fetched from
`kubernetes/website/content/en/examples/validatingadmissionpolicy/`):
- **Stable since v1.30**; `apiVersion: admissionregistration.k8s.io/v1`, kinds `ValidatingAdmissionPolicy` +
  `ValidatingAdmissionPolicyBinding` (both needed for any effect). Policy spec: `failurePolicy` (`Fail` default | `Ignore`),
  `matchConstraints.resourceRules[{apiGroups, apiVersions, operations, resources}]`, `matchConditions[{name, expression}]`,
  `variables[{name, expression}]` (use as `variables.<name>`), `validations[{expression, message, messageExpression, reason}]`
  (`reason` ∈ `Unauthorized|Forbidden|Invalid|RequestEntityTooLarge`), `auditAnnotations`, optional `paramKind`.
- Binding spec: `policyName`, **`validationActions`** (one or more of `Deny`, `Warn`, `Audit`; `Deny`+`Warn` together not allowed),
  `matchResources.namespaceSelector`/`objectSelector`, optional `paramRef` (its `parameterNotFoundAction` is required).
- CEL variables: `object` (null on DELETE), `oldObject` (null on CREATE), `request`, `params`, `namespaceObject`, `authorizer`.
  Safe navigation: `object.?metadata.labels['k'].orValue('')`, `has(object.spec.x)`. Names ending `.static.k8s.io` are reserved.
- Match-condition errors: any false → policy skipped; else `Fail` rejects, `Ignore` skips.
- A validated sample (policy requiring `runtimeClassName == 'gvisor'` for `pods` CREATE/UPDATE + a `[Deny]` binding on
  `namespaceSelector kubernetes.io/metadata.name: sandbox`) passed `kubernetes-validate --strict -k 1.34.0` this session.
**NetworkPolicy** (`concepts/services-networking/network-policies.md`, example `network-policy-default-deny-egress.yaml`):
- Needs a network plugin that enforces it. Default deny egress: `apiVersion: networking.k8s.io/v1, kind: NetworkPolicy, spec:
  {podSelector: {}, policyTypes: [Egress]}`. "A default deny-all egress policy also blocks DNS traffic" (add an explicit DNS allow if needed).
- `policyTypes` omitted → `Ingress` always set, `Egress` only if egress rules exist.
- `to` entries: `podSelector` (same namespace), `namespaceSelector`, both in ONE entry = AND (pods in those namespaces), separate
  entries = OR; `ipBlock {cidr, except}` "should be cluster-external IPs"; `ports[{protocol, port, endPort}]`. Namespace by name via the
  immutable label `kubernetes.io/metadata.name`.
- Exceptions: "traffic to and from the node where a Pod is running is always allowed"; a pod cannot block access to itself; cannot block
  localhost; no logging of blocked connections; no explicit deny rules; no FQDN/service-name targeting.
- **Startup race**: a pod created before the plugin handles a new policy "may be started unprotected"; once handled, new pods are
  isolated before they start; allow rules may arrive after isolation (pod may briefly have no connectivity).
**Job** (`concepts/workloads/controllers/job.md`): `.spec.activeDeadlineSeconds` bounds the whole Job, kills running Pods, status
`type: Failed, reason: DeadlineExceeded`, **takes precedence over `backoffLimit`** (both Job spec and Pod spec have an
`activeDeadlineSeconds` — set the right level). `.spec.backoffLimit` **defaults to 6**; retry back-off 10 s, 20 s, 40 s … capped at 6 min.
`.spec.ttlSecondsAfterFinished`: deletes Job and Pods (cascading) N s after Complete/Failed; `0` = immediately; unset = never.
`podFailurePolicy.rules[].action` ∈ `FailJob|Ignore|Count|FailIndex`, `onExitCodes {containerName, operator: In|NotIn, values}` or
`onPodConditions [{type: DisruptionTarget}]`; requires `restartPolicy: Never`.
**securityContext** (`tasks/configure-pod-container/security-context.md`): `allowPrivilegeEscalation` controls `no_new_privs` (inverted);
**defaults to true when unset**; cannot be false with `privileged` or `CAP_SYS_ADMIN`. `readOnlyRootFilesystem`. `seccompProfile.type`
`RuntimeDefault|Unconfined|Localhost` (+`localhostProfile` only for Localhost). `appArmorProfile.type` `RuntimeDefault` (default)|
`Unconfined`|`Localhost`; **explicitly setting `RuntimeDefault` makes the Pod unadmittable on nodes without AppArmor**, while leaving it
unset applies it only where AppArmor is enabled. Container fields override pod-level ones.
**Quota, limits, storage, identity**:
- ResourceQuota (`concepts/policy/resource-quotas.md`): `requests.cpu`, `limits.memory`, `requests.ephemeral-storage`,
  `limits.ephemeral-storage`, `pods`, `count/jobs.batch`, …; when a quota covers cpu/memory, **every Pod must specify them or is
  rejected** (use LimitRange defaults). Scopes `Terminating` (pod `.spec.activeDeadlineSeconds >= 0`) / `NotTerminating` refer to the
  **Pod's** field, not the Job's.
- LimitRange (`limit-range.md`): applies `default`/`defaultRequest` at admission and validates min/max; "validations occur only at Pod
  admission stage, not on running Pods".
- PIDs (`pid-limiting.md`): **no per-Pod pids field in the Pod spec**; set kubelet `--pod-max-pids` / `PodPidsLimit` (`podPidsLimit` in
  KubeletConfiguration). GKE TF: `node_config.kubelet_config.pod_pids_limit` (schema, provider 8.4.0). kind: a `KubeletConfiguration`
  kubeadm patch (kind reads it from the first node only — `$REF/kind/site/content/docs/user/configuration.md`).
- emptyDir (`$REF/k8s-volumes.md`): `sizeLimit` caps the volume on node ephemeral storage (can run out earlier if the node disk fills);
  `medium: "Memory"` = tmpfs, **counts against the container memory limit**; unsized memory-backed volumes = node allocatable memory.
- `automountServiceAccountToken: false` on the ServiceAccount or Pod (pod wins) (`tasks/configure-pod-container/configure-service-account.md`);
  security checklist: restrict metadata API `169.254.169.254` (`concepts/security/security-checklist.md`).
- `kubectl exec` needs RBAC `create` on subresource `pods/exec` (`resourceNames` allowed for subresources) (`reference/access-authn-authz/rbac.md`).
  Ephemeral containers are added via the `pods/ephemeralcontainers` subresource, cannot set `resources`/ports/probes, never restart;
  PSS restricted also covers `spec.ephemeralContainers[*]`.

## 4. kind (`$REF/kind`, `$REF/kindnetd-main.go`)
- Default CNI kindnetd (ptp/host-local + netlink routes + masquerade). **Current kindnetd enforces standard NetworkPolicy** via
  `sigs.k8s.io/kube-network-policies` (nftables table `kindnet-network-policies`, NFQUEUE 101, **`FailOpen: true`**)
  (`images/kindnetd/cmd/kindnetd/main.go` lines ~217–265). First kind release with it: **v0.24.0 (unverified — release notes not read)**;
  the repo's pinned v0.33.0 is newer. kind's docs still describe `networking.disableDefaultCNI: true` for Calico etc.
  ("power user feature with limited support").
- kind main is `0.34.0-alpha`; default node image `kindest/node:v1.37.0` (`pkg/apis/config/defaults/image.go`); the repo pins
  kind **v0.33.0** + `kindest/node:v1.34.11@sha256:44e2…` (03 `deploy/versions.env`) — reuse that.
- kind docs have no RuntimeClass/gVisor/Kata guidance. **gVisor in kind: not supported by kind (unverified whether a manual runsc install
  inside the node container works; treat as unavailable, as SPEC says)**. Minikube's gvisor addon is the documented local alternative.
- `kubeadmConfigPatches` supports `KubeletConfiguration` (for `podPidsLimit`), `containerdConfigPatches` exists
  (`private-registries.md`).

## 5. Firecracker (`$REF/firecracker`)
- `SPECIFICATION.md` (enforced by CI on m5d.metal/m6g.metal): VMM starts to API socket in **8 CPU ms** (wall 6–60 ms, typical **12 ms**);
  VMM memory overhead **≤ 5 MiB** for a 1 vCPU / 128 MiB microVM; **≤ 125 ms** from `InstanceStart` to guest `/sbin/init` (serial console
  off, minimal kernel/rootfs); guest CPU > 95 % of bare metal (test pending); network up to 14.5 Gbps (≤80 % of a core), +0.06 ms latency.
- Defaults (`README.md`): 1 vCPU, **128 MiB**; built-in: demand-fault paging and CPU oversubscription on by default; per-thread seccomp
  filters (`docs/seccomp.md`: VMM, API, vCPU threads; none on debug builds); **Jailer** (`docs/jailer.md`: `jailer --id <id> --exec-file
  <fc> --uid --gid [--cgroup-version] [--chroot-base-dir] [--netns] [--resource-limit] [--daemonize] [--new-pid-ns]`, applies
  cgroup/namespace isolation then drops privileges; id ≤ 64 chars).
- Needs **KVM** (`/dev/kvm` read/write; AWS `.metal` only for EC2) (`docs/getting-started.md`) → not on Colab/most laptops' Docker.
- Snapshots (`docs/snapshotting/snapshot-support.md`): memory file mapped `MAP_PRIVATE` → on-demand page loading ("very fast snapshot
  loading"); must keep the memory file for the VM's lifetime; network/vsock connections may drop; **resuming the same snapshot more than
  once is considered insecure** (duplicated RNG state, identifiers, tokens) unless VMGenID (Linux ≥ 5.18) etc. is handled.
- Threat model: guest vCPU threads least trusted, nested trust zones (`docs/design.md` §Threat Containment).

## 6. Kata Containers (`$REF/kata/docs`, `$REF/kata-runtimeclasses.yaml`)
- Each pod = its own lightweight VM with a dedicated guest kernel; `kata-agent` inside the guest; host files shared via virtio-fs
  (`quick-start-guide.md`). "Isolation is not multi-tenancy on its own" (network, storage, control plane too).
- Hypervisors (`hypervisors.md`): QEMU (C; GPU, TDX, SEV-SNP), Cloud Hypervisor, Firecracker, Dragonball (in-shim), StratoVirt (Rust).
  Kata+Firecracker needs containerd's `devmapper` snapshotter (`how-to/how-to-use-kata-containers-with-firecracker.md`).
- Current: **4.x**, 4.2.0 is the newest release note (`releases/4.2.0.md`); `runtime-rs` (Rust) is default since 4.0.0; Go runtime
  deprecated, still selectable (`migrating-config-go-runtime-to-runtime-rs.md`). Install: `kata-deploy` Helm chart
  `oci://ghcr.io/kata-containers/kata-deploy-charts/kata-deploy` (one RuntimeClass per shim); needs `/dev/kvm`, K8s ≥ 1.22, containerd
  ≥ v2.1.x recommended (`installation.md`).
- **RuntimeClass names**: runtime-rs → `-runtime-rs` suffix, e.g. **`kata-qemu-runtime-rs`**, `kata-clh-runtime-rs`; `kata-dragonball`
  (runtime-rs only); Go: `kata-qemu`, `kata-clh`, **`kata-fc`** (listing in `helm-configuration.md` ~line 1362). So SPEC's "kata-qemu,
  kata-fc, kata-clh" are the Go-runtime names; prefer `kata-qemu-runtime-rs` in new manifests.
- **Default `overhead.podFixed`** set by kata-deploy (`templates/runtimeclasses.yaml`): fc / clh / clh-runtime-rs / dragonball
  **130Mi, 250m**; qemu / qemu-runtime-rs **320Mi, 250m** (raised from 160Mi because the QEMU VMM was OOM-killed in small pods);
  qemu-snp/tdx 2048Mi, 1.0; qemu-nvidia-gpu 10240Mi, 1.0.
- Kata boot/cold-start ms: not in these docs **(unverified; commonly quoted ~0.5–2 s for QEMU-based pods)**.

## 7. GKE Sandbox, Terraform, Cloud Run, Agent Sandbox
- **Terraform (google 8.4.0)**: `google_container_node_pool.node_config.sandbox_config.type` (required) — provider code
  `validation.StringInSlice([]string{"GVISOR"}, false)` (**case-sensitive; must be `"GVISOR"`**; `"gvisor"` fails `terraform validate`)
  (`$REF/tf-node_config.go` line 768, identical at tag v8.4.0). `sandbox_config` is `ForceNew`. Deprecated beta field
  `sandbox_type = "gvisor"` (lowercase) — do not use. Docs: "you must specify `image_type = "COS_CONTAINERD"` and `node_version =
  "1.12.7-gke.17"` or later" (`container_cluster.html.markdown` line 1266). SPEC's `type = "gvisor"` is WRONG for Terraform (gcloud's
  `--sandbox type=gvisor` is lowercase).
- Other verified attrs: `node_config.taint{key,value,effect}` (effect enum `NO_SCHEDULE`), `node_config.spot`, `node_config.image_type`,
  `node_config.workload_metadata_config.mode` (`GKE_METADATA`), `node_config.kubelet_config.pod_pids_limit`, `autoscaling.{min,max}_node_count`
  / `total_{min,max}_node_count`; cluster: `datapath_provider` (`LEGACY_DATAPATH` default | `ADVANCED_DATAPATH` = Dataplane V2, enforces
  NetworkPolicy), `network_policy{enabled, provider}` + `addons_config.network_policy_config.disabled` (Calico path),
  `private_cluster_config.enable_private_nodes`, `node_pool.network_config.enable_private_nodes`, `workload_identity_config.workload_pool`,
  `remove_default_node_pool`, `deletion_protection`, `enable_fqdn_network_policy`, `enable_cilium_clusterwide_network_policy`,
  **`addons_config.agent_sandbox_config.enabled`** ("Agent Sandbox addon") (schema via `tfattrs.py`).
- No egress by default = private nodes (no external IPs) **and no Cloud NAT** (no router/NAT resources) — design choice; images must come
  from Artifact Registry via Private Google Access on the subnet (`google_compute_subnetwork.private_ip_google_access`) **(verify PGA
  covers `*.pkg.dev`)**.
- GKE Sandbox specifics **(unverified — GKE docs site blocked)**: GKE auto-creates RuntimeClass `gvisor` (handler likely `gvisor`, not
  `runsc`) with `scheduling` nodeSelector `sandbox.gke.io/runtime: gvisor` + toleration so pods need only `runtimeClassName: gvisor`;
  sandbox nodes tainted `sandbox.gke.io/runtime=gvisor:NoSchedule`; needs ≥ 2 node pools (the default/system pool can't be sandboxed);
  incompatible: privileged pods, hostPath, hostNetwork/hostPID, some seccomp/AppArmor/SELinux custom profiles, Windows, E2 shared-core
  (e2-micro/small/medium) machine types; GPUs supported on recent versions via nvproxy. Pricing: no gVisor surcharge; cluster
  management fee ~$0.10/h per cluster with a monthly free-tier credit for one zonal/Autopilot cluster **(verify)**. Autopilot accepts
  `runtimeClassName: gvisor` (verified via gVisor tutorial, §2).
- **Cloud Run jobs**: `google_cloud_run_v2_job.template.template.execution_environment` ∈ `EXECUTION_ENVIRONMENT_GEN1` |
  `EXECUTION_ENVIRONMENT_GEN2`; `template.template.timeout` (per attempt), `template.template.max_retries` (**default 3** — set 0 for
  non-idempotent code), `template.task_count`, `template.template.vpc_access.egress` ∈ `ALL_TRAFFIC|PRIVATE_RANGES_ONLY` (schema).
  gen1 = gVisor, gen2 = microVM-based full Linux **(unverified; docs blocked)**; Cloud Run egress to the internet is **on by default**
  (restricting needs Direct VPC egress `ALL_TRAFFIC` + firewall) **(verify)**.
- **kubernetes-sigs/agent-sandbox** (`$REF/agent-sandbox`, SIG Apps): CRDs `agents.x-k8s.io/v1beta1` `Sandbox`; extensions
  `extensions.agents.x-k8s.io/v1beta1` `SandboxTemplate`, `SandboxClaim` (`spec.warmPoolRef.name`, lifecycle `Delete|DeleteForeground|
  Retain`), `SandboxWarmPool` (`spec.replicas`, `spec.sandboxTemplateRef.name`, `updateStrategy.type: Recreate`). "delegates low-level
  container isolation to … gVisor or Kata … via `RuntimeClass`". Install pin example **`v1.0.2`**: `kubectl apply -f
  https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${VERSION}/sandbox-with-extensions.yaml`; Python SDK
  `pip install k8s-agent-sandbox`. `SandboxTemplate.networkPolicyManagement` `Managed` (default; controller ensures a shared
  NetworkPolicy) | `Unmanaged`; template `networkPolicy.ingress/egress` empty list ⇒ **default deny** (`extensions/api/v1beta1/
  sandboxtemplate_types.go`). Measured (`docs/performance-tuning.md`, GKE k8s 1.36.2, 20× e2-standard-16, 75-claim burst from a
  150-sandbox pool): pod **cold start ~42–50 s p50** on that cluster; warm adoption "sub-second when the pool is healthy". Examples use
  `runtimeClassName: gvisor` and `hostAliases` to avoid DNS. GKE's addon above is presumably this project managed by GKE **(verify)**.

## 8. Managed sandbox services
- **E2B** (`$REF/e2b`): "open-source infrastructure that allows you to run AI-generated code in secure isolated sandboxes in the cloud";
  Python `pip install e2b` → `from e2b import Sandbox; with Sandbox.create() as sandbox: sandbox.commands.run(...)`; code interpreter
  `pip install e2b-code-interpreter` → `sandbox.run_code(...)`; desktop/computer-use `e2b-desktop`; needs `E2B_API_KEY`. SDK:
  `default_sandbox_timeout = 300` s; max lifetime 24 h (Pro) / 1 h (Hobby); **egress open by default** ("If `allow_out` is not specified,
  all outbound traffic is allowed"); `allow_internet_access=False` ≡ `deny_out: ["0.0.0.0/0"]`; `network.rules[host][].transform`
  injects headers (e.g. `Authorization`) **on the host side, so the sandbox never sees the credential** — the same pattern as
  `sandboxcore.proxy` (`packages/python-sdk/e2b/sandbox/sandbox_api.py`, `sandbox/main.py`, `sandbox_sync/main.py`). Isolation
  technology (Firecracker microVMs) and cold-start claims are not in this repo **(unverified; E2B markets ~150 ms starts)**.
- Modal (gVisor-based sandboxes), Daytona, Vertex AI Agent Engine code execution / "Agent Sandbox": **(unverified — no source in ref;
  docs sites blocked)**. Put them in the §9 table as verify-marked rows with no numbers.

## 9. Docker hardening (`$REF/docker/docker-cli-container_run.md`, `none.md`, `content_manuals_engine_*`)
- `--network none`: "only the loopback device is created" (no IPv6 loopback). `--read-only`: root fs read-only; writes only to
  volumes/tmpfs. `--tmpfs /work:rw,noexec,nosuid,size=65536k` (options = `mount -t tmpfs -o`). `--cap-drop ALL` / `--cap-add`.
  `--security-opt no-new-privileges` (also `=true`): "commands that raise privileges such as `su` or `sudo` no longer work";
  `--security-opt seccomp=profile.json` | `seccomp=unconfined` | `seccomp=builtin`; `--security-opt apparmor=PROFILE`.
  `--pids-limit` (int64, default 0; `-1` unlimited). `-m/--memory` (default 0 = none). `--memory-swap` = memory+swap total; set equal to
  `--memory` ⇒ no swap; unset ⇒ may use swap equal to `--memory`; `-1` unlimited. `--cpus` (decimal). `-u/--user <uid>[:<gid>]`.
  `--runtime` (e.g. `runsc`). `--init` (reaps zombies, forwards signals). `--rm`. `--stop-timeout`. `--ulimit cpu=…,fsize=…,nofile=…`
  (soft[:hard]; `as` deprecated); **`--ulimit nproc` is per user, not per container** (use `--pids-limit`).
- Default seccomp profile (`seccomp.md`; `https://github.com/moby/profiles/blob/main/seccomp/default.json`, fetched to
  `$REF/docker/moby-profiles-seccomp-default.json`): an allowlist, `defaultAction: SCMP_ACT_ERRNO` (`defaultErrnoRet: 1`, EPERM),
  "disables around 44 system calls out of 300+". Blocked include `add_key`, `bpf`, `clone`/`unshare` (new namespaces; user-ns
  exceptions), `io_uring_setup/enter/register`, `kexec_load`, `keyctl`, `mount`, `umount2`, `open_by_handle_at`, `perf_event_open`,
  `pivot_root`, `ptrace` (kernel < 4.8), `reboot`, `setns`, `socket(AF_ALG)`, `swapon/off`, `userfaultfd`, module loading.
- User namespaces remapping exists but "not enabled by default"; rootless mode documented (`content_manuals_engine_security__index.md`).

## 10. Python process sandbox (T0) — CPython 3.11 docs + experiments run here (Linux 6.18, uid 0, 4 vCPU, 2026-09-26)
- `resource.setrlimit(res, (soft, hard))`. Measured behaviour with `subprocess.run(..., preexec_fn=...)`:
  - `RLIMIT_CPU` soft=hard=1 → child killed **SIGKILL** (`returncode -9`) after ~1.0 s CPU; soft 1 / hard 2 → **SIGXCPU**
    (`returncode -24`). Docs: SIGXCPU at the soft limit. CPU seconds ≠ wall time: a `sleep` loop never hits it → always add a wall timeout.
  - `RLIMIT_AS` 256 MiB, `bytearray(600 MiB)` → Python `MemoryError`, rc 1. AS counts virtual address space, not RSS: with AS 64/128 MiB
    plain `python -c pass` and `import json,socket,http.server` start, but **`import numpy` fails ("OpenBLAS error: Memory allocation still
    failed")**; 256 MiB works here. Set `OPENBLAS_NUM_THREADS=1` in the clean env and document the AS floor.
  - `RLIMIT_FSIZE` 1 MiB: a **Python** child gets `OSError: [Errno 27] File too large` (CPython ignores SIGXFSZ at startup); a non-Python
    child (`sh`) is killed by SIGXFSZ (shell rc 153 = 128+25). `restore_signals=True` (default) resets SIGPIPE/SIGXFZ/SIGXFSZ before exec.
  - `RLIMIT_NOFILE` 32 → `OSError: [Errno 24] Too many open files`.
  - **`RLIMIT_NPROC` is not enforced for root**: as uid 0 a limit of 10 still forked 50. As uid 65534 (`preexec_fn` setrlimit then
    setgid/setuid, or `subprocess.run(user=65534, group=65534, extra_groups=[])`) the 10th fork fails `EAGAIN`; a second run right
    after forked **0** because the first run's 9 sleeping children of the same UID still counted → the limit is **per real UID, system
    wide** (CPython docs misleadingly say "processes the current process may create"). Tests must run the fork-bomb probe as a
    non-root UID or assert the documented "not enforced as root" verdict; Colab runs as root.
- `subprocess`: `preexec_fn` "is NOT SAFE to use in the presence of threads … could deadlock before exec" (`py-subprocess.rst`);
  `start_new_session=True` → `setsid()`; `process_group=0` (3.11+) → `setpgid(0, 0)`. `subprocess.run(timeout=…)` on POSIX kills only the
  direct child (`process.kill()` then `wait()`): measured, a grandchild (`sh -c "sleep 7.77 & sleep 60"`) **survived re-parented to PID 1**.
  Fix: `Popen(..., start_new_session=True)`, on timeout `os.killpg(p.pid, signal.SIGKILL)` then `communicate()` (measured 1.0 s, rc -9).
- Alternatives to `preexec_fn`: a trampoline (`[sys.executable, "-I", "-c", <setrlimit+os.execv code>, …]`) or `resource.prlimit(pid, …)`
  (Linux only; races with the child's first instructions). `python -I` ignores `PYTHON*` env and user site-packages.
- Network: rlimits cannot stop sockets. Here `unshare -Urn` (unprivileged user+net namespace) worked as uid 65534: `connect_ex(("1.1.1.1",
  80))` → 101 (ENETUNREACH). Availability varies (Ubuntu 24.04 AppArmor userns restriction, Docker's seccomp, macOS none) → detect at
  run time, label the verdict, never assume.
- macOS **(unverified, no macOS here)**: `RLIMIT_AS` accepted but not enforced (or `setrlimit` raises `ValueError`), `RLIMIT_NPROC`
  per-user and low defaults, no `/proc`, no namespaces, SIGKILL semantics same; the core must print "not enforced on this OS".
- Measured spawn latency (30 runs, p50 / p95 ms): `/bin/true` 1.4 / 1.6; `python -c pass` 15.6 / 22.4; `python -I -S -c pass`
  10.6 / 12.8; `python -I` + `env={}` + `start_new_session` + one rlimit 15.8 / 18.0. Label as "measured on the build host".

## 11. Cold start by isolation level (for PRIMER §6; label each row's provenance)
| Level | Start cost | Source |
|---|---|---|
| fork/exec (`/bin/true`) | ~1.4 ms p50 | measured here (§10) |
| Python interpreter per execution | ~11–16 ms p50 | measured here (§10) |
| container (runc, `docker run`) | 100–500 ms | **(verify)**; gVisor docs: Docker itself dominates start time |
| gVisor (`runsc`) container | runc + tens–hundreds of ms | **(verify)** — docs show a graph, no number |
| Firecracker microVM | VMM ~12 ms typical + ≤ 125 ms to `/sbin/init` | `SPECIFICATION.md` |
| Kata pod | ~0.5–2 s | **(unverified)** |
| full VM (GCE) | 10 s+ (tens of s) | **(verify)** |
| GKE pod on a busy cluster, burst | ~42–50 s p50 | agent-sandbox `docs/performance-tuning.md` |
| warm pool adoption / `kubectl exec` | sub-second | agent-sandbox same file; exec latency (verify) |
| GPU replica (for contrast) | 380 s cold / 70 s warm | 03 PRIMER §8 `startup_latency()` |

## 12. Pool and cost arithmetic (worked; the core's `pool.py` must reproduce these in tests)
- Little's law: busy sandboxes = λ × t_exec. Replace-after-use pools also hold warming sandboxes = λ × t_cold. λ = 5/s, t_exec = 2 s,
  t_cold = 3 s → 10 busy + 15 warming = **25** sandboxes to never wait on a cold start.
- M/M/c (Erlang C), offered load a = λ·t = 10: c = 11 → P(wait) 0.682, E[Wq] 1.364 s; c = 12 → 0.449, 0.449 s; c = 13 → 0.285, 0.190 s;
  c = 14 → 0.174, 0.087 s; c = 16 → 0.057, 0.019 s. Formula: `C(a,c) = [a^c/c! · c/(c−a)] / [Σ_{k<c} a^k/k! + a^c/c! · c/(c−a)]`,
  `E[Wq] = C/(c·μ − λ)`, μ = 1/t_exec.
- Link to the scaling primer: 27.1 tool calls/s at peak (§3.2); if 20 % are `run_code`, λ ≈ **5.4/s** (27.1 × 0.2 = 5.42).
- Cost per execution = (sandbox-seconds held × $/s of the node share) + per-execution fixed cost; any $ figure is **(verify)** and must
  cite `COMPUTE.md` (which has no CPU-only prices today — add none without a verify mark).

## 13. OWASP Top 10 for Agentic Applications (2026) — from the repo (`identity primer §2`, `docs/sources.md` line 79–82)
Published 9 Dec 2025 (sources.md). ASI01 Agent Goal Hijack · ASI02 Tool Misuse & Exploitation · ASI03 Identity & Privilege Abuse ·
ASI04 Agentic Supply Chain · ASI05 Unexpected Code Execution · ASI06 Memory & Context Poisoning · ASI07 Insecure Inter-Agent
Communication · ASI08 Cascading Failures · ASI09 Human-Agent Trust Exploitation · ASI10 Rogue Agents. No upstream copy in `$REF` (the
clone failed); cite the identity primer, keep the primer's wording, and keep "OWASP wording" on the Verify list.

## 14. Model and tool ids to use
- T0 "LLM": a scripted model in the style of `agentcore.FakeLLM` (reimplemented inside `sandboxcore.agent`, no import) — no weights.
- T0 runtime: Python 3.11 stdlib only in `sandboxcore` (`subprocess`, `resource`, `signal`, `os`, `tempfile`, `socket`, `http.server`
  `ThreadingHTTPServer`, `urllib.request`, `json`, `hashlib`, `dataclasses`); `pyyaml` only in the lab (render), like `k8sgpu`.
- Sandbox image (T1 Docker/kind/GKE): **`python:3.12-slim`** (verify tag digest) for `run_code`; the 03 lab's `busybox:1.38.0` for trivial
  probes; the proxy as `python:3.12-slim` running `sandboxlab/proxy`. Run as UID/GID **65534** (numeric, so kubelet can verify
  `runAsNonRoot`).
- kind **v0.33.0**, node `kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d`
  (03 `versions.env`); validation `kubernetes-validate --strict -k 1.34.0`.
- gVisor: apt `runsc` "release" channel (latest; `runsc --version` to record) or tarball `gvisor.tar.zstd`; Docker runtime name `runsc`;
  K8s RuntimeClass `gvisor` → handler `runsc` (self-managed) / GKE-managed `gvisor`.
- GKE (T3): zonal Standard cluster; system pool `e2-standard-2` (verify cheapest); sandbox pool `e2-standard-2` Spot (the gVisor
  tutorial's example type), `image_type = "COS_CONTAINERD"`, `sandbox_config { type = "GVISOR" }`, autoscaling 0→N, taint as in §7;
  Terraform ≥ 1.9, google ≥ 8.0 (validated offline against 8.4.0). Optional: agent-sandbox **v1.0.2**, Kata `kata-qemu-runtime-rs`
  (needs nested virt/KVM — not on GKE Sandbox pools).
- Optional real LLM (not needed): reuse the 04 serving lab's T1 model path (verify id there); never download weights at T0.

## 15. Pitfalls (builders get these wrong)
1. Terraform `sandbox_config.type` must be **`"GVISOR"`** (case-sensitive validator); SPEC's `"gvisor"` fails `terraform validate`.
2. **RLIMIT_NPROC does nothing for root** and counts every process of the real UID system-wide; run probes as UID 65534 or report
   "not enforced (root)". Colab and many CI containers are root.
3. `subprocess.run(timeout=)` leaves grandchildren alive; always `start_new_session=True` + `os.killpg(pgid, SIGKILL)`.
4. `preexec_fn` is unsafe with threads (the core's proxy/agent may be threaded) — prefer a trampoline or keep the executor single-threaded.
5. RLIMIT_CPU measures CPU seconds; sleeping/blocked code needs the wall-clock timeout. Soft = hard → SIGKILL, soft < hard → SIGXCPU first.
6. RLIMIT_AS is virtual memory: too low breaks `import numpy` (OpenBLAS); set `OPENBLAS_NUM_THREADS=1`; ≥ 256 MiB here.
7. A Python child reports RLIMIT_FSIZE as `OSError EFBIG`; a shell child dies of SIGXFSZ — map both to exit reason `file_too_large`.
8. rlimits never block the network; the process sandbox must say so and the probe verdict for egress is "LEAKED" unless a netns was used.
9. PSA `enforce` rejects **Pods**, not Jobs: a bad Job is accepted and silently never gets Pods — watch events / Job conditions.
10. `allowPrivilegeEscalation` defaults to **true** when unset; restricted requires explicit `false`. `seccompProfile` must be explicit.
11. `runAsNonRoot: true` with an image whose USER is a name fails at container creation; set numeric `runAsUser: 65534` **(kubelet
    behaviour, verify message)**.
12. Explicit `appArmorProfile: {type: RuntimeDefault}` makes pods unschedulable/unadmittable on nodes without AppArmor (kind on some
    hosts); restricted does not require the field — omit it in kind manifests.
13. Default-deny egress also kills DNS; allowing DNS re-opens a DNS-exfiltration channel. Prefer: sandbox reaches only the proxy
    (by ClusterIP/`hostAliases`), the proxy resolves names.
14. NetworkPolicy never blocks traffic to/from the pod's own node; on GKE the metadata server path is node-local (verify) — deny cloud
    credentials with `automountServiceAccountToken: false`, a KSA with no IAM binding, and `GKE_METADATA` mode, not with NetworkPolicy.
15. NetworkPolicy may lag pod start ("may be started unprotected"); apply policies before the first runner pod and gate with a readiness check.
16. kindnetd's policy dataplane is **FailOpen: true**; kind is a teaching cluster, not a security boundary. No gVisor in kind.
17. Job `backoffLimit` defaults to **6** and Cloud Run jobs `max_retries` to **3** → non-idempotent code re-runs; set 0 and use
    idempotency keys (scaling primer §5.4 recipe).
18. `ttlSecondsAfterFinished` deletes Pods and their logs — collect logs/results before the TTL; enforce output truncation inside the
    sandbox wrapper, not via kubelet log rotation.
19. `ResourceQuota` on cpu/memory rejects pods without requests/limits → ship a LimitRange with defaults. Quota scope `Terminating`
    keys on the **Pod's** `activeDeadlineSeconds`, not the Job's.
20. There is no per-pod PID limit field; use kubelet `podPidsLimit` (kind patch / TF `kubelet_config.pod_pids_limit`) or Docker `--pids-limit`.
21. `emptyDir.medium: Memory` counts against the container memory limit; always set `sizeLimit`.
22. VAP `resources: ["pods"]` does not match the `pods/ephemeralcontainers` or `pods/exec` subresources; list them if you mean to police them.
    Binding needs `validationActions`; `Deny` + `Warn` together is invalid.
23. HTTPS through a CONNECT proxy cannot get headers injected without TLS interception; the credential-injecting proxy must be the TLS
    client (sandbox → `http://proxy/…` plain HTTP inside the cluster/loopback, proxy → upstream HTTPS) or terminate TLS with a
    sandbox-trusted CA. `HTTP_PROXY` env vars are advisory; enforcement is the network (NetworkPolicy / `--network none` + proxy socket).
24. gVisor does not enforce cgroup limits inside the sandbox; limits come from the host cgroup (Docker/K8s resources). gVisor does not
    stop the code from using whatever the sandbox holds (credentials, mounted data, egress).
25. gVisor install instructions changed in 2026-07 (tarball with `gvisor-bin/`); don't copy old "download the runsc binary" snippets.
26. Kata/Firecracker need `/dev/kvm` (bare metal or nested virt); not available on Colab, GKE Sandbox pools, or Docker Desktop by default.
27. Firecracker snapshot restored more than once duplicates RNG/token state — "insecure" per upstream; warm pools of restored clones
    need re-seeding (VMGenID).
28. E2B's default network is open egress; "sandbox" ≠ "no network" in managed services — state the default per provider.
29. Kata RuntimeClass names in 4.x carry `-runtime-rs` (`kata-qemu-runtime-rs`); `kata-qemu/kata-clh/kata-fc` are the deprecated Go runtime.
30. Never fabricate latencies: T0 numbers are "measured on this machine"; Docker/kind/GKE numbers are sample output "(illustrative)".
