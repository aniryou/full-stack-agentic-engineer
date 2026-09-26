# %% [markdown]
# # 01 · The threat model: what untrusted code does when nothing stops it
#
# **Tier:** T0 — laptop / Colab CPU / CI, free, about a minute. Everything runs against harmless
# stand-ins (a fake credential in a temp file, a loopback listener the notebook owns), so you can watch
# an attack *work* without any real harm. Docker, gVisor and Kubernetes come in the lab (`../sandbox-lab`).
#
# ## The one-minute version
# An agent is *a workload that turns untrusted text into privileged actions* (the identity primer's one
# line). A `run_code` tool is the sharpest version of that: the model, which cannot tell instructions
# from data, emits a program, and something runs it. If it runs with the agent's environment, home
# directory and network, then a prompt injection is a shell on your infrastructure. This notebook makes
# that concrete: a small set of **probes** stands in for what a hijacked model might emit — read an
# environment secret, read a private key, phone home, fork forever, fill the disk, spin the CPU, hang,
# print forever, outlive the call — and you run them through the **unsandboxed** executor and see what leaks. The invariant a
# sandbox must restore is **no ambient authority**: no credentials, no network by default, no persistent
# filesystem. After this you can name the blast radius of a code tool and say which control bounds each
# risk. Primer: `../PRIMER.md` §1 (why a sandbox, the threat model). The OWASP agentic risks and the
# "execute code is DESTRUCTIVE-tier" rule are the identity primer's
# (`../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md` §2, §6.2) — cited, not restated.

# %%
from sandboxcore import PROBES, ProcessSandbox, UnsafeExecutor, run_probe

print("the probes, and the OWASP-for-agents risk each stands for:")
for p in PROBES:
    print(f"  {p.name:16} {p.owasp}")
    print(f"      {p.description}")

# %% [markdown]
# ## Worked example 1 — the ambient-authority leak, unsandboxed
# The `read_env_secret` probe reads `CLOUD_API_TOKEN`. The harness plants a fake token in the executor's
# environment. `UnsafeExecutor` runs the code in a child that **inherits this process's environment** — the
# default if you just `subprocess.run` model output. Watch the token come straight back.

# %%
ue = UnsafeExecutor()
env_probe = next(p for p in PROBES if p.name == "read_env_secret")
v = run_probe(env_probe, ue, secret="sk-live-PLANTED-BY-HARNESS")
print("unsandboxed:", v.detail, "| leaked:", v.leaked)
print("  (a real deployment's child would inherit AWS_*, GOOGLE_*, the metadata server's reach, ...)")

# %% [markdown]
# ## Worked example 2 — reaching the network, unsandboxed
# `egress_connect` opens a socket to a host and sends bytes. The harness starts a **loopback listener it
# owns** and points the probe at it; if the probe connects, that is the exfiltration channel working. With
# no isolation the connection succeeds.

# %%
egress = next(p for p in PROBES if p.name == "egress_connect")
v = run_probe(egress, ue)
print("unsandboxed:", v.detail, "| leaked:", v.leaked)

# %% [markdown]
# ## Worked example 3 — the same code, through a process sandbox
# Now run the whole suite through `ProcessSandbox` (notebook 02 builds it). Read its isolation report first:
# what it can enforce depends on this machine. The environment secret is gone (clean environment) and the
# resource abuses are stopped by limits. Three probes depend on **running as a different UID**, which the
# sandbox can only do when it runs as root (Colab, most CI): the key read (a different UID cannot open your
# 0700 home — pointing `HOME` at the workspace hides nothing from an absolute path), the fork bomb
# (`RLIMIT_NPROC` counts processes per UID) and the session escape (only a per-execution UID lets the sandbox
# find and kill a process that left its process group). And **egress is the one thing a process sandbox never
# stops**, which it reports honestly. That exception is why the network is a separate control (notebook 04,
# and NetworkPolicy in the lab).

# %%
sb = ProcessSandbox()
rep = sb.isolation_report()
print("this sandbox:", {k: rep[k] for k in ("uid_dropped", "filesystem_isolated", "clean_env",
                                            "nproc_enforced", "escapes_swept", "network_blocked")})
print()
print(f"{'probe':16} {'exit':14} {'leaked':7} {'contained':9} detail")
for p in PROBES:
    v = run_probe(p, sb)
    print(f"{p.name:16} {v.exit_reason:14} {str(v.leaked):7} {str(v.contained):9} {v.detail[:44]}")

# %% [markdown]
# Read the table: with a per-execution UID every probe but `egress_connect` is contained; without one (not
# root, or `drop_to_uid=None`) the key read, the fork bomb and the escape get through too. Egress is never
# contained here, because resource limits do not touch sockets. That is not a bug in the sandbox — it is the
# boundary between the *process* layer and the *network* layer. Hold that thought for notebook 04.
#
# ## Exercise 1.1 — classify the blast radius
# For each probe, say which control bounds it. Fill `control` with one of `"clean_env"`, `"separate_uid"`,
# `"cpu_limit"`, `"wall_timeout"`, `"file_limit"`, `"pid_limit"`, `"output_truncation"`, `"uid_sweep"`,
# `"network_layer"`. (This is the mental map you carry into a review. Two of them are not what they look
# like: think about what actually stops an absolute-path read, and what finds a process that left the group.)

# %% exercise
control = {name: None for name in ("read_env_secret", "read_ssh_key", "egress_connect", "fork_bomb",
                                   "disk_fill", "cpu_spin", "sleep_forever", "output_flood", "escape_session")}
### BEGIN SOLUTION
control = {
    "read_env_secret": "clean_env",
    "read_ssh_key": "separate_uid",        # HOME redirection is not a boundary; file permissions are
    "egress_connect": "network_layer",     # NOT the process sandbox — the honest answer
    "fork_bomb": "pid_limit",
    "disk_fill": "file_limit",
    "cpu_spin": "cpu_limit",
    "sleep_forever": "wall_timeout",
    "output_flood": "output_truncation",
    "escape_session": "uid_sweep",         # setsid() leaves the group; the per-execution UID is how it is found
}
### END SOLUTION

# %% check
import hashlib
assert None not in control.values(), "fill every probe"
_digest = hashlib.sha256(repr(sorted(control.items())).encode()).hexdigest()[:12]
assert _digest == "7caf9c0b4135", (
    "not quite: re-read worked example 3 — which control stops an absolute-path read, which one finds an "
    "escaped process, and which layer the egress probe belongs to?")
print("✅ each risk mapped to the control that bounds it — and egress is the network layer, not rlimits")

# %% [markdown]
# ## Exercise 1.2 — predict the exit reasons, then measure them
# The `cpu_spin` probe burns CPU; the `sleep_forever` probe uses none. Both run under the harness budget
# `cpu_s=1, wall_s=2`. Predict the exit reason (one of `contract.EXIT_REASONS`) each will end with; the check
# runs both and compares.

# %% exercise
predicted = {"cpu_spin": None, "sleep_forever": None}
### BEGIN SOLUTION
predicted = {"cpu_spin": "cpu_time", "sleep_forever": "wall_timeout"}
### END SOLUTION

# %% check
from sandboxcore import PROBES_BY_NAME
measured = {n: run_probe(PROBES_BY_NAME[n], ProcessSandbox()).exit_reason for n in predicted}
print("predicted:", predicted, "| measured on this machine:", measured)
assert predicted == measured, "compare with the measurement: which limit fires first for each?"
print("✅ RLIMIT_CPU measures CPU seconds; blocked or sleeping code needs a wall-clock kill too")

# %% [markdown]
# ## Exercise 1.3 — predict what gets through without a separate UID
# A laptop user runs the sandbox as themselves; that is `ProcessSandbox(SandboxConfig(drop_to_uid=None))`.
# Write `uncontained(sandbox)` returning the **set of probe names you expect NOT to be contained**, using
# only the sandbox's `isolation_report()` (not by running the probes). The check runs the suite and compares.
# Hint: which report fields decide the key read, the fork bomb, the escape and egress?

# %% exercise
from sandboxcore import SandboxConfig


def uncontained(sandbox):
    rep = sandbox.isolation_report()
    ### BEGIN SOLUTION
    out = set()
    if not rep["network_blocked"]:
        out.add("egress_connect")
    if not rep["filesystem_isolated"]:
        out.add("read_ssh_key")
    if not rep["nproc_enforced"]:
        out.add("fork_bomb")
    if not rep["escapes_swept"]:
        out.add("escape_session")
    return out
    ### END SOLUTION

# %% check
as_me = ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None))
observed = {p.name for p in PROBES if not run_probe(p, as_me).contained}
print("your prediction:", sorted(uncontained(as_me)), "| measured on this machine:", sorted(observed))
assert uncontained(as_me) == observed
assert uncontained(ProcessSandbox()) >= {"egress_connect"}
print("✅ the isolation report predicts the verdicts: HOME redirection hides nothing, egress is never blocked")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A code-execution tool takes untrusted, model-written programs and runs them,
# so I treat it as the highest-risk tool there is — DESTRUCTIVE tier, in the identity primer's language. The
# question I ask is 'if the model is fully hijacked, what can this code reach?' The answer must be: no
# ambient credentials, because the process starts from a clean environment; no private files, because the
# code runs as its own UID (pointing HOME somewhere else is not a boundary); nothing left behind, because
# the workspace is thrown away and leftovers of that UID are swept; no network, because egress is denied
# by default and only an allowlisting proxy can reach out; and no runaway resource use, because CPU,
# memory, processes, file size and output are all bounded. I demonstrate this with a suite of harmless
# probes: unsandboxed they leak the token, read a key, phone home and outlive the call; sandboxed with a
# per-execution UID, all but one are contained, and that one — egress — needs the network layer, not the
# process limits. That last distinction is the one people get wrong."
#
# **Drill questions**
# 1. *Why is 'execute code' automatically the most dangerous tool?* — It turns arbitrary model output into
#    arbitrary computation with whatever authority the process holds; a single injection becomes code
#    execution. The identity primer calls it DESTRUCTIVE-tier by definition.
# 2. *A sandbox enforces CPU, memory and PID limits. Is exfiltration contained?* — No. Resource limits never
#    touch the network; egress needs a separate control (deny-by-default + an allowlisting proxy, or a
#    network namespace / `--network none`).
# 3. *What is 'no ambient authority' in one line?* — The code inherits nothing: no credentials, no network,
#    no persistent filesystem — only what the caller deliberately hands it for this one execution.
# 4. *We set `HOME` to a temp dir, so the SSH key is safe?* — No. `HOME` only changes where `~` points; the
#    key still opens by its absolute path. Only a different UID (or a filesystem that does not contain it:
#    a mount namespace, a container) protects it.
