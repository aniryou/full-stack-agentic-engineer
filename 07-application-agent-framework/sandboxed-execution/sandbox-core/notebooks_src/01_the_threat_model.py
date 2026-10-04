# %% [markdown]
# # 01 · The threat model: what untrusted code does when nothing stops it
#
# **Tier:** T0: a laptop, a Colab CPU or CI, free, about a minute. Everything runs against harmless
# stand-ins (a fake credential in a temp file, a loopback listener that the notebook owns). Thus you can watch
# an attack *work* and cause no real harm. Docker, gVisor and Kubernetes come in the lab (`../sandbox-lab`).
#
# ## The one-minute version
# An agent is *a workload that turns untrusted text into privileged actions* (the one line of the identity primer).
# A `run_code` tool is the sharpest form of that. The model cannot tell instructions from data. It emits a program,
# and something runs that program. If the program runs with the environment, the home directory and the network of
# the agent, a prompt injection is a shell on your infrastructure.
#
# This notebook makes that concrete. A small set of **probes** stands in for the programs that a hijacked model can
# emit:
#
# - read an environment secret,
# - read a private key,
# - phone home,
# - fork forever,
# - fill the disk,
# - spin the CPU,
# - hang,
# - print forever,
# - outlive the call.
#
# You run the probes through the **unsandboxed** executor and see what leaks. The invariant that a sandbox must
# restore is **no ambient authority**: no credentials, no network by default, no persistent filesystem.
#
# After this notebook, you can name the blast radius of a code tool. You can also say which control bounds each risk.
# Primer: `../PRIMER.md` §1 (why a sandbox, the threat model). The OWASP agentic risks and the "execute code is
# DESTRUCTIVE-tier" rule come from the identity primer
# (`../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md` §2, §6.2). This notebook cites
# them and does not state them again.

# %%
from sandboxcore import PROBES, ProcessSandbox, UnsafeExecutor, run_probe

print("the probes, and the OWASP-for-agents risk each stands for:")
for p in PROBES:
    print(f"  {p.name:16} {p.owasp}")
    print(f"      {p.description}")

# %% [markdown]
# ## Worked example 1 — the ambient-authority leak, unsandboxed
# The `read_env_secret` probe reads `CLOUD_API_TOKEN`. The harness puts a fake token in the environment of the
# executor. `UnsafeExecutor` runs the code in a child that **inherits this process's environment**. That is the
# default when you only give the model output to `subprocess.run`. Watch the token come back directly.

# %%
ue = UnsafeExecutor()
env_probe = next(p for p in PROBES if p.name == "read_env_secret")
v = run_probe(env_probe, ue, secret="sk-live-PLANTED-BY-HARNESS")
print("unsandboxed:", v.detail, "| leaked:", v.leaked)
print("  (a real deployment's child would inherit AWS_*, GOOGLE_*, the metadata server's reach, ...)")

# %% [markdown]
# ## Worked example 2 — reaching the network, unsandboxed
# `egress_connect` opens a socket to a host and sends bytes. The harness starts a **loopback listener it
# owns** and points the probe at it. If the probe connects, the exfiltration channel works. With no isolation,
# the connection succeeds.

# %%
egress = next(p for p in PROBES if p.name == "egress_connect")
v = run_probe(egress, ue)
print("unsandboxed:", v.detail, "| leaked:", v.leaked)

# %% [markdown]
# ## Worked example 3 — the same code, through a process sandbox
# Now run the full suite through `ProcessSandbox` (notebook 02 builds it). First, read its isolation report: the
# controls that it can enforce depend on this machine. The environment secret is gone (clean environment). The
# limits stop the resource abuses.
#
# Three probes depend on **a different UID for the code**. The sandbox can do this only when it runs as root
# (Colab, most CI):
#
# - The key read. A different UID cannot open your 0700 home. When you point `HOME` at the workspace, that hides
#   nothing from an absolute path.
# - The fork bomb, when the sandbox runs as root. `RLIMIT_NPROC` counts tasks per UID and ignores root. When the
#   sandbox runs as a user, the parent counts the process tree of the run instead.
# - The session escape. Only a per-execution UID lets the sandbox find and kill a process that left its process
#   group.
#
# Also, **egress is the one thing a process sandbox never stops**, and the sandbox reports this honestly. That
# exception is the reason that the network is a separate control (notebook 04, and NetworkPolicy in the lab).

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
# Read the table. With a per-execution UID, the sandbox contains every probe but `egress_connect`. Without one (not
# root, or `drop_to_uid=None`), the key read and the escape also get through. As root, the fork bomb also gets
# through.
#
# Here, the sandbox never contains egress, because resource limits do not touch sockets. That is not a bug in the
# sandbox. It is the boundary between the *process* layer and the *network* layer. Keep that thought for
# notebook 04.
#
# ## Exercise 1.1 — classify the blast radius
# For each probe, say which control bounds it. Fill `control` with one of `"clean_env"`, `"separate_uid"`,
# `"cpu_limit"`, `"wall_timeout"`, `"file_limit"`, `"pid_limit"`, `"output_truncation"`, `"uid_sweep"`,
# `"network_layer"`. This is the mental map that you take into a review. Two of the answers are not what they look
# like. Think about what actually stops an absolute-path read. Also think about what finds a process that left the
# group.

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
# The `cpu_spin` probe uses CPU all the time. The `sleep_forever` probe uses no CPU. Both run under the harness
# budget `cpu_s=1, wall_s=2`. Predict the exit reason (one of `contract.EXIT_REASONS`) for each probe. The check
# runs both probes and compares.

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
# A laptop user runs the sandbox as themselves. That is `ProcessSandbox(SandboxConfig(drop_to_uid=None))`.
#
# Write `uncontained(sandbox)`. It returns the **set of probe names that you expect the sandbox NOT to contain**.
# Use only the `isolation_report()` of the sandbox. Do not run the probes. The check runs the suite and compares.
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
# **The two-minute version.** "A code-execution tool takes untrusted programs that the model writes, and runs
# them. Thus I treat it as the highest-risk tool there is: DESTRUCTIVE tier, in the language of the identity primer.
#
# "The question I ask is 'if the model is fully hijacked, what can this code reach?' The answer must be as follows.
# There are no ambient credentials, because the process starts from a clean environment. There are no private
# files, because the code runs as its own UID. (A HOME that points to a different directory is not a boundary.)
#
# "Nothing stays behind, because the sandbox discards the workspace and sweeps the leftovers of that UID. There is no
# network, because the policy denies egress by default and only a proxy with an allowlist can reach out. There is no
# runaway resource use, because CPU, memory, processes, file size and output all have limits.
#
# "I show this with a suite of harmless probes. Without a sandbox, they leak the token, read a key, phone home and
# outlive the call. In a sandbox with a per-execution UID, the sandbox contains all but one. That one is egress, and
# it needs the network layer, not the process limits. That last distinction is the one where people make errors."
#
# **Drill questions**
# 1. *Why is 'execute code' automatically the most dangerous tool?* It turns arbitrary model output into
#    arbitrary computation, with all the authority that the process holds. One injection becomes code execution.
#    The identity primer calls it DESTRUCTIVE-tier by definition.
# 2. *A sandbox enforces CPU, memory and PID limits. Is exfiltration contained?* No. Resource limits never
#    touch the network. Egress needs a separate control: deny-by-default and a proxy with an allowlist, or a
#    network namespace or `--network none`.
# 3. *What is 'no ambient authority' in one line?* The code inherits nothing: no credentials, no network,
#    no persistent filesystem. It gets only what the caller deliberately gives it for this one execution.
# 4. *We set `HOME` to a temp dir, so the SSH key is safe?* No. `HOME` only changes where `~` points. The
#    key still opens by its absolute path. Only a different UID protects it, or a filesystem that does not
#    contain it (a mount namespace, a container).
