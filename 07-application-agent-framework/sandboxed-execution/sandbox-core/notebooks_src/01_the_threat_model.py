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
# environment secret, read a private key, phone home, fork forever, fill the disk, spin the CPU, print
# forever — and you run them through the **unsandboxed** executor and see what leaks. The invariant a
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
# Now run the whole suite through `ProcessSandbox` (notebook 02 builds it). The secrets are gone (clean
# environment, isolated home), the resource abuses are stopped by limits — and **egress is the one thing a
# process sandbox does not stop**, which it reports honestly. That single exception is why the network is a
# separate control (notebook 04, and NetworkPolicy in the lab).

# %%
sb = ProcessSandbox()
rep = sb.isolation_report()
print("this sandbox:", {k: rep[k] for k in ("uid_dropped", "clean_env", "isolate_home",
                                            "as_limit_enforced", "nproc_enforced", "network_blocked")})
print()
print(f"{'probe':16} {'exit':14} {'leaked':7} {'contained':9} detail")
for p in PROBES:
    v = run_probe(p, sb)
    print(f"{p.name:16} {v.exit_reason:14} {str(v.leaked):7} {str(v.contained):9} {v.detail[:44]}")

# %% [markdown]
# Read the table: seven of eight probes are contained; `egress_connect` is not, because resource limits do
# not touch sockets. That is not a bug in the sandbox — it is the boundary between the *process* layer and
# the *network* layer. Hold that thought for notebook 04.
#
# ## Exercise 1.1 — classify the blast radius
# For each probe, say which layer of control bounds it. Fill `control` with one of
# `"clean_env"`, `"isolated_home"`, `"cpu_limit"`, `"wall_timeout"`, `"file_limit"`, `"pid_limit"`,
# `"output_truncation"`, `"network_layer"`. (This is the mental map you carry into a review.)

# %% exercise
control = {name: None for name in ("read_env_secret", "read_ssh_key", "egress_connect", "fork_bomb",
                                   "disk_fill", "cpu_spin", "sleep_forever", "output_flood")}
### BEGIN SOLUTION
control = {
    "read_env_secret": "clean_env",
    "read_ssh_key": "isolated_home",
    "egress_connect": "network_layer",     # NOT the process sandbox — the honest answer
    "fork_bomb": "pid_limit",
    "disk_fill": "file_limit",
    "cpu_spin": "cpu_limit",
    "sleep_forever": "wall_timeout",
    "output_flood": "output_truncation",
}
### END SOLUTION

# %% check
assert control["egress_connect"] == "network_layer", "rlimits do not block sockets"
assert control["sleep_forever"] == "wall_timeout", "a sleeper uses no CPU; only the wall clock stops it"
assert control["cpu_spin"] == "cpu_limit" and control["fork_bomb"] == "pid_limit"
assert None not in control.values()
print("✅ each risk mapped to the control that bounds it — and egress is the network layer, not rlimits")

# %% [markdown]
# ## Exercise 1.2 — why is a wall-clock timeout separate from the CPU limit?
# The `cpu_spin` probe burns CPU; the `sleep_forever` probe uses none. Predict the exit reason a CPU-time
# limit alone would give each, then set `needs_wall_clock` to the probe name that a CPU limit cannot stop.

# %% exercise
### BEGIN SOLUTION
needs_wall_clock = "sleep_forever"
### END SOLUTION

# %% check
assert needs_wall_clock == "sleep_forever"
v_sleep = run_probe(next(p for p in PROBES if p.name == "sleep_forever"), ProcessSandbox())
assert v_sleep.exit_reason == "wall_timeout"
print("✅ RLIMIT_CPU measures CPU seconds; blocked or sleeping code needs a wall-clock kill too")

# %% [markdown]
# ## Exercise 1.3 — the honest verdict for egress
# A colleague says "our sandbox drops network packets, so egress is contained." You ran `egress_connect`
# above and saw it reach the trap. Write `egress_is_contained_by(process_sandbox)` returning the boolean the
# sandbox itself reports (from its isolation report), and confirm it is `False`.

# %% exercise
def egress_is_contained_by(sandbox):
    ### BEGIN SOLUTION
    return bool(sandbox.isolation_report()["network_blocked"])
    ### END SOLUTION

# %% check
assert egress_is_contained_by(ProcessSandbox()) is False
print("✅ a process sandbox does not block egress; say so in the review and add the network layer")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A code-execution tool takes untrusted, model-written programs and runs them,
# so I treat it as the highest-risk tool there is — DESTRUCTIVE tier, in the identity primer's language. The
# question I ask is 'if the model is fully hijacked, what can this code reach?' The answer must be: no
# ambient credentials, because the process starts from a clean environment; no private files, because home
# and the workspace are ephemeral; no network, because egress is denied by default and only an allowlisting
# proxy can reach out; and no runaway resource use, because CPU, memory, processes, file size and output are
# all bounded. I demonstrate this with a suite of harmless probes: unsandboxed they leak the token, read a
# key and phone home; sandboxed, seven of eight are contained and the eighth — egress — is contained by the
# network layer, not by the process limits. That last distinction is the one people get wrong."
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
