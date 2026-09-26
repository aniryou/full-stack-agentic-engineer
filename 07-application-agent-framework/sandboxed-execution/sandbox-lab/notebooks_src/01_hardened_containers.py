# %% [markdown]
# # 01 · Hardened containers: run model-generated code without ambient authority
#
# **Tier:** T0 by default — a process sandbox runs on this machine and the attack probes are judged
# here; the Docker and gVisor rungs print the exact commands and read bundled sample verdicts
# labelled *sample output in the documented format (illustrative)*. With Docker (T0 + Docker) the
# same probes are measured against a real container.
#
# ## The one-minute version
#
# Model-generated code is untrusted input (PRIMER §1). The invariant is **no ambient authority**:
# no credentials, no network by default, no persistent filesystem — so that a hijacked model can
# do nothing outside the model that the sandbox was not asked to allow. You get there by climbing
# the isolation ladder (PRIMER §2) and turning on one control at a time:
#
# | Rung | What it adds | An attack it stops |
# |---|---|---|
# | unsandboxed | — | everything: it has the agent's env, files, UID, network |
# | process + budgets | CPU, memory, files, output, a wall timeout | fork bombs, miners, disk fill, output floods |
# | + a dedicated UID | the agent's files and other processes become unreadable | reading `~/.ssh`, scanning `/proc` |
# | + an empty network namespace | no route anywhere | exfiltration, the metadata server |
# | a hardened container | a kernel-enforced boundary: caps, seccomp, read-only root, cgroups | most escalation paths |
# | + gVisor (`runsc`) | the syscalls go to a user-space kernel, not the host's | kernel exploits |
#
# You run each attack through each rung and read the verdict — `CONTAINED` or `LEAKED` — instead of
# arguing about it. The controls live *outside the model*; that is the whole point.

# %%
from sandboxlab import env
from sandboxlab.process import Budgets, ProcessSandbox, Unsandboxed
from sandboxlab.probes import (LEAKED, PROBES, QUICK, run_suite, standin_host, unsandboxed_for,
                               verdict_table)

caps = env.capabilities()
print(caps.describe())
print("measurable here:", ", ".join(caps.levels()))

# %% [markdown]
# ## Worked example: what an unsandboxed executor hands to a hijacked model
#
# `standin_host()` builds a throwaway "agent host": a 0700 home with a **stand-in** SSH key, a
# stand-in agent process whose environment holds an API-key *canary*, and a listener that plays the
# attacker's server. Everything secret is a canary; every resource grab is bounded. The naive
# executor is `subprocess.run([python, "-c", code])` in the agent's own environment — the thing
# people ship by accident.

# %%
BUDGETS = Budgets(cpu_s=1, wall_s=2)
with standin_host() as host:
    naive = run_suite(unsandboxed_for(host), host, QUICK, budgets=BUDGETS)
for r in naive:
    print(f"  {r.probe:<16} {r.verdict:<10} {r.evidence[:70]}")
print("LEAKED:", sum(r.verdict == LEAKED for r in naive), "of", len(naive))

# %% [markdown]
# ## Worked example: the process sandbox, and what it still cannot stop
#
# `ProcessSandbox` gives the code a clean environment, a throwaway workspace, resource budgets, and
# — when this notebook runs as root (Colab does) — a dedicated UID and an empty network namespace.
# Compare the columns: the budgets and the UID close most of the gaps, but a process sandbox is not
# a kernel boundary, and without a network namespace it does nothing to the network.

# %%
with standin_host() as host:
    cols = {"none": run_suite(unsandboxed_for(host), host, QUICK, budgets=BUDGETS),
            "process(own uid)": run_suite(ProcessSandbox(BUDGETS, drop_uid=False, netns=False), host, QUICK),
            ProcessSandbox(BUDGETS).name: run_suite(ProcessSandbox(BUDGETS), host, QUICK)}
print(verdict_table(cols))
print("\n" + ProcessSandbox(BUDGETS).describe())

# %% [markdown]
# ## Exercise 1.1 — which budget catches which abuse
#
# Each probe below abuses one resource. Return the `exit_reason` the process sandbox reports for
# each — the vocabulary of the execution contract (PRIMER §3). Run the probe's code through a
# sandbox and read `result.exit_reason`. (`infinite_loop` burns CPU; `sleep_forever` blocks;
# `memory_hog` allocates; `huge_output` prints.)

# %% exercise
def exit_reason_for(probe_name: str) -> str:
    from sandboxlab.probes import BY_NAME, probe_code
    ### BEGIN SOLUTION
    sb = ProcessSandbox(Budgets(cpu_s=1, wall_s=2), netns=False)
    with standin_host() as host:
        code = probe_code(BY_NAME[probe_name], host.params(probe_name, sb.budgets))
        return sb.run(code).exit_reason
    ### END SOLUTION

# %% check
reasons = {name: exit_reason_for(name) for name in ("infinite_loop", "sleep_forever", "memory_hog", "huge_output")}
print(reasons)
assert reasons == {"infinite_loop": "cpu_time", "sleep_forever": "wall_timeout",
                   "memory_hog": "memory", "huge_output": "output_limit"}
print("✅ each abuse hits its own budget: CPU seconds, the wall clock, address space, output bytes")

# %% [markdown]
# ## Exercise 1.2 — why the wall clock is not enough, and why CPU seconds are not either
#
# A CPU-seconds limit (`RLIMIT_CPU`) never fires on code that sleeps; a wall-clock timeout never
# fires on code that is genuinely fast but loops forever on one CPU only if you have no CPU limit.
# You need both. Predict, for a 1 s CPU / 2 s wall budget, the exit reason and whether the run ends
# in about 1 s or about 2 s, for `while True: pass` and for `time.sleep(60)`.

# %% exercise
def predict(code: str) -> tuple:
    """Return (exit_reason, approx_wall_s) for Budgets(cpu_s=1, wall_s=2)."""
    ### BEGIN SOLUTION
    if "sleep" in code:
        return ("wall_timeout", 2.0)      # sleeping uses no CPU, so only the wall clock catches it
    return ("cpu_time", 1.0)        # a busy loop burns a CPU second before the wall deadline
    ### END SOLUTION

# %% check
sb = ProcessSandbox(Budgets(cpu_s=1, wall_s=2), netns=False)
for code in ("while True: pass", "import time; time.sleep(60)"):
    reason, approx = predict(code)
    r = sb.run(code)
    assert r.exit_reason == reason, (code, r.exit_reason)
    assert abs(r.wall_s - approx) < 0.6, (code, r.wall_s)
print("✅ CPU seconds catch the loop at ~1 s; the wall clock catches the sleep at ~2 s. Keep both.")

# %% [markdown]
# ## Exercise 1.3 — the hardened `docker run`, flag by flag
#
# `DockerSandbox.hardened()` builds the command notebook 01 explains. Given a set of flags you must
# not drop, assert they are all present, and that the naive spec has none of them. Then say which
# probe each flag stops (`sandboxlab.docker.FLAGS`).

# %% exercise
def has_all_controls(argv: list) -> bool:
    needed = ["--network", "--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--memory",
              "--cpus", "--user", "--tmpfs", "--init"]
    ### BEGIN SOLUTION
    s = " ".join(argv)
    return all(flag in s for flag in needed)
    ### END SOLUTION

# %% check
from sandboxlab.docker import DockerSandbox, FLAGS
assert has_all_controls(DockerSandbox.hardened().argv())
assert not has_all_controls(DockerSandbox.naive().argv("print(1)"))
stops = {flag: what for flag, _, what in ((f.split()[0], f, w) for f, _, w in FLAGS)}
assert "fork_bomb" in dict((f, w) for f, _, w in FLAGS)["--pids-limit 64"]
print("✅ every control is present in the hardened spec and absent from the naive one")
print("   the command (copy-pasteable):")
print("  ", DockerSandbox.hardened().shell("print('hello')")[:200], "...")

# %% [markdown]
# ## Exercise 1.4 — read the Docker and gVisor verdicts (and why they must be labelled)
#
# This machine may have no Docker daemon, so the Docker rungs come from
# `sandboxlab.probes.load_sample_runs()` — sample output in the documented format, not
# measurements. Assert that they are labelled as such, that a **hardened** container leaks nothing,
# and that the **default** container still leaks the network and resources. If you do have Docker,
# `python3 -m sandboxlab probes --level docker:runc` measures it for real.

# %% exercise
def hardened_leaks(runs: dict) -> int:
    """How many probes LEAKED against docker:runc in the sample verdicts."""
    ### BEGIN SOLUTION
    return sum(r.verdict == LEAKED for r in runs["docker:runc"])
    ### END SOLUTION

# %% check
from sandboxlab.probes import load_sample_runs, sample_label
runs = load_sample_runs()
assert sample_label() == "sample output in the documented format (illustrative)"
assert all(r.label == sample_label() for rows in runs.values() for r in rows)
assert hardened_leaks(runs) == 0
assert {r.probe for r in runs["docker:default"] if r.verdict == LEAKED} >= {"egress", "fork_bomb", "memory_hog"}
print(verdict_table({k: runs[k] for k in ("docker:default", "docker:runc", "docker:runsc")}))
print("✅ hardened runc and gVisor contain every probe; the default container does not — and every")
print("   Docker row says it is illustrative, because it was not measured on this machine")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "The code the model writes is untrusted input, so I design for a fully hijacked
# model and ask what it can reach. The answer has to be: nothing with ambient authority — no
# credentials, no network, no durable filesystem. On a laptop I get most of the way with a process
# sandbox: a clean environment, a throwaway workspace, `RLIMIT_CPU`/`AS`/`FSIZE`/`NOFILE`, a wall
# timeout that kills the whole process group, output truncation, and — running as a dedicated
# unprivileged UID — the agent's files and other processes become unreadable and a fork bomb hits
# `RLIMIT_NPROC`. What a process sandbox cannot do is isolate the kernel or, without a network
# namespace, touch the network. For untrusted code from the open internet I run a hardened
# container — `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, a tight
# seccomp profile, `--pids-limit`, `--memory`, non-root — and for the highest blast radius I add
# gVisor, so the syscalls hit a user-space kernel written in Go instead of the host kernel. I prove
# each layer with the probe suite: the verdict is `CONTAINED` or `LEAKED`, measured, not asserted."
#
# **Drill 1.** *We run the executor as root in a container; isn't that fine because it's
# "contained"?* — Root in the container is root against the container's kernel surface; a kernel
# bug or a permissive seccomp profile then escapes to the host. Drop to a non-root UID, drop all
# capabilities, set `no-new-privileges`, and — if the workload is genuinely adversarial — put gVisor
# under it. Defence in depth: each layer assumes the one above it failed.
#
# **Drill 2.** *`RLIMIT_NPROC` didn't stop the fork bomb in our test.* — It is not enforced for
# root, and it counts every process of the *real UID system-wide*, so it only means anything with a
# dedicated unprivileged UID (or, in a container, the pids cgroup — `--pids-limit`). Colab and many
# CI containers run as root, where the process-level limit is a no-op; that is exactly why the
# container layer exists.
#
# **Drill 3.** *Why label the Docker numbers "illustrative" when we're confident they're right?* —
# Because they were not measured on the machine that printed them. A sandbox claim is only as good
# as its evidence; mixing a measured verdict with a remembered one is how a regression ships. The
# suite measures what it can and marks the rest, so the reader always knows which is which.
