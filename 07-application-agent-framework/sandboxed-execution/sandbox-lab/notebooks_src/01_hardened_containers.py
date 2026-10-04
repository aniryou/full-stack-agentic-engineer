# %% [markdown]
# # 01 · Hardened containers: run model-generated code without ambient authority
#
# **Tier:** T0 by default. A process sandbox runs on this machine, and the notebook gives the verdicts of
# the attack probes here. The Docker and gVisor rungs print the exact commands and read bundled sample
# verdicts. These verdicts have the label *sample output in the documented format (illustrative)*. With
# Docker (T0 + Docker), the notebook measures the same probes against a real container.
#
# ## The one-minute version
#
# Model-generated code is untrusted input (PRIMER §1). The invariant is **no ambient authority**: no
# credentials, no network by default and no persistent filesystem. The purpose is that a hijacked model can
# do nothing outside the model, except the things that you told the sandbox to permit. You get to this
# invariant when you climb the isolation ladder (PRIMER §2) and turn on one control at a time:
#
# | Rung | What it adds | An attack it stops |
# |---|---|---|
# | unsandboxed | — | everything: it has the environment, files, UID and network of the agent |
# | process + budgets | CPU, memory, files, output, a wall timeout | fork bombs, miners, disk fill, output floods |
# | + a dedicated UID | the code cannot read the files of the agent or other processes | a read of `~/.ssh`, a scan of `/proc` |
# | + an empty network namespace | no route to any destination | exfiltration, the metadata server |
# | a hardened container | a boundary that the kernel enforces: caps, seccomp, read-only root, cgroups | most escalation paths |
# | + gVisor (`runsc`) | the syscalls go to a user-space kernel, not to the kernel of the host | kernel exploits |
#
# You run each attack through each rung and read the verdict: `CONTAINED` or `LEAKED`. The verdict replaces
# an argument about the control. The controls are *outside the model*, and that is the whole point.

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
# `standin_host()` builds a temporary "agent host" with these parts:
#
# - a 0700 home with a **stand-in** SSH key,
# - a stand-in agent process, whose environment holds an API-key *canary*,
# - a listener that acts as the server of the attacker.
#
# Every secret is a canary. Every grab of a resource has a limit. The simple executor is
# `subprocess.run([python, "-c", code])` in the environment of the agent itself. People ship this executor
# by accident.

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
# `ProcessSandbox` gives the code a clean environment, a temporary workspace and resource budgets. When this
# notebook runs as root (Colab does), it also gives a dedicated UID and an empty network namespace. Compare
# the columns. The budgets and the UID close most of the gaps. But a process sandbox is not a kernel
# boundary. Without a network namespace, it does nothing to the network.

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
# Each probe in this exercise uses too much of one resource. For each probe, return the `exit_reason` that
# the process sandbox reports. These reasons are the words of the execution contract (PRIMER §3). Run the
# code of the probe through a sandbox. Then read `result.exit_reason`. The probe `infinite_loop` uses CPU
# time, `sleep_forever` blocks, `memory_hog` allocates memory and `huge_output` prints.

# %% exercise
def exit_reason_for(probe_name: str) -> str:
    from sandboxlab.probes import BY_NAME, probe_code
    ### BEGIN SOLUTION
    sb = ProcessSandbox(Budgets(cpu_s=1, wall_s=5), netns=False)
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
# A CPU-seconds limit (`RLIMIT_CPU`) never stops code that sleeps. A wall-clock timeout never stops code
# that is really fast but loops forever on one CPU, only if you have no CPU limit. You must have both. The
# budget is 1 s CPU / 5 s wall. For `while True: pass` and for `time.sleep(60)`, predict the exit reason.
# Also predict which budget runs out, and after how many seconds.
#
# Why 5 s of wall for 1 s of CPU: a busy loop adds CPU seconds only while the kernel schedules it. On a
# loaded laptop or a shared CI runner, the loop can get half a core. Then its 1 s of CPU takes 2 s of wall.
# A wall limit of two times the CPU budget then catches the loop first and reports `wall_timeout`. Keep the
# wall limit at several times the CPU budget. Then each budget catches its own abuse on a busy machine too.

# %% exercise
def predict(code: str) -> tuple:
    """Return (exit_reason, approx_s) for Budgets(cpu_s=1, wall_s=5): approx_s is the budget that ran
    out, in its own unit — CPU seconds for `cpu_time`, wall seconds for `wall_timeout`."""
    ### BEGIN SOLUTION
    if "sleep" in code:
        return ("wall_timeout", 5.0)      # sleeping uses no CPU, so only the wall clock catches it
    return ("cpu_time", 1.0)        # a busy loop burns a CPU second before the wall deadline
    ### END SOLUTION

# %% check
sb = ProcessSandbox(Budgets(cpu_s=1, wall_s=5), netns=False)
for code in ("while True: pass", "import time; time.sleep(60)"):
    reason, approx = predict(code)
    r = sb.run(code)
    assert r.exit_reason == reason, (code, r.exit_reason)
    used = r.cpu_s if reason == "cpu_time" else r.wall_s
    assert used is not None and abs(used - approx) < 0.6, (code, used)
    print(f"  {code:<28} {r.exit_reason:<12} cpu {r.cpu_s or 0:.2f} s  wall {r.wall_s:.2f} s")
print("✅ CPU seconds catch the loop after ~1 CPU s (more wall time on a busy machine); the wall clock")
print("   catches the sleep at ~5 s. Keep both, with the wall limit well above the CPU budget.")

# %% [markdown]
# ## Exercise 1.3 — the hardened `docker run`, flag by flag
#
# `DockerSandbox.hardened()` builds the command that notebook 01 explains. You get a set of flags that you
# must not remove. Assert that all of them are present, and that the simple spec
# (`DockerSandbox.naive()`) has none of them. Then say which probe each flag stops
# (`sandboxlab.docker.FLAGS`).

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
# It is possible that this machine has no Docker daemon. Thus the Docker rungs come from
# `sandboxlab.probes.load_sample_runs()`. They are sample output in the documented format, not measurements.
# Assert these three things:
#
# - the Docker rungs have the label of sample output,
# - a **hardened** container leaks nothing,
# - the **default** container still leaks the network and resources.
#
# If you have Docker, `python3 -m sandboxlab probes --level docker:runc` does the real measurement.

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
# **Two minutes.** "The code that the model writes is untrusted input. Thus I design for a fully hijacked
# model, and I ask what it can reach. The answer must be: nothing with ambient authority, that is, no
# credentials, no network and no durable filesystem.
#
# "On a laptop, a process sandbox gets me most of the way. It has a clean environment, a temporary workspace
# and `RLIMIT_CPU`/`AS`/`FSIZE`/`NOFILE`. It also has a wall timeout that stops the whole process group, and
# output truncation. When it runs as a dedicated unprivileged UID, the code cannot read the files of the
# agent or other processes. With that UID, a fork bomb also hits `RLIMIT_NPROC`. A process sandbox cannot
# isolate the kernel and, without a network namespace, it cannot touch the network.
#
# "For untrusted code from the open internet, I run a hardened container. It has `--network none`,
# `--read-only`, `--cap-drop ALL`, `no-new-privileges`, a tight seccomp profile, `--pids-limit`, `--memory`
# and a non-root user. For the highest blast radius, I add gVisor, thus the syscalls go to a user-space
# kernel written in Go, not to the host kernel. I prove each layer with the probe suite. The verdict is
# `CONTAINED` or `LEAKED`. I measure the verdict, and I do not assert it."
#
# **Drill 1.** *We run the executor as root in a container. Is that not fine, because it is "contained"?*
# Root in the container is root against the kernel surface of the container. A kernel bug or a permissive
# seccomp profile then escapes to the host.
#
# Change to a non-root UID. Drop all capabilities. Set `no-new-privileges`. If the workload is really
# adversarial, put gVisor under it. Defence in depth: each layer assumes that the layer above it failed.
#
# **Drill 2.** *`RLIMIT_NPROC` did not stop the fork bomb in our test.* The kernel does not enforce it for
# root. Also, it counts every process of the *real UID system-wide*. Thus it has an effect only with a
# dedicated unprivileged UID (or, in a container, the pids cgroup: `--pids-limit`). Colab and many CI
# containers run as root, where the process-level limit does nothing. That is exactly the reason for the
# container layer.
#
# **Drill 3.** *Why do we give the Docker numbers the label "illustrative", when we are sure that they are
# correct?* Because the machine that printed them did not measure them. A sandbox claim is only as good as
# its evidence. A mix of a measured verdict and a remembered one is how a regression ships. The suite
# measures what it can and marks the rest, thus the reader always knows which is which.
