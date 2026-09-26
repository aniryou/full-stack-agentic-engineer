"""docker.py — a hardened ``docker run`` for one execution, flag by flag, and the probes through it.

One idea: a container is a process with namespaces, cgroups, a capability set and a seccomp
filter around it — and every one of those is *off or permissive* until you ask. ``docker run
python:3.12-slim python -c CODE`` gives the code root in the container, the bridge network, all
default capabilities, no memory, CPU or process limit and a writable root filesystem. The
builder below turns each missing control on and says which probe it stops (PRIMER §2
"The isolation ladder": containers; gVisor with ``--runtime runsc``).

    spec = DockerSandbox.hardened()                        # runc; DockerSandbox.hardened("runsc") for gVisor
    print(spec.shell("print(1)"))                          # the exact command, copy-pasteable
    spec.run("print(1)")                                   # needs a Docker daemon (T0 + Docker)

Without a Docker daemon nothing is executed: notebooks print the commands and read the bundled
``data/samples/docker_verdicts.json`` — sample output in the documented format (illustrative).
"""
from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from .process import Budgets, ExecResult, parse_wrapper_output

SANDBOX_IMAGE = "python:3.12-slim"          # (verify) pin by digest before production use
WRAPPER_IN_CONTAINER = "/opt/sandbox/wrapper.py"
LAB_ROOT = Path(__file__).resolve().parents[1]
SECCOMP_FILE = LAB_ROOT / "deploy" / "docker" / "seccomp-sandbox.json"
WRAPPER_FILE = Path(__file__).with_name("wrapper.py")
SANDBOX_UID = 65534                          # numeric, so the runtime and kubelet can check "non-root"


@dataclass
class DockerSandbox:
    image: str = SANDBOX_IMAGE
    runtime: str | None = None               # "runsc" = gVisor (needs `sudo runsc install`)
    network: str = "none"                    # "none" | "bridge" | an --internal network with the egress proxy
    read_only: bool = True
    cap_drop_all: bool = True
    no_new_privileges: bool = True
    seccomp: str | None = str(SECCOMP_FILE)  # None: Docker's default profile; "unconfined": none at all
    pids_limit: int | None = 64
    memory_mib: int | None = 320             # cgroup limit: the wrapper's RLIMIT_AS budget + ~64 MiB for the wrapper
    cpus: float | None = 1.0
    user: str | None = f"{SANDBOX_UID}:{SANDBOX_UID}"
    workspace_mib: int | None = 16           # tmpfs /work: the only writable place (plus /tmp)
    tmp_mib: int | None = 16
    init: bool = True                        # tini as PID 1: reaps zombies, forwards signals
    wrap: bool = True                        # run through wrapper.py (budgets, truncation, exit reason)
    env: dict[str, str] = field(default_factory=dict)
    add_hosts: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=lambda: {"app": "sandboxlab"})
    budgets: Budgets = field(default_factory=Budgets)

    # -- presets ---------------------------------------------------------------------------------
    @classmethod
    def hardened(cls, runtime: str | None = None, network: str = "none", **kw) -> "DockerSandbox":
        return cls(runtime=runtime, network=network, **kw)

    @classmethod
    def naive(cls) -> "DockerSandbox":
        """What most first attempts look like: isolation by default settings only."""
        return cls(network="bridge", read_only=False, cap_drop_all=False, no_new_privileges=False, seccomp=None,
                   pids_limit=None, memory_mib=None, cpus=None, user=None, workspace_mib=None, tmp_mib=None,
                   init=False, wrap=False, labels={"app": "sandboxlab"})

    @property
    def name(self) -> str:
        if not self.wrap and not self.read_only:
            return "docker:default"
        return f"docker:{self.runtime or 'runc'}"

    # -- the command -----------------------------------------------------------------------------
    def argv(self, code: str | None = None, *, execution_id: str | None = None,
             wrapper_path: Path | str = WRAPPER_FILE) -> list[str]:
        """``docker run ...`` for one execution. With ``wrap`` the code goes on stdin; without it,
        it is passed as ``python3 -c CODE`` (so ``code`` is required)."""
        a = ["docker", "run", "--rm", "-i"]
        if execution_id:
            a += ["--name", f"sbx-{execution_id}"]
        for k, v in self.labels.items():
            a += ["--label", f"{k}={v}"]
        if self.runtime:
            a += ["--runtime", self.runtime]
        a += ["--network", self.network]
        for host, ip in self.add_hosts.items():
            a += ["--add-host", f"{host}:{ip}"]
        if self.read_only:
            a += ["--read-only"]
        if self.cap_drop_all:
            a += ["--cap-drop", "ALL"]
        if self.no_new_privileges:
            a += ["--security-opt", "no-new-privileges"]
        if self.seccomp:
            a += ["--security-opt", f"seccomp={self.seccomp}"]
        if self.pids_limit:
            a += ["--pids-limit", str(self.pids_limit)]
        if self.memory_mib:
            a += ["--memory", f"{self.memory_mib}m", "--memory-swap", f"{self.memory_mib}m"]   # equal = no swap
        if self.cpus:
            a += ["--cpus", f"{self.cpus:g}"]
        if self.user:
            a += ["--user", self.user]
        if self.workspace_mib:
            a += ["--tmpfs", f"/work:rw,nosuid,nodev,noexec,size={self.workspace_mib}m,mode=1777"]
        if self.tmp_mib:
            a += ["--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={self.tmp_mib}m,mode=1777"]
        if self.read_only or self.cap_drop_all:
            a += ["--ulimit", "nofile=256:256", "--ulimit", "core=0"]
        if self.init:
            a += ["--init"]
        a += ["--hostname", "sandbox", "--workdir", "/work" if self.workspace_mib else "/"]
        for k, v in self.env.items():
            a += ["--env", f"{k}={v}"]
        if self.wrap:
            a += ["--volume", f"{wrapper_path}:{WRAPPER_IN_CONTAINER}:ro", self.image,
                  "python3", "-I", WRAPPER_IN_CONTAINER, "--stdin", "--workspace", "/work",
                  "--budgets", self.budgets.wrapper_json()]
            if self.runtime == "runsc":
                a += ["--nproc"]          # inside gVisor, RLIMIT_NPROC counts only the sandbox's processes
            for k in self.env:
                a += ["--pass-env", k]
        else:
            if code is None:
                raise ValueError("an unwrapped run passes the code as `python3 -c CODE`: give the code")
            a += [self.image, "python3", "-c", code]
        return a

    def shell(self, code: str = "print('hello from the sandbox')") -> str:
        """The command as you would type it (the code on stdin for wrapped runs)."""
        argv = self.argv(code)
        cmd = " ".join(shlex.quote(x) for x in argv)
        return f"printf %s {shlex.quote(code)} | {cmd}" if self.wrap else cmd

    # -- running it (needs Docker) ---------------------------------------------------------------
    def run(self, code: str, budgets: Budgets | None = None, env: dict | None = None) -> ExecResult:
        spec = replace(self, budgets=budgets) if budgets else self
        if env:
            spec = replace(spec, env={**spec.env, **env})
        eid = uuid.uuid4().hex[:12]
        argv = spec.argv(code, execution_id=eid)
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, input=code.encode() if spec.wrap else None, capture_output=True,
                               timeout=spec.budgets.wall_s + 60)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", f"sbx-{eid}"], capture_output=True)
            return ExecResult("sandbox_error", None, "", "", isolation=spec.name,
                              total_s=time.monotonic() - t0, notes=["docker run did not return in time"])
        total = time.monotonic() - t0
        out, err = p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
        if spec.wrap:
            d = parse_wrapper_output(out)
            if d is None:     # image pull failure, unknown runtime, bad flag: Docker's own error is on stderr
                return ExecResult("sandbox_error", p.returncode, "", err[-2000:], isolation=spec.name, total_s=total,
                                  notes=[f"docker exited {p.returncode} without a result line"])
            return ExecResult.from_wrapper(d, isolation=spec.name, total_s=total)
        reason = "ok" if p.returncode == 0 else ("killed" if p.returncode >= 128 else "error")
        return ExecResult(reason, p.returncode, out, err, stdout_bytes=len(p.stdout), stderr_bytes=len(p.stderr),
                          wall_s=round(total, 4), total_s=round(total, 4), isolation=spec.name,
                          notes=["no wrapper: no budgets, no truncation"])


# ---- what each flag buys ----------------------------------------------------------------------------
FLAGS: list[tuple[str, str, str]] = [
    ("--network none", "only a loopback interface: no route anywhere, not even to the host", "egress, metadata"),
    ("--read-only", "the image's filesystem cannot be modified (no dropped binaries, no edited libraries)", "write_outside, persistence"),
    ("--tmpfs /work:size=16m,noexec", "the one writable place is small, in memory, and cannot hold executables", "disk_fill, disk_fill_many"),
    ("--cap-drop ALL", "root in the container has none of the kernel's capabilities (mount, raw sockets, chown...)", "escalation paths"),
    ("--security-opt no-new-privileges", "setuid binaries and file capabilities stop working (su, sudo, ping)", "escalation paths"),
    ("--security-opt seccomp=seccomp-sandbox.json", "syscalls outside an allowlist fail with EPERM (see seccomp.py)", "kernel attack surface"),
    ("--user 65534:65534", "not root even inside the container; numeric so the runtime can check it", "escalation paths"),
    ("--pids-limit 64", "the pids cgroup refuses the 65th task (host tasks: under gVisor, the Sentry's)", "fork_bomb"),
    ("--memory 320m --memory-swap 320m", "the memory cgroup OOM-kills above 320 MiB, and there is no swap to hide in", "memory_hog"),
    ("--cpus 1", "at most one CPU's worth of time per period: a miner cannot take the host", "infinite_loop (throughput)"),
    ("--ulimit nofile=256:256 --ulimit core=0", "bounded file descriptors, no core dumps with secrets in them", "resource abuse"),
    ("--init", "a real PID 1 that reaps zombies; the container (and every process in it) ends with the run", "escape_session"),
    ("wrapper.py (stdin)", "CPU seconds, wall clock, output bytes, one exit reason: the contract inside the box", "infinite_loop, sleep_forever, huge_output"),
    ("--runtime runsc", "gVisor: the code's syscalls go to the Sentry, a user-space kernel, not to the host kernel", "kernel exploits"),
]


def explain() -> str:
    w = max(len(f) for f, _, _ in FLAGS)
    return "\n".join(f"{f:<{w}}  {what}  [stops: {stops}]" for f, what, stops in FLAGS)


# ---- gVisor ------------------------------------------------------------------------------------------
GVISOR_INSTALL = [
    # the apt route from gVisor's install guide (g3doc/user_guide/install.md, 2026-09); it also configures Docker
    "sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg",
    "curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg",
    'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] '
    'https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null',
    "sudo apt-get update && sudo apt-get install -y runsc",
    "sudo runsc install && sudo systemctl restart docker     # registers the runtime 'runsc' in /etc/docker/daemon.json",
    "docker run --rm --runtime=runsc hello-world",
]


def docker_available() -> bool:
    from .env import has_docker
    return has_docker()


def ensure_seccomp_file() -> Path:
    """The generated profile the ``--security-opt seccomp=`` flag points at (``render.py`` writes it)."""
    if not SECCOMP_FILE.exists():
        from . import seccomp
        SECCOMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECCOMP_FILE.write_text(seccomp.to_json(seccomp.sandbox_profile()))
    return SECCOMP_FILE


def listener_hosts_for(spec: DockerSandbox) -> list[str]:
    """Where a container would look for 'the host': its own loopback, and the host gateway."""
    return ["127.0.0.1", "host.docker.internal"]


def with_host_gateway(spec: DockerSandbox) -> DockerSandbox:
    """Give bridge-networked specs a name for the host (what an attacker would try first)."""
    if spec.network == "none":
        return spec
    return replace(spec, add_hosts={**spec.add_hosts, "host.docker.internal": "host-gateway"})


def env_summary() -> str:
    from .env import docker_info
    info = docker_info()
    if not info:
        return "no Docker daemon reachable: commands are printed, verdicts come from bundled sample output"
    return f"Docker {info['server_version']}, runtimes: {', '.join(info['runtimes'])}"

