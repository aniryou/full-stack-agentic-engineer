"""env.py — find out which isolation levels this machine can actually run, instead of assuming.

One idea: a notebook that must run on a laptop, on Colab and next to a GKE cluster *detects* each
rung of the isolation ladder (PRIMER §2) and says which verdicts are measured here and which come
from bundled sample output. Nothing in this module changes the machine; every probe is a
read-only command with a short timeout.

    SANDBOXLAB_KUBE_CONTEXT=kind-sandbox-lab   a kubectl context to drive (else the current one)
    SANDBOXLAB_NO_DOCKER=1                     pretend Docker is absent (to see the T0 path)
    SANDBOXLAB_NO_NETNS=1                      pretend network namespaces are unavailable
"""
from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


def _run(cmd: list[str], timeout: float = 8) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def on_colab() -> bool:
    return "google.colab" in sys.modules


@functools.lru_cache(maxsize=None)
def netns_mode() -> str | None:
    """How this machine can give a process its own (empty) network namespace.

    ``"root"``   ``unshare -n`` works (root with CAP_SYS_ADMIN, e.g. Colab or a CI container);
    ``"userns"`` ``unshare -Urn`` works (an unprivileged user namespace; many Linux laptops);
    ``None``     neither (macOS, Docker's default seccomp profile, Ubuntu's AppArmor userns policy).
    """
    if os.environ.get("SANDBOXLAB_NO_NETNS") == "1" or not is_linux() or not shutil.which("unshare"):
        return None
    for mode, cmd in (("root", ["unshare", "-n", "--", "true"]), ("userns", ["unshare", "-Urn", "--", "true"])):
        if mode == "root" and not is_root():
            continue
        r = _run(cmd, timeout=5)
        if r is not None and r.returncode == 0:
            return mode
    return None


def netns_prefix() -> list[str]:
    mode = netns_mode()
    return {"root": ["unshare", "-n", "--"], "userns": ["unshare", "-Urn", "--"]}.get(mode, [])


@functools.lru_cache(maxsize=None)
def docker_info() -> dict | None:
    """``{"runtimes": [...], "server_version": ...}`` when a Docker daemon answers, else None."""
    if os.environ.get("SANDBOXLAB_NO_DOCKER") == "1" or not shutil.which("docker"):
        return None
    r = _run(["docker", "info", "--format", "{{json .Runtimes}}|{{.ServerVersion}}|{{.SecurityOptions}}"])
    if r is None or r.returncode != 0 or "|" not in r.stdout:
        return None
    import json
    runtimes_json, version, secopts = r.stdout.strip().split("|", 2)
    try:
        runtimes = sorted(json.loads(runtimes_json))
    except ValueError:
        runtimes = []
    return {"runtimes": runtimes, "server_version": version, "security_options": secopts}


def has_docker() -> bool:
    return docker_info() is not None


def has_runsc() -> bool:
    """gVisor registered as a Docker runtime (``sudo runsc install``)."""
    info = docker_info()
    return bool(info and "runsc" in info["runtimes"])


def kube_context() -> str | None:
    """A reachable kubectl context: ``SANDBOXLAB_KUBE_CONTEXT`` or the current one."""
    if not shutil.which("kubectl"):
        return None
    ctx = os.environ.get("SANDBOXLAB_KUBE_CONTEXT")
    if not ctx:
        r = _run(["kubectl", "config", "current-context"], timeout=5)
        ctx = r.stdout.strip() if r is not None and r.returncode == 0 else None
    if not ctx:
        return None
    r = _run(["kubectl", "--context", ctx, "get", "--raw", "/readyz"], timeout=8)
    return ctx if r is not None and r.returncode == 0 else None


def has_gcloud() -> bool:
    return shutil.which("gcloud") is not None


def has_terraform() -> bool:
    return shutil.which("terraform") is not None


@dataclass
class Capabilities:
    os: str
    root: bool
    netns: str | None
    docker: bool
    runsc: bool
    kube_context: str | None
    gcloud: bool
    colab: bool

    def levels(self) -> list[str]:
        """Isolation levels that can be *measured* here, cheapest first."""
        out = ["none", "process"]
        if self.netns:
            out.append("process+netns")
        if self.docker:
            out.append("docker:runc")
        if self.runsc:
            out.append("docker:runsc")
        if self.kube_context:
            out.append("k8s:job")
        return out

    def describe(self) -> str:
        return (f"OS {self.os} | root: {self.root} | netns: {self.netns or 'unavailable'} | "
                f"docker: {self.docker} (gVisor runsc: {self.runsc}) | kubectl context: {self.kube_context or 'none'} | "
                f"gcloud: {self.gcloud} | Colab: {self.colab}")


def capabilities() -> Capabilities:
    return Capabilities(os=f"{platform.system()} {platform.release()}", root=is_root(), netns=netns_mode(),
                        docker=has_docker(), runsc=has_runsc(), kube_context=kube_context(),
                        gcloud=has_gcloud(), colab=on_colab())
