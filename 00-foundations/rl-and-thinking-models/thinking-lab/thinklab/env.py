"""env.py — decide what can run here: which tier, which server.

One idea: a notebook that must run anywhere *detects* instead of assuming. ``THINKLAB_URL`` pointing at
a real OpenAI-compatible server (your ``vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`` on a Colab
T4, a tunnel, a Cloud Run URL) means measurements (T1/T3) — unless its ``/version`` says it is this
lab's fake server, which stays labelled *simulated*. Otherwise the fake server starts in-process and
every number is simulated (T0). Torch present means the tiny-transformer RL runs for real (T0 on a CPU,
faster at T1); absent, notebooks load the recorded curves.

    THINKLAB_URL=http://127.0.0.1:8000       a running server to measure (vLLM, SGLang, the fake one)
    THINKLAB_API_KEY=...                     sent as "Authorization: Bearer ..." (vllm --api-key)
    THINKLAB_BEARER=$(gcloud auth print-identity-token)   for a private Cloud Run service
    THINKLAB_NO_TORCH=1                      behave as if torch were absent (the recorded-run path)
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass


def has_torch() -> bool:
    if os.environ.get("THINKLAB_NO_TORCH") == "1":
        return False
    return importlib.util.find_spec("torch") is not None


def torch_device() -> str:
    if not has_torch():
        return "none"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def has_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


def has_docker() -> bool:
    """A docker CLI *and* a reachable daemon."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def on_colab() -> bool:
    return "google.colab" in sys.modules


def server_url() -> str | None:
    return os.environ.get("THINKLAB_URL") or None


def auth_headers() -> dict:
    token = os.environ.get("THINKLAB_BEARER") or os.environ.get("THINKLAB_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def describe() -> str:
    return (f"GPU: {gpu_name() or 'none'} | torch: {torch_device()} | vLLM installed: {has_vllm()} | "
            f"docker: {has_docker()} | Colab: {on_colab()} | THINKLAB_URL: {server_url() or 'unset'}")


def is_simulated(url: str, headers: dict | None = None) -> bool:
    try:
        req = urllib.request.Request(url.rstrip("/") + "/version", headers=headers or {})
        with urllib.request.urlopen(req, timeout=5) as r:   # noqa: S310
            return bool(json.loads(r.read().decode() or "{}").get("simulated"))
    except Exception:  # noqa: BLE001 — a real vLLM answers {"version": ...} only
        return False


def wait_healthy(url: str, timeout_s: float = 900, headers: dict | None = None) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + "/health", headers=headers or {}),
                                        timeout=5) as r:   # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


@dataclass
class Target:
    url: str
    simulated: bool
    tier: str
    note: str
    handle: object = None
    headers: dict | None = None

    @property
    def label(self) -> str:
        return "SIMULATED" if self.simulated else "MEASURED"

    def stop(self) -> None:
        if self.handle is not None:
            self.handle.stop()
            self.handle = None

    def __str__(self) -> str:
        return f"{self.tier}: {self.note} -> {self.url}"


def connect(profile: str = "t4-qwen3-0.6b", **fake_kw) -> Target:
    """``THINKLAB_URL`` if set (real unless its /version says simulated), else an in-process fake server."""
    url = server_url()
    if url:
        if not wait_healthy(url, timeout_s=30, headers=auth_headers()):
            raise RuntimeError(f"THINKLAB_URL={url} is not healthy (GET /health)")
        if is_simulated(url, auth_headers()):
            return Target(url, True, "T0", "fake server at THINKLAB_URL (simulated)", headers=auth_headers())
        return Target(url, False, "T1/T3", "real server from THINKLAB_URL", headers=auth_headers())
    from .fakeserver import FakeServer
    server = FakeServer(profile, **fake_kw)
    return Target(server.start(), True, "T0", f"in-process fake server, simulated profile {profile}", server)
