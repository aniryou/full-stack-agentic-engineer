"""env.py — decide what can run here: which tier, which target.

One idea: a notebook that must run anywhere *detects* instead of assuming. A server URL in
``SERVELAB_URL`` (vLLM on this box, a tunnel to Colab, a Cloud Run URL) means real measurements
(T1/T3). A GPU with vLLM installed and ``SERVELAB_START_VLLM=1`` means start one here (T1).
Otherwise the fake server starts and every number is labelled *simulated* (T0).

    SERVELAB_URL=http://127.0.0.1:8000      a running OpenAI-compatible server to measure
    SERVELAB_API_KEY=...                    sent as "Authorization: Bearer ..." (vllm --api-key)
    SERVELAB_BEARER=$(gcloud auth print-identity-token)   for a private Cloud Run service
    SERVELAB_START_VLLM=1                   allow notebooks to launch `vllm serve` on a local GPU
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass


def has_gpu() -> bool:
    return gpu_name() is not None


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
    return importlib.util.find_spec("vllm") is not None or shutil.which("vllm") is not None


def has_docker() -> bool:
    """A docker CLI *and* a reachable daemon (``docker info`` succeeds)."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def has_gcloud() -> bool:
    return shutil.which("gcloud") is not None


def on_colab() -> bool:
    return "google.colab" in sys.modules


def server_url() -> str | None:
    return os.environ.get("SERVELAB_URL") or None


def auth_headers() -> dict:
    token = os.environ.get("SERVELAB_BEARER") or os.environ.get("SERVELAB_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def describe() -> str:
    return (f"GPU: {gpu_name() or 'none'} | vLLM installed: {has_vllm()} | docker: {has_docker()} | "
            f"gcloud: {has_gcloud()} | Colab: {on_colab()} | SERVELAB_URL: {server_url() or 'unset'}")


def wait_healthy(url: str, timeout_s: float = 900, headers: dict | None = None) -> bool:
    """Poll ``/health`` (vLLM returns 200 once the engine is ready) until ``timeout_s``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url.rstrip("/") + "/health", headers=headers or {})
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 — not up yet
            pass
        time.sleep(2)
    return False


@dataclass
class Target:
    url: str
    simulated: bool
    tier: str
    note: str
    handle: object = None         # the FakeServer / VLLMBackend we started, if any
    headers: dict | None = None

    def stop(self) -> None:
        if self.handle is not None:
            self.handle.stop()
            self.handle = None

    def __str__(self) -> str:
        return f"{self.tier}: {self.note} -> {self.url}"


def connect(profile: str = "t4-qwen2.5-0.5b", config=None, vllm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
            vllm_flags: dict | None = None, **fake_kw) -> Target:
    """The best target available: ``SERVELAB_URL`` > local vLLM (opt-in) > the fake server."""
    url = server_url()
    if url:
        if not wait_healthy(url, timeout_s=30, headers=auth_headers()):
            raise RuntimeError(f"SERVELAB_URL={url} is not healthy (GET /health)")
        return Target(url, False, "T1/T3", "real server from SERVELAB_URL", headers=auth_headers())
    if os.environ.get("SERVELAB_START_VLLM") == "1" and has_gpu() and has_vllm():
        from .tune import VLLMBackend
        backend = VLLMBackend(vllm_model, base_flags=vllm_flags or {})
        return Target(backend.start({}), False, "T1", f"vllm serve {vllm_model} on {gpu_name()}", handle=backend)
    from .fakeserver import FakeServer
    server = FakeServer(profile, config, **fake_kw)
    return Target(server.start(), True, "T0", f"fake vLLM, simulated profile {profile}", handle=server)
