"""env.py — decide what can run here: which tier, which teacher server, which libraries.

One idea: a notebook that must run anywhere *detects* instead of assuming. ``DISTILLAB_URL`` pointing at a
real OpenAI-compatible server (``vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-logprobs 20`` on a Colab T4, a
tunnel, a Cloud Run URL) means measurements (T1/T3) — unless its ``/version`` says it is this lab's fake
teacher, which stays labelled *simulated*. Otherwise the fake teacher starts in-process (T0). Torch present
means the tiny transformers train for real (T0 on a CPU); absent, the notebooks load the recorded run.
``transformers`` / ``trl`` / ``peft`` and a GPU decide whether the ``distillab.hf`` trainers can run (T1).

    DISTILLAB_URL=http://127.0.0.1:8000        a running teacher to measure (vLLM, SGLang, the fake one)
    DISTILLAB_MODEL=Qwen/Qwen2.5-1.5B-Instruct the model name to send (default: the server's first model)
    DISTILLAB_API_KEY=...                      sent as "Authorization: Bearer ..." (vllm --api-key)
    DISTILLAB_BEARER=$(gcloud auth print-identity-token)   for a private Cloud Run service
    DISTILLAB_NO_TORCH=1                       behave as if torch were absent (the recorded-run path)
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


def has(module: str) -> bool:
    if module == "torch" and os.environ.get("DISTILLAB_NO_TORCH") == "1":
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):       # a module set to None in sys.modules (a "hidden" install)
        return False


def has_torch() -> bool:
    if not has("torch"):
        return False
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 — a broken or hidden install behaves as absent
        return False
    return True


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def hf_stack() -> dict:
    """Which of the T1 training libraries are importable (without importing them)."""
    return {m: has(m) for m in ("transformers", "trl", "peft", "datasets", "vllm")}


def on_colab() -> bool:
    return "google.colab" in sys.modules


def server_url() -> str | None:
    return os.environ.get("DISTILLAB_URL") or None


def auth_headers() -> dict:
    token = os.environ.get("DISTILLAB_BEARER") or os.environ.get("DISTILLAB_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def describe() -> str:
    stack = ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in hf_stack().items())
    return (f"GPU: {gpu_name() or 'none'} | torch: {'yes' if has_torch() else 'no'} | {stack} | "
            f"Colab: {on_colab()} | DISTILLAB_URL: {server_url() or 'unset'}")


def _get_json(url: str, headers: dict | None = None, timeout: float = 5) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode() or "{}")


def is_simulated(url: str, headers: dict | None = None) -> bool:
    try:
        return bool(_get_json(url.rstrip("/") + "/version", headers).get("simulated"))
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


def first_model(url: str, headers: dict | None = None) -> str | None:
    try:
        return _get_json(url.rstrip("/") + "/v1/models", headers)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Target:
    url: str
    simulated: bool
    tier: str
    note: str
    model: str | None = None
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
        return f"{self.tier}: {self.note} -> {self.url} (model {self.model})"


def connect(**fake_kw) -> Target:
    """``DISTILLAB_URL`` if set (real unless its /version says simulated), else an in-process fake teacher."""
    url = server_url()
    if url:
        h = auth_headers()
        if not wait_healthy(url, timeout_s=30, headers=h):
            raise RuntimeError(f"DISTILLAB_URL={url} is not healthy (GET /health)")
        model = os.environ.get("DISTILLAB_MODEL") or first_model(url, h)
        if is_simulated(url, h):
            return Target(url, True, "T0", "fake teacher at DISTILLAB_URL (simulated)", model, headers=h)
        return Target(url, False, "T1/T3", "real server from DISTILLAB_URL", model, headers=h)
    from .fakeserver import FakeTeacher
    server = FakeTeacher(**fake_kw)
    return Target(server.start(), True, "T0", "in-process fake teacher (simulated)", server.model, server)
