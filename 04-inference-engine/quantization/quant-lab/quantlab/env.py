"""env.py — decide what can run here: which tier, which GPU, which tools.

One idea: a notebook that must run anywhere *detects* instead of assuming. ``nvidia-smi`` gives
the GPU and its compute capability (which decides the scheme table in :mod:`quantlab.serve`);
importable packages decide whether llm-compressor, vLLM or lm-eval can run; ``QUANTLAB_URL``
points the benchmarks at a real server. With none of that, every notebook runs its T0 path and
labels its numbers simulated.

    QUANTLAB_URL=http://127.0.0.1:8000   an OpenAI-compatible server to measure (vllm serve, a Colab tunnel)
    QUANTLAB_MODEL=<served model name>    the model id that server expects (default: ask /v1/models)
    QUANTLAB_API_KEY=...                  sent as "Authorization: Bearer ..."
    QUANTLAB_RUN_T1=1                     allow notebooks to run llm-compressor / lm-eval on a local GPU
    QUANTLAB_VLLM_LOG=/path/vllm.log      a real `vllm serve` startup log for notebook 04 to read back
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request

from . import serve


def gpu_info() -> dict | None:
    """``{"name", "compute_cap", "memory_mib"}`` of GPU 0 from nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode or not out.stdout.strip():
        return None
    name, cc, mem = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")[:3]]
    return {"name": name, "compute_cap": float(cc), "memory_mib": float(mem)}


def closest_gpu(name: str) -> str | None:
    """Map an nvidia-smi name ("Tesla T4", "NVIDIA L4", "NVIDIA A100-SXM4-80GB") to a key of ``serve.GPUS``."""
    n = name.upper().replace("NVIDIA", "").replace("TESLA", "").replace("GEFORCE", "").replace(" ", "")
    if "L40" in n:
        return None                                  # L40 / L40S: not in this lab's table
    if "A100" in n:
        return "A100-80GB" if "80" in n else "A100-40GB"
    for key, k in (("RTXPRO6000", "RTXPRO6000"), ("RTX4090", "RTX4090"), ("H100", "H100-80GB"), ("B200", "B200"),
                   ("L4", "L4"), ("T4", "T4")):
        if key in n:
            return k
    return None


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def has_cli(name: str) -> bool:
    return shutil.which(name) is not None


def has_docker() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def on_colab() -> bool:
    return "google.colab" in sys.modules


def server_url() -> str | None:
    return os.environ.get("QUANTLAB_URL") or None


def auth_headers() -> dict:
    k = os.environ.get("QUANTLAB_API_KEY")
    return {"Authorization": f"Bearer {k}"} if k else {}


def served_model(url: str) -> str:
    """The first model id the server lists (or ``QUANTLAB_MODEL``)."""
    if os.environ.get("QUANTLAB_MODEL"):
        return os.environ["QUANTLAB_MODEL"]
    req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers=auth_headers())
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
        return json.loads(r.read())["data"][0]["id"]


def get_text(url: str, path: str, timeout: float = 10) -> str:
    """GET ``url + path`` (with ``QUANTLAB_API_KEY`` if set) as text, e.g. a server's ``/metrics``."""
    req = urllib.request.Request(url.rstrip("/") + path, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read().decode()


def t1_allowed() -> bool:
    return os.environ.get("QUANTLAB_RUN_T1") == "1" and gpu_info() is not None


def describe() -> str:
    g = gpu_info()
    gs = f"{g['name']} (sm_{int(round(g['compute_cap'] * 10))})" if g else "none"
    return (f"GPU: {gs} | torch: {has('torch')} | vllm: {has('vllm')} | llmcompressor: {has('llmcompressor')} | "
            f"lm_eval: {has_cli('lm_eval')} | docker: {has_docker()} | Colab: {on_colab()} | "
            f"QUANTLAB_URL: {server_url() or 'unset'}")
