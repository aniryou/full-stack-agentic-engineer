"""env.py — decide what can run here: which tier, which backend, which server.

One idea: a notebook that must run anywhere *detects* instead of assuming. Torch present means the
tiny MoE trains for real on the CPU (T0); a GPU with vLLM (and ``MOELAB_START_VLLM=1``) or a server
URL means real measurements (T1/T2); anything else falls back to simulators and bundled fixtures,
and every number says which of the three it is: **measured**, **simulated**, or **illustrative**.

    MOELAB_URL=http://127.0.0.1:8000        an OpenAI-compatible server running the MoE model
    MOELAB_DENSE_URL=http://127.0.0.1:8001  (optional) the same for a dense model to compare with
    MOELAB_API_KEY=...                      sent as "Authorization: Bearer ..." (vllm serve --api-key)
    MOELAB_START_VLLM=1                     allow notebooks to launch `vllm serve` on a local GPU
    MOELAB_HF_MODEL=allenai/OLMoE-1B-7B-0924-Instruct   let notebook 02 load a real HF MoE (T1)
    MOELAB_NO_TORCH=1                       pretend torch is absent (checks the numpy-only path)
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

MEASURED, SIMULATED, ILLUSTRATIVE = "measured", "simulated", "illustrative"
FIXTURE_LABEL = "sample output in the documented format (illustrative)"


def has_torch() -> bool:
    if os.environ.get("MOELAB_NO_TORCH") == "1":
        return False
    return importlib.util.find_spec("torch") is not None


def torch():
    """Import torch lazily; raise a clear error on the numpy-only path."""
    if not has_torch():
        raise ImportError("torch is not available (or MOELAB_NO_TORCH=1): use the numpy path / bundled data")
    return importlib.import_module("torch")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_transformers() -> bool:
    return has_module("transformers") and has_torch()


def _nvidia_smi(query: str) -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()] if out.returncode == 0 else []


def gpus() -> list[str]:
    """``name, memory.total`` per visible NVIDIA GPU (empty without a GPU)."""
    return _nvidia_smi("name,memory.total")


def gpu_count() -> int:
    return len(gpus())


def has_gpu() -> bool:
    return gpu_count() > 0


def compute_capability() -> float | None:
    cc = _nvidia_smi("compute_cap")
    try:
        return float(cc[0]) if cc else None
    except ValueError:
        return None


def has_vllm() -> bool:
    return has_module("vllm") or shutil.which("vllm") is not None


def has_docker() -> bool:
    """A docker CLI *and* a reachable daemon (``docker info`` succeeds)."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def has_kubectl_cluster() -> bool:
    """kubectl installed *and* a cluster answering (``kubectl version --request-timeout=5s``)."""
    if not shutil.which("kubectl"):
        return False
    try:
        return subprocess.run(["kubectl", "get", "--raw", "/readyz", "--request-timeout=5s"],
                              capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def on_colab() -> bool:
    return "google.colab" in sys.modules


def on_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def server_url(dense: bool = False) -> str | None:
    return os.environ.get("MOELAB_DENSE_URL" if dense else "MOELAB_URL") or None


def auth_headers() -> dict:
    token = os.environ.get("MOELAB_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def may_start_vllm() -> bool:
    return os.environ.get("MOELAB_START_VLLM") == "1" and has_gpu() and has_vllm()


def tier() -> str:
    """The highest tier this machine can measure at: T2 (2+ GPUs), T1 (1 GPU or a server URL), T0."""
    n = gpu_count()
    if n >= 2:
        return "T2"
    if n == 1 or server_url():
        return "T1"
    return "T0"


def describe() -> str:
    g = gpus()
    return (f"tier {tier()} | GPUs: {', '.join(g) if g else 'none'} | torch: {has_torch()} | "
            f"vLLM: {has_vllm()} | transformers: {has_module('transformers')} | docker daemon: {has_docker()} | "
            f"Colab: {on_colab()} | Kaggle: {on_kaggle()} | MOELAB_URL: {server_url() or 'unset'}")


def wait_healthy(url: str, timeout_s: float = 900, headers: dict | None = None, proc=None) -> bool:
    """Poll ``/health`` (vLLM returns 200 once the engine is ready); give up early if ``proc`` died."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url.rstrip("/") + "/health", headers=headers or {})
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 — not up yet
            pass
        time.sleep(2)
    return False


def served_model(url: str, headers: dict | None = None) -> str:
    req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
        return json.loads(r.read())["data"][0]["id"]


class VLLMServer:
    """Start ``vllm serve <model> <flags>`` on this machine (T1/T2), wait for ``/health``, stop it.

        with VLLMServer("allenai/OLMoE-1B-7B-0924-Instruct", ["--dtype", "half"]) as url: ...

    Opt-in only (``MOELAB_START_VLLM=1``): it takes the GPU and several minutes. The log goes to
    ``log_path`` so the startup lines (memory, KV cache, the MoE config warning) can be parsed."""

    def __init__(self, model: str, flags: list[str] | None = None, port: int = 8000,
                 log_path: str = "vllm-moelab.log", timeout_s: float = 1200):
        self.model, self.flags, self.port = model, list(flags or []), port
        self.log_path, self.timeout_s, self.proc = log_path, timeout_s, None

    @property
    def command(self) -> list[str]:
        return ["vllm", "serve", self.model, "--port", str(self.port), *self.flags]

    def __str__(self) -> str:
        return shlex.join(self.command)

    def start(self) -> str:
        url = f"http://127.0.0.1:{self.port}"
        log = open(self.log_path, "w")  # noqa: SIM115 — closed in stop()
        self.proc = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT)
        self._log = log
        if not wait_healthy(url, self.timeout_s, proc=self.proc):
            self.stop()
            raise RuntimeError(f"`{self}` did not become healthy; see {self.log_path}")
        return url

    def log_text(self) -> str:
        try:
            with open(self.log_path) as f:
                return f.read()
        except OSError:
            return ""

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if getattr(self, "_log", None):
            self._log.close()
            self._log = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
