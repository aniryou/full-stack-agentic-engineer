"""Which tier can this machine run? T0 always; T0 + Docker, T1 (a real vLLM) and the optional extras when present.

The notebooks call `describe()` first and branch on it: every T1 or Docker path detects its prerequisite and,
when absent, prints the exact commands and uses labelled bundled sample output instead of pretending.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from importlib import resources

VLLM_ENV = "GWLAB_VLLM_URL"              # e.g. http://127.0.0.1:8000 -- a `vllm serve` from deploy/any-gpu
GATEWAY_ENV = "GWLAB_URL"                # a gateway already running (deploy/local's compose stack)
SAMPLE_LABEL = "sample output in the documented format (illustrative)"


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def vllm_url() -> str | None:
    return os.environ.get(VLLM_ENV) or None


def has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def describe() -> dict:
    d = {"T0": True, "docker": docker_available(), "vllm_url": vllm_url(), "gateway_url": os.environ.get(GATEWAY_ENV),
         "dpop_signer": "ES256 (cryptography)" if has_cryptography() else "HMAC stand-in (install cryptography for ES256)"}
    print("tiers: T0 yes | docker " + ("yes" if d["docker"] else "no") + " | T1 vLLM " +
          (d["vllm_url"] or f"no (set {VLLM_ENV})") + " | DPoP signer: " + d["dpop_signer"])
    return d


def sample(name: str) -> str:
    """Bundled sample output (`gwlab/data/samples/<name>`), labelled illustrative wherever it is printed."""
    return resources.files("gwlab").joinpath("data", "samples", name).read_text()
