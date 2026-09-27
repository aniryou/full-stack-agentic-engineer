"""Which tier can this machine run? T0 always; T0 + Docker, T1 (a real vLLM) and the optional extras when present.

The notebooks call `describe()` first and branch on it: every T1 or Docker path detects its prerequisite and,
when absent, prints the exact commands and uses labelled bundled sample output instead of pretending.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import http.client
import urllib.parse
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


def vllm_answers(url: str, timeout: float = 2.0) -> bool:
    """True if `GET <url>/health` answers 200 within `timeout` seconds (vLLM's health endpoint, never behind --api-key)."""
    try:                                              # http.client, like gwlab.client: no proxy between us and it
        u = urllib.parse.urlsplit(url)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port, timeout=timeout)
        conn.request("GET", u.path.rstrip("/") + "/health")
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except (OSError, ValueError, http.client.HTTPException):
        return False


def vllm_url(quiet: bool = False) -> str | None:
    """The T1 vLLM's URL -- only if `GWLAB_VLLM_URL` is set *and* the server answers. A set variable pointing at a
    stopped server (notebook 02's T1 cell stops it on purpose) is not T1: the notebooks fall back to T0."""
    url = os.environ.get(VLLM_ENV) or None
    if url and not vllm_answers(url):
        if not quiet:
            print(f"{VLLM_ENV}={url} is set but vLLM is not answering on /health: T1 skipped "
                  "(restart it with deploy/any-gpu/serve.sh)")
        return None
    return url


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
