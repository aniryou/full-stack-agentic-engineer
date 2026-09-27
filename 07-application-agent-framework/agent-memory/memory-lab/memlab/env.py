"""env.py — find out what this machine can run, instead of assuming: the tier, the targets, the SQLite features.

One idea: every notebook runs at T0 and *upgrades* when something real is reachable. Nothing here
changes the machine; each probe is read-only with a short timeout.

    MEMLAB_LLM_URL=http://127.0.0.1:8000      an OpenAI-compatible chat server with tool calls (T1: vLLM)
    MEMLAB_LLM_MODEL=...                      its model id (default: the first of /v1/models)
    MEMLAB_EMBED_URL=http://127.0.0.1:8001    an OpenAI-compatible /v1/embeddings server (T1: vLLM --runner pooling)
    MEMLAB_API_KEY=...                        sent as "Authorization: Bearer ..." to both
    MEMLAB_PG_DSN=postgresql://...            Postgres + pgvector (T0 + Docker: deploy/local/up.sh --pgvector)
    MEMLAB_NO_DOCKER=1                        pretend Docker is absent (to see the T0 path)
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

from .store.sqlite import sqlite_features


def has_docker() -> bool:
    """A docker CLI *and* a reachable daemon."""
    if os.environ.get("MEMLAB_NO_DOCKER") == "1" or not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def on_colab() -> bool:
    return "google.colab" in sys.modules


def is_simulated(url: str, timeout_s: float = 5) -> bool:
    """True when ``url`` answers ``/version`` with ``"simulated": true`` (this lab's fake server)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=timeout_s) as r:
            return bool(json.loads(r.read()).get("simulated"))
    except Exception:
        return False


def reachable(url: str, timeout_s: float = 3) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def llm_url() -> str | None:
    return os.environ.get("MEMLAB_LLM_URL") or None


def embed_url() -> str | None:
    return os.environ.get("MEMLAB_EMBED_URL") or None


def pg_dsn() -> str | None:
    return os.environ.get("MEMLAB_PG_DSN") or None


def describe() -> dict:
    f = sqlite_features()
    return {
        "python": platform.python_version(),
        "sqlite": f["sqlite_version"], "fts5": f["fts5"], "fts5_secure_delete": f["fts5_secure_delete"],
        "numpy": has_module("numpy"), "aiohttp": has_module("aiohttp"), "psycopg": has_module("psycopg"),
        "pglast": has_module("pglast"), "docker": has_docker(), "gcloud": shutil.which("gcloud") is not None,
        "colab": on_colab(), "MEMLAB_LLM_URL": llm_url() or "unset", "MEMLAB_EMBED_URL": embed_url() or "unset",
        "MEMLAB_PG_DSN": "set" if pg_dsn() else "unset",
    }


def tier() -> str:
    """The highest tier this process can measure: T1 when a real model or embedder URL is set, T0 + Docker when
    a daemon is reachable, T0 otherwise."""
    for url in (llm_url(), embed_url()):
        if url and reachable(url) and not is_simulated(url):
            return "T1"
    return "T0 + Docker" if has_docker() else "T0"


def banner() -> str:
    d = describe()
    return (f"tier {tier()} | Python {d['python']} | SQLite {d['sqlite']} (FTS5 {d['fts5']}, secure-delete "
            f"{d['fts5_secure_delete']}) | docker {d['docker']} | psycopg {d['psycopg']} | "
            f"LLM {d['MEMLAB_LLM_URL']} | embedder {d['MEMLAB_EMBED_URL']}")
