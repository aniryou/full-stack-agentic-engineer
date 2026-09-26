"""proxy — the egress proxy (``server.py``) and a stand-in upstream API (``stub.py``).

Both are single, standard-library files so the same code runs in-process in a notebook, as a
container entrypoint, and from a ConfigMap on kind or GKE.
"""
from .server import EgressProxy, host_allowed, is_public_ip, redact, split_route
from .stub import StubAPI

__all__ = ["EgressProxy", "StubAPI", "host_allowed", "is_public_ip", "redact", "split_route"]
