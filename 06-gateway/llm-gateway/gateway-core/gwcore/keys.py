"""Keys, tenants and isolation (PRIMER §6).

The one idea: the tenant is whatever a *verified* key says -- never a header the caller sets -- and
every tenant-scoped thing (limits, cache namespace, the engine's ``cache_salt``, ledger rows, traces)
derives from that one identity. Virtual keys are stored hashed, scoped, budgeted and revocable;
provider keys live only in the gateway and rotate with an overlap; the gateway's own workload
identity (an X.509-SVID) rotates at half its life.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import math
import random
from collections import defaultdict
from dataclasses import dataclass


def key_hash(presented: str) -> str:
    return hashlib.sha256(presented.encode()).hexdigest()          # at rest only the hash (as LiteLLM's hash_token)


@dataclass
class VirtualKey:
    key_id: str
    tenant: str
    key_hash: str
    models: frozenset | None = None        # None = every model the tenant's route allows
    max_budget: float | None = None        # dollars
    tpm: int | None = None
    tier: str = "standard"
    spent: float = 0.0
    revoked: bool = False


class KeyStore:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.by_hash: dict = {}

    def issue(self, tenant: str, **scope) -> str:
        """Returns the plaintext key once; only its hash is kept."""
        plain = "sk-gw-" + "".join(self.rng.choice("abcdefghijkmnopqrstuvwxyz23456789") for _ in range(32))
        h = key_hash(plain)
        models = scope.pop("models", None)
        self.by_hash[h] = VirtualKey(h[:12], tenant, h, frozenset(models) if models else None, **scope)
        return plain

    def verify(self, presented: str) -> VirtualKey:
        vk = self.by_hash.get(key_hash(presented or ""))
        if vk is None or vk.revoked:
            raise PermissionError("invalid or revoked key")
        return vk

    def authorize(self, vk: VirtualKey, model: str) -> tuple:
        if vk.models is not None and model not in vk.models:
            return False, f"key not scoped to {model}"
        if vk.max_budget is not None and vk.spent >= vk.max_budget:
            return False, "budget exhausted"
        return True, ""

    def charge(self, vk: VirtualKey, cost: float) -> None:
        vk.spent += cost

    def revoke(self, key_id: str) -> None:
        for vk in self.by_hash.values():
            vk.revoked = vk.revoked or vk.key_id == key_id


class ProviderKeys:
    """Secrets only the gateway holds. Rotate = add the new key; the old one stays valid for ``overlap``
    seconds so in-flight requests and every gateway replica finish on it; then it expires."""

    def __init__(self):
        self.keys = defaultdict(list)            # provider -> [[secret, valid_from, valid_to]]

    def add(self, provider: str, secret: str, now: float, valid_to: float = math.inf) -> None:
        self.keys[provider].append([secret, now, valid_to])

    def rotate(self, provider: str, new_secret: str, now: float, overlap: float = 3600.0) -> None:
        for k in self.keys[provider]:
            k[2] = min(k[2], now + overlap)
        self.add(provider, new_secret, now)

    def valid(self, provider: str, secret: str, now: float) -> bool:
        return any(s == secret and a <= now < b for s, a, b in self.keys[provider])

    def current(self, provider: str, now: float) -> str:
        return max((k for k in self.keys[provider] if k[1] <= now < k[2]), key=lambda k: k[1])[0]


def cache_salt(tenant: str, secret: bytes) -> str:
    """The per-tenant ``cache_salt`` for vLLM: HMAC-SHA256 of the verified tenant, base64url without
    padding -- 43 characters for 256 bits, secret, never taken from the request."""
    return base64.urlsafe_b64encode(hmac.new(secret, tenant.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()


def valid_cache_salt(s: str) -> bool:
    """vLLM v0.30.0 ``validate_cache_salt``: non-empty, at most 128 characters, no ``@ / \\`` or NUL."""
    return 0 < len(s) <= 128 and not set(s) & {"@", "/", "\\", "\x00"}


def svid_rotation_window(ttl: float, jitter: float = 0.1) -> tuple:
    """SPIRE rotates an X.509-SVID when its remaining lifetime falls to half-life +/- 10 % of the half-life
    (``rotationutil.go``): returns (earliest, latest) remaining seconds at which rotation happens."""
    half = ttl / 2
    return half - half * jitter, half + half * jitter


class FakeWorkloadAPI:
    """Stands in for the SPIFFE Workload API's ``FetchX509SVID`` server stream: every message carries the
    *full* state (a missing SVID means it was revoked), and callers must send ``workload.spiffe.io: true``."""

    def __init__(self, spiffe_id: str, ttl: float = 3600.0, seed: int = 0):
        self.spiffe_id, self.ttl, self.rng = spiffe_id, ttl, random.Random(seed)

    def fetch_x509_svid(self, metadata: dict, until: float, start: float = 0.0):
        if metadata.get("workload.spiffe.io") != "true":
            raise PermissionError("missing workload.spiffe.io: true metadata")
        t, serial = start, 1
        while t < until:
            yield {"at": t, "svids": [{"spiffe_id": self.spiffe_id, "serial": serial, "not_after": t + self.ttl}]}
            lo, hi = svid_rotation_window(self.ttl)
            t, serial = t + self.ttl - self.rng.uniform(lo, hi), serial + 1   # rotate at the jittered half-life
