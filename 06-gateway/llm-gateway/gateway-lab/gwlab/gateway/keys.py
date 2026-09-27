"""Virtual keys for callers, provider keys for upstreams — and why the two never meet.

The one idea (PRIMER §6 Keys, tenants and isolation): a caller holds a *virtual key* the gateway issued —
hashed at rest (SHA-256, as LiteLLM's `hash_token` does), scoped to aliases, budgeted, revocable in one call.
The tenant is read from the verified key, never from a header the caller could set. The *provider* keys live
only in the gateway (the identity primer §5's gateway path; the sandbox primer §4's egress proxy is the same
pattern): a leaked virtual key is revoked in a second, a leaked provider key is a provider incident.

Provider-key rotation with overlap: the provider accepts old and new for a while, the gateway switches to the
new one, then the old one is retired at the provider. `ProviderKeys.rotate()` records both so the switch is
observable.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass

PREFIX = "gwk_"


def new_secret() -> str:
    return PREFIX + secrets.token_urlsafe(24)


def hash_key(secret: str) -> str:
    """What the gateway stores instead of the key: SHA-256, hex."""
    return hashlib.sha256(secret.encode()).hexdigest()


def display(secret: str) -> str:
    """Enough to recognise a key in a list, not enough to use it."""
    return f"{secret[:8]}…{secret[-4:]}"


def bearer(headers) -> str | None:
    """The token from `Authorization: Bearer <token>` (case-insensitive scheme), or None."""
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    scheme, _, token = auth.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class KeyError401(Exception):
    """The caller's key is missing, unknown, expired or revoked (an HTTP 401)."""


@dataclass
class VirtualKey:
    key_id: str
    tenant: str
    display: str
    aliases: list | None
    rpm: int | None
    tpm: int | None
    budget_usd: float | None
    spent_usd: float
    created: float
    expires_at: float | None
    revoked_at: float | None

    def allows(self, alias: str) -> bool:
        return self.aliases is None or alias in self.aliases

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class KeyStore:
    def __init__(self, store, clock=time.time):
        self.store, self.clock = store, clock

    def issue(self, tenant: str, aliases=None, rpm=None, tpm=None, budget_usd=None, ttl_s=None) -> tuple[str, VirtualKey]:
        """Create a key; the secret is returned once and never stored."""
        secret, now = new_secret(), self.clock()
        key_id = "key_" + uuid.uuid4().hex[:10]
        self.store.execute(
            "INSERT INTO keys (key_id, key_hash, display, tenant, aliases, rpm, tpm, budget_usd, created, expires_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (key_id, hash_key(secret), display(secret), tenant, json.dumps(aliases) if aliases is not None else None,
             rpm, tpm, budget_usd, now, now + ttl_s if ttl_s else None))
        return secret, self.get(key_id)

    def _row(self, r: dict) -> VirtualKey:
        return VirtualKey(key_id=r["key_id"], tenant=r["tenant"], display=r["display"],
                          aliases=json.loads(r["aliases"]) if r["aliases"] else None, rpm=r["rpm"], tpm=r["tpm"],
                          budget_usd=r["budget_usd"], spent_usd=r["spent_usd"], created=r["created"],
                          expires_at=r["expires_at"], revoked_at=r["revoked_at"])

    def get(self, key_id: str) -> VirtualKey:
        rows = self.store.query("SELECT * FROM keys WHERE key_id = ?", (key_id,))
        if not rows:
            raise KeyError(key_id)
        return self._row(rows[0])

    def verify(self, secret: str | None) -> VirtualKey:
        """The key behind a presented secret, or KeyError401 with the reason."""
        if not secret:
            raise KeyError401("missing bearer token")
        rows = self.store.query("SELECT * FROM keys WHERE key_hash = ?", (hash_key(secret),))
        if not rows:
            raise KeyError401("unknown key")
        k = self._row(rows[0])
        if k.revoked_at is not None:
            raise KeyError401("key revoked")
        if k.expires_at is not None and self.clock() >= k.expires_at:
            raise KeyError401("key expired")
        return k

    def revoke(self, key_id: str) -> bool:
        return self.store.execute("UPDATE keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                                  (self.clock(), key_id)) == 1

    def add_spend(self, key_id: str, usd: float) -> None:
        self.store.execute("UPDATE keys SET spent_usd = spent_usd + ? WHERE key_id = ?", (usd, key_id))

    def list(self, tenant: str | None = None) -> list[VirtualKey]:
        sql, args = ("SELECT * FROM keys WHERE tenant = ? ORDER BY created", (tenant,)) if tenant else \
            ("SELECT * FROM keys ORDER BY created", ())
        return [self._row(r) for r in self.store.query(sql, args)]


class ProviderKeys:
    """The provider credentials, held only here. `rotate()` switches to a new key and remembers the old one
    until `retire()` — the overlap window during which the provider must accept both."""

    def __init__(self, keys: dict):
        self.current = dict(keys)
        self.previous: dict = {}
        self.rotations: list = []

    def get(self, provider: str) -> str:
        return self.current.get(provider, "")

    def rotate(self, provider: str, new_key: str) -> None:
        self.previous[provider] = self.current.get(provider, "")
        self.current[provider] = new_key
        self.rotations.append((time.time(), provider, "rotated"))

    def retire(self, provider: str) -> str | None:
        self.rotations.append((time.time(), provider, "retired"))
        return self.previous.pop(provider, None)

    def fingerprint(self, provider: str) -> str:
        """A short hash to show *which* key is in use without showing the key."""
        return hashlib.sha256(self.get(provider).encode()).hexdigest()[:8]
