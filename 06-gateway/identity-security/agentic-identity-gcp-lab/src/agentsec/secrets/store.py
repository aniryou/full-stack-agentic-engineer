"""Secret handling: values that never print themselves, and stores that never keep keys in code.

Rules encoded here:

* A :class:`SecretValue` is opaque — ``repr``/``str``/JSON never reveal it; call ``reveal()`` at
  the single point of use. This makes accidental logging (and the model "seeing" a secret via a
  tool result) a type error rather than a habit.
* Stores expose ``get(name) -> SecretValue``. :class:`LocalSecretStore` reads from environment
  variables or an explicit mapping (dev only). :class:`SecretManagerStore` reads
  ``projects/P/secrets/NAME/versions/latest`` with the agent's own identity — the agent principal
  needs ``roles/secretmanager.secretAccessor`` on *that* secret only.
* Secrets are for infrastructure (DB passwords, a webhook signing key). Credentials the agent
  uses *to call tools* belong in Auth Manager (see ``identity.auth_manager``), not here.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SecretValue:
    """An opaque secret. ``reveal()`` is the only way to get the plaintext."""

    _value: str
    name: str = "secret"

    def reveal(self) -> str:
        return self._value

    @property
    def fingerprint(self) -> str:
        """Safe-to-log identifier: first 8 hex chars of SHA-256."""
        return hashlib.sha256(self._value.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return f"SecretValue(name={self.name!r}, fingerprint={self.fingerprint})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:  # constant-time compare for equality checks
        if not isinstance(other, SecretValue):
            return NotImplemented
        import hmac

        return hmac.compare_digest(self._value.encode(), other._value.encode())

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # Never let pydantic/json serialise the plaintext by accident.
    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("SecretValue must not be pickled/serialised")


class SecretNotFound(KeyError):
    pass


class SecretStore(Protocol):
    def get(self, name: str) -> SecretValue: ...


class LocalSecretStore:
    """Development store: explicit mapping first, then ``AGENTSEC_SECRET_<NAME>`` env vars."""

    def __init__(
        self, values: Mapping[str, str] | None = None, env_prefix: str = "AGENTSEC_SECRET_"
    ):
        self._values = dict(values or {})
        self._env_prefix = env_prefix
        self.access_log: list[str] = []

    def get(self, name: str) -> SecretValue:
        self.access_log.append(name)
        if name in self._values:
            return SecretValue(self._values[name], name=name)
        env_key = self._env_prefix + name.upper().replace("-", "_")
        if env_key in os.environ:
            return SecretValue(os.environ[env_key], name=name)
        raise SecretNotFound(name)


class SecretManagerStore:
    """Google Cloud Secret Manager, using Application Default Credentials (the agent's identity)."""

    def __init__(self, project: str, *, client: Any | None = None):
        self.project = project
        self._client = client

    def _get_client(self):
        if self._client is None:
            from google.cloud import secretmanager  # lazy

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get(self, name: str, version: str = "latest") -> SecretValue:
        path = f"projects/{self.project}/secrets/{name}/versions/{version}"
        resp = self._get_client().access_secret_version(request={"name": path})
        return SecretValue(resp.payload.data.decode("utf-8"), name=name)
