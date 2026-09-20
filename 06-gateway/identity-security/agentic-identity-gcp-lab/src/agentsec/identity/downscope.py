"""Credential Access Boundaries — "give this tool call access to exactly this prefix, briefly".

Google Cloud lets a broker take a normal OAuth access token and *downscope* it with a
Credential Access Boundary (CAB): a list of rules, each naming an available resource, the
permissions (as ``inRole:`` roles) and an optional CEL availability condition. Today CABs
support Cloud Storage only. The result is a short-lived token that cannot do more than the
boundary allows, no matter how broad the source credential was — ideal for handing a tool a
credential for one bucket prefix.

:func:`build_boundary` produces the JSON shape the STS accepts (and that
``google.auth.downscoped`` consumes), :func:`downscoped_credentials` wires it into google-auth
when running on GCP, and :class:`BoundaryEvaluator` emulates the check locally so notebooks
can demonstrate an out-of-boundary access being refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundaryRule:
    bucket: str
    roles: tuple[str, ...] = ("roles/storage.objectViewer",)
    prefix: str | None = None  # restricts to objects whose name starts with this prefix

    @property
    def available_resource(self) -> str:
        return f"//storage.googleapis.com/projects/_/buckets/{self.bucket}"

    @property
    def available_permissions(self) -> list[str]:
        return [f"inRole:{r}" for r in self.roles]

    @property
    def availability_condition(self) -> dict[str, str] | None:
        if not self.prefix:
            return None
        return {
            "title": f"prefix:{self.prefix}",
            "expression": (
                f"resource.name.startsWith('projects/_/buckets/{self.bucket}/objects/{self.prefix}')"
            ),
        }

    def to_json(self) -> dict[str, Any]:
        rule: dict[str, Any] = {
            "availableResource": self.available_resource,
            "availablePermissions": self.available_permissions,
        }
        if cond := self.availability_condition:
            rule["availabilityCondition"] = cond
        return rule


@dataclass(frozen=True)
class CredentialAccessBoundary:
    rules: tuple[BoundaryRule, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {"accessBoundary": {"accessBoundaryRules": [r.to_json() for r in self.rules]}}


def build_boundary(*rules: BoundaryRule) -> CredentialAccessBoundary:
    if len(rules) > 10:
        raise ValueError("Credential Access Boundaries support at most 10 rules")
    return CredentialAccessBoundary(rules=tuple(rules))


def downscoped_credentials(
    boundary: CredentialAccessBoundary, source_credentials: Any | None = None
) -> Any:
    """Return ``google.auth.downscoped.Credentials`` for use on GCP (lazy import).

    ``source_credentials`` defaults to Application Default Credentials — on Agent Engine or
    Cloud Run with Agent Identity that *is* the agent's certificate-bound identity.
    """
    from google.auth import default, downscoped  # lazy: only needed on GCP

    if source_credentials is None:
        source_credentials, _ = default()
    rules = []
    for r in boundary.rules:
        cond = None
        if r.availability_condition:
            cond = downscoped.AvailabilityCondition(
                expression=r.availability_condition["expression"],
                title=r.availability_condition["title"],
            )
        rules.append(
            downscoped.AccessBoundaryRule(
                available_resource=r.available_resource,
                available_permissions=r.available_permissions,
                availability_condition=cond,
            )
        )
    cab = downscoped.CredentialAccessBoundary(rules=rules)
    return downscoped.Credentials(
        source_credentials=source_credentials, credential_access_boundary=cab
    )


# --- local emulation -----------------------------------------------------------------------------
_ROLE_PERMISSIONS = {
    "roles/storage.objectViewer": {"storage.objects.get", "storage.objects.list"},
    "roles/storage.objectCreator": {"storage.objects.create"},
    "roles/storage.objectUser": {
        "storage.objects.get",
        "storage.objects.list",
        "storage.objects.create",
        "storage.objects.delete",
    },
    "roles/storage.objectAdmin": {"storage.objects.*"},
}


class BoundaryEvaluator:
    """Emulates how the STS-issued downscoped token would be evaluated by Cloud Storage."""

    def __init__(self, boundary: CredentialAccessBoundary):
        self.boundary = boundary

    @staticmethod
    def _parse(object_uri: str) -> tuple[str, str]:
        m = re.match(r"^gs://([^/]+)/(.*)$", object_uri)
        if not m:
            raise ValueError("expected gs://bucket/object")
        return m.group(1), m.group(2)

    def allows(self, object_uri: str, permission: str) -> bool:
        bucket, name = self._parse(object_uri)
        for rule in self.boundary.rules:
            if rule.bucket != bucket:
                continue
            if rule.prefix and not name.startswith(rule.prefix):
                continue
            perms: set[str] = set()
            for role in rule.roles:
                perms |= _ROLE_PERMISSIONS.get(role, set())
            if permission in perms or "storage.objects.*" in perms:
                return True
        return False
