"""Tool policy model: tiers, allowed principals, authority, scopes, constraints, confirmation, egress.

A policy is data (YAML under ``policies/``), reviewed and versioned like IAM. The engine is
deny-by-default: a tool that is not in the policy cannot be called, whatever the model wants.

Tiers mirror MCP tool annotations and VPC-SC's ``mcp.tool.isReadOnly``:

* ``read``        — no side effects
* ``write``       — reversible side effects
* ``destructive`` — irreversible / financial
* ``external``    — leaves the trust boundary (takes a URL, sends a message)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Tier(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"

    @property
    def read_only(self) -> bool:
        return self is Tier.READ


class Authority(str, Enum):
    OWN = "own"
    DELEGATED = "delegated"
    ANY = "any"


class Constraint(BaseModel):
    """Per-argument constraint. All set fields must hold."""

    max: float | None = None
    min: float | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    max_length: int | None = None

    def check(self, value: Any) -> str | None:
        import re

        if self.max is not None and isinstance(value, int | float) and value > self.max:
            return f"> max {self.max}"
        if self.min is not None and isinstance(value, int | float) and value < self.min:
            return f"< min {self.min}"
        if self.enum is not None and value not in self.enum:
            return f"not in {self.enum}"
        if (
            self.pattern is not None
            and isinstance(value, str)
            and not re.fullmatch(self.pattern, value)
        ):
            return f"does not match {self.pattern!r}"
        if self.max_length is not None and isinstance(value, str) and len(value) > self.max_length:
            return f"longer than {self.max_length}"
        return None


class Confirmation(BaseModel):
    required: bool = False
    unless: str | None = None  # safe expression over `args` and `ctx`, e.g. "args.amount <= 50"
    hint: str | None = None


class ToolPolicy(BaseModel):
    tier: Tier
    allow: list[str] = Field(
        default_factory=list
    )  # principal names (from `principals`) or raw members
    authority: Authority = Authority.ANY
    required_scopes: list[str] = Field(default_factory=list)
    constraints: dict[str, Constraint] = Field(default_factory=dict)
    confirmation: Confirmation = Field(default_factory=Confirmation)
    egress_hosts: list[str] = Field(default_factory=list)
    url_args: list[str] = Field(default_factory=lambda: ["url"])
    trust: str = "external"  # provenance trust level for results: internal | external
    description: str | None = None

    @model_validator(mode="after")
    def _defaults_by_tier(self) -> ToolPolicy:
        if (
            self.tier in (Tier.DESTRUCTIVE,)
            and not self.confirmation.required
            and self.confirmation.unless is None
        ):
            # Destructive tools require confirmation unless the policy explicitly says otherwise.
            self.confirmation = Confirmation(required=True, hint=self.confirmation.hint)
        return self


class Budgets(BaseModel):
    max_tool_calls_per_invocation: int = 25
    max_destructive_calls_per_invocation: int = 3


class Policy(BaseModel):
    version: int = 1
    default: str = "deny"
    principals: dict[str, str] = Field(
        default_factory=dict
    )  # alias -> principal:// | principalSet://
    tools: dict[str, ToolPolicy] = Field(default_factory=dict)
    budgets: Budgets = Field(default_factory=Budgets)

    @field_validator("default")
    @classmethod
    def _only_deny(cls, v: str) -> str:
        if v != "deny":
            raise ValueError(
                "policy default must be 'deny' (allow-by-default is not supported on purpose)"
            )
        return v

    def resolve_members(self, aliases: list[str]) -> list[str]:
        return [self.principals.get(a, a) for a in aliases]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        return cls.model_validate(data)
