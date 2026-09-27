"""records.py — one typed record for everything an agent remembers (PRIMER §1).

The one idea: working memory is the context window; everything that outlives a turn is a *record*
with the same fields whatever its kind — what it says, whose it is, where it came from, how far to
trust it, when it was true, when it expires and the key that deletes it. The fields are what later
steps need: scope partitions retrieval (§3), source and trust gate the write path and poisoning
(§2, §8), validity and supersession drive consolidation (§7), the deletion key and provenance carry a
forget to every copy (§7).

Tokens are counted the way 07.2 counts them (``agentlab.llm.types.count_tokens``): ``len(text) // 4``,
at least 1 for non-empty text. That is a stand-in for a tokenizer, used for budgets, not billing.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

KINDS = ("episodic", "semantic", "procedural")          # working memory is the context, not a record
SCOPES = ("user", "session", "agent", "tenant")          # who may read it, inside the (tenant, user) partition
SOURCES = ("human", "user", "tool", "inferred", "consolidation")
# Precedence when two sources disagree (PRIMER §7): a human reviewer > the user's own words > a tool's
# output > the model's inference. A consolidated fact carries the trust of its strongest evidence.
TRUST_ORDER = {"human": 3, "user": 2, "tool": 1, "inferred": 0}
STATUSES = ("active", "quarantined", "superseded", "flagged", "expired")


def count_tokens(text: str) -> int:
    """07.2's stand-in tokenizer: 0 for empty text, else ``max(1, len(text) // 4)``."""
    return 0 if not text else max(1, len(text) // 4)


def new_id() -> str:
    return "mem_" + uuid.uuid4().hex[:16]


def content_hash(*parts: Any) -> str:
    """sha256 of canonical JSON, first 16 hex — the identity lab's ``args_digest`` recipe."""
    canonical = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class MemoryRecord:
    tenant: str
    user: str
    text: str                          # what is embedded, indexed and shown to the model
    kind: str = "semantic"             # episodic | semantic | procedural
    scope: str = "user"                # user | session | agent | tenant
    source: str = "user"               # human | user | tool | inferred | consolidation
    trust: str = "user"                # human | user | tool | inferred: what precedence uses
    session: str | None = None
    agent: str | None = None
    slot: str | None = None            # a normalised key ("home_city") so a newer value can supersede
    value: str | None = None
    provenance: list[str] = field(default_factory=list)   # turn refs or record ids this came from
    confidence: float = 1.0
    importance: float = 5.0            # 1..10, as the generative-agents paper rates poignancy
    created_at: float = 0.0            # system time (when we learned it)
    valid_from: float = 0.0            # valid time (when it became true)
    valid_to: float | None = None      # valid time end: plays Graphiti's ``invalid_at``
    superseded_at: float | None = None  # system time it stopped being current: Graphiti's ``expired_at``
    ttl_s: float | None = None
    last_accessed: float = 0.0
    deletion_key: str = ""             # what a forget request names; shared by every copy of a fact
    status: str = "active"
    id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.trust not in TRUST_ORDER:
            raise ValueError(f"trust must be one of {tuple(TRUST_ORDER)}, got {self.trust!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if not self.deletion_key:
            self.deletion_key = default_deletion_key(self.tenant, self.user, self.slot or self.id)
        if not self.valid_from:
            self.valid_from = self.created_at
        if not self.last_accessed:
            self.last_accessed = self.created_at

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def valid_at(self, t: float) -> bool:
        """True when the fact held at valid time ``t`` (an as-of filter, PRIMER §3)."""
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)

    def expired(self, now: float) -> bool:
        return self.ttl_s is not None and now >= self.created_at + self.ttl_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_deletion_key(tenant: str, user: str, subject: str) -> str:
    """One key per (tenant, user, subject): forgetting "my address" names every copy of it."""
    return f"{tenant}/{user}/{subject}"


def outranks(a: str, b: str) -> bool:
    """True when trust ``a`` strictly outranks trust ``b`` (human > user > tool > inferred)."""
    return TRUST_ORDER[a] > TRUST_ORDER[b]
