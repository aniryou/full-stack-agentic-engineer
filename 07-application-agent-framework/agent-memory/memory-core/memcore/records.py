"""records.py - one typed record for every kind of memory.

The one idea: episodic memory (what happened, timestamped), semantic memory (a distilled fact) and procedural
memory (how to do something, a tool preference) are ONE record type with different fields filled in. Every
record says who it is for (scope), where it came from (source + provenance), how far to trust it (confidence,
source trust), how much it matters (importance), when it is true (valid_from / valid_to), how long to keep it
(ttl_s) and how to find every copy of it when it must go (deletion_key). A memory without those fields cannot
be scoped, ranked, superseded, expired or deleted - it can only be appended to a prompt.

Time is seconds on a scripted clock (DAY = 86,400). Tokens are estimated as len(text) // 4, as the 07.2 lab
counts them (agentlab.llm.types.count_tokens).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

HOUR, DAY = 3600.0, 86400.0
KINDS = ("episodic", "semantic", "procedural")
SOURCE_TRUST = {"human": 3, "user": 2, "tool": 1, "inferred": 0}   # precedence: human > user > tool > inferred
STATUSES = ("active", "quarantined", "superseded", "deleted")


def count_tokens(text: str) -> int:
    """~4 characters per token; 0 for empty text (the 07.2 lab's estimate, restated)."""
    return 0 if not text else max(1, len(text) // 4)


@dataclass(frozen=True)
class Scope:
    """Who a memory belongs to. (tenant, user) is the PARTITION: nothing is ever searched across it."""
    tenant: str
    user: str
    session: str | None = None
    agent: str | None = None

    @property
    def partition(self) -> tuple[str, str]:
        return (self.tenant, self.user)


@dataclass
class MemoryRecord:
    text: str
    kind: str
    scope: Scope
    source: str                         # human | user (a user turn) | tool (tool output) | inferred (a model/consolidation)
    key: str | None = None              # the slot a semantic fact fills ("home_city"); None for free text
    value: str | None = None
    provenance: tuple[str, ...] = ()    # ids of the turns / records it was derived from
    confidence: float = 1.0
    importance: float = 5.0             # 1..10, as the generative-agents "poignancy" rating
    created_at: float = 0.0             # system time: when it was written
    valid_from: float | None = None     # valid time: when it became true (Graphiti's valid_at)
    valid_to: float | None = None       # when it stopped being true (plays Graphiti's invalid_at)
    superseded_at: float | None = None  # system time it was closed (Graphiti's expired_at)
    last_accessed: float | None = None
    ttl_s: float | None = None
    deletion_key: str = ""              # what propagate() follows; defaults to "tenant/user"
    status: str = "active"
    id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if self.source not in SOURCE_TRUST:
            raise ValueError(f"source must be one of {tuple(SOURCE_TRUST)}, not {self.source!r}")
        if self.valid_from is None:
            self.valid_from = self.created_at
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if not self.deletion_key:
            self.deletion_key = "/".join(self.scope.partition)
        if not self.id:
            raw = repr((self.scope.partition, self.kind, self.text, self.created_at))
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())

    @property
    def trust(self) -> int:
        return SOURCE_TRUST[self.source]

    def is_valid(self, as_of: float) -> bool:
        """True if the fact held at `as_of` (valid time), whatever we knew then."""
        return self.valid_from <= as_of and (self.valid_to is None or as_of < self.valid_to)

    def expired(self, now: float) -> bool:
        return self.ttl_s is not None and now >= self.created_at + self.ttl_s

    def render(self) -> str:
        """One line, as it is injected into a prompt: the day it became true is part of the evidence."""
        until = f"-{int(self.valid_to // DAY)}" if self.valid_to is not None else ""
        return f"[{self.kind}, day {int(self.valid_from // DAY)}{until}, {self.source}] {self.text}"

    def copy(self, **changes) -> "MemoryRecord":
        return replace(self, **changes)
