"""Admission control: the brake on the overload feedback loop.

    more load → 429s → retries and longer turns → more turns in flight → more 429s …

The controller looks at three signals — turns in flight, the share of recent model calls
that were rate-limited, and how long the oldest queued turn has waited — and maps them to a
degrade level:

    0  normal
    1  cheaper model, shorter answers, optional tools off
    2  cheapest model, no writes and no slow tools
    3  shed: refuse new turns with a Retry-After (priority traffic still gets in)

Levels 1–2 are held for ``dwell`` seconds so the system does not flap; level 3 follows the
instantaneous cap because it must switch off the moment turns drain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .clock import CLOCK


@dataclass
class AdmissionConfig:
    max_inflight: int = 85               # from the token budget: see capacity.plan()["concurrency"]
    soft_ratio: float = 0.8              # level 1 above this share of the cap
    rate_limited_degrade: float = 0.05   # level 1 at 5 % 429s, level 2 at 15 %
    queue_age_degrade_s: float = 10.0
    queue_age_shed_s: float = 30.0
    dwell_s: float = 15.0


class Decision:
    def __init__(self, admitted: bool, level: int, retry_after: float | None = None):
        self.admitted, self.level, self.retry_after = admitted, level, retry_after


class AdmissionController:
    def __init__(self, cfg: AdmissionConfig = AdmissionConfig()):
        self.cfg = cfg
        self.inflight = 0
        self.level = 0
        self._raised_at = -1e9
        self._recent: deque[tuple[float, bool]] = deque()   # (time, was_rate_limited)
        self.stats = {"admitted": 0, "shed": 0}

    # --- signals -------------------------------------------------------------------
    def note_model_call(self, rate_limited: bool) -> None:
        self._recent.append((CLOCK.now(), rate_limited))

    def rate_limited_ratio(self, window: float = 60.0) -> float:
        cutoff = CLOCK.now() - window
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        return sum(1 for _, r in self._recent if r) / len(self._recent) if self._recent else 0.0

    # --- the level ------------------------------------------------------------------
    def compute_level(self, queue_age_s: float = 0.0, circuit_open: bool = False) -> int:
        c, ratio = self.cfg, self.rate_limited_ratio()
        raw = 0
        if self.inflight >= c.max_inflight * c.soft_ratio or ratio >= c.rate_limited_degrade or queue_age_s >= c.queue_age_degrade_s:
            raw = 1
        if ratio >= 3 * c.rate_limited_degrade or circuit_open:
            raw = 2
        shed = self.inflight >= c.max_inflight or queue_age_s >= c.queue_age_shed_s
        held = min(self.level, 2) if CLOCK.now() - self._raised_at < c.dwell_s else 0
        level = 3 if shed else max(raw, held)
        if level > self.level:
            self._raised_at = CLOCK.now()
        self.level = level
        return level

    # --- admit / release --------------------------------------------------------------
    def admit(self, priority: int = 5, queue_age_s: float = 0.0, circuit_open: bool = False) -> Decision:
        level = self.compute_level(queue_age_s, circuit_open)
        if level >= 3 and priority > 2:
            self.stats["shed"] += 1
            return Decision(False, level, retry_after=5 + 5 * level)
        self.inflight += 1
        self.stats["admitted"] += 1
        return Decision(True, min(level, 2))   # an admitted turn runs at level ≤ 2

    def release(self) -> None:
        self.inflight = max(0, self.inflight - 1)
