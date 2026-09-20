"""Injectable clock so tests and notebooks can fast-forward time (lease expiry,
poll back-off, approval timeouts) without sleeping."""

from __future__ import annotations

import time


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def real_clock() -> float:
    return time.time()
