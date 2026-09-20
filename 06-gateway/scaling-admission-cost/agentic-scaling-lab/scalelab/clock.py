"""A virtual clock.

Every sleep and every timestamp in the lab goes through ``CLOCK`` so a simulation can run
much faster than real time without changing any ratio: with ``time_scale = 0.02`` a 6-second
turn takes 120 ms of wall-clock and still *measures* as 6 s.
"""

from __future__ import annotations

import asyncio
import time


class VirtualClock:
    def __init__(self, time_scale: float = 1.0):
        self.time_scale = time_scale  # wall-clock seconds per virtual second
        self._start = time.monotonic()

    def now(self) -> float:
        """Virtual seconds since the clock was created."""
        return (time.monotonic() - self._start) / self.time_scale

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds * self.time_scale)

    def reset(self, time_scale: float | None = None) -> None:
        if time_scale is not None:
            self.time_scale = time_scale
        self._start = time.monotonic()


CLOCK = VirtualClock()  # the one clock everything shares; call CLOCK.reset(0.02) before a simulation
