"""A virtual clock.

Every sleep and every timestamp in the lab goes through ``CLOCK`` so a simulation can run
much faster than real time without changing any ratio: with ``time_scale = 0.02`` a 6-second
turn takes 120 ms of wall-clock and still *measures* as 6 s.

On an ordinary event loop the clock follows the wall clock, so how busy the machine is leaks into
the results a little (asyncio wakes a task late when the CPU is contended). ``run_in_virtual_time``
removes that: it runs a coroutine on an event loop whose time only moves when every task is
waiting, so sleeps take no wall-clock time and a seeded simulation gives the same numbers on every
run and every machine. The tests use it; the notebooks keep the wall clock.
"""

from __future__ import annotations

import asyncio
import selectors
import time
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def _loop_time() -> float:
    """The running event loop's time (``time.monotonic()`` on a normal loop), else the monotonic clock."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()


class VirtualClock:
    def __init__(self, time_scale: float = 1.0):
        self.time_scale = time_scale  # wall-clock seconds per virtual second
        self._start = _loop_time()

    def now(self) -> float:
        """Virtual seconds since the clock was created."""
        return (_loop_time() - self._start) / self.time_scale

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds * self.time_scale)

    def reset(self, time_scale: float | None = None) -> None:
        if time_scale is not None:
            self.time_scale = time_scale
        self._start = _loop_time()


CLOCK = VirtualClock()  # the one clock everything shares; call CLOCK.reset(0.02) before a simulation


class _AdvancingSelector(selectors.BaseSelector):
    """A real selector that never blocks: where the loop would wait ``timeout`` seconds for the next
    timer, it moves the loop's virtual time forward by ``timeout`` instead."""

    def __init__(self, loop: "VirtualTimeLoop"):
        self._real = selectors.DefaultSelector()
        self._loop = loop

    def register(self, fileobj, events, data=None):
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._real.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._real.modify(fileobj, events, data)

    def select(self, timeout=None):
        ready = self._real.select(0)
        if not ready and timeout is not None and timeout > 0:
            self._loop.virtual_now += timeout
        return ready

    def get_map(self):
        return self._real.get_map()

    def close(self):
        self._real.close()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose ``time()`` advances only when every task is asleep."""

    def __init__(self) -> None:
        self.virtual_now = 0.0
        super().__init__(selector=_AdvancingSelector(self))

    def time(self) -> float:
        return self.virtual_now


def run_in_virtual_time(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on a :class:`VirtualTimeLoop` (call ``CLOCK.reset`` inside it)."""
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
