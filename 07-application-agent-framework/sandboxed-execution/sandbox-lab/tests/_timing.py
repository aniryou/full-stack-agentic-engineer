"""Budgets and CPU contention for the tests that race CPU seconds against the wall clock.

A busy loop accrues CPU time only while it is scheduled. On a shared CI runner (two vCPUs and
other jobs) or a loaded laptop it can get half a core or less, so a loop with a 1 s CPU budget
needs 2 s of wall time or more: with ``wall_s = 2 * cpu_s`` it sometimes reaches the wall clock
first and reports ``wall_timeout`` where the test expects ``cpu_time``. Every test that expects
``cpu_time`` from a busy loop uses ``CPU_BURN``, whose wall limit is ten times its CPU budget: the
verdict flips only if the loop gets less than a tenth of a core. ``cpu_contention()`` is how the
suite checks that on any machine.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager

from sandboxlab.process import Budgets

CPU_BURN = Budgets(cpu_s=1, wall_s=10)

_BUSY = "import time\nend = time.monotonic() + {s}\nwhile time.monotonic() < end:\n    pass\n"


def can_pin() -> bool:
    return hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")


@contextmanager
def cpu_contention(loops: int = 1, seconds: float = 120.0):
    """Pin this process — and so every sandbox child it starts — to one CPU and run ``loops`` busy
    loops pinned to the same CPU, so a sandboxed busy loop gets at most about ``1 / (loops + 1)`` of
    that core whatever the machine's size (less if other work also lands on it).

    Each loop starts in its own session, like the sandbox child (``start_new_session=True``). With
    ``sched_autogroup`` on, the scheduler shares a CPU between sessions first and between the tasks
    of a session second, and a session's weight on one CPU is diluted by its load on the others: a
    loop left in the test runner's session (next to anything else that session runs) would get a
    small slice of this core and the child most of it. One session per loop makes every loop and
    the child equal peers on this core.

    The loops exit on their own after ``seconds`` even if this process dies, and are killed when the
    block ends; the original affinity is restored."""
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(before)})
    procs = []
    try:
        procs = [subprocess.Popen([sys.executable, "-c", _BUSY.format(s=seconds)], start_new_session=True)
                 for _ in range(loops)]
        yield
    finally:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        os.sched_setaffinity(0, before)
