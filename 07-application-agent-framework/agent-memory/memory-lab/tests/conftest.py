"""Shared fixtures: a virtual clock and temporary stores. Every test is offline (127.0.0.1 only)."""
from __future__ import annotations

import pytest

from memlab.extract import ts_of
from memlab.store import SQLiteMemoryStore


class Clock:
    def __init__(self, t: float = ts_of("2026-09-20")):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    s = SQLiteMemoryStore(tmp_path / "memory.db", clock=clock)
    yield s
    s.close()


@pytest.fixture
def mem_store(clock):
    s = SQLiteMemoryStore(":memory:", clock=clock)
    yield s
    s.close()
