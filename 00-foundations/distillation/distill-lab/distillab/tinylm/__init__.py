"""tinylm — a tiny teacher and a narrower student, trained four ways (torch is imported lazily).

``task`` is pure Python (problems, the two demonstration mixes, the verifier); ``model`` and ``train`` need
torch; ``curves`` loads a recorded run for machines without torch and prints any run.
"""
from .task import SumTask, Problem, data_mix, parse, render, teacher_mix, verify  # noqa: F401

__all__ = ["SumTask", "Problem", "data_mix", "parse", "render", "teacher_mix", "verify"]
