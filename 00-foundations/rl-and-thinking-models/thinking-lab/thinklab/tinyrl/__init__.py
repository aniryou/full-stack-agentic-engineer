"""tinyrl — GRPO on a tiny transformer that learns to use a scratchpad (torch is imported lazily).

``task`` is pure Python (problems, demonstrations, the verifier); ``model`` and ``train`` need torch;
``curves`` loads a recorded run for machines without torch.
"""
from .task import DigitSum, Problem, parse, render, reward  # noqa: F401
