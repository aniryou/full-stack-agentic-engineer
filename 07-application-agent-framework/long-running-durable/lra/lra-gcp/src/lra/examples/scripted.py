"""Scripted model answers for the example workflows (used by tests, notebooks and the demo)."""

from __future__ import annotations

import json
from typing import Any


def research_routes(*, first_score: int = 6, second_score: int = 9, n_subtasks: int = 3) -> dict[str, Any]:
    subtasks = [{"title": f"Subtopic {i}", "instructions": f"look into area {i}"} for i in range(n_subtasks)]
    return {
        r"Break the GOAL": json.dumps({"subtasks": subtasks}),
        r"Research this subtopic": lambda p: "- finding 1\n- finding 2\n- finding 3",
        r"Write a concise report": "Draft v1 of the report.",
        r"rigorous reviewer": lambda p: json.dumps(
            {"score": first_score if "v1" in p and "revised" not in p else second_score, "fixes": ["add citations"], "verdict": "ok"}
        ),
        r"Improve the DRAFT": "Draft v2 of the report (revised).",
    }
