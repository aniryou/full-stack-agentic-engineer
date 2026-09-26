"""Golden sets: the curated cases an agent is measured against (notebook 08).

A golden case pins down an *input*, the *trajectory* a correct agent takes
(which tools, roughly which arguments, in which order), what the final answer
must mention, and what must never happen (``forbidden_tools``). Cases carry a
``stratum`` (intent or risk class) and ``tags`` so pass rates are read per
slice — one blended number hides a broken stratum behind ten healthy ones.

Golden sets grow from production: a reviewed transcript becomes a case
(``from_transcripts``), a triaged failure becomes a regression case. That is
the flywheel — trace → triage → golden case → gate — and it only turns if
adding a case is cheap, which is why cases are plain JSON lines.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

Step = tuple[str, dict[str, Any]]


@dataclass
class GoldenCase:
    id: str
    input: str
    expected_tools: list[str] = field(default_factory=list)          # the expected trajectory, in order
    expected_args: dict[str, dict[str, Any]] = field(default_factory=dict)  # tool -> partial args
    expected_answer_contains: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    stratum: str = "default"
    difficulty: str = "medium"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)            # provenance: ticket id, payload, session id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenCase":
        return cls(**d)


class GoldenSet:
    """An ordered collection of cases with unique ids."""

    def __init__(self, cases: Iterable[GoldenCase] = (), name: str = "golden"):
        self.name = name
        self.cases: list[GoldenCase] = []
        for c in cases:
            self.add(c)

    # -- collection basics -----------------------------------------------------
    def add(self, case: GoldenCase) -> GoldenCase:
        if any(c.id == case.id for c in self.cases):
            raise ValueError(f"duplicate case id {case.id!r}")
        self.cases.append(case)
        return case

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[GoldenCase]:
        return iter(self.cases)

    def get(self, case_id: str) -> GoldenCase:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)

    # -- slicing -----------------------------------------------------------------
    def filter(self, predicate: Callable[[GoldenCase], bool], name: str | None = None) -> "GoldenSet":
        return GoldenSet((c for c in self.cases if predicate(c)), name=name or self.name)

    def by_stratum(self) -> dict[str, list[GoldenCase]]:
        out: dict[str, list[GoldenCase]] = {}
        for c in self.cases:
            out.setdefault(c.stratum, []).append(c)
        return out

    def by_tag(self, tag: str) -> "GoldenSet":
        return self.filter(lambda c: tag in c.tags, name=f"{self.name}[{tag}]")

    def strata(self) -> list[str]:
        return list(self.by_stratum())

    def split_holdout(self, fraction: float, seed: int) -> tuple["GoldenSet", "GoldenSet"]:
        """Stratified dev/holdout split. The holdout is never used to tune prompts (notebook 08).

        Deterministic for a given seed so the split can be reproduced in CI.
        """
        if not 0 < fraction < 1:
            raise ValueError("fraction must be strictly between 0 and 1")
        rng = random.Random(seed)
        dev, holdout = [], []
        for stratum_cases in self.by_stratum().values():
            shuffled = list(stratum_cases)
            rng.shuffle(shuffled)
            k = round(fraction * len(shuffled))
            holdout += shuffled[:k]
            dev += shuffled[k:]
        return (GoldenSet(self._in_original_order(dev), name=f"{self.name}-dev"),
                GoldenSet(self._in_original_order(holdout), name=f"{self.name}-holdout"))

    def stratified_sample(self, n_per_stratum: int, seed: int) -> "GoldenSet":
        """Up to ``n_per_stratum`` cases from every stratum — a cheap smoke set with the same shape as the full one."""
        rng = random.Random(seed)
        picked: list[GoldenCase] = []
        for stratum_cases in self.by_stratum().values():
            shuffled = list(stratum_cases)
            rng.shuffle(shuffled)
            picked += shuffled[:n_per_stratum]
        return GoldenSet(self._in_original_order(picked), name=f"{self.name}-sample")

    def _in_original_order(self, subset: Iterable[GoldenCase]) -> list[GoldenCase]:
        ids = {c.id for c in subset}
        return [c for c in self.cases if c.id in ids]

    # -- persistence ---------------------------------------------------------------
    def save_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in self.cases))
        return p

    @classmethod
    def load_jsonl(cls, path: str | Path, name: str | None = None) -> "GoldenSet":
        p = Path(path)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        return cls((GoldenCase.from_dict(json.loads(ln)) for ln in lines), name=name or p.stem)

    # -- reporting -------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cases": len(self.cases),
            "by_stratum": dict(Counter(c.stratum for c in self.cases)),
            "by_difficulty": dict(Counter(c.difficulty for c in self.cases)),
            "tags": dict(Counter(t for c in self.cases for t in c.tags)),
            "with_forbidden_tools": sum(1 for c in self.cases if c.forbidden_tools),
            "total_weight": sum(c.weight for c in self.cases),
        }


def from_transcripts(transcripts: Iterable[tuple[str, list[Step]]], *, stratum: str = "production",
                     tags: Iterable[str] = ("from-production",), id_prefix: str = "prod", start: int = 1,
                     name: str = "from-production") -> GoldenSet:
    """Promote reviewed ``(input, observed trajectory)`` pairs into golden cases.

    The observed trajectory becomes the expected one and the observed arguments
    become the partial-args spec. Only promote transcripts a human has judged
    correct, and prune ``expected_args`` to the keys that matter — a case that
    pins every argument breaks on harmless changes and teaches nothing.
    """
    cases = []
    for i, (user_input, trajectory) in enumerate(transcripts, start):
        expected_args: dict[str, dict[str, Any]] = {}
        for tool_name, args in trajectory:
            expected_args.setdefault(tool_name, dict(args))   # first call of each tool
        cases.append(GoldenCase(
            id=f"{id_prefix}-{i:03d}", input=user_input,
            expected_tools=[t for t, _ in trajectory], expected_args=expected_args,
            tags=list(tags), stratum=stratum,
            metadata={"source": "transcript", "reviewed": True},
        ))
    return GoldenSet(cases, name=name)
