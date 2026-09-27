"""retrieve.py - rank memories by similarity, recency and importance, then pack them into a token budget.

The one idea: similarity alone retrieves the most *on-topic* memory, not the most *useful* one. The
generative-agents score (Park et al. 2023) adds how recently a memory was used and how much it matters:

    score = w_recency * recency + w_importance * importance + w_relevance * relevance,

each term min-max normalised to [0, 1] over the candidates (all equal -> 0.5 each). It exists in two forms,
and they disagree:

- form="paper" (the paper's description, unverified here: arXiv is unreachable) - all weights 1, recency =
  0.995 ** (hours since the memory was last accessed);
- form="code" (joonspk-research/generative_agents, retrieve.py, read 2026-09-26) - weights recency 0.5,
  relevance 3, importance 2 (`gw = [0.5, 3, 2]`), and recency = 0.99 ** rank over the memories sorted by
  last access OLDEST FIRST, so the stalest memory gets the largest recency term. Rank-based: an hour and a
  year apart score alike.

Retrieval then packs the best records into the per-turn token budget, and - as in the reference code - a read
is a write: the returned records' last_accessed moves to now.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .records import HOUR, MemoryRecord, Scope

WEIGHTS = {"paper": (1.0, 1.0, 1.0), "code": (0.5, 2.0, 3.0)}      # (recency, importance, relevance)
DECAY = {"paper": 0.995, "code": 0.99}


def minmax(x) -> np.ndarray:
    """generative_agents' normalize_dict_floats(d, 0, 1): (v - min) / range; every value 0.5 if range == 0."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    span = x.max() - x.min()
    return np.full_like(x, 0.5) if span == 0 else (x - x.min()) / span


def recency(last_accessed, now: float, form: str = "paper") -> np.ndarray:
    """Raw recency before normalisation. paper: decay per hour since last access. code: decay ** rank, oldest first."""
    t = np.asarray(last_accessed, dtype=float)
    if form == "paper":
        return DECAY["paper"] ** ((now - t) / HOUR)
    ranks = np.empty(len(t))
    ranks[np.argsort(t, kind="stable")] = np.arange(1, len(t) + 1)   # oldest access -> rank 1 -> 0.99 ** 1
    return DECAY["code"] ** ranks


def score(last_accessed, importance, relevance, now: float, form: str = "paper", weights=None) -> np.ndarray:
    """The generative-agents retrieval score for each candidate (higher is better)."""
    wr, wi, wv = weights or WEIGHTS[form]
    return (wr * minmax(recency(last_accessed, now, form)) + wi * minmax(importance)
            + wv * minmax(relevance))


def pack(items, budget_tokens: int, k: int | None = None) -> list:
    """Greedy: take items in the given order while they fit; skip one that does not and keep looking."""
    out, used = [], 0
    for rec in items:
        if k is not None and len(out) >= k:
            break
        if used + rec.tokens <= budget_tokens:
            out.append(rec)
            used += rec.tokens
    return out


@dataclass
class Recall:
    records: list[MemoryRecord]
    scores: list[float]
    tokens: int
    considered: int
    trace: list = field(default_factory=list)        # (id, score, relevance) for every candidate, ranked

    def render(self) -> str:
        return "\n".join(r.render() for r in self.records)


def retrieve(store, scope: Scope, query: str, *, now: float, budget_tokens: int = 200, k: int | None = None,
             form: str = "paper", weights=None, kinds=None, as_of: float | None = None,
             touch: bool = True) -> Recall:
    """Filter (active, not expired, right kind, valid at `as_of`), score, rank, pack into the budget."""
    cands = [r for r in store.records(scope) if not r.expired(now) and (kinds is None or r.kind in kinds)
             and (as_of is None or r.is_valid(as_of))]
    if as_of is not None:        # an as-of query may need a fact that has since been superseded
        cands += [r for r in store.records(scope, status="superseded") if r.is_valid(as_of)
                  and (kinds is None or r.kind in kinds)]
    rel = store.similarity(scope, query, [r.id for r in cands])
    s = score([r.last_accessed for r in cands], [r.importance for r in cands], rel, now, form, weights)
    order = sorted(range(len(cands)), key=lambda i: (-s[i], cands[i].id))
    chosen = pack([cands[i] for i in order], budget_tokens, k)
    if touch:
        for r in chosen:
            r.last_accessed = now
    by_id = {cands[i].id: float(s[i]) for i in order}
    return Recall(chosen, [by_id[r.id] for r in chosen], sum(r.tokens for r in chosen), len(cands),
                  [(cands[i].id, float(s[i]), float(rel[i])) for i in order])
