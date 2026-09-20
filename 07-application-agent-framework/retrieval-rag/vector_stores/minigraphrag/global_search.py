"""Global search -- answer whole-corpus questions by map-reduce over reports.

Good for "what are the main themes?", "summarize the key players", "what is
this corpus about?". Local search can't answer these: no single entity's
neighbourhood contains the answer. Global search instead reads the *community
reports* -- the pre-computed summaries of the whole graph -- and does map-reduce:

    MAP:    ask the LLM, for each community report independently, which points
            in it bear on the question (+ a relevance score).          # LLM (xN)
    FILTER: drop zero-score points, keep the strongest.
    REDUCE: ask the LLM to synthesize one answer from the surviving points. # LLM

Because it consults summaries rather than raw text, one pass covers the entire
corpus regardless of size.

PERF: the MAP step is embarrassingly parallel -- real GraphRAG batches several
reports per call and runs the calls concurrently. We loop sequentially. Choosing
the community *level* trades breadth (coarse, level 0) against detail (fine).
"""

import json
from dataclasses import dataclass, field


def _parse_points(raw):
    """Parse the map step's JSON into a list of point dicts.

    Tolerant by design: prose around the object, a bare top-level array of
    points, or stray non-dict entries all degrade to a clean list rather than
    crashing (a real model may not emit exactly ``{"points": [...]}``).
    """
    for candidate in (raw, raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw else ""):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            pts = data.get("points", [])
            return [p for p in pts if isinstance(p, dict)] if isinstance(pts, list) else []
        if isinstance(data, list):  # tolerate a top-level array of points
            return [p for p in data if isinstance(p, dict)]
    return []


@dataclass
class GlobalSearchResult:
    answer: str = ""
    points: list = field(default_factory=list)  # list of (community_id, title, description, score)
    level: int = 0
    reports_considered: int = 0


class GlobalSearch:
    """Map-reduce answering over community reports of a :class:`GraphRAGIndex`."""

    def __init__(self, index, level=0, min_score=1, max_points=20):
        self.index = index
        self.level = level          # 0 = coarsest (broad themes); higher = finer
        self.min_score = min_score
        self.max_points = max_points

    def search(self, query):
        idx = self.index
        reports = idx.reports_at_level(self.level)

        # MAP: score each report against the question.
        scored = []  # (score, community_id, title, description)
        for rep in reports:
            raw = idx.llm.global_map(  # LLM
                query, rep.community_id, rep.full_content,
                report_summary=rep.summary, report_rank=rep.rank,
            )
            for p in _parse_points(raw):
                try:
                    score = float(p.get("score", 0))
                except (TypeError, ValueError):
                    score = 0.0
                desc = str(p.get("description", "")).strip()
                if score >= self.min_score and desc:
                    scored.append((score, rep.community_id, rep.title, desc))

        # FILTER: keep the strongest points.
        scored.sort(key=lambda t: -t[0])
        scored = scored[:self.max_points]

        # REDUCE: synthesize a final answer.
        point_texts = [f"[{title}] {desc}" for _, _, title, desc in scored]
        if point_texts:
            answer = idx.llm.global_reduce(query, point_texts)  # LLM
        else:
            answer = "The community reports do not contain enough information to answer this question."

        points = [(cid, title, desc, score) for score, cid, title, desc in scored]
        return GlobalSearchResult(
            answer=answer, points=points, level=self.level, reports_considered=len(reports),
        )
