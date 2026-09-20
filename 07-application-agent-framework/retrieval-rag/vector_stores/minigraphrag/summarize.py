"""Community report generation -- summarize each community with the LLM.

For every community, we gather its entities and internal relationships, hand
them to the LLM, and get back a structured report (title, summary, findings,
importance rating). These reports are what global search reads, and they are
also embedded for retrieval.

PERF: one LLM call per community, run sequentially. Real GraphRAG runs these
concurrently and, for large communities, summarizes *bottom-up* -- a coarse
community's report is built from its sub-community reports rather than from all
its raw entities, so the prompt never overflows the context window.
"""

import json

from .models import CommunityReport


def _parse_report_json(raw):
    """Parse the LLM's report JSON into a dict, tolerating prose around it.

    Always returns a dict so callers can ``.get(...)`` safely: a top-level array
    or scalar (a real model may emit one) degrades to ``{}`` rather than crashing.
    """
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    # Real models sometimes wrap JSON in prose/markdown fences; grab the
    # outermost {...} and retry. (Structured outputs make this unnecessary.)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _community_payload(community, graph):
    """Build the entity/relationship dicts (sorted by importance) for the prompt."""
    entities = sorted(
        (graph.entities[eid] for eid in community.entity_ids if eid in graph.entities),
        key=lambda e: -e.rank,
    )
    ent_dicts = [
        {"name": e.name, "type": e.type, "description": e.description, "rank": e.rank}
        for e in entities
    ]
    rel_dicts = []
    for rid in community.relationship_ids:
        # relationship ids look like "A__B"; look up via endpoints.
        for (u, v), r in graph.relationships.items():
            if r.id == rid:
                rel_dicts.append({
                    "source": graph.entities[u].name, "target": graph.entities[v].name,
                    "description": r.description, "weight": r.weight,
                })
                break
    title_hint = ent_dicts[0]["name"] if ent_dicts else f"Community {community.id}"
    return title_hint, ent_dicts, rel_dicts


def generate_reports(communities, graph, llm, on_progress=None):
    """Produce a :class:`CommunityReport` for each community. Returns a dict by id."""
    reports = {}
    n = len(communities)
    for i, community in enumerate(communities):
        title_hint, ent_dicts, rel_dicts = _community_payload(community, graph)
        # LLM: author the report (returns JSON text; parsed below).
        raw = llm.community_report(title_hint, ent_dicts, rel_dicts)
        data = _parse_report_json(raw)

        findings = data.get("findings", []) or []
        # Normalize findings into {summary, explanation} dicts.
        norm_findings = []
        for f in findings:
            if isinstance(f, dict):
                norm_findings.append({
                    "summary": str(f.get("summary", "")),
                    "explanation": str(f.get("explanation", "")),
                })
            else:
                norm_findings.append({"summary": str(f), "explanation": ""})

        report = CommunityReport(
            community_id=community.id,
            level=community.level,
            title=str(data.get("title", title_hint)),
            summary=str(data.get("summary", "")),
            findings=norm_findings,
            rank=float(data.get("rating", 0.0) or 0.0),
        )
        report.full_content = report.as_context()
        reports[community.id] = report
        if on_progress:
            on_progress(i + 1, n)
    return reports
