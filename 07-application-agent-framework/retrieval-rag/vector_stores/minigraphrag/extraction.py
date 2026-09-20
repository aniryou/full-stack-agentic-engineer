"""Extraction -- turn text units into graph elements.

Two responsibilities:

1. **Parse** the delimited records the extraction LLM emits (the format defined
   in ``prompts.py``). The same parser handles ``MockLLM`` output and real model
   output, because both speak the same format.
2. **Merge** the per-chunk extractions into one :class:`KnowledgeGraph`: the
   same entity mentioned in five chunks becomes one node whose descriptions are
   combined (and, past a threshold, summarized by the LLM).

PERF: this loop calls the LLM once per text unit, strictly sequentially. Real
GraphRAG runs the extraction calls concurrently (bounded worker pool), often
does multiple "gleaning" passes per chunk to catch missed entities, and caches
results so re-runs are free. See the ``# PERF:`` marks below.
"""

import re

from .models import Entity, Relationship, normalize_name
from .prompts import TUPLE_DELIMITER, RECORD_DELIMITER, COMPLETION_DELIMITER

# Zero-width split point placed BEFORE each record start -- a "(" that opens an
# ("entity"...) / ("relationship"...) tuple. Used to split records when the model
# omits the "##" separator; the lookahead keeps the "(" attached to the record.
_RECORD_SPLIT_RE = re.compile(r'(?=\(\s*"(?:entity|relationship)")', re.IGNORECASE)


def _record_payload(segment):
    """Return the text between a record's OUTER parens, tolerating inner parens.

    A description is a whole sentence and may itself contain "(...)", so we take
    everything from the first "(" to the LAST ")" rather than the first balanced
    pair -- otherwise a parenthetical like "(a startup)" would be mistaken for
    the record body and the real record silently dropped.
    """
    a = segment.find("(")
    b = segment.rfind(")")
    return segment[a + 1:b] if a != -1 and b > a else None


def parse_records(raw):
    """Parse a raw extraction completion into ``(entities, relationships)``.

    Tolerant by design: unparseable records are skipped rather than raising,
    because LLM output is never perfectly formatted. Returns lists of plain
    dicts (not dataclasses) keyed by surface name.
    """
    raw = raw.split(COMPLETION_DELIMITER)[0]
    entities, relationships = [], []

    # Prefer the explicit record delimiter; fall back to splitting at each
    # record start so we still work if the model forgot the "##" separators.
    if RECORD_DELIMITER in raw:
        segments = raw.split(RECORD_DELIMITER)
    else:
        segments = _RECORD_SPLIT_RE.split(raw)

    for segment in segments:
        payload = _record_payload(segment)
        if payload is None:
            continue
        fields = [f.strip().strip('"') for f in payload.split(TUPLE_DELIMITER)]
        if not fields:
            continue
        kind = fields[0].strip().strip('"').lower()
        if kind == "entity" and len(fields) >= 4:
            name = fields[1].strip()
            if not name:
                continue
            entities.append({"name": name, "type": fields[2].strip() or "UNKNOWN", "description": fields[3].strip()})
        elif kind == "relationship" and len(fields) >= 4:
            src, tgt = fields[1].strip(), fields[2].strip()
            if not src or not tgt:
                continue
            desc = fields[3].strip()
            weight = 1.0
            if len(fields) >= 5:
                try:
                    weight = float(re.findall(r"-?\d+\.?\d*", fields[4])[0])
                except (IndexError, ValueError):
                    weight = 1.0
            relationships.append({"source": src, "target": tgt, "description": desc, "weight": weight})
    return entities, relationships


def extract_graph(text_units, llm, summarize_threshold=3, on_progress=None):
    """Run extraction over all text units and build a merged knowledge graph.

    Parameters
    ----------
    text_units : list[TextUnit]
    llm : LanguageModel
    summarize_threshold : int
        If an entity accumulates more than this many distinct descriptions, ask
        the LLM to condense them into one (an extra ``# LLM:`` call).
    on_progress : callable(i, n) or None

    Returns the populated :class:`KnowledgeGraph`.
    """
    from .graph import KnowledgeGraph  # local import avoids a cycle at import time

    graph = KnowledgeGraph()
    # Accumulate raw descriptions per entity/edge so we can summarize at the end.
    entity_descs = {}   # entity id -> list[str]
    entity_names = {}    # entity id -> display name (first seen wins)

    n = len(text_units)
    for i, unit in enumerate(text_units):
        # PERF: sequential + single pass. Production: concurrent workers, N
        #       gleaning passes ("did you miss any entities?"), and a cache.
        raw = llm.extract(unit.text)
        ents, rels = parse_records(raw)

        for e in ents:
            eid = normalize_name(e["name"])
            entity_names.setdefault(eid, e["name"])
            entity_descs.setdefault(eid, [])
            if e["description"]:
                entity_descs[eid].append(e["description"])
            graph.add_entity(Entity(
                id=eid, name=entity_names[eid], type=e["type"],
                description=e["description"], text_unit_ids=[unit.id],
            ))
            if eid not in unit.entity_ids:
                unit.entity_ids.append(eid)

        for r in rels:
            sid, tid = normalize_name(r["source"]), normalize_name(r["target"])
            if sid == tid:
                continue
            graph.add_relationship(Relationship(
                id="", source=sid, target=tid, description=r["description"],
                weight=r["weight"], text_unit_ids=[unit.id],
            ))

        if on_progress:
            on_progress(i + 1, n)

    # Description summarization pass (merge duplicates into one gloss).
    for eid, descs in entity_descs.items():
        unique = list(dict.fromkeys(d for d in descs if d))
        ent = graph.entities[eid]
        if len(unique) > summarize_threshold:
            # LLM: condense many descriptions into one coherent description.
            ent.description = llm.summarize_descriptions(ent.name, unique)
        elif unique:
            ent.description = " ".join(unique[:summarize_threshold])[:400]

    graph.compute_stats()
    return graph
