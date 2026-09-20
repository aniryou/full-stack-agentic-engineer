"""The model layer -- where a real LLM and embedding model plug in.

Everything GraphRAG does that requires *intelligence* funnels through this
module: extraction, description merging, community reports, and the map/reduce
answer steps are all LLM calls; retrieval needs a text embedding model. To keep
the package runnable offline and deterministic, the default implementations are
fakes:

* ``MockLLM``      -- rule-based stand-ins for the six LLM calls. No network,
                      fully deterministic. Good enough to build a real graph and
                      answer questions on a small, well-written corpus.
* ``HashingEmbedding`` -- a deterministic lexical embedding (hashed word/char
                      n-grams). Captures word overlap, not meaning.

The class layout is the whole point of the lesson:

    LanguageModel   -- defines the SIX semantic calls GraphRAG makes, and builds
                       the exact prompts for each (see prompts.py). `complete()`
                       is abstract: it is the ONE method a real backend needs.
    MockLLM         -- implements `complete()` with deterministic heuristics.
    AnthropicLLM    -- implements `complete()` with a real Claude call.

Note the `meta=` argument threaded through `complete()`: real backends ignore
it and read the prompt text; the mock reads the structured hints in `meta` so it
can fake a response without having to parse English back out of the prompt.

# LLM:  every `self.complete(...)` below is a real model call in production.
# PERF: those calls are the pipeline's cost + latency. Real GraphRAG issues them
#       with bounded concurrency, retries, rate-limit handling and a cache keyed
#       on (prompt, model) so re-indexing is cheap. All omitted here.
"""

import json
import re
import zlib

import numpy as np

from . import prompts
from .models import normalize_name


# --------------------------------------------------------------------------- #
# Shared lightweight text helpers (used by the mock + the embedding model)
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "as", "by", "is", "was", "are", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "them", "his", "her",
    "their", "who", "whom", "which", "what", "when", "where", "why", "how", "did",
    "do", "does", "has", "have", "had", "from", "about", "into", "than", "then",
    "so", "if", "not", "no", "all", "any", "can", "will", "would", "there",
}
_WORD_RE = re.compile(r"[a-z0-9]+")

# A proper-noun run: capitalized words, optionally joined by name-internal
# connectors like "of"/"de" ("City of Calder"). Note we do NOT include "and"
# here -- "Ada and Ben" is two entities, not one. Trailing "." is excluded from
# the character class so a sentence-final period is not glued onto a name.
_PROPER_RE = re.compile(
    r"[A-Z][a-zA-Z0-9&'\-]*"
    r"(?:\s+(?:of|de|la|van|von|di|del)\s+[A-Z][a-zA-Z0-9&'\-]*"
    r"|\s+[A-Z][a-zA-Z0-9&'\-]*)*"
)
_SENT_RE = re.compile(r"[^.!?]+[.!?]?")

# Single capitalized words that are usually just sentence-initial, not entities.
_COMMON_CAPS = {
    "the", "a", "an", "this", "that", "these", "those", "he", "she", "it",
    "they", "we", "you", "i", "his", "her", "their", "its", "when", "where",
    "why", "how", "what", "who", "but", "and", "or", "if", "then", "there",
    "here", "some", "many", "most", "after", "before", "during", "while",
    "although", "however", "meanwhile", "later", "soon", "yet", "still", "no",
    "yes", "one", "two", "three", "each", "every", "both", "in", "on", "at",
}

# Keyword -> type hints for the mock extractor's type guesser.
_TYPE_KEYWORDS = {
    "ORGANIZATION": {
        "inc", "corp", "company", "institute", "university", "guild", "order",
        "council", "ministry", "corporation", "foundation", "lab", "labs",
        "laboratory", "laboratories", "union", "alliance", "committee", "agency",
        "academy", "society", "syndicate", "bureau", "department", "school",
        "authority", "board", "commission", "office", "group",
    },
    "LOCATION": {
        "city", "river", "mountain", "mountains", "kingdom", "village", "valley",
        "sea", "island", "harbor", "harbour", "forest", "castle", "bay", "port",
        "region", "province", "lake", "town", "district", "empire", "realm",
        "desert", "coast", "hills", "plains", "canyon",
    },
    "EVENT": {
        "war", "battle", "festival", "summit", "conference", "treaty",
        "revolution", "uprising", "expedition", "siege", "election", "plague",
        "famine", "coronation", "rebellion", "campaign",
    },
    "CONCEPT": {
        "project", "program", "programme", "operation", "mission", "initiative",
        "plan", "protocol", "act", "system", "algorithm", "doctrine",
    },
}


def content_words(text):
    """Lower-case content tokens (stop-words removed)."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _sentences(text):
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _guess_type(name):
    toks = set(name.lower().replace(".", " ").split())
    for etype, keywords in _TYPE_KEYWORDS.items():
        if toks & keywords:
            return etype
    words = name.split()
    if 1 <= len(words) <= 3 and name[0].isupper():
        return "PERSON"
    return "CONCEPT"


# --------------------------------------------------------------------------- #
# Deterministic heuristic extractor (stands in for the extraction LLM call)
# --------------------------------------------------------------------------- #
def heuristic_extract(text):
    """Rule-based entity/relationship extraction from one chunk.

    A crude substitute for the extraction LLM: proper-noun runs become entities,
    and two entities co-occurring in a sentence become a relationship. Returns
    ``(entities, relationships)`` as lists of plain dicts.
    """
    sentences = _sentences(text)

    # 1. Collect candidate proper-noun mentions with the sentence they occur in.
    mentions = {}  # normalized name -> {"name", "sentences": [idx...]}
    for si, sent in enumerate(sentences):
        for m in _PROPER_RE.finditer(sent):
            surface = m.group(0).strip().strip(".,:;!?")
            # Drop a leading article: "The Meridian Institute" -> "Meridian Institute".
            parts = surface.split()
            if len(parts) > 1 and parts[0].lower() == "the":
                surface = " ".join(parts[1:])
            words = surface.split()
            if not words:
                continue
            # Drop single common capitalized words (usually sentence-initial).
            if len(words) == 1 and words[0].lower() in _COMMON_CAPS:
                continue
            key = normalize_name(surface)
            rec = mentions.setdefault(key, {"name": surface, "sentences": []})
            rec["sentences"].append(si)

    # 2. Build entity records. Description = first sentence the entity appears in.
    entities = {}
    for key, rec in mentions.items():
        first_sent = sentences[rec["sentences"][0]]
        entities[key] = {
            "name": rec["name"],
            "type": _guess_type(rec["name"]),
            "description": first_sent[:240],
        }

    # 3. Relationships: unordered pairs of entities sharing a sentence.
    rels = {}  # (a_key, b_key) -> {"description", "strength"}
    keys_by_sentence = [[] for _ in sentences]
    for key, rec in mentions.items():
        for si in set(rec["sentences"]):
            keys_by_sentence[si].append(key)
    for si, keys in enumerate(keys_by_sentence):
        keys = sorted(set(keys))
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                pair = (a, b)
                if pair not in rels:
                    rels[pair] = {"description": sentences[si][:240], "strength": 1}
                else:
                    rels[pair]["strength"] += 1

    entity_list = [
        {"name": e["name"], "type": e["type"], "description": e["description"]}
        for e in entities.values()
    ]
    rel_list = [
        {
            "source": entities[a]["name"],
            "target": entities[b]["name"],
            "description": r["description"],
            "strength": r["strength"],
        }
        for (a, b), r in rels.items()
    ]
    return entity_list, rel_list


def _render_extraction(entities, relationships):
    """Serialize extracted records into the delimited format the parser expects."""
    td, rd = prompts.TUPLE_DELIMITER, prompts.RECORD_DELIMITER

    def clean(s):
        return str(s).replace(td, "/").replace(rd, " ").replace("\n", " ").strip()

    records = []
    for e in entities:
        records.append(f'("entity"{td}{clean(e["name"])}{td}{clean(e["type"])}{td}{clean(e["description"])})')
    for r in relationships:
        records.append(
            f'("relationship"{td}{clean(r["source"])}{td}{clean(r["target"])}'
            f'{td}{clean(r["description"])}{td}{r["strength"]})'
        )
    return f"\n{rd}\n".join(records) + f"\n{prompts.COMPLETION_DELIMITER}"


def _render_community(entities, relationships):
    """Human-readable community context block for the report prompt."""
    lines = ["Entities:"]
    for e in entities:
        lines.append(f"- {e['name']} ({e.get('type', 'UNKNOWN')}): {e.get('description', '')}")
    lines.append("")
    lines.append("Relationships:")
    for r in relationships:
        lines.append(f"- {r['source']} -> {r['target']}: {r.get('description', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The language-model interface
# --------------------------------------------------------------------------- #
class LanguageModel:
    """The six semantic calls GraphRAG makes, plus the prompt-building for each.

    Subclass and implement :meth:`complete` -- that is the only method that
    actually talks to a model. The semantic methods here are shared by every
    backend: they format the prompts (see ``prompts.py``) and return the raw
    completion string for the caller to parse.
    """

    def complete(self, system, user, *, task=None, meta=None):
        """Return the model's completion for a (system, user) prompt pair.

        ``task`` and ``meta`` are hints for offline fakes; real backends ignore
        them and use only ``system`` + ``user``.
        """
        raise NotImplementedError

    # ---- the six calls, in pipeline order -------------------------------- #
    def extract(self, text):
        """Extraction call: text unit -> delimited entity/relationship records."""
        system = prompts.ENTITY_EXTRACTION_SYSTEM.format(
            entity_types=", ".join(prompts.DEFAULT_ENTITY_TYPES),
            tuple_delimiter=prompts.TUPLE_DELIMITER,
            record_delimiter=prompts.RECORD_DELIMITER,
            completion_delimiter=prompts.COMPLETION_DELIMITER,
        )
        user = prompts.ENTITY_EXTRACTION_USER.format(text=text)
        return self.complete(system, user, task="extract", meta={"text": text})

    def summarize_descriptions(self, subject, descriptions):
        """Merge several descriptions of one entity/edge into a single gloss."""
        system = prompts.DESCRIPTION_SUMMARY_SYSTEM
        user = prompts.DESCRIPTION_SUMMARY_USER.format(
            subject=subject, descriptions="\n".join(f"- {d}" for d in descriptions)
        )
        return self.complete(
            system, user, task="summarize_descriptions",
            meta={"subject": subject, "descriptions": descriptions},
        )

    def community_report(self, title_hint, entities, relationships):
        """Author a community report (returns raw JSON string)."""
        context = _render_community(entities, relationships)
        system = prompts.COMMUNITY_REPORT_SYSTEM
        user = prompts.COMMUNITY_REPORT_USER.format(context=context)
        return self.complete(
            system, user, task="report",
            meta={"title_hint": title_hint, "entities": entities, "relationships": relationships},
        )

    def global_map(self, question, report_id, report_text, report_summary="", report_rank=0.0):
        """Map step: pull question-relevant points from one report (raw JSON)."""
        system = prompts.GLOBAL_MAP_SYSTEM
        user = prompts.GLOBAL_MAP_USER.format(question=question, report=report_text)
        return self.complete(
            system, user, task="map",
            meta={
                "question": question, "report_id": report_id, "report_text": report_text,
                "report_summary": report_summary, "report_rank": report_rank,
            },
        )

    def global_reduce(self, question, points):
        """Reduce step: synthesize a final answer from many mapped points."""
        system = prompts.GLOBAL_REDUCE_SYSTEM
        user = prompts.GLOBAL_REDUCE_USER.format(
            question=question, points="\n".join(f"- {p}" for p in points)
        )
        return self.complete(
            system, user, task="reduce",
            meta={"question": question, "points": points},
        )

    def local_answer(self, question, context_text, structured):
        """Answer a question from a focused local subgraph context."""
        system = prompts.LOCAL_ANSWER_SYSTEM
        user = prompts.LOCAL_ANSWER_USER.format(question=question, context=context_text)
        return self.complete(
            system, user, task="local_answer",
            meta={"question": question, "structured": structured},
        )


class MockLLM(LanguageModel):
    """Deterministic, offline stand-in for every LLM call.

    Not intelligent -- just enough structure that the pipeline produces a real
    graph, real communities, real reports, and grounded (extractive) answers
    without a network call. Swap in :class:`AnthropicLLM` for real quality.
    """

    def complete(self, system, user, *, task=None, meta=None):
        meta = meta or {}
        if task == "extract":
            return _render_extraction(*heuristic_extract(meta["text"]))
        if task == "summarize_descriptions":
            return self._merge_descriptions(meta["descriptions"])
        if task == "report":
            return json.dumps(self._report(meta))
        if task == "map":
            return json.dumps(self._map(meta))
        if task == "reduce":
            return self._reduce(meta)
        if task == "local_answer":
            return self._local(meta)
        raise ValueError(f"MockLLM cannot handle task={task!r}")

    # ---- per-task fakes -------------------------------------------------- #
    @staticmethod
    def _merge_descriptions(descriptions):
        seen, out = set(), []
        for d in descriptions:
            d = d.strip()
            if d and d.lower() not in seen:
                seen.add(d.lower())
                out.append(d)
        return " ".join(out[:3])[:400]

    @staticmethod
    def _report(meta):
        entities = sorted(meta["entities"], key=lambda e: -e.get("rank", 0.0))
        rels = meta["relationships"]
        names = [e["name"] for e in entities]
        top = names[:3]
        title = " & ".join(top[:2]) if top else (meta.get("title_hint") or "Community")
        summary = (
            f"This community is organized around {', '.join(top) if top else 'several entities'}"
            f" and contains {len(entities)} entities linked by {len(rels)} relationships."
        )
        findings = []
        for e in entities[:4]:
            findings.append({
                "summary": f"{e['name']} is a central {e.get('type', 'entity').lower()}",
                "explanation": e.get("description", ""),
            })
        if rels:
            r = max(rels, key=lambda r: r.get("weight", 1.0))
            findings.append({
                "summary": f"{r['source']} and {r['target']} are strongly connected",
                "explanation": r.get("description", ""),
            })
        total_w = sum(r.get("weight", 1.0) for r in rels)
        rating = round(min(10.0, 1.5 * len(entities) + 0.5 * total_w), 1)
        return {
            "title": title,
            "summary": summary,
            "findings": findings,
            "rating": rating,
            "rating_explanation": f"Importance scales with the community's {len(entities)} entities and {len(rels)} ties.",
        }

    @staticmethod
    def _map(meta):
        # Blend two signals a real LLM would weigh implicitly:
        #  * lexical relevance -- does the report share vocabulary with the question?
        #  * community importance -- big/central communities ARE the main themes,
        #    so broad questions ("what are the themes?") still surface them even
        #    with no word overlap.
        q_terms = set(content_words(meta["question"]))
        r_terms = set(content_words(meta["report_text"]))
        overlap = (len(q_terms & r_terms) / len(q_terms)) if q_terms else 0.0
        rank = float(meta.get("report_rank", 0.0) or 0.0)
        score = int(round(60 * overlap + 4 * min(10.0, rank)))
        if score <= 0:
            return {"points": []}
        desc = meta.get("report_summary") or " ".join(meta["report_text"].split())[:260]
        return {"points": [{"description": desc, "score": score}]}

    @staticmethod
    def _reduce(meta):
        points = meta["points"]
        q = meta["question"]
        if not points:
            return "The community reports do not contain enough information to answer this question."
        lines = [f'Drawing on the community reports, here is what bears on "{q}":', ""]
        for i, p in enumerate(points, 1):
            lines.append(f"{i}. {p}")
        return "\n".join(lines)

    @staticmethod
    def _local(meta):
        s = meta["structured"]
        q_terms = set(content_words(meta["question"]))
        entities = s.get("entities", [])

        def relevance(e):
            terms = set(content_words(e["name"] + " " + e.get("description", "")))
            return len(terms & q_terms)

        ranked = sorted(entities, key=lambda e: (-relevance(e), -e.get("rank", 0.0)))
        # If nothing matches lexically, fall back to the retrieved (top-ranked) set.
        chosen = [e for e in ranked if relevance(e) > 0][:5] or ranked[:3]

        lines = ["Based on the knowledge graph:", ""]
        for e in chosen:
            lines.append(f"- {e['name']} ({e.get('type', 'UNKNOWN')}): {e.get('description', '')}")
        rels = s.get("relationships", [])[:5]
        if rels:
            lines.append("")
            lines.append("Key relationships:")
            for r in rels:
                lines.append(f"- {r['source']} — {r['target']}: {r.get('description', '')}")
        return "\n".join(lines)


class AnthropicLLM(LanguageModel):
    """Real backend: one Claude call per :meth:`complete`.

    Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY`` (or an
    ``ant auth login`` profile). Everything above -- the six semantic calls and
    their prompts -- is inherited unchanged; only ``complete`` differs. This is
    the entire surface a production GraphRAG backend has to implement.
    """

    def __init__(self, model="claude-opus-5", max_tokens=4096, client=None):
        if client is None:
            import anthropic  # optional dependency; imported lazily
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system, user, *, task=None, meta=None):
        # LLM: the actual model call. Adaptive thinking per current Claude API
        # guidance. For "report"/"map" tasks you would additionally pass
        # output_config={"format": ...} (structured outputs) so the JSON always
        # parses -- omitted here to keep the adapter minimal.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class EmbeddingModel:
    """Turns text into a dense vector. Implement :meth:`embed`."""

    def embed(self, texts):
        """Return an ``(n, dim)`` float32 array for a list of strings."""
        raise NotImplementedError

    def embed_one(self, text):
        return self.embed([text])[0]


class HashingEmbedding(EmbeddingModel):
    """Deterministic, offline lexical embedding via the hashing trick.

    Each word (unigram + bigram) and character trigram is hashed with a stable
    CRC32 into one of ``dim`` buckets; bucket counts form the vector, then it is
    L2-normalized. This captures *lexical overlap* (a query naming an entity
    lands near that entity), but NOT semantics -- "car" and "automobile" are
    orthogonal.

    # LLM: real GraphRAG embeds with a neural text embedding model (OpenAI
    #      text-embedding-3, Cohere, a local sentence-transformer, ...). Replace
    #      this class with a call to that model; the rest of the pipeline is
    #      unchanged, since it only ever sees float vectors.
    """

    def __init__(self, dim=512):
        self.dim = int(dim)

    def _bucket(self, token):
        return zlib.crc32(token.encode("utf-8")) % self.dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = content_words(text)
            for w in words:
                out[i, self._bucket("w:" + w)] += 1.0
            for a, b in zip(words, words[1:]):
                out[i, self._bucket(f"b:{a}_{b}")] += 1.0
            low = text.lower()
            for j in range(len(low) - 2):
                tri = low[j:j + 3]
                if tri.strip():
                    out[i, self._bucket("c:" + tri)] += 0.5
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        out /= norms
        return out
