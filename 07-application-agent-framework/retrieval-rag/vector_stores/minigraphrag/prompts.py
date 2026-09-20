"""Prompt templates -- the actual text GraphRAG sends to the LLM.

The prompts are the *heart* of GraphRAG: the graph is only as good as the
extraction prompt, and the answers are only as good as the report and search
prompts. We keep close to the shapes Microsoft GraphRAG uses so the templates
are recognizable, but trimmed for readability.

Two output conventions matter, because the parsers in ``extraction.py`` /
``summarize.py`` depend on them:

* **Extraction** returns *delimited tuples* (cheap to stream, easy to parse
  incrementally). Format:

      ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
      ##
      ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH)
      ...
      <|COMPLETE|>

* **Reports / global map** return **JSON**. Real GraphRAG constrains this with
  structured outputs (``output_config.format`` / ``strict`` tool schemas) so the
  JSON always parses; see the ``# LLM:`` note in ``llm.py``.

``MockLLM`` never reads these strings (it fakes the completions directly), but
the real ``PromptedLLM`` path formats and sends exactly these.
"""

# Delimiters used by the extraction format. Multi-character and unlikely to
# occur in natural text, so parsing stays unambiguous.
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Entity types we ask the model to use. Constraining the label set keeps the
# graph tidy; GraphRAG lets you tune this per domain.
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"]


ENTITY_EXTRACTION_SYSTEM = """\
You are an information-extraction engine that builds a knowledge graph.
Identify every entity of the requested types, and every relationship between
them that is stated or clearly implied by the text.

Entity types: {entity_types}

Output format -- one record per line, records separated by "{record_delimiter}":
  ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<one-sentence description>)
  ("relationship"{tuple_delimiter}<source name>{tuple_delimiter}<target name>{tuple_delimiter}<why they are related>{tuple_delimiter}<strength 1-10>)

Use the entity's exact surface name. When done, output "{completion_delimiter}".
Extract only what the text supports; do not invent entities.
"""

ENTITY_EXTRACTION_USER = """\
-----TEXT-----
{text}
-----END TEXT-----
"""


# In real GraphRAG, when one entity accumulates many descriptions across chunks,
# an LLM condenses them into a single coherent gloss. We expose that as its own
# call so the integration point is visible.
DESCRIPTION_SUMMARY_SYSTEM = """\
You merge several descriptions of the SAME thing into one concise, factual
description (2-3 sentences). Resolve overlaps; keep every distinct fact.
"""

DESCRIPTION_SUMMARY_USER = """\
Subject: {subject}
Descriptions:
{descriptions}
"""


COMMUNITY_REPORT_SYSTEM = """\
You are an analyst writing a report about a community of related entities.
Given the entities and relationships in the community, write a report that
would help a reader understand the community's purpose and significance.

Return a JSON object with exactly these keys:
  "title": a short, specific name for the community
  "summary": one paragraph overview
  "findings": a list of 2-5 objects, each {{"summary": short claim, "explanation": supporting detail}}
  "rating": a float 0-10 for the community's overall importance
  "rating_explanation": one sentence justifying the rating

Base every statement on the provided data. Return JSON only.
"""

COMMUNITY_REPORT_USER = """\
-----COMMUNITY DATA-----
{context}
-----END COMMUNITY DATA-----
"""


# Global search = map-reduce over community reports.
GLOBAL_MAP_SYSTEM = """\
You are helping answer a question using one analyst report as evidence.
Extract the points in the report that help answer the question. Return a JSON
object {{"points": [{{"description": str, "score": int 0-100}}]}} where score is
how directly the point answers the question (0 = irrelevant). Return JSON only.
"""

GLOBAL_MAP_USER = """\
Question: {question}

-----REPORT-----
{report}
-----END REPORT-----
"""

GLOBAL_REDUCE_SYSTEM = """\
You are answering a question by synthesizing key points gathered from many
analyst reports across a corpus. Write a clear, well-organized answer that
draws on the strongest points. Note if the reports do not contain enough
information to answer.
"""

GLOBAL_REDUCE_USER = """\
Question: {question}

-----KEY POINTS (each: source report -> point)-----
{points}
-----END KEY POINTS-----
"""


# Local search = answer from a focused subgraph around the query's entities.
LOCAL_ANSWER_SYSTEM = """\
You are answering a question using a focused context assembled from a knowledge
graph: the most relevant entities, the relationships among them, community
reports, and the source text units. Ground every claim in the provided context
and be specific. If the context is insufficient, say so.
"""

LOCAL_ANSWER_USER = """\
Question: {question}

-----CONTEXT-----
{context}
-----END CONTEXT-----
"""
