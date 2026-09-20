"""Chunking -- split documents into overlapping "text units".

The first stage of the GraphRAG index. Extraction quality is sensitive to chunk
size: too large and the LLM misses entities; too small and relationships that
span a boundary are lost. The overlap gives cross-boundary relationships a
second chance to appear intact in some chunk.
"""

import re

from .models import TextUnit

# Split on whitespace. We count length in *words* purely so the demo stays
# dependency-free and deterministic.
#
# PERF: real GraphRAG chunks by *tokens* using the model's tokenizer (tiktoken),
# because the context window and cost are measured in tokens, not words. A word
# is ~1.3 tokens on average, so swap this splitter for `tiktoken.encoding_for_model(...)`
# and slice the token-id list to chunk by true token budget.
_WORD_RE = re.compile(r"\S+")


def _split_words(text):
    return _WORD_RE.findall(text)


def chunk_text(text, chunk_size=120, overlap=20):
    """Yield ``(index, chunk_text)`` windows of ``chunk_size`` words with ``overlap``.

    A sliding window advances by ``chunk_size - overlap`` words each step, so
    consecutive chunks share ``overlap`` words of context.
    """
    words = _split_words(text)
    if not words:
        return
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    index = 0
    start = 0
    n = len(words)
    while start < n:
        window = words[start:start + chunk_size]
        yield index, " ".join(window)
        index += 1
        if start + chunk_size >= n:
            break
        start += step


def chunk_documents(documents, chunk_size=120, overlap=20):
    """Chunk an iterable of :class:`Document` into a flat list of :class:`TextUnit`.

    Text-unit ids are stable and human-readable: ``"<doc id>::chunk<index>"``.
    """
    units = []
    for doc in documents:
        for index, chunk in chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap):
            units.append(
                TextUnit(
                    id=f"{doc.id}::chunk{index}",
                    text=chunk,
                    document_id=doc.id,
                    index=index,
                )
            )
    return units
