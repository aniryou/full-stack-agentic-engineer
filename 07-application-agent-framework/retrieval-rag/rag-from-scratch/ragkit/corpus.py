"""Load the bundled corpus and evaluation labels, and a shared tokenizer.

The corpus is a handful of short Markdown docs for a fictional SaaS company
("Meridian"). Each doc has `##` sections, which is what the structure-aware
chunker in notebook 02 keys on. Keep this module boring: it is not the lesson.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data"
CORPUS_DIR = DATA / "corpus"
QRELS_PATH = DATA / "eval" / "qrels.json"


@dataclass
class Chunk:
    """One retrievable unit of text plus where it came from."""

    chunk_id: str          # unique, e.g. "expense-policy#submitting-a-claim"
    doc_id: str            # source document, e.g. "expense-policy"
    text: str              # the text you embed / match / show the model
    section: str = ""      # section heading this chunk belongs to ("" if none)
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # short, notebook-friendly
        head = self.text[:70].replace("\n", " ")
        return f"Chunk({self.chunk_id!r}, {head!r}...)"


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Used by BM25 and the fixed-size chunker
    so that matching and splitting agree on what a 'token' is."""
    return _TOKEN_RE.findall(text.lower())


def load_documents() -> dict[str, str]:
    """Return {doc_id: raw_markdown} for every doc in the corpus."""
    docs = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def load_corpus() -> list[Chunk]:
    """Convenience loader: whole documents as chunks (one Chunk per doc).

    This is the *naive* chunking used by notebook 01 so the first pipeline is
    as simple as possible. Notebook 02 replaces it with real chunkers.
    """
    chunks = []
    for doc_id, text in load_documents().items():
        # Drop the leading "# Title" line from the body but keep it as meta.
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else doc_id
        chunks.append(
            Chunk(chunk_id=doc_id, doc_id=doc_id, text=text, section="",
                  meta={"title": title})
        )
    return chunks


def load_qrels() -> list[dict]:
    """Return the evaluation set: a list of
    {qid, question, gold_docs:[doc_id,...], gold_section:str|None, kind:str}.

    `gold_docs` is deliberately labelled at the *document* level so recall
    numbers stay comparable no matter which chunker you use.
    """
    return json.loads(QRELS_PATH.read_text(encoding="utf-8"))
