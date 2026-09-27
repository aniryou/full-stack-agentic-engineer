"""embedders.py — text to unit vectors: a hashing embedder at T0, any ``/v1/embeddings`` server at T1.

The one idea: the store does not care where vectors come from, only that they are L2-normalised so a
dot product is a cosine. At T0 the vectors come from ``HashingEmbedder``, a re-implementation of
``ragkit.embed.HashingEmbedder`` (07.4's rag-from-scratch): each ``[a-z0-9]+`` token of the lower-cased
text adds 1 to bucket ``crc32(token) % dim``, then the vector is normalised. It is lexical, not
semantic — "live" and "lives" share nothing, "reside" and "home city" share nothing — which is exactly
why PRIMER §4's paraphrase subset exists: it measures what a real embedder has to add. At T1,
``OpenAIEmbeddings`` calls any OpenAI-compatible ``POST /v1/embeddings`` (vLLM's pooling runner, see
``deploy/any-gpu/``); ``get_embedder()`` picks it when ``MEMLAB_EMBED_URL`` is set.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import zlib

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
FALLBACK_LABEL = "hashing embedder (T0 fallback; not semantic)"     # ragkit's label, verbatim


def tokenize(text: str) -> list[str]:
    """ragkit's tokenizer: lower-case ``[a-z0-9]+`` runs."""
    return TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Bag-of-words feature hashing, identical to ragkit's (the tests cross-check it by path import)."""

    label = FALLBACK_LABEL

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.model_name = FALLBACK_LABEL

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode(self, texts):
        """A string gives a 1-D ``(dim,)`` vector; a list gives ``(n, dim)``; rows are unit length."""
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class OpenAIEmbeddings:
    """A client for ``POST {url}/v1/embeddings`` (vLLM ``--runner pooling``, or any compatible server).

    Vectors are re-normalised here, so a server that returns unnormalised embeddings still gives cosines.
    The dimension is learned from the first response.
    """

    def __init__(self, url: str, model: str | None = None, api_key: str | None = None,
                 batch_size: int = 64, timeout_s: float = 60):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.batch_size, self.timeout_s = batch_size, timeout_s
        self.model_name = model or self._discover_model()
        self.label = f"{self.model_name} via {self.url}/v1/embeddings"
        self.dim: int | None = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _discover_model(self) -> str:
        """vLLM serves one model per process, so the first id is it; a server that lists several (this lab's
        fake serves chat and embeddings together) is asked for the one that looks like an embedder."""
        req = urllib.request.Request(self.url + "/v1/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            ids = [d["id"] for d in json.loads(r.read())["data"]]
        return next((i for i in ids if any(w in i.lower() for w in ("embed", "hashing", "bge", "minilm"))), ids[0])

    def _batch(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model_name, "input": texts}).encode()
        req = urllib.request.Request(self.url + "/v1/embeddings", data=body, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = sorted(json.loads(r.read())["data"], key=lambda d: d["index"])
        arr = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms > 0, norms, 1.0)

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        out = np.concatenate([self._batch(items[i:i + self.batch_size]) for i in range(0, len(items), self.batch_size)])
        self.dim = out.shape[1]
        return out[0] if single else out


def get_embedder(dim: int = 1024):
    """``MEMLAB_EMBED_URL`` set: a real embedder over HTTP (T1). Otherwise the hashing embedder (T0)."""
    url = os.environ.get("MEMLAB_EMBED_URL")
    if url:
        return OpenAIEmbeddings(url, os.environ.get("MEMLAB_EMBED_MODEL"), os.environ.get("MEMLAB_API_KEY"))
    return HashingEmbedder(dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0
