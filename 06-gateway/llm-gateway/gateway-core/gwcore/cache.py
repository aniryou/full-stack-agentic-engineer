"""Exact and semantic caching at the gateway (PRIMER §3).

The one idea: a cache hit is a saving only if it is the right answer *for this tenant*. An exact key
(a hash of everything that changes the answer, namespaced by tenant) is safe and rarely hits; a
semantic key (embed -> nearest cached query -> similarity threshold) hits paraphrases and also near
misses that differ in the one word that matters -- so you decide what may be cached at all, measure
hit and false-hit rates against the threshold on labelled traffic, and guard exactly what an embedding
cannot see: numbers, dates and named entities.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

EMBEDDER_LABEL = "hashing embedder (T0; lexical, not semantic)"
KEY_FIELDS = ("model", "messages", "tools", "tool_choice", "response_format", "temperature", "top_p",
              "max_completion_tokens", "seed", "reasoning_effort")
COMPLETE = frozenset({"stop"})       # the finish reasons whose answer may be stored: an allowlist, never "not an error"
_TOKEN = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b(?:[A-Z][A-Z0-9]+|\d[\d.,/:-]*)\b")      # IDs, codes, numbers, dates: SSO, Q3, 2025, 1234


def cacheable(request: dict, allowed: frozenset = frozenset({"faq"})) -> tuple:
    """What may be cached is declared by the route (``metadata.cache_class``), never guessed from the text."""
    cls = (request.get("metadata") or {}).get("cache_class")
    if request.get("tools"):
        return False, "tool calls act on the world"
    if cls not in allowed:
        return False, f"class {cls!r} is not cacheable on this route"
    return True, cls


def exact_key(namespace: str, request: dict) -> str:
    """SHA-256 of the namespace and every field that changes the answer (``stream`` does not)."""
    body = {k: request[k] for k in KEY_FIELDS if k in request}
    return hashlib.sha256(json.dumps([namespace, body], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ExactCache:
    def __init__(self, ttl: float = 3600.0):
        self.ttl, self.items = ttl, {}

    def get(self, key: str, now: float):
        hit = self.items.get(key)
        return hit[0] if hit and hit[1] > now else None

    def put(self, key: str, value, now: float) -> None:
        self.items[key] = (value, now + self.ttl)


class HashingEmbedder:
    """Built like ``ragkit.embed.HashingEmbedder`` (07.4): crc32 of each ``[a-z0-9]+`` token into ``dim``
    buckets, L2-normalised float32. Two texts are similar exactly when they share words."""

    model_name = EMBEDDER_LABEL

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in _TOKEN.findall(text.lower()):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def entities(text: str) -> frozenset:
    return frozenset(_ENTITY.findall(text))


class SemanticCache:
    """Nearest cached query per namespace; a hit needs similarity >= threshold (and equal entities if guarded)."""

    def __init__(self, threshold: float = 0.9, *, guard: bool = True, embedder=None, ttl: float = 3600.0):
        self.threshold, self.guard, self.ttl = threshold, guard, ttl
        self.embedder = embedder or HashingEmbedder()
        self.spaces = defaultdict(list)          # namespace -> [(vector, query, entities, answer, expires)]

    def lookup(self, namespace: str, query: str, now: float = 0.0):
        """(answer, similarity, cached query) of the nearest live entry that passes, else None."""
        live = [e for e in self.spaces[namespace] if e[4] > now]
        if not live:
            return None
        q = self.embedder.encode(query)
        sims = np.stack([e[0] for e in live]) @ q
        best = int(np.argmax(sims))
        vec, cached_q, ents, answer, _ = live[best]
        if sims[best] < self.threshold or (self.guard and ents != entities(query)):
            return None
        return answer, float(sims[best]), cached_q

    def store(self, namespace: str, query: str, answer, now: float = 0.0) -> None:
        self.spaces[namespace].append((self.embedder.encode(query), query, entities(query), answer, now + self.ttl))


def load_sample() -> dict:
    return json.loads((Path(__file__).parent / "data" / "traffic.json").read_text())


def sweep_thresholds(thresholds, *, guard: bool, sample: dict | None = None) -> list:
    """Per threshold, on the labelled sample: hit rate on paraphrases (right answer served), false-hit
    rate over every cacheable lookup (a wrong answer served), and precision (right / served)."""
    s = sample or load_sample()
    rows = []
    for th in thresholds:
        cache = SemanticCache(th, guard=guard)
        for seed in s["seeds"]:
            cache.store(s["tenant"], seed["q"], seed["id"])
        probes = [p for p in s["probes"] if p["label"] != "uncacheable"]
        served = [(p, cache.lookup(s["tenant"], p["q"])) for p in probes]
        right = sum(1 for p, hit in served if hit and p["label"] == "same" and hit[0] == p["seed"])
        wrong = sum(1 for p, hit in served if hit and not (p["label"] == "same" and hit[0] == p["seed"]))
        n_same = sum(p["label"] == "same" for p in probes)
        rows.append({"threshold": th, "hit_rate": right / n_same, "false_hit_rate": wrong / len(probes),
                     "precision": right / (right + wrong) if right + wrong else 1.0, "right": right, "wrong": wrong})
    return rows
