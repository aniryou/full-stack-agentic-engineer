"""Caching at the gateway: an exact cache, a semantic cache, and what must never be cached.

The one idea (PRIMER §3): a gateway can skip the model entirely — but only for requests whose answer does not
depend on who asks or when. Four caches sit on the path and save different things:

| cache | where | saves | risk |
|---|---|---|---|
| exact response | gateway | the whole call | stale answers; cross-tenant leaks without a namespace |
| semantic response | gateway | the whole call on a paraphrase | **false hits**: a near miss gets another question's answer |
| provider prompt cache | hosted API | ~90 % of cached input $ (scaling primer §3.4) | none to correctness |
| engine prefix cache | vLLM | prefill time (TTFT) (serving-engine PRIMER §5) | a timing side channel: `cache_salt` per tenant |

The semantic path here: classify (personal and time-sensitive queries are never cached) -> namespace by tenant,
alias and system prompt -> embed the last user message -> nearest neighbour -> threshold -> an **exact guard**
on numbers, dates and codes (a lexical near miss such as "order 1234" vs "order 1243" is rejected) -> serve.
The embedder is the lexical hashing embedder of `ragkit.embed.HashingEmbedder` (07.4), re-implemented: two texts
are close exactly when they share words, so it finds rephrasings that reuse words and misses the ones that do
not (a real embedder, T1, finds more of both — true paraphrases and false hits; embeddings primer §15,
"Embeddings elsewhere in agent systems"). The classifier is a regex stand-in, labelled as one.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
import zlib
from dataclasses import dataclass
from importlib import resources

import numpy as np

from ..tokens import last_user_text, system_text

EMBEDDER_LABEL = "hashing embedder (T0 stand-in; lexical, not semantic)"
CLASSIFIER_LABEL = "regex stand-in, not a classifier"
_TOKEN = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Za-z]{0,4}-?\d+(?:[.,:/]\d+)*\b")
_PERSONAL = re.compile(r"\b(my|our)\s+(order|account|invoice|balance|booking|ticket|payment|address|card|bill|"
                       r"subscription)s?\b|\b(order|account|invoice|ticket)\s*#?\s*\d{3,}\b", re.I)
_TIME = re.compile(r"\b(today|tonight|now|right now|currently|current|latest|tomorrow|yesterday|this week|"
                   r"weather|stock price|exchange rate)\b", re.I)


class HashingEmbedder:
    """crc32(token) % dim bag of words, L2-normalised, float32 — built like `ragkit.embed.HashingEmbedder`."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.model_name = EMBEDDER_LABEL

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in _TOKEN.findall(text.lower()):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def entities(text: str) -> tuple:
    """Numbers, dates and codes in `text`, normalised and sorted: the guard compares these exactly."""
    return tuple(sorted(e.lower().replace("-", "") for e in _ENTITY.findall(text)))


def classify_query(text: str) -> str:
    """general | personal | time_sensitive (a regex stand-in, not a classifier)."""
    if _PERSONAL.search(text):
        return "personal"
    if _TIME.search(text):
        return "time_sensitive"
    return "general"


_CANON_KEYS = ("model", "messages", "temperature", "top_p", "max_completion_tokens", "max_tokens", "tools",
               "tool_choice", "response_format", "stop", "seed", "reasoning_effort")


def canonical(body: dict) -> str:
    """The part of a request that decides its answer, as stable JSON (no `stream`, `user`, `metadata`)."""
    return json.dumps({k: body[k] for k in _CANON_KEYS if k in body}, sort_keys=True, separators=(",", ":"))


def cacheable(body: dict) -> tuple[bool, str]:
    """Is this request's answer safe to reuse? (deterministic, no tools, one choice, not personal or time-bound)"""
    if body.get("temperature") != 0:
        return False, "sampling (temperature is not 0)"
    if body.get("tools"):
        return False, "tools (the answer depends on tool state)"
    if (body.get("n") or 1) != 1:
        return False, "n > 1"
    cls = classify_query(last_user_text(body))
    if cls != "general":
        return False, cls
    return True, "general"


def cache_salt(tenant: str, secret: str) -> str:
    """A per-tenant vLLM `cache_salt`: HMAC-SHA256(secret, tenant), base64url without padding (43 chars, 256 bits).

    Derived from the *verified* tenant, never from a header; secret, so another tenant cannot compute it."""
    mac = hmac.new(secret.encode(), tenant.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()


def valid_cache_salt(salt) -> bool:
    """vLLM 0.30.0's rules: a non-empty string of at most 128 characters without '@', '/', '\\' or NUL."""
    return isinstance(salt, str) and 1 <= len(salt) <= 128 and not any(c in salt for c in "@/\\\x00")


@dataclass
class Lookup:
    kind: str | None                    # exact | semantic | None
    response: dict | None = None
    score: float = 0.0
    reason: str = ""                    # why it missed or was bypassed
    matched: str = ""                   # the cached query a semantic hit matched


class GatewayCache:
    def __init__(self, store, cfg, clock=time.time, embedder: HashingEmbedder | None = None):
        self.store, self.cfg, self.clock = store, cfg, clock
        self.embedder = embedder or HashingEmbedder(cfg.dim)
        self.stats = {"exact": 0, "semantic": 0, "miss": 0, "bypass": 0, "guard_rejected": 0, "stored": 0}

    @staticmethod
    def namespace(tenant: str, alias: str, body: dict) -> str:
        """tenant | alias | hash of the system prompt: a hit never crosses tenants, models or instructions."""
        return f"{tenant}|{alias}|{hashlib.sha256(system_text(body).encode()).hexdigest()[:12]}"

    @staticmethod
    def exact_key(body: dict) -> str:
        return hashlib.sha256(canonical(body).encode()).hexdigest()

    @staticmethod
    def single_turn(body: dict) -> bool:
        return sum(1 for m in body.get("messages") or [] if m.get("role") not in ("system", "developer")) == 1

    def lookup(self, tenant: str, alias: str, body: dict) -> Lookup:
        ok, why = cacheable(body)
        if not ok:
            self.stats["bypass"] += 1
            return Lookup(None, reason=why)
        ns, now = self.namespace(tenant, alias, body), self.clock()
        if self.cfg.exact:
            rows = self.store.query("SELECT response FROM cache WHERE ns=? AND key=? AND kind='exact' AND expires_at>?",
                                    (ns, self.exact_key(body), now))
            if rows:
                self.stats["exact"] += 1
                self.store.execute("UPDATE cache SET hits=hits+1 WHERE ns=? AND key=? AND kind='exact'", (ns, self.exact_key(body)))
                return Lookup("exact", json.loads(rows[0]["response"]), 1.0)
        if self.cfg.semantic and self.single_turn(body):
            q = last_user_text(body)
            rows = self.store.query("SELECT key, query, entities, vec, response FROM cache "
                                    "WHERE ns=? AND kind='semantic' AND expires_at>?", (ns, now))
            if rows:
                mat = np.stack([np.frombuffer(r["vec"], dtype="float32") for r in rows])
                scores = mat @ self.embedder.encode(q)
                i = int(np.argmax(scores))
                best = float(scores[i])
                if best >= self.cfg.threshold:
                    if self.cfg.entity_guard and json.loads(rows[i]["entities"]) != list(entities(q)):
                        self.stats["guard_rejected"] += 1
                        self.stats["miss"] += 1
                        return Lookup(None, score=best, reason="entity guard", matched=rows[i]["query"])
                    self.stats["semantic"] += 1
                    self.store.execute("UPDATE cache SET hits=hits+1 WHERE ns=? AND key=? AND kind='semantic'", (ns, rows[i]["key"]))
                    return Lookup("semantic", json.loads(rows[i]["response"]), best, matched=rows[i]["query"])
                self.stats["miss"] += 1
                return Lookup(None, score=best, reason="below threshold", matched=rows[i]["query"])
        self.stats["miss"] += 1
        return Lookup(None, reason="miss")

    def put(self, tenant: str, alias: str, body: dict, response: dict) -> bool:
        if not cacheable(body)[0]:
            return False
        ns, now = self.namespace(tenant, alias, body), self.clock()
        exp, resp = now + self.cfg.ttl_s, json.dumps(response)
        key = self.exact_key(body)
        with self.store.lock:
            db = self.store.db
            if self.cfg.exact:
                db.execute("INSERT OR REPLACE INTO cache (ns, key, kind, query, entities, vec, response, created, expires_at)"
                           " VALUES (?,?,?,?,?,?,?,?,?)", (ns, key, "exact", None, None, None, resp, now, exp))
            if self.cfg.semantic and self.single_turn(body):
                q = last_user_text(body)
                db.execute("INSERT OR REPLACE INTO cache (ns, key, kind, query, entities, vec, response, created, expires_at)"
                           " VALUES (?,?,?,?,?,?,?,?,?)", (ns, key, "semantic", q, json.dumps(list(entities(q))),
                                                          self.embedder.encode(q).tobytes(), resp, now, exp))
        self.stats["stored"] += 1
        return True

    def invalidate(self, tenant: str | None = None) -> int:
        """Drop every entry (or one tenant's): the lever for a changed prompt, model or source of truth."""
        if tenant is None:
            return self.store.execute("DELETE FROM cache")
        return self.store.execute("DELETE FROM cache WHERE ns LIKE ?", (tenant + "|%",))


# ------------------------------------------------------------------------------------------ the labelled sample
SAMPLE_LABEL = "hand-written traffic sample for this lab (illustrative), labelled with answer groups"


def load_sample() -> list[dict]:
    """The bundled labelled traffic: each query has an answer `group` (same group = same correct answer),
    a `class` (general | personal | time_sensitive) and a `kind` (seed, paraphrase, near_miss, ...)."""
    text = resources.files("gwlab").joinpath("data", "cache_traffic.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sweep(sample: list, thresholds, guard: bool = True, embedder: HashingEmbedder | None = None) -> list[dict]:
    """Replay the sample through a fresh semantic cache per threshold.

    A query that is not `general` bypasses the cache. Otherwise its nearest cached query answers it when the
    cosine clears the threshold (and, with `guard`, the entities match exactly); the answer is *correct* when
    both are in the same group. A miss stores the query. Rates are over cacheable queries:
    `hit_rate` (served from cache), `false_hit_rate` (served another question's answer), and `reachable`
    (share whose group was already cached: the most any threshold could serve).
    """
    emb = embedder or HashingEmbedder()
    eligible = [q for q in sample if classify_query(q["query"]) == "general"]
    vecs = {id(q): emb.encode(q["query"]) for q in eligible}
    out = []
    for thr in thresholds:
        entries: list = []
        hits = false_hits = reachable = 0
        for q in eligible:
            v = vecs[id(q)]
            reachable += any(e["group"] == q["group"] for e in entries)
            best, score = None, -1.0
            for e in entries:
                s = float(vecs[id(e)] @ v)
                if s > score:
                    best, score = e, s
            if best is not None and score >= thr and (not guard or entities(best["query"]) == entities(q["query"])):
                hits += 1
                false_hits += best["group"] != q["group"]
            else:
                entries.append(q)
        n = len(eligible)
        out.append({"threshold": thr, "n": n, "hits": hits, "false_hits": false_hits,
                    "hit_rate": hits / n, "false_hit_rate": false_hits / n, "reachable": reachable / n,
                    "bypassed": len(sample) - n})
    return out
