"""The record type, the token count and the hashing embedder (ragkit's, re-implemented)."""
import numpy as np
import pytest

from memlab.embedders import FALLBACK_LABEL, HashingEmbedder, cosine, tokenize
from memlab.records import MemoryRecord, count_tokens, default_deletion_key, outranks


def test_count_tokens_is_07_2s_rule():
    assert count_tokens("") == 0 and count_tokens("abc") == 1 and count_tokens("a" * 17) == 4


def test_record_defaults_and_validation():
    r = MemoryRecord("acme", "u1", "Home city: the user lives in Lisbon.", slot="home_city", created_at=100.0)
    assert r.deletion_key == "acme/u1/home_city" and r.valid_from == 100.0 and r.last_accessed == 100.0
    assert r.valid_at(100.0) and not r.valid_at(99.0)
    r.valid_to = 200.0
    assert r.valid_at(150.0) and not r.valid_at(200.0)
    for bad in ({"kind": "working"}, {"scope": "global"}, {"source": "web"}, {"trust": "admin"}, {"status": "gone"}):
        with pytest.raises(ValueError):
            MemoryRecord("acme", "u1", "x", **bad)
    assert default_deletion_key("t", "u", "address") == "t/u/address"


def test_trust_precedence():
    assert outranks("human", "user") and outranks("user", "tool") and outranks("tool", "inferred")
    assert not outranks("user", "user") and not outranks("inferred", "tool")


def test_hashing_embedder_pins_ragkits_buckets_and_cosines():
    e = HashingEmbedder(1024)
    import zlib
    buckets = {t: zlib.crc32(t.encode()) % 1024 for t in ("user", "lives", "in", "lisbon", "the", "moved", "to", "porto")}
    assert buckets == {"user": 585, "lives": 606, "in": 590, "lisbon": 181, "the": 486, "moved": 501, "to": 708,
                       "porto": 130}
    a = e.encode("user lives in lisbon")
    assert a.shape == (1024,) and a.dtype == np.float32 and abs(np.linalg.norm(a) - 1) < 1e-6
    assert round(cosine(a, e.encode("the user moved to porto")), 4) == 0.2236
    assert round(cosine(a, e.encode("Where does the user live?")), 4) == 0.2236      # "live" != "lives": by design
    assert e.encode(["a b", "c"]).shape == (2, 1024) and not e.encode("!!!").any()
    assert e.label == FALLBACK_LABEL == "hashing embedder (T0 fallback; not semantic)"
    assert tokenize("Rua das Flores, 12!") == ["rua", "das", "flores", "12"]
