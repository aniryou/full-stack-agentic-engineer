"""Records (the typed memory) and the store (partitions, a flat index that deletes, the hashing embedder)."""
import numpy as np
import pytest

from memcore import DAY, HashingEmbedder, MemoryRecord, MemoryStore, Scope, count_tokens
from memcore.store import tokenize

ALICE, BOB = Scope("acme", "alice"), Scope("acme", "bob")


def test_count_tokens_is_len_over_four():
    assert [count_tokens(t) for t in ("", "abc", "abcd", "a" * 41)] == [0, 1, 1, 10]


def test_record_defaults_and_validation():
    r = MemoryRecord("The user's home city is Lisbon.", "semantic", ALICE, "user", key="home_city", value="Lisbon",
                     created_at=2 * DAY)
    assert r.valid_from == r.last_accessed == 2 * DAY and r.deletion_key == "acme/alice" and len(r.id) == 12
    assert r.render() == "[semantic, day 2, user] The user's home city is Lisbon." and r.tokens == 13
    assert r.is_valid(3 * DAY) and not r.is_valid(DAY)
    r.valid_to = 5 * DAY
    assert r.is_valid(4.9 * DAY) and not r.is_valid(5 * DAY) and r.render().startswith("[semantic, day 2-5,")
    with pytest.raises(ValueError):
        MemoryRecord("x", "working", ALICE, "user")
    with pytest.raises(ValueError):
        MemoryRecord("x", "semantic", ALICE, "web")


def test_ttl_expiry():
    r = MemoryRecord("temp", "episodic", ALICE, "user", created_at=0, ttl_s=DAY)
    assert not r.expired(DAY - 1) and r.expired(DAY)


def test_hashing_embedder_is_ragkits_recipe():
    e = HashingEmbedder()
    v = e.encode("user lives in lisbon")
    assert v.shape == (1024,) and np.isclose(np.linalg.norm(v), 1.0)
    assert np.flatnonzero(v).tolist() == sorted([585, 606, 590, 181])
    assert round(float(v @ e.encode("the user moved to porto")), 4) == 0.2236          # 1 / sqrt(4 * 5)
    assert round(float(v @ e.encode("Where does the user live?")), 4) == 0.2236        # 'live' != 'lives'
    assert e.encode(["a", "b"]).shape == (2, 1024) and e.encode([]).shape == (0, 1024)
    assert tokenize("The user's CITY!") == ["the", "user", "s", "city"]


def test_search_never_crosses_the_partition():
    s = MemoryStore()
    s.put(MemoryRecord("The user's home city is Lisbon.", "semantic", ALICE, "user"))
    s.put(MemoryRecord("The user's home city is Porto.", "semantic", BOB, "user"))
    got = s.search(ALICE, "home city", k=10)
    assert [r.text for r, _ in got] == ["The user's home city is Lisbon."]
    assert s.search(Scope("other", "alice"), "home city") == []                    # tenant is part of the key


def test_delete_removes_record_vector_and_postings():
    s = MemoryStore()
    keep = s.put(MemoryRecord("The user's pet is cat.", "semantic", ALICE, "user"))
    gone = s.put(MemoryRecord("The user's home city is Lisbon.", "semantic", ALICE, "user"))
    before = s.counts(ALICE)
    out = s.delete(ALICE, gone.id)
    assert out == {"records": 1, "vectors": 1, "fulltext_postings": 7}     # the user s home city is lisbon
    assert s.counts(ALICE) == {"records": 1, "vectors": 1, "fulltext_postings": before["fulltext_postings"] - 7}
    assert s.fulltext_search(ALICE, "lisbon") == set() and s.fulltext_search(ALICE, "cat") == {keep.id}
    assert [r.id for r, _ in s.search(ALICE, "Lisbon", k=5)] == [keep.id]          # no tombstone left to match
    assert s.delete(ALICE, gone.id) == {"records": 0, "vectors": 0, "fulltext_postings": 0}


def test_put_same_id_replaces_all_three_copies():
    s = MemoryStore()
    r = s.put(MemoryRecord("The user's pet is cat.", "semantic", ALICE, "user", id="fixed"))
    s.put(r.copy(text="The user's pet is dog."))
    assert s.counts(ALICE)["records"] == 1 and s.counts(ALICE)["vectors"] == 1
    assert s.fulltext_search(ALICE, "cat") == set() and s.fulltext_search(ALICE, "dog") == {"fixed"}
