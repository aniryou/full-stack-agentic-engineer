"""The T0 fallback: without sentence-transformers the factories return labelled
lexical stand-ins, so every notebook runs with numpy alone."""
import sys

import numpy as np
import pytest

import ragkit.embed as embed


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # import now raises ImportError
    monkeypatch.delenv("RAGKIT_EMBEDDER", raising=False)
    monkeypatch.setattr(embed, "_CACHE", {})
    monkeypatch.setattr(embed, "_ANNOUNCED", set())


def test_factory_falls_back_and_says_so(no_sentence_transformers, capsys):
    emb = embed.get_embedder()
    assert isinstance(emb, embed.HashingEmbedder)
    assert emb.model_name == "hashing embedder (T0 fallback; not semantic)"
    assert "hashing embedder (T0 fallback; not semantic)" in capsys.readouterr().out
    assert embed.get_embedder() is emb                        # cached, announced once
    assert capsys.readouterr().out == ""


def test_env_var_forces_the_fallback(monkeypatch):
    monkeypatch.setenv("RAGKIT_EMBEDDER", "hashing")
    monkeypatch.setattr(embed, "_CACHE", {})
    assert not embed.have_sentence_transformers()
    assert isinstance(embed.get_embedder(), embed.HashingEmbedder)


def test_hashing_vectors_are_unit_deterministic_and_lexical(no_sentence_transformers):
    emb = embed.get_embedder()
    one = emb.encode("Error ERR_4290 means rate limited")
    batch = emb.encode(["Error ERR_4290 means rate limited", "annual leave policy", ""])
    assert one.shape == (emb.dim,) and one.dtype == np.float32
    assert batch.shape == (3, emb.dim)
    assert np.allclose(batch[0], one)                         # same text, same vector (crc32, not salted hash)
    assert np.isclose(np.linalg.norm(batch[0]), 1.0) and not batch[2].any()
    q = emb.encode("what does ERR_4290 mean")
    assert float(q @ batch[0]) > float(q @ batch[1]) == 0.0   # shared tokens -> similar; none -> orthogonal
    # Not semantic: synonyms with no shared token are orthogonal.
    assert float(emb.encode("holidays") @ emb.encode("vacation")) == 0.0
    assert emb.encode([]).shape == (0, emb.dim)


def test_cross_encoder_fallback(no_sentence_transformers, capsys):
    rr = embed.get_cross_encoder()
    assert "token-overlap reranker (T0 fallback" in capsys.readouterr().out
    s = rr.predict([("rate limit error", "the API rate limit error is ERR_4290"), ("rate limit error", "annual leave")])
    assert s.dtype == np.float32 and s[0] == 1.0 and s[1] == 0.0
