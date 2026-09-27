"""Configuration (references resolve, env expansion) and virtual keys (hashed at rest, scoped, revocable)."""
import pytest

from gwlab.gateway.config import load_config
from gwlab.gateway.keys import KeyError401, KeyStore, ProviderKeys, bearer, hash_key
from gwlab.gateway.store import Store


def test_bundled_configs_load(monkeypatch):
    lab = load_config("lab")
    assert set(lab.aliases) == {"chat", "chat-strong", "chat-cheap"} and lab.guardrails.input == "off"
    assert lab.models["bolt/haiku"].price_ref == "claude-haiku-4-5"
    monkeypatch.setenv("GWLAB_VLLM_URL", "http://gpu-box:8000")
    v = load_config("vllm")
    assert v.providers["local"].base_url == "http://gpu-box:8000" and v.models["local/llm"].price_ref == "self-hosted"
    assert v.aliases["chat"].targets == ["local/llm", "acme/fast"]         # vLLM first, a fake as the fallback


@pytest.mark.parametrize("patch, msg", [
    ({"aliases": {"chat": {"targets": ["nope/model"]}}}, "unknown target"),
    ({"tenants": {"team-a": {"aliases": ["ghost"]}}}, "unknown alias"),
    ({"aliases": {"chat": {"targets": ["acme/fast"], "policy": "random"}}}, "policy"),
    ({"providers": {"acme": {"dialect": "gemini"}}}, "dialect"),
    ({"models": {"acme/fast": {"price": "no-such-row"}}}, "price row"),
    ({"limits": {"mode": "maybe"}}, "limits.mode"),
])
def test_bad_references_are_rejected(patch, msg):
    with pytest.raises(ValueError, match=msg):
        load_config("lab", patch)


def test_yaml_off_is_a_placement_not_false():
    cfg = load_config("gateway: {}\nproviders: {}\nmodels: {}\naliases: {}\ntenants: {}\nguardrails: {input: off, output: window}\n")
    assert cfg.guardrails.input == "off" and cfg.guardrails.output == "window"


def test_keys_are_hashed_scoped_and_revocable():
    store = Store()
    ks = KeyStore(store)
    secret, k = ks.issue("team-a", aliases=["chat"], rpm=10, budget_usd=1.0)
    raw = str(store.query("SELECT * FROM keys"))
    assert secret not in raw and hash_key(secret) in raw                    # only the SHA-256 is stored
    assert ks.verify(secret).tenant == "team-a" and k.allows("chat") and not k.allows("chat-strong")
    for bad in (None, "", "gwk_nope"):
        with pytest.raises(KeyError401):
            ks.verify(bad)
    assert ks.revoke(k.key_id) and not ks.revoke(k.key_id)
    with pytest.raises(KeyError401, match="revoked"):
        ks.verify(secret)


def test_keys_expire():
    now = [1000.0]
    ks = KeyStore(Store(), clock=lambda: now[0])
    secret, _ = ks.issue("t", ttl_s=60)
    assert ks.verify(secret)
    now[0] += 61
    with pytest.raises(KeyError401, match="expired"):
        ks.verify(secret)


def test_bearer_and_provider_key_rotation():
    assert bearer({"Authorization": "Bearer abc"}) == "abc" and bearer({"authorization": "bearer x"}) == "x"
    assert bearer({"Authorization": "Basic abc"}) is None and bearer({}) is None
    pk = ProviderKeys({"acme": "old"})
    before = pk.fingerprint("acme")
    pk.rotate("acme", "new")
    assert pk.get("acme") == "new" and pk.previous["acme"] == "old" and pk.fingerprint("acme") != before
    assert pk.retire("acme") == "old" and "acme" not in pk.previous
