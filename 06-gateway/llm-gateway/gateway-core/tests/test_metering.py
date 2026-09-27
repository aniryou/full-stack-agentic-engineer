"""§5: usage is the bill; blended $/M; self-hosted $/M; chargeback; reconciliation."""
import pytest

from gwcore import metering as M
from gwcore.providers import CATALOGUE


def test_the_scaling_primers_call():
    assert M.price_call("gemini-3.5-flash", 5000, 350, 2700) == pytest.approx(0.007005)     # 0.00345 + 0.0004 + 0.00315
    assert M.price_call("gemini-3.5-flash", 5000, 350) == pytest.approx(0.01065)


def test_cache_writes_bill_at_their_own_price():
    """Anthropic's cache_creation_input_tokens bill at 1.25/M on claude-haiku-4-5, not the 1.00/M input rate."""
    from gwcore.providers import normalize_usage
    u = normalize_usage("anthropic", {"input_tokens": 1000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 10_000,
                                      "output_tokens": 0})
    assert M.price_call("claude-haiku-4-5", u["prompt_tokens"], 0, u["cached_tokens"], u["cache_write_tokens"]) == \
        pytest.approx((1000 * 1.00 + 10_000 * 1.25) / 1e6)
    row = M.row_from_usage("r", "t", "k", "claude-haiku-4-5", u)
    assert row.cache_write_tokens == 10_000 and row.cost == pytest.approx(0.01350)             # not $0.01100
    assert M.price_call("gemini-3.5-flash", 5000, 350, 0, 1000) == M.price_call("gemini-3.5-flash", 5000, 350)  # no write price: input


def test_thinking_bills_as_output():
    assert M.price_call("gemini-3.5-flash", 5000, 1550, 2700) == pytest.approx(0.017805)
    assert round(0.017805 / 0.007005, 2) == 2.54


@pytest.mark.parametrize("model,call,per_m", [("gemini-3.5-flash", 0.007005, 1.3093), ("gpt-5.4-mini", 0.0035025, 0.6547),
                                              ("claude-haiku-4-5", 0.00432, 0.8075), ("gemini-3.5-flash-lite", 0.001646, 0.3077)])
def test_blended_cost_per_million(model, call, per_m):
    assert M.price_call(model, 5000, 350, 2700) == pytest.approx(call)
    assert round(M.cost_per_million(model, 5000, 350, 2700), 4) == per_m


def test_self_hosted_rows():
    assert round(M.self_hosted_per_million(3.7, 6846.515877965477), 4) == 0.1501
    assert round(M.self_hosted_per_million(0.70, 294.3945617414261), 4) == 0.6605
    assert round(M.self_hosted_per_million(3.7, 6846.515877965477, utilisation=0.6), 3) == 0.250


def test_ledger_rows_totals_and_reconcile():
    led = M.Ledger()
    led.add(M.row_from_usage("r1", "acme", "k1", "gpt-5.4-mini", {"prompt_tokens": 5000, "completion_tokens": 350, "cached_tokens": 2700}))
    led.add(M.row_from_usage("r2", "acme", "k1", "gpt-5.4-mini", {"prompt_tokens": 100, "completion_tokens": 20}, estimated=True))
    led.add(M.row_from_usage("r3", "globex", "k2", "lab/llm", {"prompt_tokens": 10, "completion_tokens": 5}))
    t = led.totals()
    assert t["acme"]["requests"] == 2 and t["acme"]["estimated"] == 1 and t["acme"]["cost"] == pytest.approx(0.0035025 + 0.000165)
    assert t["globex"]["cost"] == 0.0 and CATALOGUE["lab/llm"].price is None      # self-hosted: charged back, not priced
    rep = led.reconcile({"gpt-5.4-mini": {"prompt_tokens": 5100, "completion_tokens": 400}})
    assert rep[("gpt-5.4-mini", "prompt_tokens")]["ok"] and not rep[("gpt-5.4-mini", "completion_tokens")]["ok"]
    assert rep[("gpt-5.4-mini", "completion_tokens")]["diff"] == 30


def test_chargeback_by_tokens_vs_gpu_seconds():
    u = {"rag": {"prompt_tokens": 90_000_000, "completion_tokens": 1_000_000},
         "thinking": {"prompt_tokens": 5_000_000, "completion_tokens": 10_000_000}}
    by_tok, by_gpu = M.chargeback(u, 10_000, by="tokens"), M.chargeback(u, 10_000, by="gpu_seconds")
    assert round(by_tok["rag"], 2) == 8584.91 and round(by_gpu["rag"], 2) == 4892.57
    assert sum(by_gpu.values()) == pytest.approx(10_000)
