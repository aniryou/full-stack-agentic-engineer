import pytest

from agentlab.estimation import (PRICES, Scenario, Segment, ci_half_width, compounded_reliability, human_bytes,
                               latency_budget, littles_law, schedule, throughput_for_backlog, token_cost,
                               vector_store_bytes, waterfall_text)


def support(**overrides) -> Scenario:
    """Notebook 12's support scenario: 50k conversations/day × 8 calls × (6k in, 400 out)."""
    base = dict(name="support", units_per_day=50_000, calls_per_unit=8, in_tokens=6_000, out_tokens=400)
    base.update(overrides)
    return Scenario(**base)


# --------------------------------------------------------------------- cost
def test_token_cost_components_cache_batch_and_long_context_tier():
    pro = PRICES["gemini-3.1-pro"]
    assert token_cost(6_000, 400, pro) == pytest.approx(0.0168)
    assert token_cost(6_000, 400, pro, cached_share=4_000 / 6_000) == pytest.approx(0.0096)
    assert token_cost(1_000, 100, PRICES["gemini-3.5-flash-lite"], batch=True) == pytest.approx(0.000275)
    assert token_cost(300_000, 1_000, pro) == pytest.approx(1.218)      # long-context tier: 4.00 in / 18.00 out
    assert token_cost(100_000, 1_000, pro) == pytest.approx(0.212)
    with pytest.raises(ValueError):
        token_cost(1, 1, pro, cached_share=1.5)


def test_worked_scenario_a_all_pro():
    a = support()
    assert round(a.daily_cost(), 2) == 6_720.00
    assert round(a.cost_per_unit(), 4) == 0.1344
    assert round(a.annual_cost(), 2) == 2_452_800.00


def test_worked_scenario_b_all_flash_and_c_mix():
    assert round(support(model_mix={"gemini-3-flash": 1.0}).daily_cost(), 2) == 1_680.00
    assert round(support(model_mix={"gemini-3-flash": 0.7, "gemini-3.1-pro": 0.3}).daily_cost(), 2) == 3_192.00


def test_worked_scenario_d_context_caching():
    assert round(support(cached_share=4_000 / 6_000).daily_cost(), 2) == 3_840.00


def test_worked_peak_rates_and_concurrency():
    a = support()
    assert a.calls_per_day == 400_000
    assert round(a.avg_calls_per_sec(), 2) == 4.63
    assert round(a.peak_calls_per_sec()) == 14
    assert a.peak_input_tpm() == pytest.approx(5_000_000)
    assert round(a.peak_output_tps()) == 5_556
    assert round(a.concurrency()) == 56 and littles_law(a.peak_calls_per_sec(), 4.0) == pytest.approx(55.56, abs=0.01)


def test_worked_document_backlog_online_batch_and_throughput():
    # 20M documents × 3 calls × 1,000 input tokens, 300 output tokens per document (100 per call)
    backlog = Scenario("backlog", 20_000_000, 3, 1_000, 100, model_mix={"gemini-3.5-flash-lite": 1.0})
    assert round(backlog.daily_cost(), 2) == 33_000.00
    assert round(Scenario("batch", 20_000_000, 3, 1_000, 100, model_mix={"gemini-3.5-flash-lite": 1.0}, batch=True).daily_cost(), 2) == 16_500.00
    rate = throughput_for_backlog(60e9, 30)
    assert round(rate.tokens_per_s, 2) == 23_148.15 and round(rate.tokens_per_min) == 1_388_889


def test_scenario_validation_and_report():
    with pytest.raises(ValueError):
        support(model_mix={"gemini-3-flash": 0.5})
    with pytest.raises(KeyError):
        support(model_mix={"gemini-9": 1.0})
    report = support().report()
    assert "$6,720.00" in report and "$0.1344" in report and "5,000,000" in report and "55.6" in report


# ------------------------------------------------------------------ latency
SEQUENTIAL = [Segment("plan", 1.1), Segment("lookup_a", 0.4), Segment("lookup_b", 0.5), Segment("answer", 2.7)]
PARALLEL = [Segment("plan", 1.1), Segment("lookup_a", 0.4, "tools"), Segment("lookup_b", 0.5, "tools"), Segment("answer", 2.7)]


def test_worked_latency_sequential_vs_parallel_tools():
    seq = latency_budget(SEQUENTIAL, first_token_segment="answer", first_token_offset_s=0.7)
    par = latency_budget(PARALLEL, first_token_segment="answer", first_token_offset_s=0.7)
    assert (round(seq.total_s, 2), round(seq.first_token_s, 2)) == (4.7, 2.7)
    assert (round(par.total_s, 2), round(par.first_token_s, 2)) == (4.3, 2.3)
    assert round(seq.p95_s, 3) == 7.05
    assert round(latency_budget(SEQUENTIAL).first_token_s, 2) == 4.7, "without streaming the first token is the last"
    with pytest.raises(KeyError):
        latency_budget(SEQUENTIAL, first_token_segment="nope")


def test_schedule_places_parallel_groups_side_by_side():
    spans = {s.name: (round(s.start, 2), round(s.end, 2)) for s in schedule(PARALLEL)}
    assert spans == {"plan": (0.0, 1.1), "lookup_a": (1.1, 1.5), "lookup_b": (1.1, 1.6), "answer": (1.6, 4.3)}
    text = waterfall_text(PARALLEL, first_token_segment="answer", first_token_offset_s=0.7)
    assert text.count("█") > 0 and "first token 2.30s" in text and text.splitlines()[0].startswith("plan")


# ----------------------------------------------------------------- capacity
def test_vector_store_bytes_and_human_units():
    assert vector_store_bytes(5_000_000, 768, index_overhead=1.0) == 15_360_000_000
    assert human_bytes(vector_store_bytes(5_000_000, 768, index_overhead=1.0)) == "15.36 GB"
    assert human_bytes(vector_store_bytes(5_000_000, 768)) == "23.04 GB"
    assert human_bytes(512) == "512 B" and human_bytes(2.5e12) == "2.50 TB"


def test_ci_half_width_and_compounded_reliability():
    assert round(ci_half_width(100), 3) == 0.098
    assert round(ci_half_width(400), 3) == 0.049
    assert round(ci_half_width(1_000), 3) == 0.031
    assert round(compounded_reliability(0.99, 10), 3) == 0.904
    assert round(compounded_reliability(0.999, 10), 3) == 0.990
    with pytest.raises(ValueError):
        ci_half_width(0)
