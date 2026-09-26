"""The Mistral primer and the capacity-plan doc quote the Mistral provider's plan: pin the text to the model."""

from pathlib import Path

from scalelab import capacity, mistral
from scalelab.mistral import Scenario, cost_per_call, plan

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_capacity_plan_doc_is_both_generators():
    text = (DOCS / "03-capacity-plan.md").read_text()
    assert text == capacity.plan_markdown() + "\n\n" + mistral.plan_section(), \
        "regenerate: { python -m scalelab.capacity; echo; python -m scalelab.mistral --section; } > docs/03-capacity-plan.md"


def test_mistral_primer_figures_are_the_plan():
    p = plan(Scenario())
    h, f = p["hosted"], p["self_hosted"]
    mix = h["cost_per_conversation"][h["planning_mix"]]
    small, medium = cost_per_call("mistral-small-2603", 5000, 200, 2700), cost_per_call("mistral-medium-3-5", 5000, 200, 2700)
    figures = [  # (text as the primer prints it, the model's value, the number that text stands for)
        ("$0.000505", small, 0.000505),
        ("$0.005355", medium, 0.005355),
        ("10.6 times", medium / small, 10.6),
        ("4.8 / 14.3 / 47.7 M TPM", round(p["tokens"]["average"]["total_tpm"] / 1e6, 1), 4.8),   # 4.77, printed to one decimal
        ("4.8 / 14.3 / 47.7 M TPM", p["tokens"]["peak"]["total_tpm"] / 1e6, 14.3),
        ("4.8 / 14.3 / 47.7 M TPM", p["tokens"]["incident"]["total_tpm"] / 1e6, 47.7),
        ("175 turns in flight", p["concurrency"]["max_inflight_for_limit"], 175),
        ("60 RPS, 19 M TPM and 209 B", h["limit_request"]["tokens_per_month"] / 1e9, 209),
        ("$39.7 k", h["monthly_usd"][h["planning_mix"]] / 1e3, 39.7),
        ("$55.2 k", f["monthly_usd"]["peak"] / 1e3, 55.2),
        ("293", f["max_inflight_turns_peak_fleet"], 293),
        ("$4.95", f["breakeven_gpu_usd_per_hour"], 4.95),
        ("1.39×", f["vs_hosted_by_price"]["aws on-demand"], 1.39),
        ("0.76×", f["vs_hosted_by_price"]["neocloud"], 0.76),
    ]
    text = (DOCS / "mistral" / "01-scaling-primer.md").read_text()
    for quoted, value, number in figures:
        assert quoted in text, quoted
        assert abs(value - number) <= 0.006 * number, (quoted, value)   # rounding only
    assert "4 / 11 / 34" in text and f["replicas"] == {"average": 4, "peak": 11, "incident": 34}
    assert h["limit_request"]["rps"] == 60 and h["limit_request"]["tpm"] == 19_000_000
    assert round(mix, 4) == 0.0131 and "$0.0131" in text
