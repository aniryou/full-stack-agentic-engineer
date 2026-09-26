"""Reports label every section with where its numbers came from."""
import json

import pytest

from quantlab import report


def test_sections_carry_their_source(tmp_path):
    r = report.Report("t").add("speed", "simulated", [{"scheme": "fp8", "tpot_ms": 36.1}])
    r.add("eval", "sample", [{"task": "gsm8k", "value": 0.3}])
    md = r.to_markdown()
    assert "SIMULATED" in md and "sample output in the documented format (illustrative)" in md
    assert "| fp8 | 36.1 |" in md
    md_path, js_path = r.save(tmp_path / "x")
    assert json.loads(js_path.read_text())["sections"][0]["source"] == "simulated" and md_path.exists()
    with pytest.raises(KeyError):
        r.add("x", "vibes", [])


def test_number_formatting():
    assert report.fmt(1234.5) == "1,234" or report.fmt(1234.5) == "1,235"
    assert report.fmt(0.0012) == "1.20e-03" and report.fmt(12.345) == "12.3" and report.fmt(float("nan")) == "-"
