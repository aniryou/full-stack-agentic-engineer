"""Reports carry provenance: every section is measured, simulated or illustrative."""
import json

import pytest

from moelab.report import Report


def test_report_labels_and_files(tmp_path):
    r = Report("EP on 2 x T4").add("ITL by layout", "simulated", "tp 9.2 ms", batch=1)
    r.add("bench", "illustrative", "", source="fixture")
    with pytest.raises(ValueError):
        r.add("x", "guessed")
    md, js = r.save(tmp_path)
    text = md.read_text()
    assert "[SIMULATED]" in text and "[ILLUSTRATIVE]" in text and "**batch**: 1" in text
    assert json.loads(js.read_text())["sections"][0]["label"] == "simulated"
