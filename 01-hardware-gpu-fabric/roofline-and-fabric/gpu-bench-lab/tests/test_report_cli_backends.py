"""Reports round-trip, the CLI works on saved output, and PyTorch stays optional."""
import json
import sys

import pytest

from gpubench import cli, topo
from gpubench.backends import BackendUnavailable, get_backend
from gpubench.report import Report


def test_auto_backend_falls_back_to_numpy_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)       # `import torch` now raises ImportError
    assert get_backend("auto", verbose=False).name == "numpy"
    with pytest.raises(BackendUnavailable, match="PyTorch is not installed"):
        get_backend("torch")


def test_tiny_suite_writes_a_report_that_round_trips(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cli.main(["run", "--backend", "numpy", "--tiny", "--out", str(tmp_path)]) == 0
    js = sorted(tmp_path.glob("*.json"))
    assert len(js) == 1 and js[0].with_suffix(".md").exists()
    rep = Report.load(js[0])
    ops = {m.op for m in rep.measurements}
    assert {"gemm", "stream.triad", "memcpy", "load.read"} <= ops
    assert rep.meta["tiny"] is True and "roofline" in rep.analyses and "alpha_beta" in rep.analyses
    again = Report.from_dict(json.loads(json.dumps(rep.to_dict())))
    assert [m.cost for m in again.measurements] == [m.cost for m in rep.measurements]
    md = rep.to_markdown()
    assert "### GEMM" in md and "STREAM convention" in md
    assert "**WARNING: tiny run: plumbing check only" in md and "mode `tiny`" in md   # never passes for a real run
    assert md.index("WARNING") < md.index("## Machine")
    full = Report(meta={"quick": False, "backend": "numpy"})
    assert full.mode == "full" and "WARNING" not in full.to_markdown()


def test_cli_topo_and_inventory_on_saved_output(tmp_path, capsys):
    f = tmp_path / "topo.txt"
    f.write_text(topo.load_fixture("pcie-4gpu-2socket"))
    assert cli.main(["topo", str(f), "--tp", "2"]) == 0
    assert "best 2-GPU group: GPU0, GPU1" in capsys.readouterr().out
    from gpubench import inventory
    g = tmp_path / "inv.csv"
    g.write_text(inventory.load_fixture("hgx-h100-8gpu"))
    assert cli.main(["inventory", str(g)]) == 0
    assert "x8 of x16" in capsys.readouterr().out
