"""The illustrative fixtures are reproducible: tools/make_fixtures.py regenerates them byte for byte."""
import importlib.util
from pathlib import Path

from moelab import hooks

ROOT = Path(__file__).resolve().parents[1]


def test_make_fixtures_is_deterministic(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("make_fixtures", ROOT / "tools" / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(tmp_path) == 0
    made = sorted(p.name for p in tmp_path.iterdir())
    assert len(made) == 9
    for name in made:
        assert (tmp_path / name).read_bytes() == (Path(hooks.FIXTURES) / name).read_bytes(), name


def test_every_illustrative_file_says_so():
    for p in Path(hooks.FIXTURES).iterdir():
        if p.suffix in (".txt", ".log", ".json") and p.name != "tinymoe_curves.json":
            assert "sample output in the documented format (illustrative)" in p.read_text()[:400], p.name
