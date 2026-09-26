"""The command line gives the quick answers the README promises, offline and in seconds."""
import pytest

from moelab.__main__ import main
from moelab import hooks


@pytest.mark.parametrize("argv,expect", [
    (["env"], "tier"),
    (["models"], "OLMoE-1B-7B-0924"),
    (["fit", "--model", "olmoe-1b-7b", "--gpu", "T4"], "--cpu-offload-gb 3"),
    (["touched", "--experts", "8", "--top-k", "2", "--batch", "1", "16"], "7.9"),
    (["stream", "--batch", "1", "8"], "simulated"),
    (["ep", "--batch", "1", "32"], "--enable-expert-parallel"),
    (["trace"], "illustrative"),
])
def test_cli(argv, expect, capsys):
    assert main(argv) == 0
    assert expect in capsys.readouterr().out


def test_cli_bench_parse(capsys):
    main(["bench-parse", str(hooks.FIXTURES / "bench_serve_olmoe_2xT4_tp_c16.txt")])
    assert "mean_itl_ms" in capsys.readouterr().out
