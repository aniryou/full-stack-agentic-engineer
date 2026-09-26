"""servelab — serve, measure, size and tune a real inference engine (vLLM).

Modules, in the order the notebooks use them:

    sizing        config.json + GPU + flags -> KV blocks and max concurrency (before you start vLLM)
    bench         async OpenAI-compatible load generator; TTFT/ITL/TPOT/E2E/goodput as vLLM defines them
    metrics       parse vLLM's Prometheus /metrics; histogram quantiles; windows between scrapes
    tune          sweep engine flags against an SLO on a pluggable backend (fake, vLLM, any URL)
    fakeserver    a fake vLLM (OpenAI API + vLLM metrics) over fake_engine: the T0 target
    fake_engine   the engine timing emulator: scheduler, KV blocks, prefix cache, roofline steps
    env           what is available here (GPU, vLLM, a server URL) -> which tier runs
"""
__version__ = "0.1.0"

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
# Illustrative samples in vLLM's formats (a /metrics page at two times, a startup log): not measurements.
SAMPLES_DIR = DATA_DIR / "samples"

from . import env, metrics, sizing  # light modules; bench/fakeserver/tune import aiohttp on demand

__all__ = ["env", "metrics", "sizing", "DATA_DIR", "SAMPLES_DIR", "__version__"]
