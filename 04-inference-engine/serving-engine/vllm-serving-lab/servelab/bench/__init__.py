"""bench — an async OpenAI-compatible load generator with vLLM's metric definitions.

    from servelab.bench import random_requests, run_open_loop, SLO
    run = run_open_loop("http://127.0.0.1:8000", random_requests(100), rate=5)
    print(run.report(SLO(ttft_ms=500, tpot_ms=50)))
"""
from .client import RequestResult, SSEParser, stream_request
from .report import compare, curve, table
from .runner import BenchRun, closed_loop, open_loop, run_closed_loop, run_open_loop, run_sessions, run_sync, sessions
from .summary import SLO, Stat, Summary, littles_law, percentile, summarize
from .workload import AgentSession, Lengths, Request, agent_sessions, arrival_times, mixed_requests, random_requests

__all__ = [
    "RequestResult", "SSEParser", "stream_request", "compare", "curve", "table", "BenchRun", "closed_loop",
    "open_loop", "run_closed_loop", "run_open_loop", "run_sessions", "run_sync", "sessions", "SLO", "Stat",
    "Summary", "littles_law", "percentile", "summarize", "AgentSession", "Lengths", "Request", "agent_sessions",
    "arrival_times", "mixed_requests", "random_requests",
]
