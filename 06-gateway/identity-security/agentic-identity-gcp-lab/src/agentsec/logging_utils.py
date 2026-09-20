"""Quieten chatty libraries for notebooks and demos."""

from __future__ import annotations

import logging
import warnings


def quiet_logs(level: int = logging.WARNING) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in (
        "google_adk",
        "mcp",
        "httpx",
        "httpcore",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "google.adk",
    ):
        logging.getLogger(name).setLevel(level)
    logging.getLogger("google_adk.google.adk.models.google_llm").setLevel(logging.ERROR)
    logging.getLogger("google_adk.google.adk.flows.llm_flows.base_llm_flow").setLevel(logging.ERROR)
