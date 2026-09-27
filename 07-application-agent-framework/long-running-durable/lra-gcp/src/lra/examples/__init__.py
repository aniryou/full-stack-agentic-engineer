"""Example workflows used by the services, tests and notebooks."""

from .procurement_saga import procurement
from .research_pipeline import research_pipeline, research_subtopic

ALL_WORKFLOWS = [research_pipeline, research_subtopic, procurement]
__all__ = ["ALL_WORKFLOWS", "procurement", "research_pipeline", "research_subtopic"]
