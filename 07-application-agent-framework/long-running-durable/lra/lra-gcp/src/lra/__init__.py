"""lra — Long-Running Agents on Google Cloud.

Public surface::

    from lra import Workflow, Next, Done, Wait, FanOut, StepFailed, ChildSpec, Budget
    from lra import Engine, Event, StepTask
    from lra.adapters.memory import InMemoryStateStore, InMemoryTaskQueue, InMemoryEventBus, FakeLLM, FakeClock, LocalRunner
    from lra.adapters.gcp import build_gcp_engine   # Firestore + Cloud Tasks + Pub/Sub + Gemini
"""

from .core.engine import Engine, SimulatedCrash, WorkflowRegistry
from .core.models import Budget, ChildSpec, Event, LLMResponse, LLMUsage, Run, RunStatus, StepRecord, StepTask
from .core.ports import ConflictError, LeaseHeldError
from .core.workflow import Done, FanOut, Next, StepContext, StepFailed, Wait, Workflow

__all__ = [
    "Budget", "ChildSpec", "ConflictError", "Done", "Engine", "Event", "FanOut", "LLMResponse", "LLMUsage",
    "LeaseHeldError", "Next", "Run", "RunStatus", "SimulatedCrash", "StepContext", "StepFailed", "StepRecord",
    "StepTask", "Wait", "Workflow", "WorkflowRegistry",
]
__version__ = "0.1.0"
