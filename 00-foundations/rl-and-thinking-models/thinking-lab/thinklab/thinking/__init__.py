"""thinking — talk to a thinking model: the request switches, the response fields, budgets and
test-time compute, over a generated eval set of verifiable problems."""
from .client import Completion, ThinkingClient, request_body  # noqa: F401
from .evalset import Problem, make_evalset, verify  # noqa: F401

__all__ = ["Completion", "ThinkingClient", "request_body", "Problem", "make_evalset", "verify"]
