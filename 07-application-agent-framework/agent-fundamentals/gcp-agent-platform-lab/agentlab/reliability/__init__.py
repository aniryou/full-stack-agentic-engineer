"""Reliability mechanisms for agent systems (notebook 10): retries, breakers, bulkheads, deadlines, fallbacks.

The same mechanisms at fleet scale are the scaling primer's §5.2
(06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md).
"""
from .breaker import (BreakerMetrics, BreakerState, Bulkhead, BulkheadFull, CircuitBreaker, CircuitOpen, Deadline,
                      DeadlineExceeded, per_hop_budget, with_deadline)
from .fallback import (FallbackChain, FallbackExhausted, FallbackResult, GracefulTool, Step, cached_answer_step,
                       enqueue_followup_step, lab_fallback_chain, model_step)
from .retry import (RETRYABLE_STATUSES, IdempotentCall, PermanentError, RetryableError, RetryPolicy, backoff_schedule,
                    classify_http, is_retryable, retry, with_retry)

__all__ = [
    "RETRYABLE_STATUSES", "IdempotentCall", "PermanentError", "RetryableError", "RetryPolicy", "backoff_schedule",
    "classify_http", "is_retryable", "retry", "with_retry",
    "BreakerMetrics", "BreakerState", "Bulkhead", "BulkheadFull", "CircuitBreaker", "CircuitOpen", "Deadline",
    "DeadlineExceeded", "per_hop_budget", "with_deadline",
    "FallbackChain", "FallbackExhausted", "FallbackResult", "GracefulTool", "Step", "cached_answer_step",
    "enqueue_followup_step", "lab_fallback_chain", "model_step",
]
