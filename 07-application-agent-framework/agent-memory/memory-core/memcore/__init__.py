"""memcore - agent memory from first principles: what an agent remembers, how it is written, retrieved,
consolidated and forgotten. Standard library + numpy; a scripted model and a hashing embedder keep every
number offline. Read the modules in this order: records, store, write, retrieve, budget, consolidate, forget,
agent, harness. The topic's PRIMER.md cites the function behind every number.
"""
from .agent import POISONED_PAGE, AuditEvent, MemoryAgent, UserTurn, fence, scripted_model
from .budget import (GPUS, LLMS, LAYOUTS, Budget, BudgetExceeded, PrefixCache, block_names, call_cost,
                     compute_ttft, expected_cached_tokens, hit_rate, hits_per_turn, prefill_seconds, token_ids,
                     turn_cost)
from .consolidate import Consolidator, Crash, LeaseHeld, plan_key, reflect
from .forget import DeletionReport, Surfaces, cap, expire, propagate, residue, retention
from .harness import (Question, Scenario, build_store, compare_modes, evaluate, generate, knee, read_answer,
                      recall_vs_budget, summarize, wilson_interval)
from .records import DAY, HOUR, KINDS, SOURCE_TRUST, MemoryRecord, Scope, count_tokens
from .retrieve import Recall, minmax, pack, recency, retrieve, score
from .store import HashingEmbedder, MemoryStore, tokenize
from .write import SLOTS, WritePolicy, WriteResult, Writer, extract, fact_text, idempotency_key, read_facts, screen

__all__ = [
    "AuditEvent", "Budget", "BudgetExceeded", "Consolidator", "Crash", "DAY", "DeletionReport", "GPUS", "HOUR",
    "HashingEmbedder", "KINDS", "LAYOUTS", "LLMS", "LeaseHeld", "MemoryAgent", "MemoryRecord", "MemoryStore",
    "POISONED_PAGE", "PrefixCache", "Question", "Recall", "SLOTS", "SOURCE_TRUST", "Scenario", "Scope",
    "Surfaces", "UserTurn", "WritePolicy", "WriteResult", "Writer", "block_names", "build_store", "call_cost",
    "cap", "compare_modes", "compute_ttft", "count_tokens", "evaluate", "expected_cached_tokens", "expire",
    "extract", "fact_text", "fence", "generate", "hit_rate", "hits_per_turn", "idempotency_key", "knee",
    "minmax", "pack", "plan_key", "prefill_seconds", "propagate", "read_answer", "read_facts",
    "recall_vs_budget", "recency", "reflect", "residue", "retention", "retrieve", "score", "screen",
    "scripted_model", "summarize", "token_ids", "tokenize", "turn_cost", "wilson_interval",
]
__version__ = "0.1.0"
