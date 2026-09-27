"""memcore - agent memory from first principles: what an agent remembers, how it is written, retrieved,
consolidated and forgotten. Standard library + numpy; a scripted model and a hashing embedder keep every
number offline. Read the modules in this order: records, store, write, retrieve, budget, consolidate, forget,
agent, harness. The topic's PRIMER.md cites the function behind every number.
"""
from .agent import POISONED_PAGE, AuditEvent, MemoryAgent, UserTurn, fence, scripted_model
from .budget import (CACHE_MIN_TOKENS, GPUS, LLMS, LAYOUTS, Budget, BudgetExceeded, PrefixCache, Salts, billed_cached,
                     block_names, cache_salt, call_cost, compute_ttft, expected_cached_tokens, hit_rate, hits_per_turn,
                     prefill_seconds, token_ids, turn_cost)
from .consolidate import Consolidator, Crash, LeaseHeld, LeaseLost, plan_key, reflect
from .forget import DeletionReport, Surfaces, cap, expire, mentions, propagate, residue, retention
from .harness import (Question, Scenario, build_store, cluster_interval, compare_modes, evaluate, generate, knee,
                      read_answer, recall_vs_budget, summarize, wilson_interval)
from .records import DAY, HOUR, KINDS, SOURCE_TRUST, MemoryRecord, Scope, count_tokens
from .retrieve import Recall, minmax, pack, recency, retrieve, score
from .store import HashingEmbedder, MemoryStore, tokenize
from .write import (SLOTS, IdempotencyConflict, WritePolicy, WriteResult, Writer, extract, fact_text, idempotency_key,
                    read_facts, screen, slot_info)

__all__ = [
    "AuditEvent", "Budget", "BudgetExceeded", "CACHE_MIN_TOKENS", "Consolidator", "Crash", "DAY", "DeletionReport",
    "GPUS", "HOUR", "HashingEmbedder", "IdempotencyConflict", "KINDS", "LAYOUTS", "LLMS", "LeaseHeld", "LeaseLost",
    "MemoryAgent", "MemoryRecord", "MemoryStore", "POISONED_PAGE", "PrefixCache", "Question", "Recall", "SLOTS",
    "SOURCE_TRUST", "Salts", "Scenario", "Scope", "Surfaces", "UserTurn", "WritePolicy", "WriteResult", "Writer",
    "billed_cached", "block_names", "build_store", "cache_salt", "call_cost", "cap", "cluster_interval",
    "compare_modes", "compute_ttft", "count_tokens", "evaluate", "expected_cached_tokens", "expire", "extract",
    "fact_text", "fence", "generate", "hit_rate", "hits_per_turn", "idempotency_key", "knee", "mentions", "minmax",
    "pack", "plan_key", "prefill_seconds", "propagate", "read_answer", "read_facts", "recall_vs_budget", "recency",
    "reflect", "residue", "retention", "retrieve", "score", "screen", "scripted_model", "slot_info", "summarize",
    "token_ids", "tokenize", "turn_cost", "wilson_interval",
]
__version__ = "0.1.0"
