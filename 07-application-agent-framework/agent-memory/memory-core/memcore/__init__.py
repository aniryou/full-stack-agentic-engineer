"""memcore - agent memory from first principles: what an agent remembers, how it is written, retrieved,
consolidated and forgotten. Standard library + numpy; a scripted model and a hashing embedder keep every
number offline. Read the modules in this order: records, store, write, retrieve, budget, consolidate, forget,
agent, harness. The topic's PRIMER.md cites the function behind every number.
"""
from .agent import AuditEvent, MemoryAgent, UserTurn, fence, read_answer, scripted_model
from .budget import (GPUS, LLMS, LAYOUTS, Budget, BudgetExceeded, PrefixCache, block_names, call_cost,
                     compute_ttft, expected_cached_tokens, hit_rate, hits_per_turn, prefill_seconds, token_ids,
                     turn_cost)
from .consolidate import Consolidator, Crash, LeaseHeld, plan_key, reflect
from .forget import DeletionReport, Surfaces, cap, expire, propagate, residue, retention
from .harness import (Question, Scenario, build_store, compare_modes, evaluate, generate, knee,
                      recall_vs_budget, summarize, wilson_interval)
from .records import DAY, HOUR, KINDS, SOURCE_TRUST, MemoryRecord, Scope, count_tokens
from .retrieve import Recall, minmax, pack, recency, retrieve, score
from .store import HashingEmbedder, MemoryStore, tokenize
from .write import SLOTS, WritePolicy, WriteResult, Writer, extract, fact_text, idempotency_key, read_facts, screen

__version__ = "0.1.0"
