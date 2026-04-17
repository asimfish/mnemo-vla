"""Persistent (cross-session) memory for MNEMO-VLA.

The persistent memory layer stores EpisodeSummary objects across sessions
and provides top-K retrieval at the start of a new episode. This is how
MNEMO implements "remember where I put the knife last time" style recall.

Public API:
    EpisodeSummary, EpisodeFact, summarize_rule_based, chain_summaries
    MemoryStore, RetrievalHit, HashingEmbedder, StackedEmbedder
    build_dataset, default_query_set, run_benchmark, BenchmarkResult
"""

from .episode_summary import (
    EpisodeFact,
    EpisodeSummary,
    chain_summaries,
    summarize_rule_based,
)
from .embedders import SentenceTransformerEmbedder, StackedEmbedder
from .memory_store import HashingEmbedder, MemoryStore, RetrievalHit
from .synthetic_bench import (
    BenchmarkResult,
    build_dataset,
    default_query_set,
    run_benchmark,
)

__all__ = [
    "EpisodeSummary",
    "EpisodeFact",
    "summarize_rule_based",
    "chain_summaries",
    "MemoryStore",
    "RetrievalHit",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "StackedEmbedder",
    "build_dataset",
    "default_query_set",
    "run_benchmark",
    "BenchmarkResult",
]
