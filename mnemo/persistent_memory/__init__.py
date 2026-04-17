"""Persistent (cross-session) memory for MNEMO-VLA.

The persistent memory layer stores EpisodeSummary objects across sessions
and provides top-K retrieval at the start of a new episode. This is how
MNEMO implements "remember where I put the knife last time" style recall.

Public API:
    EpisodeSummary, EpisodeFact, summarize_rule_based, chain_summaries
    MemoryStore, RetrievalHit, HashingEmbedder
"""

from .episode_summary import (
    EpisodeFact,
    EpisodeSummary,
    chain_summaries,
    summarize_rule_based,
)
from .memory_store import HashingEmbedder, MemoryStore, RetrievalHit

__all__ = [
    "EpisodeSummary",
    "EpisodeFact",
    "summarize_rule_based",
    "chain_summaries",
    "MemoryStore",
    "RetrievalHit",
    "HashingEmbedder",
]
