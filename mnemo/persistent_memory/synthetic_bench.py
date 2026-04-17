"""Synthetic cross-session retrieval benchmark for the persistent store.

This module generates a deterministic synthetic dataset of episode
summaries across several task families and evaluates Recall@K when
queried with paraphrased task descriptions. The benchmark is CPU-only
and finishes in well under 10 seconds, which makes it suitable for
continuous integration and for reporting in paper tables.

Public API:
    build_dataset(seed=42)
    run_benchmark(store, queries) -> BenchmarkResult
    default_query_set() -> list of (query_text, target_session_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .episode_summary import EpisodeFact, EpisodeSummary
from .memory_store import MemoryStore


# ---- dataset construction -------------------------------------------------

_TASK_FAMILIES = {
    "kitchen": [
        "Prepare grilled cheese sandwich",
        "Clean the countertop",
        "Unload the dishwasher",
        "Bake chocolate chip cookies",
    ],
    "workshop": [
        "Organize power tools on the pegboard",
        "Tighten bolts on the workbench",
        "Sort screws by size",
        "Sharpen the knives",
    ],
    "laundry": [
        "Fold the dry towels",
        "Sort dirty clothes by color",
        "Load the washing machine",
        "Iron the dress shirts",
    ],
    "garden": [
        "Water all indoor plants",
        "Prune the rose bush",
        "Repot the succulents",
        "Harvest ripe tomatoes",
    ],
    "office": [
        "File the monthly invoices",
        "Shred expired documents",
        "Restock the printer paper",
        "Wipe down the whiteboard",
    ],
}


def build_dataset() -> List[EpisodeSummary]:
    """Return a deterministic list of EpisodeSummary objects."""
    summaries: List[EpisodeSummary] = []
    for family, tasks in _TASK_FAMILIES.items():
        for idx, task in enumerate(tasks):
            session_id = f"{family}-{idx:02d}"
            summaries.append(
                EpisodeSummary(
                    session_id=session_id,
                    task=task,
                    outcome="success",
                    narrative=f"Completed: {task}.",
                    facts=[
                        EpisodeFact(
                            kind="task_outcome",
                            content=f"{task} finished successfully.",
                        ),
                        EpisodeFact(
                            kind="object_location",
                            content=f"{family.capitalize()} tools live in "
                                    f"the {family} cabinet.",
                        ),
                    ],
                )
            )
    return summaries


def default_query_set() -> List[Tuple[str, str]]:
    """Return a list of (query, expected_session_id) pairs.

    Queries are paraphrases of the task titles in ``build_dataset``.
    """
    return [
        ("Make a toasted cheese sandwich", "kitchen-00"),
        ("Wipe the kitchen counter", "kitchen-01"),
        ("Take dishes out of the dishwasher", "kitchen-02"),
        ("Bake cookies", "kitchen-03"),
        ("Arrange power tools neatly", "workshop-00"),
        ("Fasten bench bolts", "workshop-01"),
        ("Group screws by dimension", "workshop-02"),
        ("Sharpen kitchen knives", "workshop-03"),
        ("Fold towels that are dry", "laundry-00"),
        ("Sort the laundry by color", "laundry-01"),
        ("Run the washer", "laundry-02"),
        ("Iron my dress shirts", "laundry-03"),
        ("Water the indoor plants", "garden-00"),
        ("Prune the roses", "garden-01"),
        ("Repot succulent plants", "garden-02"),
        ("Pick ripe tomatoes", "garden-03"),
        ("File invoices for the month", "office-00"),
        ("Shred old documents", "office-01"),
        ("Refill printer paper", "office-02"),
        ("Clean the whiteboard", "office-03"),
    ]


# ---- evaluation -----------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Aggregated metrics from a synthetic retrieval benchmark run."""

    num_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    per_query: List[dict] = field(default_factory=list)

    def as_table(self) -> str:
        """Render a plain-text summary table."""
        lines = [
            f"Queries evaluated : {self.num_queries}",
            f"Recall@1          : {self.recall_at_1:.2%}",
            f"Recall@3          : {self.recall_at_3:.2%}",
            f"Recall@5          : {self.recall_at_5:.2%}",
        ]
        return "\n".join(lines)


def run_benchmark(
    store: MemoryStore,
    queries: List[Tuple[str, str]],
    top_k: int = 5,
) -> BenchmarkResult:
    """Evaluate Recall@K on the given (query, target) pairs.

    The store is expected to already contain the synthetic episode
    summaries (typically produced by ``build_dataset()``).
    """
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    per_query: List[dict] = []

    for query_text, target in queries:
        hits = store.query(query_text, top_k=top_k)
        top_ids = [h.summary.session_id for h in hits]
        rank = top_ids.index(target) + 1 if target in top_ids else None

        if rank is not None:
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1

        per_query.append(
            {
                "query": query_text,
                "target": target,
                "rank": rank,
                "top_ids": top_ids,
            }
        )

    n = len(queries)
    return BenchmarkResult(
        num_queries=n,
        recall_at_1=hits_at_1 / n if n else 0.0,
        recall_at_3=hits_at_3 / n if n else 0.0,
        recall_at_5=hits_at_5 / n if n else 0.0,
        per_query=per_query,
    )


__all__ = [
    "build_dataset",
    "default_query_set",
    "run_benchmark",
    "BenchmarkResult",
]
