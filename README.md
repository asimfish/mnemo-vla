# MNEMO-VLA

**Unified memory framework for vision-language-action models.**

MNEMO-VLA is an open-source, incremental research project aimed at giving
robot policies **hot / cold / persistent** memory — short-term video,
long-term language, and cross-session retrieval — in a single, composable
Python package.

> **Status**: v0.1 (alpha). The skeleton ships two reusable components
> (semi-structured language memory + persistent cross-session store) plus
> a ready-to-apply patch for the [Mem-0 / RMBench planner](
> https://github.com/robotwin-Platform/rmbench).
>
> The full architecture (tactile stream, joint memory-action training,
> MNEMO-Bench) is described in the research proposal under
> `docs/proposal.md`.

---

## Why MNEMO?

Memory-aware manipulation is a rapidly moving field (SAM2Act+, MemoryVLA,
ReMem-VLA, MemER, CronusVLA, MEM, Mem-0, ...). After surveying the state
of the art we identified seven recurring gaps:

| Gap | MNEMO response |
|---|---|
| Single-task training | Shared memory backbone + task adapter |
| Vision-only sensing | Tri-modal fusion (vision + text + tactile) |
| Episode-bound forgetting | FAISS-style persistent memory |
| Memory / action trained in isolation | Joint consistency loss |
| Apples-to-oranges benchmarks | Unified MNEMO-Bench |
| Black-box memory | Semi-structured, human-readable format |
| Short ⇄ long horizon mismatch | Hot / cold / persistent three-tier store |

This repo delivers the **first two** rows of that table today; the rest
are tracked in `docs/roadmap.md`.

---

## Install

```bash
git clone https://github.com/asimplefish/mnemo-vla.git
cd mnemo-vla
pip install -e .
```

Optional extras:

```bash
pip install -e .[faiss,embeddings,tests]
```

Minimum requirements: Python 3.10, numpy. FAISS and sentence-transformers
are optional and lazily imported.

---

## Quickstart

```python
from mnemo.language_memory import Memory, SubtaskStatus
from mnemo.persistent_memory import (
    HashingEmbedder, MemoryStore, summarize_rule_based,
)

# --- Episode 1: organize kitchen tools ---
memory = Memory(task="Organize kitchen tools.")
memory.add_subtask("Open drawer", SubtaskStatus.DONE)
memory.add_subtask("Place hammer in left slot", SubtaskStatus.DONE)

store = MemoryStore(root="./my_memory", embedder=HashingEmbedder())
store.add(summarize_rule_based(memory))

# --- Episode 2 (days later): find the hammer ---
hits = store.query("Find the hammer.", top_k=3)
for hit in hits:
    print(hit.score, hit.summary.narrative)
```

The full walk-through (including prompt-injection for Mem-0) lives in
`examples/quickstart.py`.

---

## Using MNEMO inside Mem-0

`mnemo.integrations.mem0_planner_patch` provides a drop-in adapter. Full
step-by-step guide: [`mnemo/integrations/mem0_README.md`](mnemo/integrations/mem0_README.md).

Two-line summary:

```python
from mnemo.integrations.mem0_planner_patch import inject_into_prompt
full_prompt_msg = inject_into_prompt(
    full_prompt_msg,
    store=self.persistent_store,
    task_goal=self.global_task,
)
```

No retraining required for the hook — Qwen3-VL consumes the injected
block as extra context.

---

## Package layout

```
mnemo/
├── language_memory/          # Semi-structured, human-readable memory
│   ├── schema.py             #   Memory / Subtask / CrossSessionFact / FailureEvent
│   └── formatter.py          #   Text <-> Memory round-trip parser
├── persistent_memory/        # Cross-session memory store
│   ├── episode_summary.py    #   EpisodeSummary + rule-based summarizer
│   └── memory_store.py       #   FAISS-ready store (numpy-backed for now)
├── integrations/             # Adapters for existing VLAs
│   ├── mem0_planner_patch.py #   Drop-in for robotwin-Platform/rmbench
│   └── mem0_README.md        #   Step-by-step integration guide
└── utils/                    # Helpers (currently empty)

tests/                        # pytest suite (pure Python, no GPU needed)
examples/                     # Runnable demos (quickstart.py)
docs/                         # Architecture, roadmap, paper index
```

---

## Tests

```bash
pip install -e .[tests]
pytest -q
```

The entire test suite runs on CPU in a few seconds and covers:

* Memory serialization + text round-trip
* FAISS-free persistent store persistence & retrieval
* Rule-based episode summarization
* Mem-0 prompt-injection helpers

---

## Design principles

1. **Readable memory by default.** `Memory.to_text()` is the canonical
   representation — we do not invent a yet-another embedding format for
   long-term memory.
2. **Zero required heavy deps.** numpy only. FAISS and transformers are
   opt-in extras.
3. **Pluggable encoders.** `Embedder` is a `typing.Protocol`; swap in any
   model you like (the default `HashingEmbedder` is for tests).
4. **Compose with, don't replace, existing systems.** Integrations live
   in `mnemo.integrations/*` as small patches so that Mem-0, MemoryVLA,
   ReMem-VLA, and others can adopt MNEMO piece by piece.

---

## Documentation

* [`docs/architecture.md`](docs/architecture.md) — three-tier memory design
* [`docs/roadmap.md`](docs/roadmap.md) — milestone plan
* [`docs/papers.md`](docs/papers.md) — annotated bibliography (6 methods + 4 benchmarks)
* [`docs/proposal.md`](docs/proposal.md) — the long-form research proposal

---

## Citing

This codebase is a research scaffolding. The accompanying proposal is
hosted in the literature review that seeded the project:
**"MNEMO-VLA: Unified Memory Framework for VLAs"** (draft, 2026).

```bibtex
@misc{mnemo-vla-2026,
  title  = {MNEMO-VLA: Unified Memory Framework for Vision-Language-Action Models},
  author = {MNEMO-VLA authors},
  year   = {2026},
  note   = {Draft. \url{https://github.com/asimplefish/mnemo-vla}},
}
```

---

## License

MIT. See [`LICENSE`](LICENSE).
