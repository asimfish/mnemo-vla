# MNEMO-VLA Architecture

## 1. Three-tier memory

MNEMO separates memory by **time scale** and **modality**:

| Tier | Horizon | Modality | Where it lives today |
|---|---|---|---|
| **Hot** | seconds | vision (+ tactile planned) | *external encoder* (e.g. MEM / Mem-0 sliding window) |
| **Cold** | minutes | natural language | `mnemo.language_memory` |
| **Persistent** | days+ | text + embeddings | `mnemo.persistent_memory` |

The three tiers interact at a single touchpoint: the **planner prompt**.
The planner receives:

```
<global_task>            (str)
<hot>                    (video tokens, handled outside MNEMO)
<cold: structured memory>
<cross-session memory>   (from persistent store)
```

and produces the next subtask + an updated cold memory.

## 2. Semi-structured cold memory

Cold memory is not a dense embedding — it is a **semi-structured string**
that is easy to read, diff, and feed back into an LLM. The schema:

```
[TASK] <goal>
[SESSION] <id>
[SUBTASKS]
  1. [X] <done subtask>
  2. [>] <current subtask>
  3. [ ] <pending subtask>
[CROSS-SESSION]
  - <fact> (source: <session>, conf=<0..1>)
[FAILURES]
  - <error> -> <intervention>
```

The string round-trips losslessly via `Memory.to_text` and
`parse_memory_text`. This makes memory **debuggable**, unlike
vector-only alternatives.

## 3. Persistent memory mechanics

A persistent store keeps episode summaries across sessions. Each summary
contains:

```
EpisodeSummary
  session_id
  task
  outcome: success | failure | partial
  narrative: free-text
  facts: [(kind, content, entities)]
  timestamp
```

Facts are typed: `object_location`, `task_outcome`, `failure_cause`,
`user_preference`. Typing enables downstream filtering.

Retrieval flow:

```
new task goal  ─►  embed                   ──┐
                                             │
                                             ▼
                 [numpy / FAISS dot product]   top-K
                                             │
                                             ▼
                  EpisodeSummary[] ────► prompt block
```

The store ships two persistence files:

```
store/
  summaries.jsonl    # one JSON per episode
  vectors.npy        # (N, dim) float32 matrix
```

Garbage collection (size-bounded, staleness-aware) is on the roadmap.

## 4. Integrations

Today we target a single existing system: **Mem-0 (RMBench)**. The
integration is a two-line patch inside the planner plus a one-line hook
at episode end (see `mnemo/integrations/mem0_README.md`).

Planned integrations:

- `mnemo.integrations.memoryvla` — PCMB adapter (needs code clone).
- `mnemo.integrations.memer` — keyframe voting + persistent layer.
- `mnemo.integrations.cronusvla` — feature chunking + anchor.

## 5. Non-goals (for v0.x)

- No training loops. MNEMO is inference-time glue.
- No hardware drivers.
- No model-specific fine-tuning code.

These are tracked as milestones, not deferred indefinitely — the
training hooks will ship in v0.3 as documented in `roadmap.md`.
