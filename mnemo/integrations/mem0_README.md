# Mem-0 Integration Guide

This document describes how to drop MNEMO's persistent memory layer into
the Mem-0 codebase (`robotwin-Platform/rmbench`). The goal is to keep the
Mem-0 changes **minimal** -- just a few edits in the Planner.

## Target files in Mem-0

```
rmbench/policy/Mem-0/source/
├── models/planning_module/
│   └── memorymatters_planner.py          # ← patch here
└── agent/
    └── memorymatters_agent.py            # ← persistence hook here
```

## Step 1. Add MNEMO as a dependency

From the Mem-0 conda env:

```bash
pip install -e /path/to/mnemo-vla
```

Or add to `requirements.txt`:

```
mnemo @ file:///path/to/mnemo-vla
```

## Step 2. Patch `MemoryMattersPlanner.__init__`

Create the persistent memory store when the planner is built.

```python
# memorymatters_planner.py
from mnemo.persistent_memory import MemoryStore, HashingEmbedder

class MemoryMattersPlanner(nn.Module):
    def __init__(self, config, device=None, vllm_url=..., global_task=None, **kwargs):
        super().__init__()
        # ... existing init ...
        store_root = config.planning_module.get(
            "persistent_memory_root",
            "./persistent_memory_store",
        )
        self.persistent_store = MemoryStore(
            root=store_root,
            embedder=HashingEmbedder(dim=64),  # swap in SentenceTransformer for real use
        )
```

## Step 3. Inject cross-session memory into the prompt

In `prepare_qwen_input`, right before returning, call the MNEMO adapter:

```python
# memorymatters_planner.py
from mnemo.integrations.mem0_planner_patch import inject_into_prompt

def prepare_qwen_input(self):
    # ... existing logic that builds full_prompt_msg ...
    full_prompt_msg = inject_into_prompt(
        full_prompt_msg,
        store=self.persistent_store,
        task_goal=self.global_task,
        top_k=3,
    )
    return full_prompt_msg
```

The Qwen3-VL prompt now contains a `<cross_session_memory>` block before
`<finished_subtasks>`. No training changes are required for this hook;
the LLM will treat the block as extra context.

## Step 4. Persist memory at episode end

In `MemoryMattersAgent.reset()` (which is called at the end of an
episode), add an episode-summary write.

```python
# memorymatters_agent.py
from mnemo.language_memory import Memory, SubtaskStatus
from mnemo.persistent_memory import summarize_rule_based

def reset(self):
    if self.high_model is not None:
        # Build a Memory object from the planner's finished subtasks.
        memory = Memory(task=self.config.get("global_task", ""))
        for subtask in self.high_model.finished_subtasks:
            memory.add_subtask(subtask, status=SubtaskStatus.DONE)
        summary = summarize_rule_based(memory)
        self.high_model.persistent_store.add(summary)
    # ... existing reset logic ...
```

Replace `summarize_rule_based` with an LLM-based summarizer when ready.

## Step 5. Smoke-test the integration

1. Launch Mem-0 evaluation as usual (`bash eval.sh`).
2. Confirm that the first episode runs unchanged.
3. After the first episode, check `./persistent_memory_store/summaries.jsonl`
   for a new line.
4. On the second episode, check the Qwen3-VL prompt (via the planner log)
   for the `<cross_session_memory>` block.

## Known caveats

- **Embedding quality**: `HashingEmbedder` is a toy. Swap in a
  real sentence encoder (SentenceTransformer, Qwen3-VL text encoder, ...)
  before running user studies.
- **Concurrent writes**: `MemoryStore` appends to JSONL + rewrites the
  vectors file. Serialize episodes if you run Mem-0 in parallel.
- **Namespace collisions**: if you run multiple robots, scope the
  `store_root` by robot id to keep memories separate.

## Verifying the patch without running the robot

The MNEMO package ships a pure-Python test suite that exercises these
helpers:

```bash
cd mnemo-vla
python -m pytest tests/ -q
```

The tests cover schema serialization, text round-trip parsing, FAISS-free
memory store persistence, and the Mem-0 prompt injection helper.
