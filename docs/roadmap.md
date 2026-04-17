# MNEMO-VLA Roadmap

## v0.1 (this release) — foundational primitives

- [x] `mnemo.language_memory` with text round-trip
- [x] `mnemo.persistent_memory` with FAISS-ready but numpy-backed store
- [x] `mnemo.integrations.mem0` adapter + step-by-step guide
- [x] pytest suite (no GPU, no network)
- [x] Project docs (README, architecture, this roadmap, papers index)

## v0.2 — real embeddings

- [ ] `SentenceTransformerEmbedder`: sentence-transformers backend
- [ ] `QwenVLEmbedder`: reuse Qwen3-VL text encoder for tight alignment
  with Mem-0 / MemER / ReMem-VLA
- [ ] Optional FAISS-CPU index for N > 50k summaries
- [ ] Benchmark: retrieval P@1 on synthetic data

## v0.3 — tactile stream

- [ ] `mnemo.hot_memory.tactile`: 1D CNN encoder for 6-axis F/T
- [ ] Data collection helper for TAMEn-style handheld data
- [ ] Synthetic tactile augmentation
- [ ] Integration hook for Mem-0 `SubtaskEndClassifier` (boost Press Button)

## v0.4 — training hooks

- [ ] `mnemo.training.losses`: consistency + POP + keyframe losses
- [ ] Demo fine-tune script on RMBench Press Button
- [ ] Repro target: match Mem-0 baseline without regression

## v0.5 — MNEMO-Bench

- [ ] Port RMBench tasks into MNEMO harness
- [ ] Port MemoryBench and MIKASA-Robo task configs
- [ ] Unified results dashboard (per-memory-type radar chart)

## v0.6 — cross-embodiment

- [ ] Action head adapters for Piper / Franka / UR5 / ARX-X5 / Aloha
- [ ] Embodiment config generator
- [ ] Cross-embodiment evaluation protocol

## v1.0 — full framework

- [ ] Three-tier memory wired end-to-end on a single agent
- [ ] Real-world MNEMO-Real suite (10 tasks)
- [ ] Paper draft + reproducibility bundle

## Beyond v1.0

- Continual learning hooks (update persistent store online).
- Privacy / safety: redaction tools for memory contents.
- Multi-robot shared memory (team memory).

Each milestone is designed to ship independently; there is no hard
dependency across rows in the table. Community contributions targeting
any single row are welcome.
