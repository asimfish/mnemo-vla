# Annotated Bibliography

All citations below were analyzed in depth as part of the literature
review that seeded this project. Local analysis files live in the
companion directory `rm/related_work/` (see the repo root of the parent
research workspace).

## Memory methods

- **Mem-0 / RMBench**. *RMBench: Memory-Dependent Robotic Manipulation
  Benchmark with Insights into Policy Design*. arXiv:2603.01229v2. The
  direct code baseline we integrate with. Three-level buffer
  (anchor / sliding / key) + hierarchical planner. See
  `RMBench_Mem0_深度分析.md` (574 lines, code-level).

- **MemoryVLA**. *Perceptual-Cognitive Memory in VLAs*. arXiv:2508.19236.
  ICLR 2026. Perceptual + cognitive dual bank with consolidation.

- **ReMem-VLA**. *Dual-Level Recurrent Queries*. arXiv:2603.12942.
  Gradient-free EMA updates on frame + chunk level queries. No public
  code.

- **SAM2Act+**. *Integrating SAM2 Memory with Manipulation*.
  arXiv:2501.18564. Keyframe-level spatial memory built on RVT-2.

- **MemER**. *Scaling Up Memory for Robot Control via Experience
  Retrieval*. arXiv:2510.20328. Qwen3-VL learns to vote keyframes;
  `EpisodicMemory` class is only 90 lines.

- **CronusVLA**. *Multi-Frame VLA Modeling*. arXiv:2506.19816. Feature
  chunking as implicit short-term memory.

- **MEM**. *Multi-Scale Embodied Memory*. arXiv:2603.03596. The Physical
  Intelligence component powering π0.6-MEM / π0.7. Video encoder +
  natural-language summary. No public code.

## Benchmarks

- **RMBench** (see above): 9 bimanual tasks, M(1) + M(n) taxonomy.
- **MemoryBench**. Subset of SAM2Act paper, 3 spatial tasks.
- **MIKASA-Robo**. arXiv:2502.10550. ICLR 2026. 32 tasks × 12 memory
  types, RL-friendly.
- **RoboCerebra**. arXiv:2506.06677. NeurIPS 2025. Long-horizon
  (~2972 frames) System 2 benchmark.

## Infrastructure

- **RoboTwin 2.0**. arXiv:2506.18088. Bimanual manipulation data
  generator with MLLM-assisted task code synthesis. Underlies RMBench
  and, indirectly, Mem-0 training data.

## VLA backbones

- **π0.5 / π*0.6 / π0.7**. Physical Intelligence series. π0.7 (pi07.pdf)
  builds on MEM for production-grade long-horizon memory.

## Cross-links to detailed analyses

| Paper | Deep analysis (this workspace) |
|---|---|
| Mem-0 | `RMBench_Mem0_深度分析.md` |
| RoboTwin 2.0 | `RoboTwin2_深度分析.md` |
| ReMem-VLA | `ReMem-VLA_深度分析.md` |
| 6 memory methods | `记忆策略横向对比.md` |
| 4 benchmarks | `记忆基准横向对比.md` |
| Memory survey | `_综述_Memory_Aware_Manipulation_2025-2026.md` |
| Code-level cheat-sheet | `记忆策略代码速查.md` |
