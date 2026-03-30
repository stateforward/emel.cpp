---
phase: 28-qwen3-parity-and-regression
plan: 01
completed: 2026-03-28
commit: c19e824
---

# Phase 28 Plan 01 Summary

Stored-baseline parity is now aligned for the canonical Qwen3 generation slice. Compare-mode
reference tokenization, attribution replay, and output rendering now consume the same explicit
formatter/runtime contract as the shipped generator, and the approved `1/10/100/1000` parity
artifacts were refreshed under `snapshots/parity/`.
