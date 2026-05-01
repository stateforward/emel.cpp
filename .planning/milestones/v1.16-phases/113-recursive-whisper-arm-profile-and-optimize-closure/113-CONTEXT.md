---
phase: 113
title: Recursive Whisper ARM Profile And Optimize Closure
status: superseded
created: 2026-04-27
requirements: [CLOSE-01, PERF-03]
---

# Phase 113 Context

## Supersession Note

Phase 113 is superseded by the 2026-04-27 source-backed audit. The active work moved to Phase 114
for runtime-surface contract repair, Phase 115 for evidence truth repair, and Phase 116 for final
benchmark/closeout proof. The original performance direction below is retained as historical
context only and must not be executed as a standalone plan.

## Goal

Restore the v1.16 performance contract by recursively profiling and optimizing the maintained ARM
Whisper runtime until EMEL beats the matched single-thread CPU `whisper.cpp` reference lane, then
rerun source-backed closeout.

## Audit Trigger

The v1.16 milestone audit found that `build/whisper_benchmark/benchmark_summary.json` records
`status: ok` while EMEL is slower than the matched reference:

- EMEL mean: `241,398,167 ns`
- Reference mean: `59,409,666 ns`

This contradicts the archived `PERF-03` requirement and the active milestone performance contract.

## Historical Direction

- Preserve `PERF-03`; do not weaken the milestone performance contract.
- Make benchmark publication fail when EMEL is not faster than the matched single-thread reference.
- Profile recursively from benchmark summary to stage timing to source-owned kernel/runtime code.
- Optimize only EMEL-owned `src/` kernel/runtime paths.
- Preserve exact transcript parity, lane isolation, SML routing rules, and allocation constraints.

## Planning Notes

The first plan for this phase should identify the current dominant ARM bottleneck before choosing
implementation tasks. The phase is not complete until the maintained single-thread benchmark shows
EMEL faster than reference and `$gsd-audit-milestone` passes.
