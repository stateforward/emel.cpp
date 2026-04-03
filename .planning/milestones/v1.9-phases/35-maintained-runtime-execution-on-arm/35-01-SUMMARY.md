---
phase: 35-maintained-runtime-execution-on-arm
plan: 01
subsystem: generator
tags: [liquid, runtime, arm, quantized]
requires:
  - phase: 34
    provides: explicit lfm2 model-contract acceptance
provides:
  - maintained Liquid generator initialization and bounded generation on ARM
  - truthful quantized runtime contract publication for the maintained Q4_K_M slice
affects: [36, 37, 38]
tech-stack:
  added: []
  patterns:
    - maintained runtime contract published on the shipped generation path
key-files:
  created:
    - .planning/phases/35-maintained-runtime-execution-on-arm/35-01-SUMMARY.md
  modified:
    - src/emel/generator/detail.hpp
    - src/emel/generator/sm.hpp
    - tests/generator/lifecycle_tests.cpp
    - tools/paritychecker/parity_runner.cpp
    - tools/bench/generation_bench.cpp
key-decisions:
  - "Keep the maintained runtime claim fixed to the official Q4_K_M Liquid fixture."
  - "Publish truthful quantized-path evidence rather than claiming broader Liquid quant support."
patterns-established:
  - "Runtime contract truth is published on maintained generation, parity, and benchmark surfaces together."
requirements-completed: [RUN-04, RUN-06]
duration: reconstructed
completed: 2026-04-02
---

# Phase 35 Plan 01: Maintained Runtime Execution On ARM Summary

The maintained Liquid runtime slice shipped on the branch, but the current v1.9 phase directory
never recorded the closeout. This summary reconstructs the delivered runtime evidence.

## Accomplishments

- Brought the maintained Liquid `Q4_K_M` slice up on the shipped generator path for ARM.
- Preserved bounded maintained generation on the official fixture.
- Published a truthful quantized runtime contract for the maintained Liquid slice on maintained
  parity and benchmark surfaces.

## Evidence

- [detail.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/generator/detail.hpp)
- [sm.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/generator/sm.hpp)
- [lifecycle_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tests/generator/lifecycle_tests.cpp)
- [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/parity_runner.cpp)
- [generation_bench.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/generation_bench.cpp)

---
*Phase: 35-maintained-runtime-execution-on-arm*
*Completed: 2026-04-02*
