---
phase: 27-qwen3-runtime-architecture-bring-up
plan: 02
subsystem: generator
tags: [qwen3, generator, paritychecker, planner]
requires:
  - phase: 26.1-native-q8-0-projection-and-output-runtime-support-for-canonical-qwen3
    provides: native q8_0 hot-path support and backend-unblock proof
  - phase: 27-qwen3-runtime-architecture-bring-up
    provides: canonical qwen3 execution-view binding
provides:
  - shipped generator runtime bring-up for the canonical Qwen3 slice
  - maintained paritychecker generation proof on the temp-baseline write lane
  - explicit planner metadata and error classification for structured-message requests
affects: [28, 29]
tech-stack:
  added: []
  patterns:
    - maintained generation bring-up is proven before stored parity publication
    - explicit planner metadata contract for structured-message requests
key-files:
  created:
    - .planning/phases/27-qwen3-runtime-architecture-bring-up/27-02-SUMMARY.md
  modified:
    - src/emel/generator/actions.hpp
    - src/emel/generator/detail.hpp
    - src/emel/generator/guards.hpp
    - src/emel/generator/sm.hpp
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
key-decisions:
  - "The canonical Qwen runtime slice is complete when the shipped generator initializes and generates truthfully on the maintained path; stored snapshot-backed parity stays in Phase 28."
  - "Dedicated maintained-generation tests own the canonical Qwen fixture; the generic tiny-model tokenizer sweep excludes that fixture."
patterns-established:
  - "Planner invalid bitmasks are classified explicitly and must not be downgraded into backend errors."
requirements-completed: [RUN-01]
completed: 2026-03-28
commit: fed9950
---

# Phase 27 Plan 02: Canonical Qwen Runtime Bring-Up Summary

**The shipped EMEL generator path now initializes and generates on the canonical Qwen3 slice, and
the maintained paritychecker generation surface succeeds on the temp-baseline write lane without
claiming stored parity completion yet.**

## Accomplishments
- Carried the explicit Qwen execution-view and native `q8_0` support through the shipped generator
  runtime path.
- Fixed the planning/request metadata boundary that was misreporting canonical Qwen failures as
  backend initialization errors.
- Updated maintained Qwen subprocess coverage to prove successful generation, formatter-contract
  publication, flash attribution, and native `q8_0` dispatch on the canonical fixture.

## Verification
- `./build/zig/emel_tests_bin --test-case='*qwen3*generator*' --no-breaks`
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Qwen3-0.6B-Q8_0.gguf --text hello --max-tokens 1 --write-generation-baseline <tmp>`
- `scripts/quality_gates.sh`

## Issues Encountered
- The remaining bring-up failure was not kernel compute after `26.1`; it was a planner boundary
  bug where single-sequence prompt planning emitted invalid sequence metadata and then misclassified
  the resulting planner bits as backend failure.

## Next Phase Readiness
- Phase 28 can focus on stored parity/regression work. Runtime bring-up is no longer the blocker.
