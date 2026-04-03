---
phase: 35-maintained-runtime-execution-on-arm
plan: 01
subsystem: reference-comparator
tags: [embedded-size, qwen3, llama.cpp, smoke, comparator]
requires:
  - phase: 34-initializer-surface-shrink-and-proof
    provides: corrected EMEL probe and shared executable-size harness
provides:
  - matched llama.cpp reference executable row
  - shared runtime smoke proof for published rows
  - narrowed comparator evidence for v1.8
affects: [36-01 publication plumbing, 39-01 publication refresh]
tech-stack:
  added: []
  patterns: [matched executable comparison, shared smoke proof, reference row publication]
key-files:
  created: []
  modified:
    [scripts/embedded_size.sh, tools/embedded_size/reference_probe/main.cpp, tools/embedded_size/emel_probe/main.cpp]
key-decisions:
  - "The published comparison stays limited to EMEL and one matched llama.cpp executable."
  - "Runtime smoke is part of publication truth, not a separate optional lane."
requirements-completed: []
duration: 0min
completed: 2026-04-02
---

# Phase 35 Plan 01 Summary

**The matched `llama.cpp` reference row and shared smoke proof are now explicit**

## Accomplishments

- Verified the maintained reference row in
  [main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/embedded_size/reference_probe/main.cpp)
  builds as a final linked executable on the same canonical Qwen3 slice.
- Confirmed the local executable-size run smoke-passes the same `hello` -> first-token path for
  both EMEL and reference rows.
- Kept the published comparator set narrow to EMEL and one matched `llama.cpp` executable row.

## Verification

- `./scripts/embedded_size.sh --json`

## Deviations from Plan

- None in scope. The phase stayed on one comparator row and shared smoke proof only.

---
*Phase: 35-maintained-runtime-execution-on-arm*
*Completed: 2026-04-02*
