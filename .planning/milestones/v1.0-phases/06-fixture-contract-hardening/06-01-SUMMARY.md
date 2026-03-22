---
phase: 06-fixture-contract-hardening
plan: 01
subsystem: paritychecker
tags: [generation, paritychecker, fixture-contract, regression]
requires: []
provides:
  - Canonical fixture-path enforcement for the first Llama-68M generation slice
  - A subprocess regression that rejects same-basename impostor fixtures outside `tests/models/`
  - Repo-root-aware paritychecker fixture resolution for subprocess execution
affects: [paritychecker-runtime, paritychecker-tests]
tech-stack:
  added: []
  patterns: [Canonical-path fixture gating at the CLI boundary]
key-files:
  created: []
  modified:
    - tools/paritychecker/CMakeLists.txt
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
key-decisions:
  - "Closed `HARN-02` by comparing normalized canonical fixture paths instead of only checking the pinned basename."
  - "Passed the repo root into the paritychecker executable at compile time so subprocess tests and direct runs resolve the same canonical fixture path."
patterns-established:
  - "Pattern: fixture contracts that matter to milestone acceptance should be enforced with normalized path equality plus a subprocess regression that proves the negative case."
requirements-completed: [HARN-02]
duration: 16min
completed: 2026-03-08
---

# Phase 6 Plan 01 Summary

**Paritychecker now accepts only the canonical first-slice Llama-68M fixture**

## Accomplishments
- Replaced basename-only generation fixture matching in [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) with normalized canonical-path comparison against `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
- Added the repo-root compile definition to [CMakeLists.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/CMakeLists.txt) so the standalone `paritychecker` executable resolves the same canonical fixture path as the test binary.
- Added a new subprocess regression in [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) that copies the same GGUF basename into a temp directory and proves generation mode rejects that impostor path with `generation requires canonical fixture`.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Deviations from Plan
- The original plan only implied paritychecker-local runtime work, but the executable also needed the repo-root compile definition so the canonical-path contract remained stable when invoked from the build directory.

## Verification
- `cmake --build build/paritychecker_zig --target paritychecker_tests paritychecker`
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/does-not-exist.gguf --text hello --max-tokens 1`

## Next Readiness
- Wave 2 could treat the runtime fixture contract as closed and focus on making the CLI/help wording and standard gate evidence match the new canonical rule.
