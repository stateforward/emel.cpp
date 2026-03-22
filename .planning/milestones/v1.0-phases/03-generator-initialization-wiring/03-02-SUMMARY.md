---
phase: 03-generator-initialization-wiring
plan: 02
subsystem: paritychecker
tags: [generation, initialize, diagnostics, tests]
requires: [03-01]
provides:
  - Deterministic initialize success output at the paritychecker boundary
  - Deterministic initialize failure output for rejected generator init requests
  - Focused generator lifecycle coverage for invalid initialize requests
affects: [paritychecker, generator-tests]
tech-stack:
  added: []
  patterns: [Explicit initialize callback-to-cli outcome mapping]
key-files:
  created: []
  modified:
    - tools/paritychecker/parity_runner.cpp
    - tests/generator/lifecycle_tests.cpp
key-decisions:
  - "Mapped success strictly to `initialize_done` and failure to rejected dispatch or `initialize_error`, rather than inferring readiness from the earlier load callback."
  - "Kept init failure coverage at the generator lifecycle level instead of introducing test-only CLI knobs."
patterns-established:
  - "Pattern: `initialize_done` drives user-visible success evidence and invalid init requests drive deterministic stderr plus exit code 1."
requirements-completed: [INIT-01, INIT-02]
duration: 26min
completed: 2026-03-08
---

# Phase 3 Plan 02 Summary

**Initialize success and failure signals are now explicit at the paritychecker boundary**

## Accomplishments
- Refined [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) so generation mode now reports `generation initialize ok ...` only after the generator initialize contract succeeds, and reports `generation initialize failed ...` for init-time failures.
- Added [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp) coverage for invalid initialize requests so the paritychecker failure mapping is backed by a direct generator lifecycle assertion.
- Verified that the success path now identifies generator initialization explicitly and that the broader repo gates remain green with `90.3%` line coverage and `56.3%` branch coverage.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Verification
- `build/zig/emel_tests_bin --dt-test-case="*generator*"`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello | rg "initialize"`
- `scripts/test_with_coverage.sh`
- `scripts/quality_gates.sh`

## Next Readiness
- Phase 4 can assume a deterministic initialize-ready contract and focus on bounded generation execution rather than generator bootstrap plumbing.
