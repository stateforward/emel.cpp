---
phase: 05-gate-hardening
plan: 01
subsystem: paritychecker
tags: [generation, paritychecker, regression, failure-path]
requires: []
provides:
  - One deterministic generation failure-path subprocess regression
  - A generalized paritychecker subprocess capture helper reusable across success and failure cases
  - Stable stderr assertions for the missing-model generation contract
affects: [paritychecker-tests]
tech-stack:
  added: []
  patterns: [Shared subprocess capture helper for paritychecker CLI regressions]
key-files:
  created: []
  modified:
    - tools/paritychecker/paritychecker_tests.cpp
key-decisions:
  - "Used the existing missing-model generation error as the narrowest deterministic negative case instead of adding a synthetic mismatch path."
  - "Kept the entire hardening slice inside `tools/paritychecker/`; no `src/` or runtime diagnostic changes were needed."
patterns-established:
  - "Pattern: generation subprocess tests now share one argument-driven capture helper so future negative cases do not need bespoke command construction."
requirements-completed: [VER-02]
duration: 8min
completed: 2026-03-08
---

# Phase 5 Plan 01 Summary

**Paritychecker now has automated generation failure-path subprocess coverage**

## Accomplishments
- Generalized the subprocess capture helper in [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) so generation CLI tests can pass arbitrary argument sets while preserving the existing cross-platform stdout/stderr capture pattern.
- Added a deterministic negative doctest for the missing-model generation path and asserted the stable `generation load failed: missing model file` stderr contract alongside the non-zero exit code.
- Preserved the existing Phase 4 success-path generation subprocess regression unchanged, so both positive and negative generation checks now live in the same `paritychecker_tests` surface.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Deviations from Plan
- The plan allowed for a tiny paritychecker runtime diagnostic adjustment if the stderr contract proved unstable. That was not necessary; the existing missing-model message was already specific enough for deterministic substring assertions.

## Verification
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- `scripts/paritychecker.sh`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/does-not-exist.gguf --text hello --max-tokens 1`

## Next Readiness
- Wave 2 could treat `VER-02` as closed at the test surface and focus only on updating stale Phase 5 wording plus confirming the default gate chain already carries the final regression mix.
