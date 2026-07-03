---
phase: 231-deterministic-error-taxonomy
plan: "01"
subsystem: io-staged-read-errors
tags: [stateforward-sml, staged-read, esg-01, esg-02a, esg-03, esg-04]
requirements-completed: [ESG-01, ESG-02A, ESG-03, ESG-04]
requirements-deferred: [ESG-02B]
completed: 2026-05-07
---

# Phase 231 Plan 01: deterministic error taxonomy — Summary

Phase 231 implemented deterministic named error categories for current source-backed
`io/staged_read` behavior and explicitly deferred file-backed `ESG-02B`.

## What changed

- Added source-backed error categories in `src/emel/io/staged_read/errors.hpp`:
  `invalid_callbacks`, `invalid_stage_contract`, `invalid_target_window`,
  `unsupported_platform`, `null_source_span`, `source_span_size_mismatch`,
  `insufficient_source_span`.
- Added explicit source-span guard predicates in `src/emel/io/staged_read/guards.hpp`
  for null/mismatch/insufficient source surfaces.
- Added corresponding error-marking actions in `src/emel/io/staged_read/actions.hpp`.
- Extended `src/emel/io/staged_read/sm.hpp` with explicit decision/error states and
  transitions for source-surface categories; no runtime choice moved into actions/detail.
- Updated `tests/io/staged_read/lifecycle_tests.cpp` to assert new category outcomes
  through `process_event(...)` callbacks only.
- Verified supported-host doctests cover `invalid_callbacks`, `invalid_stage_contract`,
  `invalid_target_window`, plus source-span taxonomy; `unsupported_platform` remains
  explicitly modeled in compiled guard/transition paths and is not force-triggered on
  current supported-host test runs.
- Updated planning truth (`ROADMAP`, `REQUIREMENTS`, `STATE`) with approved split:
  `ESG-02A` satisfied in Phase 231, `ESG-02B` deferred/future.
- Kept generated architecture docs in sync with the updated source-span taxonomy
  graph (`io_staged_read` state diagram includes null/mismatch/insufficient-source branches).

## Changed files

- `src/emel/io/staged_read/errors.hpp`
- `src/emel/io/staged_read/guards.hpp`
- `src/emel/io/staged_read/actions.hpp`
- `src/emel/io/staged_read/sm.hpp`
- `tests/io/staged_read/lifecycle_tests.cpp`
- `.planning/phases/231-deterministic-error-taxonomy/231-CONTEXT.md`
- `.planning/phases/231-deterministic-error-taxonomy/231-01-PLAN.md`
- `.planning/phases/231-deterministic-error-taxonomy/231-01-SUMMARY.md`
- `.planning/phases/231-deterministic-error-taxonomy/231-VERIFICATION.md`
- `.planning/architecture/io_staged_read.md`
- `.planning/architecture/mermaid/io_staged_read.mmd`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`

## Scope truths

- No OS file I/O path was added to staged_read.
- No synthetic fault knobs or test-only production fields were added.
- No snapshot baseline updates were landed; timing snapshot was restored to HEAD after gate runs.
