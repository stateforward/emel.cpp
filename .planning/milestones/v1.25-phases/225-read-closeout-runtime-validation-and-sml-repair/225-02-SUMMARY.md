---
phase: 225-read-closeout-runtime-validation-and-sml-repair
plan: 02
subsystem: io
tags: [stateforward-sml, io-loader, io-read, batch-route, doctest]
requires:
  - phase: 225-read-closeout-runtime-validation-and-sml-repair
    provides: Plan 01 shared tensor span contract and public io/read batch copy dispatch.
provides:
  - Public `io/loader` `load_tensor_batch` request and `_done`/`_error` events.
  - Guard-selected `read_copy` batch route through one public `io/read` child dispatch.
  - Same-RTC stack status callbacks for concrete read batch result capture.
  - Public-dispatch doctests and source guardrail for one batch child dispatch.
affects: [io-loader, io-read, model-loader, phase-225]
tech-stack:
  added: []
  patterns:
    - Wrapper-local `batch_runtime_status` carrier referenced by internal runtime events.
    - One selected child dispatch for batch routing with result choice in guards/transitions.
key-files:
  created:
    - .planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-02-SUMMARY.md
  modified:
    - src/emel/io/loader/events.hpp
    - src/emel/io/loader/detail.hpp
    - src/emel/io/loader/guards.hpp
    - src/emel/io/loader/actions.hpp
    - src/emel/io/loader/sm.hpp
    - tests/io/loader/lifecycle_tests.cpp
key-decisions:
  - "Batch read/copy result status lives in wrapper-local stack storage, not io/loader context."
  - "Batch route actions use separate batch-specific action symbols so SML completion propagation resolves the originating runtime event."
patterns-established:
  - "io/loader batch path: guards select read_copy+actor, one action dispatches io/read, guards classify accepted/ok status, then callbacks publish done/error."
requirements-completed: [TIO-03, VAL-04, VAL-01]
duration: 14min
completed: 2026-05-06
---

# Phase 225 Plan 02: io/loader Batch Read/Copy Route Summary

**Public io/loader batch route that reports read_copy only after one successful io/read batch dispatch.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-05-06T14:39:16Z
- **Completed:** 2026-05-06T14:52:45Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added `load_tensor_batch`, `load_tensor_batch_done`, and `load_tensor_batch_error`.
- Replaced the loader-local tensor span definition with the shared `emel::io::event::tensor_load_span` alias.
- Added explicit batch guards/transitions for span validity, read-copy actor availability, and read batch success/failure.
- Added one `effect_dispatch_read_tensor_batch` action that calls `ctx.io_read->process_event(read)` once and records callbacks into stack status.
- Added public-dispatch doctests for batch success, missing actor, invalid spans, callback-absent legs, and read failure propagation.

## Task Commits

1. **Task 1: Add io/loader batch route doctests and guardrail** - `8c4b8fc2` (`test`)
2. **Task 2: Implement io/loader load_tensor_batch public route** - `56295e65` (`feat`)

## Files Created/Modified

- `src/emel/io/loader/events.hpp` - Shared tensor span alias and public batch request/result events.
- `src/emel/io/loader/detail.hpp` - Wrapper-local batch status and runtime carrier.
- `src/emel/io/loader/guards.hpp` - Batch validity, strategy+actor, read success/failure, and callback predicates.
- `src/emel/io/loader/actions.hpp` - Batch begin, one child dispatch, read callbacks, done/error publication, and record actions.
- `src/emel/io/loader/sm.hpp` - Explicit batch route states/transitions and public `process_event(load_tensor_batch)`.
- `tests/io/loader/lifecycle_tests.cpp` - Batch route doctests and source guardrail for one read batch dispatch.

## Decisions Made

- Kept batch result capture in `detail::batch_runtime_status` allocated by the public wrapper, so no dispatch-local request/status data is stored in `action::context`.
- Split batch invalid/unsupported actions from existing single-tensor actions after SML action deduction rejected overloaded functors on completion transitions.
- Kept snapshot files unchanged; the lint snapshot issue was resolved by clang-formatting the Plan 02 files.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Split overloaded action functors**
- **Found during:** Task 2
- **Issue:** Adding batch overloads to existing single-tensor action structs caused SML action argument deduction failures during the build.
- **Fix:** Added batch-specific action symbols for invalid and unsupported batch outcomes and updated transitions to use them.
- **Files modified:** `src/emel/io/loader/actions.hpp`, `src/emel/io/loader/sm.hpp`
- **Verification:** `cmake --build build/zig --target emel_tests_bin` passed.
- **Committed in:** `56295e65`

**2. [Rule 3 - Blocking] Added batch error/no-callback coverage for the scoped quality gate**
- **Found during:** Task 2
- **Issue:** Focused tests passed, but changed-file coverage was below the 90% line threshold.
- **Fix:** Added public-dispatch doctests for invalid batch spans, success without a done callback, read failure propagation, and missing actor without an error callback.
- **Files modified:** `tests/io/loader/lifecycle_tests.cpp`
- **Verification:** Scoped quality gate passed with 98.0% line and 71.4% branch coverage for changed io/loader headers.
- **Committed in:** `56295e65`

**3. [Rule 3 - Blocking] Formatted Plan 02 files instead of updating lint snapshot**
- **Found during:** Task 2 quality gate
- **Issue:** The lint snapshot lane flagged `tests/io/loader/lifecycle_tests.cpp` as a new clang-format regression.
- **Fix:** Ran clang-format on the Plan 02 source/test files; no snapshots were updated.
- **Files modified:** Plan 02 source/test files
- **Verification:** `scripts/quality_gates.sh` completed successfully after formatting.
- **Committed in:** `56295e65`

**Total deviations:** 3 auto-fixed blocking issues.
**Impact on plan:** All fixes were required to build or satisfy existing gates; no runtime scope was widened.

## Issues Encountered

- The RED build failed as expected on missing `load_tensor_batch` event/result types.
- The pre-existing worktree had many unrelated dirty files. Task commits staged only the Plan 02 source/test files.

## Validation

- `cmake --build build/zig --target emel_tests_bin` - failed during RED as expected; passed after implementation.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` - passed.
- `rg -n "struct load_tensor_batch|load_tensor_batch_done|load_tensor_batch_error" src/emel/io/loader/events.hpp` - found expected declarations.
- `rg -n "effect_dispatch_read_tensor_batch|on_read_batch_done|on_read_batch_error|process_event\\(read\\)" src/emel/io/loader/actions.hpp` - found expected dispatch and callback symbols.
- `rg -n "struct batch_runtime_status|accepted|failed_index" src/emel/io/loader/detail.hpp` - found expected status fields.
- `rg -n "strategy_read_copy_batch_with_actor|process_event\\(const event::load_tensor_batch" src/emel/io/loader/guards.hpp src/emel/io/loader/sm.hpp` - found expected guard and wrapper.
- `rg -n "for \\(.*process_event|while \\(.*process_event|process_queue|defer_queue" src/emel/io/loader` - no matches.
- `scripts/check_domain_boundaries.sh` - passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/io/loader/events.hpp:src/emel/io/loader/detail.hpp:src/emel/io/loader/guards.hpp:src/emel/io/loader/actions.hpp:src/emel/io/loader/sm.hpp:tests/io/loader/lifecycle_tests.cpp" EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh` - passed.

## Known Stubs

None. Default null pointers, empty callbacks, and empty spans are intentional existing request/status defaults, not UI or behavior stubs.

## Threat Flags

None. The new trust boundary is the planned `io/loader -> io/read` public batch dispatch and is covered by T-225-04..06.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 03 can replace model-loader per-tensor child dispatch with the public `io/loader` batch route added here.

## Self-Check: PASSED

- Found `src/emel/io/loader/events.hpp`.
- Found `src/emel/io/loader/actions.hpp`.
- Found `src/emel/io/loader/sm.hpp`.
- Found `tests/io/loader/lifecycle_tests.cpp`.
- Found `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-02-SUMMARY.md`.
- Found task commit `8c4b8fc2`.
- Found task commit `56295e65`.

---
*Phase: 225-read-closeout-runtime-validation-and-sml-repair*
*Completed: 2026-05-06*
