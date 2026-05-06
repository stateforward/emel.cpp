---
phase: 225-read-closeout-runtime-validation-and-sml-repair
plan: 01
subsystem: io
tags: [stateforward-sml, io-read, batch-copy, doctest, coverage]
requires:
  - phase: 225-read-closeout-runtime-validation-and-sml-repair
    provides: Phase context, research, SML repair constraints, and io/read ownership rules.
provides:
  - Shared `emel::io::event::tensor_load_span` contract.
  - Public `io/read` `read_tensor_batch` request and `_done`/`_error` events.
  - Guard-selected batch validation chain with bounded source-span copy in `io/read`.
  - Public-dispatch doctests for copied-byte success and explicit batch failures.
affects: [io-read, io-loader, model-loader, phase-225]
tech-stack:
  added: []
  patterns:
    - Request-owned `std::span<const tensor_load_span>` batch payloads.
    - Guard-selected SML batch outcome chain before action-owned byte copy.
key-files:
  created:
    - src/emel/io/events.hpp
  modified:
    - src/emel/io/read/events.hpp
    - src/emel/io/read/detail.hpp
    - src/emel/io/read/guards.hpp
    - src/emel/io/read/actions.hpp
    - src/emel/io/read/sm.hpp
    - tests/io/read/lifecycle_tests.cpp
key-decisions:
  - "Batch failed-index scans use monotonic branchless helper logic so actions do not call helper-local if/break control flow."
  - "Coverage gate changed-file input must be colon-separated; space-separated input is parsed as one filename by the existing script."
patterns-established:
  - "Batch copy path: guards classify aggregate validation/source outcomes, then one action copies the already-selected valid spans."
requirements-completed: [VAL-01, TIO-03]
duration: 21min
completed: 2026-05-06
---

# Phase 225 Plan 01: io/read Batch Copy Surface Summary

**Shared tensor-span contract plus public `io/read` batch dispatch that copies caller-owned spans and reports explicit batch outcomes.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-05-06T14:15:07Z
- **Completed:** 2026-05-06T14:35:50Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added `src/emel/io/events.hpp` with the shared `tensor_load_span` contract.
- Added `read_tensor_batch`, `read_tensor_batch_done`, and `read_tensor_batch_error`.
- Added an explicit SML batch path in `io/read` with guard-selected validation/error states and one bounded copy action.
- Added public-dispatch doctests for batch success, first failing span error publication, and representative batch error legs.

## Task Commits

1. **Task 1: Add public-dispatch io/read batch doctests** - `3ba20968` (`test`)
2. **Task 2: Implement shared tensor span and io/read batch copy path** - `f2f38cde` (`feat`)

## Files Created/Modified

- `src/emel/io/events.hpp` - Shared `emel::io::event::tensor_load_span`.
- `src/emel/io/read/events.hpp` - Public batch request and batch done/error events.
- `src/emel/io/read/detail.hpp` - Batch same-RTC runtime/status carriers.
- `src/emel/io/read/guards.hpp` - Batch validation, source-result, success/error predicates.
- `src/emel/io/read/actions.hpp` - Batch begin/error publication and bounded `std::memcpy` copy action.
- `src/emel/io/read/sm.hpp` - Batch transition chain and public `process_event(read_tensor_batch)`.
- `tests/io/read/lifecycle_tests.cpp` - Public-dispatch batch success and failure coverage.

## Decisions Made

- Kept batch source-byte copy inside `src/emel/io/read`; higher layers can route spans without owning byte-copy work.
- Used branchless failed-index scans for batch error payloads to avoid action-called `if`/`break` helper control flow.
- Treated `EMEL_QUALITY_GATES_CHANGED_FILES` as colon-separated input after discovering the script does not split on spaces.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added broader batch error coverage for the changed-file quality gate**
- **Found during:** Task 2
- **Issue:** The implementation passed focused doctests, but exact changed-header coverage stayed below threshold until representative batch error legs were covered.
- **Fix:** Added public-dispatch doctests for invalid request, unsupported resource, source open failure, source seek failure, file read failure, fail-closed missing callback, and same-RTC error capture.
- **Files modified:** `tests/io/read/lifecycle_tests.cpp`
- **Verification:** Changed-file quality gate passed with 98.5% line and 78.1% branch coverage.
- **Committed in:** `f2f38cde`

**2. [Rule 1 - Bug] Removed action-called helper branching from failed-index scans**
- **Found during:** Task 2 SML rule review
- **Issue:** Initial batch error actions called first-failure helpers containing `if`/`break`, which was too close to hidden action-called runtime control flow.
- **Fix:** Replaced those scans with monotonic branchless index accumulation helpers.
- **Files modified:** `src/emel/io/read/guards.hpp`
- **Verification:** `rg -n "if \\(|break;" src/emel/io/read/guards.hpp src/emel/io/read/actions.hpp` returned no matches; focused tests and quality gate passed.
- **Committed in:** `f2f38cde`

**Total deviations:** 2 auto-fixed (1 blocking verification, 1 SML-rule bug)
**Impact on plan:** Both changes strengthened the planned behavior and rule compliance without widening runtime scope.

## Issues Encountered

- The RED CTest command initially passed because it used the stale test binary. Rebuilding `emel_tests_bin` produced the expected RED failure: missing `emel/io/events.hpp`.
- A pre-existing staged `.planning/REQUIREMENTS.md` deletion was accidentally included in the first Task 1 commit attempt. The commit was immediately amended so the final Task 1 commit contains only `tests/io/read/lifecycle_tests.cpp`.
- The first quality-gate runs used a space-separated `EMEL_QUALITY_GATES_CHANGED_FILES`; the script parses comma/colon/newline separators, so gcovr matched zero changed files. The final gate used colon-separated paths and passed.

## Validation

- `cmake --build build/zig --target emel_tests_bin` - failed during RED as expected on missing `emel/io/events.hpp`; passed after Task 2.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` - passed.
- `rg -n "struct read_tensor_batch|read_tensor_batch_done|read_tensor_batch_error" src/emel/io/read/events.hpp` - found expected declarations.
- `rg -n "effect_mark_read_tensor_batch_done|std::memcpy" src/emel/io/read/actions.hpp` - found expected copy action and `std::memcpy`.
- `rg -n "read_tensor_batch_runtime|process_event\\(const event::read_tensor_batch" src/emel/io/read/detail.hpp src/emel/io/read/sm.hpp` - found expected runtime carrier and wrapper.
- `rg -n "process_queue|defer_queue|for \\(.*process_event|io_loader->process_event" src/emel/io/read src/emel/io/events.hpp` - no matches.
- `rg -n "std::vector|new |malloc|process_queue|defer_queue" src/emel/io/read src/emel/io/events.hpp` - no matches.
- `scripts/check_domain_boundaries.sh` - passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/io/events.hpp:src/emel/io/read/events.hpp:src/emel/io/read/detail.hpp:src/emel/io/read/guards.hpp:src/emel/io/read/actions.hpp:src/emel/io/read/sm.hpp:tests/io/read/lifecycle_tests.cpp" EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh` - passed.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02 can route batch load requests through `io/loader` using the shared span contract and the public `io/read` batch dispatch surface.

## Self-Check: PASSED

- Found `src/emel/io/events.hpp`.
- Found `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-01-SUMMARY.md`.
- Found task commit `3ba20968`.
- Found task commit `f2f38cde`.

---
*Phase: 225-read-closeout-runtime-validation-and-sml-repair*
*Completed: 2026-05-06*
