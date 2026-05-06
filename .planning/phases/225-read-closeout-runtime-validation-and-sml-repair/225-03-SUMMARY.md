---
phase: 225-read-closeout-runtime-validation-and-sml-repair
plan: 03
subsystem: model-loader
tags: [stateforward-sml, model-loader, io-loader, read-copy, batch-dispatch, doctest]
requires:
  - phase: 225-read-closeout-runtime-validation-and-sml-repair
    provides: Plan 01 shared io tensor spans and Plan 02 public io/loader load_tensor_batch route.
provides:
  - Model-loader request-owned `io_load_spans` scratch for batch read/copy metadata.
  - One public `io/loader::event::load_tensor_batch` dispatch from model-loader.
  - Guard-selected batch scratch readiness and read_copy used-strategy evidence after batch success.
  - Public-dispatch doctests and source guardrails for the batch route.
affects: [model-loader, io-loader, io-read, phase-225]
tech-stack:
  added: []
  patterns:
    - Request-owned batch scratch spans in public model-loader load events.
    - Same-RTC concrete callbacks record io/loader batch done/error into model-loader phase events.
key-files:
  created:
    - .planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-03-SUMMARY.md
  modified:
    - src/emel/io/loader/events.hpp
    - src/emel/io/loader/actions.hpp
    - src/emel/model/loader/events.hpp
    - src/emel/model/loader/guards.hpp
    - src/emel/model/loader/actions.hpp
    - src/emel/model/loader/sm.hpp
    - tests/model/loader/lifecycle_tests.cpp
key-decisions:
  - "Model-loader batch readiness is guarded in sm.hpp/guards.hpp; actions only bind already-selected tensor metadata and dispatch one child event."
  - "used_io_strategy is marked only after io_load_done_all observes the public io/loader batch success."
  - "io/loader batch errors now publish failed_index so model-loader can preserve concrete same-RTC error evidence."
patterns-established:
  - "model/loader read_copy path: tensor plan -> scratch-ready guard -> one load_tensor_batch dispatch -> guarded success/error publication."
requirements-completed: [TIO-03, VAL-04, VAL-01]
duration: 20min
completed: 2026-05-06
---

# Phase 225 Plan 03: Model Loader Batch I/O Dispatch Summary

**Model-loader read_copy now prepares request-owned batch spans and dispatches one public io/loader batch event before reporting used-strategy evidence.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-05-06T14:56:10Z
- **Completed:** 2026-05-06T15:16:27Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added failing model-loader doctests for two-tensor read_copy batch success, missing request-owned batch scratch, and source guardrails against the audited loop.
- Added `event::load::io_load_spans` and explicit guards for batch scratch ready/missing.
- Replaced `effect_dispatch_io_loads` with `effect_dispatch_io_load_batch`, which fills request-owned spans and calls `io_loader->process_event(load)` once.
- Kept `used_io_strategy == read_copy` behind `io_load_done_all`; missing scratch and batch error paths keep it `none`.

## Task Commits

1. **Task 1: Add model-loader batch orchestration regression tests** - `69e6039f` (`test`)
2. **Task 2: Replace model-loader per-tensor I/O loop with one batch dispatch** - `5c87f763` (`feat`)

## Files Created/Modified

- `src/emel/model/loader/events.hpp` - Added request-owned `io_load_spans`, `bytes_done`, `strategy_err`, and `failed_index` phase carriers.
- `src/emel/model/loader/guards.hpp` - Added batch scratch readiness/missing predicates and combined route guards.
- `src/emel/model/loader/actions.hpp` - Added batch callbacks and one public `load_tensor_batch` dispatch action; removed the audited per-tensor dispatch action.
- `src/emel/model/loader/sm.hpp` - Routed tensor plan completion through explicit batch scratch guards and kept used-strategy marking after batch done.
- `src/emel/io/loader/events.hpp` - Added batch error `failed_index` needed by the model-loader callback.
- `src/emel/io/loader/actions.hpp` - Published `failed_index` in batch error events.
- `tests/model/loader/lifecycle_tests.cpp` - Added RED/PASS batch route tests, missing-scratch tests, batch-error evidence tests, and source guardrails.

## Decisions Made

- Kept model-loader orchestration as one selected batch phase rather than a per-tensor completion chain.
- Used request-owned scratch spans instead of storing per-dispatch batch scratch in model-loader context.
- Preserved tensor actor reuse on missing scratch by running existing tensor effect error cleanup before publishing `io_strategy_unavailable`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Published `failed_index` from io/loader batch errors**
- **Found during:** Task 2
- **Issue:** Plan 02 recorded `failed_index` internally but `load_tensor_batch_error` did not expose it, while Plan 03's callback contract required `ev.failed_index`.
- **Fix:** Added `failed_index` to `src/emel/io/loader/events.hpp` and populated it in `src/emel/io/loader/actions.hpp`.
- **Verification:** Focused model/batch tests passed; changed-file quality gate passed.
- **Committed in:** `5c87f763`

**2. [Rule 1 - Bug] Preserved tensor actor reuse on missing batch scratch**
- **Found during:** Task 2 focused doctests
- **Issue:** The first missing-scratch route went directly to model-loader error, leaving the tensor actor in a non-reusable planned state.
- **Fix:** Routed missing scratch through `effect_dispatch_tensor_apply_error_results` cleanup before marking `io_strategy_unavailable`.
- **Verification:** `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` passed.
- **Committed in:** `5c87f763`

**3. [Rule 3 - Blocking] Added public-dispatch coverage for scoped quality gate**
- **Found during:** Task 2 quality gate
- **Issue:** Changed-file coverage initially failed at 84.4% line / 40.5% branch for changed model-loader headers.
- **Fix:** Added public-dispatch tests for parse error classification, validation policy branches, and batch error callback evidence.
- **Verification:** Changed-file quality gate passed at 91.9% line / 51.7% branch coverage.
- **Committed in:** `5c87f763`

**Total deviations:** 3 auto-fixed (2 blocking, 1 bug).
**Impact on plan:** All fixes were required to satisfy the declared callback contract, preserve actor reuse, or pass required gates. Runtime scope was not widened.

## Issues Encountered

- The first RED `ctest` run used a stale `build/zig` cache configured with `EMEL_TEST_SHARDS=io`, so the model-loader test file was not compiled. Reconfiguring with `EMEL_ZIG_TEST_SHARDS=model_and_batch` produced the expected RED compile failure on missing `io_load_spans`.
- The first implementation used a metadata `for (` loop in `effect_dispatch_io_load_batch`; the source guardrail required no `for (` in that function slice, so the binding loop was written as a monotonic `while` loop with the child dispatch outside the loop.

## Validation

- `rg -n "model loader read copy uses one io loader batch dispatch|model loader read copy requires request owned io batch span|effect_dispatch_io_loads" tests/model/loader/lifecycle_tests.cpp` - found the new tests/guardrail in RED.
- `EMEL_ZIG_TEST_SHARDS=model_and_batch scripts/build_with_zig.sh` - failed during RED as expected on missing `event::load::io_load_spans`; passed after implementation.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` - passed after implementation.
- `rg -n "io_load_spans|io_load_batch_span_ready|effect_dispatch_io_load_batch|effect_record_io_load_batch_done_event|effect_record_io_load_batch_error_event" src/emel/model/loader/events.hpp src/emel/model/loader/guards.hpp src/emel/model/loader/actions.hpp src/emel/model/loader/sm.hpp` - found expected declarations/usages.
- `test "$(rg -n "effect_dispatch_io_loads|for \\(.*io_loader->process_event|emel/io/read/events.hpp" src/emel/model/loader/actions.hpp | wc -l | tr -d ' ')" = "0"` - passed.
- `rg -n "for \\(.*io_loader->process_event|emel/io/read/events.hpp|read_tensor_request" src/emel/model/loader` - no matches.
- `scripts/check_domain_boundaries.sh` - passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/io/loader/events.hpp:src/emel/io/loader/actions.hpp:src/emel/model/loader/events.hpp:src/emel/model/loader/guards.hpp:src/emel/model/loader/actions.hpp:src/emel/model/loader/sm.hpp:tests/model/loader/lifecycle_tests.cpp" EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh` - passed.

## Known Stubs

None. Empty spans, null actor pointers, null buffers, and empty callbacks in the touched event types are intentional request defaults used for validation/error paths, not unimplemented behavior.

## Threat Flags

None. The touched trust boundaries are the planned `model/loader -> io/loader` batch dispatch and public evidence callbacks from the plan threat model.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 04 can update maintained callers to provide request-owned `io_load_spans` and continue proving the public read_copy evidence path through model-loader.

## Self-Check: PASSED

- Found `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-03-SUMMARY.md`.
- Found task commit `69e6039f`.
- Found task commit `5c87f763`.

---
*Phase: 225-read-closeout-runtime-validation-and-sml-repair*
*Completed: 2026-05-06*
