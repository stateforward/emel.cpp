---
phase: 225-read-closeout-runtime-validation-and-sml-repair
plan: 04
subsystem: model-loader
tags: [stateforward-sml, model-loader, read-copy, maintained-tools, guardrails]
requires:
  - phase: 225-read-closeout-runtime-validation-and-sml-repair
    provides: Plan 03 model-loader request-owned `io_load_spans` batch route.
provides:
  - Maintained generation, Sortformer, embedded probe, and paritychecker caller wiring for model-loader `io_load_spans`.
  - Source guardrails requiring maintained EMEL lanes to keep public source loading, strategy evidence, and batch scratch wiring.
affects: [model-loader, io-loader, io-read, maintained-tools, phase-225]
tech-stack:
  added: []
  patterns:
    - Request-lifetime `std::vector<emel::io::event::tensor_load_span>` scratch resized before model-loader dispatch.
    - Source guardrails over maintained tools for public read/copy evidence.
key-files:
  created:
    - .planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-04-SUMMARY.md
  modified:
    - tools/bench/generation_bench.cpp
    - tools/bench/diarization/sortformer_fixture.hpp
    - tools/embedded_size/emel_probe/main.cpp
    - tools/paritychecker/parity_engines.cpp
    - tests/model/loader/lifecycle_tests.cpp
key-decisions:
  - "Maintained caller scratch is owned by the request fixture/state and resized before dispatch, matching the existing effect request/result setup pattern."
  - "Guardrails stay source-scoped to maintained EMEL lanes and reject actor-internal io/read reach-through."
patterns-established:
  - "Maintained model-loader callers bind `effect_requests`, `effect_results`, and `io_load_spans` together before selecting the public load strategy."
requirements-completed: [TIO-03, VAL-04, VAL-01]
duration: 7min
completed: 2026-05-06
---

# Phase 225 Plan 04: Maintained Caller Batch Scratch Summary

**Maintained read/copy tool lanes now provide request-owned model-loader batch scratch while preserving public source loading and strategy evidence.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-06T15:20:27Z
- **Completed:** 2026-05-06T15:27:20Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `io_load_spans` storage to the maintained generation, Sortformer, embedded probe, and paritychecker EMEL lane state.
- Resized that scratch storage to `emel::model::data::k_max_tensors` before each model-loader dispatch and assigned it to `event::load::io_load_spans`.
- Preserved public `emel::io::source::load_file_bytes` setup-time source loading and `.used_io_strategy = ev.used_io_strategy` propagation.
- Extended the maintained-tool guardrail to require `io_load_spans` in all four maintained caller files.

## Task Commits

1. **Task 1: Wire maintained callers to request-owned io_load_spans** - `724a12ff` (`feat`)
2. **Task 2: Extend maintained caller guardrails** - `c04b8901` (`test`)

## Files Created/Modified

- `tools/bench/generation_bench.cpp` - Generation benchmark EMEL fixture owns and passes `io_load_spans`.
- `tools/bench/diarization/sortformer_fixture.hpp` - Sortformer fixture owns and passes `io_load_spans`.
- `tools/embedded_size/emel_probe/main.cpp` - Embedded probe fixture owns and passes `io_load_spans`.
- `tools/paritychecker/parity_engines.cpp` - Paritychecker EMEL generation state owns and passes `io_load_spans`.
- `tests/model/loader/lifecycle_tests.cpp` - Maintained-tool guardrail now requires `io_load_spans`.
- `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-04-SUMMARY.md` - Execution summary.

## Decisions Made

- Used the same request-lifetime fixture/state storage style as existing `effect_requests` and `effect_results`, resizing before dispatch so the SML dispatch path does not allocate scratch.
- Kept maintained tools on public `emel::io::source::load_file_bytes` and model-loader callbacks; no `io/read` detail/events or direct `read_tensor_request` construction were introduced.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The worktree contained many unrelated dirty and untracked files before execution. Task commits staged only the Plan 04 files listed above.

## Validation

- `rg -n "io_load_spans|emel::io::source::load_file_bytes|\\.used_io_strategy = ev.used_io_strategy" tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_engines.cpp` - passed; found all required caller wiring and evidence.
- `test "$(rg -n "emel/io/read/detail.hpp|emel/io/read/events.hpp|read_tensor_request|EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION|--update" tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_engines.cpp | wc -l | tr -d ' ')" = "0"` - passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/generation_bench.cpp:tools/bench/diarization/sortformer_fixture.hpp:tools/embedded_size/emel_probe/main.cpp:tools/paritychecker/parity_engines.cpp" scripts/quality_gates.sh` - passed; generation and Sortformer benchmark snapshots checked, paritychecker tests passed.
- `cmake --build build/zig --target emel_tests_bin` - passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` - passed.
- `scripts/check_domain_boundaries.sh` - passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="tests/model/loader/lifecycle_tests.cpp" scripts/quality_gates.sh` - passed.
- `rg -n "io_load_spans|emel::io::source::load_file_bytes|used_io_strategy" tests/model/loader/lifecycle_tests.cpp` - passed; found the retained and new guardrail checks.

## Known Stubs

None. The touched empty vectors, null pointers, and empty callback/default fields are existing fixture/request initialization patterns and do not represent incomplete Plan 04 behavior.

## Threat Flags

None. The touched trust boundary is the planned maintained-tool to model-loader public event boundary.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 05 can reconcile documentation and closeout path truth with maintained callers now wired to the Plan 03 batch scratch surface.

## Self-Check: PASSED

- Found `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-04-SUMMARY.md`.
- Found task commit `724a12ff`.
- Found task commit `c04b8901`.

---
*Phase: 225-read-closeout-runtime-validation-and-sml-repair*
*Completed: 2026-05-06*
