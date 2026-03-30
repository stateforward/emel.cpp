---
phase: 26-canonical-qwen3-fixture-and-conditioning-contract
plan: 04
subsystem: tooling
tags: [paritychecker, bench, gguf, formatter-contract]
requires:
  - phase: 26-canonical-qwen3-fixture-and-conditioning-contract
    provides: structured generator request boundary
provides:
  - primary-template formatter contract classification on maintained tool paths
  - formatter-contract publication on truthful parity and bench pre-runtime failure surfaces
  - shared tool-only formatter adapter for the maintained Qwen slice
affects: [phase-27, phase-28, phase-29, tools/paritychecker, tools/bench]
tech-stack:
  added: []
  patterns: [tool-only GGUF formatter contract helper, explicit pre-runtime failure publication]
key-files:
  created:
    - .planning/phases/26-canonical-qwen3-fixture-and-conditioning-contract/26-04-SUMMARY.md
    - tools/generation_formatter_contract.hpp
  modified:
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
    - tools/bench/generation_bench.cpp
    - tools/bench/bench_main.cpp
    - .planning/ROADMAP.md
    - .planning/STATE.md
key-decisions:
  - "Kept formatter contract classification and adapter logic tool-only under tools/."
  - "Resolved the maintained formatter contract directly from GGUF KV entries instead of partially populating model metadata."
patterns-established:
  - "Maintained tool surfaces publish formatter contract before truthful pre-runtime failure."
  - "Primary-template support is classified explicitly with no raw fallback on maintained paths."
requirements-completed: [COND-01]
duration: 22min
completed: 2026-03-28
---

# Phase 26: Maintained Formatter Contract Publication Summary

**Maintained paritychecker and bench now derive one explicit formatter contract from the Qwen
GGUF primary template and publish it on truthful pre-runtime failure surfaces**

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-28T03:31:49Z
- **Completed:** 2026-03-28T03:53:18Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added a shared tool-only formatter contract helper for the maintained Qwen slice under
  `tools/`, with explicit supported/unsupported classification and no `src/` runtime expansion.
- Bound paritychecker and bench setup to the primary GGUF `tokenizer.chat_template`, published the
  resolved formatter contract, and kept both paths on truthful pre-runtime failure.
- Added formatter-contract subprocess coverage and verified the maintained parity and bench failure
  surfaces both show the approved contract string before `model_invalid` / `prepare_emel_fixture`.

## Task Commits

Each task was committed atomically:

1. **Task 1 + Task 2: Primary-template maintained formatter binding and publication** - `7c0f80a`
   (feat)

**Plan metadata:** `7c0f80a` (feat: wave summary/state checkpoint)

## Files Created/Modified
- `.planning/phases/26-canonical-qwen3-fixture-and-conditioning-contract/26-04-SUMMARY.md` -
  Captures Wave 4 outcomes and follow-on context.
- `tools/generation_formatter_contract.hpp` - Tool-only primary-template classifier and maintained
  formatter adapter.
- `tools/paritychecker/parity_runner.cpp` - Resolves and publishes `formatter_contract=` on
  maintained generation failure/setup paths.
- `tools/paritychecker/paritychecker_tests.cpp` - Pins maintained formatter contract behavior and
  helper classification.
- `tools/bench/generation_bench.cpp` - Resolves and publishes bench formatter contract on the real
  maintained compare failure path.
- `tools/bench/bench_main.cpp` - Prints `# generation_formatter_contract: ...` in maintained
  compare output.

## Decisions Made
- Kept the shared contract helper under `tools/` only after explicit user approval.
- Derived support directly from GGUF KV entries instead of forcing partially populated metadata into
  `model::data`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Tooling metadata edge] Removed partial model-metadata storage from the tool contract path**
- **Found during:** Task 2 (maintained formatter binding)
- **Issue:** `model::data::metadata::tokenizer_data.chat_template` is not a `std::string_view`
  sink for raw GGUF strings, so using it here would have been a misleading half-implementation.
- **Fix:** Resolved the formatter contract directly from tool-local GGUF KV entries on both parity
  and bench paths.
- **Files modified:** `tools/paritychecker/parity_runner.cpp`, `tools/bench/generation_bench.cpp`
- **Verification:** maintained parity/bench commands both publish the contract and fail truthfully
- **Committed in:** `7c0f80a` (feat)

---

**Total deviations:** 1 auto-fixed (tooling metadata edge)
**Impact on plan:** Keeps the change narrower and more honest. No expansion into `src/` runtime
metadata behavior.

## Issues Encountered
- Bench compilation needed one forward declaration after adding the tool-local formatter helper;
  fixed without changing behavior.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 26 is complete: canonical fixture identity, structured request boundary, and maintained
  formatter contract publication are all in place.
- Phase 27 can now focus purely on truthful Qwen runtime bring-up on the maintained generator path.

---
*Phase: 26-canonical-qwen3-fixture-and-conditioning-contract*
*Completed: 2026-03-28*
