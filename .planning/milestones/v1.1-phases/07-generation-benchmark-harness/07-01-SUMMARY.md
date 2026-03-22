---
phase: 07-generation-benchmark-harness
plan: 01
subsystem: benchmarking
tags: [benchmark, generation, gguf, llama.cpp, tools/bench]
requires:
  - phase: 06-fixture-contract-hardening
    provides: canonical `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` fixture contract
provides:
  - canonical generation benchmark case wired into `bench_runner`
  - preloaded request-latency harness for the pinned Llama-68M EMEL generation path
  - paired EMEL/reference case appenders for later compare-mode work
affects: [07-02 workload contract, 08-generation-compare-output, 09-benchmark-integration-hardening]
tech-stack:
  added: []
  patterns: [cached benchmark fixture setup outside timed loop, per-iteration request session setup]
key-files:
  created: [tools/bench/generation_bench.cpp]
  modified: [tools/bench/CMakeLists.txt, tools/bench/bench_cases.hpp, tools/bench/bench_main.cpp]
key-decisions:
  - "Keep the generation harness inside `tools/bench` and mirror the shipped paritychecker load/initialize/generate flow instead of adding benchmark-only runtime hooks."
  - "Cache the canonical fixture outside the timed loop and rebuild request-local generation state per iteration so the benchmark measures preloaded request latency rather than model load time."
  - "Clamp the generation case through dedicated generation env knobs so `bench_runner --mode=emel` stays bounded even though the global bench defaults are very large."
patterns-established:
  - "Generation benchmark cases should preload fixture state once, then time only request-local setup and bounded generation."
  - "Heavy benchmark fixture/session storage should avoid stack allocation when it carries `emel::model::data` or loader arenas."
requirements-completed: [BENCH-01, BENCH-02]
duration: 43min
completed: 2026-03-09
---

# Phase 7 Plan 1: Generation Benchmark Harness Summary

**Canonical Llama-68M preloaded request-latency benchmark wiring in `tools/bench` with one bounded EMEL generation case and paired case registration for future compare flow**

## Performance

- **Duration:** 43 min
- **Started:** 2026-03-09T03:02:00Z
- **Completed:** 2026-03-09T03:45:41Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `tools/bench/generation_bench.cpp` and wired it into `bench_runner`.
- Registered one canonical generation benchmark case in the standard bench case list.
- Reused the proven generation slice so the timed path covers request-local preparation and bounded generation against `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.

## Task Commits

None - the user requested no commits for this execution and `commit_docs` was `false`.

## Files Created/Modified

- `tools/bench/generation_bench.cpp` - canonical generation benchmark harness and paired EMEL/reference appenders
- `tools/bench/CMakeLists.txt` - adds the benchmark source and repo-root compile definition for fixture resolution
- `tools/bench/bench_cases.hpp` - declares generation case appenders
- `tools/bench/bench_main.cpp` - registers the generation case in the default bench registry

## Decisions Made

- Used a cached canonical fixture and per-run generation session setup so timed iterations exclude one-time model loading.
- Kept the benchmark contract narrow to one prompt (`hello`) and one bounded token budget, with dedicated generation env overrides on top of the existing bench config.
- Preserved paired EMEL/reference bench registration now so later compare work can reuse the same case surface.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed stack-overflow paths from benchmark-local generation state**
- **Found during:** Task 2 (Implement the EMEL generation benchmark harness)
- **Issue:** Heavy benchmark fixture/session state containing model data and loader-owned storage caused the new generation case to crash during isolated execution.
- **Fix:** Reworked the harness to keep canonical fixture state cached outside the timed loop and to allocate per-request generation session state safely for each measured iteration.
- **Files modified:** `tools/bench/generation_bench.cpp`
- **Verification:** `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=emel` and `scripts/bench.sh --emel-only`
- **Committed in:** None - no commits requested for this execution

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix was required for correctness and kept the work inside the planned `tools/bench` scope.

## Issues Encountered

- `scripts/quality_gates.sh` builds and passes `emel_tests`, but the coverage-report step fails because `gcovr` tries to enter a missing `build/coverage/CMakeFiles/emel_tests_bin.dir/tests/kernel` directory. This was recorded in `.planning/phases/07-generation-benchmark-harness/deferred-items.md` as an out-of-scope repo issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `bench_runner --mode=emel` now includes `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.
- Phase `07-02` can tighten the benchmark contract and surface wording without needing more bench harness plumbing.
- Phase `08` can build compare-mode reporting on top of the registered paired generation case.

## Verification

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --emel-only`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=emel | rg 'generation/'`
- `scripts/quality_gates.sh` (tests pass; coverage-report step fails on an existing `gcovr` path issue outside this plan)

---
*Phase: 07-generation-benchmark-harness*
*Completed: 2026-03-09*
