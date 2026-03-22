---
phase: 15-runtime-adoption-and-proof
plan: 03
subsystem: parity-flash-attribution
tags: [paritychecker, generation, aarch64, proof, regression]
requires:
  - phase: 15-01
    provides: runtime optimized/shared flash accessors
  - phase: 15-02
    provides: negative zero-claim runtime proof
provides:
  - paritychecker generation output with optimized/shared flash attribution
  - ARM-specific proof that canonical parity runs stay on the optimized flash backend
  - regression closure under the repo's existing warning-only benchmark drift policy
affects: [16-benchmark-attribution-and-evidence]
tech-stack:
  added: []
  patterns: [CLI proof comments with backend attribution, ARM-only strict proof gates]
key-files:
  created: []
  modified:
    [tools/paritychecker/parity_runner.cpp, tools/paritychecker/paritychecker_tests.cpp, tests/generator/lifecycle_tests.cpp]
key-decisions:
  - "Extend the existing `flash_dispatch:` parity proof line instead of inventing a new output channel."
  - "Enforce optimized > 0 and shared == 0 only when the active runtime kernel kind is `aarch64`."
patterns-established:
  - "Maintained CLI proof should distinguish backend selection, not only aggregate flash dispatch."
  - "Repo-wide warning-only benchmark drift remains documented but untouched until explicit publication work begins."
requirements-completed: [ARCH-01, PAR-03, VER-02]
duration: 0min
completed: 2026-03-22
---

# Phase 15 Plan 3 Summary

**Paritychecker now publishes optimized-vs-shared ARM flash attribution and the full gate still passes**

## Accomplishments

- Extended
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  so generation proof now emits both `optimized_flash_dispatch_calls` and
  `shared_flash_dispatch_calls` in the summary line plus `optimized=` and `shared=` on the durable
  `flash_dispatch:` line.
- Added an ARM-specific parity guard in
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  that fails canonical generation proof if the active runtime kernel is `aarch64` but the run does
  not stay on the optimized flash path.
- Updated
  [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp)
  so maintained parity tests verify the new attribution fields and require optimized > 0 with
  shared == 0 on ARM, while keeping non-ARM builds explicit and benign.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/paritychecker_zig_latest --target paritychecker_tests paritychecker --parallel 8`
- `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `scripts/quality_gates.sh`

## Deviations from Plan

- `build/zig` does not own the paritychecker targets, so focused parity verification ran from the
  dedicated Zig paritychecker build tree at `build/paritychecker_zig_latest`.
- `scripts/quality_gates.sh` reported warning-only benchmark regressions for
  `batch/planner_equal`, `gbnf/rule_parser_complex`, and `logits/sampler_raw/vocab_256000`, then
  exited successfully under the repo's current policy.

## Next Readiness

- Phase 16 can build on the new parity attribution fields when measuring and publishing benchmark
  evidence.
- Checked-in benchmark snapshots remain untouched so the next phase can stop cleanly at the user
  approval gate if publication needs artifact refresh.

---
*Phase: 15-runtime-adoption-and-proof*
*Completed: 2026-03-22*
