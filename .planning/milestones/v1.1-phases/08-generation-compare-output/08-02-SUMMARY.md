---
phase: 08-generation-compare-output
plan: 02
subsystem: generation-compare-publishing
tags: [generation, benchmark, compare, publishing]
requires:
  - phase: 08-generation-compare-output
    provides: canonical compare contract hardening from 08-01
provides:
  - a clean standard `scripts/bench.sh --compare` stdout surface for compare rows
  - stderr-only build chatter and seam-audit diagnostics
  - published canonical generation compare evidence through the normal script workflow
affects: [09-benchmark-integration-hardening]
tech-stack:
  added: []
  patterns: [stdout compare rows, stderr diagnostics, script-surface hardening]
key-files:
  created: []
  modified: [scripts/bench.sh]
key-decisions:
  - "Keep the compare row format unchanged and move script build chatter to stderr instead of inventing a new report."
  - "Preserve seam-audit diagnostics as stderr-only support evidence so the normal compare stdout surface stays stable."
  - "Reuse the existing `bench_runner --mode=compare` output as the published evidence surface instead of adding a second compare command."
patterns-established:
  - "The standard compare script now behaves like a publishable report surface: compare rows on stdout, operational diagnostics on stderr."
requirements-completed: [COMP-01, COMP-02]
duration: 0min
completed: 2026-03-10
---

# Phase 08 Plan 2 Summary

**The normal compare script now publishes the canonical generation row cleanly**

## Accomplishments

- Updated `scripts/bench.sh` so compare-path configure/build chatter now goes to stderr instead of
  stdout.
- Verified that the standard `scripts/bench.sh --compare` stdout stream still includes the
  canonical generation compare row in the existing `emel.cpp ... llama.cpp ... ratio=...` style.
- Verified that seam-audit evidence remains opt-in and stderr-only, with the EMEL path showing
  zero reference-wrapper hits and the explicit reference path still showing positive seam hits.

## Task Commits

None. Execution stayed local with `commit_docs` disabled.

## Verification

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "emel\\.cpp .* llama\\.cpp .* ratio="`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
- `EMEL_BENCH_CASE_INDEX=7 EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

## Deviations from Plan

- No additional `tools/bench/bench_main.cpp` edits were needed in this wave; the row-shape and
  canonical-pairing work from `08-01` was already sufficient once the script stopped polluting
  stdout with build chatter.

## Next Readiness

- Phase 08 is ready to close.
- Phase 09 can focus on snapshot and tooling integration rather than compare-surface cleanup.

---
*Phase: 08-generation-compare-output*
*Completed: 2026-03-10*
