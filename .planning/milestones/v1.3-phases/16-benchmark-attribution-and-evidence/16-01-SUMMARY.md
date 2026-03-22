---
phase: 16-benchmark-attribution-and-evidence
plan: 01
subsystem: bench-flash-attribution
tags: [bench, aarch64, flash-attention, compare-surface, proof]
requires:
  - phase: 15-runtime-adoption-and-proof
    provides: runtime optimized/shared flash observability and parity proof
provides:
  - live benchmark compare metadata for optimized vs shared ARM flash dispatch
  - ARM-specific compare-surface validation that rejects shared-fallback claims
  - verified benchmark-runner changes without touching checked-in benchmark artifacts
affects: [16-02 publication checkpoint]
tech-stack:
  added: []
  patterns: [compare-surface attribution comments, live ARM benchmark proof]
key-files:
  created: []
  modified:
    [tools/bench/generation_bench.cpp, tools/bench/bench_main.cpp]
key-decisions:
  - "Extend the existing `# generation_flash_evidence:` compare comment instead of creating a separate benchmark proof channel."
  - "Treat non-AArch64 hosts explicitly by requiring zero optimized/shared flash counts there."
patterns-established:
  - "Benchmark proof should distinguish optimized-vs-shared backend execution, not only aggregate flash dispatch."
  - "Checked-in snapshot publication remains a separate approval gate even when live compare proof is ready."
requirements-completed: [BENCH-04, BENCH-05]
duration: 0min
completed: 2026-03-22
---

# Phase 16 Plan 1 Summary

**The live compare runner now publishes optimized-vs-shared ARM flash attribution**

## Accomplishments

- Extended
  [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp)
  so the canonical generation evidence now captures optimized and shared flash dispatch deltas from
  the shipped generator seam alongside the existing aggregate flash count.
- Updated
  [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp)
  so `bench_runner --mode=compare` publishes
  `optimized_flash_dispatch_calls=` and `shared_flash_dispatch_calls=` on the maintained
  `# generation_flash_evidence:` line.
- Added ARM-specific validation in
  [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp)
  that rejects canonical compare runs which claim AArch64 benchmarking but fail to stay on the
  optimized flash path.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^# generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .*optimized_flash_dispatch_calls=[1-9][0-9]* .*shared_flash_dispatch_calls=0 '`
- `scripts/quality_gates.sh`

## Deviations from Plan

- The full gate reported a new warning-only benchmark regression on
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` after the live compare change,
  but `scripts/quality_gates.sh` still exited successfully under the repo's current policy.
- Plan 1 intentionally stopped short of snapshot or docs publication. The maintained compare
  artifact and generated benchmark docs remain unchanged until approval is explicit.

## Next Readiness

- Phase 16 is ready for the repo-mandated approval checkpoint before any benchmark snapshot or
  generated docs artifact changes.
- The exact publication set is now clear: preserve the current shared-scalar ARM baseline artifact,
  refresh the compare snapshot, and regenerate benchmark docs.

---
*Phase: 16-benchmark-attribution-and-evidence*
*Completed: 2026-03-22*
