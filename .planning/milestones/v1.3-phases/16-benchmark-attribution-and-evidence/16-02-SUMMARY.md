---
phase: 16-benchmark-attribution-and-evidence
plan: 02
subsystem: benchmark-publication
tags: [bench, docsgen, snapshots, aarch64, evidence]
requires:
  - phase: 16-01
    provides: live compare attribution and ARM compare-surface proof
provides:
  - preserved shared-scalar ARM baseline artifact for the canonical short compare case
  - refreshed maintained compare snapshot with optimized/shared flash attribution
  - generated benchmark docs that publish measured improvement over the preserved ARM baseline
affects: [milestone-audit]
tech-stack:
  added: []
  patterns: [approval-gated benchmark publication, backward-compatible metadata parsing]
key-files:
  created:
    [snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt]
  modified:
    [snapshots/bench/benchmarks_compare.txt, tools/docsgen/docsgen.cpp, docs/benchmarks.md]
key-decisions:
  - "Preserve the pre-refresh canonical short compare row as a dedicated ARM baseline artifact before refreshing the maintained compare snapshot."
  - "Keep `tools/docsgen` compatible with older aggregate-only `generation_flash_evidence` metadata while publishing optimized/shared counts when present."
patterns-established:
  - "Maintained benchmark publication preserves prior evidence in a separate artifact before refreshing snapshot-driven docs."
  - "Optimized/shared flash attribution is a durable publication surface, not only a live CLI proof."
requirements-completed: [BENCH-04, BENCH-05, BENCH-06]
duration: 0min
completed: 2026-03-22
---

# Phase 16 Plan 2 Summary

**Phase 16 now publishes maintained optimized ARM flash evidence with an explicit preserved
baseline**

## Accomplishments

- Added `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt` to preserve the
  canonical short compare row from the last maintained shared-scalar ARM snapshot before refresh.
- Updated `tools/docsgen/docsgen.cpp` so benchmark publication reads the new preserved baseline
  artifact, publishes optimized/shared flash attribution, and remains compatible with older
  aggregate-only `# generation_flash_evidence:` metadata lines.
- Refreshed `snapshots/bench/benchmarks_compare.txt` and regenerated `docs/benchmarks.md` after
  explicit approval. The maintained canonical short compare row now reports
  `emel.cpp 6133750.000 ns/op`, `llama.cpp 3028833.000 ns/op`, and
  `optimized_flash_dispatch_calls=2 shared_flash_dispatch_calls=0`.
- Verified the preserved-baseline proof with
  `python3 tools/bench/compare_flash_baseline.py`, which reported `speedup=1.140x` and
  `latency_drop_pct=12.3` for
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `rg 'NEVER update snapshots without explicit user consent' AGENTS.md`
- approval checkpoint recorded via `request_user_input` with `Approve updates`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 scripts/bench.sh --compare-update`
- `scripts/generate_docs.sh`
- `scripts/generate_docs.sh --check`
- `python3 tools/bench/compare_flash_baseline.py --baseline snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt --current snapshots/bench/benchmarks_compare.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `scripts/quality_gates.sh`

## Deviations from Plan

- The maintained compare refresh updates the whole checked-in compare snapshot, not only the
  canonical flash row, because `scripts/bench.sh --compare-update` publishes the full benchmark
  suite by design.

## Next Readiness

- Phase 16 is ready to close and satisfies the milestone's maintained benchmark-evidence goal.
- With all milestone phases complete, the autonomous flow can move to milestone audit and archive
  work.

---
*Phase: 16-benchmark-attribution-and-evidence*
*Completed: 2026-03-22*
