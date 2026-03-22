---
phase: 13-benchmark-evidence
plan: 04
subsystem: docs
tags: [benchmark, snapshots, docsgen, publication, flash-attention]
requires:
  - phase: 13-03
    provides: explicit user approval for checked-in benchmark artifact updates
provides:
  - a preserved canonical short-case pre-flash baseline artifact
  - refreshed compare snapshot proof comments on the maintained benchmark surface
  - generated benchmark docs that publish flash evidence plus the baseline comparison
affects: [13, benchmark, snapshots, docs]
tech-stack:
  added: []
  patterns:
    - preserved benchmark history is published as a checked-in key-value artifact plus generated docs instead of git-history-only evidence
    - benchmark docs derive flash-proof comments and preserved baseline math from maintained artifacts at docs-generation time
key-files:
  created:
    - snapshots/bench/generation_pre_flash_baseline.txt
  modified:
    - snapshots/bench/benchmarks_compare.txt
    - tools/docsgen/docsgen.cpp
    - docs/templates/benchmarks.md.j2
    - docs/benchmarks.md
key-decisions:
  - "The canonical short pre-flash baseline is preserved as a standalone repo artifact seeded from `git show 2acd4fe^:snapshots/bench/benchmarks_compare.txt`."
  - "The existing compare -> snapshot -> docsgen workflow stays intact; publication proof lives in snapshot comments and generated benchmark docs."
patterns-established:
  - "Flash benchmark publication now requires both a durable compare proof surface and a separately preserved pre-flash artifact."
  - "Generated docs can publish benchmark proof and speedup math without inventing a second benchmark output format."
requirements-completed: [BENCH-02, BENCH-03]
duration: 119min
completed: 2026-03-22
---

# Phase 13 Plan 04: Published Baseline And Benchmark Docs Summary

**The repo now preserves the canonical short pre-flash baseline as a maintained artifact, and the
generated benchmark docs publish the current flash proof plus a measured improvement over that
baseline.**

## Performance

- **Duration:** 119 min
- **Started:** 2026-03-22T03:54:00-0500
- **Completed:** 2026-03-22T05:52:36-0500
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Materialized `snapshots/bench/generation_pre_flash_baseline.txt` from
  `git show 2acd4fe^:snapshots/bench/benchmarks_compare.txt`, preserving the canonical short-case
  pre-flash EMEL and reference timings in the comparator's key-value contract.
- Refreshed `snapshots/bench/benchmarks_compare.txt` through
  `scripts/bench.sh --compare-update` so the maintained compare artifact carries both
  `# reference_impl:` and `# generation_flash_evidence:` proof comments.
- Extended docs generation so `docs/benchmarks.md` now publishes the preserved baseline artifact,
  the current flash-evidence row, and the computed short-case improvement summary
  `speedup=9.126x latency_drop_pct=89.0`.

## Task Commits

No git commits were created in this workspace run.

## Files Created/Modified
- `snapshots/bench/generation_pre_flash_baseline.txt` - preserves the canonical short pre-flash
  benchmark record in the repo-local comparator format.
- `snapshots/bench/benchmarks_compare.txt` - now includes durable compare-surface proof comments
  ahead of the maintained benchmark rows.
- `tools/docsgen/docsgen.cpp` - parses benchmark proof comments and the preserved baseline artifact,
  then computes the flash-vs-baseline publication section.
- `docs/templates/benchmarks.md.j2` - renders dedicated `Current Flash Evidence` and
  `Pre-Flash Baseline Comparison` sections above the normal compare table.
- `docs/benchmarks.md` - regenerated benchmark publication proving the canonical short flash case
  improved over the preserved pre-flash baseline.

## Decisions Made
- Kept the existing benchmark publication flow unchanged and added flash/baseline evidence through
  maintained artifacts instead of introducing a new report format or command surface.
- Preserved the historical baseline as a single-case artifact because BENCH-03 remains scoped to
  the canonical short case until a trustworthy maintained long-case baseline exists.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `scripts/quality_gates.sh` still emits the repo's existing non-blocking benchmark warning policy
  elsewhere in the suite, but Phase 13's new short-case proof and publication artifacts completed
  successfully and the timing snapshot updated normally.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
Phase 13 now has checked-in proof that the maintained benchmark workflow can publish flash evidence
truthfully and show a measurable short-case improvement over the preserved non-flash baseline. The
milestone is ready for phase verification and milestone audit.

---
*Phase: 13-benchmark-evidence*
*Completed: 2026-03-22*
