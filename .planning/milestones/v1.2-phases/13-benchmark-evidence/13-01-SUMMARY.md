---
phase: 13-benchmark-evidence
plan: 01
subsystem: infra
tags: [benchmark, flash-attention, compare, reference, truthfulness]
requires:
  - phase: 12-parity-and-verification-closure
    provides: deterministic reference policy and durable flash-proof expectations for the canonical workload
provides:
  - deterministic fetched reference sourcing for `tools/bench`
  - compare-surface reference identity and canonical flash-evidence comments
  - preserved benchmark row format for the maintained generation compare workflow
affects: [13, benchmark, snapshots, docs, paritychecker]
tech-stack:
  added: []
  patterns:
    - compare-mode publication can add machine-checkable proof comments without changing the maintained benchmark row shape
    - benchmark reference truth now mirrors paritychecker's fetched-or-pinned reference policy
key-files:
  created: []
  modified:
    - tools/bench/CMakeLists.txt
    - tools/bench/bench_main.cpp
    - tools/bench/generation_bench.cpp
key-decisions:
  - "The benchmark tool now always uses the configured fetched reference instead of a local `tmp/llama.cpp` override."
  - "Existing `scripts/bench.sh` compare/update behavior was kept unchanged because it already preserves the benchmark tool's stdout, including the new proof comments."
patterns-established:
  - "Canonical flash proof on benchmark publication lives in comment metadata ahead of the normal compare rows."
  - "Compare execution hard-fails when the canonical short EMEL case cannot prove flash dispatch truthfully."
requirements-completed: [BENCH-01]
duration: 18min
completed: 2026-03-22
---

# Phase 13 Plan 01: Deterministic Compare Proof Summary

**The maintained benchmark compare surface now proves which reference build ran and that the
canonical short EMEL generation case actually executed flash attention.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-22T03:32:26-0500
- **Completed:** 2026-03-22T03:50:52-0500
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Removed the benchmark tool's machine-local `tmp/llama.cpp` preference and replaced it with the
  deterministic fetched-or-pinned reference policy already used by paritychecker.
- Added durable `# reference_impl:` and `# generation_flash_evidence:` comment lines to
  `bench_runner --mode=compare` without changing the existing compare row format.
- Verified the canonical short compare row still appears on the normal `scripts/bench.sh --compare`
  surface and that full `scripts/quality_gates.sh` remained green.

## Task Commits

No git commits were created in this workspace run.

## Files Created/Modified
- `tools/bench/CMakeLists.txt` - removes the local reference override and publishes resolved
  reference metadata to `bench_runner`.
- `tools/bench/bench_main.cpp` - emits durable compare-surface proof comments and validates the
  canonical flash-evidence contract before printing rows.
- `tools/bench/generation_bench.cpp` - captures canonical short-case flash-dispatch and seam-proof
  data for compare-mode publication.

## Decisions Made
- Kept the compare row names and row format unchanged so existing snapshots, grep checks, and
  docsgen inputs continue to work.
- Left `scripts/bench.sh` unmodified because its compare and compare-update paths already preserve
  the benchmark tool's stdout exactly, including the new proof comments.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `scripts/quality_gates.sh` spent most of its time in the normal coverage plus docsgen path, but
  it completed successfully with the new compare proof in place.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
Plan 13-02's comparator and runbook can now rely on a deterministic compare surface that publishes
reference identity plus canonical flash-execution proof.
The phase is ready to stop at the explicit approval checkpoint before any checked-in snapshot
artifact change.

---
*Phase: 13-benchmark-evidence*
*Completed: 2026-03-22*
