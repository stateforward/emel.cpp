# Phase 16: Benchmark Attribution And Evidence - Research

**Date:** 2026-03-22
**Status:** Complete

## Key Findings

1. `tools/bench/generation_bench.cpp` currently records only aggregate
   `flash_dispatch_calls`; it does not yet capture the optimized/shared backend counts that Phase
   15 now exposes through `emel::generator::sm`.
2. `tools/bench/bench_main.cpp` hard-fails compare mode when aggregate flash evidence is missing
   and prints a stable `# generation_flash_evidence:` line, so the benchmark publication seam is
   already centralized and easy to extend.
3. `tools/docsgen/docsgen.cpp` parses the snapshot metadata with a strict regex that matches only
   the old aggregate flash evidence fields. It will need to accept the new optimized/shared fields
   before refreshed snapshot publication can succeed.
4. The current committed
   `snapshots/bench/benchmarks_compare.txt` is the last maintained compare artifact from before the
   ARM flash optimization. It is therefore the right source for a preserved shared-scalar ARM
   baseline artifact before the compare snapshot is refreshed.
5. `AGENTS.md` and the prior v1.2 benchmark-evidence plans require explicit user approval before
   any checked-in benchmark snapshot or generated benchmark doc is updated.

## Constraints

- Phase 16 cannot be declared done honestly unless maintained publication artifacts are refreshed.
- Approval is required before modifying `snapshots/bench/benchmarks_compare.txt`, creating a new
  preserved ARM baseline artifact under `snapshots/bench/`, or regenerating `docs/benchmarks.md`.
- Until approval is granted, any code changes must remain backward-compatible with the currently
  committed snapshot metadata.

## Chosen Direction

- First extend live compare output and validation to publish optimized/shared flash attribution on
  the benchmark runner.
- Then stop for explicit approval before preserving the current compare snapshot as
  `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`, refreshing
  `snapshots/bench/benchmarks_compare.txt`, and regenerating `docs/benchmarks.md`.

---
*Phase: 16-benchmark-attribution-and-evidence*
*Research completed: 2026-03-22*
