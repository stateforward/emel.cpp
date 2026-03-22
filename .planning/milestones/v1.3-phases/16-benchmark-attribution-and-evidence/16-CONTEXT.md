# Phase 16: Benchmark Attribution And Evidence - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 16 publishes maintained benchmark evidence for the optimized ARM flash path on the canonical
compare surface. This phase extends benchmark attribution, preserves the prior shared-scalar ARM
baseline as a separate artifact, and refreshes the generated benchmark publication only after the
required user approval for checked-in snapshot changes.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Keep the compare workflow on the existing `tools/bench` -> `scripts/bench.sh` ->
  `snapshots/bench` -> `tools/docsgen` -> `docs/benchmarks.md` path.
- Prepare live compare attribution and validation automatically, but stop for explicit approval
  before updating checked-in benchmark snapshots or generated benchmark docs.
- Preserve the current committed compare snapshot as the shared-scalar ARM baseline in a dedicated
  artifact before refreshing the compare surface for optimized-path publication.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_bench.cpp` already captures canonical generation flash-dispatch evidence
  for the compare surface.
- `tools/bench/bench_main.cpp` already publishes `# reference_impl:` and
  `# generation_flash_evidence:` comments ahead of compare rows.
- `tools/docsgen/docsgen.cpp` already parses the compare snapshot and publishes a `Current Flash
  Evidence` section in `docs/benchmarks.md`.
- `tools/bench/compare_flash_baseline.py` already computes improvement summaries from a preserved
  baseline artifact plus the current compare snapshot.

### Established Patterns
- Durable publication lives in committed snapshot comments and generated benchmark docs, not only
  in one-off local command output.
- Baseline preservation should materialize as a dedicated snapshot artifact before the current
  compare snapshot is refreshed.
- Approval must be explicit before any checked-in benchmark snapshot or generated docs change.

### Integration Points
- `tools/bench/generation_bench.cpp` and `tools/bench/bench_main.cpp` are the live attribution
  seams for optimized/shared flash counts.
- `snapshots/bench/benchmarks_compare.txt` is the maintained compare artifact that will need a
  refresh after approval.
- `tools/docsgen/docsgen.cpp`, `docs/templates/benchmarks.md.j2`, and `docs/benchmarks.md` are the
  maintained publication surfaces that must reflect the new attribution fields and preserved ARM
  baseline.

</code_context>

<specifics>
## Specific Ideas

- Extend benchmark proof metadata from aggregate flash dispatch counts to optimized/shared AArch64
  counts.
- Preserve the current compare snapshot's canonical short generation row as
  `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt` before refreshing the compare
  snapshot.
- Refresh generated benchmark docs only after the compare snapshot and preserved ARM baseline
  artifact are both in place.

</specifics>

<deferred>
## Deferred Ideas

- Broader non-ARM benchmark publication remains out of scope for v1.3.

</deferred>

---
*Phase: 16-benchmark-attribution-and-evidence*
*Context gathered: 2026-03-22*
