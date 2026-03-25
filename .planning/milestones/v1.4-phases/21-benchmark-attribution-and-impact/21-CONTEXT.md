# Phase 21: Benchmark Attribution And Impact - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 21 stays on the maintained benchmark publication surface. It does not widen the runtime path,
change Boost.SML structure, or revisit longer-decode parity scope. Its job is to make the
maintained ARM compare workflow truthfully publish the vectorized quantized path attribution, then
refresh maintained benchmark evidence for the active `1/10/100/1000` generation coverage.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Attribution
- Keep the canonical benchmark proof on the existing Q2_K GGUF generation fixture already used by
  `tools/bench/generation_bench.cpp`.
- Reuse the runtime attribution accessors already exposed by `generator::sm`; do not add new
  runtime hooks or public APIs.
- Publish q2/q3/q6 optimized/shared dispatch counts from compare mode alongside the existing flash
  evidence so maintained ARM compare output can distinguish vectorized quantized execution from
  shared row helpers.

### Benchmark Evidence Scope
- Keep the maintained generation benchmark set at `1`, `10`, `100`, and `1000` tokens, matching
  the widened benchmark cases already present in the worktree.
- Refresh both EMEL snapshot and compare snapshot baselines intentionally under explicit user
  approval.
- Record measurable improvement where it exists, but do not overstate unsupported cases if some
  lengths still regress versus the current v1.3 baseline.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_bench.cpp` already captures flash attribution for the canonical
  generation case and already widened the maintained cases to `1/10/100/1000`.
- `tools/bench/bench_main.cpp` already validates compare-mode flash evidence and is the natural
  place to extend compare publication for q2/q3/q6 attribution.
- `snapshots/bench/benchmarks.txt` and `snapshots/bench/benchmarks_compare.txt` are currently
  stale for the widened generation coverage and still reflect the old `1/8` surface.

### Integration Points
- The maintained benchmark gate uses `scripts/bench.sh --snapshot --compare`, so Phase 21 must
  land in the exact output format consumed by that script.
- `tools/bench/testdata/generation_compare_current.txt` is a maintained compare artifact that also
  needs regeneration once the compare output changes.

</code_context>

<specifics>
## Specific Ideas

- Extend the benchmark evidence state from flash-only to combined flash plus quantized q2/q3/q6
  attribution for the canonical `max_tokens=1` case.
- Keep compare-mode validation strict on AArch64: optimized q2/q3/q6 counts must be non-zero and
  shared counts must stay zero.
- Refresh compare and snapshot baselines with the maintained benchmark scripts rather than
  hand-editing numbers.

</specifics>

<deferred>
## Deferred Ideas

- Longer-decode `100/1000` parity beyond the user-approved short-length gate remains deferred.
- Any performance tuning beyond truthful benchmark publication is out of scope for this phase.

</deferred>

---
*Phase: 21-benchmark-attribution-and-impact*
*Context gathered: 2026-03-23*
