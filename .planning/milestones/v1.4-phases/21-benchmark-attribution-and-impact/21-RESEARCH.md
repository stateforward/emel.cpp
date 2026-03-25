# Phase 21: Benchmark Attribution And Impact - Research

**Researched:** 2026-03-23
**Domain:** maintained benchmark attribution and baseline refresh
**Confidence:** HIGH

<user_constraints>
## User Constraints

- The user explicitly approved proceeding with benchmark and baseline work in this session.
- Snapshot updates are allowed for this phase.
- The maintained truth bar for parity remains `1/10`; benchmark publication may still report
  `100/1000` timing results without claiming longer-decode parity is solved.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-08 | Maintained ARM compare workflow must publish attribution that proves the vectorized quantized path. | `generator::sm` already exposes q2/q3/q6 optimized/shared counters; `tools/bench` simply does not publish them yet. |
| BENCH-09 | Maintained benchmark evidence must republish `1/10/100/1000` results against the current v1.3 baseline. | The benchmark cases are already widened in `tools/bench`; only compare publication and baseline refresh are missing. |

</phase_requirements>

## Summary

Phase 21 is mostly a publication and evidence-refresh pass, not a new runtime implementation.

1. `tools/bench/generation_bench.cpp` already captures the canonical generation benchmark surface,
   but it only stores flash evidence today.
2. `tools/bench/bench_main.cpp` already validates the canonical generation compare case, but it
   only prints and checks flash evidence.
3. The maintained baseline files are stale for the widened generation case set and still carry the
   old `max_tokens_8` entry.

That means the safest phase split is:

1. extend the benchmark evidence state to include q2/q3/q6 optimized/shared attribution
2. validate and print that attribution from compare mode
3. regenerate `benchmarks.txt`, `benchmarks_compare.txt`, and the generation compare testdata
4. record whether at least one generation length improved versus the v1.3 reference baseline

## Likely File Changes

| File | Why |
|------|-----|
| `tools/bench/generation_bench.cpp` | Capture generation-time q2/q3/q6 optimized/shared counters for canonical benchmark evidence. |
| `tools/bench/bench_main.cpp` | Validate and print canonical quantized benchmark attribution in compare mode. |
| `snapshots/bench/benchmarks.txt` | Refresh maintained EMEL snapshot baseline for widened generation coverage. |
| `snapshots/bench/benchmarks_compare.txt` | Refresh maintained compare evidence and attribution header. |
| `tools/bench/testdata/generation_compare_current.txt` | Refresh maintained generation compare artifact. |

## Anti-Patterns To Avoid

- Do not claim quantized benchmark proof from a different fixture than the maintained canonical
  Q2_K generation workload.
- Do not hand-edit benchmark numbers when the maintained scripts can regenerate them.
- Do not describe all widened generation lengths as improvements if some still regress.

---
*Phase: 21-benchmark-attribution-and-impact*
*Research completed: 2026-03-23*
