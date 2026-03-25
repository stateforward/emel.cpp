---
phase: 21-benchmark-attribution-and-impact
plan: 02
subsystem: benchmark-baselines-and-publication
tags: [bench, baselines, snapshots, docs, generation]
requires:
  - phase: 21-benchmark-attribution-and-impact
    plan: 01
    provides: canonical benchmark compare output with quantized attribution
provides:
  - refreshed maintained `1/10/100/1000` generation snapshot baselines
  - refreshed generation compare artifact for the widened maintained surface
  - generated docs aligned to the updated compare snapshot
affects: [21 verification, milestone closeout]
tech-stack:
  added: []
  patterns: [generated baseline refresh, truthful performance publication]
key-files:
  created: []
  modified:
    [snapshots/bench/benchmarks.txt, snapshots/bench/benchmarks_compare.txt, tools/bench/testdata/generation_compare_current.txt, docs/benchmarks.md]
key-decisions:
  - "Refresh the maintained baselines under the same `1000/3` benchmark env the repo quality gate uses to keep the active gate and stored evidence aligned."
  - "Publish the real performance story: `max_tokens=1` improved materially, while `10/100/1000` remain slower than the current v1.3 reference baseline."
requirements-completed: [BENCH-09]
duration: 0min
completed: 2026-03-23
---

# Phase 21 Plan 2 Summary

**The maintained benchmark evidence is refreshed for `1/10/100/1000`**

## Accomplishments

- Refreshed
  [benchmarks.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks.txt)
  with maintained `1/10/100/1000` generation entries under the active gate environment.
- Refreshed
  [benchmarks_compare.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks_compare.txt)
  so compare output now republishes the widened generation surface plus flash and quantized
  attribution headers.
- Refreshed
  [generation_compare_current.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/testdata/generation_compare_current.txt)
  to carry all maintained generation compare rows.
- Regenerated
  [benchmarks.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/docs/benchmarks.md)
  from the updated compare snapshot so the public benchmark page matches the maintained evidence.

## Verification

- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 build/bench_tools_ninja/bench_runner --mode=compare > /tmp/bench_compare_phase21.txt`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 build/bench_tools_ninja/bench_runner --mode=emel > /tmp/bench_emel_phase21.txt`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 BENCH_TOLERANCE=0.30 scripts/bench.sh --snapshot --compare`
- `build/docsgen/docsgen --root . --check`

## Deviations from Plan

- The full `scripts/quality_gates.sh` run still emits non-blocking benchmark variance warnings on a
  few unrelated non-generation cases, but the maintained benchmark publication artifacts and the
  standalone benchmark gate for this phase verify successfully.

---
*Phase: 21-benchmark-attribution-and-impact*
*Completed: 2026-03-23*
