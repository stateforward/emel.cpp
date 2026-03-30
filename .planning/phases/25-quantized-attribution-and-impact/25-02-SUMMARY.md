---
phase: 25-quantized-attribution-and-impact
plan: 02
subsystem: benchmark-publication
tags: [bench, snapshot, docs, publication, approval]
requires:
  - phase: 25-quantized-attribution-and-impact
    plan: 01
    provides: maintained compare output and docsgen support for runtime-contract attribution
provides:
  - refreshed stored compare evidence with the shipped runtime contract
  - regenerated benchmark docs that explain the approved dense-f32-by-contract seams honestly
  - repo-gate verification for the approved publication workflow
affects: [milestone v1.5 closeout]
tech-stack:
  added: []
  patterns: [approval-gated snapshot refresh, truthful docs publication, repo-gate verification]
key-files:
  created: []
  modified: [snapshots/bench/benchmarks_compare.txt, docs/benchmarks.md]
key-decisions:
  - "Stop for explicit user approval before touching stored snapshots or generated docs."
  - "Use the maintained compare-update and docs-generation workflow rather than inventing a custom publication path."
requirements-completed: [BENCH-10]
duration: 0min
completed: 2026-03-25
---

# Phase 25 Plan 2 Summary

**Stored benchmark artifacts now publish the approved runtime contract**

## Accomplishments

- After explicit user approval, refreshed
  [benchmarks_compare.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks_compare.txt)
  through the maintained compare-update workflow so the canonical benchmark case now stores
  `generation_runtime_contract: ... native_quantized=8 approved_dense_f32_by_contract=4
  disallowed_fallback=0 explicit_no_claim=0`.
- Regenerated
  [benchmarks.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/docs/benchmarks.md)
  so the published benchmark docs mirror the stored runtime contract and include a contract
  summary that keeps token embedding and norm-vector seams in the approved dense-f32-by-contract
  bucket.
- Closed with a full `scripts/quality_gates.sh` pass, then restored
  [timing.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/quality_gates/timing.txt)
  to the preserved baseline values after the generated gate run.

## Verification

- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 scripts/bench.sh --compare-update`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`

## Deviations from Plan

- None in accepted scope. Publication waited for explicit approval, used the maintained workflow,
  and closed under the existing warning-only benchmark-regression policy.

---
*Phase: 25-quantized-attribution-and-impact*
*Completed: 2026-03-25*
