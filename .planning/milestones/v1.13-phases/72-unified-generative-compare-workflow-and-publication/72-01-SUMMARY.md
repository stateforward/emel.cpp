---
phase: 72-unified-generative-compare-workflow-and-publication
plan: 01
status: complete
completed: 2026-04-20
requirements-completed:
  - CMP-01
  - CMP-02
  - CMP-03
---

# Phase 72 Summary

## Outcome

Phase 72 is complete. Operators now have one documented generation compare workflow that emits raw
lane JSONL, dumped outputs, backend identity, workload identity, and machine-readable compare
verdicts.

## Delivered

- Published `scripts/bench_generation_compare.sh --reference-backend llama_cpp_generation` as the
  operator-facing generation compare entrypoint.
- Added `--workload-id` selection so reproducible proof runs can pin one manifest-selected
  workload without executing the full workload matrix.
- Extended summary semantics to distinguish `exact_match`, `bounded_drift`, `non_comparable`,
  `missing`, and `error`.
- Documented artifact layout and verdict semantics in `docs/benchmarking.md`.

## Verification Result

- Focused `generation_compare_tests` proved exact-match, bounded-drift, non-comparable, and error
  outcomes.
- The documented wrapper path remained separate from benchmark snapshot publication while sharing
  the same `generation_compare/v1` raw lane contract.
