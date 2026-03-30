# Phase 25: Quantized Attribution And Impact - Research

**Researched:** 2026-03-25
**Domain:** Maintained benchmark attribution after Phase 24 proof
**Confidence:** HIGH

## Summary

The smallest truthful Phase 25 gap is publication, not proof. `tools/paritychecker` now publishes
and enforces the shipped `8/4/0/0` runtime contract, but the maintained benchmark surface still
only publishes dispatch evidence (`generation_flash_evidence` and
`generation_quantized_evidence`). The compare snapshot and generated docs do not yet say that the
canonical workload stayed on the approved runtime contract or that the remaining cost includes
approved dense-f32-by-contract seams rather than a hidden disallowed fallback.

`tools/bench/bench_main.cpp` already hard-fails invalid flash/quantized dispatch evidence, so the
right Phase 25 move is additive: publish benchmark-time runtime contract counts from the shipped
generator surface, then update `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md`
to tell the new story honestly. But that second step crosses the repo’s explicit consent boundary,
because snapshot updates are not allowed without user approval.

## Evidence

- `tools/bench/bench_main.cpp:308-429`
  Compare mode already validates and prints flash and quantized dispatch evidence, but it does not
  print the Phase 24 runtime contract counts.
- `tools/bench/generation_bench.cpp:1744-1887`
  Generation bench capture already records dispatch evidence from the shipped generator wrapper and
  can be extended to capture additional runtime contract counts.
- `snapshots/bench/benchmarks_compare.txt:1-6`
  The maintained compare snapshot currently publishes `generation_flash_evidence` and
  `generation_quantized_evidence`, but no explicit runtime contract or approved dense-f32-by-
  contract attribution line.
- `docs/benchmarks.md:10-20`
  The generated benchmark docs mirror the current snapshot and therefore also omit the Phase 24
  runtime-contract story.
- `scripts/bench.sh`
  Snapshot and compare publication are already centralized here, so Phase 25 should continue using
  the maintained operator workflow rather than inventing a second publication path.

## Recommended Plan Split

### 25-01: Add Runtime-Contract Benchmark Attribution

- Extend the benchmark capture/publication path so compare output prints the shipped runtime
  contract counts alongside the existing dispatch evidence.
- Keep the benchmark proof additive and aligned with the approved `8/4/0/0` contract without
  rewriting the current warning-only regression policy.
- Verify locally with focused compare runs, but do not update stored snapshots yet.

### 25-02: Refresh Stored Compare Artifacts And Docs

- Regenerate `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` so the stored
  publication matches the new attribution surface and current benchmark truth.
- Record the current warning-only benchmark regressions honestly instead of hiding them.
- Do **not** execute this step without explicit user approval, because it updates stored snapshots.

## User Approval Check

Phase 25 planning and pre-publication code work do not require approval. Executing the snapshot and
docs refresh step does require explicit user approval because `AGENTS.md` forbids snapshot updates
without consent.
