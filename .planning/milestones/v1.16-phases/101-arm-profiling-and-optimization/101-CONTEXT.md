# Phase 101: ARM Profiling And Optimization - Context

**Gathered:** 2026-04-26
**Status:** Complete

<domain>
## Phase Boundary

Profile the maintained EMEL Whisper runtime, optimize source-owned runtime/kernel code, and prove
the optimized EMEL lane beats the matched single-thread CPU `whisper.cpp` lane. Phase 101 does not
own milestone closeout or broad documentation generation.

</domain>

<decisions>
## Profiling Findings

- Phase 100 benchmark records showed EMEL already ahead of the pinned reference on process wall
  time: EMEL mean `327,299,736 ns`, reference mean `419,345,402 ns`.
- EMEL stage timing identified encoder execution as the dominant runtime cost, around
  `254-258 ms` per run before optimization.
- Decoder execution was the second largest stage, around `49-50 ms` before optimization.

## Optimization Direction

- Optimize source-owned Whisper kernel code, not benchmark scaffolding.
- Keep runtime variant routing unchanged in SML guards/transitions.
- Keep hot-path dispatch allocation-free; use caller-provided workspace only.
- Preserve deterministic token transcript output.

</decisions>

<code_context>
## Changes Attempted

- `src/emel/kernel/whisper/detail.hpp`
  - Added caller-workspace Hann/chirp tables so mel preparation does not recompute those trig
    values per frame.
  - Added an AArch64 NEON q8_0 row dot helper for Whisper linear projections, with scalar fallback
    on non-AArch64 hosts.
- `tests/whisper/kernel/detail_tests.cpp`
  - Added focused tests for q4/q8 helper paths, tensor lookup/shape helpers, softmax, spectral
    helper paths, transcript writing, and the AArch64 q8 row helper.
- `CMakeLists.txt`, `scripts/test_with_coverage.sh`, `scripts/quality_gates.sh`
  - Added missing `whisper` compile-time shard support so Whisper kernel changes can be scoped to
    the matching tests.

</code_context>

<specifics>
## Current Benchmark Evidence

After the NEON q8_0 row optimization and final scoped gate:

- EMEL mean process wall time: `143,873,444 ns`
- Reference mean process wall time: `431,662,486 ns`
- EMEL encode stage: about `102-103 ms`
- EMEL decode stage: about `14.8-15.0 ms`
- Transcript remained `token:50257`

</specifics>

<deferred>
## Deferred

No Phase 101 blocker remains. The final changed-file quality gate passed with:

- `src/emel/kernel/whisper/detail.hpp` line coverage: `100.0%`
- `src/emel/kernel/whisper/detail.hpp` branch coverage: `55.3%`
- Required branch threshold: `50.0%`

A broader coverage attempt hit an unrelated generator test failure:

`generator_quantized_path_audit_marks_unsupported_quantized_stage_no_claim`

Phase 101 should not be marked complete until the coverage gate is resolved without weakening the
threshold.

</deferred>
