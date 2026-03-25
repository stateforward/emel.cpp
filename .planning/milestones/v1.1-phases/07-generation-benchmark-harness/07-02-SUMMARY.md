---
phase: 07-generation-benchmark-harness
plan: 02
subsystem: benchmarking
tags: [benchmark, generation, contract, compare-smoke, tools/bench]
requires: [07-01]
provides:
  - explicit preloaded-request generation benchmark contract in the case surface
  - deterministic canonical prompt/token defaults for the Llama-68M benchmark
  - compare-smoke proof that the generation case participates in the existing bench flow
affects: [07 phase closeout, 08-generation-compare-output, 09-benchmark-integration-hardening]
tech-stack:
  added: []
  patterns: [contract-in-case-name, dedicated heavy-case bench env overrides]
key-files:
  created: []
  modified: [tools/bench/generation_bench.cpp]
key-decisions:
  - "Use the benchmark case name itself to publish the measured segment: `generation/preloaded_request/...`."
  - "Keep one canonical prompt (`hello`) and one bounded token budget (`max_tokens=1`) as the default contract."
  - "Preserve the existing compare parser shape so Phase 7 stays compatible with `scripts/bench.sh` instead of adding a second reporting path."
patterns-established:
  - "Heavy generation benchmark cases should expose their measurement contract directly in the case name."
  - "Bounded generation cases should use dedicated generation env knobs instead of inheriting large generic bench iteration defaults."
requirements-completed: [BENCH-03]
duration: 18min
completed: 2026-03-09
---

# Phase 7 Plan 2 Summary

**The generation benchmark contract is now explicit, bounded, and visible through the existing bench surfaces**

## Accomplishments

- Locked one canonical generation contract in [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp): prompt `hello`, `max_tokens=1`, and a preloaded-request latency path.
- Made the measured segment user-visible through the case name `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`, so users can tell from output what the benchmark is timing without reading code.
- Proved the new case participates in both `bench_runner --mode=emel` and the existing `scripts/bench.sh --compare` surface.

## Task Commits

None. The plan was completed locally with `commit_docs` disabled.

## Files Created/Modified

- `tools/bench/generation_bench.cpp` - finalizes the visible contract and bounded generation defaults

## Decisions Made

- Kept the contract entirely inside the benchmark case surface instead of adding bench-only CLI flags or a second compare format.
- Used dedicated generation benchmark env variables so the heavy case remains bounded even when global bench defaults are much larger.
- Left the standard compare output format intact; the new generation case now fits into it without widening the parser surface.

## Deviations from Plan

- The plan listed [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp) as a possible Phase 7 plan 2 touchpoint. In practice, the visible measurement contract was fully expressed in the generation case name and harness defaults, so no additional `bench_main.cpp` edit was required after wave 1.

## Verification

- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=emel | rg "generation/"`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

## Next Readiness

- Phase 7 is ready to close.
- Phase 8 can now focus on whether any additional compare-surface hardening or output shaping is still needed beyond the now-working generation compare path.

---
*Phase: 07-generation-benchmark-harness*
*Completed: 2026-03-09*
