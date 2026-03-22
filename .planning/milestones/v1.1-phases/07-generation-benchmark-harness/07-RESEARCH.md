# Phase 07: Generation Benchmark Harness Research

**Phase:** 07
**Date:** 2026-03-08
**Requirement IDs:** BENCH-01, BENCH-02, BENCH-03
**Source:** `.planning/research/SUMMARY.md`, `tools/bench/*`, and the shipped v1.0 generation slice

## Goal

Add one canonical generation benchmark case to `tools/bench` for
`tests/models/Llama-68M-Chat-v1-Q2_K.gguf`, define a deterministic bounded workload, and make the
benchmark surface explicit about what portion of generation it measures.

## What Exists Today

- `tools/bench/CMakeLists.txt` already builds `bench_runner` with both EMEL and `llama.cpp`
  linkage.
- `tools/bench/bench_cases.hpp` and `tools/bench/bench_main.cpp` already support paired
  EMEL/reference benchmark cases with `--mode=emel`, `--mode=reference`, and `--mode=compare`.
- `scripts/bench.sh` already configures, builds, and runs benchmark compare/snapshot flows.
- The shipped v1.0 generation path lives in `tools/paritychecker/parity_runner.cpp`, but there is
  no generation benchmark case in `tools/bench/`.

## Narrowest Correct Benchmark Contract

Phase 7 should lock one explicit benchmark contract before Phase 8 adds compare-mode reporting:

1. The fixture is the canonical local model `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
2. The workload is one deterministic bounded generation request with pinned prompt text and
   token-budget defaults.
3. The measured segment is one **preloaded request latency** path:
   model loading and one-time benchmark setup happen outside the timed loop, while each timed
   iteration covers request-local preparation and bounded generation.
4. The benchmark case name itself should expose that contract so BENCH-03 is satisfied without
   making users inspect code.

This keeps the milestone truthful and avoids conflating model-load cost with generation latency.

## Likely Edit Surface

- `tools/bench/CMakeLists.txt`
  Add a generation benchmark source file to `BENCH_RUNNER_SOURCES`.
- `tools/bench/bench_cases.hpp`
  Declare paired generation case appenders.
- `tools/bench/bench_main.cpp`
  Register the new generation benchmark case in the existing case list.
- `tools/bench/generation_bench.cpp`
  Implement the canonical generation benchmark fixture and EMEL case.

Phase 7 should avoid widening `scripts/bench.sh` beyond the existing runner contract unless a
small wiring change is required for deterministic case discovery.

## Implementation Constraints

- Keep all benchmark work in `tools/bench/`; do not widen public API surface.
- Reuse the shipped generation slice rather than inventing a benchmark-only orchestration path.
- Keep the benchmark deterministic and narrow: one canonical fixture, one prompt, one token budget.
- Avoid multi-model or multi-prompt scope creep in this phase.
- Preserve the repo’s tool boundary: `llama.cpp` linkage remains in `tools/bench/` only.

## Risks And Pitfalls

- If the benchmark times a different workload on EMEL than the one eventually used for reference
  compare, Phase 8’s ratios will be misleading.
- If the benchmark measures model loading or one-time setup inside the timed loop, results will be
  noisy and harder to interpret.
- If the case name does not encode the measured segment, BENCH-03 remains unmet even if the code is
  correct.
- If Phase 7 reaches into `src/` to create benchmark-only hooks, the milestone will drift away from
  the already-proven generation path.

## Recommended Phase Shape

Two plans are sufficient:

1. **Benchmark harness wiring**
   Add the new benchmark source, declarations, case registration, and an EMEL-only canonical
   generation benchmark path.
2. **Deterministic workload contract hardening**
   Lock the prompt/token defaults, make the measured segment explicit in the benchmark surface, and
   prove the new case runs through the existing emel-only bench flow.

## Validation Architecture

**Quick feedback loop**
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --emel-only`

**Behavior checks**
- `build/bench_tools_ninja/bench_runner --mode=emel`
- `build/bench_tools_ninja/bench_runner --mode=emel | rg "generation/"`

**Full gate for this phase**
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

The validation focus should stay on the harness contract:
- one canonical generation benchmark case exists
- the workload is bounded and deterministic
- the benchmark name/output tells the user what segment is being measured

## Planning Recommendation

Keep Phase 7 EMEL-first. Do not add the full reference compare output yet; Phase 8 owns the final
truthful EMEL-vs-`llama.cpp` comparison surface.
