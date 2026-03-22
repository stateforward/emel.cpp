---
phase: 08-generation-compare-output
verified: 2026-03-10T23:30:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 8 Verification Report

**Phase Goal:** Compare EMEL and `llama.cpp` timings for the same canonical Llama-68M generation
workload and publish truthful compare output through the existing bench surface.
**Verified:** 2026-03-10T23:30:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Compare mode now enforces a canonical generation pairing contract instead of relying only on sorted position. | ✓ VERIFIED | [bench_cases.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_cases.hpp) exports the shared canonical case name, [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp) now rejects duplicate case names plus missing canonical generation rows on either side, and `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare` prints the expected generation row. |
| 2 | The published compare row exposes EMEL timing, `llama.cpp` timing, and ratio for the canonical generation workload through the normal compare flow. | ✓ VERIFIED | `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "emel\\.cpp .* llama\\.cpp .* ratio="` matches the canonical `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` row, and `scripts/bench.sh --compare` publishes the same row through the standard script surface. |
| 3 | Truth-support diagnostics stay opt-in and off the standard compare stdout surface. | ✓ VERIFIED | [scripts/bench.sh](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/scripts/bench.sh) now routes compare-path configure/build chatter to stderr, and `EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 scripts/bench.sh --compare` shows seam-audit lines only on stderr while stdout remains the compare row stream. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| COMP-01 | ✓ SATISFIED | - |
| COMP-02 | ✓ SATISFIED | - |

## Automated Checks

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "emel\\.cpp .* llama\\.cpp .* ratio="`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
- `EMEL_BENCH_CASE_INDEX=7 EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
- `scripts/quality_gates.sh`

## Verification Notes

- The canonical generation compare row remained in the existing bench style; Phase 8 did not add a
  second report format.
- `scripts/bench.sh --compare` now keeps operational build chatter on stderr, which makes the
  stdout compare surface usable as published evidence without changing the row format itself.
- `scripts/quality_gates.sh` passed. It still reported the repo's known benchmark snapshot
  regressions and the missing baseline entry for the generation benchmark as non-blocking policy,
  then exited successfully with the existing warning `benchmark snapshot regression ignored by
  quality gates`.
